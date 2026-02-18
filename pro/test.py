import asyncio
import json
import os
import sys
from pathlib import Path

from openai import OpenAI


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("export "):
            raw = raw[len("export ") :].strip()
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _parse_viewport(raw: str | None) -> dict | None:
    if not raw:
        return None
    text = raw.lower().replace(" ", "")
    if "x" in text:
        width_str, height_str = text.split("x", 1)
    elif "," in text:
        width_str, height_str = text.split(",", 1)
    else:
        return None
    try:
        return {"width": int(width_str), "height": int(height_str)}
    except ValueError:
        return None


def _looks_like_challenge(title: str) -> bool:
    lowered = (title or "").lower()
    return "just a moment" in lowered or "attention required" in lowered


def run_unit_tests() -> dict:
    from SOTASearch import (
        _clean_ai_tags,
        _extract_browse_page_links,
        _parse_markdown_links,
        _parse_sse_chat_completions,
        _strip_urls,
        _unwrap_redirect_url,
    )

    ai_content = (
        "这里是正文内容，模型偶尔会夹带链接："
        "[xAI Reasoning Guide](https://docs.x.ai/docs/guides/reasoning)。"
        "也可能出现裸链接 (https://example.com/bare)。"
        "甚至是尖括号链接 <https://example.com/angle>。"
        "\n\n参考来源（reasoning中列出）：\n"
        "- https://example.com/ref1\n"
        "- https://example.com/ref2\n"
    )
    ai_reasoning = "\n".join(
        [
            "Sources:",
            "https://docs.x.ai/docs/guides/reasoning",
            "[Prompt Engineering](https://docs.x.ai/docs/guides/grok-code-prompt-engineering)",
        ]
    )

    links, summary = _parse_markdown_links(ai_content, extra_text=ai_reasoning)
    summary_clean = _strip_urls(summary)
    content_clean = _strip_urls(_clean_ai_tags(ai_content))

    assert "http://" not in summary_clean and "https://" not in summary_clean
    assert "http://" not in content_clean and "https://" not in content_clean
    assert "参考来源" not in summary_clean and "参考来源" not in content_clean

    urls = {link.get("url") for link in links}
    assert "https://docs.x.ai/docs/guides/reasoning" in urls
    assert "https://example.com/bare" in urls
    assert "https://example.com/ref1" in urls

    # SSE parsing (upstream returns text/event-stream)
    sse = "\n".join(
        [
            'data: {"choices":[{"delta":{"role":"assistant","content":"Hello "}}]}',
            "",
            'data: {"choices":[{"delta":{"content":"world"}}]}',
            "",
            "data: [DONE]",
            "",
        ]
    )
    sse_content, sse_reasoning = _parse_sse_chat_completions(sse)
    assert sse_content == "Hello world"
    assert sse_reasoning == ""

    # Redirect unwrap
    ddg = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fb%3Dc"
    assert _unwrap_redirect_url(ddg) == "https://example.com/a?b=c"
    brave = "https://r.search.brave.com/redirect?url=https%3A%2F%2Fexample.com%2Fp"
    assert _unwrap_redirect_url(brave) == "https://example.com/p"
    zhihu = "https://link.zhihu.com/?target=https%3A%2F%2Fexample.com%2Fz"
    assert _unwrap_redirect_url(zhihu) == "https://example.com/z"

    # Link parsing should accept www/ // / json-url and unwrap redirects
    content2 = (
        "Sources:\n"
        "- [Example](www.example.com/path)\n"
        "- //example.com/p2\n"
        '- {"url": "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fp3"}\n'
    )
    links2, _ = _parse_markdown_links(content2)
    urls2 = {link.get("url") for link in links2}
    assert "https://www.example.com/path" in urls2
    assert "https://example.com/p2" in urls2
    assert "https://example.com/p3" in urls2

    # Grok browse_page trace URL extraction
    browse_text = (
        'browse_page {"url":"https://openai.com/index/introducing-gpt-5-3-codex/",'
        '"instructions":"Confirm release date and safety notes."}\\n'
        'browse_page {"url":"https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fwrapped",'
        '"instructions":"unwrap redirect"}'
    )
    browse_links = _extract_browse_page_links(browse_text)
    browse_urls = [link.get("url") for link in browse_links]
    assert "https://openai.com/index/introducing-gpt-5-3-codex/" in browse_urls
    assert "https://example.com/wrapped" in browse_urls

    return {
        "success": True,
        "links_extracted": len(links),
        "ai_links": links,
        "ai_summary_clean": summary_clean,
        "ai_content_clean": content_clean,
    }


def call_llm(content: str) -> dict:
    _load_env_file(Path(__file__).with_name(".env"))
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    if not api_key or not base_url:
        raise SystemExit("Missing OPENAI_API_KEY or OPENAI_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url)

    text = content if content and content.strip() else " "
    messages = [{"role": "user", "content": text}]
    response = client.chat.completions.create(model=model, messages=messages)
    try:
        return response.model_dump()
    except Exception:
        return {"raw": str(response)}


async def call_fetch(url: str, mode: str = "text") -> dict:
    from SOTASearch import fetch

    mode = (mode or "text").lower()
    _ = mode  # reserved for compatibility
    return await fetch(url)


async def call_playwright(url: str, mode: str = "text") -> dict:
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "error": f"playwright not installed: {e}",
        }
    try:
        from playwright_stealth import Stealth
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "error": f"playwright-stealth not installed: {e}",
        }

    mode = (mode or "text").lower()
    proxy = os.getenv("PROXY")
    headless = os.getenv("PW_HEADLESS", "1").lower() not in ("0", "false", "no")
    launch_args = {"headless": headless}
    if proxy:
        launch_args["proxy"] = {"server": proxy}

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(**launch_args)
            user_agent = os.getenv(
                "PW_USER_AGENT",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36",
            )
            locale = os.getenv("PW_LOCALE", "zh-CN")
            timezone_id = os.getenv("PW_TIMEZONE", "Asia/Shanghai")
            viewport = _parse_viewport(os.getenv("PW_VIEWPORT", "1366x768"))
            device_scale_factor = float(os.getenv("PW_DEVICE_SCALE", "2"))

            context_kwargs = {
                "user_agent": user_agent,
                "locale": locale,
                "timezone_id": timezone_id,
                "color_scheme": "light",
                "device_scale_factor": device_scale_factor,
            }
            if viewport:
                context_kwargs["viewport"] = viewport

            context = await browser.new_context(**context_kwargs)
            await context.set_extra_http_headers(
                {"Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7"}
            )
            page = await context.new_page()
            await Stealth().apply_stealth_async(page)
            await page.goto(url, wait_until="domcontentloaded", timeout=60_000)

            for _ in range(20):
                title = await page.title()
                if not _looks_like_challenge(title):
                    break
                await page.wait_for_timeout(1000)

            if mode == "html":
                content = await page.content()
                result = {"success": True, "url": url, "via_playwright": True, "html": content}
            else:
                text = await page.inner_text("body")
                result = {"success": True, "url": url, "via_playwright": True, "text": text}

            await context.close()
            await browser.close()
            return result
    except Exception as e:
        return {"success": False, "url": url, "error": str(e), "via_playwright": True}


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--unit":
        result = run_unit_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "--llm":
        content = " ".join(sys.argv[2:]).strip()
        result = call_llm(content)
    elif len(sys.argv) > 1 and sys.argv[1] in ("--playwright", "--pw"):
        url = sys.argv[2].strip() if len(sys.argv) > 2 else "https://linux.do/t/topic/1496040"
        mode = sys.argv[3].strip() if len(sys.argv) > 3 else "text"
        result = asyncio.run(call_playwright(url, mode=mode))
    else:
        url = sys.argv[1].strip() if len(sys.argv) > 1 else "https://linux.do/t/topic/1496040"
        mode = sys.argv[2].strip() if len(sys.argv) > 2 else "text"
        result = asyncio.run(call_fetch(url, mode=mode))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
