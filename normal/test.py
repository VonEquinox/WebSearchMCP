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


def run_unit_tests() -> dict:
    from SOTASearch import _clean_ai_tags, _parse_markdown_links, _strip_urls

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
    from SOTASearch import fetch_html, fetch_metadata, fetch_text

    mode = (mode or "text").lower()
    if mode == "html":
        return await fetch_html(url)
    if mode in ("meta", "metadata"):
        return await fetch_metadata(url)
    return await fetch_text(url)


async def call_playwright(url: str, mode: str = "text") -> dict:
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "error": f"playwright not installed: {e}",
        }

    mode = (mode or "text").lower()
    proxy = os.getenv("PROXY")
    launch_args = {"headless": True}
    if proxy:
        launch_args["proxy"] = {"server": proxy}

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(**launch_args)
            page = await browser.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            if mode == "html":
                content = await page.content()
                result = {"success": True, "url": url, "via_playwright": True, "html": content}
            else:
                text = await page.inner_text("body")
                result = {"success": True, "url": url, "via_playwright": True, "text": text}

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
