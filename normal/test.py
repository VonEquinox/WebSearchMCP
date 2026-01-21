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
    if len(sys.argv) > 1 and sys.argv[1] == "--llm":
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
