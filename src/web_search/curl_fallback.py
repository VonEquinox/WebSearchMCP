from __future__ import annotations

import asyncio
import json
from html.parser import HTMLParser
from typing import Any

from curl_cffi import requests as curl_requests

from .config import config
from .logger import log_info

_DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}
_BLOCK_TAGS = {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr"}


class _HtmlTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._ignored_depth += 1
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._ignored_depth > 0:
            self._ignored_depth -= 1
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return
        text = " ".join(data.split())
        if text:
            self._parts.append(text)

    def text(self) -> str:
        lines = [line.strip() for line in "".join(self._parts).splitlines()]
        merged = "\n".join(line for line in lines if line)
        return merged.strip()


def _render_body_as_text(content_type: str, body: str) -> str:
    lowered_type = (content_type or "").lower()
    stripped = (body or "").strip()
    if not stripped:
        return ""

    if "application/json" in lowered_type:
        try:
            parsed = json.loads(stripped)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            return stripped

    if "text/html" not in lowered_type and "<html" not in stripped[:512].lower():
        return stripped

    parser = _HtmlTextParser()
    parser.feed(stripped)
    return parser.text()


def _request_with_curl(url: str, timeout_seconds: float) -> tuple[int, str, str]:
    response = curl_requests.get(
        url,
        headers=_DEFAULT_HEADERS,
        timeout=timeout_seconds,
        allow_redirects=True,
        impersonate="chrome124",
    )
    content_type = response.headers.get("Content-Type", "") if response.headers else ""
    return response.status_code, content_type, response.text


async def call_curl_cffi_extract(url: str, ctx: Any = None) -> str | None:
    attempts = max(config.retry_max_attempts, 1)
    for attempt in range(1, attempts + 1):
        try:
            status_code, content_type, body = await asyncio.to_thread(_request_with_curl, url, 45.0)
        except Exception as exc:
            await log_info(
                ctx,
                f"curl_cffi error (attempt {attempt}/{attempts}): {type(exc).__name__}: {exc}",
                config.debug_enabled,
            )
            continue

        if status_code >= 400:
            await log_info(
                ctx,
                f"curl_cffi HTTP {status_code} (attempt {attempt}/{attempts}) for {url}",
                config.debug_enabled,
            )
            continue

        rendered = _render_body_as_text(content_type, body)
        if rendered:
            return rendered

        await log_info(ctx, f"curl_cffi empty body (attempt {attempt}/{attempts}) for {url}", config.debug_enabled)
    return None
