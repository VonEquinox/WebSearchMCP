from __future__ import annotations

import re
from typing import List

from .prompts import (
    SearchPromptProfile,
    build_search_prompt,
    classify_query_complexity,
    fetch_prompt,
    rank_sources_prompt,
    search_prompt,
    url_describe_prompt,
)
from .providers.base import SearchResult

_URL_PATTERN = re.compile(r"https?://[^\s<>\"'`，。、；：！？》）】\)]+")


def extract_unique_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_PATTERN.finditer(text or ""):
        url = match.group().rstrip('.,;:!?')
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def format_extra_sources(tavily_results: list[dict] | None, firecrawl_results: list[dict] | None) -> str:
    sections: list[str] = []
    seen_urls: set[str] = set()
    start_index = 1
    for provider, results, text_key in (
        ("Firecrawl", firecrawl_results, "description"),
        ("Tavily", tavily_results, "content"),
    ):
        lines = [f"## Extra Sources [{provider}]"]
        for item in results or []:
            url = (item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = (item.get("title") or "Untitled").strip()
            lines.append(f"{start_index}. **[{title}]({url})**")
            extra_text = (item.get(text_key) or "").strip()
            if extra_text:
                lines.append(f"   {extra_text}")
            start_index += 1
        if len(lines) > 1:
            sections.append("\n".join(lines))
    return "\n\n".join(sections)


def format_search_results(results: List[SearchResult]) -> str:
    if not results:
        return "No results found."
    blocks: list[str] = []
    for index, result in enumerate(results, 1):
        parts = [f"## Result {index}: {result.title}"]
        if result.url:
            parts.append(f"**URL:** {result.url}")
        if result.snippet:
            parts.append(f"**Summary:** {result.snippet}")
        if result.source:
            parts.append(f"**Source:** {result.source}")
        if result.published_date:
            parts.append(f"**Published:** {result.published_date}")
        blocks.append("\n".join(parts))
    return "\n\n---\n\n".join(blocks)


__all__ = [
    "SearchPromptProfile",
    "build_search_prompt",
    "classify_query_complexity",
    "extract_unique_urls",
    "fetch_prompt",
    "format_extra_sources",
    "format_search_results",
    "rank_sources_prompt",
    "search_prompt",
    "url_describe_prompt",
]
