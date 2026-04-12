import ast
import json
import re
import uuid
from collections import OrderedDict
from typing import Any

import asyncio

from .utils import extract_unique_urls


_MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_SOURCES_HEADING_PATTERN = re.compile(
    r"(?im)^"
    r"(?:#{1,6}\s*)?"
    r"(?:\*\*|__)?\s*"
    r"(sources?|references?|citations?|信源|参考资料|参考|引用|来源列表|来源)"
    r"\s*(?:\*\*|__)?"
    r"(?:\s*[（(][^)\n]*[)）])?"
    r"\s*[:：]?\s*$"
)
_SOURCES_FUNCTION_PATTERN = re.compile(
    r"(?im)(^|\n)\s*(sources|source|citations|citation|references|reference|citation_card|source_cards|source_card)\s*\("
)
_SEARCH_ARTIFACT_PATTERN = re.compile(r"\[(webSearchResults|browse_page)\](?=\[|\{)", re.IGNORECASE)


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


class SourcesCache:
    def __init__(self, max_size: int = 256):
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._cache: OrderedDict[str, list[dict]] = OrderedDict()

    async def set(self, session_id: str, sources: list[dict]) -> None:
        async with self._lock:
            self._cache[session_id] = sources
            self._cache.move_to_end(session_id)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    async def get(self, session_id: str) -> list[dict] | None:
        async with self._lock:
            sources = self._cache.get(session_id)
            if sources is None:
                return None
            self._cache.move_to_end(session_id)
            return sources


def merge_sources(*source_lists: list[dict]) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []
    for sources in source_lists:
        for item in sources or []:
            url = (item or {}).get("url")
            if not isinstance(url, str) or not url.strip():
                continue
            url = url.strip()
            if url in seen:
                continue
            seen.add(url)
            merged.append(item)
    return merged


def split_answer_and_sources(text: str) -> tuple[str, list[dict]]:
    raw = (text or "").strip()
    if not raw:
        return "", []

    split = _split_function_call_sources(raw)
    if split and _answer_has_substance(split[0]):
        return split

    split = _split_heading_sources(raw)
    if split and _answer_has_substance(split[0]):
        return split

    split = _split_details_block_sources(raw)
    if split and _answer_has_substance(split[0]):
        return split

    split = _split_tail_link_block(raw)
    if split and _answer_has_substance(split[0]):
        return split

    return raw, []


def extract_search_artifacts(text: str) -> tuple[str, list[dict], list[dict]]:
    raw = text or ""
    if not raw.strip():
        return "", [], []

    cleaned_parts: list[str] = []
    cursor = 0
    web_results_payloads: list[Any] = []
    browse_page_payloads: list[Any] = []

    for match in _SEARCH_ARTIFACT_PATTERN.finditer(raw):
        open_index = match.end()
        if open_index >= len(raw):
            continue
        opening_char = raw[open_index]
        artifact_name = match.group(1).lower()
        if artifact_name == "browse_page":
            extracted = _extract_balanced_curly_block(raw, open_index) if opening_char == "{" else None
        elif opening_char == "[":
            extracted = _extract_balanced_square_block(raw, open_index)
        else:
            extracted = None
        if not extracted:
            continue

        close_index, payload = extracted
        cleaned_parts.append(raw[cursor:match.start()])
        parsed = _parse_jsonish(payload)
        if parsed is not None:
            if artifact_name == "websearchresults":
                web_results_payloads.append(parsed)
            else:
                browse_page_payloads.append(parsed)
        cursor = close_index + 1

    cleaned_parts.append(raw[cursor:])
    cleaned_text = "".join(cleaned_parts)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

    return (
        cleaned_text,
        _normalize_web_results(web_results_payloads),
        _normalize_browse_pages(browse_page_payloads),
    )


def _answer_has_substance(answer: str) -> bool:
    cleaned = (answer or "").strip()
    if not cleaned:
        return False
    if _SOURCES_HEADING_PATTERN.fullmatch(cleaned):
        return False
    return True


def _split_function_call_sources(text: str) -> tuple[str, list[dict]] | None:
    matches = list(_SOURCES_FUNCTION_PATTERN.finditer(text))
    if not matches:
        return None

    for m in reversed(matches):
        open_paren_idx = m.end() - 1
        extracted = _extract_balanced_call_at_end(text, open_paren_idx)
        if not extracted:
            continue

        close_paren_idx, args_text = extracted
        sources = _parse_sources_payload(args_text)
        if not sources:
            continue

        answer = text[: m.start()].rstrip()
        return answer, sources

    return None


def _extract_balanced_call_at_end(text: str, open_paren_idx: int) -> tuple[int, str] | None:
    if open_paren_idx < 0 or open_paren_idx >= len(text) or text[open_paren_idx] != "(":
        return None

    depth = 1
    in_string: str | None = None
    escape = False

    for idx in range(open_paren_idx + 1, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == in_string:
                in_string = None
            continue

        if ch in ("'", '"'):
            in_string = ch
            continue

        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            if depth == 0:
                if text[idx + 1 :].strip():
                    return None
                args_text = text[open_paren_idx + 1 : idx]
                return idx, args_text

    return None


def _extract_balanced_square_block(text: str, open_bracket_idx: int) -> tuple[int, str] | None:
    if open_bracket_idx < 0 or open_bracket_idx >= len(text) or text[open_bracket_idx] != "[":
        return None

    depth = 1
    in_string: str | None = None
    escape = False

    for idx in range(open_bracket_idx + 1, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == in_string:
                in_string = None
            continue

        if ch in ("'", '"'):
            in_string = ch
            continue

        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                return idx, text[open_bracket_idx : idx + 1]

    return None


def _extract_balanced_curly_block(text: str, open_brace_idx: int) -> tuple[int, str] | None:
    if open_brace_idx < 0 or open_brace_idx >= len(text) or text[open_brace_idx] != "{":
        return None

    depth = 1
    in_string: str | None = None
    escape = False

    for idx in range(open_brace_idx + 1, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == in_string:
                in_string = None
            continue

        if ch in ("'", '"'):
            in_string = ch
            continue

        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return idx, text[open_brace_idx : idx + 1]

    return None


def _split_heading_sources(text: str) -> tuple[str, list[dict]] | None:
    matches = list(_SOURCES_HEADING_PATTERN.finditer(text))
    if not matches:
        return None

    for m in reversed(matches):
        start = m.start()
        sources_text = text[start:]
        sources = _extract_sources_from_text(sources_text)
        if not sources:
            continue
        answer = text[:start].rstrip()
        return answer, sources
    return None


def _split_tail_link_block(text: str) -> tuple[str, list[dict]] | None:
    lines = text.splitlines()
    if not lines:
        return None

    idx = len(lines) - 1
    while idx >= 0 and not lines[idx].strip():
        idx -= 1
    if idx < 0:
        return None

    tail_end = idx
    link_like_count = 0
    while idx >= 0:
        line = lines[idx].strip()
        if not line:
            idx -= 1
            continue
        if not _is_link_only_line(line):
            break
        link_like_count += 1
        idx -= 1

    tail_start = idx + 1
    if link_like_count < 2:
        return None

    block_text = "\n".join(lines[tail_start : tail_end + 1])
    sources = _extract_sources_from_text(block_text)
    if not sources:
        return None

    answer = "\n".join(lines[:tail_start]).rstrip()
    return answer, sources


def _split_details_block_sources(text: str) -> tuple[str, list[dict]] | None:
    lower = text.lower()
    close_idx = lower.rfind("</details>")
    if close_idx == -1:
        return None
    tail = text[close_idx + len("</details>") :].strip()
    if tail:
        return None

    open_idx = lower.rfind("<details", 0, close_idx)
    if open_idx == -1:
        return None

    block_text = text[open_idx : close_idx + len("</details>")]
    sources = _extract_sources_from_text(block_text)
    if len(sources) < 2:
        return None

    answer = text[:open_idx].rstrip()
    return answer, sources


def _is_link_only_line(line: str) -> bool:
    stripped = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", line).strip()
    if not stripped:
        return False
    if stripped.startswith(("http://", "https://")):
        return True
    if _MD_LINK_PATTERN.search(stripped):
        return True
    return False


def _parse_sources_payload(payload: str) -> list[dict]:
    payload = (payload or "").strip().rstrip(";")
    if not payload:
        return []

    data: Any = None
    try:
        data = json.loads(payload)
    except Exception:
        try:
            data = ast.literal_eval(payload)
        except Exception:
            data = None

    if data is None:
        return _extract_sources_from_text(payload)

    if isinstance(data, dict):
        for key in ("sources", "citations", "references", "urls"):
            if key in data:
                return _normalize_sources(data[key])
        return _normalize_sources(data)

    return _normalize_sources(data)


def _parse_jsonish(payload: str) -> Any:
    payload = (payload or "").strip()
    if not payload:
        return None

    try:
        return json.loads(payload)
    except Exception:
        try:
            return ast.literal_eval(payload)
        except Exception:
            return None


def _normalize_sources(data: Any) -> list[dict]:
    items: list[Any]
    if isinstance(data, (list, tuple)):
        items = list(data)
    elif isinstance(data, dict):
        items = [data]
    else:
        items = [data]

    normalized: list[dict] = []
    seen: set[str] = set()

    for item in items:
        if isinstance(item, str):
            for url in extract_unique_urls(item):
                if url not in seen:
                    seen.add(url)
                    normalized.append({"url": url})
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title, url = item[0], item[1]
            if isinstance(url, str) and url.startswith(("http://", "https://")) and url not in seen:
                seen.add(url)
                out: dict = {"url": url}
                if isinstance(title, str) and title.strip():
                    out["title"] = title.strip()
                normalized.append(out)
            continue

        if isinstance(item, dict):
            url = item.get("url") or item.get("href") or item.get("link")
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                continue
            if url in seen:
                continue
            seen.add(url)
            out: dict = {"url": url}
            title = item.get("title") or item.get("name") or item.get("label")
            if isinstance(title, str) and title.strip():
                out["title"] = title.strip()
            desc = item.get("description") or item.get("snippet") or item.get("content") or item.get("preview")
            if isinstance(desc, str) and desc.strip():
                out["description"] = desc.strip()
            normalized.append(out)
            continue

    return normalized


def _normalize_web_results(data: Any) -> list[dict]:
    items: list[Any]
    if isinstance(data, (list, tuple)):
        items = list(data)
    else:
        items = [data]

    flattened: list[Any] = []
    for item in items:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)

    normalized: list[dict] = []
    seen: set[str] = set()
    for item in flattened:
        if isinstance(item, str):
            for url in extract_unique_urls(item):
                if url in seen:
                    continue
                seen.add(url)
                normalized.append({"url": url})
            continue

        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("href") or item.get("link")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        result: dict[str, Any] = {"url": url}
        title = item.get("title") or item.get("name") or item.get("label")
        if isinstance(title, str) and title.strip():
            result["title"] = title.strip()
        preview = (
            item.get("preview")
            or item.get("description")
            or item.get("snippet")
            or item.get("content")
            or item.get("instructions")
        )
        if isinstance(preview, str) and preview.strip():
            result["preview"] = preview.strip()
        instructions = item.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            result["instructions"] = instructions.strip()
        normalized.append(result)

    return normalized


def _normalize_browse_pages(data: Any) -> list[dict]:
    results = _normalize_web_results(data)
    enriched: list[dict] = []
    for item in results:
        enriched.append(
            {
                **item,
                "high_value": True,
                "note": "高价值网页，建议优先打开或抓取正文。",
            }
        )
    return enriched


def _extract_sources_from_text(text: str) -> list[dict]:
    sources: list[dict] = []
    seen: set[str] = set()

    for title, url in _MD_LINK_PATTERN.findall(text or ""):
        url = (url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = (title or "").strip()
        if title:
            sources.append({"title": title, "url": url})
        else:
            sources.append({"url": url})

    for url in extract_unique_urls(text or ""):
        if url in seen:
            continue
        seen.add(url)
        sources.append({"url": url})

    return sources
