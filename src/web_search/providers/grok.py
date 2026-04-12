import httpx
import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, List, Optional
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential
from tenacity.wait import wait_base
from zoneinfo import ZoneInfo
from .base import BaseSearchProvider, SearchResult
from ..utils import (
    fetch_prompt,
    url_describe_prompt,
    rank_sources_prompt,
)
from ..logger import log_info
from ..config import build_openai_chat_completions_url, config


def get_local_time_info() -> str:
    """获取本地时间信息，用于注入到搜索查询中"""
    try:
        # 尝试获取系统本地时区
        local_tz = datetime.now().astimezone().tzinfo
        local_now = datetime.now(local_tz)
    except Exception:
        # 降级使用 UTC
        local_now = datetime.now(timezone.utc)

    # 格式化时间信息
    weekdays_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays_cn[local_now.weekday()]

    return (
        f"[Current Time Context]\n"
        f"- Date: {local_now.strftime('%Y-%m-%d')} ({weekday})\n"
        f"- Time: {local_now.strftime('%H:%M:%S')}\n"
        f"- Timezone: {local_now.tzname() or 'Local'}\n"
    )


def _needs_time_context(query: str) -> bool:
    """检查查询是否需要时间上下文"""
    # 中文时间相关关键词
    cn_keywords = [
        "当前", "现在", "今天", "明天", "昨天",
        "本周", "上周", "下周", "这周",
        "本月", "上月", "下月", "这个月",
        "今年", "去年", "明年",
        "最新", "最近", "近期", "刚刚", "刚才",
        "实时", "即时", "目前",
    ]
    # 英文时间相关关键词
    en_keywords = [
        "current", "now", "today", "tomorrow", "yesterday",
        "this week", "last week", "next week",
        "this month", "last month", "next month",
        "this year", "last year", "next year",
        "latest", "recent", "recently", "just now",
        "real-time", "realtime", "up-to-date",
    ]

    query_lower = query.lower()

    for keyword in cn_keywords:
        if keyword in query:
            return True

    for keyword in en_keywords:
        if keyword in query_lower:
            return True

    return False

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_THINK_BLOCK_RE = re.compile(r"<think>\s*.*?\s*</think>\s*", re.IGNORECASE | re.DOTALL)
_LEADING_THINK_RE = re.compile(r"^\s*<think>\s*", re.IGNORECASE)
_TRAILING_THINK_RE = re.compile(r"\s*</think>\s*$", re.IGNORECASE)


def _is_retryable_exception(exc) -> bool:
    """检查异常是否可重试"""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


def _coerce_text_content(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "".join(_coerce_text_content(item) for item in value)

    if isinstance(value, dict):
        text_type = str(value.get("type", "")).lower()
        if text_type in {"text", "output_text"} and isinstance(value.get("text"), str):
            return value["text"]

        for key in ("content", "text", "value", "output_text", "answer"):
            extracted = _coerce_text_content(value.get(key))
            if extracted:
                return extracted
        return ""

    return str(value)


def _extract_choice_text(choice: dict[str, Any]) -> str:
    for container_name in ("delta", "message"):
        container = choice.get(container_name)
        if isinstance(container, dict):
            for field in ("content", "text", "output_text", "answer"):
                extracted = _coerce_text_content(container.get(field))
                if extracted:
                    return extracted

    for field in ("content", "text", "output_text", "answer"):
        extracted = _coerce_text_content(choice.get(field))
        if extracted:
            return extracted

    return ""


def _unwrap_response_payload(payload: Any) -> Any:
    current = payload
    while isinstance(current, dict) and isinstance(current.get("data"), dict):
        current = current["data"]
    return current


def _extract_response_text(payload: Any) -> str:
    unwrapped = _unwrap_response_payload(payload)

    if isinstance(unwrapped, dict):
        choices = unwrapped.get("choices")
        if isinstance(choices, list):
            text = "".join(
                _extract_choice_text(choice) for choice in choices if isinstance(choice, dict)
            )
            if text:
                return text

        for field in ("output_text", "answer", "content", "text"):
            extracted = _coerce_text_content(unwrapped.get(field))
            if extracted:
                return extracted

    return _coerce_text_content(unwrapped)


def _sanitize_model_output(content: str) -> str:
    without_think_blocks = _THINK_BLOCK_RE.sub("", content or "")
    without_unclosed_leading = _LEADING_THINK_RE.sub("", without_think_blocks)
    without_orphan_trailing = _TRAILING_THINK_RE.sub("", without_unclosed_leading)
    return without_orphan_trailing.strip()


class _WaitWithRetryAfter(wait_base):
    """等待策略：优先使用 Retry-After 头，否则使用指数退避"""

    def __init__(self, multiplier: float, max_wait: int):
        self._base_wait = wait_random_exponential(multiplier=multiplier, max=max_wait)
        self._protocol_error_base = 3.0

    def __call__(self, retry_state):
        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                retry_after = self._parse_retry_after(exc.response)
                if retry_after is not None:
                    return retry_after
            if isinstance(exc, httpx.RemoteProtocolError):
                return self._base_wait(retry_state) + self._protocol_error_base
        return self._base_wait(retry_state)

    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        """解析 Retry-After 头（支持秒数或 HTTP 日期格式）"""
        header = response.headers.get("Retry-After")
        if not header:
            return None
        header = header.strip()

        if header.isdigit():
            return float(header)

        try:
            retry_dt = parsedate_to_datetime(header)
            if retry_dt.tzinfo is None:
                retry_dt = retry_dt.replace(tzinfo=timezone.utc)
            delay = (retry_dt - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, delay)
        except (TypeError, ValueError):
            return None


class GrokSearchProvider(BaseSearchProvider):
    def __init__(self, api_url: str, api_key: str, model: str = "grok-4.20-0309-reasoning"):
        super().__init__(api_url, api_key)
        self.model = model

    def get_provider_name(self) -> str:
        return "Grok"

    async def search(
        self,
        query: str,
        platform: str = "",
        min_results: int = 3,
        max_results: int = 10,
        ctx=None,
        planning_context: Optional[dict] = None,
    ) -> List[SearchResult]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = [{"role": "user", "content": query}]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        await log_info(
            ctx,
            f"raw_search_prompt={query}",
            config.debug_enabled,
        )

        return await self._execute_stream_with_retry(headers, payload, ctx)

    async def fetch(self, url: str, ctx=None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": fetch_prompt,
                },
                {"role": "user", "content": url + "\n获取该网页内容并返回其结构化Markdown格式" },
            ],
            "stream": True,
        }
        return await self._execute_stream_with_retry(headers, payload, ctx)

    async def _parse_streaming_response(self, response, ctx=None) -> str:
        content = ""
        full_body_buffer = []
        
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            
            full_body_buffer.append(line)

            # 兼容 "data: {...}" 和 "data:{...}" 两种 SSE 格式
            if line.startswith("data:"):
                if line in ("data: [DONE]", "data:[DONE]"):
                    continue
                try:
                    # 去掉 "data:" 前缀，并去除可能的空格
                    json_str = line[5:].lstrip()
                    data = json.loads(json_str)
                    chunk_text = _extract_response_text(data)
                    if chunk_text:
                        content += chunk_text
                except (json.JSONDecodeError, IndexError):
                    continue
                
        if not content and full_body_buffer:
            try:
                full_text = "".join(full_body_buffer)
                if full_text.startswith("data:"):
                    parsed_chunks = []
                    for buffered_line in full_body_buffer:
                        if not buffered_line.startswith("data:") or buffered_line in ("data: [DONE]", "data:[DONE]"):
                            continue
                        try:
                            parsed_chunks.append(json.loads(buffered_line[5:].lstrip()))
                        except json.JSONDecodeError:
                            continue
                    content = "".join(_extract_response_text(chunk) for chunk in parsed_chunks)
                else:
                    data = json.loads(full_text)
                    content = _extract_response_text(data)
            except json.JSONDecodeError:
                pass

        content = _sanitize_model_output(content)
        
        await log_info(ctx, f"content: {content}", config.debug_enabled)

        return content

    async def _execute_stream_with_retry(self, headers: dict, payload: dict, ctx=None) -> str:
        """执行带重试机制的流式 HTTP 请求"""
        timeout = httpx.Timeout(connect=6.0, read=120.0, write=10.0, pool=None)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(config.retry_max_attempts + 1),
                wait=_WaitWithRetryAfter(config.retry_multiplier, config.retry_max_wait),
                retry=retry_if_exception(_is_retryable_exception),
                reraise=True,
            ):
                with attempt:
                    async with client.stream(
                        "POST",
                        build_openai_chat_completions_url(self.api_url),
                        headers=headers,
                        json=payload,
                    ) as response:
                        response.raise_for_status()
                        return await self._parse_streaming_response(response, ctx)

    async def describe_url(self, url: str, ctx=None) -> dict:
        """让 Grok 阅读单个 URL 并返回 title + extracts"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": url_describe_prompt},
                {"role": "user", "content": url},
            ],
            "stream": True,
        }
        result = await self._execute_stream_with_retry(headers, payload, ctx)
        title, extracts = url, ""
        for line in result.strip().splitlines():
            if line.startswith("Title:"):
                title = line[6:].strip() or url
            elif line.startswith("Extracts:"):
                extracts = line[9:].strip()
        return {"title": title, "extracts": extracts, "url": url}

    async def rank_sources(self, query: str, sources_text: str, total: int, ctx=None) -> list[int]:
        """让 Grok 按查询相关度对信源排序，返回排序后的序号列表"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": rank_sources_prompt},
                {"role": "user", "content": f"Query: {query}\n\n{sources_text}"},
            ],
            "stream": True,
        }
        result = await self._execute_stream_with_retry(headers, payload, ctx)
        order: list[int] = []
        seen: set[int] = set()
        for token in result.strip().split():
            try:
                n = int(token)
                if 1 <= n <= total and n not in seen:
                    seen.add(n)
                    order.append(n)
            except ValueError:
                continue
        # 补齐遗漏的序号
        for i in range(1, total + 1):
            if i not in seen:
                order.append(i)
        return order
