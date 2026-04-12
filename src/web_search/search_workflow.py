from __future__ import annotations

import asyncio
from typing import Any

from .config import config
from .errors import SearchExecutionError
from .logger import log_info
from .providers.grok import GrokSearchProvider
from .service_support import (
    call_firecrawl_search,
    call_tavily_search,
)
from .sources import (
    extract_search_artifacts,
    merge_sources,
    new_session_id,
    split_answer_and_sources,
)


def _extra_results_to_sources(
    tavily_results: list[dict] | None,
    firecrawl_results: list[dict] | None,
) -> list[dict]:
    sources: list[dict] = []
    seen: set[str] = set()
    for provider_name, results, desc_key in (
        ("firecrawl", firecrawl_results, "description"),
        ("tavily", tavily_results, "content"),
    ):
        for item in results or []:
            url = (item.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            source: dict[str, Any] = {"url": url, "provider": provider_name}
            title = (item.get("title") or "").strip()
            if title:
                source["title"] = title
            description = (item.get(desc_key) or "").strip()
            if description:
                source["description"] = description
            sources.append(source)
    return sources


def _build_sources_preview(sources: list[dict], limit: int = 3) -> list[dict]:
    preview: list[dict] = []
    for item in sources[:limit]:
        url = (item.get("url") or "").strip()
        if not url:
            continue
        current = {"url": url}
        for key in ("title", "provider"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                current[key] = value.strip()
        description = item.get("description")
        if isinstance(description, str) and description.strip():
            current["description"] = description.strip()[:240]
        preview.append(current)
    return preview


def _build_sparse_search_fallback(sources: list[dict]) -> str:
    preview = _build_sources_preview(sources)
    if not preview:
        return (
            "当前查询未返回足够完整的正文。"
            "这通常意味着主题较冷门、证据分散，或检索范围仍然过宽。"
            "建议缩小查询范围、补充实体名或时间范围后再试。"
        )
    lines = [
        "当前查询没有拿到足够完整的正文输出，但已经检索到一些可供核验的相关来源。",
        "这通常说明主题较冷门、证据分散，或上游模型在证据不足时选择了保守输出。",
        "",
        "可先核验这些来源：",
    ]
    for item in preview:
        label = item.get("title") or item["url"]
        lines.append(f"- [{label}]({item['url']})")
    lines.extend([
        "",
        "如果需要更完整的结论，建议缩小查询范围、补充实体名或时间范围，再继续搜索。",
    ])
    return "\n".join(lines)


def _build_search_response(
    prompt: str,
    model: str,
    sources: list[dict],
    *,
    content: str = "",
    include_summary: bool = False,
    status: str = "ok",
    error_code: str = "",
    error_message: str = "",
    answer_ready: bool | None = None,
    web_search_results: list[dict] | None = None,
    browse_pages: list[dict] | None = None,
) -> dict[str, Any]:
    normalized_content = (content or "").strip()
    normalized_web_results = web_search_results or []
    normalized_browse_pages = browse_pages or []
    default_ready = (
        status == "ok" and bool(normalized_content)
        if include_summary
        else status == "ok" and bool(normalized_web_results or normalized_browse_pages or sources)
    )
    response: dict[str, Any] = {
        "session_id": new_session_id(),
        "prompt": prompt,
        "model": model,
        "sources_count": len(sources),
        "status": status,
        "answer_ready": answer_ready if answer_ready is not None else default_ready,
        "webSearchResults": normalized_web_results,
        "webSearchResults_count": len(normalized_web_results),
        "browse_page": normalized_browse_pages,
        "browse_page_count": len(normalized_browse_pages),
    }
    if include_summary and normalized_content:
        response["content"] = normalized_content
        if status == "ok":
            response["summary_warning"] = "summary 不一定准确，最好结合 webSearchResults、browse_page 和 fetch 结果自行验证。"
    preview = _build_sources_preview(sources)
    if preview:
        response["sources_preview"] = preview
    if normalized_browse_pages:
        response["browse_page_guidance"] = "browse_page 中的网页是高价值网页，建议优先打开或抓取正文。"
    if status != "ok":
        response["error"] = {
            "code": error_code or "search_error",
            "message": error_message or normalized_content or "search failed",
            "retry_same_query": False,
        }
    return response


async def _execute_search(
    prompt: str,
    *,
    model: str = "",
    extra_sources: int = 0,
    ctx: Any = None,
    include_summary: bool,
) -> dict[str, Any]:
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        raise SearchExecutionError("prompt 不能为空")
    try:
        api_url = config.grok_api_url
        api_key = config.grok_api_key
    except ValueError as exc:
        return _build_search_response(
            cleaned_prompt,
            config.grok_model,
            [],
            content=f"配置错误: {exc}",
            include_summary=include_summary,
            status="error",
            error_code="config_error",
            error_message=f"配置错误: {exc}",
        )

    effective_model = config.grok_model
    provider = GrokSearchProvider(api_url, api_key, effective_model)
    has_tavily = config.tavily_enabled and bool(config.tavily_api_keys)
    has_firecrawl = bool(config.firecrawl_api_key)
    tavily_count = extra_sources if has_tavily else 0
    firecrawl_count = extra_sources if has_firecrawl and not tavily_count else 0

    async def safe_tavily() -> list[dict] | None:
        return await call_tavily_search(cleaned_prompt, tavily_count) if tavily_count else None

    async def safe_firecrawl() -> list[dict] | None:
        return await call_firecrawl_search(cleaned_prompt, firecrawl_count) if firecrawl_count else None

    coroutines: list[Any] = [provider.search(cleaned_prompt, ctx=ctx)]
    if tavily_count:
        coroutines.append(safe_tavily())
    if firecrawl_count:
        coroutines.append(safe_firecrawl())
    gathered = await asyncio.gather(*coroutines, return_exceptions=True)

    grok_outcome = gathered[0]
    tavily_results = gathered[1] if tavily_count else None
    firecrawl_results = gathered[2 if tavily_count else 1] if firecrawl_count else None
    tavily_results = None if isinstance(tavily_results, Exception) else tavily_results
    firecrawl_results = None if isinstance(firecrawl_results, Exception) else firecrawl_results

    if isinstance(grok_outcome, Exception):
        message = f"上游搜索失败: {type(grok_outcome).__name__}: {grok_outcome}"
        await log_info(ctx, message, config.debug_enabled)
        return _build_search_response(
            cleaned_prompt,
            effective_model,
            merge_sources(_extra_results_to_sources(tavily_results, firecrawl_results)),
            content=message,
            include_summary=include_summary,
            status="error",
            error_code="upstream_search_failed",
            error_message=message,
            answer_ready=False,
        )

    cleaned_output, web_search_results, browse_pages = extract_search_artifacts(grok_outcome or "")
    answer, grok_sources = split_answer_and_sources(cleaned_output)
    all_sources = merge_sources(
        browse_pages,
        web_search_results,
        grok_sources,
        _extra_results_to_sources(tavily_results, firecrawl_results),
    )
    if include_summary and not answer.strip():
        await log_info(ctx, "搜索未返回可用答案正文，已降级返回稀疏结果", config.debug_enabled)
        return _build_search_response(
            cleaned_prompt,
            effective_model,
            all_sources,
            content=_build_sparse_search_fallback(all_sources),
            include_summary=True,
            answer_ready=False,
            web_search_results=web_search_results,
            browse_pages=browse_pages,
        )
    return _build_search_response(
        cleaned_prompt,
        effective_model,
        all_sources,
        content=answer,
        include_summary=include_summary,
        web_search_results=web_search_results,
        browse_pages=browse_pages,
    )


async def search_web(
    prompt: str,
    *,
    model: str = "",
    extra_sources: int = 0,
    ctx: Any = None,
) -> dict[str, Any]:
    return await _execute_search(
        prompt,
        model=model,
        extra_sources=extra_sources,
        ctx=ctx,
        include_summary=False,
    )


async def search_web_with_summary(
    prompt: str,
    *,
    model: str = "",
    extra_sources: int = 0,
    ctx: Any = None,
) -> dict[str, Any]:
    return await _execute_search(
        prompt,
        model=model,
        extra_sources=extra_sources,
        ctx=ctx,
        include_summary=True,
    )
