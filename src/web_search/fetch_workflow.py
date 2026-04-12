from __future__ import annotations

from typing import Any

from .config import config
from .errors import SearchExecutionError
from .logger import log_info
from .service_support import call_firecrawl_scrape, call_tavily_extract, get_tavily_client


async def fetch_url(url: str, ctx: Any = None) -> str:
    await log_info(ctx, f"Begin Fetch: {url}", config.debug_enabled)
    result = await call_tavily_extract(url)
    if result:
        await log_info(ctx, "Fetch Finished (Tavily)!", config.debug_enabled)
        return result
    await log_info(ctx, "Tavily unavailable or failed, trying Firecrawl...", config.debug_enabled)
    result = await call_firecrawl_scrape(url, ctx)
    if result:
        await log_info(ctx, "Fetch Finished (Firecrawl)!", config.debug_enabled)
        return result
    if not config.tavily_api_keys and not config.firecrawl_api_key:
        raise SearchExecutionError("配置错误: TAVILY_API_KEY / TAVILY_API_KEYS 和 FIRECRAWL_API_KEY 均未配置")
    raise SearchExecutionError(f"提取失败: {url}")


async def map_site(
    url: str,
    *,
    instructions: str = "",
    max_depth: int = 1,
    max_breadth: int = 20,
    limit: int = 50,
    timeout: int = 150,
) -> dict[str, Any]:
    client = get_tavily_client()
    if not client.is_configured:
        raise SearchExecutionError("配置错误: TAVILY_API_KEY / TAVILY_API_KEYS 未配置，请设置环境变量或本地 .env")
    try:
        data = await client.map(url, instructions, max_depth, max_breadth, limit, timeout)
    except Exception as exc:
        raise SearchExecutionError(f"映射失败: {exc}") from exc
    if not data:
        raise SearchExecutionError("映射失败: Tavily 未返回可用结果")
    return {
        "base_url": data.get("base_url", ""),
        "results": data.get("results", []),
        "response_time": data.get("response_time", 0),
    }
