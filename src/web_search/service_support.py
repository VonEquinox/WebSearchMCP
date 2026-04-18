from __future__ import annotations

import asyncio
import shlex
import time
from pathlib import Path
from typing import Any

import httpx

from .config import build_openai_models_url, bundled_repo_root, config
from .providers.tavily import TavilyClient

_AVAILABLE_MODELS_CACHE: dict[tuple[str, str], list[str]] = {}
_AVAILABLE_MODELS_LOCK = asyncio.Lock()
_TAVILY_CLIENT: TavilyClient | None = None
_TAVILY_CLIENT_FINGERPRINT: tuple[str, tuple[str, ...], int] | None = None


def repo_root(start: Path | None = None) -> Path:
    fallback = bundled_repo_root().resolve()
    current = (start or fallback).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return fallback


def in_repo_root(path: Path | None = None) -> bool:
    return (path or Path.cwd()).resolve() == mcp_repository_root()


def mcp_repository_root() -> Path:
    return bundled_repo_root().resolve()


def cli_command_prefix() -> str:
    return f"uv run --project {shlex.quote(str(mcp_repository_root()))} web-search-cli"


async def _fetch_available_models(api_url: str, api_key: str) -> list[str]:
    models_url = build_openai_models_url(api_url)
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            models_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()
    return [
        item["id"]
        for item in (data or {}).get("data", []) or []
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


async def get_available_models(api_url: str | None = None, api_key: str | None = None) -> list[str]:
    resolved_api_url = api_url or config.grok_api_url
    resolved_api_key = api_key or config.grok_api_key
    key = (resolved_api_url, resolved_api_key)
    async with _AVAILABLE_MODELS_LOCK:
        cached = _AVAILABLE_MODELS_CACHE.get(key)
        if cached is not None:
            return cached
    try:
        models = await _fetch_available_models(resolved_api_url, resolved_api_key)
    except Exception:
        models = []
    async with _AVAILABLE_MODELS_LOCK:
        _AVAILABLE_MODELS_CACHE[key] = models
    return models


async def test_grok_connection() -> dict[str, Any]:
    result: dict[str, Any] = {"status": "未测试", "message": "", "response_time_ms": 0}
    try:
        api_url = config.grok_api_url
        api_key = config.grok_api_key
        models_url = build_openai_models_url(api_url)
        start = time.time()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                models_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        result["response_time_ms"] = round((time.time() - start) * 1000, 2)
        if response.status_code != 200:
            result["status"] = "⚠️ 连接异常"
            result["message"] = f"HTTP {response.status_code}: {response.text[:100]}"
            return result
        payload = response.json()
        models = [
            item["id"]
            for item in payload.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        if models and config.grok_model not in models:
            result["status"] = "⚠️ 固定模型不可用"
            result["message"] = "API 已连通，但固定搜索模型当前不可用，请检查上游配置。"
            return result
        result["status"] = "✅ 连接成功"
        result["message"] = "API 已连通，固定搜索模型检查通过。"
        return result
    except httpx.TimeoutException:
        result["status"] = "❌ 连接超时"
        result["message"] = "请求超时（10秒），请检查网络连接或 API URL"
    except httpx.RequestError as exc:
        result["status"] = "❌ 连接失败"
        result["message"] = f"网络错误: {exc}"
    except ValueError as exc:
        result["status"] = "❌ 配置错误"
        result["message"] = str(exc)
    except Exception as exc:
        result["status"] = "❌ 测试失败"
        result["message"] = f"未知错误: {exc}"
    return result


def _display_env_files() -> list[str]:
    root_env = repo_root() / ".env"
    config_env = Path.home() / ".config" / "web-search" / ".env"
    labels: list[str] = []
    for path in config._iter_env_files():
        if not path.exists():
            continue
        resolved = path.resolve(strict=False)
        if resolved == root_env.resolve(strict=False):
            labels.append("./.env")
        elif resolved == config_env.resolve(strict=False):
            labels.append("~/.config/web-search/.env")
        else:
            labels.append(path.name)
    return labels


async def get_doctor_info() -> dict[str, Any]:
    try:
        grok_api_url_configured = bool(config.grok_api_url)
    except ValueError:
        grok_api_url_configured = False
    try:
        grok_api_key_configured = bool(config.grok_api_key)
    except ValueError:
        grok_api_key_configured = False
    connection_test = await test_grok_connection()
    command_prefix = cli_command_prefix()
    root = str(mcp_repository_root())
    return {
        "repository_root": root,
        "working_directory": {
            "rule": "You can run the provided commands from any directory because they already include --project <repository_root>.",
            "is_repo_root": in_repo_root(),
        },
        "configuration": {
            "grok_api_url_configured": grok_api_url_configured,
            "grok_api_key_configured": grok_api_key_configured,
            "tavily_enabled": config.tavily_enabled,
            "tavily_api_keys_count": len(config.tavily_api_keys),
            "firecrawl_configured": bool(config.firecrawl_api_key),
            "env_files_loaded": _display_env_files(),
        },
        "connection_test": connection_test,
        "quickstart": [
            f"{command_prefix} doctor",
            f'{command_prefix} search "<FULL_PROMPT>"',
            f'{command_prefix} searchwithsummary "<FULL_PROMPT>"  # summary 仅供草稿参考，最好自行验证',
            f'{command_prefix} fetch "<url>"',
        ],
    }


def get_tavily_client() -> TavilyClient:
    global _TAVILY_CLIENT, _TAVILY_CLIENT_FINGERPRINT
    fingerprint = (
        config.tavily_api_url,
        tuple(config.tavily_api_keys),
        config.tavily_key_cooldown_seconds,
    )
    if _TAVILY_CLIENT is None or _TAVILY_CLIENT_FINGERPRINT != fingerprint:
        _TAVILY_CLIENT = TavilyClient(
            api_url=fingerprint[0],
            api_keys=list(fingerprint[1]),
            cooldown_seconds=fingerprint[2],
        )
        _TAVILY_CLIENT_FINGERPRINT = fingerprint
    return _TAVILY_CLIENT


async def call_tavily_search(query: str, max_results: int = 6) -> list[dict] | None:
    client = get_tavily_client()
    if not client.is_configured:
        return None
    return await client.search(query, max_results)


async def call_tavily_extract(url: str) -> str | None:
    client = get_tavily_client()
    if not client.is_configured:
        return None
    return await client.extract(url)


async def call_firecrawl_search(query: str, limit: int = 14) -> list[dict] | None:
    api_key = config.firecrawl_api_key
    if not api_key:
        return None
    endpoint = f"{config.firecrawl_api_url.rstrip('/')}/search"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(endpoint, headers=headers, json={"query": query, "limit": limit})
        response.raise_for_status()
        data = response.json()
    results = data.get("data", {}).get("web", [])
    if not results:
        return None
    return [
        {"title": item.get("title", ""), "url": item.get("url", ""), "description": item.get("description", "")}
        for item in results
    ]
