import httpx
import pytest

import web_search
from web_search.prompts import build_search_prompt, classify_query_complexity
from web_search.providers import tavily as tavily_module
from web_search.providers.tavily import TavilyClient


def test_package_exports_mcp():
    assert hasattr(web_search, "mcp")
    assert web_search.mcp.name == "web-search"


def test_build_search_prompt_reflects_complexity_profile():
    profile = classify_query_complexity(
        "Compare Claude, Gemini, and GPT for coding agents, benchmarks, tool use, and production tradeoffs"
    )

    prompt = build_search_prompt(
        "Compare Claude, Gemini, and GPT for coding agents, benchmarks, tool use, and production tradeoffs"
    )

    assert profile.level == 3
    assert profile.mode == "deep"
    assert f"Complexity Level: {profile.level} ({profile.mode})" in prompt
    assert "Preferred Source Target" in prompt


@pytest.mark.asyncio
async def test_tavily_client_rotates_to_next_key_after_retryable_failure(monkeypatch):
    calls = []

    class _FakeClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, endpoint, headers, json):
            calls.append(headers["Authorization"])
            request = httpx.Request("POST", endpoint, headers=headers, json=json)
            if headers["Authorization"] == "Bearer key-1":
                return httpx.Response(500, request=request, json={"error": "boom"})
            return httpx.Response(
                200,
                request=request,
                json={
                    "results": [
                        {
                            "title": "Alpha",
                            "url": "https://example.com",
                            "content": "snippet",
                            "score": 0.9,
                        }
                    ]
                },
            )

    monkeypatch.setattr(tavily_module.httpx, "AsyncClient", _FakeClient)

    client = TavilyClient("https://api.tavily.com", ["key-1", "key-2"], cooldown_seconds=60)
    result = await client.search("latest uv release")

    assert result == [
        {
            "title": "Alpha",
            "url": "https://example.com",
            "content": "snippet",
            "score": 0.9,
        }
    ]
    assert calls == ["Bearer key-1", "Bearer key-2"]
    assert client._cooldowns["key-1"] > 0
