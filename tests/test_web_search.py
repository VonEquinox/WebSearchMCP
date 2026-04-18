import pytest

import web_search.config as config_module
from web_search.config import config
from web_search import search_workflow, fetch_workflow
from web_search.sources import extract_search_artifacts
from web_search.service_support import _AVAILABLE_MODELS_CACHE


@pytest.fixture(autouse=True)
def _reset_runtime_state(monkeypatch, tmp_path):
    monkeypatch.setattr(config_module, "bundled_repo_root", lambda: tmp_path / "repo-root")
    config.reset_cache()
    _AVAILABLE_MODELS_CACHE.clear()
    yield
    config.reset_cache()
    _AVAILABLE_MODELS_CACHE.clear()


@pytest.mark.asyncio
async def test_search_web_returns_explicit_error_when_upstream_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("GROK_API_URL", "https://example.com")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_ENABLED", "false")
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WEB_SEARCH_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setattr(config, "_config_file", tmp_path / "config.json")
    config.reset_cache()

    class FailingProvider:
        async def search(self, query, platform="", ctx=None, planning_context=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(search_workflow, "GrokSearchProvider", lambda api_url, api_key, model: FailingProvider())

    result = await search_workflow.search_web("latest status")

    assert result["status"] == "error"
    assert result["answer_ready"] is False
    assert result["error"]["code"] == "upstream_search_failed"
    assert "boom" in result["error"]["message"]
    assert "content" not in result


@pytest.mark.asyncio
async def test_search_web_returns_sparse_fallback_for_empty_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("GROK_API_URL", "https://example.com")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_ENABLED", "false")
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WEB_SEARCH_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setattr(config, "_config_file", tmp_path / "config.json")
    config.reset_cache()

    class EmptyProvider:
        async def search(self, query, platform="", ctx=None, planning_context=None):
            return ""

    monkeypatch.setattr(search_workflow, "GrokSearchProvider", lambda api_url, api_key, model: EmptyProvider())

    result = await search_workflow.search_web("latest status")

    assert result["status"] == "ok"
    assert result["answer_ready"] is False
    assert "content" not in result


@pytest.mark.asyncio
async def test_search_web_with_summary_returns_sparse_fallback_for_empty_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("GROK_API_URL", "https://example.com")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_ENABLED", "false")
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WEB_SEARCH_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setattr(config, "_config_file", tmp_path / "config.json")
    config.reset_cache()

    class EmptyProvider:
        async def search(self, query, platform="", ctx=None, planning_context=None):
            return ""

    monkeypatch.setattr(search_workflow, "GrokSearchProvider", lambda api_url, api_key, model: EmptyProvider())

    result = await search_workflow.search_web_with_summary("latest status")

    assert result["status"] == "ok"
    assert result["answer_ready"] is False
    assert "当前查询未返回足够完整的正文" in result["content"]
    assert result["summary_warning"]


@pytest.mark.asyncio
async def test_fetch_url_falls_back_to_curl_cffi_when_tavily_returns_none(monkeypatch):
    async def fake_tavily_extract(_: str):
        return None

    async def fake_curl_extract(_: str, ctx=None):
        return "curl fallback content"

    monkeypatch.setattr(fetch_workflow, "call_tavily_extract", fake_tavily_extract)
    monkeypatch.setattr(fetch_workflow, "call_curl_cffi_extract", fake_curl_extract)

    result = await fetch_workflow.fetch_url("https://example.com")

    assert result == "curl fallback content"


@pytest.mark.asyncio
async def test_fetch_url_raises_when_tavily_and_curl_cffi_both_fail(monkeypatch):
    async def fake_tavily_extract(_: str):
        return None

    async def fake_curl_extract(_: str, ctx=None):
        return None

    monkeypatch.setattr(fetch_workflow, "call_tavily_extract", fake_tavily_extract)
    monkeypatch.setattr(fetch_workflow, "call_curl_cffi_extract", fake_curl_extract)

    with pytest.raises(fetch_workflow.SearchExecutionError, match="提取失败: https://example.com"):
        await fetch_workflow.fetch_url("https://example.com")


@pytest.mark.asyncio
async def test_search_web_always_uses_fixed_reasoning_model(monkeypatch, tmp_path):
    monkeypatch.setenv("GROK_API_URL", "https://example.com")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.setenv("GROK_MODEL", "legacy-model")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WEB_SEARCH_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setattr(config, "_config_file", tmp_path / "config.json")
    config.reset_cache()

    captured = {}

    class InspectingProvider:
        async def search(self, query, platform="", ctx=None, planning_context=None):
            return "Final answer"

    def fake_provider(api_url, api_key, model):
        captured["model"] = model
        return InspectingProvider()

    monkeypatch.setattr(search_workflow, "GrokSearchProvider", fake_provider)

    result = await search_workflow.search_web("latest status", model="deepseek-chat")

    assert result["status"] == "ok"
    assert result["model"] == "grok-4.20-0309-reasoning"
    assert captured["model"] == "grok-4.20-0309-reasoning"


def test_iter_env_files_includes_repo_root_env(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo-root"
    repo_root.mkdir()
    repo_env = repo_root / ".env"
    repo_env.write_text("GROK_API_URL=https://example.com\n", encoding="utf-8")

    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(config, "_config_file", tmp_path / "config.json")
    config.reset_cache()

    env_files = config._iter_env_files()

    assert repo_env in env_files


def test_extract_search_artifacts_parses_web_results_and_browse_pages():
    text = (
        '[webSearchResults][{"url":"https://example.com/a","title":"Alpha","preview":"alpha preview"}]\n'
        '[browse_page]{"url":"https://example.com/b","title":"Beta","preview":"beta preview"}\n'
        'Final answer\n'
        '[1] https://example.com/a'
    )

    cleaned, web_results, browse_pages = extract_search_artifacts(text)

    assert cleaned.startswith("Final answer")
    assert web_results == [
        {
            "url": "https://example.com/a",
            "title": "Alpha",
            "preview": "alpha preview",
        }
    ]
    assert browse_pages == [
        {
            "url": "https://example.com/b",
            "title": "Beta",
            "preview": "beta preview",
            "high_value": True,
            "note": "高价值网页，建议优先打开或抓取正文。",
        }
    ]


def test_extract_search_artifacts_ignores_non_explicit_browse_page_text():
    text = (
        "Summary paragraph.\n\n"
        "**Browse_page candidates for deeper details:**\n"
        "- Official Docs: https://example.com/docs (latest specs)\n"
        "- API Guide: https://example.com/api (integration steps)\n"
    )

    cleaned, web_results, browse_pages = extract_search_artifacts(text)

    assert cleaned == text.strip()
    assert web_results == []
    assert browse_pages == []


def test_extract_search_artifacts_ignores_browse_page_array_form():
    text = '[browse_page][{"url":"https://example.com/b","title":"Beta"}]\nFinal answer'

    cleaned, web_results, browse_pages = extract_search_artifacts(text)

    assert cleaned == text
    assert web_results == []
    assert browse_pages == []


@pytest.mark.asyncio
async def test_search_web_returns_structured_web_results_and_browse_pages(monkeypatch, tmp_path):
    monkeypatch.setenv("GROK_API_URL", "https://example.com")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WEB_SEARCH_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setattr(config, "_config_file", tmp_path / "config.json")
    config.reset_cache()

    class StructuredProvider:
        async def search(self, query, platform="", ctx=None, planning_context=None):
            return (
                '[webSearchResults][{"url":"https://example.com/a","title":"Alpha","preview":"alpha preview"}]\n'
                '[browse_page]{"url":"https://example.com/b","title":"Beta","preview":"beta preview"}\n'
                'Final answer\n'
                '[1] https://example.com/a'
            )

    monkeypatch.setattr(search_workflow, "GrokSearchProvider", lambda api_url, api_key, model: StructuredProvider())

    result = await search_workflow.search_web("latest status")

    assert result["status"] == "ok"
    assert "content" not in result
    assert result["webSearchResults_count"] == 1
    assert result["browse_page_count"] == 1
    assert result["browse_page_guidance"]
    assert result["sources_count"] == 2


@pytest.mark.asyncio
async def test_search_web_with_summary_returns_summary_and_structured_results(monkeypatch, tmp_path):
    monkeypatch.setenv("GROK_API_URL", "https://example.com")
    monkeypatch.setenv("GROK_API_KEY", "test-key")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WEB_SEARCH_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.setattr(config, "_config_file", tmp_path / "config.json")
    config.reset_cache()

    class StructuredProvider:
        async def search(self, query, platform="", ctx=None, planning_context=None):
            return (
                '[webSearchResults][{"url":"https://example.com/a","title":"Alpha","preview":"alpha preview"}]\n'
                '[browse_page]{"url":"https://example.com/b","title":"Beta","preview":"beta preview"}\n'
                'Final answer\n'
                '[1] https://example.com/a'
            )

    monkeypatch.setattr(search_workflow, "GrokSearchProvider", lambda api_url, api_key, model: StructuredProvider())

    result = await search_workflow.search_web_with_summary("latest status")

    assert result["status"] == "ok"
    assert result["content"].startswith("Final answer")
    assert result["summary_warning"]
    assert result["webSearchResults_count"] == 1
    assert result["browse_page_count"] == 1
