import pytest
from pathlib import Path

from web_search import guide, server, service_support


def test_get_usage_guide_mentions_uv_run_and_non_search_mcp():
    result = server.get_usage_guide()
    expected_root = str(Path(__file__).resolve().parents[1])

    assert "does not run web search directly" in result["purpose"]
    assert "uv run --project" in result["purpose"]
    assert result["repository_root"] == expected_root
    assert result["commands"]["search"].startswith(f"uv run --project {expected_root}")
    assert "doctor" in result["commands"]
    assert "<FULL_PROMPT>" in result["commands"]["search"]
    assert "<FULL_PROMPT>" in result["commands"]["searchwithsummary"]
    assert "may be inaccurate" in result["command_notes"]["searchwithsummary"]
    assert "must_split_when" in result["complex_research_rules"]
    assert "Do not send one giant prompt" in result["complex_research_rules"]["do_not"][0]


def test_recommend_command_prefers_fetch_for_known_url():
    result = server.recommend_command("Read https://docs.astral.sh/uv/ and summarize it")
    expected_root = str(Path(__file__).resolve().parents[1])

    assert result["recommended_command"] == f"uv run --project {expected_root} web-search-cli fetch https://docs.astral.sh/uv/"


def test_recommend_command_returns_prompt_template_for_search_tasks():
    result = guide.recommend_command("Find RAG benchmark papers and arXiv sources")
    expected_root = str(Path(__file__).resolve().parents[1])

    assert result["recommended_command"] == f'uv run --project {expected_root} web-search-cli search "<FULL_PROMPT>"'
    assert "original papers" in result["prompt_guidance"]


def test_recommend_command_uses_searchwithsummary_for_summary_tasks():
    result = guide.recommend_command("Summarize the latest Pixel 10 news with sources")
    expected_root = str(Path(__file__).resolve().parents[1])

    assert result["recommended_command"] == f'uv run --project {expected_root} web-search-cli searchwithsummary "<FULL_PROMPT>"'
    assert "不一定准确" in result["summary_warning"]


def test_recommend_command_marks_complex_research_for_decomposition():
    result = guide.recommend_command(
        "Find primary sources (papers, arXiv, conference pages, official lab blogs) about whether continued training "
        "or domain-specific finetuning of LLMs causes catastrophic forgetting, negative transfer, or cross-domain "
        "skill degradation. Focus on: (a) data quality effects; (b) math reasoning or code skill degradation."
    )

    assert result["decomposition_required"] is True
    assert result["orchestration_strategy"] == "split_then_search"
    assert result["single_shot_not_recommended"] is True
    assert "too broad for a single search prompt" in result["orchestration_warning"]
    assert any("one `search` command per sub-query" in item for item in result["decomposition_rules"])


def test_in_repo_root_only_true_for_actual_repo_root(tmp_path):
    assert service_support.in_repo_root(Path(__file__).resolve().parents[1]) is True
    assert service_support.in_repo_root(tmp_path) is False


@pytest.mark.asyncio
async def test_doctor_info_omits_model_fields(monkeypatch, tmp_path):
    expected_root = str(Path(__file__).resolve().parents[1])
    monkeypatch.setenv("GROK_API_URL", "https://example.com")
    monkeypatch.setenv("GROK_API_KEY", "test-key")

    async def fake_connection():
        return {"status": "✅ 连接成功", "message": "API 已连通，固定搜索模型检查通过。", "response_time_ms": 12.3}

    monkeypatch.setattr(service_support, "test_grok_connection", fake_connection)

    result = await service_support.get_doctor_info()

    assert result["repository_root"] == expected_root
    assert "grok_model" not in result["configuration"]
    assert "model_available" not in result["configuration"]
    assert "recommended_model" not in result["configuration"]
    assert "available_models" not in result["connection_test"]


@pytest.mark.asyncio
async def test_doctor_tool_returns_next_step(monkeypatch):
    async def fake_doctor_info():
        return {"working_directory": {"is_repo_root": True}}

    monkeypatch.setattr(guide, "get_doctor_info", fake_doctor_info)

    result = await server.doctor()

    assert result["working_directory"]["is_repo_root"] is True
    assert "recommend_command" in result["next_step"]
