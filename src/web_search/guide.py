from __future__ import annotations

import re
import shlex
from typing import Any

from .runtime import get_doctor_info
from .service_support import cli_command_prefix, mcp_repository_root

_URL_RE = re.compile(r"https?://\S+")
_GITHUB_RE = re.compile(r"github\.com/([\w.-]+/[\w.-]+)")
_SYMBOL_RE = re.compile(r"[`\"']([A-Za-z_][A-Za-z0-9_./:-]*)[`\"']")


def _quote(value: str) -> str:
    return shlex.quote(value)


def _extract_url(task: str) -> str:
    match = _URL_RE.search(task)
    return match.group(0).rstrip(").,]") if match else ""


def _extract_repo(task: str) -> str:
    match = _GITHUB_RE.search(task)
    return match.group(1) if match else ""




def _extract_symbol(task: str) -> str:
    match = _SYMBOL_RE.search(task)
    if match:
        return match.group(1)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_./:-]{2,}", task)
    return tokens[-1] if tokens else "<pattern>"


def _is_complex_research_task(task: str) -> bool:
    lowered = (task or "").lower()
    score = 0

    if len(task) >= 220:
        score += 1
    if any(marker in lowered for marker in ("focus on", "cover", "including", "(a)", "(b)", "(c)")):
        score += 1
    if lowered.count(" and ") + lowered.count(" or ") >= 4:
        score += 1

    research_markers = (
        "primary source",
        "papers",
        "paper",
        "arxiv",
        "conference",
        "official lab",
        "catastrophic forgetting",
        "negative transfer",
        "degradation",
        "reasoning",
        "code skill",
        "whether",
        "compare",
    )
    if sum(marker in lowered for marker in research_markers) >= 3:
        score += 1

    return score >= 2


def build_usage_guide() -> dict[str, Any]:
    root = str(mcp_repository_root())
    command_prefix = cli_command_prefix()
    return {
        "repository_root": root,
        "command_prefix": command_prefix,
        "purpose": (
            "This MCP does not run web search directly. It teaches the model how to use this repository's "
            "bash/uv CLI commands. When search is needed, the calling model must author the full search prompt "
            "itself, then execute the provided `uv run --project <repository_root> web-search-cli ...` commands. "
            "The absolute repository root is returned in `repository_root`."
        ),
        "working_directory_rule": "The command templates already include the absolute repository path, so they can run from any directory.",
        "workflow": [
            "1. Call doctor to verify configuration and repository state.",
            "2. Call recommend_command with the user's task.",
            "3. If recommend_command returns `<FULL_PROMPT>`, replace it with a fully written search prompt before execution.",
            "4. Execute the returned command with bash; the command already pins the project path.",
            "5. If the first command is insufficient, use the fallback commands from recommend_command.",
        ],
        "commands": {
            "search": f'{command_prefix} search "<FULL_PROMPT>"',
            "searchwithsummary": f'{command_prefix} searchwithsummary "<FULL_PROMPT>"',
            "fetch": f'{command_prefix} fetch "<url>"',
            "map": f'{command_prefix} map --instructions "only documentation pages" "<url>"',
            "models": f"{command_prefix} models",
            "doctor": f"{command_prefix} doctor",
            "github_api": f"{command_prefix} github-api /repos/<owner>/<repo>",
            "repo_file": f"{command_prefix} repo-file <owner>/<repo> <path>",
            "repo_grep": f'{command_prefix} repo-grep <owner>/<repo> "<pattern>"',
        },
        "command_notes": {
            "search": "Return structured webSearchResults and explicit [browse_page]{...} only; do not rely on it for a summary.",
            "searchwithsummary": "Also returns a draft summary, but the summary may be inaccurate and should be verified with webSearchResults, browse_page, and fetch.",
        },
        "selection_rules": [
            "Use `search` when you need the upstream model to execute a caller-authored web research prompt and return structured results without a summary.",
            "Use `searchwithsummary` only when a draft summary is useful; that summary may be inaccurate, so verify it yourself.",
            "If the task is a broad research question with multiple hypotheses, source classes, focus areas, or evaluation dimensions, do not single-shot it into one search prompt.",
            "The full prompt should include scope, source preferences, date constraints, and output requirements.",
            "If `search` or `searchwithsummary` returns `browse_page`, treat those URLs as high-value pages and prioritize opening or fetching them.",
            "Use `fetch` when the exact target URL is already known.",
            "Use `map` only for docs-style site structure exploration.",
            "Use `github-api`, `repo-file`, or `repo-grep` once the relevant GitHub repository is known.",
        ],
        "complex_research_rules": {
            "must_split_when": [
                "The task asks about more than one hypothesis, failure mode, skill area, or evidence source class.",
                "The task contains lists like (a)/(b), multiple focus clauses, or a long compare/survey-style question.",
                "The task needs primary-source discovery first and synthesis later.",
            ],
            "workflow": [
                "Split the problem into 2-4 narrow sub-queries before running any search command.",
                "Each sub-query should isolate one question or one evidence dimension only.",
                "Run one `search` command per sub-query for source discovery; do not ask a single search call to synthesize everything.",
                "After identifying high-value pages, use `fetch` on the best pages before writing the final synthesis.",
            ],
            "do_not": [
                "Do not send one giant prompt that mixes many hypotheses, domains, source types, and evaluation angles into a single search call."
            ],
        },
    }


def recommend_command(task: str) -> dict[str, Any]:
    cleaned = (task or "").strip()
    lowered = cleaned.lower()
    url = _extract_url(cleaned)
    repo = _extract_repo(cleaned)
    symbol = _extract_symbol(cleaned)
    wants_summary = any(keyword in lowered for keyword in ("summary", "summarize", "总结", "概述", "摘要"))
    complex_research = _is_complex_research_task(cleaned)
    root = str(mcp_repository_root())
    command_prefix = cli_command_prefix()

    command = f'{command_prefix} search "<FULL_PROMPT>"'
    fallback = []
    reason = "Search expects a fully authored prompt from the calling model."
    prompt_guidance = (
        "Write the full prompt yourself. Include the user goal, scope limits, preferred source types, any date "
        "constraints, and the desired answer format."
    )

    if url and any(keyword in lowered for keyword in ("map", "site map", "sitemap", "站点结构", "导航")):
        command = f"{command_prefix} map {_quote(url)}"
        fallback = [f"{command_prefix} fetch {_quote(url)}"]
        reason = "A known docs-like URL with structure/navigation intent fits `map`."
        prompt_guidance = ""
    elif url:
        command = f"{command_prefix} fetch {_quote(url)}"
        fallback = [f'{command_prefix} search "<FULL_PROMPT>"']
        reason = "A known target URL should be fetched directly before broader search."
        prompt_guidance = ""
    elif repo and any(keyword in lowered for keyword in ("release", "stars", "metadata", "repo info", "仓库信息", "发布")):
        command = f"{command_prefix} github-api /repos/{repo}"
        fallback = [f"{command_prefix} repo-file {repo} README.md"]
        reason = "The task asks for GitHub metadata rather than general web search."
        prompt_guidance = ""
    elif repo and any(keyword in lowered for keyword in ("readme", "file", "contents", "目录", "文件")):
        command = f"{command_prefix} repo-file {repo} README.md"
        fallback = [f"{command_prefix} github-api /repos/{repo}/contents"]
        reason = "The task mentions a known repository file or directory."
        prompt_guidance = ""
    elif repo and any(keyword in lowered for keyword in ("implementation", "source code", "grep", "search in repo", "实现", "源码", "出现位置")):
        command = f"{command_prefix} repo-grep {repo} {_quote(symbol)}"
        fallback = [f"{command_prefix} repo-file {repo} README.md"]
        reason = "Implementation-detail questions should inspect the repository directly."
        prompt_guidance = ""
    elif wants_summary:
        command = f'{command_prefix} searchwithsummary "<FULL_PROMPT>"'
        fallback = [f'{command_prefix} search "<FULL_PROMPT>"']
        reason = "The task explicitly asks for a summary, so the draft-summary variant is a better first pass."
    elif any(keyword in lowered for keyword in ("paper", "papers", "arxiv", "论文", "学术", "benchmark")):
        reason = "This is still a search task, but the calling model should write a paper-first prompt."
        prompt_guidance = (
            "Write a full prompt that explicitly prioritizes original papers, arXiv pages, publisher pages, "
            "benchmarks, and official repositories."
        )
    elif any(keyword in lowered for keyword in ("official", "api", "docs", "spec", "标准", "官方", "文档")):
        reason = "This is still a search task, but the calling model should write an official-source-first prompt."
        prompt_guidance = (
            "Write a full prompt that explicitly prioritizes official docs, official repositories, official APIs, "
            "standards, specs, and current version dates."
        )

    result = {
        "task": cleaned,
        "repository_root": root,
        "recommended_command": command,
        "fallback_commands": fallback,
        "reason": reason,
        "working_directory_rule": "The command already includes the absolute repository path and can run from any directory.",
    }
    if prompt_guidance:
        result["prompt_guidance"] = prompt_guidance
    if complex_research:
        result["decomposition_required"] = True
        result["orchestration_strategy"] = "split_then_search"
        result["orchestration_warning"] = (
            "This task is too broad for a single search prompt. Split it into 2-4 narrow source-discovery queries before running search."
        )
        result["decomposition_rules"] = [
            "One sub-query should cover one hypothesis, one skill area, or one evidence dimension only.",
            "Run one `search` command per sub-query instead of one giant prompt.",
            "Use `fetch` on the highest-value pages before writing the final synthesis.",
        ]
        result["single_shot_not_recommended"] = True
    if " web-search-cli searchwithsummary " in f" {command} ":
        result["summary_warning"] = (
            "searchwithsummary 返回的 summary 不一定准确，最好结合 webSearchResults、browse_page 和 fetch 结果自行验证。"
        )
    return result


async def build_doctor_report() -> dict[str, Any]:
    report = await get_doctor_info()
    report["next_step"] = "If configuration looks healthy, call recommend_command and then execute the returned uv run command."
    return report
