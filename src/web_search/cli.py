from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .github_tools import GitHubToolError, fetch_repo_file, format_grep_results, github_api_request, grep_repo
from .runtime import (
    SearchExecutionError,
    fetch_url,
    get_available_models,
    get_doctor_info,
    map_site,
    search_web,
    search_web_with_summary,
)


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _format_search_output(result: dict[str, Any], *, include_summary: bool) -> str:
    sections: list[str] = []
    if include_summary:
        warning = (result.get("summary_warning") or "").strip()
        if warning:
            sections.append(f"> ⚠️ {warning}")
        content = (result.get("content") or "").strip()
        if content:
            sections.append(content)

    browse_pages = result.get("browse_page") or []
    if browse_pages:
        lines = ["## browse_page（高价值网页，建议优先打开/抓取）"]
        for item in browse_pages:
            title = item.get("title") or item.get("url", "")
            url = item.get("url", "")
            preview = item.get("preview") or item.get("description") or ""
            instructions = item.get("instructions") or ""
            note = item.get("note") or ""
            lines.append(f"- [{title}]({url})")
            if preview:
                lines.append(f"  - {preview}")
            if instructions and instructions != preview:
                lines.append(f"  - 指引：{instructions}")
            if note:
                lines.append(f"  - {note}")
        sections.append("\n".join(lines))

    web_results = result.get("webSearchResults") or []
    if web_results:
        lines = ["## webSearchResults"]
        for item in web_results:
            title = item.get("title") or item.get("url", "")
            url = item.get("url", "")
            preview = item.get("preview") or item.get("description") or ""
            lines.append(f"- [{title}]({url})")
            if preview:
                lines.append(f"  - {preview}")
        sections.append("\n".join(lines))

    return "\n\n".join(section for section in sections if section.strip())


async def _run_search(args: argparse.Namespace) -> int:
    result = await search_web(
        " ".join(args.prompt),
        extra_sources=args.extra_sources,
    )
    if args.json:
        _print_json(result)
    else:
        print(_format_search_output(result, include_summary=False))
    return 0 if result["status"] == "ok" else 1


async def _run_search_with_summary(args: argparse.Namespace) -> int:
    result = await search_web_with_summary(
        " ".join(args.prompt),
        extra_sources=args.extra_sources,
    )
    if args.json:
        _print_json(result)
    else:
        print(_format_search_output(result, include_summary=True))
    return 0 if result["status"] == "ok" else 1


async def _run_fetch(args: argparse.Namespace) -> int:
    try:
        print(await fetch_url(args.url))
        return 0
    except (SearchExecutionError, Exception) as exc:
        print(str(exc), file=sys.stderr)
        return 1


async def _run_map(args: argparse.Namespace) -> int:
    try:
        result = await map_site(
            args.url,
            instructions=args.instructions,
            max_depth=args.max_depth,
            max_breadth=args.max_breadth,
            limit=args.limit,
            timeout=args.timeout,
        )
    except (SearchExecutionError, Exception) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _print_json(result)
    return 0


async def _run_models(args: argparse.Namespace) -> int:
    try:
        models = await get_available_models()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        _print_json({"models": models})
    else:
        print("\n".join(models))
    return 0


async def _run_doctor(_: argparse.Namespace) -> int:
    _print_json(await get_doctor_info())
    return 0


async def _run_github_api(args: argparse.Namespace) -> int:
    try:
        _print_json(await github_api_request(args.endpoint))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


async def _run_repo_file(args: argparse.Namespace) -> int:
    try:
        result = await fetch_repo_file(args.repo, args.path, ref=args.ref)
    except (GitHubToolError, Exception) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json or result.get("kind") != "file":
        _print_json(result)
    else:
        print(result.get("content", ""))
    return 0


async def _run_repo_grep(args: argparse.Namespace) -> int:
    try:
        result = await grep_repo(
            args.repo,
            args.pattern,
            ref=args.ref,
            use_regex=args.regex,
            case_sensitive=args.case_sensitive,
            max_matches=args.max_matches,
        )
    except (GitHubToolError, Exception) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        _print_json(result)
    else:
        print(format_grep_results(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="web-search-cli", description="Repository-local bash/uv entrypoints for search workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="Send a full caller-authored prompt to the configured Grok endpoint.")
    search.add_argument("prompt", nargs="+")
    search.add_argument("--extra-sources", type=int, default=0)
    search.add_argument("--json", action="store_true")
    search.set_defaults(handler=_run_search)

    search_with_summary = subparsers.add_parser(
        "searchwithsummary",
        help="Send a full caller-authored prompt and also keep the upstream summary in the result.",
    )
    search_with_summary.add_argument("prompt", nargs="+")
    search_with_summary.add_argument("--extra-sources", type=int, default=0)
    search_with_summary.add_argument("--json", action="store_true")
    search_with_summary.set_defaults(handler=_run_search_with_summary)

    fetch = subparsers.add_parser("fetch", help="Fetch a known page through Tavily/Firecrawl.")
    fetch.add_argument("url")
    fetch.set_defaults(handler=_run_fetch)

    site_map = subparsers.add_parser("map", help="Map a docs-style site through Tavily.")
    site_map.add_argument("url")
    site_map.add_argument("--instructions", default="")
    site_map.add_argument("--max-depth", type=int, default=1)
    site_map.add_argument("--max-breadth", type=int, default=20)
    site_map.add_argument("--limit", type=int, default=50)
    site_map.add_argument("--timeout", type=int, default=150)
    site_map.set_defaults(handler=_run_map)

    models = subparsers.add_parser("models", help="List models from the configured Grok endpoint.")
    models.add_argument("--json", action="store_true")
    models.set_defaults(handler=_run_models)

    doctor = subparsers.add_parser("doctor", help="Show configuration and connectivity diagnostics.")
    doctor.set_defaults(handler=_run_doctor)

    github_api = subparsers.add_parser("github-api", help="Query the GitHub REST API.")
    github_api.add_argument("endpoint")
    github_api.set_defaults(handler=_run_github_api)

    repo_file = subparsers.add_parser("repo-file", help="Fetch a GitHub repository file or directory listing.")
    repo_file.add_argument("repo")
    repo_file.add_argument("path", nargs="?", default="")
    repo_file.add_argument("--ref", default="")
    repo_file.add_argument("--json", action="store_true")
    repo_file.set_defaults(handler=_run_repo_file)

    repo_grep = subparsers.add_parser("repo-grep", help="Shallow-clone and search a GitHub repository.")
    repo_grep.add_argument("repo")
    repo_grep.add_argument("pattern")
    repo_grep.add_argument("--ref", default="")
    repo_grep.add_argument("--regex", action="store_true")
    repo_grep.add_argument("--case-sensitive", action="store_true")
    repo_grep.add_argument("--max-matches", type=int, default=20)
    repo_grep.add_argument("--json", action="store_true")
    repo_grep.set_defaults(handler=_run_repo_grep)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
