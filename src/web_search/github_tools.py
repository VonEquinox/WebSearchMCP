from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx

GITHUB_API_BASE = "https://api.github.com"
_GITHUB_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


class GitHubToolError(RuntimeError):
    """Raised when a GitHub helper cannot complete."""


async def github_api_request(endpoint_or_url: str) -> Any:
    url = endpoint_or_url if endpoint_or_url.startswith(("http://", "https://")) else f"{GITHUB_API_BASE}/{endpoint_or_url.lstrip('/')}"
    headers = dict(_GITHUB_HEADERS)
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


def parse_github_repo(repo: str) -> str:
    value = repo.strip().rstrip("/")
    for prefix in ("https://github.com/", "http://github.com/", "git@github.com:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    value = value.removesuffix(".git")
    if "/" not in value:
        raise GitHubToolError(f"Unsupported GitHub repo spec: {repo}")
    return value


async def fetch_repo_file(repo: str, path: str = "", ref: str = "") -> dict[str, Any]:
    normalized_repo = parse_github_repo(repo)
    endpoint = f"/repos/{normalized_repo}/contents/{path.lstrip('/')}"
    if ref:
        endpoint = f"{endpoint}?ref={ref}"
    data = await github_api_request(endpoint)
    if isinstance(data, list):
        return {
            "kind": "directory",
            "repo": normalized_repo,
            "path": path or ".",
            "entries": [
                {"name": item.get("name", ""), "path": item.get("path", ""), "type": item.get("type", "")}
                for item in data
                if isinstance(item, dict)
            ],
        }
    if not isinstance(data, dict):
        raise GitHubToolError("Unexpected GitHub contents API response")
    if data.get("type") != "file":
        return {"kind": data.get("type", "unknown"), "repo": normalized_repo, "path": data.get("path", path)}

    content = data.get("content") or ""
    encoding = data.get("encoding") or ""
    decoded = ""
    if encoding == "base64" and isinstance(content, str):
        decoded = base64.b64decode(content.encode("utf-8")).decode("utf-8", errors="replace")
    return {
        "kind": "file",
        "repo": normalized_repo,
        "path": data.get("path", path),
        "sha": data.get("sha", ""),
        "content": decoded,
    }


def _search_file_lines(
    file_path: Path,
    pattern: str,
    *,
    use_regex: bool,
    case_sensitive: bool,
    max_matches: int,
) -> list[dict[str, Any]]:
    flags = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(pattern if use_regex else re.escape(pattern), flags)
    matches: list[dict[str, Any]] = []
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not regex.search(line):
                    continue
                matches.append(
                    {
                        "path": str(file_path),
                        "line": line_number,
                        "text": line.rstrip(),
                    }
                )
                if len(matches) >= max_matches:
                    break
    except OSError:
        return []
    return matches


async def grep_repo(
    repo: str,
    pattern: str,
    *,
    ref: str = "",
    use_regex: bool = False,
    case_sensitive: bool = False,
    max_matches: int = 20,
) -> dict[str, Any]:
    normalized_repo = parse_github_repo(repo)
    git_bin = shutil.which("git")
    if not git_bin:
        raise GitHubToolError("git 不可用，无法执行 repo-grep")

    clone_url = f"https://github.com/{normalized_repo}.git"
    with tempfile.TemporaryDirectory(prefix="web-search-repo-") as tmp_dir:
        target = Path(tmp_dir) / "repo"
        command = [git_bin, "clone", "--depth", "1"]
        if ref:
            command.extend(["--branch", ref, "--single-branch"])
        command.extend([clone_url, str(target)])
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise GitHubToolError(completed.stderr.strip() or "git clone 失败")

        matches: list[dict[str, Any]] = []
        for root, _, files in os.walk(target):
            if ".git" in Path(root).parts:
                continue
            for name in files:
                file_matches = _search_file_lines(
                    Path(root) / name,
                    pattern,
                    use_regex=use_regex,
                    case_sensitive=case_sensitive,
                    max_matches=max_matches - len(matches),
                )
                for item in file_matches:
                    item["path"] = str(Path(item["path"]).relative_to(target))
                matches.extend(file_matches)
                if len(matches) >= max_matches:
                    break
            if len(matches) >= max_matches:
                break

    return {
        "repo": normalized_repo,
        "pattern": pattern,
        "use_regex": use_regex,
        "case_sensitive": case_sensitive,
        "matches": matches,
        "match_count": len(matches),
    }


def format_grep_results(result: dict[str, Any]) -> str:
    if not result.get("matches"):
        return json.dumps(result, ensure_ascii=False, indent=2)
    lines = [f"repo: {result['repo']}", f"pattern: {result['pattern']}", ""]
    for item in result["matches"]:
        lines.append(f"{item['path']}:{item['line']}: {item['text']}")
    return "\n".join(lines)
