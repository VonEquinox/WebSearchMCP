import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _limit_chars(text: str, max_chars: int = 200_000) -> str:
    content = text or ""
    if len(content) <= max_chars:
        return content
    return content[:max_chars]


def _safe_name(text: str) -> str:
    text = (text or "").strip() or "case"
    text = re.sub(r"[^\w\-\.]+", "_", text, flags=re.UNICODE)
    return text[:80].strip("_") or "case"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content or "", encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _contains_any(haystack: str, needles: List[str]) -> bool:
    if not needles:
        return True
    text = (haystack or "").lower()
    for needle in needles:
        if (needle or "").lower() in text:
            return True
    return False


def _contains_forbidden(haystack: str, forbidden: List[str]) -> List[str]:
    text = haystack or ""
    hits: List[str] = []
    for token in forbidden or []:
        if token and token in text:
            hits.append(token)
    return hits


def _markdown_to_text(markdown: str) -> str:
    text = markdown or ""
    # Remove fenced code blocks but keep their content.
    text = re.sub(r"```[a-zA-Z0-9_-]*\n", "", text)
    text = text.replace("```", "")
    # Drop markdown links: [title](url) -> title
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    # Drop headings markers
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    # Drop list markers
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.M)
    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _direct_html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "lxml")
    for tag in soup(["script", "style", "noscript"]):
        try:
            tag.decompose()
        except Exception:
            pass
    return soup.get_text(separator="\n", strip=True)


async def _run_fetch_cases(
    cases: List[Dict[str, Any]],
    *,
    concurrency: int,
    out_dir: Path,
    also_text: bool,
    also_direct: bool,
) -> Tuple[List[Dict[str, Any]], int, int]:
    from SOTASearch import fetch

    sem = asyncio.Semaphore(max(1, concurrency))
    stamp = _now_stamp()
    run_dir = out_dir / "fetch" / stamp

    hard_failures = 0
    total_hard = 0

    async def _one(case: Dict[str, Any]) -> Dict[str, Any]:
        name = case.get("name") or case.get("url") or "case"
        url = case.get("url") or ""
        min_chars = int(case.get("min_chars") or 0)
        require_any = list(case.get("require_any") or [])
        forbid = list(case.get("forbid") or [])
        allow_blocked = bool(case.get("allow_blocked") or False)
        hard_fail = bool(case.get("hard_fail", True))

        case_dir = run_dir / _safe_name(name)
        async with sem:
            md_result = await fetch(url)

        _write_json(case_dir / "result.json", md_result)
        _write_text(case_dir / "content.md", md_result.get("markdown", "") or "")
        if also_text:
            _write_text(case_dir / "content.txt", _markdown_to_text(md_result.get("markdown", "") or ""))
        if also_direct:
            loop = asyncio.get_running_loop()

            def _direct_fetch() -> Dict[str, Any]:
                try:
                    from SOTASearch import _curl_get_with_retries, _get_target_url, FETCH_TIMEOUT_S

                    resp = _curl_get_with_retries(
                        _get_target_url(url),
                        headers=None,
                        timeout_s=FETCH_TIMEOUT_S,
                        retries=1,
                    )
                    html = resp.text or ""
                    return {"success": True, "status_code": resp.status_code, "html": html}
                except Exception as e:
                    return {"success": False, "error": str(e)}

            direct = await loop.run_in_executor(None, _direct_fetch)
            _write_json(case_dir / "direct.json", direct)
            if direct.get("success") and direct.get("html"):
                direct_txt = _direct_html_to_text(direct.get("html") or "")
                _write_text(case_dir / "direct.txt", _limit_chars(direct_txt))

        status = "PASS"
        errors: List[str] = []

        if not md_result.get("success"):
            status = "FAIL"
            errors.append(f"success=false error={md_result.get('error')!r}")
        else:
            content = md_result.get("markdown", "") or ""
            if min_chars and len(content) < min_chars:
                status = "FAIL"
                errors.append(f"len<{min_chars} (got {len(content)})")

            if require_any and not _contains_any(content, require_any):
                status = "FAIL"
                errors.append(f"missing require_any={require_any}")

            forbidden_hits = _contains_forbidden(content, forbid)
            if forbidden_hits:
                status = "FAIL"
                errors.append(f"forbidden hits={forbidden_hits}")

            if md_result.get("blocked") and allow_blocked:
                status = "SKIP"
                errors.append("blocked=true (allowed)")

        nonlocal hard_failures, total_hard
        if hard_fail:
            total_hard += 1
            if status == "FAIL":
                hard_failures += 1

        return {
            "name": name,
            "url": url,
            "status": status,
            "errors": errors,
            "extractor": md_result.get("extractor"),
            "quality_score": md_result.get("quality_score"),
            "degraded": md_result.get("degraded"),
            "blocked": md_result.get("blocked"),
            "artifact_dir": str(case_dir),
        }

    results = await asyncio.gather(*[_one(c) for c in cases])
    return results, hard_failures, total_hard


async def _run_search_cases(
    cases: List[Dict[str, Any]],
    *,
    concurrency: int,
    out_dir: Path,
) -> Tuple[List[Dict[str, Any]], int, int]:
    from SOTASearch import web_search, _normalize_url_for_dedup

    sem = asyncio.Semaphore(max(1, concurrency))
    stamp = _now_stamp()
    run_dir = out_dir / "search" / stamp

    hard_failures = 0
    total_hard = 0

    async def _one(case: Dict[str, Any]) -> Dict[str, Any]:
        name = case.get("name") or case.get("query") or "case"
        query = case.get("query") or ""
        min_links = int(case.get("min_links") or 1)
        hard_fail = bool(case.get("hard_fail", True))

        case_dir = run_dir / _safe_name(name)
        async with sem:
            result = await web_search(query)

        _write_json(case_dir / "result.json", result)

        status = "PASS"
        errors: List[str] = []

        if not result.get("success"):
            status = "FAIL"
            errors.append(f"success=false error={result.get('error')!r}")
        else:
            links = list(result.get("links") or [])
            if len(links) < min_links:
                status = "FAIL"
                errors.append(f"links<{min_links} (got {len(links)})")
            if len(links) > 15:
                status = "FAIL"
                errors.append(f"links>15 (got {len(links)})")

            urls = [l.get("url", "") for l in links if isinstance(l, dict)]
            empty_urls = [u for u in urls if not u]
            if empty_urls:
                status = "FAIL"
                errors.append("empty url present")

            normalized = [_normalize_url_for_dedup(u) or u for u in urls if u]
            if len(set(normalized)) != len(normalized):
                status = "FAIL"
                errors.append("normalized duplicate urls present")

        nonlocal hard_failures, total_hard
        if hard_fail:
            total_hard += 1
            if status == "FAIL":
                hard_failures += 1

        return {
            "name": name,
            "query": query,
            "status": status,
            "errors": errors,
            "artifact_dir": str(case_dir),
        }

    results = await asyncio.gather(*[_one(c) for c in cases])
    return results, hard_failures, total_hard


def main() -> None:
    parser = argparse.ArgumentParser(description="WebSearchMCP integration regression runner (live)")
    parser.add_argument("--cases-dir", type=str, default="cases")
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--fetch", action="store_true", help="Run fetch regression cases")
    parser.add_argument("--search", action="store_true", help="Run web_search regression cases")
    parser.add_argument("--also-text", action="store_true", help="Also save a plain-text view of fetched markdown")
    parser.add_argument("--also-direct", action="store_true", help="Also save a direct HTML->text snapshot for comparison")
    parser.add_argument("--concurrency", type=int, default=int(os.getenv("REGRESSION_CONCURRENCY", "3")))

    args = parser.parse_args()

    cases_dir = Path(args.cases_dir)
    out_dir = Path(args.out_dir)

    run_fetch = args.fetch or (not args.fetch and not args.search)
    run_search = args.search or (not args.fetch and not args.search)

    fetch_cases_path = cases_dir / "fetch_cases.json"
    search_cases_path = cases_dir / "search_cases.json"

    fetch_cases = _read_json(fetch_cases_path) if fetch_cases_path.exists() else []
    search_cases = _read_json(search_cases_path) if search_cases_path.exists() else []

    async def _run() -> Dict[str, Any]:
        summary: Dict[str, Any] = {"timestamp": _now_stamp(), "fetch": None, "search": None}
        hard_failures = 0
        total_hard = 0

        if run_fetch:
            results, hf, th = await _run_fetch_cases(
                fetch_cases,
                concurrency=args.concurrency,
                out_dir=out_dir,
                also_text=args.also_text,
                also_direct=args.also_direct,
            )
            summary["fetch"] = {"results": results, "hard_failures": hf, "total_hard": th}
            hard_failures += hf
            total_hard += th

        if run_search:
            results, hf, th = await _run_search_cases(
                search_cases,
                concurrency=args.concurrency,
                out_dir=out_dir,
            )
            summary["search"] = {"results": results, "hard_failures": hf, "total_hard": th}
            hard_failures += hf
            total_hard += th

        summary["hard_failures"] = hard_failures
        summary["total_hard"] = total_hard
        return summary

    summary = asyncio.run(_run())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get("hard_failures", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
