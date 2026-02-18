"""
SOTA Search MCP Server - AI 增强搜索

功能：
- 原有 web_search: 使用 Brave Search 进行普通搜索
- 新增 ai_search: 使用 OpenAI API 进行 AI 深度搜索，返回搜索链接和总结
- 支持 --cf-worker 参数，通过 Cloudflare Worker 代理流量

变更(curl_cffi 版）：
- 统一使用 curl_cffi 发起 HTTP 请求并解析 HTML
- Pro 版在被 Cloudflare 挑战时回退 Playwright + stealth
"""

import argparse
import asyncio
import html as html_lib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, quote, urlparse, urlunparse, parse_qsl, urlencode


def _load_env_file(path: Path) -> None:
    """Parse a .env-like file but ignore non KEY=VALUE lines.

    This project sometimes keeps human-readable sections in `.env` without `#`,
    which makes python-dotenv noisy. We keep a tolerant parser instead.
    """

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if raw.startswith("export "):
            raw = raw[len("export ") :].strip()
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ.setdefault(key, value)


_load_env_file(Path(__file__).with_name(".env"))

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

# ============================================================================
# 全局配置
# ============================================================================
parser = argparse.ArgumentParser(description="SOTA Search MCP Server")
parser.add_argument(
    "--proxy",
    type=str,
    default=os.getenv("PROXY"),
    help="本地代理设置，例如: http://127.0.0.1:7890",
)
parser.add_argument(
    "--cf-worker",
    type=str,
    default=os.getenv("CF_WORKER"),
    help="Cloudflare Worker 地址，例如: https://xxx.xxx.workers.dev",
)
parser.add_argument(
    "--openai-api-key",
    type=str,
    default=os.getenv("OPENAI_API_KEY"),
    help="OpenAI API Key",
)
parser.add_argument(
    "--openai-base-url",
    type=str,
    default=os.getenv("OPENAI_BASE_URL"),
    help="OpenAI API Base URL，例如: https://api.openai.com/v1",
)
parser.add_argument(
    "--openai-model",
    type=str,
    default=os.getenv("OPENAI_MODEL", "gpt-4o"),
    help="OpenAI 模型名称，默认: gpt-4o",
)

CLI_ARGS, _ = parser.parse_known_args()
PROXY_CONFIG = CLI_ARGS.proxy
CF_WORKER_URL = CLI_ARGS.cf_worker
OPENAI_API_KEY = CLI_ARGS.openai_api_key
OPENAI_BASE_URL = CLI_ARGS.openai_base_url
OPENAI_MODEL = CLI_ARGS.openai_model

# 兼容处理：MCP 客户端可能把 "--arg value" 合并成一个字符串
for arg in sys.argv[1:]:
    if arg.startswith("--proxy ") and PROXY_CONFIG is None:
        PROXY_CONFIG = arg.split(" ", 1)[1]
    elif arg.startswith("--cf-worker ") and CF_WORKER_URL is None:
        CF_WORKER_URL = arg.split(" ", 1)[1]
    elif arg.startswith("--openai-api-key ") and OPENAI_API_KEY is None:
        OPENAI_API_KEY = arg.split(" ", 1)[1]
    elif arg.startswith("--openai-base-url ") and OPENAI_BASE_URL is None:
        OPENAI_BASE_URL = arg.split(" ", 1)[1]
    elif arg.startswith("--openai-model ") and OPENAI_MODEL == "gpt-4o":
        OPENAI_MODEL = arg.split(" ", 1)[1]

# 调试：打印接收到的参数
logger.info(f"[DEBUG] sys.argv: {sys.argv}")
logger.info(f"[DEBUG] PROXY_CONFIG: {PROXY_CONFIG}")
logger.info(f"[DEBUG] CF_WORKER_URL: {CF_WORKER_URL}")
logger.info(f"[DEBUG] OPENAI_API_KEY: {'***' if OPENAI_API_KEY else None}")
logger.info(f"[DEBUG] OPENAI_BASE_URL: {OPENAI_BASE_URL}")
logger.info(f"[DEBUG] OPENAI_MODEL: {OPENAI_MODEL}")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

MAX_TOKEN_LIMIT = 10000
CURL_IMPERSONATE = "chrome110"
HTTP_VERSION = "v1"
FETCH_TIMEOUT_S = 15
SEARCH_TIMEOUT_S = 60
SEARCH_RESULT_LIMIT = 25
PLAYWRIGHT_FALLBACK = os.getenv("PLAYWRIGHT_FALLBACK", "1").lower() not in ("0", "false", "no")
PLAYWRIGHT_TIMEOUT_MS = int(os.getenv("PLAYWRIGHT_TIMEOUT_MS", "60000"))
PLAYWRIGHT_CHALLENGE_WAIT = int(os.getenv("PLAYWRIGHT_CHALLENGE_WAIT", "20"))

# OpenAI 客户端（懒加载，只创建一次）
_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> Optional[OpenAI]:
    """获取 OpenAI 客户端（懒加载）"""
    global _openai_client
    if _openai_client is None and OPENAI_API_KEY and OPENAI_BASE_URL:
        _openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
    return _openai_client


def _llm_configured() -> bool:
    return bool(OPENAI_API_KEY and OPENAI_BASE_URL)


# ============================================================================
# 辅助函数：处理 URL 路由
# ============================================================================
def _get_target_url(original_url: str) -> str:
    """
    如果配置了 CF Worker，则将请求包装到 Worker URL 中。
    否则返回原始 URL。
    """
    if CF_WORKER_URL:
        # 移除末尾斜杠以防万一
        worker_base = CF_WORKER_URL.rstrip("/")
        # 构造类似 https://worker.dev?url=https://google.com 的请求
        encoded_target = quote(original_url) # 使用 quote 而不是 quote_plus 以保持 :// 的完整性兼容性
        return f"{worker_base}?url={encoded_target}"
    return original_url


def _get_proxies() -> Optional[Dict[str, str]]:
    """
    获取代理配置。
    注意：如果使用了 CF Worker，通常不需要本地代理去连接 CF Worker（除非本地无法直连 CF）。
    这里保持逻辑：如果配置了 proxy，就让 requests/scraper 走这个 proxy。
    """
    if PROXY_CONFIG:
        return {
            "http": PROXY_CONFIG,
            "https": PROXY_CONFIG,
        }
    return None


def _limit_content_length(content: str) -> Tuple[str, bool]:
    estimated_tokens = len(content) // 4
    if estimated_tokens > MAX_TOKEN_LIMIT:
        chars_to_keep = MAX_TOKEN_LIMIT * 4
        truncated_content = content[:chars_to_keep]
        return truncated_content, True
    return content, False


def _parse_viewport(raw: Optional[str]) -> Optional[Dict[str, int]]:
    if not raw:
        return None
    text = raw.lower().replace(" ", "")
    if "x" in text:
        width_str, height_str = text.split("x", 1)
    elif "," in text:
        width_str, height_str = text.split(",", 1)
    else:
        return None
    try:
        return {"width": int(width_str), "height": int(height_str)}
    except ValueError:
        return None


def _looks_like_challenge_text(content: str) -> bool:
    lowered = (content or "").lower()
    return (
        "just a moment" in lowered
        or "checking your browser" in lowered
        or "attention required" in lowered
        or "cf-browser-verification" in lowered
        or ("cloudflare" in lowered and "ray id" in lowered)
    )


def _looks_like_blocked_text(content: str) -> bool:
    """Detect generic blocks/login walls/captcha pages (best-effort)."""
    if not content:
        return False
    if _looks_like_challenge_text(content):
        return True

    # Prefer visible text over raw HTML to avoid false positives from scripts.
    visible_text = content
    if "<" in content and ">" in content:
        try:
            visible_text = _html_to_text(content)
        except Exception:
            visible_text = content

    lowered = (visible_text or "").lower()
    english_hints = (
        "captcha",
        "robot check",
        "access denied",
        "verify you are human",
        "unusual traffic",
    )
    for hint in english_hints:
        if hint in lowered:
            return True

    chinese_hints = (
        "访问异常",
        "安全验证",
        "滑动验证",
        "验证码",
        "请完成验证",
        "检测到异常",
        "系统检测到",
        "访问过于频繁",
        "请稍后再试",
        "请先登录",
        "登录后查看更多",
        "请登录后继续访问",
        "马上登录",
        "立即登录",
        "登录即可",
    )
    for hint in chinese_hints:
        if hint in (visible_text or ""):
            return True
    return False


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form", "button", "svg"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


_TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "igshid",
    "spm",
    "spm_id_from",
    "from",
    "from_source",
    "source",
    "sourcefrom",
    "share_source",
    "share_medium",
    "share_platform",
    "share_id",
    "share_from",
    "shareuid",
    "scene",
    "platform",
    "ref",
    "refer",
    "ref_source",
    "referrer",
    "vd_source",
    "_t",
    "_r",
    "mpshare",
}


def _normalize_url_for_dedup(url: str) -> str:
    if not url:
        return ""
    raw = url.strip()
    if raw.startswith("//"):
        raw = "https:" + raw
    elif raw.startswith("www."):
        raw = "https://" + raw
    try:
        parsed = urlparse(raw)
    except Exception:
        return raw

    scheme = (parsed.scheme or "https").lower()
    netloc = (parsed.netloc or "").lower()
    path = parsed.path or ""

    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    filtered_pairs = []
    try:
        for key, value in parse_qsl(parsed.query, keep_blank_values=False):
            if key.lower() in _TRACKING_QUERY_KEYS:
                continue
            filtered_pairs.append((key, value))
    except Exception:
        filtered_pairs = []
    query = urlencode(sorted(filtered_pairs), doseq=True)

    normalized_path = path.rstrip("/") or "/"
    return urlunparse((scheme, netloc, normalized_path, "", query, ""))


_SITE_QUERY_RE = re.compile(r"(?<!\S)site\s*:\s*([^\s]+)", re.IGNORECASE)


def _is_site_query(query: str) -> bool:
    return bool(_SITE_QUERY_RE.search(query or ""))


_REDIRECT_PARAM_CANDIDATES = ("uddg", "target", "url", "q", "u", "to", "dest", "destination", "redir", "redirect")


def _unwrap_redirect_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""

    if raw.startswith("//"):
        raw = "https:" + raw
    elif raw.startswith("www."):
        raw = "https://" + raw

    try:
        parsed = urlparse(raw)
    except Exception:
        return raw

    netloc = (parsed.netloc or "").lower()
    path = parsed.path or ""

    try:
        params = dict(parse_qsl(parsed.query, keep_blank_values=False))
    except Exception:
        params = {}

    # DuckDuckGo redirect: https://duckduckgo.com/l/?uddg=...
    if netloc.endswith("duckduckgo.com") and path.startswith("/l/"):
        uddg = params.get("uddg")
        if uddg and isinstance(uddg, str) and uddg.startswith("http"):
            return uddg

    # Zhihu redirect: https://link.zhihu.com/?target=...
    if netloc == "link.zhihu.com":
        target = params.get("target")
        if target and isinstance(target, str) and target.startswith("http"):
            return target

    # Brave redirect: https://r.search.brave.com/redirect?url=...
    if netloc.endswith("search.brave.com") and ("redirect" in path or netloc.startswith("r.")):
        target = params.get("url") or params.get("q")
        if target and isinstance(target, str) and target.startswith("http"):
            return target

    # Google redirect: https://www.google.com/url?q=...
    if netloc.endswith("google.com") and path.startswith("/url"):
        target = params.get("q") or params.get("url")
        if target and isinstance(target, str) and target.startswith("http"):
            return target

    # YouTube redirect: https://www.youtube.com/redirect?q=...
    if netloc.endswith("youtube.com") and path.startswith("/redirect"):
        target = params.get("q") or params.get("url")
        if target and isinstance(target, str) and target.startswith("http"):
            return target

    # Steam linkfilter: https://steamcommunity.com/linkfilter/?url=...
    if netloc.endswith("steamcommunity.com") and "linkfilter" in path:
        target = params.get("url")
        if target and isinstance(target, str) and target.startswith("http"):
            return target

    # Facebook outbound: https://l.facebook.com/l.php?u=...
    if netloc == "l.facebook.com":
        target = params.get("u")
        if target and isinstance(target, str) and target.startswith("http"):
            return target

    # Generic fallback for known redirect hosts.
    if netloc in {"t.co"}:
        return raw

    if netloc in {"redirect.pinterest.com"}:
        for key in _REDIRECT_PARAM_CANDIDATES:
            target = params.get(key)
            if target and isinstance(target, str) and target.startswith("http"):
                return target

    return raw


def _parse_sse_chat_completions(text: str) -> Tuple[str, str]:
    content_parts: List[str] = []
    reasoning_parts: List[str] = []

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data:
            continue
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue

        for choice in obj.get("choices") or []:
            delta = (choice or {}).get("delta") or {}
            if not isinstance(delta, dict):
                continue
            piece = delta.get("content")
            if piece:
                content_parts.append(str(piece))
            reasoning_piece = (
                delta.get("reasoning_content")
                or delta.get("reasoning")
                or delta.get("analysis")
                or delta.get("thinking")
            )
            if reasoning_piece:
                reasoning_parts.append(str(reasoning_piece))

    return "".join(content_parts), "".join(reasoning_parts)


def _call_openai_chat_completions(prompt: str) -> Tuple[str, str]:
    if not OPENAI_API_KEY or not OPENAI_BASE_URL:
        raise RuntimeError("OpenAI client not configured")

    url = f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    response = curl_requests.post(
        url,
        json=payload,
        headers=headers,
        proxies=_get_proxies(),
        allow_redirects=True,
        impersonate=CURL_IMPERSONATE,
        http_version=HTTP_VERSION,
        stream=True,
    )
    response.raise_for_status()

    content_type = (response.headers.get("content-type") or "").lower()
    is_sse = "text/event-stream" in content_type

    if is_sse:
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        pending = ""
        done = False

        def _consume_line(raw_line: str) -> None:
            nonlocal done
            line = (raw_line or "").strip()
            if not line.startswith("data:"):
                return
            data = line[len("data:") :].strip()
            if not data:
                return
            if data == "[DONE]":
                done = True
                return
            try:
                obj = json.loads(data)
            except Exception:
                return
            for choice in obj.get("choices") or []:
                delta = (choice or {}).get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                piece = delta.get("content")
                if piece:
                    content_parts.append(str(piece))
                reasoning_piece = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or delta.get("analysis")
                    or delta.get("thinking")
                )
                if reasoning_piece:
                    reasoning_parts.append(str(reasoning_piece))

        try:
            for chunk in response.iter_content():
                if not chunk:
                    continue
                pending += chunk.decode("utf-8", errors="ignore")
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    _consume_line(line.rstrip("\r"))
                    if done:
                        break
                if done:
                    break
        except Exception as e:
            # AI 长响应在超时前可能已收到大量 chunk，优先返回可用的部分结果。
            if not content_parts and not reasoning_parts:
                raise
            logger.warning("AI SSE 流读取中断，返回部分结果: %s", e)
        finally:
            try:
                response.close()
            except Exception:
                pass

        if pending and not done:
            _consume_line(pending.rstrip("\r"))
        return "".join(content_parts), "".join(reasoning_parts)

    raw_chunks: List[bytes] = []
    try:
        for chunk in response.iter_content():
            if chunk:
                raw_chunks.append(chunk)
    finally:
        try:
            response.close()
        except Exception:
            pass
    text = b"".join(raw_chunks).decode("utf-8", errors="replace")

    try:
        data = json.loads(text)
    except Exception:
        return text, ""

    content = ""
    reasoning = ""
    try:
        choice0 = (data.get("choices") or [{}])[0]
        message = choice0.get("message") or {}
        content_value = message.get("content") or ""
        reasoning_value = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("analysis")
            or ""
        )
        if isinstance(content_value, list):
            content = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content_value
            )
        else:
            content = str(content_value)
        if isinstance(reasoning_value, list):
            reasoning = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in reasoning_value
            )
        else:
            reasoning = str(reasoning_value)
    except Exception:
        pass

    return content, reasoning


def _get_hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _prefer_playwright_for_url(url: str) -> bool:
    host = _get_hostname(url)
    if not host:
        return False
    if host.endswith(("xiaohongshu.com", "xhslink.com")):
        return True
    # 知乎非回答页通常需要渲染才能得到完整正文
    if host.endswith("zhihu.com"):
        return True
    return False


def _resolve_playwright_executable_path(path: str) -> Optional[str]:
    """Work around occasional Playwright arch/path mismatches (e.g. mac-x64 vs mac-arm64)."""
    if not path:
        return None
    candidate = path
    if os.path.exists(candidate):
        return candidate

    # Common macOS path mismatch: Playwright expects x64 but browsers are arm64.
    replacements = (
        ("chrome-mac-x64", "chrome-mac-arm64"),
        ("chrome-headless-shell-mac-x64", "chrome-headless-shell-mac-arm64"),
        ("mac-x64", "mac-arm64"),
    )
    for old, new in replacements:
        if old in path:
            alt = path.replace(old, new)
            if alt != path and os.path.exists(alt):
                return alt
    return None


_NOISE_LINE_PATTERNS = [
    re.compile(r"^\s*(skip to main content|back to top|ctrl\s*\+\s*k|github)\s*$", re.I),
    re.compile(r"^\s*you signed (?:in|out) with another tab or window\.?\s*$", re.I),
    re.compile(r"^\s*reload\s*$", re.I),
    re.compile(r"^\s*dismiss alert\s*$", re.I),
    re.compile(r"^\s*you must be signed in to change notification settings\s*$", re.I),
    re.compile(r"^\s*repository files navigation\s*$", re.I),
    re.compile(r"^\s*view all files\s*$", re.I),
    re.compile(r"^\s*(目录|返回顶部|上一篇|下一篇|上一页|下一页|展开阅读全文|查看更多|查看全部)\s*$"),
    re.compile(r"^\s*(打开\s*app|下载\s*app|在\s*app\s*内\s*打开|立即打开|立即下载|扫码|扫一扫)\s*$", re.I),
    re.compile(r"^\s*(登录|注册|请登录|请先登录)\s*$"),
    re.compile(r"^\s*(点赞|收藏|分享|举报|评论|关注|投币|转发)\s*$"),
    re.compile(r"^\s*(跳转到最后一条回复|跳转到顶部|跳到主要内容)\s*$"),
    re.compile(r"^\s*您已选择\s*\d+\s*个帖子。?\s*$"),
    re.compile(r"^\s*(全选|取消选择)\s*$"),
    re.compile(r"^\s*invalid date\s*$", re.I),
]

_NOISE_SUBSTRINGS = (
    "打开app",
    "下载app",
    "在app内打开",
    "立即打开",
    "立即下载",
    "扫码",
    "扫一扫",
    "登录后查看更多",
    "请登录后继续访问",
    "访问异常",
    "安全验证",
    "滑动验证",
    "验证码",
    "检测到异常",
    "访问过于频繁",
)


def _is_noise_line(line: str) -> bool:
    if not line:
        return False
    stripped = line.strip()
    if not stripped:
        return False

    for pattern in _NOISE_LINE_PATTERNS:
        if pattern.match(stripped):
            return True

    compact = re.sub(r"[\s\u200b\u200c\u200d\ufeff]+", "", stripped.lower())
    compact = re.sub(r"[·•|丨>»«:：;；,.，。!?！？()（）\[\]{}【】<>《》“”\"'`]+", "", compact)
    if len(compact) <= 32:
        for needle in _NOISE_SUBSTRINGS:
            if needle in compact:
                return True
    return False


def _clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    lines: List[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if _is_noise_line(line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _clean_extracted_markdown(markdown: str) -> str:
    if not markdown:
        return ""
    lines = []
    in_code_block = False
    for raw_line in (markdown or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            lines.append(line)
            continue
        if in_code_block:
            lines.append(line)
            continue
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith("#"):
            line = re.sub(r"\s*#\s*$", "", line)
            stripped = line.strip()
        # Keep headings/code fences but still allow noise removal for short UI-only headings.
        candidate = stripped.lstrip("#").strip() if stripped.startswith("#") else stripped
        if _is_noise_line(candidate):
            continue
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _score_content(content: str) -> Dict[str, Any]:
    content = (content or "").strip()
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    char_len = len(content)
    line_count = len(lines)

    # Avoid penalizing structured Markdown where some helper lines legitimately repeat
    # (e.g. code fences like ```).
    meaningful_lines: List[str] = []
    for ln in lines:
        if re.match(r"^\s*```", ln):
            continue
        meaningful_lines.append(ln)
    unique_source = meaningful_lines or lines
    unique_ratio = (len(set(unique_source)) / len(unique_source)) if unique_source else 0.0
    noise_hits = sum(1 for ln in lines if _is_noise_line(ln))
    noise_ratio = (noise_hits / line_count) if line_count else 0.0
    short_source = meaningful_lines or lines
    short_hits = sum(1 for ln in short_source if len(ln) <= 12)
    short_ratio = (short_hits / len(short_source)) if short_source else 0.0

    is_markdown_like = (
        "```" in content
        or bool(re.search(r"^\s*#{1,6}\s+\S", content, flags=re.M))
        or bool(re.search(r"^\s*[-*]\s+\S", content, flags=re.M))
    )
    paragraph_count = content.count("\n\n")
    code_fence_count = content.count("```")
    heading_count = len(re.findall(r"^\s*#{1,6}\s+\S", content, flags=re.M))
    bullet_count = len(re.findall(r"^\s*[-*]\s+\S", content, flags=re.M))
    structure_bonus = 0.0
    if is_markdown_like:
        structure_bonus += 6.0 if code_fence_count >= 2 else (3.0 if code_fence_count else 0.0)
        structure_bonus += min(6.0, float(paragraph_count))
        structure_bonus += min(4.0, float(line_count) / 8.0 * 4.0) if line_count else 0.0
        structure_bonus += min(2.0, float(heading_count))
        structure_bonus += min(2.0, float(bullet_count) / 3.0 * 2.0) if bullet_count else 0.0

    length_score = min(60.0, (char_len / 2000.0) * 60.0)  # ~2000 chars gets full points
    unique_score = min(20.0, unique_ratio * 20.0)
    noise_penalty = min(50.0, noise_ratio * 100.0 * 0.7)

    # Many websites dump navigation/menu items as lots of very short lines.
    short_line_penalty = 0.0
    if line_count >= 40 and short_ratio >= 0.6:
        short_line_penalty = min(30.0, (short_ratio - 0.6) * 100.0)

    score = max(
        0.0,
        min(100.0, length_score + unique_score - noise_penalty - short_line_penalty + structure_bonus),
    )
    return {
        "quality_score": int(round(score)),
        "char_len": char_len,
        "line_count": line_count,
        "unique_line_ratio": round(unique_ratio, 3),
        "noise_line_ratio": round(noise_ratio, 3),
    }


def _extract_title_and_description(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html or "", "lxml")

    def _meta(attrs: Dict[str, str]) -> str:
        tag = soup.find("meta", attrs=attrs)
        if not tag:
            return ""
        return (tag.get("content") or "").strip()

    title = (
        _meta({"property": "og:title"})
        or _meta({"name": "twitter:title"})
        or (soup.find("title").get_text(strip=True) if soup.find("title") else "")
    )
    description = (
        _meta({"property": "og:description"})
        or _meta({"name": "twitter:description"})
        or _meta({"name": "description"})
    )
    return title.strip(), description.strip()


def _trafilatura_extract(
    html: str,
    *,
    url: Optional[str],
    output_format: str,
    favor_precision: bool = False,
    favor_recall: bool = False,
    fast: bool = False,
    include_links: bool = False,
) -> Optional[str]:
    try:
        from trafilatura import extract
    except Exception as e:
        logger.warning("trafilatura not available: %s", e)
        return None
    try:
        max_tree_size_raw = os.getenv("TRAFILATURA_MAX_TREE_SIZE", "").strip()
        max_tree_size = int(max_tree_size_raw) if max_tree_size_raw else None
    except Exception:
        max_tree_size = None

    try:
        return extract(
            html,
            url=url,
            output_format=output_format,
            include_comments=False,
            include_tables=True,
            include_images=False,
            include_links=include_links,
            deduplicate=True,
            favor_precision=favor_precision,
            favor_recall=favor_recall,
            fast=fast,
            max_tree_size=max_tree_size,
        )
    except Exception as e:
        logger.debug("trafilatura.extract failed: %s", e)
        return None


def _trafilatura_baseline(html: str) -> Optional[str]:
    try:
        from trafilatura import baseline
    except Exception:
        return None
    try:
        _postbody, text, _len_text = baseline(html)
        return text
    except Exception:
        return None


def _extract_csdn_html_pruned(html: str) -> Optional[str]:
    soup = BeautifulSoup(html or "", "lxml")
    title = ""
    title_tag = soup.select_one("h1.title-article") or soup.select_one("h1")
    if title_tag:
        title = title_tag.get_text(strip=True)

    main = soup.select_one("#content_views") or soup.select_one("article")
    if not main:
        return None

    for selector in (
        "script",
        "style",
        "header",
        "footer",
        "nav",
        "aside",
        ".hide-article-box",
        ".recommend-box",
        ".tool-box",
        ".blog-tags-box",
        ".article-info-box",
        ".operating",
        ".csdn-toolbar",
        "#passportbox",
        "#toolBarBox",
    ):
        for node in main.select(selector):
            try:
                node.decompose()
            except Exception:
                pass

    title_html = f"<h1>{html_lib.escape(title)}</h1>" if title else ""
    return f"<html><body>{title_html}{str(main)}</body></html>"


def _extract_github_html_pruned(html: str) -> Optional[str]:
    soup = BeautifulSoup(html or "", "lxml")
    title, description = _extract_title_and_description(html)

    readme = (
        soup.select_one("#readme article.markdown-body")
        or soup.select_one("#readme .markdown-body")
        or soup.select_one("article.markdown-body")
    )
    if not readme:
        return None

    # GitHub adds various UI helpers inside markdown rendering; remove obvious non-content.
    for selector in (
        "svg",
        "button",
        "summary",
        "details",
        "clipboard-copy",
        "a.anchor",
        "a.anchorjs-link",
        ".octicon",
    ):
        for node in readme.select(selector):
            try:
                node.decompose()
            except Exception:
                pass

    title_html = f"<h1>{html_lib.escape(title)}</h1>" if title else ""
    desc_html = f"<p>{html_lib.escape(description)}</p>" if description else ""
    return f"<html><body>{title_html}{desc_html}{str(readme)}</body></html>"


def _extract_discourse_html_pruned(html: str) -> Optional[str]:
    soup = BeautifulSoup(html or "", "lxml")
    root = soup.select_one("#main-outlet") or soup.select_one("main") or soup.body
    if not root:
        return None

    articles = root.select("article[data-post-id]") or root.select("article.topic-post")

    cooked_blocks: List[str] = []
    if articles:
        for article in articles:
            cooked = article.select_one(".cooked")
            if not cooked:
                continue
            # Remove obvious UI elements embedded in posts.
            for selector in ("svg", "button", ".post-menu-area", ".topic-map", ".names"):
                for node in cooked.select(selector):
                    try:
                        node.decompose()
                    except Exception:
                        pass
            cooked_blocks.append(str(cooked))
    else:
        # Some instances render posts without <article>; fall back to cooked blocks within the outlet.
        for cooked in root.select(".cooked"):
            if not cooked:
                continue
            for selector in ("svg", "button", ".post-menu-area", ".topic-map", ".names"):
                for node in cooked.select(selector):
                    try:
                        node.decompose()
                    except Exception:
                        pass
            cooked_blocks.append(str(cooked))

    if not cooked_blocks:
        return None

    title, _ = _extract_title_and_description(html)
    title_html = f"<h1>{html_lib.escape(title)}</h1>" if title else ""
    body_html = "".join(cooked_blocks)
    return f"<html><body>{title_html}{body_html}</body></html>"


def _extract_bangumi_html_pruned(html: str) -> Optional[str]:
    soup = BeautifulSoup(html or "", "lxml")
    title, description = _extract_title_and_description(html)

    col_a = soup.select_one("#columnA")
    col_b = soup.select_one("#columnB")
    if not col_a and not col_b:
        return None

    parts: List[str] = []
    if title:
        parts.append(f"<h1>{html_lib.escape(title)}</h1>")
    if description:
        parts.append(f"<p>{html_lib.escape(description)}</p>")
    if col_a:
        parts.append(str(col_a))
    if col_b:
        parts.append(str(col_b))

    combined = "".join(parts)
    return f"<html><body>{combined}</body></html>"


def _extract_steamcommunity_html_pruned(html: str) -> Optional[str]:
    soup = BeautifulSoup(html or "", "lxml")
    title, description = _extract_title_and_description(html)

    main = soup.select_one("#responsive_page_template_content") or soup.select_one(".responsive_page_template_content")
    if not main:
        return None

    # Remove obvious non-content blocks inside the main container.
    for selector in (
        "#global_header",
        "#global_actions",
        "#footer",
        ".responsive_page_menu_ctn",
        ".responsive_header",
        ".responsive_page_menu",
        ".responsive_local_menu",
        ".pulldown",
    ):
        for node in main.select(selector):
            try:
                node.decompose()
            except Exception:
                pass

    title_html = f"<h1>{html_lib.escape(title)}</h1>" if title else ""
    desc_html = f"<p>{html_lib.escape(description)}</p>" if description else ""
    return f"<html><body>{title_html}{desc_html}{str(main)}</body></html>"


def _extract_discourse_text_pruned(html: str, *, url: str) -> Optional[str]:
    title, _ = _extract_title_and_description(html)
    title = (title or "").strip()
    if not title:
        return None

    # Common Discourse title shape: "{topic} - {category} - {site}"
    topic_title = re.split(r"\s+-\s+", title, maxsplit=1)[0].strip()
    if not topic_title or len(topic_title) < 4:
        return None

    raw_text = _html_to_text(html)
    if not raw_text:
        return None
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return None

    start_idx = 0
    for i, ln in enumerate(lines):
        if ln == topic_title:
            start_idx = i
            break
    else:
        for i, ln in enumerate(lines):
            if topic_title in ln and len(ln) <= len(topic_title) + 12:
                start_idx = i
                break

    end_idx = len(lines)
    for marker in ("相关话题", "话题列表", "Related topics", "Topic list"):
        for i in range(start_idx + 1, len(lines)):
            if lines[i] == marker or marker in lines[i]:
                end_idx = min(end_idx, i)
                break

    sliced = lines[start_idx:end_idx]

    cleaned_lines: List[str] = []
    for ln in sliced:
        if not ln:
            continue
        if ln in ("您已选择", "个帖子", "个帖子。"):
            continue
        if ln in (",", "，", "。"):
            continue
        if re.fullmatch(r"\d{1,2}", ln):
            # Often UI counters/avatars rendered as standalone digits.
            continue
        if re.fullmatch(r"\d+\s*/\s*\d+", ln):
            # Pagination-like counters, e.g. "1 / 19"
            continue
        cleaned_lines.append(ln)

    kept = "\n".join(cleaned_lines).strip()
    kept = _clean_extracted_text(kept)
    return kept or None


def _build_degraded_markdown(html: str) -> Optional[str]:
    title, description = _extract_title_and_description(html)
    if not title and not description:
        return None
    parts = []
    if title:
        parts.append(f"# {title}")
    if description:
        parts.append(description)
    return "\n\n".join(parts).strip()


def _build_degraded_text(html: str) -> Optional[str]:
    title, description = _extract_title_and_description(html)
    if not title and not description:
        return None
    parts = []
    if title:
        parts.append(title)
    if description:
        parts.append(description)
    return "\n\n".join(parts).strip()


def _extract_best_content(html: str, *, url: str, output_format: str) -> Dict[str, Any]:
    host = _get_hostname(url)

    candidates: List[Dict[str, Any]] = []

    def _add_candidate(content: Optional[str], extractor: str) -> None:
        if not content:
            return
        cleaned = (
            _clean_extracted_markdown(content)
            if output_format == "markdown"
            else _clean_extracted_text(content)
        )
        if not cleaned:
            return
        metrics = _score_content(cleaned)
        candidates.append(
            {
                "content": cleaned,
                "extractor": extractor,
                **metrics,
                "degraded": False,
            }
        )

    # Site adapter: CSDN
    if host.endswith("csdn.net"):
        pruned = _extract_csdn_html_pruned(html)
        if pruned:
            _add_candidate(
                _trafilatura_extract(pruned, url=url, output_format=output_format, favor_precision=True),
                "adapter:csdn+trafilatura",
            )

    # Site adapter: GitHub (prefer README/markdown body, keep links)
    if host.endswith("github.com"):
        pruned = _extract_github_html_pruned(html)
        if pruned:
            _add_candidate(
                _trafilatura_extract(
                    pruned,
                    url=url,
                    output_format=output_format,
                    favor_precision=True,
                    include_links=True,
                ),
                "adapter:github+trafilatura",
            )

    # Site adapter: Bangumi user pages
    if host.endswith(("bgm.tv", "bangumi.tv", "chii.in")):
        pruned = _extract_bangumi_html_pruned(html)
        if pruned:
            pruned_baseline = _trafilatura_baseline(pruned)
            if pruned_baseline:
                _add_candidate(pruned_baseline, "adapter:bangumi+baseline")
            _add_candidate(_html_to_text(pruned), "adapter:bangumi+bs4")

    # Site adapter: Steam Community pages
    if host.endswith("steamcommunity.com"):
        pruned = _extract_steamcommunity_html_pruned(html)
        if pruned:
            pruned_baseline = _trafilatura_baseline(pruned)
            if pruned_baseline:
                _add_candidate(pruned_baseline, "adapter:steamcommunity+baseline")
            _add_candidate(_html_to_text(pruned), "adapter:steamcommunity+bs4")

    # Site adapter: Discourse topic pages (prefer cooked post bodies, keep links)
    if "/t/" in (url or ""):
        pruned = _extract_discourse_html_pruned(html)
        if pruned:
            _add_candidate(
                _trafilatura_extract(
                    pruned,
                    url=url,
                    output_format=output_format,
                    favor_precision=True,
                    include_links=True,
                ),
                "adapter:discourse+trafilatura",
            )
        pruned_text = _extract_discourse_text_pruned(html, url=url)
        if pruned_text:
            _add_candidate(pruned_text, "adapter:discourse:text_pruned")

    # Main extractor: Trafilatura (precision first)
    _add_candidate(
        _trafilatura_extract(html, url=url, output_format=output_format, favor_precision=True),
        "trafilatura:precision",
    )
    _add_candidate(
        _trafilatura_extract(html, url=url, output_format=output_format, favor_recall=True),
        "trafilatura:recall",
    )
    _add_candidate(
        _trafilatura_extract(html, url=url, output_format=output_format, favor_precision=True, fast=True),
        "trafilatura:fast",
    )

    baseline_text = _trafilatura_baseline(html)
    if baseline_text:
        _add_candidate(baseline_text, "trafilatura:baseline")

    # Last-resort fallback: plain text
    _add_candidate(_html_to_text(html), "bs4:text")

    def _extractor_bonus(extractor: str) -> int:
        if extractor.startswith("adapter:"):
            return 15
        if extractor.startswith("trafilatura:precision"):
            return 10
        if extractor.startswith("trafilatura:recall"):
            return 9
        if extractor.startswith("trafilatura:fast"):
            return 8
        if extractor.startswith("trafilatura:baseline"):
            return 6
        return 0

    def _rank_key(item: Dict[str, Any]) -> Tuple[int, int, int]:
        q = int(item.get("quality_score", 0) or 0)
        bonus = _extractor_bonus(item.get("extractor", "") or "")
        char_len = int(item.get("char_len", 0) or 0)
        return (q + bonus, q, char_len)

    ranked = sorted(candidates, key=_rank_key, reverse=True)
    min_chars = 120 if output_format == "markdown" else 200

    # Prefer site adapters when they produce a reasonable amount of content, even if
    # the generic scorer thinks the result is "low quality" (many profiles are lists/short lines).
    adapter_candidates = [c for c in ranked if (c.get("extractor", "") or "").startswith("adapter:")]
    for candidate in adapter_candidates:
        if candidate.get("char_len", 0) >= min_chars and candidate.get("quality_score", 0) >= 10:
            return candidate

    for candidate in ranked:
        if candidate.get("quality_score", 0) >= 30 and candidate.get("char_len", 0) >= min_chars:
            return candidate

    best = ranked[0] if ranked else None

    degraded = (
        _build_degraded_markdown(html)
        if output_format == "markdown"
        else _build_degraded_text(html)
    )
    if degraded:
        cleaned = (
            _clean_extracted_markdown(degraded)
            if output_format == "markdown"
            else _clean_extracted_text(degraded)
        )
        metrics = _score_content(cleaned)
        return {
            "content": cleaned,
            "extractor": "meta:degraded",
            **metrics,
            "degraded": True,
        }

    return best or {
        "content": "",
        "extractor": "none",
        "quality_score": 0,
        "char_len": 0,
        "line_count": 0,
        "unique_line_ratio": 0.0,
        "noise_line_ratio": 0.0,
        "degraded": True,
    }


def _extract_metadata(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html or "", "lxml")

    def _meta(attrs: Dict[str, str]) -> str:
        tag = soup.find("meta", attrs=attrs)
        if not tag:
            return ""
        return (tag.get("content") or "").strip()

    title = (
        _meta({"property": "og:title"})
        or _meta({"name": "twitter:title"})
        or (soup.find("title").get_text(strip=True) if soup.find("title") else "")
    )

    description = (
        _meta({"property": "og:description"})
        or _meta({"name": "twitter:description"})
        or _meta({"name": "description"})
    )

    canonical_url = ""
    canonical = soup.find("link", attrs={"rel": "canonical"})
    if canonical:
        canonical_url = (canonical.get("href") or "").strip()

    links = []
    for a in soup.find_all("a", href=True, limit=50):
        links.append({"text": a.get_text(strip=True), "href": a["href"]})

    links_str = str(links)
    _, was_truncated = _limit_content_length(links_str)
    if was_truncated:
        avg_length = len(links_str) / len(links) if links else 0
        keep_count = max(1, int(MAX_TOKEN_LIMIT * 4 / avg_length) if avg_length > 0 else 0)
        links = links[:keep_count]

    return {
        "title": title,
        "description": description,
        "canonical_url": canonical_url,
        "links": links,
        "truncated": was_truncated,
    }


_ZHIHU_ANSWER_RE = re.compile(r"zhihu\.com/(?:question/\d+/)?answer/(\d+)", re.IGNORECASE)


def _extract_zhihu_answer_id(url: str) -> Optional[str]:
    match = _ZHIHU_ANSWER_RE.search(url or "")
    if match:
        return match.group(1)
    return None


def _clean_ai_tags(text: str) -> str:
    """清理 AI 返回内容中的特殊标签"""
    # 清理 <think>...</think> 推理标签（常见于部分模型）
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    # 清理 grok:render 标签
    text = re.sub(r'<grok:render[^>]*>[\s\S]*?</grok:render>', '', text)
    # 清理其他可能的 XML 标签
    text = re.sub(r'<[a-z_]+:[^>]+>[\s\S]*?</[a-z_]+:[^>]+>', '', text)
    # 清理多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _strip_urls(text: str) -> str:
    """移除正文中的 URL（保留 markdown 链接的标题文本）"""
    if not text:
        return ""

    # 将 markdown 链接 [title](url) 解包为 title
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'\1', text)
    # 移除 <https://...> 形式的链接
    text = re.sub(r'<(https?://[^>]+)>', '', text)
    # 移除裸 URL
    url_pattern = r'https?://[^\s<>\"\'\)\]，。、；：）】}]+'
    text = re.sub(url_pattern, '', text)

    # 清理可能残留的空括号/空方括号与多余空白
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 移除“参考来源/References/Sources”尾部段落（避免 strip URL 后出现空列表）
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.match(
            r"^\s*(参考来源|参考资料|参考链接|Sources|References)\b.*[:：]\s*$",
            line,
            flags=re.IGNORECASE,
        ):
            lines = lines[:index]
            break
    # 移除空的列表项（例如仅剩 '-'）
    cleaned_lines: List[str] = []
    for line in lines:
        if re.match(r"^\s*[-*]\s*$", line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _parse_markdown_links(content: str, extra_text: str = "") -> Tuple[List[Dict[str, str]], str]:
    """
    从 AI 返回的内容中解析链接
    支持格式：
    1. markdown 链接 [title](url)
    2. 纯 URL https://...
    3. 带括号的链接 (https://...)
    返回：(链接列表, 清理后的分析内容)
    """
    content = content or ""
    extra_text = extra_text or ""
    link_source = f"{content}\n{extra_text}" if extra_text else content
    links = []
    seen_urls = set()

    def _normalize_candidate(raw_url: str) -> str:
        url = (raw_url or "").strip()
        if not url:
            return ""
        # 清理末尾常见标点/括号
        url = re.sub(r"[\s\)\]\}>,，。、；：]+$", "", url)
        url = re.sub(r"[.,;:!?]+$", "", url)
        if url.startswith("//"):
            url = "https:" + url
        elif url.startswith("www."):
            url = "https://" + url
        url = _unwrap_redirect_url(url)
        return url.strip()

    # 1. 匹配 markdown 链接格式 [title](url)
    md_pattern = r'\[([^\]]+)\]\(((?:https?://|//|www\.)[^)\s]+)\)'
    for match in re.finditer(md_pattern, link_source):
        title = match.group(1).strip()
        url = _normalize_candidate(match.group(2))
        if not url.startswith("http"):
            continue
        dedup_key = _normalize_url_for_dedup(url) or url
        if dedup_key not in seen_urls:
            seen_urls.add(dedup_key)
            links.append({"title": title, "url": url, "description": ""})

    # 2. 匹配纯 URL（不在 markdown 格式中的）
    # 先移除已匹配的 markdown 链接，再提取剩余 URL
    content_without_md = re.sub(md_pattern, '', link_source)
    # 兼容 https:// / http:// / //example.com / www.example.com
    url_pattern = r'(?:https?://|//|www\.)[^\s<>\"\'\)\]，。、；：）】}]+'
    for match in re.finditer(url_pattern, content_without_md):
        url = _normalize_candidate(match.group(0))
        if not url.startswith("http") or len(url) <= 10:
            continue
        dedup_key = _normalize_url_for_dedup(url) or url
        if dedup_key not in seen_urls:
            seen_urls.add(dedup_key)
            # 尝试从 URL 生成标题
            title = url.split('//')[-1].split('/')[0]
            links.append({"title": title, "url": url, "description": ""})

    # 3. 兼容 JSON 结构中的 "url": "..."
    json_url_pattern = r'"url"\s*:\s*"([^"]+)"'
    for match in re.finditer(json_url_pattern, link_source):
        url = _normalize_candidate(match.group(1))
        if not url.startswith("http") or len(url) <= 10:
            continue
        dedup_key = _normalize_url_for_dedup(url) or url
        if dedup_key not in seen_urls:
            seen_urls.add(dedup_key)
            title = url.split("//")[-1].split("/")[0]
            links.append({"title": title, "url": url, "description": ""})

    # 提取分析内容（找到总结部分）
    summary_patterns = [
        r'###\s*详细总结分析([\s\S]*)',
        r'###\s*总结分析([\s\S]*)',
        r'##\s*总结([\s\S]*)',
        r'####\s*结论([\s\S]*)',
    ]

    summary = ""
    summary_source = content.strip() or link_source
    for pattern in summary_patterns:
        match = re.search(pattern, summary_source)
        if match:
            summary = match.group(0).strip()
            break

    if not summary:
        summary = summary_source

    # 清理特殊标签
    summary = _clean_ai_tags(summary)

    return links, summary


def _extract_browse_page_links(content: str, extra_text: str = "") -> List[Dict[str, str]]:
    """Extract URLs from Grok-style tool trace lines:
    browse_page {"url":"https://...","instructions":"..."}
    """
    source = f"{content}\n{extra_text}" if extra_text else (content or "")
    if not source:
        return []

    links: List[Dict[str, str]] = []
    seen: set[str] = set()

    # Most common shape in merged SSE text: browse_page {"url":"...","instructions":"..."}
    browse_pattern = re.compile(
        r"browse_page\s*\{\s*\"url\"\s*:\s*\"((?:[^\"\\]|\\.)+)\"(?:\s*,\s*\"instructions\"\s*:\s*\"((?:[^\"\\]|\\.)*)\")?\s*\}",
        flags=re.IGNORECASE,
    )

    def _unescape_json_fragment(value: str) -> str:
        raw = value or ""
        raw = raw.replace("\\/", "/")
        raw = raw.replace('\\"', '"')
        return raw

    for match in browse_pattern.finditer(source):
        raw_url = _unescape_json_fragment(match.group(1).strip())
        instruction = _unescape_json_fragment((match.group(2) or "").strip())
        url = _unwrap_redirect_url(raw_url)
        if not url or not url.startswith("http"):
            continue
        key = _normalize_url_for_dedup(url) or url
        if key in seen:
            continue
        seen.add(key)
        title = f"browse_page: {instruction[:80].strip()}" if instruction else url.split("//")[-1].split("/")[0]
        links.append({"title": title, "url": url, "description": ""})

    return links


mcp = FastMCP("sota-search")


# ============================================================================
# Brave Search 核心
# ============================================================================
async def _search_brave_core(
    query: str,
    max_results: int = 20,
) -> List[Dict[str, str]]:
    try:
        def _fetch_and_parse() -> List[Dict[str, str]]:
            # 构造原始 Brave 搜索 URL
            target_url = f"https://search.brave.com/search?q={quote_plus(query)}"
            visit_url = _get_target_url(target_url)

            logger.info(f"正在搜索: {query}")
            if CF_WORKER_URL:
                logger.info(f"Via Cloudflare Worker: {visit_url}")

            request_headers = {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "close",
            }

            response = curl_requests.get(
                visit_url,
                headers=request_headers,
                proxies=_get_proxies(),
                timeout=SEARCH_TIMEOUT_S,
                allow_redirects=True,
                impersonate=CURL_IMPERSONATE,
                http_version=HTTP_VERSION,
                stream=True,
            )
            try:
                response.raise_for_status()

                buf = bytearray()
                for chunk in response.iter_content():
                    if not chunk:
                        continue
                    buf.extend(chunk)

                    # 收到一部分 HTML 后就尝试解析；如果已经能提取到足够结果就提前结束，避免卡在响应尾部
                    if len(buf) < 16_000:
                        continue
                    if b"data-type" not in buf and b"snippet" not in buf:
                        continue

                    html = bytes(buf).decode("utf-8", errors="replace")
                    parsed = _extract_brave_results(html, max_results)
                    desired = max(1, min(max_results, SEARCH_RESULT_LIMIT))
                    if len(parsed) >= desired:
                        break

                html = bytes(buf).decode("utf-8", errors="replace")
                results = _extract_brave_results(html, max_results)
                logger.info(f"搜索完成，找到 {len(results)} 个结果")
                return results
            finally:
                try:
                    response.close()
                except Exception:
                    pass

        def _extract_brave_results(html: str, limit: int) -> List[Dict[str, str]]:
            soup = BeautifulSoup(html, "lxml")
            items = soup.select('[data-type="web"]')
            if not items:
                items = soup.select(".snippet")

            if limit and limit > 0:
                items = items[:limit]

            extracted: List[Dict[str, str]] = []
            for item in items:
                link = item.select_one("a[href]")
                if not link:
                    continue
                href = link.get("href", "")
                if not href.startswith("http"):
                    continue
                if CF_WORKER_URL and CF_WORKER_URL in href:
                    continue

                title_elem = item.select_one(".snippet-title, .title")
                desc_elem = item.select_one(".snippet-description, .snippet-content, .description")

                extracted.append(
                    {
                        "title": title_elem.get_text(strip=True) if title_elem else "No Title",
                        "url": href,
                        "description": desc_elem.get_text(strip=True) if desc_elem else "",
                    }
                )
            return extracted

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _fetch_and_parse)
    except Exception as e:
        logger.error(f"搜索过程发生错误: {e}")
        return []


async def _search_duckduckgo_core(
    query: str,
    max_results: int = 20,
) -> List[Dict[str, str]]:
    try:
        def _decode_ddg_url(href: str) -> str:
            if not href:
                return ""
            if href.startswith("//"):
                href = "https:" + href
            if href.startswith("/"):
                href = "https://duckduckgo.com" + href
            if not href.startswith("http"):
                return ""
            try:
                parsed = urlparse(href)
                if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
                    params = dict(parse_qsl(parsed.query))
                    uddg = params.get("uddg")
                    if uddg:
                        return uddg
            except Exception:
                return href
            return href

        def _fetch_and_parse() -> List[Dict[str, str]]:
            target_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            visit_url = _get_target_url(target_url)

            logger.info(f"正在搜索(DDG): {query}")
            if CF_WORKER_URL:
                logger.info(f"Via Cloudflare Worker: {visit_url}")

            request_headers = {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "close",
            }

            response = curl_requests.get(
                visit_url,
                headers=request_headers,
                proxies=_get_proxies(),
                timeout=SEARCH_TIMEOUT_S,
                allow_redirects=True,
                impersonate=CURL_IMPERSONATE,
                http_version=HTTP_VERSION,
            )
            response.raise_for_status()
            html = response.text or ""

            soup = BeautifulSoup(html, "lxml")
            results: List[Dict[str, str]] = []
            for item in soup.select(".results .result"):
                link = item.select_one("a.result__a[href]")
                if not link:
                    continue
                href = _decode_ddg_url(link.get("href", ""))
                if not href.startswith("http"):
                    continue
                title = link.get_text(strip=True) or "No Title"
                desc_elem = item.select_one(".result__snippet") or item.select_one(".result__body")
                desc = desc_elem.get_text(strip=True) if desc_elem else ""
                results.append({"title": title, "url": href, "description": desc})
                if max_results and len(results) >= max_results:
                    break
            logger.info(f"搜索完成(DDG)，找到 {len(results)} 个结果")
            return results

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _fetch_and_parse)
    except Exception as e:
        logger.error(f"DDG 搜索过程发生错误: {e}")
        return []


def _curl_get_with_retries(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: int = FETCH_TIMEOUT_S,
    retries: int = 2,
) -> curl_requests.Response:
    request_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "close",
        **(headers or {}),
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            response = curl_requests.get(
                url,
                headers=request_headers,
                proxies=_get_proxies(),
                timeout=timeout_s,
                allow_redirects=True,
                impersonate=CURL_IMPERSONATE,
                http_version=HTTP_VERSION,
            )
            response.raise_for_status()
            return response
        except Exception as e:
            last_error = e
            message = str(e)
            is_timeout = ("curl: (28)" in message) or ("Operation timed out" in message)
            logger.warning(
                "HTTP ERROR attempt=%s/%s url=%s timeout_s=%s is_timeout=%s err=%s",
                attempt,
                retries,
                url,
                timeout_s,
                is_timeout,
                message,
            )
            if attempt >= retries or not is_timeout:
                raise
            timeout_s = max(timeout_s * 2, timeout_s + 10)
            # 简单退避
            time.sleep(0.3 * attempt)
    assert last_error is not None
    raise last_error


def _fetch_zhihu_answer_content(url: str, mode: str) -> Optional[Dict[str, Any]]:
    answer_id = _extract_zhihu_answer_id(url)
    if not answer_id:
        return None

    api_url = (
        "https://www.zhihu.com/api/v4/answers/"
        f"{answer_id}?include=content,excerpt,content_need_truncated,segment_infos"
    )

    def _build_result(content_html: str, via_worker: bool) -> Dict[str, Any]:
        if mode == "html":
            limited_html, was_truncated = _limit_content_length(content_html)
            return {
                "success": True,
                "url": url,
                "via_worker": via_worker,
                "via_playwright": False,
                "via_zhihu_api": True,
                "html": limited_html,
                "truncated": was_truncated,
            }
        wrapped_html = f"<html><body>{content_html}</body></html>"
        if mode == "markdown":
            extracted = _extract_best_content(wrapped_html, url=url, output_format="markdown")
            limited_md, was_truncated = _limit_content_length(extracted.get("content", ""))
            return {
                "success": True,
                "url": url,
                "via_worker": via_worker,
                "via_playwright": False,
                "via_zhihu_api": True,
                "markdown": limited_md,
                "truncated": was_truncated,
                "extractor": extracted.get("extractor"),
                "quality_score": extracted.get("quality_score"),
                "quality_metrics": {
                    "char_len": extracted.get("char_len"),
                    "line_count": extracted.get("line_count"),
                    "unique_line_ratio": extracted.get("unique_line_ratio"),
                    "noise_line_ratio": extracted.get("noise_line_ratio"),
                },
                "degraded": extracted.get("degraded", False),
            }

        extracted = _extract_best_content(wrapped_html, url=url, output_format="txt")
        limited_text, was_truncated = _limit_content_length(extracted.get("content", ""))
        return {
            "success": True,
            "url": url,
            "via_worker": via_worker,
            "via_playwright": False,
            "via_zhihu_api": True,
            "text": limited_text,
            "truncated": was_truncated,
            "extractor": extracted.get("extractor"),
            "quality_score": extracted.get("quality_score"),
            "quality_metrics": {
                "char_len": extracted.get("char_len"),
                "line_count": extracted.get("line_count"),
                "unique_line_ratio": extracted.get("unique_line_ratio"),
                "noise_line_ratio": extracted.get("noise_line_ratio"),
            },
            "degraded": extracted.get("degraded", False),
        }

    def _try_fetch(target_url: str, via_worker: bool) -> Optional[Dict[str, Any]]:
        response = _curl_get_with_retries(
            target_url,
            headers={"Accept": "application/json"},
            timeout_s=FETCH_TIMEOUT_S,
            retries=2,
        )
        try:
            data = response.json()
        except Exception:
            return None
        content_html = data.get("content") if isinstance(data, dict) else None
        if not content_html:
            return None
        if data.get("content_need_truncated") and data.get("segment_infos"):
            compact_content = re.sub(r"\s+", "", content_html)
            extra_parts = []
            for segment in data.get("segment_infos") or []:
                text = (segment or {}).get("text") or ""
                if not text.strip():
                    continue
                compact_text = re.sub(r"\s+", "", text)
                if compact_text and compact_text[:20] in compact_content:
                    continue
                extra_parts.append(f"<p>{html_lib.escape(text.strip())}</p>")
            if extra_parts:
                content_html = content_html + "".join(extra_parts)
        return _build_result(content_html, via_worker)

    if CF_WORKER_URL:
        try:
            result = _try_fetch(_get_target_url(api_url), True)
        except Exception:
            result = None
        if result:
            return result
        try:
            return _try_fetch(api_url, False)
        except Exception:
            return None
    try:
        return _try_fetch(api_url, False)
    except Exception:
        return None


def _discourse_topic_json_url(url: str) -> Optional[str]:
    parsed = urlparse(url or "")
    if not parsed.netloc:
        return None
    path = parsed.path or ""
    if path.endswith(".json"):
        return urlunparse((parsed.scheme or "https", parsed.netloc, path, "", "", ""))

    segments = [seg for seg in path.split("/") if seg]
    if "t" not in segments:
        return None

    t_index = segments.index("t")
    topic_id_index: Optional[int] = None
    for i in range(t_index + 1, len(segments)):
        if segments[i].isdigit():
            topic_id_index = i
            break
    if topic_id_index is None:
        return None

    json_path = "/" + "/".join(segments[: topic_id_index + 1]) + ".json"
    return urlunparse((parsed.scheme or "https", parsed.netloc, json_path, "", "", ""))


def _extract_discourse_topic_markdown(data: Any, *, url: str) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    title = (data.get("title") or "").strip()
    post_stream = data.get("post_stream") if isinstance(data.get("post_stream"), dict) else {}
    posts = post_stream.get("posts") if isinstance(post_stream.get("posts"), list) else []
    if not posts:
        return None

    parts: List[str] = []
    if title:
        parts.append(f"# {title}")

    for post in posts:
        if not isinstance(post, dict):
            continue
        cooked = post.get("cooked") or ""
        cooked = cooked.strip()
        if not cooked:
            continue
        username = (post.get("username") or "").strip()
        post_number = post.get("post_number")
        if username:
            header = f"## {username}"
            if isinstance(post_number, int):
                header = f"{header} · #{post_number}"
            parts.append(header)

        wrapped = f"<html><body>{cooked}</body></html>"
        md = _trafilatura_extract(
            wrapped,
            url=url,
            output_format="markdown",
            favor_precision=True,
            include_links=True,
        )
        if not md:
            md = _html_to_text(wrapped)
        md = _clean_extracted_markdown(md)
        if md:
            parts.append(md)

    combined = "\n\n".join([p for p in parts if p and p.strip()]).strip()
    return combined or None


def _fetch_discourse_topic_content(url: str) -> Optional[Dict[str, Any]]:
    json_url = _discourse_topic_json_url(url)
    if not json_url:
        return None

    try:
        response = _curl_get_with_retries(
            _get_target_url(json_url),
            headers={"Accept": "application/json"},
            timeout_s=FETCH_TIMEOUT_S,
            retries=2,
        )
    except Exception:
        return None

    raw = response.text or ""
    if _looks_like_blocked_text(raw):
        return None

    try:
        data = response.json()
    except Exception:
        return None

    markdown = _extract_discourse_topic_markdown(data, url=url)
    if not markdown:
        return None

    limited_md, was_truncated = _limit_content_length(markdown)
    metrics = _score_content(limited_md)
    return {
        "success": True,
        "url": url,
        "via_worker": bool(CF_WORKER_URL),
        "via_playwright": False,
        "status_code": response.status_code,
        "markdown": limited_md,
        "truncated": was_truncated,
        "blocked": False,
        "extractor": "adapter:discourse:topic_json",
        "quality_score": metrics.get("quality_score"),
        "quality_metrics": {
            "char_len": metrics.get("char_len"),
            "line_count": metrics.get("line_count"),
            "unique_line_ratio": metrics.get("unique_line_ratio"),
            "noise_line_ratio": metrics.get("noise_line_ratio"),
        },
        "degraded": False,
    }


async def _fetch_with_playwright(
    url: str,
    mode: str,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    if not PLAYWRIGHT_FALLBACK:
        return {"success": False, "url": url, "error": "Playwright fallback disabled"}

    try:
        from playwright.async_api import async_playwright
        from playwright_stealth import Stealth
    except Exception as e:
        return {"success": False, "url": url, "error": f"Playwright not available: {e}"}

    headless = os.getenv("PW_HEADLESS", "1").lower() not in ("0", "false", "no")
    user_agent = os.getenv("PW_USER_AGENT", USER_AGENT)
    accept_language = os.getenv(
        "PW_ACCEPT_LANGUAGE",
        "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    )
    locale = os.getenv("PW_LOCALE", "zh-CN")
    timezone_id = os.getenv("PW_TIMEZONE", "Asia/Shanghai")
    viewport = _parse_viewport(os.getenv("PW_VIEWPORT", "1366x768"))
    device_scale_factor = float(os.getenv("PW_DEVICE_SCALE", "2"))

    extra_headers: Dict[str, str] = {"Accept-Language": accept_language}
    if headers:
        for key, value in headers.items():
            if key.lower() == "user-agent":
                user_agent = value
            elif key.lower() == "accept-language":
                extra_headers["Accept-Language"] = value
            else:
                extra_headers[key] = value

    context_kwargs: Dict[str, Any] = {
        "user_agent": user_agent,
        "locale": locale,
        "timezone_id": timezone_id,
        "color_scheme": "light",
        "device_scale_factor": device_scale_factor,
    }
    if viewport:
        context_kwargs["viewport"] = viewport

    try:
        launch_args: Dict[str, Any] = {"headless": headless}
        if PROXY_CONFIG:
            launch_args["proxy"] = {"server": PROXY_CONFIG}

        async with async_playwright() as p:
            executable_override = (
                os.getenv("PW_CHROMIUM_EXECUTABLE_PATH")
                or os.getenv("PW_EXECUTABLE_PATH")
                or os.getenv("PLAYWRIGHT_EXECUTABLE_PATH")
            )
            if executable_override and os.path.exists(executable_override):
                launch_args["executable_path"] = executable_override
            else:
                resolved = _resolve_playwright_executable_path(getattr(p.chromium, "executable_path", ""))
                if resolved:
                    launch_args["executable_path"] = resolved

            browser = await p.chromium.launch(**launch_args)
            context = await browser.new_context(**context_kwargs)
            await context.set_extra_http_headers(extra_headers)
            page = await context.new_page()
            await Stealth().apply_stealth_async(page)
            await page.goto(url, wait_until="domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT_MS)
            try:
                await page.wait_for_load_state("networkidle", timeout=min(PLAYWRIGHT_TIMEOUT_MS, 5000))
            except Exception:
                # Some sites keep long-polling; ignore if networkidle never settles.
                pass

            for _ in range(max(1, PLAYWRIGHT_CHALLENGE_WAIT)):
                try:
                    title = await page.title()
                except Exception:
                    # Page may be mid-navigation; wait and retry.
                    await page.wait_for_timeout(1000)
                    continue
                if not _looks_like_challenge_text(title):
                    break
                await page.wait_for_timeout(1000)

            try:
                html = await page.content()
            except Exception:
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=PLAYWRIGHT_TIMEOUT_MS)
                    html = await page.content()
                except Exception as e:
                    return {"success": False, "url": url, "error": str(e), "via_playwright": True}
            blocked = _looks_like_blocked_text(html)

            if mode == "html":
                limited_html, was_truncated = _limit_content_length(html)
                result = {
                    "success": True,
                    "url": url,
                    "via_worker": False,
                    "via_playwright": True,
                    "html": limited_html,
                    "truncated": was_truncated,
                    "blocked": blocked,
                }
            elif mode == "text":
                loop = asyncio.get_running_loop()

                def _extract_sync() -> Dict[str, Any]:
                    return _extract_best_content(html, url=url, output_format="txt")

                extracted = await loop.run_in_executor(None, _extract_sync)
                if blocked and extracted.get("quality_score", 0) < 65:
                    degraded = _build_degraded_text(html) or ""
                    degraded = _clean_extracted_text(degraded)
                    metrics = _score_content(degraded)
                    extracted = {
                        "content": degraded,
                        "extractor": "meta:blocked",
                        "degraded": True,
                        **metrics,
                    }
                limited_text, was_truncated = _limit_content_length(extracted.get("content", ""))
                result = {
                    "success": True,
                    "url": url,
                    "via_worker": False,
                    "via_playwright": True,
                    "text": limited_text,
                    "truncated": was_truncated,
                    "blocked": blocked,
                    "extractor": extracted.get("extractor"),
                    "quality_score": extracted.get("quality_score"),
                    "quality_metrics": {
                        "char_len": extracted.get("char_len"),
                        "line_count": extracted.get("line_count"),
                        "unique_line_ratio": extracted.get("unique_line_ratio"),
                        "noise_line_ratio": extracted.get("noise_line_ratio"),
                    },
                    "degraded": extracted.get("degraded", False),
                }
            elif mode == "markdown":
                loop = asyncio.get_running_loop()

                def _extract_sync() -> Dict[str, Any]:
                    return _extract_best_content(html, url=url, output_format="markdown")

                extracted = await loop.run_in_executor(None, _extract_sync)
                if blocked and extracted.get("quality_score", 0) < 65:
                    degraded = _build_degraded_markdown(html) or ""
                    degraded = _clean_extracted_markdown(degraded)
                    metrics = _score_content(degraded)
                    extracted = {
                        "content": degraded,
                        "extractor": "meta:blocked",
                        "degraded": True,
                        **metrics,
                    }
                limited_md, was_truncated = _limit_content_length(extracted.get("content", ""))
                result = {
                    "success": True,
                    "url": url,
                    "via_worker": False,
                    "via_playwright": True,
                    "markdown": limited_md,
                    "truncated": was_truncated,
                    "blocked": blocked,
                    "extractor": extracted.get("extractor"),
                    "quality_score": extracted.get("quality_score"),
                    "quality_metrics": {
                        "char_len": extracted.get("char_len"),
                        "line_count": extracted.get("line_count"),
                        "unique_line_ratio": extracted.get("unique_line_ratio"),
                        "noise_line_ratio": extracted.get("noise_line_ratio"),
                    },
                    "degraded": extracted.get("degraded", False),
                }
            elif mode in ("meta", "metadata"):
                metadata = _extract_metadata(html)
                result = {
                    "success": True,
                    "url": url,
                    "via_worker": False,
                    "via_playwright": True,
                    "blocked": blocked,
                    **metadata,
                }
            else:
                result = {
                    "success": False,
                    "url": url,
                    "error": f"Unsupported mode: {mode}",
                    "via_playwright": True,
                }

            await context.close()
            await browser.close()
            return result
    except Exception as e:
        return {"success": False, "url": url, "error": str(e), "via_playwright": True}

# ============================================================================
# Fetch 工具 (支持 CF Worker)
# ============================================================================
@mcp.tool()
async def fetch(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    def _fetch():
        zhihu_result = _fetch_zhihu_answer_content(url, mode="markdown")
        if zhihu_result:
            return zhihu_result

        discourse_result = _fetch_discourse_topic_content(url)
        if discourse_result:
            return discourse_result

        if _prefer_playwright_for_url(url) and PLAYWRIGHT_FALLBACK:
            return {"success": False, "url": url, "needs_playwright": True}

        target_url = _get_target_url(url)
        response = _curl_get_with_retries(
            target_url,
            headers=headers,
            timeout_s=FETCH_TIMEOUT_S,
            retries=2,
        )

        raw_html = response.text or ""
        if _looks_like_blocked_text(raw_html) and PLAYWRIGHT_FALLBACK:
            return {
                "success": False,
                "url": url,
                "needs_playwright": True,
                "via_worker": bool(CF_WORKER_URL),
                "status_code": response.status_code,
            }

        blocked = _looks_like_blocked_text(raw_html)
        extracted = _extract_best_content(raw_html, url=url, output_format="markdown")
        if blocked and extracted.get("quality_score", 0) < 65:
            degraded = _build_degraded_markdown(raw_html) or ""
            degraded = _clean_extracted_markdown(degraded)
            metrics = _score_content(degraded)
            extracted = {
                "content": degraded,
                "extractor": "meta:blocked",
                "degraded": True,
                **metrics,
            }
        limited_md, was_truncated = _limit_content_length(extracted.get("content", ""))

        return {
            "success": True,
            "url": url,
            "via_worker": bool(CF_WORKER_URL),
            "via_playwright": False,
            "status_code": response.status_code,
            "markdown": limited_md,
            "truncated": was_truncated,
            "blocked": blocked,
            "extractor": extracted.get("extractor"),
            "quality_score": extracted.get("quality_score"),
            "quality_metrics": {
                "char_len": extracted.get("char_len"),
                "line_count": extracted.get("line_count"),
                "unique_line_ratio": extracted.get("unique_line_ratio"),
                "noise_line_ratio": extracted.get("noise_line_ratio"),
            },
            "degraded": extracted.get("degraded", False),
        }

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(None, _fetch)
    except Exception as e:
        logger.error(f"抓取失败 {url}: {e}")
        if PLAYWRIGHT_FALLBACK:
            return await _fetch_with_playwright(url, mode="markdown", headers=headers)
        return {"success": False, "url": url, "error": str(e)}

    if result.get("needs_playwright") and PLAYWRIGHT_FALLBACK:
        return await _fetch_with_playwright(url, mode="markdown", headers=headers)

    if _looks_like_blocked_text(result.get("markdown", "")) and PLAYWRIGHT_FALLBACK:
        pw_result = await _fetch_with_playwright(url, mode="markdown", headers=headers)
        if pw_result.get("success"):
            return pw_result
        result["blocked"] = True
        result["playwright_error"] = pw_result.get("error", "Playwright fallback failed")

    return result

@mcp.tool()
async def web_search(query: str) -> Dict[str, Any]:
    """
    搜索工具：结合 AI 深度搜索与浏览器搜索结果（Brave/DDG）。

    - 固定最多返回 25 条链接（不支持参数控制数量）
    - 普通搜索：browse_page URL > 其它 AI URL > 浏览器 URL
    - site: 搜索：browse_page URL > 浏览器 URL > 其它 AI URL（并关闭同域名上限）
    """

    logger.info(f"收到搜索请求: query='{query}'")

    is_site = _is_site_query(query)

    ai_summary = ""
    ai_error = ""
    ai_priority_links: List[Dict[str, str]] = []
    ai_links_only: List[Dict[str, str]] = []
    browser_links: List[Dict[str, str]] = []

    # 定义 AI 搜索的异步包装函数
    async def _ai_search_async() -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str, str]:
        if not _llm_configured():
            return [], [], "", "llm_not_configured"

        def _ai_search() -> Tuple[str, str]:
            prompt = f"""你是一个研究型搜索助手。请通过联网检索与交叉验证，给出高质量、细节充分的回答，避免编造。

输出要求：
1) 正文：自然语言写作，不要输出任何 URL/链接（包括 http/https/www 开头的内容），也不要出现“参考来源/References/Sources”等段落标题。
2) 末尾追加一段 SOURCES（必须以单独一行 'SOURCES:' 开头），其后每行一个你参考过的来源 URL（最多 30 条）。

用户问题：{query}"""

            return _call_openai_chat_completions(prompt)

        loop = asyncio.get_running_loop()
        try:
            raw_content, raw_reasoning = await loop.run_in_executor(None, _ai_search)
        except Exception as e:
            logger.warning("AI 搜索不可用，已降级: %s", e)
            return [], [], "", str(e)

        priority_links = _extract_browse_page_links(raw_content, extra_text=raw_reasoning)
        ai_links, summary = _parse_markdown_links(raw_content, extra_text=raw_reasoning)
        priority_keys = {_normalize_url_for_dedup(l.get("url", "")) or l.get("url", "") for l in priority_links}
        ai_links_others = [
            link
            for link in ai_links
            if ((_normalize_url_for_dedup(link.get("url", "")) or link.get("url", "")) not in priority_keys)
        ]
        summary = _strip_urls(summary)
        logger.info(
            "AI 搜索完成，提取到 %s 个链接（browse_page 优先链接 %s）",
            len(ai_links),
            len(priority_links),
        )
        return priority_links, ai_links_others, summary, ""

    async def _browser_search_async() -> List[Dict[str, str]]:
        internal_limit = max(SEARCH_RESULT_LIMIT * 2, 20)
        results = await _search_brave_core(query=query, max_results=internal_limit)
        if results:
            logger.info(f"普通搜索完成(Brave)，获取到 {len(results)} 个结果")
            return results
        fallback = await _search_duckduckgo_core(query=query, max_results=internal_limit)
        logger.info(f"普通搜索完成(DDG)，获取到 {len(fallback)} 个结果")
        return fallback

    # 并行执行两个搜索
    use_ai = _llm_configured()
    browser_task = asyncio.create_task(_browser_search_async())
    if use_ai:
        ai_task = asyncio.create_task(_ai_search_async())
        ai_result, browser_result = await asyncio.gather(ai_task, browser_task, return_exceptions=True)

        if isinstance(ai_result, Exception):
            logger.warning("AI 搜索失败，已降级为普通搜索: %s", ai_result)
            ai_error = str(ai_result)
        else:
            ai_priority_links, ai_links_only, ai_summary, ai_error = ai_result

        if isinstance(browser_result, Exception):
            logger.error("普通搜索失败: %s", browser_result)
        else:
            browser_links = browser_result
    else:
        browser_links = await browser_task

    merged_links = (
        ai_priority_links + browser_links + ai_links_only
        if is_site
        else ai_priority_links + ai_links_only + browser_links
    )

    # 去重（按 URL，含规范化去追踪参数 + 常见重定向解包）
    seen_urls: set[str] = set()
    unique_links: List[Dict[str, str]] = []
    for link in merged_links:
        if not isinstance(link, dict):
            continue
        raw_url = link.get("url", "")
        url = _unwrap_redirect_url(raw_url)
        if not url or not url.startswith("http"):
            continue
        dedup_key = _normalize_url_for_dedup(url) or url
        if dedup_key in seen_urls:
            continue
        seen_urls.add(dedup_key)
        cleaned = {
            "title": str(link.get("title") or ""),
            "url": url,
        }
        unique_links.append(cleaned)

    # 结果裁剪：固定 15，并限制同域名占比（site: 查询关闭同域上限）
    limit = SEARCH_RESULT_LIMIT
    try:
        max_per_domain = int(os.getenv("SEARCH_MAX_PER_DOMAIN", "2"))
    except Exception:
        max_per_domain = 2
    if max_per_domain < 0:
        max_per_domain = 0
    if is_site:
        max_per_domain = 0

    domain_counts: Dict[str, int] = {}
    limited_links: List[Dict[str, str]] = []
    for link in unique_links:
        url = link.get("url", "")
        host = _get_hostname(url) if url else ""
        if max_per_domain > 0 and host:
            if domain_counts.get(host, 0) >= max_per_domain:
                continue
        limited_links.append(link)
        if host:
            domain_counts[host] = domain_counts.get(host, 0) + 1
        if len(limited_links) >= limit:
            break

    return {
        "success": True,
        "query": query,
        "links": limited_links,
        "ai_summary": ai_summary,
        "ai_error": ai_error,
    }

def main():
    logger.info("SOTA Search MCP Server 启动中...")
    if CF_WORKER_URL:
        logger.info(f"启用 Cloudflare Worker 代理: {CF_WORKER_URL}")
        logger.info("注意：流量将通过 Worker 转发，目标网站看到的 IP 为 Cloudflare 节点 IP")

    if PROXY_CONFIG:
        logger.info(f"使用本地代理: {PROXY_CONFIG}")

    if OPENAI_API_KEY and OPENAI_BASE_URL:
        logger.info(f"启用 AI 搜索，模型: {OPENAI_MODEL}")

    logger.info("等待 MCP 客户端连接...")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
