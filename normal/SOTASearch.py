"""
SOTA Search MCP Server - AI 增强搜索

功能：
- 原有 web_search: 使用 Brave Search 进行普通搜索
- 新增 ai_search: 使用 OpenAI API 进行 AI 深度搜索，返回搜索链接和总结
- 支持 --cf-worker 参数，通过 Cloudflare Worker 代理流量

变更(curl_cffi 版）：
- 移除 Playwright / cloudscraper
- 统一使用 curl_cffi 发起 HTTP 请求并解析 HTML
"""

import argparse
import asyncio
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, quote
from dotenv import load_dotenv

load_dotenv()

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


def _clean_ai_tags(text: str) -> str:
    """清理 AI 返回内容中的特殊标签"""
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
        if re.match(r"^\s*(参考来源|参考资料|参考链接|Sources|References)\b.*[:：]\s*$", line):
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

    # 1. 匹配 markdown 链接格式 [title](url)
    md_pattern = r'\[([^\]]+)\]\((https?://[^)]+)\)'
    for match in re.finditer(md_pattern, link_source):
        title = match.group(1).strip()
        url = match.group(2).strip()
        if url not in seen_urls:
            seen_urls.add(url)
            links.append({"title": title, "url": url, "description": ""})

    # 2. 匹配纯 URL（不在 markdown 格式中的）
    # 先移除已匹配的 markdown 链接，再提取剩余 URL
    content_without_md = re.sub(md_pattern, '', link_source)
    url_pattern = r'https?://[^\s<>\"\'\)\]，。、；：）】}]+'
    for match in re.finditer(url_pattern, content_without_md):
        url = match.group(0).strip()
        # 清理 URL 末尾可能的标点
        url = re.sub(r'[.,;:!?]+$', '', url)
        if url not in seen_urls and len(url) > 10:
            seen_urls.add(url)
            # 尝试从 URL 生成标题
            title = url.split('//')[-1].split('/')[0]
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
                    if len(parsed) >= max(1, min(max_results, 3)):
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
        return [{"title": "搜索失败", "url": "", "description": f"错误: {str(e)}"}]


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

# ============================================================================
# Fetch 工具 (支持 CF Worker)
# ============================================================================
@mcp.tool()
async def fetch_html(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        def _fetch():
            target_url = _get_target_url(url)

            response = _curl_get_with_retries(
                target_url,
                headers=headers,
                timeout_s=FETCH_TIMEOUT_S,
                retries=2,
            )

            html_content, was_truncated = _limit_content_length(response.text)
            return {
                "success": True,
                "url": url,
                "via_worker": bool(CF_WORKER_URL),
                "status_code": response.status_code,
                "html": html_content,
                "truncated": was_truncated,
            }

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _fetch)
        return result
    except Exception as e:
        logger.error(f"抓取失败 {url}: {e}")
        return {"success": False, "url": url, "error": str(e)}


@mcp.tool()
async def fetch_text(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        def _fetch():
            target_url = _get_target_url(url)
            response = _curl_get_with_retries(
                target_url,
                headers=headers,
                timeout_s=FETCH_TIMEOUT_S,
                retries=2,
            )

            soup = BeautifulSoup(response.text, "lxml")
            for tag in soup(["script", "style", "header", "footer", "nav"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            limited_text, was_truncated = _limit_content_length(text)

            return {
                "success": True,
                "url": url,
                "via_worker": bool(CF_WORKER_URL),
                "text": limited_text,
                "truncated": was_truncated,
            }

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _fetch)
        return result
    except Exception as e:
        logger.error(f"抓取文本失败 {url}: {e}")
        return {"success": False, "url": url, "error": str(e)}


@mcp.tool()
async def fetch_metadata(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        def _fetch():
            target_url = _get_target_url(url)
            response = _curl_get_with_retries(
                target_url,
                headers=headers,
                timeout_s=FETCH_TIMEOUT_S,
                retries=2,
            )

            soup = BeautifulSoup(response.text, "lxml")
            title = ""
            if soup.find("title"):
                title = soup.find("title").get_text(strip=True)

            description = ""
            desc_tag = soup.find("meta", attrs={"name": "description"})
            if desc_tag:
                description = desc_tag.get("content", "")

            links = []
            for a in soup.find_all("a", href=True, limit=50):
                # 注意：经过 Worker 后，HTML 中的相对路径链接可能会失效
                # 解析出来的 href 仍然是原始 HTML 中的样子
                links.append({"text": a.get_text(strip=True), "href": a["href"]})

            links_str = str(links)
            _, was_truncated = _limit_content_length(links_str)
            if was_truncated:
                avg_length = len(links_str) / len(links) if links else 0
                keep_count = max(1, int(MAX_TOKEN_LIMIT * 4 / avg_length) if avg_length > 0 else 0)
                links = links[:keep_count]

            return {
                "success": True,
                "url": url,
                "via_worker": bool(CF_WORKER_URL),
                "title": title,
                "description": description,
                "links": links,
                "truncated": was_truncated,
            }

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _fetch)
        return result
    except Exception as e:
        logger.error(f"提取元数据失败 {url}: {e}")
        return {"success": False, "url": url, "error": str(e)}

@mcp.tool()
async def web_search(query: str, max_results: int = 10) -> Dict[str, Any]:
    """
    搜索工具：结合 AI 深度搜索和普通搜索
    返回合并的链接列表和 AI 分析总结
    """
    logger.info(f"收到搜索请求: query='{query}'")

    all_links = []
    ai_summary = ""
    ai_content = ""
    ai_links_only: List[Dict[str, str]] = []

    # 定义 AI 搜索的异步包装函数
    async def _ai_search_async() -> Tuple[List[Dict[str, str]], str, str]:
        if not _llm_configured():
            return [], "", ""

        def _ai_search():
            client = _get_openai_client()
            if client is None:
                raise RuntimeError("OpenAI client not configured")

            prompt = f"""你是一个研究型搜索助手。目标：尽可能通过广泛检索与交叉验证，给出高质量、细节充分的回答，避免编造。

请覆盖：原理/实现/最佳实践/对比/限制/最新进展。请在内部使用中英文关键词做同义扩展（含缩写、别名、版本号、标准号），生成多组检索 query 并迭代检索；优先官方文档、标准/RFC、学术论文、项目主页、权威媒体，尽量近 3 年（经典标准除外）；关键结论尽量做到多来源交叉验证，无法验证的内容请明确标注“不确定/推测”；去重：同域名最多 2 条，宁可少而准。

输出要求：最终回答请用自然语言写作，不要使用固定模板/标题/列表；正文不要输出任何 URL/链接（包括以 http/https 开头的内容），也不要出现“参考来源/References/Sources”等段落或占位符。你参考过的来源 URL 仅写在推理/思考（reasoning）里，按行列出即可（不要在正文提及或重复）。

用户问题：{query}"""

            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            message = response.choices[0].message
            content = getattr(message, "content", "") or ""
            reasoning = getattr(message, "reasoning_content", "") or ""
            return content, reasoning

        loop = asyncio.get_running_loop()
        try:
            ai_content, ai_reasoning = await loop.run_in_executor(None, _ai_search)
        except Exception as e:
            logger.warning("AI 搜索不可用，已降级: %s", e)
            return [], "", ""
        ai_links, summary = _parse_markdown_links(ai_content, extra_text=ai_reasoning)
        summary = _strip_urls(summary)
        cleaned_content = _strip_urls(_clean_ai_tags(ai_content))
        logger.info(f"AI 搜索完成，提取到 {len(ai_links)} 个链接")
        return ai_links, summary, cleaned_content

    # 定义普通搜索的异步包装函数
    async def _brave_search_async() -> List[Dict[str, str]]:
        results = await _search_brave_core(query=query, max_results=max_results)
        logger.info(f"普通搜索完成，获取到 {len(results)} 个结果")
        return results

    # 并行执行两个搜索
    use_ai = _llm_configured()
    brave_task = asyncio.create_task(_brave_search_async())
    if use_ai:
        ai_task = asyncio.create_task(_ai_search_async())
        results = await asyncio.gather(ai_task, brave_task, return_exceptions=True)
        ai_result = results[0]
        brave_result = results[1]

        if isinstance(ai_result, Exception):
            logger.warning("AI 搜索失败，已降级为普通搜索: %s", ai_result)
        else:
            ai_links_only, ai_summary, ai_content = ai_result
            all_links.extend(ai_links_only)

        if isinstance(brave_result, Exception):
            logger.error("普通搜索失败: %s", brave_result)
        else:
            all_links.extend(brave_result)
    else:
        brave_result = await brave_task
        all_links.extend(brave_result)

    # 去重（按 URL）
    seen_urls = set()
    unique_links = []
    for link in all_links:
        url = link.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_links.append(link)

    return {
        "success": True,
        "query": query,
        "links": unique_links,
        "ai_links": ai_links_only,
        "ai_content": ai_content,
        "ai_summary": ai_summary,
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
