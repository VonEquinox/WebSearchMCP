"""
SOTA Search MCP Server - AI 增强搜索

功能：
- 原有 web_search: 使用 Brave Search 进行普通搜索
- 新增 ai_search: 使用 OpenAI API 进行 AI 深度搜索，返回搜索链接和总结
- 支持 --cf-worker 参数，通过 Cloudflare Worker 代理流量
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus, quote

# 设置 Playwright 浏览器路径为项目本地目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = os.path.join(_SCRIPT_DIR, ".playwright-browsers")

import cloudscraper
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
from openai import OpenAI
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

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
    default=None,
    help="本地代理设置，例如: http://127.0.0.1:7890",
)
parser.add_argument(
    "--cf-worker",
    type=str,
    default=None,
    help="Cloudflare Worker 地址，例如: https://xxx.xxx.workers.dev",
)
parser.add_argument(
    "--openai-api-key",
    type=str,
    default=None,
    help="OpenAI API Key",
)
parser.add_argument(
    "--openai-base-url",
    type=str,
    default=None,
    help="OpenAI API Base URL，例如: https://api.openai.com/v1",
)
parser.add_argument(
    "--openai-model",
    type=str,
    default="gpt-4o",
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

# 全局 scraper 实例
_scraper: Optional[cloudscraper.CloudScraper] = None

# Playwright配置
HEADLESS = True
TIMEOUT_MS = 20_000
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

MAX_TOKEN_LIMIT = 10000


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


def _get_scraper() -> cloudscraper.CloudScraper:
    global _scraper
    if _scraper is None:
        _scraper = cloudscraper.create_scraper()
    return _scraper


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


def _parse_markdown_links(content: str) -> Tuple[List[Dict[str, str]], str]:
    """
    从 AI 返回的内容中解析链接
    支持格式：
    1. markdown 链接 [title](url)
    2. 纯 URL https://...
    3. 带括号的链接 (https://...)
    返回：(链接列表, 清理后的分析内容)
    """
    links = []
    seen_urls = set()

    # 1. 匹配 markdown 链接格式 [title](url)
    md_pattern = r'\[([^\]]+)\]\((https?://[^)]+)\)'
    for match in re.finditer(md_pattern, content):
        title = match.group(1).strip()
        url = match.group(2).strip()
        if url not in seen_urls:
            seen_urls.add(url)
            links.append({"title": title, "url": url, "description": ""})

    # 2. 匹配纯 URL（不在 markdown 格式中的）
    # 先移除已匹配的 markdown 链接，再提取剩余 URL
    content_without_md = re.sub(md_pattern, '', content)
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
    for pattern in summary_patterns:
        match = re.search(pattern, content)
        if match:
            summary = match.group(0).strip()
            break

    if not summary:
        summary = content

    # 清理特殊标签
    summary = _clean_ai_tags(summary)

    return links, summary


# ============================================================================
# 全局浏览器管理器
# ============================================================================
class BrowserManager:
    def __init__(self):
        self._playwright = None
        self._browser = None
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def get_browser(self):
        async with self._get_lock():
            if self._browser is None or not self._browser.is_connected():
                await self._start_browser()
            return self._browser

    async def _start_browser(self):
        try:
            await self._cleanup()
            self._playwright = await async_playwright().start()

            launch_kwargs = {"headless": HEADLESS}
            # 如果配置了本地代理，Playwright 依然使用它
            # 注意：如果使用了 CF Worker，浏览器访问的是 Worker 地址
            if PROXY_CONFIG:
                launch_kwargs["proxy"] = {"server": PROXY_CONFIG}
                logger.info(f"浏览器使用本地代理: {PROXY_CONFIG}")

            self._browser = await self._playwright.chromium.launch(**launch_kwargs)
            logger.info("持久化浏览器已启动")

        except Exception as e:
            logger.error(f"启动浏览器失败: {e}")
            raise

    async def new_page(self):
        browser = await self.get_browser()
        context = await browser.new_context(user_agent=USER_AGENT, java_script_enabled=False)
        page = await context.new_page()
        page.set_default_timeout(TIMEOUT_MS)
        return page, context

    async def _cleanup(self):
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None

        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    async def close(self):
        async with self._get_lock():
            await self._cleanup()
            logger.info("持久化浏览器已关闭")


_browser_manager = BrowserManager()
mcp = FastMCP("sota-search")


# ============================================================================
# Brave Search 核心
# ============================================================================
async def _search_brave_core(
    query: str,
    max_results: int = 20,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    context = None

    try:
        page, context = await _browser_manager.new_page()

        # 构造原始 Brave 搜索 URL
        target_url = f"https://search.brave.com/search?q={quote_plus(query)}"
        
        # 转换为实际访问的 URL (直连 或 经 CF Worker)
        visit_url = _get_target_url(target_url)

        logger.info(f"正在搜索: {query}")
        if CF_WORKER_URL:
            logger.info(f"Via Cloudflare Worker: {visit_url}")

        # 【修改点1】使用 'commit' 策略
        # 只要服务器发回了 HTML 响应头就算成功，不再等待 DOM 加载完成
        # 配合禁用 JS，速度极快
        await page.goto(visit_url, wait_until="commit", timeout=60000)

        # 【修改点2】移除 wait_for_selector，改用简单的缓冲
        # 因为没有 JS，DOM 事件可能不完整，直接等 1 秒让 HTML 接收完毕即可
        await asyncio.sleep(1.5)

        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        
        # 【修改点3】Brave 静态 HTML 的选择器优化
        # 你的 curl 结果显示 content 都在 #results 里面
        items = soup.select('[data-type="web"]') # 保持原样，通常有效
        
        # 如果标准选择器没找到，尝试备用方案 (针对无 JS 版页面结构)
        if not items:
            items = soup.select('.snippet') 

        if max_results and max_results > 0:
            items = items[:max_results]

        for item in items:
            link = item.select_one("a[href]")
            if not link:
                continue
            href = link.get("href", "")
            
            # 过滤
            if not href.startswith("http"):
                continue
            if CF_WORKER_URL and CF_WORKER_URL in href:
                continue

            # 提取标题和描述
            title_elem = item.select_one(".snippet-title, .title")
            desc_elem = item.select_one(".snippet-description, .snippet-content, .description")

            title = (title_elem.get_text(strip=True) if title_elem else "No Title")
            body = (desc_elem.get_text(strip=True) if desc_elem else "")

            results.append(
                {
                    "title": title,
                    "url": href,
                    "description": body,
                }
            )

        logger.info(f"搜索完成，找到 {len(results)} 个结果")

    except PlaywrightError as e:
        logger.error(f"Playwright 错误: {e}")
        # 返回部分结果而不是直接报错
        if results:
            return results
        return [{"title": "搜索失败", "url": "", "description": f"Playwright错误: {str(e)}"}]
    except Exception as e:
        logger.error(f"搜索过程发生错误: {e}")
        return [{"title": "搜索失败", "url": "", "description": f"错误: {str(e)}"}]
    finally:
        if context:
            try:
                await context.close()
            except Exception:
                pass

    return results

# ============================================================================
# Fetch 工具 (支持 CF Worker)
# ============================================================================
@mcp.tool()
async def fetch_html(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        def _fetch():
            scraper = _get_scraper()
            proxies = _get_proxies()
            
            # 使用 Worker 包装 URL
            target_url = _get_target_url(url)
            
            # 如果用了 Worker，headers 中的 Host 可能需要调整，通常 cloudscraper 会自动处理
            # 但我们要确保传给 Worker 的请求头是合理的。
            # 这里简单起见，直接请求 Worker，Headers 由 cloudscraper 默认处理
            
            response = scraper.get(
                target_url, headers=headers or {}, proxies=proxies, timeout=15
            )
            response.raise_for_status()

            html_content, was_truncated = _limit_content_length(response.text)
            return {
                "success": True,
                "url": url,
                "via_worker": bool(CF_WORKER_URL),
                "status_code": response.status_code,
                "html": html_content,
                "truncated": was_truncated,
            }

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _fetch)
        return result
    except Exception as e:
        logger.error(f"抓取失败 {url}: {e}")
        return {"success": False, "url": url, "error": str(e)}


@mcp.tool()
async def fetch_text(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        def _fetch():
            scraper = _get_scraper()
            proxies = _get_proxies()
            target_url = _get_target_url(url)

            response = scraper.get(
                target_url, headers=headers or {}, proxies=proxies, timeout=15
            )
            response.raise_for_status()

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

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _fetch)
        return result
    except Exception as e:
        logger.error(f"抓取文本失败 {url}: {e}")
        return {"success": False, "url": url, "error": str(e)}


@mcp.tool()
async def fetch_metadata(url: str, *, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    try:
        def _fetch():
            scraper = _get_scraper()
            proxies = _get_proxies()
            target_url = _get_target_url(url)

            response = scraper.get(
                target_url, headers=headers or {}, proxies=proxies, timeout=15
            )
            response.raise_for_status()

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

        loop = asyncio.get_event_loop()
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

    # 1. AI 搜索（走代理）
    if OPENAI_API_KEY and OPENAI_BASE_URL:
        try:
            def _ai_search():
                client = OpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL,
                )

                prompt = f"""你是一个专业的搜索助手。请针对用户的问题进行深度搜索和分析。

要求：
1. 尽可能多地搜索相关信息，从多个角度和来源获取数据
2. 搜索时使用多种关键词组合，包括中文和英文
3. 优先搜索最新的信息和权威来源
4. 对搜索结果进行整理和总结

输出格式要求：
1. 首先列出所有搜索到的相关链接（格式：[标题](URL)）
2. 然后提供详细的总结分析，可以包含表格、要点总结、结论等

用户问题：{query}

请开始搜索并提供详细回答。"""

                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content

            loop = asyncio.get_event_loop()
            ai_content = await loop.run_in_executor(None, _ai_search)

            # 解析 AI 返回的链接和总结
            ai_links, ai_summary = _parse_markdown_links(ai_content)
            all_links.extend(ai_links)
            logger.info(f"AI 搜索完成，提取到 {len(ai_links)} 个链接")

        except Exception as e:
            logger.error(f"AI 搜索失败: {e}")
            ai_summary = f"AI 搜索失败: {str(e)}"

    # 2. 普通搜索（走 CF Worker）
    try:
        basic_results = await _search_brave_core(query=query, max_results=max_results)
        all_links.extend(basic_results)
        logger.info(f"普通搜索完成，获取到 {len(basic_results)} 个结果")
    except Exception as e:
        logger.error(f"普通搜索失败: {e}")

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
