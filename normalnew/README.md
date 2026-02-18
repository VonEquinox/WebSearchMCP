#  WebSearch MCP Server (normalnew)

一个基于 MCP (Model Context Protocol) 的智能网页搜索服务器，支持 AI 增强搜索、Brave 搜索和网页抓取功能。

本目录为 `normalnew` 版本：基于 `pro` 版本代码，禁用 Playwright 回退，仅使用 `curl_cffi` 抓取。

> **推荐**: 本项目推荐使用 `SOTASearch.py`，它结合了 AI 深度搜索和 Brave 搜索，**并行执行**两种搜索以提升速度。适合与 [CherryStudio](https://github.com/kangfenmao/cherry-studio) 配合使用。

## 功能

| 工具 | 说明 |
|------|------|
| `web_search` | AI 深度搜索 + Brave 搜索（并行执行），返回链接列表和 AI 总结 |
| `fetch` | 抓取网页并提取正文 Markdown（尽量去掉导航/按钮/广告等噪声） |

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/WebSearchMCP.git
cd WebSearchMCP
```

### 2. 安装依赖

**使用 uv（推荐）：**

```bash
uv sync
```

**使用 pip：**

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env` 并填入配置：

```bash
cp .env.example .env
```

主要配置项：

```env
# 代理配置（可选）
PROXY=http://127.0.0.1:7890
CF_WORKER=https://your-worker.workers.dev

# OpenAI API 配置（SOTASearch 必需）
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

## 快速开始

```bash
# 使用 uv（推荐）
uv run SOTASearch.py

# 或使用 python
python SOTASearch.py
```

## 配置 CherryStudio

在 CherryStudio 的 MCP 服务器设置中添加（必须使用虚拟环境中的 Python 绝对路径）：

**Windows 示例：**

```json
{
  "mcpServers": {
    "sota-search": {
      "name": "Websearch",
      "type": "stdio",
      "command": "D:\\Code\\github\\WebSearchMCP\\.venv\\Scripts\\python.exe",
      "args": ["D:/Code/github/WebSearchMCP/SOTASearch.py"]
    }
  }
}
```

**macOS/Linux 示例：**

```json
{
  "mcpServers": {
    "sota-search": {
      "name": "Websearch",
      "type": "stdio",
      "command": "/path/to/WebSearchMCP/.venv/bin/python",
      "args": ["/path/to/WebSearchMCP/SOTASearch.py"]
    }
  }
}
```

> **注意**: 推荐使用项目虚拟环境中的 Python 解释器（`.venv/Scripts/python.exe` 或 `.venv/bin/python`），确保依赖正确加载。系统 Python 也可以使用，但需确保已安装所有依赖。

> **提示**: 代理和 API 配置建议写在 `.env` 文件中，无需在命令行参数中指定。

## 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--proxy` | 本地代理地址 | `--proxy http://127.0.0.1:7890` |
| `--cf-worker` | Cloudflare Worker 地址 | `--cf-worker https://xxx.workers.dev` |
| `--openai-api-key` | OpenAI API Key | `--openai-api-key sk-xxx` |
| `--openai-base-url` | OpenAI API Base URL | `--openai-base-url https://api.openai.com/v1` |
| `--openai-model` | 模型名称（默认 gpt-4o） | `--openai-model gpt-4o` |

## 工具说明

### web_search

AI 深度搜索 + Brave 搜索（并行执行）。

参数：
- `query`: 搜索关键词（必填）
  - 固定最多返回 15 条链接（不支持参数控制数量）
  - 普通搜索：AI 链接优先占据 15 个名额，不足再用浏览器结果补齐（默认同域名最多 2 条，可通过环境变量 `SEARCH_MAX_PER_DOMAIN` 调整）
  - `site:` 搜索：浏览器结果优先占据 15 个名额，不足再用 AI 补齐（并关闭同域名上限）

返回格式：

```json
{
  "success": true,
  "query": "搜索关键词",
  "links": [
    {"title": "标题", "url": "链接", "description": "描述"}
  ],
  "ai_summary": "AI 总结内容"
}
```

### fetch

抓取网页并提取正文内容（Markdown），尽量去掉导航/按钮/广告等噪声。

参数：
- `url`: 目标网址（必填）
- `headers`: 可选的请求头

## 回归测试（集成 / Live）

项目提供了一个集成回归脚本用于反复调试抓取与搜索质量：

```bash
python regression_suite.py --fetch
python regression_suite.py --search
python regression_suite.py --fetch --also-text
```

用例文件位于 `cases/` 目录（可按需增删 URL/Query）。产物默认落盘到 `artifacts/` 目录，便于肉眼对比与迭代调参。

## 特性

- AI 深度搜索 + Brave 搜索**并行执行**
- 支持本地代理 (`--proxy`)
- 支持 Cloudflare Worker 代理 (`--cf-worker`)
- 自动内容截断，防止响应过大
- 支持 `.env` 文件配置

## 依赖

- Python 3.10+
- fastmcp
- beautifulsoup4
- curl_cffi
- lxml
- openai
