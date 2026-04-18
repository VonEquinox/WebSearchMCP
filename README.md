# WebSearchMCP

<div align="center">

简体中文 | [English](./docs/README_EN.md)

**引导型 MCP + 仓库内 CLI：让模型先学会如何在 bash 中调用本项目搜索**

</div>

---

## 项目定位

本仓库不再把搜索能力直接暴露为 MCP 搜索工具，而是拆成两层：

1. **Guide MCP**：向模型说明“何时该用哪个命令”，并返回准确的 `uv run web-search-cli ...` 命令。
2. **Repository-local CLI**：真正执行搜索、抓取、站点映射、GitHub API 查询和仓库 grep。

这样做的目标是：

- 不再依赖 `~/.codex/skills/web-search-bash`
- 不绑定任何本机绝对路径
- 只要 `git clone` + 配置 `.env`，模型就能在仓库根目录自动 `uv run` 调用搜索能力

---

## 架构

```text
Model / Codex
  ├─ 调用 Guide MCP: get_usage_guide / recommend_command / doctor
  └─ 在仓库根目录执行 bash 命令
       └─ uv run web-search-cli <subcommand>
            ├─ Grok (/chat/completions)
            ├─ Tavily (/search /extract /map)
            ├─ Firecrawl (/search)
            └─ GitHub REST API / shallow clone grep
```

---

## 安装与准备

### 前置条件

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- `git`

### 克隆仓库

```bash
git clone <your-fork-or-this-repo-url>
cd WebSearchMCP
uv sync --dev
```

### 配置 `.env`

可以直接复制模板：

```bash
cp .env.example .env
```

最少需要：

```env
GROK_API_URL=https://your-api-endpoint.com/v1
GROK_API_KEY=your-grok-api-key
GROK_MODEL=grok-4.20-0309-reasoning
```

可选：

```env
TAVILY_API_URL=https://api.tavily.com
TAVILY_API_KEYS=["tvly-key-1","tvly-key-2"]
FIRECRAWL_API_URL=https://api.firecrawl.dev/v2
FIRECRAWL_API_KEY=your-firecrawl-api-key
```

配置读取顺序：

1. 显式环境变量
2. 仓库根目录 `.env`
3. `~/.config/web-search/.env`
4. `WEB_SEARCH_ENV_FILE` / `GROK_SEARCH_ENV_FILE` 指定文件

---

## 启动 Guide MCP

在仓库根目录运行：

```bash
uv run web-search
```

这个 MCP 只提供三个工具：

- `get_usage_guide`
- `recommend_command`
- `doctor`

它们**不会直接联网搜索**，而是告诉模型接下来应该执行哪个 `uv run web-search-cli ...` 命令；其中 `search` / `searchwithsummary` 都要求**调用方自己写完整 prompt**。

---

## CLI 命令

所有命令都应在**仓库根目录**执行。

### 搜索与抓取

```bash
uv run web-search-cli doctor
uv run web-search-cli models
uv run web-search-cli search "Search the web for the latest uv release. Return webSearchResults and explicit [browse_page]{...} blocks."
uv run web-search-cli searchwithsummary "Search the web for the latest uv release. Return a short markdown summary plus webSearchResults and explicit [browse_page]{...} blocks."
uv run web-search-cli fetch "https://docs.astral.sh/uv/"
uv run web-search-cli map --instructions "only documentation pages" "https://docs.astral.sh/uv/"
```

- `search`：只保留结构化结果，不返回 summary
- `searchwithsummary`：额外返回草稿 summary，但 **summary 不一定准确，最好自己验证**

### GitHub 辅助能力

```bash
uv run web-search-cli github-api /repos/openai/openai-python
uv run web-search-cli repo-file openai/openai-python README.md
uv run web-search-cli repo-grep openai/openai-python Responses
```

---

## 推荐给模型的工作流

1. 先调用 `doctor` 检查配置
2. 再调用 `recommend_command` 获取推荐命令
3. 如果返回的是 `uv run web-search-cli search "<FULL_PROMPT>"` 或 `searchwithsummary "<FULL_PROMPT>"`，由模型自己把 `<FULL_PROMPT>` 替换成完整搜索 prompt
4. 模型在仓库根目录执行最终命令
5. 如有需要，再执行 fallback command

---

## 破坏性变更说明

当前版本**不再提供旧的直接搜索型 MCP 工具**，例如：

- `web_search`
- `web_fetch`
- `web_map`
- `plan_*`
- `toggle_builtin_tools`

如果你的客户端之前直接依赖这些 MCP 工具，需要迁移到：

- Guide MCP：负责告诉模型怎么用
- CLI：负责真正执行搜索

---

## 从旧 `web-search-bash` skill 迁移

| 旧 skill 能力 | 新仓库命令 |
|---|---|
| `web-search.sh` | `uv run web-search-cli search "<FULL_PROMPT>"` |
| `discover-sources.sh` | `uv run web-search-cli search "<DISCOVERY_PROMPT>"` |
| `web-fetch.sh` | `uv run web-search-cli fetch ...` |
| `web-map.sh` | `uv run web-search-cli map ...` |
| `models.sh` | `uv run web-search-cli models` |
| `github-api.sh` | `uv run web-search-cli github-api ...` |
| `repo-file.sh` | `uv run web-search-cli repo-file ...` |
| `repo-grep.sh` | `uv run web-search-cli repo-grep ...` |

---

## 开发与测试

```bash
uv run pytest -q
```

如果你在改 CLI 或 Guide MCP，建议至少验证：

```bash
uv run web-search-cli doctor
uv run web-search-cli search "Search the web for the latest uv release. Return webSearchResults and explicit [browse_page]{...} blocks."
uv run web-search-cli searchwithsummary "Search the web for the latest uv release. Return a short markdown summary plus webSearchResults and explicit [browse_page]{...} blocks."
```
