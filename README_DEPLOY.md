# WebSearchMCP 部署指南

本文只讲部署，不讲功能细节。目标是让你在一台机器上稳定运行：

- Guide MCP 服务（`web-search`）
- 仓库内 CLI（`web-search-cli`）

---

## 1. 前置条件

- Python 3.10+
- `uv`
- `git`
- 可用的上游 API 凭证（至少 `GROK_API_URL`、`GROK_API_KEY`）

---

## 2. 拉取与安装

```bash
git clone <your-repo-url>
cd WebSearchMCP
uv sync --dev
```

---

## 3. 配置环境变量

复制模板并填写：

```bash
cp .env.example .env
```

最小必填：

```env
GROK_API_URL=https://your-api-endpoint.com/v1
GROK_API_KEY=your-grok-api-key
```

> 当前版本搜索模型固定为 `grok-4.20-0309-reasoning`，`GROK_MODEL` 不作为运行时切换入口。

---

## 4. 本地验证部署

```bash
uv run web-search-cli doctor
```

如果返回中 `grok_api_url_configured` 与 `grok_api_key_configured` 为 `true`，说明配置已生效。

再做一次连通性验证：

```bash
uv run web-search-cli search "Find the official Python 3.13 What's New page on docs.python.org. Return webSearchResults only."
```

---

## 5. 启动 MCP 服务

前台启动（开发/调试）：

```bash
uv run web-search
```

MCP 工具仅包含：

- `get_usage_guide`
- `recommend_command`
- `doctor`

---

## 6. 在 Codex 中注册 MCP（推荐）

```bash
codex mcp add websearch -- uv run --directory /absolute/path/to/WebSearchMCP web-search
codex mcp list
codex mcp get websearch
```

验证应看到 `enabled: true`。

---

## 7. 生产化建议（最小可用）

如果你需要常驻进程，建议用进程管理器（systemd/supervisord/pm2）托管：

- 工作目录固定为仓库根目录
- 环境变量固定加载仓库 `.env`
- 异常自动重启
- 日志落盘

---

## 8. 常见问题

### 1) `Failed to spawn: web-search-cli`

使用带项目路径的命令：

```bash
uv run --project /absolute/path/to/WebSearchMCP web-search-cli doctor
```

### 2) `.env` 未生效

先看：

```bash
uv run --project /absolute/path/to/WebSearchMCP web-search-cli doctor
```

检查返回中的 `env_files_loaded` 是否包含仓库 `.env`。

### 3) 返回有 summary 但不稳定

`searchwithsummary` 仅供草稿参考，最终结论应基于 `webSearchResults` + `fetch` 自行核验。

