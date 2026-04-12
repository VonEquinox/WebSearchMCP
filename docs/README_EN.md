# WebSearchMCP

[简体中文](../README.md) | English

**Guide MCP + repository-local CLI for bash-driven search workflows**

---

## What changed

This repository no longer exposes search itself as MCP tools. It is now split into two layers:

1. **Guide MCP** — tells the model which command to run and how to run it.
2. **Repository-local CLI** — performs the actual search, fetch, site map, GitHub API lookup, and repo grep.

The goal is to make the workflow:

- independent from `~/.codex/skills/web-search-bash`
- free of machine-specific absolute paths
- easy to move to another computer after `git clone` + `.env`

---

## Architecture

```text
Model / Codex
  ├─ calls Guide MCP: get_usage_guide / recommend_command / doctor
  └─ runs bash in the repository root
       └─ uv run web-search-cli <subcommand>
            ├─ Grok (/chat/completions)
            ├─ Tavily (/search /extract /map)
            ├─ Firecrawl (/search /scrape)
            └─ GitHub REST API / shallow clone grep
```

---

## Setup

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- `git`

### Clone and install

```bash
git clone <your-fork-or-this-repo-url>
cd WebSearchMCP
uv sync --dev
```

### Configure `.env`

```bash
cp .env.example .env
```

Minimum config:

```env
GROK_API_URL=https://your-api-endpoint.com/v1
GROK_API_KEY=your-grok-api-key
GROK_MODEL=grok-4.20-0309-reasoning
```

Optional config:

```env
TAVILY_API_URL=https://api.tavily.com
TAVILY_API_KEYS=["tvly-key-1","tvly-key-2"]
FIRECRAWL_API_URL=https://api.firecrawl.dev/v2
FIRECRAWL_API_KEY=your-firecrawl-api-key
```

Config lookup order:

1. explicit environment variables
2. repository-root `.env`
3. `~/.config/web-search/.env`
4. `WEB_SEARCH_ENV_FILE` / `GROK_SEARCH_ENV_FILE`

---

## Start the Guide MCP

Run this from the repository root:

```bash
uv run web-search
```

It exposes only three tools:

- `get_usage_guide`
- `recommend_command`
- `doctor`

These tools **do not search directly**. They tell the model which `uv run web-search-cli ...` command to execute next; both `search` and `searchwithsummary` expect the caller to author the full prompt itself.

---

## CLI commands

Run all commands from the **repository root**.

### Search and fetch

```bash
uv run web-search-cli doctor
uv run web-search-cli models
uv run web-search-cli search "Search the web for the latest uv release. Return webSearchResults and explicit [browse_page]{...} blocks."
uv run web-search-cli searchwithsummary "Search the web for the latest uv release. Return a short markdown summary plus webSearchResults and explicit [browse_page]{...} blocks."
uv run web-search-cli fetch "https://docs.astral.sh/uv/"
uv run web-search-cli map --instructions "only documentation pages" "https://docs.astral.sh/uv/"
```

- `search`: keep structured results only, without a summary
- `searchwithsummary`: also returns a draft summary, but **the summary may be inaccurate and should be verified**

### GitHub helpers

```bash
uv run web-search-cli github-api /repos/openai/openai-python
uv run web-search-cli repo-file openai/openai-python README.md
uv run web-search-cli repo-grep openai/openai-python Responses
```

---

## Recommended model workflow

1. Call `doctor`
2. Call `recommend_command`
3. If the result is `uv run web-search-cli search "<FULL_PROMPT>"` or `searchwithsummary "<FULL_PROMPT>"`, replace `<FULL_PROMPT>` with a fully written search prompt
4. Execute the final command from the repository root
5. Use fallback commands if needed

---

## Breaking change

This version no longer provides the old direct-search MCP tools such as:

- `web_search`
- `web_fetch`
- `web_map`
- `plan_*`
- `toggle_builtin_tools`

If your client used those MCP tools directly, migrate to:

- Guide MCP for instructions
- CLI for execution

---

## Migration from the old `web-search-bash` skill

| Old skill | New repo command |
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

## Test

```bash
uv run pytest -q
```
