# Repository Guidelines

## Project Structure & Module Organization
The Python package lives in `src/web_search/`. Keep MCP entrypoints and tool wiring in `server.py`, shared configuration in `config.py`, planning/session logic in `planning.py`, source handling in `sources.py`, and provider integrations under `src/web_search/providers/` (`grok.py`, `tavily.py`). Tests live in `tests/` and mirror behavior-focused areas such as search flow and prompt construction. User-facing docs are in `README.md` and `docs/README_EN.md`; screenshots and static images belong in `images/`. Use `.env.example` as the template for local configuration.

## Build, Test, and Development Commands
- `uv sync --dev` — install runtime and test dependencies into the local environment.
- `uv run pytest -q` — run the full test suite.
- `uv run web-search` — start the MCP server over stdio using the packaged entrypoint.
- `uv run python -m web_search.server` — alternate way to launch the server while debugging imports.

Set `GROK_API_URL` and `GROK_API_KEY` before running locally; optional Tavily and Firecrawl settings can be loaded from a root `.env` file.

## Coding Style & Naming Conventions
Target Python 3.10+ and follow existing PEP 8-style conventions: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for module-level constants. Prefer explicit type hints and small, single-purpose helpers. Keep provider-specific HTTP behavior inside `providers/` instead of `server.py`. No formatter or linter config is committed, so keep diffs minimal and avoid unrelated reformatting.

## Testing Guidelines
Use `pytest` with `pytest-asyncio` for async tool behavior. Name files `test_*.py` and test functions `test_*`. Add regression tests for new planning rules, provider fallback logic, and configuration edge cases. Run `uv run pytest -q` before opening a PR; there is no published coverage threshold, but new behavior should ship with tests.

## Commit & Pull Request Guidelines
Match the repository’s Conventional Commit style seen in history, for example `fix(search): prevent planning mismatch` or `docs(readme): clarify env loading`. PRs should include: a short summary, linked issue or rationale, config/env impact, and the exact test command/output you ran. Only include screenshots when updating README assets or other user-facing documentation.

## Security & Configuration Tips
Never commit real API keys or populated `.env` files. Prefer `TAVILY_API_KEYS` as a JSON array for multi-key setups, and document any new environment variables in both `.env.example` and the README.
