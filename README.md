# WebSearchMCP

本仓库包含三个版本：

- `normal/`: 轻量版，仅 curl_cffi 抓取，适合常规站点
- `pro/`: 增强版，遇到 Cloudflare 挑战自动回退 Playwright + stealth
- `aionly/`: AI Only 版，`web_search` 仅走 AI 深度搜索（不包含 Brave 搜索），保留网页抓取工具

使用方法请查看对应目录的说明：

- `normal/README.md`
- `pro/README.md`
- `aionly/README.md`
