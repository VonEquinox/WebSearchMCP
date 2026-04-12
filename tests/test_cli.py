from web_search import cli


def test_cli_search_joins_multiword_query_and_prints_json(monkeypatch, capsys):
    captured = {}

    async def fake_search(query, **kwargs):
        captured["query"] = query
        return {
            "status": "ok",
            "content": "done",
            "webSearchResults": [],
            "browse_page": [],
        }

    monkeypatch.setattr(cli, "search_web", fake_search)

    exit_code = cli.main(["search", "--json", "hello", "world"])

    assert exit_code == 0
    assert captured["query"] == "hello world"
    assert '"content": "done"' in capsys.readouterr().out


def test_cli_search_plain_output_includes_browse_page_and_results(monkeypatch, capsys):
    async def fake_search(query, **kwargs):
        return {
            "status": "ok",
            "webSearchResults": [
                {"url": "https://example.com/a", "title": "Alpha", "preview": "alpha preview"}
            ],
            "browse_page": [
                {
                    "url": "https://example.com/b",
                    "title": "Beta",
                    "preview": "beta preview",
                    "note": "高价值网页，建议优先打开或抓取正文。",
                }
            ],
        }

    monkeypatch.setattr(cli, "search_web", fake_search)

    exit_code = cli.main(["search", "hello"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "browse_page（高价值网页" in output
    assert "webSearchResults" in output


def test_cli_searchwithsummary_plain_output_includes_warning_and_summary(monkeypatch, capsys):
    async def fake_search(query, **kwargs):
        return {
            "status": "ok",
            "content": "summary",
            "summary_warning": "summary 不一定准确，最好自己验证。",
            "webSearchResults": [],
            "browse_page": [],
        }

    monkeypatch.setattr(cli, "search_web_with_summary", fake_search)

    exit_code = cli.main(["searchwithsummary", "hello"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "summary 不一定准确" in output
    assert "summary" in output


def test_cli_repo_file_prints_file_content_by_default(monkeypatch, capsys):
    async def fake_repo_file(repo, path, ref=""):
        return {"kind": "file", "repo": repo, "path": path, "content": "README body"}

    monkeypatch.setattr(cli, "fetch_repo_file", fake_repo_file)

    exit_code = cli.main(["repo-file", "owner/repo", "README.md"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "README body"


def test_cli_doctor_prints_json(monkeypatch, capsys):
    async def fake_doctor():
        return {"working_directory": {"is_repo_root": True}}

    monkeypatch.setattr(cli, "get_doctor_info", fake_doctor)

    exit_code = cli.main(["doctor"])

    assert exit_code == 0
    assert '"is_repo_root": true' in capsys.readouterr().out
