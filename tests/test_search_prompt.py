import pytest

from web_search.providers.grok import GrokSearchProvider


@pytest.mark.asyncio
async def test_grok_search_uses_caller_supplied_prompt_as_only_message(monkeypatch):
    captured = {}

    async def fake_execute(self, headers, payload, ctx=None):
        captured['payload'] = payload
        return 'Final answer'

    monkeypatch.setattr(GrokSearchProvider, '_execute_stream_with_retry', fake_execute)

    provider = GrokSearchProvider('https://example.com', 'test-key', 'test-model')
    await provider.search('Search the web for official docs only. Return markdown sources.')

    assert captured['payload']['messages'] == [
        {
            'role': 'user',
            'content': 'Search the web for official docs only. Return markdown sources.',
        }
    ]


@pytest.mark.asyncio
async def test_grok_search_preserves_multiline_raw_prompt(monkeypatch):
    captured = {}

    async def fake_execute(self, headers, payload, ctx=None):
        captured['payload'] = payload
        return 'Final answer'

    monkeypatch.setattr(GrokSearchProvider, '_execute_stream_with_retry', fake_execute)

    prompt = 'Goal: verify latest uv release\nConstraints: use official sources\nOutput: answer + sources'
    provider = GrokSearchProvider('https://example.com', 'test-key', 'test-model')
    await provider.search(prompt)

    assert captured['payload']['messages'][0]['content'] == prompt
