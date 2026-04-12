import json

import pytest

from web_search.config import (
    build_openai_chat_completions_url,
    build_openai_models_url,
    normalize_openai_api_base_url,
)
from web_search.providers.grok import (
    GrokSearchProvider,
    _extract_response_text,
    _sanitize_model_output,
)


class _FakeStreamingResponse:
    def __init__(self, lines: list[str]):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def test_openai_url_helpers_accept_base_or_full_endpoint_urls():
    assert normalize_openai_api_base_url("https://example.com/v1/chat/completions") == "https://example.com/v1"
    assert normalize_openai_api_base_url("https://example.com/v1/models") == "https://example.com/v1"
    assert build_openai_chat_completions_url("https://example.com/v1") == "https://example.com/v1/chat/completions"
    assert build_openai_chat_completions_url("https://example.com/v1/models") == "https://example.com/v1/chat/completions"
    assert build_openai_models_url("https://example.com/v1/chat/completions") == "https://example.com/v1/models"


def test_extract_response_text_supports_wrapped_payloads_and_typed_content():
    payload = {
        "data": {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "text", "text": " world"},
                        ]
                    }
                }
            ]
        }
    }

    assert _extract_response_text(payload) == "hello world"


def test_sanitize_model_output_removes_think_blocks():
    raw = "<think>\nprivate reasoning\n</think>\nfinal answer"

    assert _sanitize_model_output(raw) == "final answer"


@pytest.mark.asyncio
async def test_parse_streaming_response_discards_think_chunks_and_keeps_answer():
    provider = GrokSearchProvider("https://example.com/v1/chat/completions", "test-key", model="grok-4.20-beta")
    lines = [
        'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
        'data: {"choices":[{"delta":{"content":"<think>\\n"}}]}',
        'data: {"choices":[{"delta":{"content":"private reasoning"}}]}',
        'data: {"choices":[{"delta":{"content":"\\n</think>\\n"}}]}',
        'data: {"choices":[{"delta":{"content":"pong"}}]}',
        "data: [DONE]",
    ]

    content = await provider._parse_streaming_response(_FakeStreamingResponse(lines))

    assert content == "pong"


@pytest.mark.asyncio
async def test_parse_streaming_response_supports_single_json_body_without_sse():
    provider = GrokSearchProvider("https://example.com/v1", "test-key", model="grok-4.20-beta")
    body = json.dumps({"data": {"choices": [{"message": {"content": "plain answer"}}]}})

    content = await provider._parse_streaming_response(_FakeStreamingResponse([body]))

    assert content == "plain answer"
