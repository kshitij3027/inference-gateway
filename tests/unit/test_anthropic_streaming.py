import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.backends.anthropic import stream_chat_completion
from gateway.config import BackendConfig
from gateway.models import ChatMessage, ChatCompletionRequest


async def _async_iter(lines):
    for line in lines:
        yield line


def _make_mock_stream_client(lines):
    mock_response = MagicMock()
    mock_response.aiter_lines = lambda: _async_iter(lines)
    mock_response.raise_for_status = MagicMock()

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)
    return mock_client


def _make_backend(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return BackendConfig(
        name="mock-anthropic",
        provider="anthropic",
        base_url="http://mock:3000/anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        models=["mock-claude-markdown"],
    )


def _make_request():
    return ChatCompletionRequest(
        model="mock-claude-markdown",
        messages=[ChatMessage(role="user", content="Hello")],
        stream=True,
    )


async def _collect_stream(gen):
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)
    return chunks


ANTHROPIC_SSE_LINES = [
    'event: message_start',
    'data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","model":"mock-claude-markdown","usage":{"input_tokens":10}}}',
    '',
    'event: content_block_start',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    '',
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
    '',
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
    '',
    'event: content_block_stop',
    'data: {"type":"content_block_stop","index":0}',
    '',
    'event: message_delta',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}',
    '',
    'event: message_stop',
    'data: {"type":"message_stop"}',
]


class TestAnthropicStreaming:
    async def test_happy_path(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        client = _make_mock_stream_client(ANTHROPIC_SSE_LINES)
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))

        # First chunk: role
        first = json.loads(chunks[0][6:].strip())
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"]["role"] == "assistant"

        # Content chunks
        second = json.loads(chunks[1][6:].strip())
        assert second["choices"][0]["delta"]["content"] == "Hello"

        third = json.loads(chunks[2][6:].strip())
        assert third["choices"][0]["delta"]["content"] == " world"

        # Final chunk with finish_reason
        fourth = json.loads(chunks[3][6:].strip())
        assert fourth["choices"][0]["finish_reason"] == "stop"  # end_turn -> stop

        # [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"

    async def test_stop_reason_max_tokens(self, monkeypatch):
        lines = [
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}',
            'event: message_delta',
            'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":100}}',
            'event: message_stop',
            'data: {"type":"message_stop"}',
        ]
        backend = _make_backend(monkeypatch)
        client = _make_mock_stream_client(lines)
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))

        # Find the final chunk (second to last before [DONE])
        final = json.loads(chunks[-2][6:].strip())
        assert final["choices"][0]["finish_reason"] == "length"  # max_tokens -> length

    async def test_non_text_delta_ignored(self, monkeypatch):
        lines = [
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"key\\""}}',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}',
            'event: message_delta',
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}',
            'event: message_stop',
            'data: {"type":"message_stop"}',
        ]
        backend = _make_backend(monkeypatch)
        client = _make_mock_stream_client(lines)
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))

        # Should have: role chunk, "Hi" content chunk, final chunk, [DONE]
        content_chunks = [c for c in chunks if '"content"' in c and '"Hi"' in c]
        assert len(content_chunks) == 1

    async def test_all_chunks_share_same_id(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        client = _make_mock_stream_client(ANTHROPIC_SSE_LINES)
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))

        ids = set()
        for chunk in chunks:
            if chunk.startswith("data: {"):
                data = json.loads(chunk[6:].strip())
                if "id" in data:
                    ids.add(data["id"])

        assert len(ids) == 1
        assert list(ids)[0].startswith("chatcmpl-")

    async def test_event_lines_skipped(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        client = _make_mock_stream_client(ANTHROPIC_SSE_LINES)
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))

        # All chunks should be SSE format
        for chunk in chunks:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")

    async def test_done_terminator(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        client = _make_mock_stream_client(ANTHROPIC_SSE_LINES)
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))
        assert chunks[-1] == "data: [DONE]\n\n"
