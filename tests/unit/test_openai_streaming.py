import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.backends.openai import stream_chat_completion
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
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return BackendConfig(
        name="mock-openai",
        provider="openai",
        base_url="http://mock:3000",
        api_key_env="OPENAI_API_KEY",
        models=["mock-gpt-markdown"],
    )


def _make_request():
    return ChatCompletionRequest(
        model="mock-gpt-markdown",
        messages=[ChatMessage(role="user", content="Hello")],
        stream=True,
    )


async def _collect_stream(gen):
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)
    return chunks


OPENAI_SSE_LINES = [
    'data: {"id":"chatcmpl-backend","object":"chat.completion.chunk","created":1700000000,"model":"mock-gpt-markdown","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
    "",
    'data: {"id":"chatcmpl-backend","object":"chat.completion.chunk","created":1700000000,"model":"mock-gpt-markdown","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
    "",
    'data: {"id":"chatcmpl-backend","object":"chat.completion.chunk","created":1700000000,"model":"mock-gpt-markdown","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
    "",
    'data: {"id":"chatcmpl-backend","object":"chat.completion.chunk","created":1700000000,"model":"mock-gpt-markdown","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    "",
    "data: [DONE]",
]


class TestOpenAIStreaming:
    async def test_happy_path(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        client = _make_mock_stream_client(OPENAI_SSE_LINES)
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))

        # First yielded chunk should have role
        first = json.loads(chunks[0][6:].strip())
        assert first["choices"][0]["delta"]["role"] == "assistant"

        # Content chunks
        second = json.loads(chunks[1][6:].strip())
        assert second["choices"][0]["delta"]["content"] == "Hello"

        # Last data chunk has finish_reason
        last_data = json.loads(chunks[-2][6:].strip())
        assert last_data["choices"][0]["finish_reason"] == "stop"

        # [DONE] terminator
        assert chunks[-1] == "data: [DONE]\n\n"

    async def test_id_replaced_with_gateway_id(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        client = _make_mock_stream_client(OPENAI_SSE_LINES)
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))

        ids = set()
        for chunk in chunks:
            if chunk.startswith("data: {"):
                data = json.loads(chunk[6:].strip())
                ids.add(data["id"])

        assert len(ids) == 1
        chunk_id = list(ids)[0]
        assert chunk_id.startswith("chatcmpl-")
        assert chunk_id != "chatcmpl-backend"  # Backend ID replaced

    async def test_empty_lines_skipped(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        client = _make_mock_stream_client(OPENAI_SSE_LINES)
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))

        # No empty chunks — all should start with "data: "
        for chunk in chunks:
            assert chunk.startswith("data: ")

    async def test_done_terminator(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        client = _make_mock_stream_client(OPENAI_SSE_LINES)
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))
        assert chunks[-1] == "data: [DONE]\n\n"

    async def test_sse_format(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        client = _make_mock_stream_client(OPENAI_SSE_LINES)
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))
        for chunk in chunks:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")
