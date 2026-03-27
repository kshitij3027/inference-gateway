import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.backends.ollama import stream_chat_completion
from gateway.config import BackendConfig
from gateway.models import ChatCompletionRequest, ChatMessage


async def _async_iter(lines):
    for line in lines:
        yield line


def _make_mock_stream_client(lines):
    """Create a mock httpx client that streams given lines."""
    mock_response = MagicMock()
    mock_response.aiter_lines = lambda: _async_iter(lines)
    mock_response.raise_for_status = MagicMock()

    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)
    return mock_client


def _make_backend():
    return BackendConfig(
        name="ollama-local",
        provider="ollama",
        base_url="http://ollama:11434",
        models=["tinyllama"],
    )


def _make_request():
    return ChatCompletionRequest(
        model="tinyllama",
        messages=[ChatMessage(role="user", content="Hello")],
        stream=True,
    )


async def _collect_stream(gen):
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)
    return chunks


OLLAMA_NDJSON_LINES = [
    '{"model":"tinyllama","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":"Hello"},"done":false}',
    '{"model":"tinyllama","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":" world"},"done":false}',
    '{"model":"tinyllama","created_at":"2024-01-01T00:00:00Z","message":{"role":"assistant","content":""},"done":true,"prompt_eval_count":10,"eval_count":5}',
]


class TestOllamaStreaming:
    async def test_happy_path(self):
        client = _make_mock_stream_client(OLLAMA_NDJSON_LINES)
        backend = _make_backend()
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))

        # First chunk: role announcement
        assert chunks[0].startswith("data: ")
        first = json.loads(chunks[0][6:].strip())
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"]["role"] == "assistant"
        assert first["choices"][0]["finish_reason"] is None

        # Content chunks
        second = json.loads(chunks[1][6:].strip())
        assert second["choices"][0]["delta"]["content"] == "Hello"

        third = json.loads(chunks[2][6:].strip())
        assert third["choices"][0]["delta"]["content"] == " world"

        # Final chunk with finish_reason
        fourth = json.loads(chunks[3][6:].strip())
        assert fourth["choices"][0]["delta"] == {"role": None, "content": None}
        assert fourth["choices"][0]["finish_reason"] == "stop"

        # [DONE] terminator
        assert chunks[-1] == "data: [DONE]\n\n"

    async def test_all_chunks_share_same_id(self):
        client = _make_mock_stream_client(OLLAMA_NDJSON_LINES)
        backend = _make_backend()
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))

        # Parse all non-[DONE] chunks
        ids = set()
        for chunk in chunks:
            if chunk.startswith("data: {"):
                data = json.loads(chunk[6:].strip())
                ids.add(data["id"])

        assert len(ids) == 1
        assert list(ids)[0].startswith("chatcmpl-")

    async def test_empty_content_lines_skipped(self):
        lines = [
            '{"model":"tinyllama","message":{"role":"assistant","content":""},"done":false}',
            '{"model":"tinyllama","message":{"role":"assistant","content":"Hi"},"done":false}',
            '{"model":"tinyllama","message":{"role":"assistant","content":""},"done":true}',
        ]
        client = _make_mock_stream_client(lines)
        backend = _make_backend()
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))

        # Should have: role chunk, "Hi" content chunk, final chunk, [DONE]
        content_chunks = [c for c in chunks if '"content"' in c and '"Hi"' in c]
        assert len(content_chunks) == 1

    async def test_done_terminator(self):
        client = _make_mock_stream_client(OLLAMA_NDJSON_LINES)
        backend = _make_backend()
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))
        assert chunks[-1] == "data: [DONE]\n\n"

    async def test_sse_format(self):
        client = _make_mock_stream_client(OLLAMA_NDJSON_LINES)
        backend = _make_backend()
        req = _make_request()

        chunks = await _collect_stream(stream_chat_completion(client, backend, req))

        for chunk in chunks:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")
