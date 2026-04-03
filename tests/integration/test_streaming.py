import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import app
from gateway.models import (
    ChatCompletionResponse,
    ChatMessageResponse,
    Choice,
    Usage,
)


@pytest.fixture
async def client(test_env):
    """Client fixture that triggers app lifespan (loads config, creates registry)."""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


async def _mock_stream_generator(**kwargs):
    """Mock async generator that yields OpenAI SSE format."""
    yield 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"tinyllama","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
    yield 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"tinyllama","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
    yield 'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1700000000,"model":"tinyllama","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    yield "data: [DONE]\n\n"


class TestStreamingDispatch:
    async def test_stream_returns_sse_content_type(self, client):
        with patch.dict(
            "gateway.routes.chat.STREAM_TRANSLATORS",
            {"ollama": lambda **kw: _mock_stream_generator(**kw)},
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "tinyllama",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    async def test_stream_body_contains_sse_chunks(self, client):
        with patch.dict(
            "gateway.routes.chat.STREAM_TRANSLATORS",
            {"ollama": lambda **kw: _mock_stream_generator(**kw)},
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "tinyllama",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        body = resp.text
        # Body should contain SSE data lines
        assert "data: " in body
        assert "chat.completion.chunk" in body
        assert "data: [DONE]" in body

    async def test_stream_ends_with_done(self, client):
        with patch.dict(
            "gateway.routes.chat.STREAM_TRANSLATORS",
            {"ollama": lambda **kw: _mock_stream_generator(**kw)},
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "tinyllama",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        body = resp.text
        # Last non-empty line should be data: [DONE]
        lines = [l for l in body.strip().split("\n") if l.strip()]
        assert lines[-1].strip() == "data: [DONE]"

    async def test_non_streaming_still_works(self, client):
        """Non-streaming requests should still return JSON, not SSE."""
        mock_response = ChatCompletionResponse(
            model="tinyllama",
            choices=[Choice(message=ChatMessageResponse(content="Hi!"))],
            usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )
        mock_chat = AsyncMock(return_value=mock_response)
        with patch.dict(
            "gateway.routes.chat.TRANSLATORS",
            {"ollama": mock_chat},
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "tinyllama",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                },
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert "text/event-stream" not in resp.headers.get("content-type", "")


class TestStreamingAnalytics:
    """Tests for TTFT/ITL/generation duration streaming analytics."""

    async def test_stream_contains_ttft_comment(self, client):
        """Streaming response should contain SSE comment with TTFT."""
        with patch.dict(
            "gateway.routes.chat.STREAM_TRANSLATORS",
            {"ollama": lambda **kw: _mock_stream_generator(**kw)},
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "tinyllama",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 200
        body = resp.text
        # Should contain an SSE comment with ttft_ms
        lines = body.split("\n")
        ttft_lines = [l for l in lines if l.startswith(": ttft_ms=")]
        assert len(ttft_lines) == 1
        # Value should be a valid non-negative float
        ttft_val = float(ttft_lines[0].split("=")[1])
        assert ttft_val >= 0

    async def test_ttft_comment_is_valid_sse(self, client):
        """The TTFT SSE comment should be a valid SSE comment line (starts with :)."""
        with patch.dict(
            "gateway.routes.chat.STREAM_TRANSLATORS",
            {"ollama": lambda **kw: _mock_stream_generator(**kw)},
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "tinyllama",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        body = resp.text
        # All original data lines should still be present
        data_lines = [l for l in body.split("\n") if l.startswith("data: ")]
        assert len(data_lines) >= 3  # role + content + [DONE]
        # Comment lines start with : (valid SSE)
        comment_lines = [l for l in body.split("\n") if l.startswith(":")]
        assert len(comment_lines) >= 1
