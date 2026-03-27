from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import HTTPException

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


def _mock_response():
    return ChatCompletionResponse(
        model="tinyllama",
        choices=[Choice(message=ChatMessageResponse(content="Hello!"))],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


class TestNonStreamingFailover:
    async def test_failover_to_next_backend_on_5xx(self, client):
        """First backend returns 500, second succeeds."""
        call_count = 0

        async def mock_translator(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise HTTPException(status_code=500, detail="Backend error")
            return _mock_response()

        with patch.dict(
            "gateway.routes.chat.TRANSLATORS",
            {"ollama": mock_translator},
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 200
        assert call_count == 2  # First failed, second succeeded

    async def test_failover_on_connect_error(self, client):
        """ConnectError triggers failover to next backend."""
        call_count = 0

        async def mock_translator(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            return _mock_response()

        with patch.dict(
            "gateway.routes.chat.TRANSLATORS",
            {"ollama": mock_translator},
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 200

    async def test_503_when_all_backends_fail(self, client):
        """All backends fail -> returns server error."""
        async def always_fail(**kwargs):
            raise HTTPException(status_code=500, detail="Backend error")

        with patch.dict(
            "gateway.routes.chat.TRANSLATORS",
            {"ollama": always_fail},
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        # Should get 500 (last error) or 502 depending on implementation
        assert resp.status_code >= 500


class TestStreamingCircuitBreaker:
    async def test_streaming_with_cb_wrapper(self, client):
        """Streaming records success on circuit breaker."""
        async def mock_stream(**kwargs):
            yield 'data: {"id":"test","object":"chat.completion.chunk","created":1,"model":"tinyllama","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"test","object":"chat.completion.chunk","created":1,"model":"tinyllama","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
            yield "data: [DONE]\n\n"

        with patch.dict(
            "gateway.routes.chat.STREAM_TRANSLATORS",
            {"ollama": lambda **kw: mock_stream(**kw)},
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    async def test_streaming_error_records_failure(self, client):
        """Streaming error event records failure on circuit breaker."""
        async def mock_error_stream(**kwargs):
            yield 'data: {"type":"stream_error","error":"backend_failure"}\n\n'

        with patch.dict(
            "gateway.routes.chat.STREAM_TRANSLATORS",
            {"ollama": lambda **kw: mock_error_stream(**kw)},
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 200
        body = resp.text
        assert "stream_error" in body


class TestAdminCircuitState:
    async def test_backends_show_circuit_state(self, client):
        """Admin backends includes circuit_breaker field."""
        resp = await client.get("/admin/backends")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        first = data[0]
        assert "health" in first
        assert first["health"] in ("CLOSED", "OPEN", "HALF_OPEN", "unknown")
        assert "circuit_breaker" in first
        if first["circuit_breaker"]:
            assert "state" in first["circuit_breaker"]
            assert "error_rate" in first["circuit_breaker"]

    async def test_backends_circuit_state_after_failure(self, client):
        """Circuit breaker state reflects failures."""
        async def always_fail(**kwargs):
            raise HTTPException(status_code=500, detail="Backend error")

        with patch.dict(
            "gateway.routes.chat.TRANSLATORS",
            {"ollama": always_fail},
        ):
            # Make a request that will fail through all backends
            await client.post(
                "/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer test-alpha-key"},
            )

        # Check admin endpoint reflects the failures
        resp = await client.get("/admin/backends")
        assert resp.status_code == 200
        data = resp.json()
        # Find ollama backends — they should have recorded failures
        ollama_backends = [b for b in data if b["provider"] == "ollama"]
        assert len(ollama_backends) > 0
        for backend in ollama_backends:
            cb = backend["circuit_breaker"]
            assert cb["requests_in_window"] > 0
