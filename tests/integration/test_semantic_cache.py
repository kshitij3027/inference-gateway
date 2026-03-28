"""Integration tests for semantic cache in the chat route."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import app
from gateway.models import ChatCompletionResponse, ChatMessageResponse, Choice, Usage


def _make_response(model="tinyllama", content="Paris is the capital"):
    return ChatCompletionResponse(
        model=model,
        choices=[Choice(message=ChatMessageResponse(content=content))],
        usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
    )


@pytest.fixture
def test_env(monkeypatch):
    monkeypatch.setenv("TENANT_ALPHA_KEY", "test-alpha-key")
    monkeypatch.setenv("TENANT_BETA_KEY", "test-beta-key")
    monkeypatch.setenv("CONFIG_PATH", "config/backends.yaml")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")


@pytest.fixture
def mock_translator():
    """Mock the ollama translator to return a fixed response."""
    response = _make_response()

    async def fake_chat_completion(client, backend, request):
        return response

    with patch.dict(
        "gateway.routes.chat.TRANSLATORS",
        {"ollama": fake_chat_completion, "openai": fake_chat_completion, "anthropic": fake_chat_completion},
    ):
        yield response


class TestCacheIntegration:
    async def test_first_request_cache_miss(self, test_env, mock_translator):
        """First request should get X-Cache: MISS header."""
        async with app.router.lifespan_context(app):
            # Replace semantic_cache with a mock that returns miss
            mock_cache = AsyncMock()
            mock_cache.lookup = AsyncMock(return_value=(None, None))
            mock_cache.record_miss = AsyncMock()
            mock_cache.store = AsyncMock()
            app.state.semantic_cache = mock_cache

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "What is the capital of France?"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 200
            assert resp.headers.get("X-Cache") == "MISS"
            mock_cache.record_miss.assert_called_once()
            mock_cache.store.assert_called_once()

    async def test_second_request_cache_hit(self, test_env, mock_translator):
        """When cache returns a hit, X-Cache: HIT and X-Cache-Similarity headers should be set."""
        cached = _make_response(content="cached: Paris")
        async with app.router.lifespan_context(app):
            mock_cache = AsyncMock()
            mock_cache.lookup = AsyncMock(return_value=(cached, 0.9812))
            mock_cache.record_hit = AsyncMock()
            app.state.semantic_cache = mock_cache

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "What is the capital of France?"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 200
            assert resp.headers.get("X-Cache") == "HIT"
            assert resp.headers.get("X-Cache-Similarity") == "0.9812"
            body = resp.json()
            assert body["choices"][0]["message"]["content"] == "cached: Paris"
            mock_cache.record_hit.assert_called_once()

    async def test_cache_similarity_header_present(self, test_env, mock_translator):
        """X-Cache-Similarity should be present on HIT responses."""
        cached = _make_response()
        async with app.router.lifespan_context(app):
            mock_cache = AsyncMock()
            mock_cache.lookup = AsyncMock(return_value=(cached, 0.9567))
            mock_cache.record_hit = AsyncMock()
            app.state.semantic_cache = mock_cache

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert "X-Cache-Similarity" in resp.headers
            assert float(resp.headers["X-Cache-Similarity"]) == pytest.approx(0.9567)

    async def test_graceful_degradation_no_cache(self, test_env, mock_translator):
        """When semantic_cache is None, requests should proceed normally."""
        async with app.router.lifespan_context(app):
            app.state.semantic_cache = None

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 200
            assert "X-Cache" not in resp.headers

    async def test_cache_exception_graceful_degradation(self, test_env, mock_translator):
        """If cache lookup throws, request should proceed normally with MISS."""
        async with app.router.lifespan_context(app):
            mock_cache = AsyncMock()
            mock_cache.lookup = AsyncMock(side_effect=Exception("Redis down"))
            mock_cache.store = AsyncMock()
            app.state.semantic_cache = mock_cache

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 200
            assert resp.headers.get("X-Cache") == "MISS"

    async def test_streaming_cache_hit_returns_sse(self, test_env, mock_translator):
        """Streaming cache hit should return SSE-formatted chunks."""
        cached = _make_response(content="cached streamed response")
        async with app.router.lifespan_context(app):
            mock_cache = AsyncMock()
            mock_cache.lookup = AsyncMock(return_value=(cached, 0.9900))
            mock_cache.record_hit = AsyncMock()
            app.state.semantic_cache = mock_cache

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={
                        "model": "tinyllama",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 200
            assert resp.headers.get("content-type") == "text/event-stream; charset=utf-8"
            assert resp.headers.get("X-Cache") == "HIT"
            # Verify SSE format
            text = resp.text
            assert "data: " in text
            assert "data: [DONE]" in text
            assert "cached streamed response" in text

    async def test_streaming_cache_miss_wraps_with_tee(self, test_env):
        """Streaming cache miss should wrap generator with tee for caching."""
        async def fake_stream(client, backend, request):
            yield 'data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"tinyllama","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"tinyllama","choices":[{"index":0,"delta":{"content":"hello world"},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"x","object":"chat.completion.chunk","created":1,"model":"tinyllama","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        with patch.dict(
            "gateway.routes.chat.STREAM_TRANSLATORS",
            {"ollama": fake_stream, "openai": fake_stream, "anthropic": fake_stream},
        ):
            async with app.router.lifespan_context(app):
                mock_cache = AsyncMock()
                mock_cache.lookup = AsyncMock(return_value=(None, None))
                mock_cache.record_miss = AsyncMock()
                mock_cache.store = AsyncMock()
                app.state.semantic_cache = mock_cache

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post(
                        "/v1/chat/completions",
                        json={
                            "model": "tinyllama",
                            "messages": [{"role": "user", "content": "hello"}],
                            "stream": True,
                        },
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )
                assert resp.status_code == 200
                assert resp.headers.get("X-Cache") == "MISS"
                text = resp.text
                assert "hello world" in text
                assert "data: [DONE]" in text
                # The tee should have called store
                mock_cache.store.assert_called_once()
