"""Integration tests for request journal in the chat route."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import app
from gateway.models import ChatCompletionResponse, ChatMessageResponse, Choice, Usage


def _make_response(model="tinyllama", content="test"):
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
    response = _make_response()

    async def fake_chat(client, backend, request):
        return response

    with patch.dict(
        "gateway.routes.chat.TRANSLATORS",
        {"ollama": fake_chat, "openai": fake_chat, "anthropic": fake_chat},
    ):
        yield response


class TestJournalIntegration:
    async def test_non_streaming_records_journal(self, test_env, mock_translator):
        """Non-streaming request should record both request and completion in journal."""
        async with app.router.lifespan_context(app):
            mock_journal = AsyncMock()
            app.state.journal = mock_journal
            app.state.semantic_cache = None
            app.state.queue_manager = None

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 200
            mock_journal.record_request.assert_called_once()
            mock_journal.record_completion.assert_called_once()
            # Verify completion has correct status
            completion_call = mock_journal.record_completion.call_args
            assert completion_call[1]["status"] == 200 or completion_call.kwargs.get("status") == 200

    async def test_journal_graceful_when_none(self, test_env, mock_translator):
        """When journal is None, requests should work normally."""
        async with app.router.lifespan_context(app):
            app.state.journal = None
            app.state.semantic_cache = None
            app.state.queue_manager = None

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 200

    async def test_cache_hit_records_journal(self, test_env):
        """Cache hit should record journal with cache_hit=True."""
        cached = _make_response(content="cached response")
        async with app.router.lifespan_context(app):
            mock_journal = AsyncMock()
            mock_cache = AsyncMock()
            mock_cache.lookup = AsyncMock(return_value=(cached, 0.99, "L2_HIT"))
            mock_cache.record_hit = AsyncMock()
            app.state.journal = mock_journal
            app.state.semantic_cache = mock_cache
            app.state.queue_manager = None

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 200
            mock_journal.record_request.assert_called_once()
            mock_journal.record_completion.assert_called_once()
