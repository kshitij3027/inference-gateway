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


def _mock_response():
    """Create a mock ChatCompletionResponse for testing."""
    return ChatCompletionResponse(
        model="tinyllama",
        choices=[Choice(message=ChatMessageResponse(content="Hello!"))],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


class TestAuthenticatedChatFlow:
    """Test the full auth flow through the real app."""

    @patch("gateway.routes.chat.ollama.chat_completion", new_callable=AsyncMock)
    async def test_valid_key_returns_200(self, mock_chat, client):
        mock_chat.return_value = _mock_response()
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer test-alpha-key"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["usage"]["total_tokens"] == 8
        mock_chat.assert_called_once()

    async def test_bad_key_returns_401(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer bad-key"},
        )
        assert resp.status_code == 401

    async def test_missing_auth_returns_401(self, client):
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 401

    @patch("gateway.routes.chat.ollama.chat_completion", new_callable=AsyncMock)
    async def test_disallowed_model_returns_403(self, mock_chat, client):
        """tenant-alpha only has tinyllama in allowed_models."""
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer test-alpha-key"},
        )
        assert resp.status_code == 403
        assert "not allowed" in resp.json()["detail"]
        mock_chat.assert_not_called()

    @patch("gateway.routes.chat.ollama.chat_completion", new_callable=AsyncMock)
    async def test_wildcard_tenant_passes_auth(self, mock_chat, client):
        """tenant-beta has allowed_models: ["*"], should pass auth but get 404 for unknown model."""
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "nonexistent-model", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer test-beta-key"},
        )
        # Auth passes (wildcard), but no backend serves this model -> 404
        assert resp.status_code == 404
        assert "No backend available" in resp.json()["detail"]
        mock_chat.assert_not_called()
