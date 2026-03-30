from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import app
from gateway.models import ChatCompletionResponse, ChatMessageResponse, Choice, Usage


def _mock_response(model="mock-gpt-markdown"):
    return ChatCompletionResponse(
        model=model,
        choices=[Choice(message=ChatMessageResponse(content="Hello!"))],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


@pytest.fixture
async def client(test_env):
    """Client fixture that triggers app lifespan."""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


class TestRoutingStrategies:
    async def test_admin_routing_endpoint(self, client):
        resp = await client.get("/admin/routing")
        assert resp.status_code == 200
        data = resp.json()
        # mock-gpt-markdown is configured as latency_aware
        assert data["mock-gpt-markdown"]["strategy"] == "latency_aware"
        assert data["mock-gpt-markdown"]["hedge_enabled"] is True
        # mock-claude-markdown is configured as cost_aware
        assert data["mock-claude-markdown"]["strategy"] == "cost_aware"
        assert data["mock-claude-markdown"]["hedge_enabled"] is False
        # tinyllama is default (consistent_hash)
        assert data["tinyllama"]["strategy"] == "consistent_hash"

    async def test_cost_aware_picks_cheapest(self, client):
        """Cost-aware strategy should pick the cheapest backend."""
        mock_chat = AsyncMock(return_value=_mock_response("mock-claude-markdown"))
        with patch.dict("gateway.routes.chat.TRANSLATORS", {"anthropic": mock_chat}):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "mock-claude-markdown", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 200
        # mock-anthropic-1 has cost 0.01 (cheapest)
        assert resp.headers.get("x-backend") == "mock-anthropic-1"

    async def test_hedge_returns_winner_header(self, client):
        """Hedge request should return X-Hedge-Winner header."""
        mock_chat = AsyncMock(return_value=_mock_response())
        with patch.dict("gateway.routes.chat.TRANSLATORS", {"openai": mock_chat}):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "mock-gpt-markdown", "messages": [{"role": "user", "content": "Hi"}]},
                headers={
                    "Authorization": "Bearer test-alpha-key",
                    "X-Hedge": "true",
                },
            )
        assert resp.status_code == 200
        assert "x-hedge-winner" in resp.headers
        assert "x-hedge-loser" in resp.headers
        # Winner and loser should be different
        assert resp.headers["x-hedge-winner"] != resp.headers["x-hedge-loser"]

    async def test_no_hedge_without_header(self, client):
        """Normal request should not have hedge headers."""
        mock_chat = AsyncMock(return_value=_mock_response())
        with patch.dict("gateway.routes.chat.TRANSLATORS", {"openai": mock_chat}):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "mock-gpt-markdown", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 200
        assert "x-hedge-winner" not in resp.headers

    async def test_no_hedge_on_non_hedge_model(self, client):
        """Hedge header on model with hedge_enabled=false should be ignored."""
        mock_chat = AsyncMock(return_value=_mock_response("mock-claude-markdown"))
        with patch.dict("gateway.routes.chat.TRANSLATORS", {"anthropic": mock_chat}):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "mock-claude-markdown", "messages": [{"role": "user", "content": "Hi"}]},
                headers={
                    "Authorization": "Bearer test-alpha-key",
                    "X-Hedge": "true",
                },
            )
        assert resp.status_code == 200
        assert "x-hedge-winner" not in resp.headers
