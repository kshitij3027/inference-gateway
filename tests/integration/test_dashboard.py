"""Integration tests for the live web dashboard."""

import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import app


@pytest.fixture
def test_env(monkeypatch):
    """Set required environment variables for gateway startup."""
    monkeypatch.setenv("TENANT_ALPHA_KEY", "test-alpha-key")
    monkeypatch.setenv("TENANT_BETA_KEY", "test-beta-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")


@pytest.fixture
async def client(test_env):
    """Create test client with full app lifecycle (Redis mocked out)."""
    with patch("gateway.main.aioredis") as mock_redis_module:
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("no redis in test"))
        mock_redis_module.from_url.return_value = mock_redis

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                yield ac


class TestDashboardStaticFiles:
    @pytest.mark.asyncio
    async def test_dashboard_html_loads(self, client):
        resp = await client.get("/dashboard/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Inference Gateway" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_js_loads(self, client):
        resp = await client.get("/dashboard/dashboard.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_dashboard_css_loads(self, client):
        resp = await client.get("/dashboard/dashboard.css")
        assert resp.status_code == 200
        assert "css" in resp.headers.get("content-type", "")


class TestDashboardWebSocket:
    @pytest.mark.asyncio
    async def test_websocket_endpoint_exists(self, test_env):
        """Verify WebSocket endpoint is registered (check via route list)."""
        routes = [r.path for r in app.routes]
        assert "/ws/dashboard" in routes
