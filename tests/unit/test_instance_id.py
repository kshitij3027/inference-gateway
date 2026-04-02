"""Tests for X-Instance-ID response header."""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import app
from gateway.models import ChatCompletionResponse, ChatMessageResponse, Choice, Usage


def _mock_response():
    return ChatCompletionResponse(
        model="tinyllama",
        choices=[Choice(message=ChatMessageResponse(content="Hello!"))],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


class TestInstanceIdHeader:
    async def test_instance_id_from_env(self, test_env, monkeypatch):
        """X-Instance-ID should match the INSTANCE_ID env var."""
        monkeypatch.setenv("INSTANCE_ID", "gw-test-42")
        mock_chat = AsyncMock(return_value=_mock_response())

        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                with patch.dict("gateway.routes.chat.TRANSLATORS", {"ollama": mock_chat}):
                    resp = await ac.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )
            assert resp.status_code == 200
            assert resp.headers.get("x-instance-id") == "gw-test-42"

    async def test_instance_id_default_hostname(self, test_env):
        """Without INSTANCE_ID env var, defaults to hostname."""
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/health")
            assert resp.status_code == 200
            # Should have some instance ID (hostname)
            instance_id = resp.headers.get("x-instance-id")
            assert instance_id is not None
            assert len(instance_id) > 0

    async def test_instance_id_on_health_endpoint(self, test_env, monkeypatch):
        """Health endpoint should also include X-Instance-ID."""
        monkeypatch.setenv("INSTANCE_ID", "gw-health")
        async with app.router.lifespan_context(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.get("/health")
            assert resp.status_code == 200
            assert resp.headers.get("x-instance-id") == "gw-health"
