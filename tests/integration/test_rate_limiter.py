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
    return ChatCompletionResponse(
        model="tinyllama",
        choices=[Choice(message=ChatMessageResponse(content="Hello!"))],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


class _MockRateLimiter:
    """Configurable mock rate limiter for testing."""

    def __init__(self, rps_limit=None, deny_after=None):
        self._call_count = 0
        self._rps_limit = rps_limit
        self._deny_after = deny_after
        self._tokens_recorded = 0

    async def check_rate_limit(self, tenant_id, request_id, rps_limit, rpm_limit):
        self._call_count += 1
        if self._deny_after is not None and self._call_count > self._deny_after:
            return False, {
                "limit_type": "rps",
                "limit": rps_limit or 10,
                "current": rps_limit or 10,
                "retry_after": 1.0,
            }
        return True, None

    async def check_token_budget(self, tenant_id, budget):
        return True, None

    async def record_tokens(self, tenant_id, tokens):
        self._tokens_recorded += tokens
        return self._tokens_recorded

    async def get_remaining(self, tenant_id, rps_limit, rpm_limit):
        remaining = {}
        if rps_limit is not None:
            remaining["rps"] = max(0, rps_limit - self._call_count)
        if rpm_limit is not None:
            remaining["rpm"] = max(0, (rpm_limit or 60) - self._call_count)
        return remaining


class TestRateLimitEnforcement:
    async def test_429_when_rate_limit_exceeded(self, client):
        """Requests beyond the limit get 429."""
        mock_rl = _MockRateLimiter(deny_after=3)  # Allow first 3, deny rest
        mock_chat = AsyncMock(return_value=_mock_response())

        with (
            patch.object(client._transport.app.state, "rate_limiter", mock_rl),
            patch.dict("gateway.routes.chat.TRANSLATORS", {"ollama": mock_chat}),
        ):
            results = []
            for i in range(5):
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
                results.append(resp.status_code)

        assert results.count(200) == 3
        assert results.count(429) == 2

    async def test_429_includes_retry_after_header(self, client):
        """429 response has Retry-After header."""
        mock_rl = _MockRateLimiter(deny_after=0)  # Deny all
        with patch.object(client._transport.app.state, "rate_limiter", mock_rl):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 429
        assert "retry-after" in resp.headers

    async def test_429_json_body_structure(self, client):
        """429 response has structured JSON body."""
        mock_rl = _MockRateLimiter(deny_after=0)
        with patch.object(client._transport.app.state, "rate_limiter", mock_rl):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 429
        body = resp.json()["detail"]
        assert body["error"] == "rate_limit_exceeded"
        assert body["type"] == "rps"
        assert "limit" in body
        assert "retry_after" in body


class TestRateLimitHeaders:
    async def test_remaining_headers_on_success(self, client):
        """Successful requests include X-Ratelimit-Remaining headers."""
        mock_rl = _MockRateLimiter()
        mock_chat = AsyncMock(return_value=_mock_response())

        with (
            patch.object(client._transport.app.state, "rate_limiter", mock_rl),
            patch.dict("gateway.routes.chat.TRANSLATORS", {"ollama": mock_chat}),
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 200
        # Headers should be present since tenant-alpha has rate limits configured
        assert "x-ratelimit-remaining-rps" in resp.headers
        assert "x-ratelimit-remaining-rpm" in resp.headers

    async def test_remaining_decreases_with_requests(self, client):
        """Remaining count decreases as requests accumulate."""
        mock_rl = _MockRateLimiter()
        mock_chat = AsyncMock(return_value=_mock_response())

        with (
            patch.object(client._transport.app.state, "rate_limiter", mock_rl),
            patch.dict("gateway.routes.chat.TRANSLATORS", {"ollama": mock_chat}),
        ):
            remaining_values = []
            for i in range(3):
                resp = await client.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
                remaining_values.append(int(resp.headers.get("x-ratelimit-remaining-rps", 0)))

        # Remaining should decrease
        assert remaining_values[0] > remaining_values[2]


class TestGracefulDegradation:
    async def test_no_rate_limiter_passes_through(self, client):
        """When rate_limiter is None (Redis down), requests pass through."""
        mock_chat = AsyncMock(return_value=_mock_response())

        with (
            patch.object(client._transport.app.state, "rate_limiter", None),
            patch.dict("gateway.routes.chat.TRANSLATORS", {"ollama": mock_chat}),
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 200
        # No rate limit headers when rate limiter is disabled
        assert "x-ratelimit-remaining-rps" not in resp.headers


class TestTokenBudget:
    async def test_token_recording_after_response(self, client):
        """Tokens are recorded after successful non-streaming response."""
        mock_rl = _MockRateLimiter()
        mock_chat = AsyncMock(return_value=_mock_response())

        with (
            patch.object(client._transport.app.state, "rate_limiter", mock_rl),
            patch.dict("gateway.routes.chat.TRANSLATORS", {"ollama": mock_chat}),
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 200
        # Token budget is 500 for tenant-alpha, mock response has total_tokens=8
        assert mock_rl._tokens_recorded == 8
