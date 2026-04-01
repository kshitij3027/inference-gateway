"""Integration tests: chaos + circuit breaker + retry + rate limiter.

Validates that when chaos injection is active the gateway still produces
structured JSON responses, circuit breakers trip under sustained failure,
and rate limiting continues to function independently of backend chaos.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from gateway.chaos import ChaosConfig, ChaosHttpClient
from gateway.main import app
from gateway.models import ChatCompletionResponse, ChatMessageResponse, Choice, Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_MODEL = "mock-gpt-markdown"
AUTH_ALPHA = {"Authorization": "Bearer test-alpha-key"}
AUTH_BETA = {"Authorization": "Bearer test-beta-key"}
CHAT_PAYLOAD = {
    "model": MOCK_MODEL,
    "messages": [{"role": "user", "content": "hello"}],
}


def _mock_response(model: str = MOCK_MODEL) -> ChatCompletionResponse:
    """Build a minimal valid ChatCompletionResponse."""
    return ChatCompletionResponse(
        model=model,
        choices=[Choice(message=ChatMessageResponse(content="Hello!"))],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


def _make_noop_client() -> AsyncMock:
    """Create a mock httpx.AsyncClient whose post() returns a 200 response.

    This is used as the *underlying* client inside ChaosHttpClient so that
    when chaos does NOT inject a fault the call succeeds without hitting the
    network.
    """
    mock_resp = httpx.Response(200, json={"ok": True})
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(return_value=mock_resp)
    return mock_client


def _make_chaos_translator(chaos_client: ChaosHttpClient):
    """Return a translator coroutine that routes through the chaos client.

    The real openai/anthropic translators call ``client.post(url, ...)``
    which hits the ChaosHttpClient wrapper.  A plain AsyncMock would
    bypass chaos entirely.  This helper simulates the same flow: it calls
    ``chaos_client.post()`` with a dummy URL so chaos can inject faults,
    then converts the httpx exceptions into the same HTTPException the real
    translators would raise.

    The ChaosHttpClient wraps a no-op mock client (see ``_make_noop_client``)
    so non-fault calls succeed without network access.
    """

    async def _translator(*, client, backend, request):  # noqa: ARG001
        try:
            await chaos_client.post(
                f"http://fake-backend:3000/v1/chat/completions",
                json={"model": request.model},
            )
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="Backend request timed out")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"Backend error: {exc.response.status_code}",
            )
        return _mock_response(request.model)

    return _translator


def _setup_chaos(config: ChaosConfig) -> ChaosHttpClient:
    """Create a ChaosHttpClient backed by a no-op mock client.

    Also sets ``app.state.http_client`` to the chaos wrapper so that
    any code path reading from app state picks it up.
    """
    noop_client = _make_noop_client()
    chaos_client = ChaosHttpClient(noop_client, config)
    app.state.http_client = chaos_client
    return chaos_client


# ---------------------------------------------------------------------------
# Test 1 — Structured responses under moderate chaos
# ---------------------------------------------------------------------------


class TestChaosStructuredResponses:
    """Every request must get a valid JSON response even under chaos."""

    async def test_all_requests_get_structured_response(self, test_env):
        """Send 20 requests with 30 % error + 10 % timeout chaos.

        Every response must be either:
        - 200 with a ``choices`` list
        - 4xx/5xx with a ``detail`` key (or 429 from rate limiter)
        No unstructured errors, no crashes, no hangs.
        """
        async with app.router.lifespan_context(app):
            chaos_client = _setup_chaos(
                ChaosConfig(
                    error_rate=0.3, timeout_rate=0.1, latency_rate=0, seed=42
                )
            )
            translator = _make_chaos_translator(chaos_client)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS", {"openai": translator}
                ):
                    for i in range(20):
                        resp = await ac.post(
                            "/v1/chat/completions",
                            json={
                                "model": MOCK_MODEL,
                                "messages": [
                                    {"role": "user", "content": f"test {i}"}
                                ],
                            },
                            headers=AUTH_ALPHA,
                        )
                        body = resp.json()
                        assert isinstance(body, dict), (
                            f"Request {i}: non-dict response"
                        )
                        if resp.status_code == 200:
                            assert "choices" in body, (
                                f"Request {i}: 200 but no 'choices'"
                            )
                        else:
                            # 429 from rate-limiter, or 5xx/4xx with detail
                            assert "detail" in body or resp.status_code == 429, (
                                f"Request {i}: status {resp.status_code} "
                                f"with no 'detail': {body}"
                            )

    async def test_successful_requests_have_valid_shape(self, test_env):
        """Successful (200) responses under chaos have the expected fields."""
        async with app.router.lifespan_context(app):
            chaos_client = _setup_chaos(
                ChaosConfig(
                    error_rate=0.1, timeout_rate=0.05, latency_rate=0, seed=99
                )
            )
            translator = _make_chaos_translator(chaos_client)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS", {"openai": translator}
                ):
                    successes = 0
                    for i in range(30):
                        resp = await ac.post(
                            "/v1/chat/completions",
                            json={
                                "model": MOCK_MODEL,
                                "messages": [
                                    {"role": "user", "content": f"shape {i}"}
                                ],
                            },
                            headers=AUTH_ALPHA,
                        )
                        if resp.status_code == 200:
                            body = resp.json()
                            assert "choices" in body
                            assert len(body["choices"]) > 0
                            assert "message" in body["choices"][0]
                            assert "usage" in body
                            successes += 1

                    # With 10 % error + 5 % timeout and 5 backends (3 retries),
                    # most requests should still succeed
                    assert successes > 0, "Expected at least some 200s"


# ---------------------------------------------------------------------------
# Test 2 — Circuit breaker trips under heavy chaos
# ---------------------------------------------------------------------------


class TestChaosCircuitBreaker:
    """High chaos error rate should trip circuit breakers."""

    async def test_circuit_breaker_trips_under_chaos(self, test_env):
        """With 90 % error rate, enough failures accumulate to trip breakers.

        The circuit breaker has min_requests=10 and failure_threshold=50 %.
        At 90 % error rate across 30+ requests the breaker(s) should move
        to OPEN or HALF_OPEN.
        """
        async with app.router.lifespan_context(app):
            chaos_client = _setup_chaos(
                ChaosConfig(
                    error_rate=0.9, timeout_rate=0, latency_rate=0, seed=1
                )
            )
            translator = _make_chaos_translator(chaos_client)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS", {"openai": translator}
                ):
                    # Send enough requests to exceed min_requests (10) and
                    # failure_threshold (50 %) across all 5 openai backends
                    for i in range(30):
                        await ac.post(
                            "/v1/chat/completions",
                            json={
                                "model": MOCK_MODEL,
                                "messages": [
                                    {"role": "user", "content": f"chaos {i}"}
                                ],
                            },
                            headers=AUTH_BETA,
                        )

                    # Check backends state via admin endpoint
                    admin_resp = await ac.get("/admin/backends")
                    assert admin_resp.status_code == 200
                    backends = admin_resp.json()

                    # Collect circuit breaker states for mock-openai backends
                    openai_backends = [
                        b for b in backends if b["name"].startswith("mock-openai")
                    ]
                    assert len(openai_backends) > 0, "No mock-openai backends found"

                    states = [
                        b["circuit_breaker"].get("state", "unknown")
                        for b in openai_backends
                        if b.get("circuit_breaker")
                    ]
                    # At least one backend should have tripped
                    has_tripped = any(s in ("OPEN", "HALF_OPEN") for s in states)
                    assert has_tripped, (
                        f"No breakers tripped under 90 % error rate: {states}"
                    )

    async def test_circuit_breaker_error_rate_reflects_chaos(self, test_env):
        """Admin endpoint error_rate should be non-zero after chaos failures."""
        async with app.router.lifespan_context(app):
            chaos_client = _setup_chaos(
                ChaosConfig(
                    error_rate=0.7, timeout_rate=0, latency_rate=0, seed=7
                )
            )
            translator = _make_chaos_translator(chaos_client)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS", {"openai": translator}
                ):
                    for i in range(20):
                        await ac.post(
                            "/v1/chat/completions",
                            json={
                                "model": MOCK_MODEL,
                                "messages": [
                                    {"role": "user", "content": f"err {i}"}
                                ],
                            },
                            headers=AUTH_BETA,
                        )

                    admin_resp = await ac.get("/admin/backends")
                    backends = admin_resp.json()
                    openai_backends = [
                        b for b in backends if b["name"].startswith("mock-openai")
                    ]

                    # At least one backend should show a non-zero error rate
                    error_rates = [
                        b["circuit_breaker"].get("error_rate", 0.0)
                        for b in openai_backends
                        if b.get("circuit_breaker")
                    ]
                    assert any(r > 0 for r in error_rates), (
                        f"Expected non-zero error rates under 70 % chaos: "
                        f"{error_rates}"
                    )


# ---------------------------------------------------------------------------
# Test 3 — Rate limiter holds under chaos
# ---------------------------------------------------------------------------


class TestChaosRateLimiter:
    """Rate limiter should still enforce limits even with chaos enabled.

    Rate limit check happens BEFORE backend calls, so chaos (which affects
    backend calls) must not prevent 429 responses from being returned.
    """

    async def test_rate_limiter_holds_under_chaos(self, test_env):
        """All responses remain structured even with chaos + rate limiting.

        tenant-alpha has rate_limit_rps=10, rate_limit_rpm=60.  In unit
        tests Redis is not available, so the rate limiter is gracefully
        skipped.  Even so, every response must be a well-formed JSON with
        a known status code.
        """
        async with app.router.lifespan_context(app):
            chaos_client = _setup_chaos(
                ChaosConfig(
                    error_rate=0.2, timeout_rate=0.05, latency_rate=0, seed=42
                )
            )
            translator = _make_chaos_translator(chaos_client)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS", {"openai": translator}
                ):
                    status_codes: list[int] = []
                    for i in range(30):
                        resp = await ac.post(
                            "/v1/chat/completions",
                            json={
                                "model": MOCK_MODEL,
                                "messages": [
                                    {"role": "user", "content": f"rate {i}"}
                                ],
                            },
                            headers=AUTH_ALPHA,
                        )
                        status_codes.append(resp.status_code)

                    # All responses should have a recognised status code
                    allowed = {200, 429, 500, 502, 503, 504}
                    for i, code in enumerate(status_codes):
                        assert code in allowed, (
                            f"Request {i}: unexpected status {code}"
                        )

    async def test_chaos_does_not_corrupt_error_responses(self, test_env):
        """Error responses under chaos are valid JSON with expected keys.

        Uses 80 % error rate to guarantee some requests exhaust all 3
        retry attempts.  P(all 3 fail) = 0.8^3 = 51 %, so over 25
        requests we reliably see errors.
        """
        async with app.router.lifespan_context(app):
            chaos_client = _setup_chaos(
                ChaosConfig(
                    error_rate=0.8, timeout_rate=0.1, latency_rate=0, seed=77
                )
            )
            translator = _make_chaos_translator(chaos_client)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS", {"openai": translator}
                ):
                    errors_seen = 0
                    for i in range(25):
                        resp = await ac.post(
                            "/v1/chat/completions",
                            json={
                                "model": MOCK_MODEL,
                                "messages": [
                                    {"role": "user", "content": f"corrupt {i}"}
                                ],
                            },
                            headers=AUTH_ALPHA,
                        )
                        body = resp.json()
                        if resp.status_code != 200:
                            errors_seen += 1
                            # Error response must be a dict with 'detail'
                            # (or a 429 which may have a different shape)
                            if resp.status_code != 429:
                                assert "detail" in body, (
                                    f"Request {i}: {resp.status_code} "
                                    f"missing 'detail': {body}"
                                )

                    assert errors_seen > 0, (
                        "Expected at least one error with 80 % chaos rate"
                    )


# ---------------------------------------------------------------------------
# Test 4 — Retry + failover under chaos
# ---------------------------------------------------------------------------


class TestChaosRetryFailover:
    """Verify the retry loop successfully fails over across backends."""

    async def test_retry_loop_survives_partial_chaos(self, test_env):
        """With moderate chaos and 5 backends, requests can still succeed.

        The retry loop tries up to 3 backends.  With 30 % error rate, most
        requests should find at least one healthy backend within 3 attempts.
        """
        async with app.router.lifespan_context(app):
            chaos_client = _setup_chaos(
                ChaosConfig(
                    error_rate=0.3, timeout_rate=0, latency_rate=0, seed=123
                )
            )
            translator = _make_chaos_translator(chaos_client)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS", {"openai": translator}
                ):
                    successes = 0
                    for i in range(15):
                        resp = await ac.post(
                            "/v1/chat/completions",
                            json={
                                "model": MOCK_MODEL,
                                "messages": [
                                    {"role": "user", "content": f"retry {i}"}
                                ],
                            },
                            headers=AUTH_BETA,
                        )
                        if resp.status_code == 200:
                            successes += 1

                    # P(all 3 attempts fail) = 0.3^3 = 2.7 %
                    # Over 15 requests we expect most to succeed
                    assert successes >= 5, (
                        f"Too few successes ({successes}/15) with 30 % "
                        f"error rate and 3 retries"
                    )

    async def test_total_failure_returns_last_error(self, test_env):
        """When chaos makes every attempt fail, we still get a structured error."""
        async with app.router.lifespan_context(app):
            # 100 % error rate — every call fails
            chaos_client = _setup_chaos(
                ChaosConfig(
                    error_rate=1.0, timeout_rate=0, latency_rate=0, seed=0
                )
            )
            translator = _make_chaos_translator(chaos_client)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS", {"openai": translator}
                ):
                    resp = await ac.post(
                        "/v1/chat/completions",
                        json=CHAT_PAYLOAD,
                        headers=AUTH_BETA,
                    )
                    assert resp.status_code >= 500
                    body = resp.json()
                    assert "detail" in body
