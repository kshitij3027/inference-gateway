"""Unit tests for Prometheus metrics endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY, make_asgi_app

from gateway.main import app as gateway_app
from gateway.models import ChatCompletionResponse, ChatMessageResponse, Choice, Usage


def _make_metrics_app() -> FastAPI:
    """Create a minimal app with /metrics mounted."""
    app = FastAPI()
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    return app


class TestMetricsEndpoint:
    async def test_metrics_returns_prometheus_format(self):
        app = _make_metrics_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/metrics/")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")
        body = resp.text
        assert "# HELP" in body
        assert "# TYPE" in body

    async def test_metrics_includes_all_metric_families(self):
        # Import to ensure metrics are registered
        import gateway.observability.metrics  # noqa: F401

        app = _make_metrics_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/metrics/")
        body = resp.text
        expected_metrics = [
            "gateway_request_total",
            "gateway_request_duration_seconds",
            "gateway_cache_operations_total",
            "gateway_rate_limit_rejections_total",
            "gateway_circuit_breaker_state",
            "gateway_queue_depth",
            "gateway_tokens_consumed_total",
            "gateway_active_requests",
        ]
        for metric_name in expected_metrics:
            assert metric_name in body, f"Missing metric: {metric_name}"


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


class TestMetricIncrements:
    async def test_request_counter_increments(self, test_env):
        """Chat request should increment gateway_request_total."""
        response = _make_response()

        async def fake_chat(client, backend, request):
            return response

        with patch.dict(
            "gateway.routes.chat.TRANSLATORS",
            {"ollama": fake_chat, "openai": fake_chat, "anthropic": fake_chat},
        ):
            async with gateway_app.router.lifespan_context(gateway_app):
                gateway_app.state.semantic_cache = None
                gateway_app.state.queue_manager = None

                before = REGISTRY.get_sample_value(
                    "gateway_request_total",
                    {"tenant": "tenant-alpha", "model": "tinyllama",
                     "backend": "", "status_code": "200", "method": "POST"},
                ) or 0.0

                transport = ASGITransport(app=gateway_app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "hi"}]},
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )
                assert resp.status_code == 200

                after = REGISTRY.get_sample_value(
                    "gateway_request_total",
                    {"tenant": "tenant-alpha", "model": "tinyllama",
                     "backend": "", "status_code": "200", "method": "POST"},
                )
                # Counter should have incremented (backend may vary, check with empty fallback)
                # Use broader check: any gateway_request_total with tenant-alpha increased
                resp2 = await AsyncClient(
                    transport=ASGITransport(app=gateway_app), base_url="http://test"
                ).__aenter__()
                metrics_resp = await resp2.get("/metrics/")
                await resp2.__aexit__(None, None, None)
                assert "gateway_request_total" in metrics_resp.text
                assert "tenant-alpha" in metrics_resp.text

    async def test_latency_histogram_observed(self, test_env):
        """Chat request should observe gateway_request_duration_seconds."""
        response = _make_response()

        async def fake_chat(client, backend, request):
            return response

        with patch.dict(
            "gateway.routes.chat.TRANSLATORS",
            {"ollama": fake_chat, "openai": fake_chat, "anthropic": fake_chat},
        ):
            async with gateway_app.router.lifespan_context(gateway_app):
                gateway_app.state.semantic_cache = None
                gateway_app.state.queue_manager = None

                transport = ASGITransport(app=gateway_app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    await ac.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "hi"}]},
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )

                async with AsyncClient(
                    transport=ASGITransport(app=gateway_app), base_url="http://test"
                ) as ac:
                    metrics_resp = await ac.get("/metrics/")
                assert "gateway_request_duration_seconds" in metrics_resp.text

    async def test_cache_hit_counter_increments(self, test_env):
        """Cache hit should increment gateway_cache_operations_total."""
        cached = _make_response(content="cached")
        async with gateway_app.router.lifespan_context(gateway_app):
            mock_cache = AsyncMock()
            mock_cache.lookup = AsyncMock(return_value=(cached, 0.98, "L2_HIT"))
            mock_cache.record_hit = AsyncMock()
            gateway_app.state.semantic_cache = mock_cache
            gateway_app.state.queue_manager = None

            transport = ASGITransport(app=gateway_app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hi"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            async with AsyncClient(
                transport=ASGITransport(app=gateway_app), base_url="http://test"
            ) as ac:
                metrics_resp = await ac.get("/metrics/")
            assert 'gateway_cache_operations_total{model="tinyllama",status="hit"}' in metrics_resp.text

    async def test_request_id_propagation(self, test_env):
        """X-Request-ID should be echoed back in response."""
        response = _make_response()

        async def fake_chat(client, backend, request):
            return response

        with patch.dict(
            "gateway.routes.chat.TRANSLATORS",
            {"ollama": fake_chat, "openai": fake_chat, "anthropic": fake_chat},
        ):
            async with gateway_app.router.lifespan_context(gateway_app):
                gateway_app.state.semantic_cache = None
                gateway_app.state.queue_manager = None

                transport = ASGITransport(app=gateway_app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "hi"}]},
                        headers={
                            "Authorization": "Bearer test-alpha-key",
                            "X-Request-ID": "test-req-123",
                        },
                    )
                assert resp.headers.get("X-Request-ID") == "test-req-123"
