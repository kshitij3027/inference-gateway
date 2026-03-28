"""Unit tests for Prometheus metrics endpoint."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from prometheus_client import make_asgi_app


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
