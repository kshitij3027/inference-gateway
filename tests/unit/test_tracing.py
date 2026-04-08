import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway import auth as _auth_module
from gateway.auth import get_current_tenant
from gateway.config import GatewayConfig, Registry, TenantConfig
from gateway.observability.tracing import get_tracer, init_tracing, shutdown_tracing


# ── helpers ────────────────────────────────────────────────────────────


def _make_registry(monkeypatch) -> Registry:
    """Create a test registry with one tenant."""
    monkeypatch.setenv("TEST_KEY", "valid-key")
    config = GatewayConfig.model_validate({
        "backends": [{
            "name": "b1",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "models": ["tinyllama"],
        }],
        "tenants": [{
            "id": "test-tenant",
            "api_key_env": "TEST_KEY",
            "allowed_models": ["tinyllama"],
        }],
    })
    return Registry(config)


def _make_test_app(registry: Registry) -> FastAPI:
    """Create a minimal app with auth dependency for testing."""
    app = FastAPI()
    app.state.registry = registry

    @app.get("/test-auth")
    async def test_route(tenant: TenantConfig = Depends(get_current_tenant)):
        return {"tenant_id": tenant.id}

    return app


# ── unit tests for tracing module ──────────────────────────────────────


def _reset_tracer_provider():
    """Reset the global tracer provider so tests can set their own."""
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace._TRACER_PROVIDER = None


class TestInitTracing:
    def test_init_tracing_returns_none_when_no_endpoint(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        result = init_tracing()
        assert result is None

    def test_init_tracing_returns_provider_when_endpoint_set(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        _reset_tracer_provider()
        provider = init_tracing()
        try:
            assert isinstance(provider, TracerProvider)
        finally:
            if provider:
                provider.shutdown()
            _reset_tracer_provider()

    def test_get_tracer_returns_tracer(self):
        tracer = get_tracer()
        assert tracer is not None

    def test_shutdown_tracing_with_none(self):
        # Should not raise
        shutdown_tracing(None)


# ── auth span integration tests ────────────────────────────────────────


class TestAuthSpan:
    @pytest.fixture(autouse=True)
    def _setup_tracing(self):
        """Set up an in-memory TracerProvider for span capture."""
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        _reset_tracer_provider()
        # Reset the cached real tracer on the module-level proxy in auth.py
        # so it picks up the new provider we are about to install.
        _auth_module.tracer._real_tracer = None
        trace.set_tracer_provider(provider)
        yield
        provider.shutdown()
        _reset_tracer_provider()
        _auth_module.tracer._real_tracer = None

    async def test_auth_span_created_on_success(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/test-auth", headers={"Authorization": "Bearer valid-key"}
            )
        assert resp.status_code == 200

        spans = self.exporter.get_finished_spans()
        auth_spans = [s for s in spans if s.name == "gateway.auth"]
        assert len(auth_spans) == 1

        span = auth_spans[0]
        assert span.attributes.get("tenant.id") == "test-tenant"

    async def test_auth_span_created_on_failure(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/test-auth", headers={"Authorization": "Bearer wrong-key"}
            )
        assert resp.status_code == 401

        spans = self.exporter.get_finished_spans()
        auth_spans = [s for s in spans if s.name == "gateway.auth"]
        assert len(auth_spans) == 1
