from unittest.mock import AsyncMock, patch

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
from gateway.models import ChatCompletionResponse, ChatMessageResponse, Choice, Usage
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


# ── pre-routing span tests ────────────────────────────────────────────


def _mock_response():
    return ChatCompletionResponse(
        model="tinyllama",
        choices=[Choice(message=ChatMessageResponse(content="Hello!"))],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


class TestPreRoutingSpans:
    @pytest.fixture(autouse=True)
    def _setup_tracing(self):
        """Set up an in-memory TracerProvider for span capture."""
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        _reset_tracer_provider()
        trace.set_tracer_provider(provider)
        yield
        provider.shutdown()
        _reset_tracer_provider()

    async def test_rate_limit_span_created(self, monkeypatch, test_env):
        from gateway.main import app

        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS",
                    {"ollama": AsyncMock(return_value=_mock_response())},
                ):
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )
        assert resp.status_code == 200
        spans = self.exporter.get_finished_spans()
        rate_limit_spans = [s for s in spans if s.name == "gateway.rate_limit"]
        assert len(rate_limit_spans) == 1
        assert rate_limit_spans[0].attributes.get("tenant.id") == "tenant-alpha"
        assert rate_limit_spans[0].attributes.get("rate_limit.allowed") is True

    async def test_cache_lookup_span_created(self, monkeypatch, test_env):
        from gateway.main import app

        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS",
                    {"ollama": AsyncMock(return_value=_mock_response())},
                ):
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )
        assert resp.status_code == 200
        spans = self.exporter.get_finished_spans()
        cache_spans = [s for s in spans if s.name == "gateway.cache.lookup"]
        assert len(cache_spans) == 1
        assert cache_spans[0].attributes.get("cache.hit") is False

    async def test_journal_write_span_created(self, monkeypatch, test_env):
        from gateway.main import app

        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS",
                    {"ollama": AsyncMock(return_value=_mock_response())},
                ):
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )
        assert resp.status_code == 200
        spans = self.exporter.get_finished_spans()
        journal_spans = [s for s in spans if s.name == "gateway.journal.write"]
        assert len(journal_spans) >= 1
        request_spans = [s for s in journal_spans if s.attributes.get("journal.phase") == "request"]
        assert len(request_spans) >= 1

    async def test_router_span_created(self, monkeypatch, test_env):
        from gateway.main import app

        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS",
                    {"ollama": AsyncMock(return_value=_mock_response())},
                ):
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )
        assert resp.status_code == 200
        spans = self.exporter.get_finished_spans()
        router_spans = [s for s in spans if s.name == "gateway.router"]
        assert len(router_spans) >= 1
        assert router_spans[0].attributes.get("route.backend") is not None

    async def test_translator_span_created(self, monkeypatch, test_env):
        from gateway.main import app

        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS",
                    {"ollama": AsyncMock(return_value=_mock_response())},
                ):
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )
        assert resp.status_code == 200
        spans = self.exporter.get_finished_spans()
        translator_spans = [s for s in spans if s.name == "gateway.translator.request"]
        assert len(translator_spans) >= 1
        assert translator_spans[0].attributes.get("translator.streaming") is False

    async def test_circuit_breaker_span_created(self, monkeypatch, test_env):
        from gateway.main import app

        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS",
                    {"ollama": AsyncMock(return_value=_mock_response())},
                ):
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )
        assert resp.status_code == 200
        spans = self.exporter.get_finished_spans()
        cb_spans = [s for s in spans if s.name == "gateway.circuit_breaker"]
        assert len(cb_spans) >= 1
        assert cb_spans[0].attributes.get("cb.outcome") == "success"

    async def test_cache_store_span_created(self, monkeypatch, test_env):
        from gateway.main import app

        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS",
                    {"ollama": AsyncMock(return_value=_mock_response())},
                ):
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )
        assert resp.status_code == 200
        spans = self.exporter.get_finished_spans()
        store_spans = [s for s in spans if s.name == "gateway.cache.store"]
        assert len(store_spans) >= 1

    async def test_journal_completion_span_created(self, monkeypatch, test_env):
        from gateway.main import app

        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                with patch.dict(
                    "gateway.routes.chat.TRANSLATORS",
                    {"ollama": AsyncMock(return_value=_mock_response())},
                ):
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )
        assert resp.status_code == 200
        spans = self.exporter.get_finished_spans()
        journal_spans = [s for s in spans if s.name == "gateway.journal.write"]
        completion_spans = [s for s in journal_spans if s.attributes.get("journal.phase") == "completion"]
        assert len(completion_spans) >= 1
        assert completion_spans[0].attributes.get("journal.status") == 200
