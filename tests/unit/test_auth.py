import pytest
from fastapi import FastAPI, Depends
from httpx import ASGITransport, AsyncClient

from gateway.auth import get_current_tenant
from gateway.config import BackendConfig, GatewayConfig, Registry, TenantConfig


def _make_test_app(registry: Registry) -> FastAPI:
    """Create a minimal app with auth dependency for testing."""
    app = FastAPI()
    app.state.registry = registry

    @app.get("/test-auth")
    async def test_route(tenant: TenantConfig = Depends(get_current_tenant)):
        return {"tenant_id": tenant.id}

    return app


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


class TestGetCurrentTenant:
    async def test_valid_key_returns_tenant(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/test-auth", headers={"Authorization": "Bearer valid-key"})
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "test-tenant"

    async def test_invalid_key_returns_401(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/test-auth", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401

    async def test_missing_header_returns_401(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/test-auth")
        assert resp.status_code == 401

    async def test_malformed_header_returns_401(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/test-auth", headers={"Authorization": "Basic abc123"})
        assert resp.status_code == 401

    async def test_empty_bearer_returns_401(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/test-auth", headers={"Authorization": "Bearer "})
        assert resp.status_code == 401

    async def test_bearer_case_sensitive(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/test-auth", headers={"Authorization": "bearer valid-key"})
        assert resp.status_code == 401
