import pytest
import yaml
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from gateway.circuit_breaker import CircuitBreakerRegistry
from gateway.config import GatewayConfig, Registry
from gateway.routes.admin import router as admin_router


def _make_test_app(registry: Registry, config_path: str = "") -> FastAPI:
    """Create a test app with admin router and mock registry."""
    app = FastAPI()
    app.include_router(admin_router)
    app.state.registry = registry
    app.state.config_path = config_path
    app.state.circuit_breakers = CircuitBreakerRegistry(
        list(registry.backends.keys())
    )
    return app


def _make_registry(monkeypatch) -> Registry:
    monkeypatch.setenv("TEST_KEY", "key1")
    config = GatewayConfig.model_validate({
        "backends": [{
            "name": "test-backend",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "models": ["tinyllama"],
        }],
        "tenants": [{
            "id": "t1",
            "api_key_env": "TEST_KEY",
            "allowed_models": ["tinyllama"],
        }],
    })
    return Registry(config)


class TestListBackends:
    async def test_returns_all_backends(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/backends")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "test-backend"
        assert data[0]["provider"] == "ollama"
        assert data[0]["models"] == ["tinyllama"]
        assert data[0]["health"] == "CLOSED"
        assert "circuit_breaker" in data[0]
        assert data[0]["circuit_breaker"]["state"] == "CLOSED"


class TestReloadConfig:
    async def test_reload_success(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TEST_KEY", "key1")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({
            "backends": [{
                "name": "reloaded",
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "models": ["tinyllama"],
            }],
            "tenants": [{
                "id": "t1",
                "api_key_env": "TEST_KEY",
                "allowed_models": ["tinyllama"],
            }],
        }))
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry, config_path=str(config_file))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/admin/reload")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reloaded"
        assert resp.json()["backends"] == 1
        # Verify registry was swapped
        assert "reloaded" in app.state.registry.backends

    async def test_reload_bad_config_returns_400(self, monkeypatch, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid")
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry, config_path=str(config_file))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/admin/reload")
        assert resp.status_code == 400
        # Old registry should be unchanged
        assert "test-backend" in app.state.registry.backends

    async def test_reload_missing_file_returns_400(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry, config_path="/nonexistent/config.yaml")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/admin/reload")
        assert resp.status_code == 400
