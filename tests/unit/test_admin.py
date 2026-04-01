from unittest.mock import AsyncMock, patch

import pytest
import yaml
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from gateway.circuit_breaker import CircuitBreakerRegistry
from gateway.config import GatewayConfig, Registry
from gateway.models import ChatCompletionResponse, Choice, ChatMessageResponse, Usage
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


class TestCacheStats:
    async def test_returns_stats(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        mock_cache = AsyncMock()
        mock_cache.get_stats = AsyncMock(return_value={
            "hits": 10, "misses": 90, "hit_rate": 0.1, "entries": 5,
        })
        app.state.semantic_cache = mock_cache
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["hits"] == 10
        assert data["entries"] == 5

    async def test_returns_disabled_when_no_cache(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        app.state.semantic_cache = None
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/cache/stats")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False


class TestCacheFlush:
    async def test_flush_returns_count(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        mock_cache = AsyncMock()
        mock_cache.flush = AsyncMock(return_value=42)
        app.state.semantic_cache = mock_cache
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/admin/cache")
        assert resp.status_code == 200
        assert resp.json()["status"] == "flushed"
        assert resp.json()["entries_deleted"] == 42

    async def test_flush_returns_503_when_no_cache(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        app.state.semantic_cache = None
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.delete("/admin/cache")
        assert resp.status_code == 503


class TestQueueStats:
    async def test_returns_stats(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        mock_qm = AsyncMock()
        mock_qm.get_concurrency = lambda name: 3 if name == "test-backend" else 0
        mock_qm.get_queue_depth = AsyncMock(return_value=2)
        mock_qm.max_queue_depth = 100
        app.state.queue_manager = mock_qm
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["concurrency"]["test-backend"]["active"] == 3
        assert data["concurrency"]["test-backend"]["max"] == 10
        assert data["queues"]["tinyllama"]["depth"] == 2

    async def test_returns_disabled_when_no_queue(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        app.state.queue_manager = None
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/queue")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False


class TestJournalStats:
    async def test_returns_stats(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        mock_journal = AsyncMock()
        mock_journal.get_stats = AsyncMock(return_value={
            "total": 100, "inflight": 2, "entries_per_min": 5.5,
        })
        app.state.journal = mock_journal
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/journal/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["total"] == 100
        assert data["inflight"] == 2

    async def test_returns_disabled_when_no_journal(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        app.state.journal = None
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/journal/stats")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False


class TestJournalQuery:
    async def test_returns_entries(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        mock_journal = AsyncMock()
        mock_journal.query = AsyncMock(return_value=[
            {"request_id": "req-1", "tenant_id": "t-a", "model": "gpt-4", "status": "200"},
        ])
        app.state.journal = mock_journal
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/journal?last=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["count"] == 1
        assert data["entries"][0]["request_id"] == "req-1"

    async def test_caps_last_at_100(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        mock_journal = AsyncMock()
        mock_journal.query = AsyncMock(return_value=[])
        app.state.journal = mock_journal
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/admin/journal?last=500")
        mock_journal.query.assert_called_once_with(tenant_id=None, last=100)

    async def test_returns_disabled_when_no_journal(self, monkeypatch):
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        app.state.journal = None
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/admin/journal")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False


def _mock_response() -> ChatCompletionResponse:
    """Create a mock ChatCompletionResponse for testing."""
    return ChatCompletionResponse(
        model="mock-model",
        choices=[Choice(index=0, message=ChatMessageResponse(content="Hello!"))],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


class TestCacheWarm:
    """Tests for POST /admin/cache/warm endpoint."""

    async def test_warm_success(self, monkeypatch):
        """Warming with valid prompts returns warmed count."""
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        mock_cache = AsyncMock()
        mock_cache.store = AsyncMock()
        mock_translator = AsyncMock(return_value=_mock_response())
        app.state.semantic_cache = mock_cache
        app.state.http_client = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            with patch.dict("gateway.routes.chat.TRANSLATORS", {"ollama": mock_translator}):
                resp = await client.post(
                    "/admin/cache/warm",
                    json={
                        "prompts": [
                            {
                                "model": "tinyllama",
                                "messages": [{"role": "user", "content": "What is Python?"}],
                            }
                        ]
                    },
                )
        assert resp.status_code == 200
        data = resp.json()
        assert data["warmed"] == 1
        assert data["errors"] == 0
        mock_cache.store.assert_called_once()

    async def test_warm_no_cache(self, monkeypatch):
        """Returns 503 when semantic cache is not available."""
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        app.state.semantic_cache = None

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/admin/cache/warm",
                json={"prompts": [{"model": "m1", "messages": [{"role": "user", "content": "Hi"}]}]},
            )
        assert resp.status_code == 503

    async def test_warm_empty_prompts(self, monkeypatch):
        """Empty prompts list returns warmed=0, errors=0."""
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        mock_cache = AsyncMock()
        app.state.semantic_cache = mock_cache
        app.state.http_client = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/admin/cache/warm",
                json={"prompts": []},
            )
        assert resp.status_code == 200
        assert resp.json() == {"status": "completed", "warmed": 0, "errors": 0}

    async def test_warm_invalid_model_counts_error(self, monkeypatch):
        """Prompt with missing model counts as error."""
        registry = _make_registry(monkeypatch)
        app = _make_test_app(registry)
        mock_cache = AsyncMock()
        app.state.semantic_cache = mock_cache
        app.state.http_client = AsyncMock()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/admin/cache/warm",
                json={
                    "prompts": [
                        {"messages": [{"role": "user", "content": "Hi"}]},  # missing model
                    ]
                },
            )
        assert resp.status_code == 200
        assert resp.json()["errors"] == 1
        assert resp.json()["warmed"] == 0
