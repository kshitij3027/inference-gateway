import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def test_env(monkeypatch):
    """Set environment variables required by the gateway app."""
    monkeypatch.setenv("TENANT_ALPHA_KEY", "test-alpha-key")
    monkeypatch.setenv("TENANT_BETA_KEY", "test-beta-key")
    monkeypatch.setenv("CONFIG_PATH", "config/backends.yaml")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")


@pytest.fixture
async def client(test_env):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
