from httpx import ASGITransport, AsyncClient

from gateway.main import app


async def test_health_returns_ok(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ready_returns_ok_with_healthy_backends(client):
    """Readiness probe returns 200 when at least 1 backend is healthy."""
    response = await client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["healthy_backends"] > 0
    assert data["total_backends"] > 0


async def test_503_during_shutdown(test_env):
    """Chat requests return 503 during shutdown."""
    async with app.router.lifespan_context(app):
        app.state.shutting_down = True
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/v1/chat/completions",
                json={"model": "tinyllama", "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer test-alpha-key"},
            )
        assert resp.status_code == 503
        assert resp.json()["error"] == "shutting_down"
        assert "Retry-After" in resp.headers


async def test_health_allowed_during_shutdown(test_env):
    """Health endpoint still works during shutdown."""
    async with app.router.lifespan_context(app):
        app.state.shutting_down = True
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
        assert resp.status_code == 200


async def test_ready_allowed_during_shutdown(test_env):
    """Ready endpoint still works during shutdown."""
    async with app.router.lifespan_context(app):
        app.state.shutting_down = True
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/ready")
        assert resp.status_code == 200
