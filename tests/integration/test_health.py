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
