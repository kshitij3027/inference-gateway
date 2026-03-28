"""Integration tests for priority queue in the chat route."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import app
from gateway.models import ChatCompletionResponse, ChatMessageResponse, Choice, Usage
from gateway.priority_queue import PriorityQueueManager


def _make_response(model="tinyllama", content="test response"):
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


@pytest.fixture
def mock_translator():
    response = _make_response()

    async def fake_chat_completion(client, backend, request):
        return response

    with patch.dict(
        "gateway.routes.chat.TRANSLATORS",
        {"ollama": fake_chat_completion, "openai": fake_chat_completion, "anthropic": fake_chat_completion},
    ):
        yield response


class TestQueueIntegration:
    async def test_request_proceeds_when_slots_available(self, test_env, mock_translator):
        """When slots are available, request proceeds without queuing."""
        async with app.router.lifespan_context(app):
            # Mock queue manager that always has slots
            mock_qm = AsyncMock(spec=PriorityQueueManager)
            mock_qm.acquire_slot = AsyncMock(return_value=True)
            mock_qm.release_slot = AsyncMock()
            app.state.queue_manager = mock_qm

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 200
            assert "X-Queue-Wait-Ms" not in resp.headers
            mock_qm.acquire_slot.assert_called()
            mock_qm.release_slot.assert_called()

    async def test_503_when_queue_full(self, test_env, mock_translator):
        """When queue is full, return 503 with Retry-After."""
        from gateway.priority_queue import QueueFullError

        async with app.router.lifespan_context(app):
            mock_qm = AsyncMock(spec=PriorityQueueManager)
            mock_qm.acquire_slot = AsyncMock(return_value=False)
            mock_qm.enqueue = AsyncMock(side_effect=QueueFullError(depth=100))
            app.state.queue_manager = mock_qm

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 503
            assert resp.headers.get("retry-after") == "5"

    async def test_504_on_queue_timeout(self, test_env, mock_translator):
        """When queue wait times out, return 504."""
        from gateway.priority_queue import QueueTimeoutError

        async with app.router.lifespan_context(app):
            mock_qm = AsyncMock(spec=PriorityQueueManager)
            mock_qm.acquire_slot = AsyncMock(return_value=False)
            mock_qm.enqueue = AsyncMock()
            mock_qm.wait_for_slot = AsyncMock(side_effect=QueueTimeoutError())
            mock_qm.remove_from_queue = AsyncMock()
            app.state.queue_manager = mock_qm

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 504
            mock_qm.remove_from_queue.assert_called_once()

    async def test_graceful_degradation_no_queue(self, test_env, mock_translator):
        """When queue_manager is None, requests proceed normally."""
        async with app.router.lifespan_context(app):
            app.state.queue_manager = None

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 200
            assert "X-Queue-Wait-Ms" not in resp.headers

    async def test_x_queue_wait_ms_header_on_dequeued_request(self, test_env, mock_translator):
        """Dequeued requests should have X-Queue-Wait-Ms header."""
        async with app.router.lifespan_context(app):
            mock_qm = AsyncMock(spec=PriorityQueueManager)
            mock_qm.acquire_slot = AsyncMock(side_effect=[False, True])  # first fails, second succeeds (after dequeue re-route)
            mock_qm.enqueue = AsyncMock()
            mock_qm.wait_for_slot = AsyncMock(return_value=150.5)  # 150.5ms wait
            mock_qm.release_slot = AsyncMock()
            app.state.queue_manager = mock_qm

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 200
            assert resp.headers.get("X-Queue-Wait-Ms") == "150.5"

    async def test_slot_released_on_backend_failure(self, test_env):
        """Slot must be released even when backend fails."""
        async def failing_translator(client, backend, request):
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="Backend error")

        with patch.dict(
            "gateway.routes.chat.TRANSLATORS",
            {"ollama": failing_translator, "openai": failing_translator, "anthropic": failing_translator},
        ):
            async with app.router.lifespan_context(app):
                mock_qm = AsyncMock(spec=PriorityQueueManager)
                mock_qm.acquire_slot = AsyncMock(return_value=True)
                mock_qm.release_slot = AsyncMock()
                app.state.queue_manager = mock_qm

                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as ac:
                    resp = await ac.post(
                        "/v1/chat/completions",
                        json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                        headers={"Authorization": "Bearer test-alpha-key"},
                    )
                # Should still get a response (may be error)
                # Important: release_slot was called despite failure
                assert mock_qm.release_slot.call_count > 0

    async def test_circuit_opens_while_queued(self, test_env, mock_translator):
        """If circuit breaker opens while request is queued, return 503 after dequeue."""
        async with app.router.lifespan_context(app):
            mock_qm = AsyncMock(spec=PriorityQueueManager)
            mock_qm.acquire_slot = AsyncMock(return_value=False)
            mock_qm.enqueue = AsyncMock()
            mock_qm.wait_for_slot = AsyncMock(return_value=100.0)

            # After dequeue, trip ALL circuit breakers so no backend is available
            original_find = app.state.registry.find_backend_for_model
            call_count = [0]

            def mock_find(model, routing_key=None, exclude=frozenset()):
                call_count[0] += 1
                if call_count[0] <= 1:
                    # First call (before enqueue): return normally
                    return original_find(model, routing_key=routing_key, exclude=exclude)
                # Second call (after dequeue): return None (all unavailable)
                return None

            app.state.registry.find_backend_for_model = mock_find
            app.state.queue_manager = mock_qm

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post(
                    "/v1/chat/completions",
                    json={"model": "tinyllama", "messages": [{"role": "user", "content": "hello"}]},
                    headers={"Authorization": "Bearer test-alpha-key"},
                )
            assert resp.status_code == 503
            assert "after dequeue" in resp.json()["detail"]
