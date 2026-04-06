"""Unit tests for the EventBroadcaster."""

import asyncio
import json

import pytest

from gateway.events import EventBroadcaster


class MockWebSocket:
    """Minimal mock WebSocket for testing."""

    def __init__(self, *, fail_on_send: bool = False) -> None:
        self.accepted = False
        self.sent: list[str] = []
        self.fail_on_send = fail_on_send

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, data: str) -> None:
        if self.fail_on_send:
            raise RuntimeError("connection closed")
        self.sent.append(data)

    async def receive_text(self) -> str:
        await asyncio.sleep(999)  # Block forever
        return ""


class TestEventBroadcaster:
    def test_starts_empty(self):
        b = EventBroadcaster()
        assert b.client_count == 0

    @pytest.mark.asyncio
    async def test_connect_increments_count(self):
        b = EventBroadcaster()
        ws = MockWebSocket()
        await b.connect(ws)
        assert b.client_count == 1
        assert ws.accepted

    @pytest.mark.asyncio
    async def test_disconnect_decrements_count(self):
        b = EventBroadcaster()
        ws = MockWebSocket()
        await b.connect(ws)
        b.disconnect(ws)
        assert b.client_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_idempotent(self):
        b = EventBroadcaster()
        ws = MockWebSocket()
        await b.connect(ws)
        b.disconnect(ws)
        b.disconnect(ws)  # Should not raise
        assert b.client_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all_clients(self):
        b = EventBroadcaster()
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        await b.connect(ws1)
        await b.connect(ws2)
        await b.broadcast("test_event", {"key": "value"})
        assert len(ws1.sent) == 1
        assert len(ws2.sent) == 1
        msg1 = json.loads(ws1.sent[0])
        assert msg1["type"] == "test_event"
        assert msg1["data"] == {"key": "value"}
        assert "ts" in msg1

    @pytest.mark.asyncio
    async def test_broadcast_prunes_dead_connections(self):
        b = EventBroadcaster()
        ws_ok = MockWebSocket()
        ws_dead = MockWebSocket(fail_on_send=True)
        await b.connect(ws_ok)
        await b.connect(ws_dead)
        assert b.client_count == 2
        await b.broadcast("test", {"x": 1})
        assert b.client_count == 1
        assert len(ws_ok.sent) == 1

    @pytest.mark.asyncio
    async def test_broadcast_noop_when_no_clients(self):
        b = EventBroadcaster()
        await b.broadcast("test", {"x": 1})  # Should not raise

    @pytest.mark.asyncio
    async def test_event_json_format(self):
        b = EventBroadcaster()
        ws = MockWebSocket()
        await b.connect(ws)
        await b.broadcast("cache_hit", {"model": "gpt-4", "tenant_id": "t1"})
        msg = json.loads(ws.sent[0])
        assert set(msg.keys()) == {"type", "data", "ts"}
        assert isinstance(msg["ts"], float)

    def test_emit_noop_when_no_clients(self):
        """emit() should not create a task if no clients are connected."""
        b = EventBroadcaster()
        b.emit("test", {"x": 1})  # Should not raise, no task created
