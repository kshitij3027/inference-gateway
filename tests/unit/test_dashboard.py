"""Unit tests for dashboard event types and circuit breaker callback."""

import json

import pytest

from gateway.circuit_breaker import CircuitBreaker, CircuitState
from gateway.events import EventBroadcaster


# All 6 event types and their required data fields
EVENT_TYPES = {
    "new_request": {"request_id": "req-1", "tenant_id": "t1", "model": "gpt-4", "stream": False},
    "request_complete": {"request_id": "req-1", "tenant_id": "t1", "model": "gpt-4", "backend": "b1", "duration_ms": 100.5, "tokens": 50, "status": 200},
    "cache_hit": {"request_id": "req-1", "model": "gpt-4", "tenant_id": "t1"},
    "cache_miss": {"request_id": "req-1", "model": "gpt-4", "tenant_id": "t1"},
    "circuit_state_change": {"backend": "b1", "old_state": "CLOSED", "new_state": "OPEN"},
    "rate_limit_hit": {"request_id": "req-1", "tenant_id": "t1", "limit_type": "rps"},
}


class MockWebSocket:
    def __init__(self):
        self.sent = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, data):
        self.sent.append(data)


class TestEventSerialization:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("event_type,data", EVENT_TYPES.items())
    async def test_all_event_types_produce_valid_json(self, event_type, data):
        b = EventBroadcaster()
        ws = MockWebSocket()
        await b.connect(ws)
        await b.broadcast(event_type, data)
        msg = json.loads(ws.sent[0])
        assert msg["type"] == event_type
        assert msg["data"] == data
        assert isinstance(msg["ts"], float)


class TestCircuitBreakerCallback:
    def test_callback_fires_on_transition(self):
        cb = CircuitBreaker("test-backend", min_requests=1, failure_threshold=0.5)
        changes = []
        cb.on_state_change = lambda backend, old, new: changes.append((backend, old, new))
        # Force a failure to trip the breaker
        cb.record_failure()
        cb.record_failure()
        assert len(changes) == 1
        assert changes[0] == ("test-backend", "CLOSED", "OPEN")

    def test_callback_none_by_default(self):
        cb = CircuitBreaker("test-backend")
        assert cb.on_state_change is None

    def test_callback_exception_is_swallowed(self):
        cb = CircuitBreaker("test-backend", min_requests=1, failure_threshold=0.5)
        cb.on_state_change = lambda *args: (_ for _ in ()).throw(ValueError("boom"))
        cb.record_failure()
        cb.record_failure()  # Should not raise despite callback error
        assert cb.state == CircuitState.OPEN
