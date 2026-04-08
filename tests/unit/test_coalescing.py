"""Unit tests for the RequestCoalescer."""

import asyncio
import time

import pytest

from gateway.coalescing import RequestCoalescer


class TestRequestCoalescer:
    def test_hash_request_deterministic(self):
        h1 = RequestCoalescer.hash_request('{"model":"gpt-4","messages":[]}')
        h2 = RequestCoalescer.hash_request('{"model":"gpt-4","messages":[]}')
        assert h1 == h2

    def test_hash_request_different_for_different_bodies(self):
        h1 = RequestCoalescer.hash_request('{"model":"gpt-4"}')
        h2 = RequestCoalescer.hash_request('{"model":"gpt-3"}')
        assert h1 != h2

    def test_check_returns_none_for_new_request(self):
        c = RequestCoalescer()
        assert c.check("hash1") is None

    @pytest.mark.asyncio
    async def test_check_returns_future_for_duplicate(self):
        c = RequestCoalescer()
        c.register("hash1")
        future = c.check("hash1")
        assert future is not None
        assert isinstance(future, asyncio.Future)

    @pytest.mark.asyncio
    async def test_resolve_delivers_to_waiters(self):
        c = RequestCoalescer()
        future = c.register("hash1")
        # Simulate a waiter
        waiter_future = c.check("hash1")
        assert waiter_future is future
        # Resolve
        c.resolve("hash1", {"status": "ok"})
        result = await future
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_reject_propagates_error(self):
        c = RequestCoalescer()
        future = c.register("hash1")
        c.reject("hash1", ValueError("backend failed"))
        with pytest.raises(ValueError, match="backend failed"):
            await future

    @pytest.mark.asyncio
    async def test_cleanup_after_resolve(self):
        c = RequestCoalescer()
        c.register("hash1")
        c.resolve("hash1", "result")
        # After resolve, check should return None (cleaned up)
        assert c.check("hash1") is None

    @pytest.mark.asyncio
    async def test_window_expiry(self):
        """Requests outside the 100ms window are not coalesced."""
        c = RequestCoalescer(window_ms=50)  # Short window for testing
        c.register("hash1")
        await asyncio.sleep(0.1)  # 100ms > 50ms window
        # Should NOT coalesce — window expired
        assert c.check("hash1") is None

    @pytest.mark.asyncio
    async def test_done_future_not_returned(self):
        """Already-resolved futures should not be returned by check."""
        c = RequestCoalescer()
        future = c.register("hash1")
        c.resolve("hash1", "result")
        # Future is done, check should return None
        assert c.check("hash1") is None

    def test_resolve_idempotent(self):
        """Calling resolve on non-existent hash doesn't error."""
        c = RequestCoalescer()
        c.resolve("nonexistent", "result")  # Should not raise

    def test_reject_idempotent(self):
        """Calling reject on non-existent hash doesn't error."""
        c = RequestCoalescer()
        c.reject("nonexistent", ValueError("err"))  # Should not raise
