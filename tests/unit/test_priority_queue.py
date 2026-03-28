import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.priority_queue import (
    PriorityQueueManager,
    QueueFullError,
    QueueTimeoutError,
)


class TestConcurrencyTracking:
    async def test_acquire_slot_under_limit(self):
        mgr = PriorityQueueManager(redis_client=AsyncMock())
        result = await mgr.acquire_slot("backend-1", max_concurrent=5)
        assert result is True
        assert mgr.get_concurrency("backend-1") == 1

    async def test_acquire_multiple_slots(self):
        mgr = PriorityQueueManager(redis_client=AsyncMock())
        for _ in range(5):
            await mgr.acquire_slot("backend-1", max_concurrent=5)
        assert mgr.get_concurrency("backend-1") == 5

    async def test_acquire_slot_at_limit(self):
        mgr = PriorityQueueManager(redis_client=AsyncMock())
        for _ in range(5):
            await mgr.acquire_slot("backend-1", max_concurrent=5)
        result = await mgr.acquire_slot("backend-1", max_concurrent=5)
        assert result is False
        assert mgr.get_concurrency("backend-1") == 5

    async def test_release_slot_decrements(self):
        redis_mock = AsyncMock()
        redis_mock.zpopmin = AsyncMock(return_value=[])
        mgr = PriorityQueueManager(redis_client=redis_mock)
        await mgr.acquire_slot("backend-1", max_concurrent=5)
        await mgr.acquire_slot("backend-1", max_concurrent=5)
        await mgr.release_slot("backend-1", "model-a")
        assert mgr.get_concurrency("backend-1") == 1

    async def test_release_slot_floors_at_zero(self):
        redis_mock = AsyncMock()
        redis_mock.zpopmin = AsyncMock(return_value=[])
        mgr = PriorityQueueManager(redis_client=redis_mock)
        await mgr.release_slot("backend-1", "model-a")
        assert mgr.get_concurrency("backend-1") == 0

    async def test_get_concurrency_unknown_backend(self):
        mgr = PriorityQueueManager(redis_client=AsyncMock())
        assert mgr.get_concurrency("nonexistent") == 0

    async def test_independent_backend_tracking(self):
        mgr = PriorityQueueManager(redis_client=AsyncMock())
        await mgr.acquire_slot("backend-1", max_concurrent=5)
        await mgr.acquire_slot("backend-2", max_concurrent=10)
        await mgr.acquire_slot("backend-2", max_concurrent=10)
        assert mgr.get_concurrency("backend-1") == 1
        assert mgr.get_concurrency("backend-2") == 2


class TestEnqueue:
    async def test_enqueue_adds_to_sorted_set(self):
        redis_mock = AsyncMock()
        redis_mock.zcard = AsyncMock(return_value=0)
        redis_mock.zadd = AsyncMock()
        mgr = PriorityQueueManager(redis_client=redis_mock)

        await mgr.enqueue("model-a", "req-1", priority=1)
        redis_mock.zadd.assert_called_once()
        call_args = redis_mock.zadd.call_args
        assert call_args[0][0] == "queue:model-a"

    async def test_depth_limit_raises_queue_full(self):
        redis_mock = AsyncMock()
        redis_mock.zcard = AsyncMock(return_value=100)
        mgr = PriorityQueueManager(redis_client=redis_mock, max_queue_depth=100)

        with pytest.raises(QueueFullError) as exc_info:
            await mgr.enqueue("model-a", "req-1", priority=1)
        assert exc_info.value.depth == 100

    async def test_enqueue_under_depth_limit_succeeds(self):
        redis_mock = AsyncMock()
        redis_mock.zcard = AsyncMock(return_value=99)
        redis_mock.zadd = AsyncMock()
        mgr = PriorityQueueManager(redis_client=redis_mock, max_queue_depth=100)

        await mgr.enqueue("model-a", "req-1", priority=1)
        redis_mock.zadd.assert_called_once()


class TestScoreDesign:
    async def test_lower_priority_number_gets_lower_score(self):
        redis_mock = AsyncMock()
        redis_mock.zcard = AsyncMock(return_value=0)
        scores = []

        async def capture_zadd(key, mapping):
            scores.append(list(mapping.values())[0])

        redis_mock.zadd = AsyncMock(side_effect=capture_zadd)
        mgr = PriorityQueueManager(redis_client=redis_mock)

        await mgr.enqueue("model-a", "req-high", priority=1)
        await mgr.enqueue("model-a", "req-low", priority=2)
        assert scores[0] < scores[1]  # priority 1 score < priority 2 score

    async def test_same_priority_fifo(self):
        redis_mock = AsyncMock()
        redis_mock.zcard = AsyncMock(return_value=0)
        scores = []

        async def capture_zadd(key, mapping):
            scores.append(list(mapping.values())[0])

        redis_mock.zadd = AsyncMock(side_effect=capture_zadd)
        mgr = PriorityQueueManager(redis_client=redis_mock)

        await mgr.enqueue("model-a", "req-first", priority=1)
        await asyncio.sleep(0.01)  # ensure different timestamps
        await mgr.enqueue("model-a", "req-second", priority=1)
        assert scores[0] < scores[1]  # first enqueued has lower score

    async def test_priority_dominates_timestamp(self):
        """Priority 1 at a later time still has lower score than priority 2 at an earlier time."""
        redis_mock = AsyncMock()
        redis_mock.zcard = AsyncMock(return_value=0)
        scores = []

        async def capture_zadd(key, mapping):
            scores.append(list(mapping.values())[0])

        redis_mock.zadd = AsyncMock(side_effect=capture_zadd)
        mgr = PriorityQueueManager(redis_client=redis_mock)

        # Enqueue priority 2 first (earlier timestamp)
        await mgr.enqueue("model-a", "req-low-priority", priority=2)
        await asyncio.sleep(0.01)
        # Enqueue priority 1 second (later timestamp)
        await mgr.enqueue("model-a", "req-high-priority", priority=1)

        # priority=1 (second enqueued) should still have LOWER score
        assert scores[1] < scores[0]


class TestWaitForSlot:
    async def test_wait_returns_on_event_set(self):
        mgr = PriorityQueueManager(redis_client=AsyncMock(), queue_timeout=5.0)

        async def signal_after_delay():
            await asyncio.sleep(0.05)
            event = mgr._waiters.get("req-1")
            if event:
                event.set()

        task = asyncio.create_task(signal_after_delay())
        wait_ms = await mgr.wait_for_slot("req-1")
        await task
        assert wait_ms > 0
        assert "req-1" not in mgr._waiters

    async def test_wait_timeout_raises(self):
        mgr = PriorityQueueManager(redis_client=AsyncMock(), queue_timeout=0.05)
        with pytest.raises(QueueTimeoutError):
            await mgr.wait_for_slot("req-1")
        assert "req-1" not in mgr._waiters

    async def test_waiter_cleaned_up_on_success(self):
        mgr = PriorityQueueManager(redis_client=AsyncMock(), queue_timeout=5.0)

        async def signal_immediately():
            await asyncio.sleep(0.01)
            mgr._waiters["req-1"].set()

        task = asyncio.create_task(signal_immediately())
        await mgr.wait_for_slot("req-1")
        await task
        assert "req-1" not in mgr._waiters

    async def test_waiter_cleaned_up_on_timeout(self):
        mgr = PriorityQueueManager(redis_client=AsyncMock(), queue_timeout=0.05)
        with pytest.raises(QueueTimeoutError):
            await mgr.wait_for_slot("req-1")
        assert "req-1" not in mgr._waiters


class TestSignalNextWaiter:
    async def test_pops_from_redis_and_signals(self):
        redis_mock = AsyncMock()
        redis_mock.zpopmin = AsyncMock(return_value=[("req-1", 1000000001234.5)])
        mgr = PriorityQueueManager(redis_client=redis_mock)

        event = asyncio.Event()
        mgr._waiters["req-1"] = event

        await mgr._signal_next_waiter("model-a")
        redis_mock.zpopmin.assert_called_once_with("queue:model-a", count=1)
        assert event.is_set()

    async def test_noop_on_empty_queue(self):
        redis_mock = AsyncMock()
        redis_mock.zpopmin = AsyncMock(return_value=[])
        mgr = PriorityQueueManager(redis_client=redis_mock)

        await mgr._signal_next_waiter("model-a")  # Should not raise

    async def test_skips_stale_entry_and_retries(self):
        redis_mock = AsyncMock()
        # First pop returns stale entry, second returns valid entry
        redis_mock.zpopmin = AsyncMock(
            side_effect=[
                [("stale-req", 1000.0)],
                [("valid-req", 2000.0)],
            ]
        )
        mgr = PriorityQueueManager(redis_client=redis_mock)

        event = asyncio.Event()
        mgr._waiters["valid-req"] = event
        # "stale-req" is NOT in _waiters

        await mgr._signal_next_waiter("model-a")
        assert event.is_set()
        assert redis_mock.zpopmin.call_count == 2

    async def test_gives_up_after_3_stale_entries(self):
        redis_mock = AsyncMock()
        redis_mock.zpopmin = AsyncMock(
            side_effect=[
                [("stale-1", 1000.0)],
                [("stale-2", 2000.0)],
                [("stale-3", 3000.0)],
            ]
        )
        mgr = PriorityQueueManager(redis_client=redis_mock)
        # None of these are in _waiters

        await mgr._signal_next_waiter("model-a")
        assert redis_mock.zpopmin.call_count == 3


class TestRemoveFromQueue:
    async def test_removes_from_redis(self):
        redis_mock = AsyncMock()
        mgr = PriorityQueueManager(redis_client=redis_mock)

        await mgr.remove_from_queue("model-a", "req-1")
        redis_mock.zrem.assert_called_once_with("queue:model-a", "req-1")


class TestGetQueueDepth:
    async def test_returns_zcard(self):
        redis_mock = AsyncMock()
        redis_mock.zcard = AsyncMock(return_value=42)
        mgr = PriorityQueueManager(redis_client=redis_mock)

        depth = await mgr.get_queue_depth("model-a")
        assert depth == 42
        redis_mock.zcard.assert_called_once_with("queue:model-a")


class TestReleaseSlotSignaling:
    async def test_release_signals_next_waiter(self):
        """release_slot should call _signal_next_waiter for the model."""
        redis_mock = AsyncMock()
        redis_mock.zpopmin = AsyncMock(return_value=[("req-1", 1000.0)])
        mgr = PriorityQueueManager(redis_client=redis_mock)

        # Set up a waiter
        event = asyncio.Event()
        mgr._waiters["req-1"] = event

        # Acquire and release a slot
        await mgr.acquire_slot("backend-1", max_concurrent=5)
        await mgr.release_slot("backend-1", "model-a")

        assert event.is_set()
        assert mgr.get_concurrency("backend-1") == 0
