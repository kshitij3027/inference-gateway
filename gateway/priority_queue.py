"""Priority queue with per-backend concurrency tracking.

Enforces max_concurrent per backend. Overflow requests are queued in
Redis sorted sets with priority-based ordering (lower priority number = higher priority).
Uses asyncio.Event for zero-latency dequeue signaling.
"""

import asyncio
import time

import structlog

logger = structlog.get_logger()


class QueueFullError(Exception):
    """Raised when the queue depth exceeds the configured limit."""

    def __init__(self, depth: int) -> None:
        self.depth = depth
        super().__init__(f"Queue full: {depth} entries")


class QueueTimeoutError(Exception):
    """Raised when a queued request exceeds the wait timeout."""

    pass


class PriorityQueueManager:
    """Per-backend concurrency tracking with priority queue for overflow.

    Concurrency tracking is in-process (dict + asyncio.Lock) for fast-path checks.
    Queue storage uses Redis sorted sets for priority ordering and atomic ZPOPMIN.
    Wait signaling uses in-process asyncio.Event for zero-latency notification.
    """

    def __init__(
        self,
        redis_client,
        max_queue_depth: int = 100,
        queue_timeout: float = 30.0,
    ) -> None:
        self.redis = redis_client
        self.max_queue_depth = max_queue_depth
        self.queue_timeout = queue_timeout
        self._concurrency: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._waiters: dict[str, asyncio.Event] = {}

    async def acquire_slot(self, backend_name: str, max_concurrent: int) -> bool:
        """Try to acquire a concurrency slot for a backend.

        Returns True if slot acquired (counter < max_concurrent), False if at capacity.
        """
        async with self._lock:
            current = self._concurrency.get(backend_name, 0)
            if current < max_concurrent:
                self._concurrency[backend_name] = current + 1
                return True
            return False

    async def release_slot(self, backend_name: str, model: str) -> None:
        """Release a concurrency slot and signal the next queued request if any."""
        async with self._lock:
            current = self._concurrency.get(backend_name, 0)
            self._concurrency[backend_name] = max(0, current - 1)

        # Try to wake the highest-priority queued request for this model
        await self._signal_next_waiter(model)

    def get_concurrency(self, backend_name: str) -> int:
        """Return the current active request count for a backend."""
        return self._concurrency.get(backend_name, 0)

    async def enqueue(self, model: str, request_id: str, priority: int) -> None:
        """Enqueue a request in the priority queue.

        Score = priority * 1e12 + timestamp. Lower score = dequeued first.
        Raises QueueFullError if queue depth >= max_queue_depth.
        """
        queue_key = f"queue:{model}"

        # Check depth limit
        depth = await self.redis.zcard(queue_key)
        if depth >= self.max_queue_depth:
            raise QueueFullError(depth=depth)

        score = priority * 1_000_000_000_000 + time.time()
        await self.redis.zadd(queue_key, {request_id: score})
        logger.info(
            "queue_enqueued",
            model=model,
            request_id=request_id,
            priority=priority,
            depth=depth + 1,
        )

    async def wait_for_slot(self, request_id: str) -> float:
        """Wait for a slot to become available via Event signaling.

        Returns the wait time in milliseconds.
        Raises QueueTimeoutError if timeout exceeded.
        """
        event = asyncio.Event()
        self._waiters[request_id] = event
        start = time.monotonic()
        try:
            await asyncio.wait_for(event.wait(), timeout=self.queue_timeout)
            wait_ms = round((time.monotonic() - start) * 1000, 2)
            logger.info("queue_dequeued", request_id=request_id, wait_ms=wait_ms)
            return wait_ms
        except asyncio.TimeoutError:
            logger.warning(
                "queue_timeout", request_id=request_id, timeout=self.queue_timeout
            )
            raise QueueTimeoutError()
        finally:
            self._waiters.pop(request_id, None)

    async def remove_from_queue(self, model: str, request_id: str) -> None:
        """Remove a request from the Redis queue (e.g., on timeout cleanup)."""
        await self.redis.zrem(f"queue:{model}", request_id)

    async def get_queue_depth(self, model: str) -> int:
        """Return the current queue depth for a model."""
        return await self.redis.zcard(f"queue:{model}")

    async def _signal_next_waiter(self, model: str) -> None:
        """Pop the highest-priority request from Redis and signal its Event.

        Handles stale entries (timed-out waiters) by retrying up to 3 times.
        """
        queue_key = f"queue:{model}"
        for _ in range(3):
            result = await self.redis.zpopmin(queue_key, count=1)
            if not result:
                return  # Queue is empty

            # result is a list of (member, score) tuples
            request_id = (
                result[0][0] if isinstance(result[0], (list, tuple)) else result[0]
            )
            event = self._waiters.get(request_id)
            if event is not None:
                event.set()
                return
            # Stale entry (waiter timed out) -- try next
            logger.debug("queue_stale_entry", request_id=request_id, model=model)
