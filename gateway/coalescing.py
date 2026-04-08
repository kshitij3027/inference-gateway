"""Request coalescing — deduplicate identical in-flight requests."""

import asyncio
import hashlib
import time

import structlog

logger = structlog.get_logger()


class RequestCoalescer:
    """Coalesce identical in-flight requests within a time window.

    If an identical request (same body hash) is already being processed
    within the window, new arrivals wait on the same Future instead of
    making a separate backend call.
    """

    def __init__(self, window_ms: float = 100.0) -> None:
        self._window_ms = window_ms
        self._inflight: dict[str, tuple[asyncio.Future, float]] = {}

    @staticmethod
    def hash_request(body_json: str) -> str:
        """Hash a serialized request body."""
        return hashlib.sha256(body_json.encode()).hexdigest()

    def check(self, request_hash: str) -> asyncio.Future | None:
        """Check if an identical request is in-flight within the window.

        Returns the Future to await if coalesced, None if this is a new request.
        """
        if request_hash in self._inflight:
            future, created_at = self._inflight[request_hash]
            elapsed_ms = (time.monotonic() - created_at) * 1000
            if elapsed_ms <= self._window_ms and not future.done():
                return future
            # Expired or done — clean up
            del self._inflight[request_hash]
        return None

    def register(self, request_hash: str) -> asyncio.Future:
        """Register a new in-flight request. Caller must resolve/reject the Future."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._inflight[request_hash] = (future, time.monotonic())
        return future

    def resolve(self, request_hash: str, result) -> None:
        """Resolve the Future with a successful result."""
        if request_hash in self._inflight:
            future, _ = self._inflight[request_hash]
            if not future.done():
                future.set_result(result)
            del self._inflight[request_hash]

    def reject(self, request_hash: str, error: Exception) -> None:
        """Reject the Future with an error."""
        if request_hash in self._inflight:
            future, _ = self._inflight[request_hash]
            if not future.done():
                future.set_exception(error)
            del self._inflight[request_hash]
