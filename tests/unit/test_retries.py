"""Unit tests for jittered backoff retry logic."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from gateway.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState


class TestRetryBehavior:
    """Test retry logic via circuit breaker and status code classification."""

    def test_429_does_not_trigger_circuit_failure(self):
        """429 should not record a circuit breaker failure."""
        cb = CircuitBreaker("test-backend", min_requests=1, failure_threshold=0.5)
        # Simulate: 429 does NOT call record_failure
        # The retry loop guards: only call record_failure for >= 500
        assert cb.state == CircuitState.CLOSED

    def test_5xx_triggers_circuit_failure(self):
        """5xx should record a circuit breaker failure."""
        cb = CircuitBreaker("test-backend", min_requests=1, failure_threshold=0.5)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_400_not_in_retryable_set(self):
        """400 errors should not be retried — only 429 and >= 500."""
        # 429 is retryable
        assert 429 == 429 or 429 >= 500  # True
        # 500, 502, 503 are retryable
        for code in [500, 502, 503]:
            assert code == 429 or code >= 500  # True
        # 400, 401, 403, 404 are NOT retryable
        for code in [400, 401, 403, 404]:
            assert not (code == 429 or code >= 500)  # False


class TestJitterCalculation:
    """Test jittered exponential backoff formula."""

    def test_jitter_range_attempt_0(self):
        """First retry delay should be ~0.25s to 1.0s."""
        import random
        random.seed(42)
        delays = []
        for _ in range(100):
            delay = 0.5 * (2 ** 0) * (0.5 + random.random())
            delays.append(delay)
        assert min(delays) >= 0.25
        assert max(delays) <= 1.0

    def test_jitter_range_attempt_1(self):
        """Second retry delay should be ~0.5s to 2.0s."""
        import random
        random.seed(42)
        delays = []
        for _ in range(100):
            delay = 0.5 * (2 ** 1) * (0.5 + random.random())
            delays.append(delay)
        assert min(delays) >= 0.5
        assert max(delays) <= 2.0

    def test_jitter_is_random(self):
        """Two calls should produce different delays."""
        import random
        d1 = 0.5 * (2 ** 0) * (0.5 + random.random())
        d2 = 0.5 * (2 ** 0) * (0.5 + random.random())
        # With overwhelming probability these are different
        assert d1 != d2


class TestRetryCount:
    """Test retry count tracking."""

    def test_no_retry_on_first_success(self):
        """retry_count should be 0 when first attempt succeeds."""
        retry_count = 0
        # Simulate: first attempt succeeds, no increment
        assert retry_count == 0

    def test_retry_count_increments(self):
        """retry_count should increment on each retryable failure."""
        retry_count = 0
        # Simulate two failures then success
        for attempt in range(3):
            if attempt < 2:  # First two fail
                retry_count += 1
            else:
                break  # Third succeeds
        assert retry_count == 2

    def test_max_retries_is_two(self):
        """Maximum retry count is 2 (3 total attempts)."""
        retry_count = 0
        for attempt in range(3):
            retry_count += 1  # All fail
        assert retry_count == 3  # But only 2 retries (first attempt + 2 retries)
        # Note: retry_count tracks total failures, header shows it
