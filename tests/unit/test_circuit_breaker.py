from unittest.mock import patch

import pytest

from gateway.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
)


class TestCircuitBreakerTransitions:
    def test_starts_closed(self):
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker("test", min_requests=10, failure_threshold=0.5)
        # 4 failures, 6 successes = 40% < 50%
        for _ in range(6):
            cb.record_success()
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_trips_to_open_at_threshold(self):
        cb = CircuitBreaker("test", min_requests=10, failure_threshold=0.5)
        # 5 successes then 5 failures = 50% >= 50%
        for _ in range(5):
            cb.record_success()
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_does_not_trip_below_min_requests(self):
        cb = CircuitBreaker("test", min_requests=10, failure_threshold=0.5)
        # 5 failures out of 5 = 100% but below min_requests
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_allow_request_false_when_open(self):
        cb = CircuitBreaker("test", min_requests=10, failure_threshold=0.5)
        for _ in range(5):
            cb.record_success()
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    @patch("gateway.circuit_breaker.time.monotonic")
    def test_transitions_to_half_open_after_cooldown(self, mock_time):
        cb = CircuitBreaker("test", min_requests=10, failure_threshold=0.5, cooldown=30.0)
        mock_time.return_value = 100.0

        # Trip the breaker
        for _ in range(5):
            cb.record_success()
        for _ in range(5):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Advance past cooldown
        mock_time.return_value = 131.0
        assert cb.allow_request() is True
        assert cb.state == CircuitState.HALF_OPEN

    @patch("gateway.circuit_breaker.time.monotonic")
    def test_half_open_success_closes(self, mock_time):
        mock_time.return_value = 100.0
        cb = CircuitBreaker("test", min_requests=10, failure_threshold=0.5, cooldown=30.0)

        for _ in range(5):
            cb.record_success()
        for _ in range(5):
            cb.record_failure()

        mock_time.return_value = 131.0
        cb.allow_request()  # Triggers HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    @patch("gateway.circuit_breaker.time.monotonic")
    def test_half_open_failure_reopens(self, mock_time):
        mock_time.return_value = 100.0
        cb = CircuitBreaker("test", min_requests=10, failure_threshold=0.5, cooldown=30.0)

        for _ in range(5):
            cb.record_success()
        for _ in range(5):
            cb.record_failure()

        mock_time.return_value = 131.0
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    @patch("gateway.circuit_breaker.time.monotonic")
    def test_half_open_only_one_probe(self, mock_time):
        mock_time.return_value = 100.0
        cb = CircuitBreaker("test", min_requests=10, failure_threshold=0.5, cooldown=30.0)

        for _ in range(5):
            cb.record_success()
        for _ in range(5):
            cb.record_failure()

        mock_time.return_value = 131.0
        assert cb.allow_request() is True  # First: allowed (probe)
        assert cb.allow_request() is False  # Second: blocked


class TestExponentialBackoff:
    @patch("gateway.circuit_breaker.time.monotonic")
    def test_cooldown_doubles_on_repeated_failure(self, mock_time):
        mock_time.return_value = 100.0
        cb = CircuitBreaker("test", min_requests=10, failure_threshold=0.5, cooldown=30.0)

        # Trip
        for _ in range(5):
            cb.record_success()
        for _ in range(5):
            cb.record_failure()
        assert cb._current_cooldown == 30.0

        # First HALF_OPEN failure
        mock_time.return_value = 131.0
        cb.allow_request()
        cb.record_failure()
        assert cb._current_cooldown == 60.0

        # Second HALF_OPEN failure
        mock_time.return_value = 192.0
        cb.allow_request()
        cb.record_failure()
        assert cb._current_cooldown == 120.0

    @patch("gateway.circuit_breaker.time.monotonic")
    def test_cooldown_caps_at_max(self, mock_time):
        mock_time.return_value = 100.0
        cb = CircuitBreaker(
            "test",
            min_requests=10,
            failure_threshold=0.5,
            cooldown=30.0,
            max_cooldown=300.0,
        )

        for _ in range(5):
            cb.record_success()
        for _ in range(5):
            cb.record_failure()

        # Keep failing: 30 -> 60 -> 120 -> 240 -> 300 (cap)
        t = 131.0
        for expected in [60.0, 120.0, 240.0, 300.0]:
            mock_time.return_value = t
            cb.allow_request()
            cb.record_failure()
            assert cb._current_cooldown == expected
            t += expected + 1

        # One more: should stay at 300
        mock_time.return_value = t
        cb.allow_request()
        cb.record_failure()
        assert cb._current_cooldown == 300.0

    @patch("gateway.circuit_breaker.time.monotonic")
    def test_cooldown_resets_on_success(self, mock_time):
        mock_time.return_value = 100.0
        cb = CircuitBreaker("test", min_requests=10, failure_threshold=0.5, cooldown=30.0)

        for _ in range(5):
            cb.record_success()
        for _ in range(5):
            cb.record_failure()

        mock_time.return_value = 131.0
        cb.allow_request()
        cb.record_failure()
        assert cb._current_cooldown == 60.0

        mock_time.return_value = 192.0
        cb.allow_request()
        cb.record_success()
        assert cb._current_cooldown == 30.0  # Reset to initial


class TestRollingWindow:
    @patch("gateway.circuit_breaker.time.monotonic")
    def test_old_entries_pruned(self, mock_time):
        cb = CircuitBreaker("test", window_size=60.0, min_requests=10)

        # Add entries at t=0
        mock_time.return_value = 0.0
        for _ in range(10):
            cb.record_failure()

        # At t=61, window should be empty after prune
        mock_time.return_value = 61.0
        cb._prune_window()
        assert len(cb._requests) == 0


class TestCircuitBreakerRegistry:
    def test_get_open_backends(self):
        registry = CircuitBreakerRegistry(
            ["a", "b", "c"],
            min_requests=2,
            failure_threshold=0.5,
        )
        # Trip breaker for "a"
        for _ in range(2):
            registry.get("a").record_failure()

        open_backends = registry.get_open_backends()
        assert "a" in open_backends
        assert "b" not in open_backends
        assert "c" not in open_backends

    def test_get_all_snapshots(self):
        registry = CircuitBreakerRegistry(["a", "b"])
        snapshots = registry.get_all_snapshots()
        assert "a" in snapshots
        assert "b" in snapshots
        assert snapshots["a"]["state"] == "CLOSED"

    def test_sync_backends_adds_new(self):
        registry = CircuitBreakerRegistry(["a", "b"])
        registry.sync_backends(["a", "b", "c"])
        assert "c" in registry.breakers
        assert registry.get("c").state == CircuitState.CLOSED

    def test_sync_backends_removes_stale(self):
        registry = CircuitBreakerRegistry(["a", "b", "c"])
        registry.sync_backends(["a", "b"])
        assert "c" not in registry.breakers

    def test_sync_backends_preserves_existing_state(self):
        registry = CircuitBreakerRegistry(
            ["a", "b"],
            min_requests=2,
            failure_threshold=0.5,
        )
        # Trip "a"
        registry.get("a").record_failure()
        registry.get("a").record_failure()

        registry.sync_backends(["a", "b", "c"])
        assert registry.get("a").state == CircuitState.OPEN  # State preserved

    def test_snapshot_structure(self):
        registry = CircuitBreakerRegistry(["a"])
        snap = registry.get("a").snapshot()
        assert "state" in snap
        assert "error_rate" in snap
        assert "requests_in_window" in snap
        assert "current_cooldown_s" in snap
