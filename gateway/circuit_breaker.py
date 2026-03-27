"""Per-backend circuit breaker with rolling window error tracking.

States: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
- CLOSED: requests flow normally, failures tracked in rolling window
- OPEN: requests blocked, waiting for cooldown to expire
- HALF_OPEN: one probe request allowed, success->CLOSED, failure->OPEN with doubled cooldown
"""

import time
from collections import deque
from enum import Enum

import structlog

logger = structlog.get_logger()


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Per-backend circuit breaker with rolling window and exponential backoff."""

    def __init__(
        self,
        backend_name: str,
        window_size: float = 60.0,
        failure_threshold: float = 0.5,
        min_requests: int = 10,
        cooldown: float = 30.0,
        max_cooldown: float = 300.0,
    ) -> None:
        self.backend_name = backend_name
        self.state = CircuitState.CLOSED
        self.window_size = window_size
        self.failure_threshold = failure_threshold
        self.min_requests = min_requests
        self._initial_cooldown = cooldown
        self._current_cooldown = cooldown
        self.max_cooldown = max_cooldown
        self._opened_at: float | None = None
        self._requests: deque[tuple[float, bool]] = deque()
        self._half_open_probe_sent = False

    def allow_request(self) -> bool:
        """Check if a request should be allowed through this backend."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = time.monotonic() - (self._opened_at or 0)
            if elapsed >= self._current_cooldown:
                self._transition(CircuitState.HALF_OPEN)
                self._half_open_probe_sent = True
                return True  # Allow one probe
            return False

        if self.state == CircuitState.HALF_OPEN:
            if not self._half_open_probe_sent:
                self._half_open_probe_sent = True
                return True
            return False  # Only one probe at a time

        return True

    def record_success(self) -> None:
        """Record a successful request."""
        self._requests.append((time.monotonic(), True))
        if self.state == CircuitState.HALF_OPEN:
            self._transition(CircuitState.CLOSED)
            self._current_cooldown = self._initial_cooldown
            self._half_open_probe_sent = False

    def record_failure(self) -> None:
        """Record a failed request (5xx, timeout, connection error)."""
        self._requests.append((time.monotonic(), False))
        if self.state == CircuitState.HALF_OPEN:
            self._current_cooldown = min(
                self._current_cooldown * 2, self.max_cooldown
            )
            self._half_open_probe_sent = False
            self._transition(CircuitState.OPEN)
        elif self.state == CircuitState.CLOSED:
            if self._should_trip():
                self._transition(CircuitState.OPEN)

    def _should_trip(self) -> bool:
        """Check if error rate exceeds threshold in rolling window."""
        self._prune_window()
        total = len(self._requests)
        if total < self.min_requests:
            return False
        failures = sum(1 for _, success in self._requests if not success)
        return (failures / total) >= self.failure_threshold

    def _prune_window(self) -> None:
        """Remove entries older than window_size."""
        cutoff = time.monotonic() - self.window_size
        while self._requests and self._requests[0][0] <= cutoff:
            self._requests.popleft()

    def _transition(self, new_state: CircuitState) -> None:
        """Transition to a new state with logging."""
        old_state = self.state
        self.state = new_state
        if new_state == CircuitState.OPEN:
            self._opened_at = time.monotonic()
        logger.info(
            "circuit_state_change",
            backend=self.backend_name,
            old_state=old_state.value,
            new_state=new_state.value,
            current_cooldown=self._current_cooldown,
        )

    def snapshot(self) -> dict:
        """Return current state for admin endpoint."""
        self._prune_window()
        total = len(self._requests)
        failures = sum(1 for _, success in self._requests if not success)
        return {
            "state": self.state.value,
            "error_rate": round(failures / total, 2) if total > 0 else 0.0,
            "requests_in_window": total,
            "current_cooldown_s": self._current_cooldown,
        }


class CircuitBreakerRegistry:
    """Manages circuit breakers for all backends."""

    def __init__(self, backend_names: list[str], **kwargs) -> None:
        """Create a circuit breaker for each backend.

        kwargs are forwarded to CircuitBreaker constructor (for testing overrides).
        """
        self._kwargs = kwargs
        self.breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker(name, **kwargs) for name in backend_names
        }

    def get(self, backend_name: str) -> CircuitBreaker | None:
        """Get circuit breaker for a backend."""
        return self.breakers.get(backend_name)

    def get_open_backends(self) -> frozenset[str]:
        """Return names of backends that should be excluded from routing.

        Note: Calling allow_request() has the side effect of transitioning
        OPEN -> HALF_OPEN when cooldown expires. This is intentional -- passive probing.
        """
        return frozenset(
            name
            for name, cb in self.breakers.items()
            if not cb.allow_request()
        )

    def get_all_snapshots(self) -> dict[str, dict]:
        """Get all circuit breaker states for admin endpoint."""
        return {name: cb.snapshot() for name, cb in self.breakers.items()}

    def sync_backends(self, backend_names: list[str]) -> None:
        """Sync circuit breakers with current config.

        Adds breakers for new backends, removes stale ones.
        Preserves state for existing backends.
        """
        desired = set(backend_names)
        current = set(self.breakers.keys())

        for name in desired - current:
            self.breakers[name] = CircuitBreaker(name, **self._kwargs)

        for name in current - desired:
            del self.breakers[name]
