"""Routing strategy protocol and implementations.

Strategies determine which backend handles a given request for a model.
"""

from __future__ import annotations

from typing import Protocol

from gateway.routing import ConsistentHashRing


class RoutingStrategy(Protocol):
    """Protocol for backend selection strategies."""

    def select(
        self,
        candidates: list[str],
        exclude: frozenset[str] = frozenset(),
        routing_key: str | None = None,
    ) -> str | None:
        """Select a backend name from candidates.

        Args:
            candidates: Backend names serving this model.
            exclude: Backend names to skip (circuit-broken).
            routing_key: Optional key for deterministic routing.

        Returns:
            Selected backend name, or None if all excluded/empty.
        """
        ...


class ConsistentHashStrategy:
    """Wraps ConsistentHashRing to implement RoutingStrategy."""

    def __init__(self, ring: ConsistentHashRing) -> None:
        self._ring = ring

    def select(
        self,
        candidates: list[str],
        exclude: frozenset[str] = frozenset(),
        routing_key: str | None = None,
    ) -> str | None:
        if routing_key is not None:
            return self._ring.get_node(routing_key, exclude=exclude)
        # Fallback: first non-excluded candidate
        for name in candidates:
            if name not in exclude:
                return name
        return None
