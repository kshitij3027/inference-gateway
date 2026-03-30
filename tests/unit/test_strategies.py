from gateway.routing import ConsistentHashRing
from gateway.strategies import ConsistentHashStrategy


class TestConsistentHashStrategy:
    def _make_strategy(self) -> tuple[ConsistentHashStrategy, ConsistentHashRing]:
        ring = ConsistentHashRing([("a", 1), ("b", 1), ("c", 1)])
        return ConsistentHashStrategy(ring), ring

    def test_deterministic(self):
        """Same routing_key always returns the same backend."""
        strategy, _ = self._make_strategy()
        results = [
            strategy.select(["a", "b", "c"], routing_key="stable-key")
            for _ in range(100)
        ]
        assert len(set(results)) == 1

    def test_different_keys_distribute(self):
        """Different routing keys hit multiple backends."""
        strategy, _ = self._make_strategy()
        nodes_hit = {
            strategy.select(["a", "b", "c"], routing_key=f"key-{i}")
            for i in range(100)
        }
        assert len(nodes_hit) >= 2

    def test_exclude_skips_backend(self):
        """Excluded backend is never returned."""
        strategy, _ = self._make_strategy()
        for i in range(50):
            result = strategy.select(
                ["a", "b", "c"],
                exclude=frozenset({"a"}),
                routing_key=f"key-{i}",
            )
            assert result != "a"

    def test_exclude_all_returns_none(self):
        """All candidates excluded returns None."""
        strategy, _ = self._make_strategy()
        result = strategy.select(
            ["a", "b", "c"],
            exclude=frozenset({"a", "b", "c"}),
            routing_key="any-key",
        )
        assert result is None

    def test_no_routing_key_returns_first_candidate(self):
        """Without routing_key, returns first non-excluded candidate."""
        strategy, _ = self._make_strategy()
        result = strategy.select(["b", "a", "c"])
        assert result == "b"

    def test_no_routing_key_with_exclude(self):
        """Without routing_key, skips excluded and returns next candidate."""
        strategy, _ = self._make_strategy()
        result = strategy.select(["a", "b", "c"], exclude=frozenset({"a"}))
        assert result == "b"

    def test_empty_candidates(self):
        """Empty candidates list returns None."""
        strategy, _ = self._make_strategy()
        result = strategy.select([])
        assert result is None

    def test_matches_ring_behavior(self):
        """Strategy select produces identical results to ring get_node."""
        strategy, ring = self._make_strategy()
        for i in range(200):
            key = f"verify-{i}"
            ring_result = ring.get_node(key)
            strategy_result = strategy.select(
                ["a", "b", "c"], routing_key=key
            )
            assert strategy_result == ring_result, (
                f"key={key}: ring={ring_result}, strategy={strategy_result}"
            )

        # Also verify with exclusions
        for i in range(50):
            key = f"exclude-verify-{i}"
            exclude = frozenset({"a"})
            ring_result = ring.get_node(key, exclude=exclude)
            strategy_result = strategy.select(
                ["a", "b", "c"], exclude=exclude, routing_key=key
            )
            assert strategy_result == ring_result, (
                f"key={key}: ring={ring_result}, strategy={strategy_result}"
            )
