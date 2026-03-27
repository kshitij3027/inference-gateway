import math

from gateway.routing import ConsistentHashRing


class TestConsistentHashRing:
    def test_deterministic(self):
        """Same key always returns same node."""
        ring = ConsistentHashRing([("a", 1), ("b", 1), ("c", 1)])
        results = [ring.get_node("test-key") for _ in range(100)]
        assert len(set(results)) == 1

    def test_different_keys_hit_multiple_nodes(self):
        """Different keys distribute across nodes."""
        ring = ConsistentHashRing([("a", 1), ("b", 1), ("c", 1)])
        nodes_hit = {ring.get_node(f"key-{i}") for i in range(100)}
        assert len(nodes_hit) >= 2  # At least 2 of 3 nodes hit

    def test_single_node_always_returns_it(self):
        ring = ConsistentHashRing([("only-node", 1)])
        for i in range(50):
            assert ring.get_node(f"key-{i}") == "only-node"

    def test_empty_ring_returns_none(self):
        ring = ConsistentHashRing([])
        assert ring.get_node("any-key") is None

    def test_weight_proportional_distribution(self):
        """Node with weight 3 should get ~75% of keys (within 5% tolerance)."""
        ring = ConsistentHashRing([("light", 1), ("heavy", 3)])
        counts: dict[str | None, int] = {"light": 0, "heavy": 0}
        total = 10000
        for i in range(total):
            node = ring.get_node(f"key-{i}")
            counts[node] += 1

        heavy_ratio = counts["heavy"] / total
        assert 0.70 <= heavy_ratio <= 0.80, f"heavy got {heavy_ratio:.2%}, expected ~75%"

    def test_add_backend_redistributes_less_than_1_over_n(self):
        """Adding a 4th node should move < ~33% of keys (1/3 + tolerance)."""
        ring_3 = ConsistentHashRing([("a", 1), ("b", 1), ("c", 1)])
        ring_4 = ConsistentHashRing([("a", 1), ("b", 1), ("c", 1), ("d", 1)])

        total = 1000
        changed = 0
        for i in range(total):
            key = f"key-{i}"
            if ring_3.get_node(key) != ring_4.get_node(key):
                changed += 1

        max_expected = math.ceil(total / 4) + total * 0.05  # 1/N + 5% tolerance
        assert changed <= max_expected, f"{changed} keys changed, expected <= {max_expected}"

    def test_remove_backend_only_redistributes_that_backends_keys(self):
        """Removing a node only moves keys that were on that node."""
        ring_3 = ConsistentHashRing([("a", 1), ("b", 1), ("c", 1)])
        ring_2 = ConsistentHashRing([("a", 1), ("b", 1)])

        for i in range(500):
            key = f"key-{i}"
            old_node = ring_3.get_node(key)
            new_node = ring_2.get_node(key)
            if old_node != "c":
                # Keys not on the removed node should stay put
                assert old_node == new_node, f"key-{i}: {old_node} -> {new_node}"

    def test_wrap_around(self):
        """Key with very high hash value wraps to first vnode."""
        ring = ConsistentHashRing([("node-a", 1), ("node-b", 1)])
        # Any key should resolve without error
        result = ring.get_node("wrap-test-key")
        assert result in ("node-a", "node-b")

    def test_exclude_skips_node(self):
        """Excluding a node routes all its keys to other nodes."""
        ring = ConsistentHashRing([("a", 1), ("b", 1)])
        for i in range(50):
            result = ring.get_node(f"key-{i}", exclude=frozenset({"a"}))
            assert result == "b"

    def test_exclude_all_returns_none(self):
        ring = ConsistentHashRing([("a", 1), ("b", 1)])
        result = ring.get_node("any-key", exclude=frozenset({"a", "b"}))
        assert result is None

    def test_node_count(self):
        ring = ConsistentHashRing([("a", 1), ("b", 2), ("c", 1)])
        assert ring.node_count == 3

    def test_vnode_count(self):
        ring = ConsistentHashRing([("a", 1), ("b", 2)], vnodes_per_unit=100)
        assert ring.vnode_count == 300  # 1*100 + 2*100

    def test_get_distribution(self):
        ring = ConsistentHashRing([("a", 1), ("b", 2)], vnodes_per_unit=100)
        dist = ring.get_distribution()
        assert dist["a"] == 100
        assert dist["b"] == 200
