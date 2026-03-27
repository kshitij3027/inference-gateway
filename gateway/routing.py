"""Consistent hash ring for backend load distribution.

Uses MD5 hashing with virtual nodes for even distribution.
Sorted list + bisect for O(log n) lookup.
"""

import bisect
import hashlib

VNODES_PER_UNIT = 150


class ConsistentHashRing:
    """Consistent hash ring with virtual nodes and weight support.

    Each node gets `weight * vnodes_per_unit` virtual nodes on the ring.
    Lookup is O(log n) via binary search on sorted positions.
    """

    def __init__(
        self,
        nodes: list[tuple[str, int]],
        vnodes_per_unit: int = VNODES_PER_UNIT,
    ) -> None:
        """Initialize the ring with nodes.

        Args:
            nodes: List of (node_name, weight) tuples.
            vnodes_per_unit: Virtual nodes per unit of weight.
        """
        self._ring: list[tuple[int, str]] = []
        self._node_names: set[str] = set()

        for name, weight in nodes:
            self._node_names.add(name)
            num_vnodes = weight * vnodes_per_unit
            for i in range(num_vnodes):
                vnode_key = f"{name}:{i}"
                position = self._hash(vnode_key)
                self._ring.append((position, name))

        self._ring.sort(key=lambda x: x[0])
        self._positions = [pos for pos, _ in self._ring]

    @staticmethod
    def _hash(key: str) -> int:
        """Hash a key to a 32-bit integer position on the ring."""
        digest = hashlib.md5(key.encode()).digest()  # noqa: S324
        return int.from_bytes(digest[:4], "big")

    def get_node(
        self,
        key: str,
        exclude: frozenset[str] = frozenset(),
    ) -> str | None:
        """Find the node responsible for the given key.

        Uses bisect to find the clockwise-nearest vnode, then walks
        clockwise skipping excluded nodes. Returns None if all nodes
        are excluded or the ring is empty.

        Args:
            key: The routing key to hash.
            exclude: Set of node names to skip (for failover).
        """
        if not self._ring:
            return None

        position = self._hash(key)
        start_idx = bisect.bisect_right(self._positions, position) % len(self._ring)

        # Walk clockwise, skipping excluded nodes
        for offset in range(len(self._ring)):
            idx = (start_idx + offset) % len(self._ring)
            _, node_name = self._ring[idx]
            if node_name not in exclude:
                return node_name

        return None  # All nodes excluded

    @property
    def node_count(self) -> int:
        """Number of distinct nodes on the ring."""
        return len(self._node_names)

    @property
    def vnode_count(self) -> int:
        """Total number of virtual nodes on the ring."""
        return len(self._ring)

    def get_distribution(self) -> dict[str, int]:
        """Count virtual nodes per node."""
        counts: dict[str, int] = {}
        for _, name in self._ring:
            counts[name] = counts.get(name, 0) + 1
        return counts
