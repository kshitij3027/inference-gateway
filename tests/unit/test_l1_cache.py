from unittest.mock import patch

from gateway.l1_cache import L1Cache


class TestL1Lookup:
    def test_hit_above_threshold(self):
        cache = L1Cache()
        emb = [1.0, 0.0, 0.0]
        cache.store("e1", "scope:gpt4", emb, '{"response": "hello"}')

        result, sim = cache.lookup("scope:gpt4", [1.0, 0.0, 0.0], threshold=0.9)
        assert result == '{"response": "hello"}'
        assert sim is not None
        assert sim >= 0.9

    def test_miss_below_threshold(self):
        cache = L1Cache()
        cache.store("e1", "scope:gpt4", [1.0, 0.0, 0.0], '{"response": "hello"}')

        result, sim = cache.lookup("scope:gpt4", [0.0, 1.0, 0.0], threshold=0.9)
        assert result is None
        assert sim is None

    def test_miss_empty_scope(self):
        cache = L1Cache()
        cache.store("e1", "scope:gpt4", [1.0, 0.0, 0.0], '{"response": "hello"}')

        result, sim = cache.lookup("scope:other", [1.0, 0.0, 0.0], threshold=0.9)
        assert result is None
        assert sim is None

    def test_best_match_selected(self):
        cache = L1Cache()
        cache.store("e1", "scope:gpt4", [1.0, 0.0, 0.0], '{"response": "first"}')
        cache.store("e2", "scope:gpt4", [0.9, 0.1, 0.0], '{"response": "second"}')

        # Query closer to e2
        result, sim = cache.lookup("scope:gpt4", [0.85, 0.15, 0.0], threshold=0.8)
        assert result == '{"response": "second"}'

    def test_lru_touch_on_hit(self):
        cache = L1Cache(max_entries=3)
        cache.store("a", "s", [1.0, 0.0, 0.0], "resp_a")
        cache.store("b", "s", [0.0, 1.0, 0.0], "resp_b")
        cache.store("c", "s", [0.0, 0.0, 1.0], "resp_c")

        # Hit a — moves it to the end
        cache.lookup("s", [1.0, 0.0, 0.0], threshold=0.9)

        # Evict oldest — should be b (a was touched, moved to end)
        cache._evict_oldest()
        assert "b" not in cache._entries
        assert "a" in cache._entries
        assert "c" in cache._entries

    @patch("gateway.l1_cache.time")
    def test_expired_entry_evicted(self, mock_time):
        mock_time.monotonic.return_value = 100.0
        cache = L1Cache(ttl_seconds=60.0)

        cache.store("e1", "scope:gpt4", [1.0, 0.0, 0.0], '{"response": "hello"}')

        # Advance past TTL
        mock_time.monotonic.return_value = 161.0
        result, sim = cache.lookup("scope:gpt4", [1.0, 0.0, 0.0], threshold=0.9)
        assert result is None
        assert sim is None
        # Entry should have been evicted
        assert "e1" not in cache._entries
        assert "scope:gpt4" not in cache._scope_index


class TestL1Store:
    def test_store_and_retrieve(self):
        cache = L1Cache()
        emb = [1.0, 0.0, 0.0]
        cache.store("e1", "scope:gpt4", emb, '{"result": "ok"}')

        result, sim = cache.lookup("scope:gpt4", emb, threshold=0.9)
        assert result == '{"result": "ok"}'
        assert sim is not None

    def test_dedup_existing_entry(self):
        cache = L1Cache()
        cache.store("e1", "scope:gpt4", [1.0, 0.0, 0.0], "first")
        cache.store("e1", "scope:gpt4", [1.0, 0.0, 0.0], "second")

        assert len(cache._entries) == 1
        # Original response preserved on dedup (move_to_end, not replace)
        assert cache._entries["e1"].response_json == "first"

    def test_capacity_eviction(self):
        cache = L1Cache(max_entries=3)
        cache.store("e1", "s", [1.0, 0.0, 0.0], "r1")
        cache.store("e2", "s", [0.0, 1.0, 0.0], "r2")
        cache.store("e3", "s", [0.0, 0.0, 1.0], "r3")

        # This should evict e1 (oldest)
        cache.store("e4", "s", [0.5, 0.5, 0.0], "r4")

        assert len(cache._entries) == 3
        assert "e1" not in cache._entries
        assert "e4" in cache._entries

    def test_scope_index_populated(self):
        cache = L1Cache()
        cache.store("e1", "scope:gpt4", [1.0, 0.0, 0.0], "resp")

        assert "scope:gpt4" in cache._scope_index
        assert "e1" in cache._scope_index["scope:gpt4"]


class TestL1Eviction:
    def test_evict_oldest_removes_front(self):
        cache = L1Cache(max_entries=3)
        cache.store("e1", "s", [1.0, 0.0, 0.0], "r1")
        cache.store("e2", "s", [0.0, 1.0, 0.0], "r2")
        cache.store("e3", "s", [0.0, 0.0, 1.0], "r3")

        cache._evict_oldest()
        assert "e1" not in cache._entries
        assert len(cache._entries) == 2

    def test_evict_cleans_scope_index(self):
        cache = L1Cache()
        cache.store("e1", "scope:gpt4", [1.0, 0.0, 0.0], "resp")
        cache.store("e2", "scope:gpt4", [0.0, 1.0, 0.0], "resp2")

        cache._evict("e1")
        assert "e1" not in cache._scope_index.get("scope:gpt4", set())
        # scope still exists because e2 is there
        assert "scope:gpt4" in cache._scope_index

    def test_evict_removes_empty_scope(self):
        cache = L1Cache()
        cache.store("e1", "scope:gpt4", [1.0, 0.0, 0.0], "resp")

        cache._evict("e1")
        assert "scope:gpt4" not in cache._scope_index


class TestL1Stats:
    def test_hit_miss_counters(self):
        cache = L1Cache()
        cache.store("e1", "scope:gpt4", [1.0, 0.0, 0.0], "resp")

        # Hit
        cache.lookup("scope:gpt4", [1.0, 0.0, 0.0], threshold=0.9)
        # Miss (different scope)
        cache.lookup("scope:other", [1.0, 0.0, 0.0], threshold=0.9)
        # Miss (below threshold)
        cache.lookup("scope:gpt4", [0.0, 1.0, 0.0], threshold=0.9)

        s = cache.stats()
        assert s["l1_hits"] == 1
        assert s["l1_misses"] == 2
        assert s["l1_entries"] == 1

    def test_flush_returns_count(self):
        cache = L1Cache()
        cache.store("e1", "s", [1.0, 0.0, 0.0], "r1")
        cache.store("e2", "s", [0.0, 1.0, 0.0], "r2")
        cache.store("e3", "s", [0.0, 0.0, 1.0], "r3")

        count = cache.flush()
        assert count == 3
        assert len(cache._entries) == 0
        assert len(cache._scope_index) == 0

        s = cache.stats()
        assert s["l1_hits"] == 0
        assert s["l1_misses"] == 0
        assert s["l1_entries"] == 0
