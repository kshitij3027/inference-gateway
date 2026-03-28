import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.models import ChatCompletionResponse, ChatMessage, ChatMessageResponse, Choice, Usage
from gateway.semantic_cache import SemanticCache


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert SemanticCache.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert SemanticCache.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert SemanticCache.cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_similar_vectors(self):
        a = [1.0, 0.1, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = SemanticCache.cosine_similarity(a, b)
        assert 0.9 < sim < 1.0

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert SemanticCache.cosine_similarity(a, b) == 0.0


class TestExtractUserText:
    def test_single_user_message(self):
        messages = [ChatMessage(role="user", content="hello")]
        assert SemanticCache._extract_user_text(messages) == "hello"

    def test_multiple_user_messages(self):
        messages = [
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi"),
            ChatMessage(role="user", content="world"),
        ]
        assert SemanticCache._extract_user_text(messages) == "hello\nworld"

    def test_ignores_system_and_assistant(self):
        messages = [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="assistant", content="asst"),
            ChatMessage(role="user", content="user_msg"),
        ]
        assert SemanticCache._extract_user_text(messages) == "user_msg"


class TestExtractSystemHash:
    def test_no_system_prompt(self):
        messages = [ChatMessage(role="user", content="hello")]
        h = SemanticCache._extract_system_hash(messages)
        # Hash of empty string
        expected = hashlib.sha256(b"").hexdigest()[:16]
        assert h == expected

    def test_different_system_prompts(self):
        m1 = [ChatMessage(role="system", content="be a teacher")]
        m2 = [ChatMessage(role="system", content="be a comedian")]
        assert SemanticCache._extract_system_hash(m1) != SemanticCache._extract_system_hash(m2)

    def test_same_system_prompt(self):
        m1 = [ChatMessage(role="system", content="be a teacher")]
        m2 = [ChatMessage(role="system", content="be a teacher")]
        assert SemanticCache._extract_system_hash(m1) == SemanticCache._extract_system_hash(m2)


class TestBuildScopeKey:
    def _make_cache(self):
        return SemanticCache(redis_client=AsyncMock())

    def test_shared_mode(self):
        cache = self._make_cache()
        key = cache._build_scope_key("gpt-4", "abc123", "tenant-a", "shared")
        assert key == "cache:scope:gpt-4:abc123"
        assert "tenant-a" not in key

    def test_tenant_isolated_mode(self):
        cache = self._make_cache()
        key = cache._build_scope_key("gpt-4", "abc123", "tenant-a", "tenant")
        assert key == "cache:scope:tenant-a:gpt-4:abc123"


def _make_response(model="test-model", content="test response"):
    return ChatCompletionResponse(
        model=model,
        choices=[Choice(message=ChatMessageResponse(content=content))],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class TestLookup:
    @pytest.fixture
    def cache(self):
        redis_mock = AsyncMock()
        c = SemanticCache(redis_mock, similarity_threshold=0.95)
        # Mock compute_embedding to return a fixed vector
        c.compute_embedding = MagicMock(return_value=[1.0, 0.0, 0.0])
        return c

    async def test_hit_above_threshold(self, cache):
        response = _make_response()
        entry_data = {
            "embedding": json.dumps([1.0, 0.0, 0.0]),  # identical
            "response": response.model_dump_json(),
            "entry_id": "abc123",
        }
        cache.redis.smembers = AsyncMock(return_value={"abc123"})
        cache.redis.hgetall = AsyncMock(return_value=entry_data)
        cache.redis.hincrby = AsyncMock()

        messages = [ChatMessage(role="user", content="hello")]
        result, similarity = await cache.lookup("test-model", messages, "t1", "shared")
        assert result is not None
        assert similarity == pytest.approx(1.0)

    async def test_miss_below_threshold(self, cache):
        response = _make_response()
        entry_data = {
            "embedding": json.dumps([0.0, 1.0, 0.0]),  # orthogonal
            "response": response.model_dump_json(),
            "entry_id": "abc123",
        }
        cache.redis.smembers = AsyncMock(return_value={"abc123"})
        cache.redis.hgetall = AsyncMock(return_value=entry_data)

        messages = [ChatMessage(role="user", content="hello")]
        result, similarity = await cache.lookup("test-model", messages, "t1", "shared")
        assert result is None

    async def test_miss_empty_scope(self, cache):
        cache.redis.smembers = AsyncMock(return_value=set())
        messages = [ChatMessage(role="user", content="hello")]
        result, similarity = await cache.lookup("test-model", messages, "t1", "shared")
        assert result is None

    async def test_model_scoping(self, cache):
        """Entry for model A should not be returned when looking up model B."""
        # Scope for model-b is empty
        cache.redis.smembers = AsyncMock(return_value=set())
        messages = [ChatMessage(role="user", content="hello")]
        result, _ = await cache.lookup("model-b", messages, "t1", "shared")
        assert result is None

    async def test_expired_entry_cleaned_up(self, cache):
        cache.redis.smembers = AsyncMock(return_value={"expired-id"})
        cache.redis.hgetall = AsyncMock(return_value={})  # expired
        cache.redis.srem = AsyncMock()

        messages = [ChatMessage(role="user", content="hello")]
        result, _ = await cache.lookup("test-model", messages, "t1", "shared")
        assert result is None
        cache.redis.srem.assert_called_once()

    async def test_no_user_text_returns_none(self, cache):
        """Messages with no user role should return None immediately."""
        messages = [ChatMessage(role="system", content="sys")]
        result, similarity = await cache.lookup("test-model", messages, "t1", "shared")
        assert result is None
        assert similarity is None

    async def test_best_match_selected(self, cache):
        """When multiple entries exist, the highest similarity is returned."""
        response_close = _make_response(content="close match")
        response_exact = _make_response(content="exact match")

        entry_close = {
            "embedding": json.dumps([0.96, 0.28, 0.0]),  # similar but not identical
            "response": response_close.model_dump_json(),
            "entry_id": "close-id",
        }
        entry_exact = {
            "embedding": json.dumps([1.0, 0.0, 0.0]),  # identical
            "response": response_exact.model_dump_json(),
            "entry_id": "exact-id",
        }

        cache.redis.smembers = AsyncMock(return_value={"close-id", "exact-id"})

        async def hgetall_side_effect(key):
            if "close-id" in key:
                return entry_close
            return entry_exact

        cache.redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)
        cache.redis.hincrby = AsyncMock()

        messages = [ChatMessage(role="user", content="hello")]
        result, similarity = await cache.lookup("test-model", messages, "t1", "shared")
        assert result is not None
        assert similarity == pytest.approx(1.0)

    async def test_hit_increments_hit_count(self, cache):
        """A cache hit should increment the hit_count on the matched entry."""
        response = _make_response()
        entry_data = {
            "embedding": json.dumps([1.0, 0.0, 0.0]),
            "response": response.model_dump_json(),
            "entry_id": "abc123",
        }
        cache.redis.smembers = AsyncMock(return_value={"abc123"})
        cache.redis.hgetall = AsyncMock(return_value=entry_data)
        cache.redis.hincrby = AsyncMock()

        messages = [ChatMessage(role="user", content="hello")]
        await cache.lookup("test-model", messages, "t1", "shared")
        cache.redis.hincrby.assert_called_once_with("cache:entry:abc123", "hit_count", 1)


class TestStore:
    async def test_stores_entry_and_index(self):
        redis_mock = AsyncMock()
        pipe_mock = AsyncMock()
        redis_mock.pipeline = MagicMock(return_value=pipe_mock)

        cache = SemanticCache(redis_mock, default_ttl=3600)
        cache.compute_embedding = MagicMock(return_value=[1.0, 0.0, 0.0])

        response = _make_response()
        messages = [ChatMessage(role="user", content="hello")]
        await cache.store("test-model", messages, response, "t1", "shared")

        # Verify pipeline was used
        pipe_mock.hset.assert_called_once()
        pipe_mock.expire.assert_called()
        pipe_mock.sadd.assert_called_once()
        pipe_mock.execute.assert_called_once()

    async def test_sets_ttl(self):
        redis_mock = AsyncMock()
        pipe_mock = AsyncMock()
        redis_mock.pipeline = MagicMock(return_value=pipe_mock)

        cache = SemanticCache(redis_mock, default_ttl=7200)
        cache.compute_embedding = MagicMock(return_value=[1.0, 0.0, 0.0])

        response = _make_response()
        messages = [ChatMessage(role="user", content="hello")]
        await cache.store("test-model", messages, response, "t1", "shared")

        # Check TTL was set to 7200
        expire_calls = pipe_mock.expire.call_args_list
        assert any(call.args[1] == 7200 for call in expire_calls)

    async def test_skips_empty_user_text(self):
        """Store should be a no-op when there are no user messages."""
        redis_mock = AsyncMock()
        cache = SemanticCache(redis_mock)
        cache.compute_embedding = MagicMock()

        response = _make_response()
        messages = [ChatMessage(role="system", content="sys")]
        await cache.store("test-model", messages, response, "t1", "shared")

        cache.compute_embedding.assert_not_called()
        redis_mock.pipeline.assert_not_called()

    async def test_entry_data_contains_required_fields(self):
        redis_mock = AsyncMock()
        pipe_mock = AsyncMock()
        redis_mock.pipeline = MagicMock(return_value=pipe_mock)

        cache = SemanticCache(redis_mock)
        cache.compute_embedding = MagicMock(return_value=[1.0, 0.0, 0.0])

        response = _make_response()
        messages = [ChatMessage(role="user", content="hello")]
        await cache.store("test-model", messages, response, "t1", "shared")

        # Inspect the mapping passed to hset
        call_kwargs = pipe_mock.hset.call_args
        mapping = call_kwargs.kwargs.get("mapping") or call_kwargs[1].get("mapping")
        assert "embedding" in mapping
        assert "response" in mapping
        assert "model" in mapping
        assert "tenant_id" in mapping
        assert "entry_id" in mapping
        assert "created_at" in mapping
        assert mapping["hit_count"] == "0"


class TestTenantIsolation:
    @pytest.fixture
    def cache(self):
        redis_mock = AsyncMock()
        c = SemanticCache(redis_mock, similarity_threshold=0.95)
        c.compute_embedding = MagicMock(return_value=[1.0, 0.0, 0.0])
        return c

    async def test_shared_mode_cross_tenant(self, cache):
        """In shared mode, tenant A's cache entry is visible to tenant B."""
        response = _make_response()
        entry_data = {
            "embedding": json.dumps([1.0, 0.0, 0.0]),
            "response": response.model_dump_json(),
            "entry_id": "abc123",
        }
        cache.redis.smembers = AsyncMock(return_value={"abc123"})
        cache.redis.hgetall = AsyncMock(return_value=entry_data)
        cache.redis.hincrby = AsyncMock()

        messages = [ChatMessage(role="user", content="hello")]
        # Both tenants use same scope key in shared mode
        result_a, _ = await cache.lookup("test-model", messages, "tenant-a", "shared")
        result_b, _ = await cache.lookup("test-model", messages, "tenant-b", "shared")
        assert result_a is not None
        assert result_b is not None

    async def test_isolated_mode_no_cross_tenant(self, cache):
        """In tenant mode, different tenants have different scope keys."""
        messages = [ChatMessage(role="user", content="hello")]

        # Simulate: tenant-a has entries, tenant-b does not
        async def smembers_side_effect(key):
            if "tenant-a" in key:
                return {"abc123"}
            return set()

        cache.redis.smembers = AsyncMock(side_effect=smembers_side_effect)

        response = _make_response()
        entry_data = {
            "embedding": json.dumps([1.0, 0.0, 0.0]),
            "response": response.model_dump_json(),
            "entry_id": "abc123",
        }
        cache.redis.hgetall = AsyncMock(return_value=entry_data)
        cache.redis.hincrby = AsyncMock()

        result_a, _ = await cache.lookup("test-model", messages, "tenant-a", "tenant")
        result_b, _ = await cache.lookup("test-model", messages, "tenant-b", "tenant")
        assert result_a is not None
        assert result_b is None


class TestStampedeGuard:
    async def test_lock_acquired(self):
        redis_mock = AsyncMock()
        redis_mock.set = AsyncMock(return_value=True)
        cache = SemanticCache(redis_mock)

        messages = [ChatMessage(role="user", content="hello")]
        acquired, lock_key = await cache.acquire_stampede_lock("model", messages)
        assert acquired is True
        assert lock_key.startswith("cache:lock:")

    async def test_lock_denied(self):
        redis_mock = AsyncMock()
        redis_mock.set = AsyncMock(return_value=False)
        cache = SemanticCache(redis_mock)

        messages = [ChatMessage(role="user", content="hello")]
        acquired, lock_key = await cache.acquire_stampede_lock("model", messages)
        assert acquired is False

    async def test_release_lock(self):
        redis_mock = AsyncMock()
        cache = SemanticCache(redis_mock)
        await cache.release_stampede_lock("cache:lock:abc")
        redis_mock.delete.assert_called_once_with("cache:lock:abc")


class TestStats:
    async def test_get_stats(self):
        redis_mock = AsyncMock()
        redis_mock.hgetall = AsyncMock(return_value={"hits": "10", "misses": "90"})

        # Mock scan_iter
        async def mock_scan_iter(**kwargs):
            for key in ["cache:entry:1", "cache:entry:2", "cache:entry:3"]:
                yield key

        redis_mock.scan_iter = mock_scan_iter

        cache = SemanticCache(redis_mock)
        stats = await cache.get_stats()
        assert stats["hits"] == 10
        assert stats["misses"] == 90
        assert stats["hit_rate"] == 0.1
        assert stats["entries"] == 3

    async def test_get_stats_no_data(self):
        redis_mock = AsyncMock()
        redis_mock.hgetall = AsyncMock(return_value={})

        async def mock_scan_iter(**kwargs):
            return
            yield  # make it an async generator

        redis_mock.scan_iter = mock_scan_iter

        cache = SemanticCache(redis_mock)
        stats = await cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["entries"] == 0

    async def test_flush(self):
        redis_mock = AsyncMock()

        async def mock_scan_iter(**kwargs):
            for key in ["cache:entry:1", "cache:scope:m:h", "cache:stats"]:
                yield key

        redis_mock.scan_iter = mock_scan_iter
        redis_mock.delete = AsyncMock(return_value=3)

        cache = SemanticCache(redis_mock)
        count = await cache.flush()
        assert count == 3

    async def test_flush_no_keys(self):
        redis_mock = AsyncMock()

        async def mock_scan_iter(**kwargs):
            return
            yield  # make it an async generator

        redis_mock.scan_iter = mock_scan_iter

        cache = SemanticCache(redis_mock)
        count = await cache.flush()
        assert count == 0

    async def test_record_hit(self):
        redis_mock = AsyncMock()
        cache = SemanticCache(redis_mock)
        await cache.record_hit()
        redis_mock.hincrby.assert_called_once_with("cache:stats", "hits", 1)

    async def test_record_miss(self):
        redis_mock = AsyncMock()
        cache = SemanticCache(redis_mock)
        await cache.record_miss()
        redis_mock.hincrby.assert_called_once_with("cache:stats", "misses", 1)
