"""Semantic response cache using sentence-transformers embeddings and Redis.

Computes 384-dimensional embeddings via all-MiniLM-L6-v2, stores in Redis,
and uses cosine similarity (threshold 0.95) for semantic matching.
"""

import asyncio
import hashlib
import json
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np
import structlog

from gateway.l1_cache import L1Cache

if TYPE_CHECKING:
    from gateway.models import ChatCompletionResponse, ChatMessage

logger = structlog.get_logger()


class SemanticCache:
    """Redis-backed semantic cache with embedding-based similarity lookup."""

    def __init__(
        self,
        redis_client,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.95,
        default_ttl: int = 3600,
        l1_max_entries: int = 500,
        l1_ttl: int | None = None,
    ) -> None:
        self.redis = redis_client
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        self._model = None  # Lazy loaded
        self._prefix_cache: dict[str, list[float]] = {}  # sys_hash → system prompt embedding
        self._l1 = L1Cache(
            max_entries=l1_max_entries,
            ttl_seconds=float(l1_ttl if l1_ttl is not None else default_ttl),
        )

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            logger.info("embedding_model_loaded", model=self.model_name)

    def compute_embedding(self, text: str) -> list[float]:
        """Compute a 384-dim embedding for the given text."""
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    @staticmethod
    def _extract_user_text(messages: list) -> str:
        """Concatenate all user-role message contents."""
        return "\n".join(m.content for m in messages if m.role == "user")

    @staticmethod
    def _extract_system_hash(messages: list) -> str:
        """SHA256 hash of concatenated system-role message contents."""
        system_text = "\n".join(m.content for m in messages if m.role == "system")
        return hashlib.sha256(system_text.encode()).hexdigest()[:16]

    def compute_system_embedding(self, messages: list) -> tuple[str, list[float]]:
        """Compute and cache the system prompt embedding.

        Returns (sys_hash, embedding). The embedding is cached by sys_hash
        so repeated system prompts avoid redundant embedding computation.
        """
        system_text = "\n".join(m.content for m in messages if m.role == "system")
        sys_hash = self._extract_system_hash(messages)

        if sys_hash not in self._prefix_cache and system_text:
            self._prefix_cache[sys_hash] = self.compute_embedding(system_text)

        return sys_hash, self._prefix_cache.get(sys_hash, [])

    def _build_scope_key(
        self, model: str, sys_hash: str, tenant_id: str, cache_isolation: str
    ) -> str:
        """Build the Redis scope key for cache partitioning."""
        if cache_isolation == "tenant":
            return f"cache:scope:{tenant_id}:{model}:{sys_hash}"
        return f"cache:scope:{model}:{sys_hash}"

    def _compute_prompt_hash(self, model: str, messages: list) -> str:
        """Compute a hash for stampede lock grouping."""
        user_text = self._extract_user_text(messages)
        sys_hash = self._extract_system_hash(messages)
        raw = f"{model}:{sys_hash}:{user_text}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    async def lookup(
        self,
        model: str,
        messages: list,
        tenant_id: str,
        cache_isolation: str = "shared",
    ) -> tuple["ChatCompletionResponse | None", float | None, str | None]:
        """Look up a semantically similar cached response.

        Returns (ChatCompletionResponse, similarity, tier) where tier is
        "L1_HIT", "L2_HIT", or None (miss).
        """
        from gateway.models import ChatCompletionResponse

        user_text = self._extract_user_text(messages)
        if not user_text:
            return None, None, None

        sys_hash, _sys_embedding = self.compute_system_embedding(messages)
        scope_key = self._build_scope_key(model, sys_hash, tenant_id, cache_isolation)

        # Compute query embedding
        query_embedding = self.compute_embedding(user_text)

        # Check L1 first
        l1_response_json, l1_similarity = self._l1.lookup(
            scope_key, query_embedding, self.similarity_threshold
        )
        if l1_response_json is not None:
            response = ChatCompletionResponse.model_validate_json(l1_response_json)
            return response, l1_similarity, "L1_HIT"

        # Check L2 (Redis)
        entry_ids = await self.redis.smembers(scope_key)
        if not entry_ids:
            return None, None, None

        # Batch-fetch embeddings and find best match
        best_similarity = -1.0
        best_entry_data: dict | None = None

        for entry_id in entry_ids:
            entry_key = f"cache:entry:{entry_id}"
            entry_data = await self.redis.hgetall(entry_key)
            if not entry_data:
                # Entry expired but still in scope set — clean up
                await self.redis.srem(scope_key, entry_id)
                continue

            stored_embedding = json.loads(entry_data["embedding"])
            similarity = self.cosine_similarity(query_embedding, stored_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_entry_data = entry_data

        if best_entry_data is not None and best_similarity >= self.similarity_threshold:
            response = ChatCompletionResponse.model_validate_json(best_entry_data["response"])
            entry_id = best_entry_data.get("entry_id", "")
            if entry_id:
                await self.redis.hincrby(f"cache:entry:{entry_id}", "hit_count", 1)
                # Promote to L1 on L2 hit
                stored_embedding = json.loads(best_entry_data["embedding"])
                self._l1.store(
                    entry_id=entry_id,
                    scope_key=scope_key,
                    embedding=stored_embedding,
                    response_json=best_entry_data["response"],
                )
            return response, round(best_similarity, 4), "L2_HIT"

        return None, None, None

    async def store(
        self,
        model: str,
        messages: list,
        response: "ChatCompletionResponse",
        tenant_id: str,
        cache_isolation: str = "shared",
    ) -> None:
        """Store a response in the cache with its embedding."""
        user_text = self._extract_user_text(messages)
        if not user_text:
            return

        sys_hash, _sys_embedding = self.compute_system_embedding(messages)
        scope_key = self._build_scope_key(model, sys_hash, tenant_id, cache_isolation)

        embedding = self.compute_embedding(user_text)
        entry_id = uuid.uuid4().hex[:16]
        entry_key = f"cache:entry:{entry_id}"

        entry_data = {
            "embedding": json.dumps(embedding),
            "response": response.model_dump_json(),
            "model": model,
            "sys_hash": sys_hash,
            "tenant_id": tenant_id,
            "created_at": str(time.time()),
            "hit_count": "0",
            "entry_id": entry_id,
        }

        pipe = self.redis.pipeline()
        pipe.hset(entry_key, mapping=entry_data)
        pipe.expire(entry_key, self.default_ttl)
        pipe.sadd(scope_key, entry_id)
        pipe.expire(scope_key, self.default_ttl)
        await pipe.execute()

        # Also store in L1
        self._l1.store(
            entry_id=entry_id,
            scope_key=scope_key,
            embedding=embedding,
            response_json=response.model_dump_json(),
        )

    async def record_hit(self) -> None:
        """Increment the global cache hit counter."""
        await self.redis.hincrby("cache:stats", "hits", 1)

    async def record_miss(self) -> None:
        """Increment the global cache miss counter."""
        await self.redis.hincrby("cache:stats", "misses", 1)

    async def get_stats(self) -> dict:
        """Return cache statistics: hits, misses, hit rate, entry count."""
        stats_raw = await self.redis.hgetall("cache:stats")
        hits = int(stats_raw.get("hits", 0))
        misses = int(stats_raw.get("misses", 0))
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0

        entry_count = 0
        async for _ in self.redis.scan_iter(match="cache:entry:*", count=100):
            entry_count += 1

        l1_stats = self._l1.stats()
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hit_rate, 4),
            "entries": entry_count,
            **l1_stats,
        }

    async def flush(self) -> int:
        """Delete all cache keys (entries, scopes, stats). Returns count deleted."""
        l1_count = self._l1.flush()
        keys_to_delete = []
        async for key in self.redis.scan_iter(match="cache:*", count=100):
            keys_to_delete.append(key)
        l2_count = 0
        if keys_to_delete:
            l2_count = await self.redis.delete(*keys_to_delete)
        return l1_count + l2_count

    async def acquire_stampede_lock(
        self, model: str, messages: list
    ) -> tuple[bool, str]:
        """Try to acquire a stampede lock for a prompt. Returns (acquired, lock_key)."""
        prompt_hash = self._compute_prompt_hash(model, messages)
        lock_key = f"cache:lock:{prompt_hash}"
        acquired = await self.redis.set(lock_key, "1", nx=True, ex=30)
        return bool(acquired), lock_key

    async def release_stampede_lock(self, lock_key: str) -> None:
        """Release a previously acquired stampede lock."""
        await self.redis.delete(lock_key)

    async def wait_for_cached_result(
        self,
        model: str,
        messages: list,
        tenant_id: str,
        cache_isolation: str,
        timeout: float = 30.0,
    ) -> tuple["ChatCompletionResponse | None", float | None]:
        """Poll for a cached result until timeout, with exponential backoff."""
        deadline = time.monotonic() + timeout
        delay = 0.1
        while time.monotonic() < deadline:
            await asyncio.sleep(delay)
            result, similarity, _tier = await self.lookup(
                model, messages, tenant_id, cache_isolation
            )
            if result is not None:
                return result, similarity
            delay = min(delay * 2, 2.0)
        return None, None
