"""Request journal using Redis Streams for audit trail.

Records every request's lifecycle (start and completion) as entries in a
Redis Stream. Uses a Redis SET for O(1) in-flight request tracking.
Journal operations never raise exceptions — failures are logged silently.
"""

import hashlib
import time

import structlog

logger = structlog.get_logger()


class RequestJournal:
    """Redis Stream-backed request journal with in-flight tracking."""

    STREAM_KEY = "journal:requests"
    INFLIGHT_KEY = "journal:inflight"

    def __init__(self, redis_client, max_len: int = 100000) -> None:
        self.redis = redis_client
        self.max_len = max_len

    @staticmethod
    def compute_prompt_hash(messages: list) -> str:
        """SHA256 hash of concatenated user messages (privacy: never log full prompts)."""
        user_text = "\n".join(m.content or "" for m in messages if m.role == "user")
        return hashlib.sha256(user_text.encode()).hexdigest()[:16]

    async def record_request(
        self,
        request_id: str,
        tenant_id: str,
        model: str,
        prompt_hash: str,
        timestamp: float,
    ) -> None:
        """Record a request entry in the journal. Never raises."""
        try:
            fields = {
                "request_id": request_id,
                "tenant_id": tenant_id,
                "model": model,
                "prompt_hash": prompt_hash,
                "timestamp": str(timestamp),
                "phase": "request",
            }
            await self.redis.xadd(
                self.STREAM_KEY, fields, maxlen=self.max_len, approximate=True
            )
            await self.redis.sadd(self.INFLIGHT_KEY, request_id)
        except Exception as e:
            logger.warning("journal_record_request_failed", error=str(e))

    async def record_completion(
        self,
        request_id: str,
        status: int,
        latency_ms: float,
        backend: str,
        cache_hit: bool,
        tokens_prompt: int,
        tokens_completion: int,
    ) -> None:
        """Record a completion entry in the journal. Never raises."""
        try:
            fields = {
                "request_id": request_id,
                "status": str(status),
                "latency_ms": str(round(latency_ms, 2)),
                "backend": backend,
                "cache_hit": str(cache_hit),
                "tokens_prompt": str(tokens_prompt),
                "tokens_completion": str(tokens_completion),
                "phase": "completion",
            }
            await self.redis.xadd(
                self.STREAM_KEY, fields, maxlen=self.max_len, approximate=True
            )
            await self.redis.srem(self.INFLIGHT_KEY, request_id)
        except Exception as e:
            logger.warning("journal_record_completion_failed", error=str(e))

    async def get_stats(self) -> dict:
        """Return journal statistics: total entries, in-flight count, entries/min."""
        total = await self.redis.xlen(self.STREAM_KEY)
        inflight = await self.redis.scard(self.INFLIGHT_KEY)

        # Calculate entries per minute from stream info
        entries_per_min = 0.0
        if total > 1:
            try:
                info = await self.redis.xinfo_stream(self.STREAM_KEY)
                first_id = info.get("first-entry", [None])[0] if info.get("first-entry") else None
                last_id = info.get("last-entry", [None])[0] if info.get("last-entry") else None
                if first_id and last_id:
                    # Stream IDs are millisecond timestamps: "1234567890123-0"
                    first_ts = int(str(first_id).split("-")[0]) / 1000
                    last_ts = int(str(last_id).split("-")[0]) / 1000
                    span_minutes = max((last_ts - first_ts) / 60, 0.001)
                    entries_per_min = round(total / span_minutes, 2)
            except Exception:
                pass

        return {
            "total": total,
            "inflight": inflight,
            "entries_per_min": entries_per_min,
        }

    async def query(
        self, tenant_id: str | None = None, last: int = 20
    ) -> list[dict]:
        """Query recent journal entries, optionally filtered by tenant.

        Returns entries grouped by request_id with merged request+completion fields.
        """
        # Fetch extra to account for filtering and request/completion pairing
        fetch_count = min(last * 4, 500)
        raw_entries = await self.redis.xrevrange(
            self.STREAM_KEY, "+", "-", count=fetch_count
        )

        # Group entries by request_id
        grouped: dict[str, dict] = {}
        for entry_id, fields in raw_entries:
            rid = fields.get("request_id", "")
            if not rid:
                continue
            if rid not in grouped:
                grouped[rid] = {"request_id": rid, "stream_id": str(entry_id)}
            grouped[rid].update(fields)

        # Filter by tenant if specified
        results = list(grouped.values())
        if tenant_id is not None:
            results = [e for e in results if e.get("tenant_id") == tenant_id]

        return results[:last]
