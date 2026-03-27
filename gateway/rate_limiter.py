"""Distributed sliding window rate limiter using Redis sorted sets.

Uses a Lua script for atomic check-and-increment to prevent TOCTOU races.
Three independent dimensions: RPS (1s), RPM (60s), daily token budget.
"""

import time
from datetime import date, datetime, timezone

import structlog

logger = structlog.get_logger()

# Atomic sliding window Lua script
# Returns {1, count} if allowed, {0, count} if denied
_SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local window_start = ARGV[1]
local now = ARGV[2]
local member = ARGV[3]
local limit = tonumber(ARGV[4])
local expire_s = tonumber(ARGV[5])

-- Remove entries outside the window
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

-- Count entries in the current window
local count = redis.call('ZCARD', key)

-- Check if adding one more would exceed the limit
if count >= limit then
    return {0, count}
end

-- Add the new entry
redis.call('ZADD', key, now, member)

-- Set TTL so the key auto-expires
redis.call('EXPIRE', key, expire_s)

return {1, count}
"""


class RateLimiter:
    """Per-tenant rate limiter backed by Redis sorted sets."""

    def __init__(self, redis_client) -> None:
        self.redis = redis_client
        self._script = None

    async def _ensure_script(self):
        """Lazily register the Lua script."""
        if self._script is None:
            self._script = self.redis.register_script(_SLIDING_WINDOW_SCRIPT)

    async def check_rate_limit(
        self,
        tenant_id: str,
        request_id: str,
        rps_limit: int | None,
        rpm_limit: int | None,
    ) -> tuple[bool, dict | None]:
        """Check RPS and RPM limits. Returns (allowed, deny_info|None)."""
        await self._ensure_script()
        now = time.time()

        # Check RPS first
        if rps_limit is not None:
            allowed, count = await self._check_window(
                key=f"ratelimit:{tenant_id}:rps",
                now=now,
                window_size=1.0,
                member=f"{request_id}:rps",
                limit=rps_limit,
                expire_seconds=2,
            )
            if not allowed:
                return False, {
                    "limit_type": "rps",
                    "limit": rps_limit,
                    "current": count,
                    "retry_after": 1.0,
                }

        # Check RPM
        if rpm_limit is not None:
            allowed, count = await self._check_window(
                key=f"ratelimit:{tenant_id}:rpm",
                now=now,
                window_size=60.0,
                member=f"{request_id}:rpm",
                limit=rpm_limit,
                expire_seconds=120,
            )
            if not allowed:
                return False, {
                    "limit_type": "rpm",
                    "limit": rpm_limit,
                    "current": count,
                    "retry_after": 60.0,
                }

        return True, None

    async def _check_window(
        self,
        key: str,
        now: float,
        window_size: float,
        member: str,
        limit: int,
        expire_seconds: int,
    ) -> tuple[bool, int]:
        """Execute the sliding window Lua script."""
        window_start = now - window_size
        result = await self._script(
            keys=[key],
            args=[
                str(window_start),
                str(now),
                member,
                str(limit),
                str(expire_seconds),
            ],
        )
        allowed = int(result[0]) == 1
        count = int(result[1])
        return allowed, count

    async def check_token_budget(
        self,
        tenant_id: str,
        budget: int | None,
    ) -> tuple[bool, dict | None]:
        """Check if tenant has exceeded daily token budget."""
        if budget is None:
            return True, None

        today = date.today().isoformat()
        key = f"ratelimit:{tenant_id}:tokens:{today}"
        raw = await self.redis.get(key)
        current = int(raw) if raw else 0

        if current >= budget:
            # Calculate seconds until midnight UTC
            now_utc = datetime.now(timezone.utc)
            midnight = now_utc.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            midnight_tomorrow = midnight.replace(day=midnight.day + 1)
            retry_after = (midnight_tomorrow - now_utc).total_seconds()
            return False, {
                "limit_type": "token_budget_daily",
                "limit": budget,
                "current": current,
                "retry_after": max(1.0, retry_after),
            }

        return True, None

    async def record_tokens(self, tenant_id: str, tokens: int) -> int:
        """Record token usage for daily budget. Returns new total."""
        today = date.today().isoformat()
        key = f"ratelimit:{tenant_id}:tokens:{today}"
        new_total = await self.redis.incrby(key, tokens)
        await self.redis.expire(key, 90000)  # 25 hours TTL
        return new_total

    async def get_remaining(
        self,
        tenant_id: str,
        rps_limit: int | None,
        rpm_limit: int | None,
    ) -> dict:
        """Get remaining request counts for response headers."""
        now = time.time()
        remaining = {}

        if rps_limit is not None:
            key = f"ratelimit:{tenant_id}:rps"
            # Clean old entries first
            await self.redis.zremrangebyscore(key, "-inf", str(now - 1.0))
            count = await self.redis.zcard(key)
            remaining["rps"] = max(0, rps_limit - count)

        if rpm_limit is not None:
            key = f"ratelimit:{tenant_id}:rpm"
            await self.redis.zremrangebyscore(key, "-inf", str(now - 60.0))
            count = await self.redis.zcard(key)
            remaining["rpm"] = max(0, rpm_limit - count)

        return remaining
