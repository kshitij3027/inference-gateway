"""Per-request cost estimation with Redis daily accumulation."""

import time
from datetime import datetime, timedelta, timezone

import structlog

logger = structlog.get_logger()


class CostTracker:
    """Tracks estimated cost per request, accumulated daily per tenant in Redis."""

    def __init__(self, redis_client) -> None:
        self.redis = redis_client

    @staticmethod
    def calculate_cost(backend, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate estimated cost in dollars for a request.

        Uses input_cost_per_1k_tokens / output_cost_per_1k_tokens if available,
        otherwise falls back to cost_per_1k_tokens for both.
        Returns 0.0 if no pricing is configured.
        """
        input_price = backend.input_cost_per_1k_tokens
        if input_price is None:
            input_price = backend.cost_per_1k_tokens
        output_price = backend.output_cost_per_1k_tokens
        if output_price is None:
            output_price = backend.cost_per_1k_tokens

        if input_price is None and output_price is None:
            return 0.0

        input_cost = (prompt_tokens * (input_price or 0.0)) / 1000
        output_cost = (completion_tokens * (output_price or 0.0)) / 1000
        return round(input_cost + output_cost, 8)

    async def record_cost(self, tenant_id: str, model: str, cost: float) -> float:
        """Record cost in Redis and Prometheus. Returns new daily total."""
        if cost <= 0:
            return 0.0

        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"cost:{tenant_id}:{date_str}"

        try:
            new_total = await self.redis.incrbyfloat(key, cost)
            await self.redis.expire(key, 90000)  # 25 hours

            # Prometheus metric
            from gateway.observability.metrics import ESTIMATED_COST
            ESTIMATED_COST.labels(tenant=tenant_id, model=model).inc(cost)

            return float(new_total)
        except Exception as e:
            logger.warning("cost_record_failed", error=str(e), tenant=tenant_id)
            return 0.0

    async def get_daily_cost(self, tenant_id: str, date_str: str | None = None) -> float:
        """Get accumulated cost for a tenant on a given date."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        key = f"cost:{tenant_id}:{date_str}"
        val = await self.redis.get(key)
        return float(val) if val else 0.0

    async def get_cost_summary(self, tenant_id: str, days: int = 7) -> dict:
        """Get cost summary for a tenant over the last N days."""
        today = datetime.now(timezone.utc).date()
        costs_by_date = {}
        for i in range(days):
            d = today - timedelta(days=i)
            date_str = d.strftime("%Y-%m-%d")
            costs_by_date[date_str] = await self.get_daily_cost(tenant_id, date_str)

        return {
            "tenant_id": tenant_id,
            "today": costs_by_date.get(today.strftime("%Y-%m-%d"), 0.0),
            "costs_by_date": costs_by_date,
        }

    async def get_all_tenants_cost(self, tenant_ids: list[str], days: int = 7) -> list[dict]:
        """Get cost summaries for multiple tenants."""
        return [await self.get_cost_summary(tid, days) for tid in tenant_ids]
