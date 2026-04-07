import structlog
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel as PydanticBaseModel

from gateway.config import ConfigError, ModelRoutingConfig, Registry, load_config

router = APIRouter(prefix="/admin", tags=["admin"])
logger = structlog.get_logger()


@router.post("/reload")
async def reload_config(request: Request):
    """Hot-reload config from disk. Atomic swap of registry."""
    config_path = request.app.state.config_path
    tracker = getattr(request.app.state, "latency_tracker", None)
    try:
        config = load_config(config_path)
        new_registry = Registry(config, latency_tracker=tracker)
    except ConfigError as e:
        logger.error("config_reload_failed", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

    request.app.state.registry = new_registry
    request.app.state.circuit_breakers.sync_backends(
        list(new_registry.backends.keys())
    )
    logger.info(
        "config_reloaded",
        backends=len(config.backends),
        tenants=len(config.tenants),
    )
    return {
        "status": "reloaded",
        "backends": len(config.backends),
        "tenants": len(config.tenants),
    }


@router.get("/ring")
async def ring_state(request: Request):
    """Return consistent hash ring state per model."""
    registry = request.app.state.registry
    return registry.ring_state()


@router.get("/backends")
async def list_backends(request: Request):
    """List all registered backends with circuit breaker state."""
    registry = request.app.state.registry
    cb_registry = getattr(request.app.state, "circuit_breakers", None)
    cb_snapshots = cb_registry.get_all_snapshots() if cb_registry else {}

    return [
        {
            "name": b.name,
            "provider": b.provider,
            "models": b.models,
            "health": cb_snapshots.get(b.name, {}).get("state", "unknown"),
            "circuit_breaker": cb_snapshots.get(b.name, {}),
        }
        for b in registry.backends.values()
    ]


@router.get("/cache/stats")
async def cache_stats(request: Request):
    """Return semantic cache statistics."""
    semantic_cache = getattr(request.app.state, "semantic_cache", None)
    if semantic_cache is None:
        return {"enabled": False, "message": "Semantic cache not available"}
    try:
        stats = await semantic_cache.get_stats()
        return {"enabled": True, **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache stats error: {e}")


@router.delete("/cache")
async def flush_cache(request: Request):
    """Flush all cached responses."""
    semantic_cache = getattr(request.app.state, "semantic_cache", None)
    if semantic_cache is None:
        raise HTTPException(status_code=503, detail="Semantic cache not available")
    try:
        count = await semantic_cache.flush()
        logger.info("cache_flushed", entries_deleted=count)
        return {"status": "flushed", "entries_deleted": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache flush error: {e}")


@router.get("/queue")
async def queue_stats(request: Request):
    """Return priority queue statistics: concurrency per backend, depth per model."""
    queue_manager = getattr(request.app.state, "queue_manager", None)
    if queue_manager is None:
        return {"enabled": False, "message": "Priority queue not available"}

    registry = request.app.state.registry

    concurrency = {}
    for backend_name, backend_config in registry.backends.items():
        concurrency[backend_name] = {
            "active": queue_manager.get_concurrency(backend_name),
            "max": backend_config.max_concurrent,
        }

    queues = {}
    for model in registry.model_to_backends:
        depth = await queue_manager.get_queue_depth(model)
        queues[model] = {"depth": depth, "max_depth": queue_manager.max_queue_depth}

    return {
        "enabled": True,
        "concurrency": concurrency,
        "queues": queues,
    }


@router.get("/journal/stats")
async def journal_stats(request: Request):
    """Return journal statistics: total entries, entries/min, in-flight count."""
    journal = getattr(request.app.state, "journal", None)
    if journal is None:
        return {"enabled": False, "message": "Request journal not available"}
    try:
        stats = await journal.get_stats()
        return {"enabled": True, **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Journal stats error: {e}")


@router.get("/journal")
async def journal_query(
    request: Request, tenant: str | None = None, last: int = 20
):
    """Query recent journal entries, optionally filtered by tenant."""
    journal = getattr(request.app.state, "journal", None)
    if journal is None:
        return {"enabled": False, "message": "Request journal not available"}
    try:
        entries = await journal.query(tenant_id=tenant, last=min(last, 100))
        return {"enabled": True, "entries": entries, "count": len(entries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Journal query error: {e}")


@router.get("/routing")
async def routing_state(request: Request):
    """Return per-model routing strategy and latency data."""
    registry = request.app.state.registry
    tracker = getattr(request.app.state, "latency_tracker", None)
    result = {}
    for model in registry.model_strategies:
        cfg = registry.model_routing_config.get(model, ModelRoutingConfig())
        entry: dict = {
            "strategy": cfg.strategy,
            "hedge_enabled": cfg.hedge_enabled,
        }
        if tracker:
            entry["p95_latencies"] = tracker.get_all_p95(model)
        result[model] = entry
    return result


class CacheWarmRequest(PydanticBaseModel):
    """Request body for cache warming."""

    prompts: list[dict]  # [{"model": "m1", "messages": [{"role": "user", "content": "..."}]}]


@router.post("/cache/warm")
async def warm_cache(request: Request, body: CacheWarmRequest):
    """Pre-populate cache by sending prompts to backends and storing results.

    Each prompt is sent to the appropriate backend, and the response is
    stored in the semantic cache. Useful for pre-warming before traffic spikes.
    """
    semantic_cache = getattr(request.app.state, "semantic_cache", None)
    if semantic_cache is None:
        raise HTTPException(status_code=503, detail="Semantic cache not available")

    registry = request.app.state.registry
    http_client = request.app.state.http_client
    cb_registry = request.app.state.circuit_breakers

    warmed = 0
    errors = 0

    for prompt_spec in body.prompts:
        model = prompt_spec.get("model")
        raw_messages = prompt_spec.get("messages", [])

        if not model or not raw_messages:
            errors += 1
            continue

        try:
            from gateway.models import ChatCompletionRequest, ChatMessage

            messages = [ChatMessage(**m) for m in raw_messages]
            chat_req = ChatCompletionRequest(model=model, messages=messages)
        except Exception as e:
            logger.warning("cache_warm_invalid_prompt", error=str(e))
            errors += 1
            continue

        # Find backend
        exclude = cb_registry.get_open_backends()
        backend = registry.find_backend_for_model(model, exclude=exclude)
        if backend is None:
            logger.warning("cache_warm_no_backend", model=model)
            errors += 1
            continue

        # Get translator
        from gateway.routes.chat import TRANSLATORS

        translator = TRANSLATORS.get(backend.provider)
        if translator is None:
            errors += 1
            continue

        try:
            result = await translator(
                client=http_client,
                backend=backend,
                request=chat_req,
            )
            await semantic_cache.store(
                model=model,
                messages=messages,
                response=result,
                tenant_id="__warm__",
                cache_isolation="shared",
            )
            warmed += 1
        except Exception as e:
            logger.warning("cache_warm_failed", model=model, error=str(e))
            errors += 1

    logger.info("cache_warm_completed", warmed=warmed, errors=errors)
    return {"status": "completed", "warmed": warmed, "errors": errors}


@router.get("/tenants")
async def list_tenants(request: Request):
    """List configured tenants (without exposing API keys)."""
    registry = request.app.state.registry
    seen = set()
    tenants = []
    for tenant in registry.api_key_to_tenant.values():
        if tenant.id in seen:
            continue
        seen.add(tenant.id)
        tenants.append({
            "id": tenant.id,
            "allowed_models": tenant.allowed_models,
            "priority": tenant.priority,
            "rate_limit_rps": tenant.rate_limit_rps,
            "rate_limit_rpm": tenant.rate_limit_rpm,
            "token_budget_daily": tenant.token_budget_daily,
        })
    return tenants


@router.get("/cost")
async def cost_summary(request: Request, tenant: str | None = None, days: int = 7):
    """Return estimated cost summary per tenant."""
    cost_tracker = getattr(request.app.state, "cost_tracker", None)
    if cost_tracker is None:
        return {"enabled": False, "message": "Cost tracking requires Redis"}

    registry = request.app.state.registry
    days = min(days, 30)

    if tenant:
        summary = await cost_tracker.get_cost_summary(tenant, days=days)
        return {"enabled": True, **summary}

    # All tenants
    seen = set()
    tenant_ids = []
    for t in registry.api_key_to_tenant.values():
        if t.id not in seen:
            seen.add(t.id)
            tenant_ids.append(t.id)
    summaries = await cost_tracker.get_all_tenants_cost(tenant_ids, days=days)
    return {"enabled": True, "tenants": summaries}
