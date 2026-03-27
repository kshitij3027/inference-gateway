import structlog
from fastapi import APIRouter, HTTPException, Request

from gateway.config import ConfigError, Registry, load_config

router = APIRouter(prefix="/admin", tags=["admin"])
logger = structlog.get_logger()


@router.post("/reload")
async def reload_config(request: Request):
    """Hot-reload config from disk. Atomic swap of registry."""
    config_path = request.app.state.config_path
    try:
        config = load_config(config_path)
        new_registry = Registry(config)
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
