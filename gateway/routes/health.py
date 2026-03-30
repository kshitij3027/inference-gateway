from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def readiness(request: Request):
    """Readiness probe: returns 200 only if at least 1 backend is healthy."""
    cb_registry = getattr(request.app.state, "circuit_breakers", None)
    registry = getattr(request.app.state, "registry", None)

    if not cb_registry or not registry:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "not_initialized"},
        )

    all_backends = set(registry.backends.keys())
    open_backends = cb_registry.get_open_backends()
    healthy_count = len(all_backends - open_backends)

    if healthy_count == 0:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "healthy_backends": 0,
                "total_backends": len(all_backends),
            },
        )

    return {
        "status": "ready",
        "healthy_backends": healthy_count,
        "total_backends": len(all_backends),
    }
