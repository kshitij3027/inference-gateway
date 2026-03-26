import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI, Request, Response

from gateway.observability.logging import setup_logging
from gateway.routes.health import router as health_router

setup_logging(log_level=os.getenv("LOG_LEVEL", "info"))
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    logger.info("gateway_started")
    yield
    await app.state.http_client.aclose()
    logger.info("gateway_stopped")


app = FastAPI(title="Inference Gateway", lifespan=lifespan)
app.include_router(health_router)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    start = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response
