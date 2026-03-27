import os
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI, Request, Response

from gateway.circuit_breaker import CircuitBreakerRegistry
from gateway.config import ConfigError, Registry, load_config
from gateway.rate_limiter import RateLimiter
from gateway.observability.logging import setup_logging
from gateway.routes.admin import router as admin_router
from gateway.routes.chat import router as chat_router
from gateway.routes.health import router as health_router

setup_logging(log_level=os.getenv("LOG_LEVEL", "info"))
logger = structlog.get_logger()

CONFIG_PATH = os.getenv("CONFIG_PATH", "config/backends.yaml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load config
    try:
        config = load_config(CONFIG_PATH)
    except ConfigError as e:
        logger.error("config_load_failed", error=str(e))
        sys.exit(1)

    app.state.config_path = CONFIG_PATH
    app.state.registry = Registry(config)
    app.state.circuit_breakers = CircuitBreakerRegistry(
        list(app.state.registry.backends.keys())
    )
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    # Redis + rate limiter (best-effort — gateway works without Redis)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        app.state.redis = aioredis.from_url(redis_url, decode_responses=True)
        await app.state.redis.ping()
        app.state.rate_limiter = RateLimiter(app.state.redis)
        logger.info("redis_connected", url=redis_url)
    except Exception as e:
        logger.warning("redis_unavailable", error=str(e))
        app.state.redis = None
        app.state.rate_limiter = None

    # SIGHUP handler for hot-reload
    def handle_sighup(signum, frame):
        try:
            new_config = load_config(CONFIG_PATH)
            app.state.registry = Registry(new_config)
            app.state.circuit_breakers.sync_backends(
                list(app.state.registry.backends.keys())
            )
            logger.info("config_reloaded_via_sighup")
        except ConfigError as e:
            logger.error("sighup_reload_failed", error=str(e))

    try:
        signal.signal(signal.SIGHUP, handle_sighup)
    except (OSError, AttributeError):
        logger.warning("sighup_not_available")

    logger.info(
        "gateway_started",
        backends=len(config.backends),
        tenants=len(config.tenants),
    )
    yield
    if getattr(app.state, "redis", None):
        await app.state.redis.aclose()
    await app.state.http_client.aclose()
    logger.info("gateway_stopped")


app = FastAPI(title="Inference Gateway", lifespan=lifespan)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(admin_router)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    request.state.request_id = request_id

    start = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    response.headers["X-Request-ID"] = request_id
    backend_name = getattr(request.state, "backend_name", None)
    if backend_name:
        response.headers["X-Backend"] = backend_name
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response
