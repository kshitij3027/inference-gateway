import os
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI, Request, Response

from gateway.config import ConfigError, Registry, load_config
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
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    # SIGHUP handler for hot-reload
    def handle_sighup(signum, frame):
        try:
            new_config = load_config(CONFIG_PATH)
            app.state.registry = Registry(new_config)
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
