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

    # Semantic cache (requires Redis)
    if app.state.redis is not None:
        from gateway.semantic_cache import SemanticCache

        cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        similarity_threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
        app.state.semantic_cache = SemanticCache(
            app.state.redis,
            similarity_threshold=similarity_threshold,
            default_ttl=cache_ttl,
        )
        logger.info(
            "semantic_cache_initialized", ttl=cache_ttl, threshold=similarity_threshold
        )
    else:
        app.state.semantic_cache = None

    # Priority queue (requires Redis)
    if app.state.redis is not None:
        from gateway.priority_queue import PriorityQueueManager

        queue_max_depth = int(os.getenv("QUEUE_MAX_DEPTH", "100"))
        queue_timeout = float(os.getenv("QUEUE_TIMEOUT", "30"))
        app.state.queue_manager = PriorityQueueManager(
            app.state.redis,
            max_queue_depth=queue_max_depth,
            queue_timeout=queue_timeout,
        )
        logger.info(
            "queue_manager_initialized",
            max_depth=queue_max_depth,
            timeout=queue_timeout,
        )
    else:
        app.state.queue_manager = None

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

# Prometheus metrics endpoint
from prometheus_client import make_asgi_app

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


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
    rate_limit_remaining = getattr(request.state, "rate_limit_remaining", None)
    if rate_limit_remaining:
        if "rps" in rate_limit_remaining:
            response.headers["X-Ratelimit-Remaining-Rps"] = str(rate_limit_remaining["rps"])
        if "rpm" in rate_limit_remaining:
            response.headers["X-Ratelimit-Remaining-Rpm"] = str(rate_limit_remaining["rpm"])
    cache_status = getattr(request.state, "cache_status", None)
    if cache_status:
        response.headers["X-Cache"] = cache_status
    cache_similarity = getattr(request.state, "cache_similarity", None)
    if cache_similarity is not None:
        response.headers["X-Cache-Similarity"] = f"{cache_similarity:.4f}"
    queue_wait_ms = getattr(request.state, "queue_wait_ms", None)
    if queue_wait_ms is not None:
        response.headers["X-Queue-Wait-Ms"] = str(queue_wait_ms)

    # Prometheus metrics
    from gateway.observability.metrics import REQUEST_COUNT, REQUEST_LATENCY

    tenant_id = getattr(request.state, "tenant_id", "")
    model_name = getattr(request.state, "model_name", "")
    backend = getattr(request.state, "backend_name", "") or ""
    REQUEST_COUNT.labels(
        tenant=tenant_id,
        model=model_name,
        backend=backend,
        status_code=str(response.status_code),
        method=request.method,
    ).inc()
    REQUEST_LATENCY.labels(
        tenant=tenant_id, model=model_name, backend=backend
    ).observe(duration_ms / 1000)

    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response
