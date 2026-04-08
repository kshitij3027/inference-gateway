import asyncio
import os
import signal
import socket
import sys
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import redis.asyncio as aioredis
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from gateway.circuit_breaker import CircuitBreakerRegistry
from gateway.config import ConfigError, Registry, load_config
from gateway.latency_tracker import LatencyTracker
from gateway.rate_limiter import RateLimiter
from gateway.observability.logging import setup_logging
from gateway.routes.admin import router as admin_router
from gateway.routes.chat import router as chat_router
from gateway.routes.health import router as health_router
from gateway.events import EventBroadcaster
from gateway.routes.dashboard import router as dashboard_router

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
    app.state.latency_tracker = LatencyTracker()
    app.state.registry = Registry(config, latency_tracker=app.state.latency_tracker)
    app.state.circuit_breakers = CircuitBreakerRegistry(
        list(app.state.registry.backends.keys())
    )
    app.state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    # Chaos mode (disabled by default)
    if os.getenv("CHAOS_ENABLED", "false").lower() == "true":
        from gateway.chaos import ChaosConfig, ChaosHttpClient

        chaos_config = ChaosConfig(
            error_rate=float(os.getenv("CHAOS_ERROR_RATE", "0.10")),
            timeout_rate=float(os.getenv("CHAOS_TIMEOUT_RATE", "0.05")),
            latency_rate=float(os.getenv("CHAOS_LATENCY_RATE", "0.30")),
            latency_min_ms=float(os.getenv("CHAOS_LATENCY_MIN_MS", "50")),
            latency_max_ms=float(os.getenv("CHAOS_LATENCY_MAX_MS", "2000")),
        )
        app.state.http_client = ChaosHttpClient(
            app.state.http_client, chaos_config
        )
        logger.warning("chaos_mode_enabled", config=vars(chaos_config))

    # Initialize circuit breaker gauges to CLOSED (0) for all backends
    from gateway.observability.metrics import CIRCUIT_BREAKER_STATE

    for backend_name in app.state.registry.backends:
        CIRCUIT_BREAKER_STATE.labels(backend=backend_name).set(0)

    # Redis + rate limiter (best-effort — gateway works without Redis)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        app.state.redis = aioredis.from_url(redis_url, decode_responses=True)
        await app.state.redis.ping()
        app.state.rate_limiter = RateLimiter(app.state.redis)
        from gateway.cost_tracker import CostTracker
        app.state.cost_tracker = CostTracker(app.state.redis)
        logger.info("redis_connected", url=redis_url)
        logger.info("cost_tracker_initialized")
    except Exception as e:
        logger.warning("redis_unavailable", error=str(e))
        app.state.redis = None
        app.state.rate_limiter = None
        app.state.cost_tracker = None

    # Semantic cache (requires Redis)
    if app.state.redis is not None:
        from gateway.semantic_cache import SemanticCache

        cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        similarity_threshold = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
        l1_max_entries = int(os.getenv("L1_MAX_ENTRIES", "500"))
        l1_ttl = int(os.getenv("L1_TTL", str(cache_ttl)))
        app.state.semantic_cache = SemanticCache(
            app.state.redis,
            similarity_threshold=similarity_threshold,
            default_ttl=cache_ttl,
            l1_max_entries=l1_max_entries,
            l1_ttl=l1_ttl,
        )
        logger.info(
            "semantic_cache_initialized",
            ttl=cache_ttl,
            threshold=similarity_threshold,
            l1_max_entries=l1_max_entries,
            l1_ttl=l1_ttl,
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

    # Request journal (requires Redis)
    if app.state.redis is not None:
        from gateway.journal import RequestJournal

        journal_max_len = int(os.getenv("JOURNAL_MAX_LEN", "100000"))
        app.state.journal = RequestJournal(
            app.state.redis, max_len=journal_max_len
        )
        logger.info("journal_initialized", max_len=journal_max_len)
    else:
        app.state.journal = None

    # SIGHUP handler for hot-reload
    def handle_sighup(signum, frame):
        try:
            new_config = load_config(CONFIG_PATH)
            app.state.registry = Registry(
                new_config, latency_tracker=app.state.latency_tracker
            )
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

    # Instance identification
    app.state.instance_id = os.getenv("INSTANCE_ID", socket.gethostname())

    # Event broadcaster for live dashboard
    app.state.event_broadcaster = EventBroadcaster()

    # Wire circuit breaker state changes to dashboard events
    def _on_cb_state_change(backend: str, old_state: str, new_state: str):
        app.state.event_broadcaster.emit("circuit_state_change", {
            "backend": backend,
            "old_state": old_state,
            "new_state": new_state,
        })

    for cb in app.state.circuit_breakers.breakers.values():
        cb.on_state_change = _on_cb_state_change

    # Graceful shutdown state
    app.state.shutting_down = False
    app.state.inflight_count = 0
    app.state.inflight_lock = asyncio.Lock()
    app.state.inflight_zero = asyncio.Event()
    app.state.inflight_zero.set()  # Starts at zero

    # SIGTERM handler for graceful shutdown
    loop = asyncio.get_running_loop()

    def handle_sigterm():
        logger.info("sigterm_received", message="starting graceful shutdown")
        app.state.shutting_down = True

    try:
        loop.add_signal_handler(signal.SIGTERM, handle_sigterm)
    except (NotImplementedError, OSError):
        logger.warning("sigterm_handler_not_available")

    logger.info(
        "gateway_started",
        instance_id=app.state.instance_id,
        backends=len(config.backends),
        tenants=len(config.tenants),
    )
    yield

    # Graceful drain: wait for in-flight requests up to 10s
    if app.state.inflight_count > 0:
        logger.info("draining_inflight", count=app.state.inflight_count)
        try:
            await asyncio.wait_for(app.state.inflight_zero.wait(), timeout=10.0)
            logger.info("drain_complete")
        except asyncio.TimeoutError:
            logger.warning(
                "drain_timeout", remaining=app.state.inflight_count
            )

    if getattr(app.state, "redis", None):
        await app.state.redis.aclose()
    await app.state.http_client.aclose()
    logger.info("gateway_stopped")


app = FastAPI(title="Inference Gateway", lifespan=lifespan)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(dashboard_router)

# Dashboard static files (served if directory exists)
import os as _os
from starlette.staticfiles import StaticFiles as _StaticFiles

_dashboard_dir = _os.path.join(_os.path.dirname(__file__), "dashboard")
if _os.path.isdir(_dashboard_dir):
    app.mount("/dashboard", _StaticFiles(directory=_dashboard_dir, html=True), name="dashboard")

# Prometheus metrics endpoint
from prometheus_client import make_asgi_app

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    # Reject new requests during shutdown (allow health/ready/metrics)
    if getattr(request.app.state, "shutting_down", False):
        if request.url.path not in ("/health", "/ready", "/metrics", "/metrics/") and not request.url.path.startswith("/dashboard"):
            return JSONResponse(
                status_code=503,
                content={"error": "shutting_down", "detail": "Server is shutting down"},
                headers={"Retry-After": "5"},
            )

    # Track in-flight requests
    async with request.app.state.inflight_lock:
        request.app.state.inflight_count += 1
        request.app.state.inflight_zero.clear()

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    request.state.request_id = request_id

    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        response.headers["X-Request-ID"] = request_id
        instance_id = getattr(request.app.state, "instance_id", None)
        if instance_id:
            response.headers["X-Instance-ID"] = instance_id
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

        hedge_winner = getattr(request.state, "hedge_winner", None)
        if hedge_winner:
            response.headers["X-Hedge-Winner"] = hedge_winner
        hedge_loser = getattr(request.state, "hedge_loser", None)
        if hedge_loser:
            response.headers["X-Hedge-Loser"] = hedge_loser

        retry_count = getattr(request.state, "retry_count", None)
        if retry_count and retry_count > 0:
            response.headers["X-Retry-Count"] = str(retry_count)

        estimated_cost = getattr(request.state, "estimated_cost", None)
        if estimated_cost is not None:
            response.headers["X-Estimated-Cost"] = f"{estimated_cost:.6f}"

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
    finally:
        async with request.app.state.inflight_lock:
            request.app.state.inflight_count -= 1
            if request.app.state.inflight_count <= 0:
                request.app.state.inflight_count = 0
                request.app.state.inflight_zero.set()
