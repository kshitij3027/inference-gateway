"""Prometheus metrics for the inference gateway.

All metrics use the ``gateway_`` prefix namespace. Metrics are module-level
globals (prometheus_client convention -- thread-safe singletons).
"""

from prometheus_client import Counter, Gauge, Histogram

# --- Request metrics (recorded in middleware) ---

REQUEST_COUNT = Counter(
    "gateway_request_total",
    "Total HTTP requests processed",
    ["tenant", "model", "backend", "status_code", "method"],
)

REQUEST_LATENCY = Histogram(
    "gateway_request_duration_seconds",
    "Request latency in seconds",
    ["tenant", "model", "backend"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# --- Cache metrics ---

CACHE_OPERATIONS = Counter(
    "gateway_cache_operations_total",
    "Semantic cache operations",
    ["model", "status"],  # status: "hit" or "miss"
)

# --- Rate limit metrics ---

RATE_LIMIT_REJECTIONS = Counter(
    "gateway_rate_limit_rejections_total",
    "Rate limit rejections",
    ["tenant", "limit_type"],  # limit_type: "rps", "rpm", "token_budget_daily"
)

# --- Circuit breaker metrics ---

CIRCUIT_BREAKER_STATE = Gauge(
    "gateway_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["backend"],
)

# --- Queue metrics ---

QUEUE_DEPTH = Gauge(
    "gateway_queue_depth",
    "Current queue depth per model",
    ["model"],
)

# --- Token metrics ---

TOKENS_CONSUMED = Counter(
    "gateway_tokens_consumed_total",
    "Tokens consumed",
    ["tenant", "model", "type"],  # type: "prompt" or "completion"
)

ESTIMATED_COST = Counter(
    "gateway_estimated_cost_dollars",
    "Estimated cost in dollars",
    ["tenant", "model"],
)

# --- Concurrency metrics ---

ACTIVE_REQUESTS = Gauge(
    "gateway_active_requests",
    "Active concurrent requests per backend",
    ["backend"],
)

# --- Hedge metrics ---

HEDGE_REQUESTS = Counter(
    "gateway_hedge_requests_total",
    "Total hedge requests",
    ["model"],
)

HEDGE_WIN_RATE = Counter(
    "gateway_hedge_win_rate",
    "Hedge wins per backend",
    ["backend", "model"],
)

# --- Streaming analytics metrics ---

TTFT = Histogram(
    "gateway_ttft_seconds",
    "Time to first content token in seconds",
    ["model", "backend"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

ITL = Histogram(
    "gateway_itl_seconds",
    "Inter-token latency in seconds",
    ["model", "backend"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

GENERATION_DURATION = Histogram(
    "gateway_generation_duration_seconds",
    "Duration from first token to last token in seconds",
    ["model", "backend"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)
