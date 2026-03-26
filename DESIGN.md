# InferenceGateway — Design Document

## Request/Response Translation — Ollama

### Why this exists

Clients should speak one protocol: the OpenAI chat completions API. Without a translation layer, every client that wants to use Ollama must implement Ollama-specific request formatting (the `/api/chat` endpoint, the `options` dict for parameters, the `num_predict` naming convention) and response parsing (extracting `eval_count` for token usage, reshaping the flat message into a `choices` array). This coupling multiplies across clients and backends.

The gateway absorbs this complexity. Clients send standard OpenAI-format requests, and the gateway handles the bidirectional translation to Ollama's native protocol. When additional backends (Anthropic, vLLM) are added in later phases, clients require zero changes.

### How it works

1. Client sends `POST /v1/chat/completions` with an OpenAI-format JSON body (model, messages, optional temperature/max_tokens/top_p).
2. FastAPI validates the request body against the `ChatCompletionRequest` Pydantic model. Invalid requests (missing model, empty messages, bad role) are rejected with 422.
3. The `request_id_middleware` in `gateway/main.py` either extracts `X-Request-ID` from the incoming header or generates a UUID4. It binds this ID to structlog's contextvars so every log line within the request carries the same `request_id`.
4. The route handler in `gateway/routes/chat.py` calls `ollama.chat_completion()`, passing the shared `httpx.AsyncClient` from `app.state` and the `OLLAMA_BASE_URL` from the environment.
5. `translate_request()` in `gateway/backends/ollama.py` converts to Ollama format: messages pass through directly (Ollama supports the same role values), `max_tokens` maps to `num_predict` inside an `options` dict, `temperature` and `top_p` also go into `options`, and `stream` is hardcoded to `false`.
6. `httpx.AsyncClient` sends an async POST to `{OLLAMA_BASE_URL}/api/chat` with the translated body. The client uses a 120-second timeout configured in the lifespan.
7. `translate_response()` converts back to OpenAI format: generates a `chatcmpl-{uuid_hex[:24]}` ID, maps `prompt_eval_count` to `prompt_tokens` and `eval_count` to `completion_tokens` (defaulting to 0 when absent), wraps the assistant message in a `choices` array with `finish_reason: "stop"`, and sets `object: "chat.completion"`.
8. The route handler logs the completion event with model, token counts, and duration, then returns the `ChatCompletionResponse`.
9. The middleware logs the overall request (method, path, status code, duration) and sets the `X-Request-ID` response header.

### Implementation

**Models** (`gateway/models.py`):
- `ChatMessage` — role (Literal["system", "user", "assistant"]) + content. Strict role validation via Literal type.
- `ChatCompletionRequest` — model, messages (min_length=1), optional temperature/max_tokens/top_p/n/stop/stream. Uses `ConfigDict(extra="allow")` to accept unknown fields without failing.
- `ChatCompletionResponse` — auto-generated `chatcmpl-{uuid}` ID, `created` timestamp, model, choices list, usage.
- `Choice` — index (default 0), `ChatMessageResponse`, finish_reason (default "stop").
- `Usage` — prompt_tokens, completion_tokens, total_tokens.
- `OllamaRequest` — model, messages (list of `OllamaMessage`), stream (bool), options (optional dict).
- `OllamaResponse` — model, created_at, message (`OllamaMessage`), done, plus optional duration and token count fields.

**Translator** (`gateway/backends/ollama.py`):
- `translate_request(ChatCompletionRequest) -> OllamaRequest` — maps messages 1:1, builds options dict from temperature/max_tokens/top_p, hardcodes stream=False.
- `translate_response(OllamaResponse, model) -> ChatCompletionResponse` — maps token counts with 0-defaults, wraps message in choices array, generates server-side ID.
- `chat_completion(client, base_url, request) -> ChatCompletionResponse` — async orchestration: translate, POST, validate response, translate back. Handles `TimeoutException` (504), `HTTPStatusError` (propagated status), and `ValidationError` (502).

**Route** (`gateway/routes/chat.py`):
- `POST /v1/chat/completions` — extracts shared httpx client from `request.app.state.http_client`, reads `OLLAMA_BASE_URL` from environment (default `http://ollama:11434`), delegates to `ollama.chat_completion()`.

**App setup** (`gateway/main.py`):
- Lifespan context manager creates `httpx.AsyncClient(timeout=120.0)` on startup, stores on `app.state`, closes on shutdown.
- `request_id_middleware` — generates/propagates X-Request-ID, binds to structlog contextvars, logs request completion with method/path/status/duration.

**Logging** (`gateway/observability/logging.py`):
- `setup_logging()` configures structlog with: contextvars merging, log level injection, ISO timestamps, stack info rendering, exception formatting, JSON output.

**Infrastructure**:
- `Dockerfile` — three-stage build: `base` (installs deps), `test` (adds pytest + test files), `runtime` (non-root `appuser`, copies only gateway code, runs uvicorn on port 8080).
- `docker-compose.yaml` — three services: gateway (builds from Dockerfile runtime target, port 8080), ollama (official image with custom entrypoint that pulls TinyLlama), redis (alpine with healthcheck). Gateway depends on ollama (service_started) and redis (service_healthy).
- `scripts/ollama-entrypoint.sh` — starts `ollama serve` in background, polls readiness for up to 30 seconds, then pulls tinyllama model.

### Key design decisions

1. **Shared httpx.AsyncClient on app.state with lifespan management.** One connection pool for the entire application. The lifespan context manager guarantees `aclose()` runs on shutdown, preventing connection leaks. This avoids the overhead of creating a new TCP connection per request.

2. **120-second timeout.** LLM inference on local hardware is slow. TinyLlama on CPU can take 30-60 seconds per completion. The timeout is deliberately generous to avoid false timeouts during development while still bounding runaway requests.

3. **Pydantic `extra="allow"` on ChatCompletionRequest.** The OpenAI API has many fields the gateway does not yet handle (function_call, tools, response_format). Accepting unknown fields means clients using newer SDK versions do not get 422 errors. The gateway ignores what it does not understand rather than rejecting it.

4. **Server-side ID generation (`chatcmpl-{uuid}`).** Ollama does not return an ID in the OpenAI format. Generating IDs at the gateway guarantees a consistent `chatcmpl-` prefix regardless of which backend responds, making the gateway a drop-in replacement for OpenAI's API.

5. **Token counts default to 0 when absent.** Ollama's `prompt_eval_count` and `eval_count` are optional. Returning 0 instead of null or raising an error is graceful degradation — clients that display token usage see zeros rather than crashes.

6. **structlog with JSON from day one.** Retrofitting structured logging into a codebase is painful. Starting with JSON output and contextvars-based request ID propagation means logs are machine-queryable from the first commit.

7. **Ollama native `/api/chat` instead of its built-in OpenAI compatibility endpoint.** The native endpoint gives full control over option mapping (e.g., `num_predict` in the options dict) and demonstrates real protocol translation — the core value of this project.

8. **Multi-stage Dockerfile with non-root runtime user.** The `test` stage includes pytest but not application dependencies it does not need. The `runtime` stage copies only compiled packages and application code, runs as `appuser`, keeping the image lean and the attack surface small.

### Alternatives considered

1. **aiohttp vs httpx.** httpx was chosen for its cleaner async context manager API, better type annotations, and familiar requests-like interface. aiohttp has a larger ecosystem for WebSocket support, but that is not needed here.

2. **Ollama's built-in OpenAI compatibility endpoint (`/v1/chat/completions`) vs native `/api/chat`.** The built-in endpoint would eliminate the need for translation code, but it hides the complexity this project is designed to showcase. Using the native endpoint demonstrates real parameter mapping (max_tokens to num_predict), response restructuring (flat message to choices array), and error normalization.

3. **Server-side ID generation vs pass-through from backend.** Ollama does not return an `id` field in its response. Generating at the gateway guarantees format consistency. If a future backend does return an ID, the gateway can still override it to maintain the `chatcmpl-` contract.

4. **sync requests vs async httpx.** A proxy that blocks the event loop on every backend call cannot handle concurrent requests. Async is essential for non-blocking I/O when multiple clients are waiting for slow LLM completions simultaneously.

5. **Per-request httpx client vs shared client.** Creating a client per request means a new TCP handshake and TLS negotiation (if applicable) for every call. The shared client enables connection pooling and reuse. The tradeoff is that a misbehaving request cannot poison the pool, but httpx handles this internally with per-connection timeouts.

### Failure modes and edge cases

- **Backend timeout (>120s):** `httpx.TimeoutException` caught in `chat_completion()`, returns 504 with "Backend request timed out". Logged as `ollama_timeout` with base_url and model.
- **Backend HTTP error (e.g., Ollama returns 500):** `httpx.HTTPStatusError` caught, the backend's status code is propagated to the client with the error text. Logged as `ollama_error` with status_code.
- **Malformed backend response (JSON parses but fails Pydantic validation):** `ValidationError` caught, returns 502 "Invalid backend response". Logged as `ollama_invalid_response`.
- **Ollama not ready (model still pulling on startup):** Connection refused from httpx results in a 502 to the client. The `depends_on: service_started` condition only waits for the Ollama container process, not for model readiness. The entrypoint script mitigates this by blocking until the model is pulled, but there is a race window.
- **Model not found in Ollama:** Ollama returns 404, which is propagated through the `HTTPStatusError` path as a 404 to the client.
- **Missing token counts (`prompt_eval_count` is None):** `translate_response()` defaults to 0 via `or 0`. `total_tokens` is computed as the sum, so it is also 0. No error raised.
- **Empty message list:** Caught by Pydantic's `min_length=1` validation on `ChatCompletionRequest.messages`, returns 422 Unprocessable Entity before reaching the translator.
- **Invalid role in message:** Caught by the `Literal["system", "user", "assistant"]` constraint on `ChatMessage.role`, returns 422.
- **Concurrent requests:** `httpx.AsyncClient` manages a connection pool internally. FastAPI's async route handlers yield the event loop during `await`, so other requests proceed. No shared mutable state exists between requests.

### Observability

**Structured logging (structlog, JSON output):**
- Every request: `request_completed` event with `request_id`, `method`, `path`, `status_code`, `duration_ms`.
- Chat completions: `chat_request_received` (model, message_count) and `chat_request_completed` (model, prompt_tokens, completion_tokens, duration_ms) in the route handler.
- Ollama backend: `ollama_completion` event with model, prompt_tokens, completion_tokens, duration_ms.
- Errors: `ollama_timeout` (base_url, model), `ollama_error` (base_url, model, status_code), `ollama_invalid_response` (error details).
- Lifecycle: `gateway_started` and `gateway_stopped` events.

**Request tracing:**
- `X-Request-ID` header propagated from client or generated as UUID4.
- Bound to structlog contextvars so every log line within a request carries the same `request_id`.
- Returned in the response `X-Request-ID` header for client-side correlation.

**Health endpoint:**
- `GET /health` returns `{"status": "ok"}` with 200. Serves as a basic liveness check (no deep dependency checks in Phase 1).

### Testing

**Unit tests (21 tests across 2 files):**
- `tests/unit/test_models.py` (12 tests): `ChatMessage` role validation (valid system/user/assistant, invalid role rejected), `ChatCompletionRequest` validation (minimal valid, all optional fields, missing model, missing messages, empty messages, extra fields tolerated), `ChatCompletionResponse` round-trip serialization (id prefix, object field, created timestamp, role default, finish_reason default), Usage total equals sum.
- `tests/unit/test_ollama_translator.py` (9 tests): `translate_request` (minimal request with no options, request with temperature/max_tokens/top_p mapped to options dict, system message pass-through), `translate_response` (full response with token counts, missing token counts default to 0), `chat_completion` async tests using `httpx.MockTransport` (happy path returns correct model/content/tokens, timeout returns 504, backend 500 propagates status, malformed response returns 502).

**Integration test (1 test):**
- `tests/integration/test_health.py`: Uses `httpx.ASGITransport` with the FastAPI app to test `GET /health` returns 200 with `{"status": "ok"}`. No running server needed — ASGI transport runs the app in-process.

**E2E (manual via Docker):**
- `make up` starts gateway + ollama + redis.
- `curl` to `POST /v1/chat/completions` with TinyLlama model and a user message.
- Verify response has `chatcmpl-` ID, choices array, usage tokens, and valid content from TinyLlama.

**Test infrastructure:**
- `tests/conftest.py` provides `anyio_backend` fixture (asyncio) and `client` fixture (httpx.AsyncClient with ASGITransport).
- Dockerfile `test` stage installs pytest, pytest-asyncio, pytest-httpx alongside the application code. `make test` builds and runs this stage in an isolated container.

### Production gaps

- **No TLS termination.** In production, a reverse proxy (NGINX, Envoy, or a cloud load balancer) would handle TLS. The gateway listens on plain HTTP.
- **No authentication or authorization.** Any client can call any endpoint. Tenant auth is planned for Phase 2.
- **No rate limiting.** No protection against a single client consuming all backend capacity. Planned for Phase 7.
- **No config file.** Ollama URL is a single environment variable. Phase 2 adds a YAML-based backend registry supporting multiple backends.
- **No backend health checking.** The gateway does not proactively check if Ollama is healthy. It discovers failures at request time. Phase 6 adds circuit breakers.
- **No graceful shutdown.** In-flight requests are not drained on SIGTERM. Phase 11 addresses this.
- **Single backend instance.** No failover or load balancing across multiple Ollama instances.
- **No connection pool tuning.** Uses httpx defaults (100 max connections, 10 max keepalive). Production workloads would need tuning based on expected concurrency.
- **No request/response size limits.** A client could send an arbitrarily large message list.
- **`depends_on: service_started` race condition.** The gateway can start before Ollama finishes pulling the model. The entrypoint script mitigates this but does not eliminate the race entirely.

### Interview talking points

- **Why translate at the gateway instead of the client.** Decouples client code from provider-specific APIs, enables transparent backend switching without client changes, and centralizes error handling and observability in one place. Each new backend requires one translator module in the gateway, not N changes across N clients.
- **Why Pydantic for both request AND response validation.** Request validation catches malformed input early (422) with structured error messages. Response validation catches malformed backend output (502) before it reaches the client. Together they form a contract enforcement layer at both boundaries of the gateway.
- **Why async httpx with a shared client on app.state.** Connection pooling avoids per-request TCP handshake overhead. The lifespan context manager guarantees cleanup. Async I/O means the event loop is free during the 30-60 second LLM inference wait, allowing other requests to proceed concurrently.

### Likely interview questions

**Q: "Why not just use Ollama's built-in OpenAI-compatible endpoint?"**
**A:** Using the native `/api/chat` endpoint gives full control over parameter mapping (e.g., `max_tokens` to `num_predict` in the options dict) and demonstrates real protocol translation. The built-in compatibility endpoint is a shortcut that hides the translation complexity — exactly the complexity this project is designed to showcase. It also means the gateway controls the response format precisely, rather than depending on Ollama's compatibility implementation to match the exact OpenAI schema.

**Q: "How would you handle the Ollama container not being ready when the gateway starts?"**
**A:** Currently `depends_on` with `service_started` only waits for the container process, not for the model to load. The custom entrypoint script (`scripts/ollama-entrypoint.sh`) polls for readiness and pulls the model before accepting connections, but there is still a race window. The gateway returns 502 if Ollama is not ready. Phase 2 adds a backend registry with health state tracking, and Phase 6 adds circuit breakers that detect and route around unhealthy backends automatically.

**Q: "What happens if two requests arrive simultaneously?"**
**A:** `httpx.AsyncClient` manages concurrent requests through its internal connection pool. FastAPI's async route handlers release the event loop during `await`, so the second request proceeds without waiting for the first to complete. There is no shared mutable state between requests — each request gets its own Pydantic model instances, its own structlog context (via contextvars), and its own response object. The connection pool handles multiplexing at the transport layer.

**Q: "Why generate IDs server-side instead of using what the backend returns?"**
**A:** Ollama's native API does not return an `id` field in the chat completion response. Even if it did, generating IDs at the gateway guarantees a consistent `chatcmpl-{hex}` format regardless of which backend handles the request. When the gateway supports multiple backends, clients see a uniform ID format. This also means ID generation is deterministic in testing — the format is always known.

## Backend Configuration & Provider Registry

### Why this exists

Phase 1 hardcoded the Ollama backend URL in a single environment variable (`OLLAMA_BASE_URL`). Adding a second backend or a second tenant required code changes to the route handler. There was no access control — any client could call any model on any backend. There was no way to reload configuration without restarting the entire gateway process. This phase introduces a declarative YAML configuration file, a typed provider registry, per-tenant API key authentication, model-level authorization, and hot-reload without downtime.

### How it works

1. On startup, the `lifespan()` context manager in `gateway/main.py` reads the `CONFIG_PATH` environment variable (default: `config/backends.yaml`), calls `load_config()` from `gateway/config.py` which parses the YAML and validates it through Pydantic models (`GatewayConfig` containing `BackendConfig` and `TenantConfig` lists). If validation fails, `ConfigError` is raised and `sys.exit(1)` terminates the process immediately.
2. The validated `GatewayConfig` is passed to `Registry.__init__()` in `gateway/config.py`, which builds three data structures: a `backends` dict (name to `BackendConfig`), a `model_to_backends` reverse index (model name to list of `BackendConfig`), and an `api_key_to_tenant` dict (resolved API key string to `TenantConfig`). API keys are resolved from environment variables via `os.environ.get(tenant.api_key_env)` — if the env var is missing or empty, the tenant is logged as a warning and skipped.
3. The `Registry` instance is stored on `app.state.registry` alongside `app.state.config_path` and `app.state.http_client`.
4. When a request arrives at `POST /v1/chat/completions`, FastAPI's dependency injection calls `get_current_tenant()` in `gateway/auth.py`. This function extracts the `Authorization` header, validates it starts with `"Bearer "`, strips the prefix, and looks up the token in `registry.api_key_to_tenant`. If the token is missing, malformed, or not found, it raises `HTTPException(401)`.
5. The route handler in `gateway/routes/chat.py` receives the authenticated `TenantConfig` via `Depends(get_current_tenant)`. It checks whether the requested model is in `tenant.allowed_models` (or if the tenant has wildcard `"*"` access). If denied, it raises `HTTPException(403)`.
6. The handler calls `registry.find_backend_for_model()`, which returns the first `BackendConfig` whose `models` list contains the requested model, or `None` (resulting in a 404).
7. The backend's `base_url` is passed to the Ollama translator (`ollama.chat_completion()`), replacing the hardcoded `OLLAMA_BASE_URL` from Phase 1.
8. Hot-reload via `POST /admin/reload` in `gateway/routes/admin.py`: re-reads the YAML file from `app.state.config_path`, validates it, constructs a new `Registry`, and atomically swaps `app.state.registry`. If validation fails, the old registry is retained and a 400 is returned.
9. Hot-reload via SIGHUP: the `handle_sighup()` signal handler registered in `lifespan()` performs the same load-validate-swap sequence. On non-Unix platforms (no `SIGHUP`), the handler registration silently skips via `except (OSError, AttributeError)`.

### Implementation

**Config models** (`gateway/config.py`):
- `BackendConfig` — name, provider (Literal: ollama/openai/anthropic/vllm), base_url, optional api_key_env, models (min_length=1), weight (default 1), max_concurrent (default 10), timeout_ms (default 120000).
- `TenantConfig` — id, optional name, api_key_env, allowed_models (min_length=1), priority (default 1), optional rate_limit_rps/rate_limit_rpm/token_budget_daily.
- `GatewayConfig` — backends list (min_length=1), tenants list (min_length=1), plus a `@model_validator(mode="after")` (`validate_uniqueness`) that rejects duplicate backend names and duplicate tenant IDs.
- `ConfigError` — custom exception used by `load_config()` to wrap file-not-found, YAML parse, and Pydantic validation failures into a single error type.

**Config loader** (`gateway/config.py: load_config()`):
- Reads file via `Path.read_text()`, catches `FileNotFoundError` and `OSError`.
- Parses with `yaml.safe_load()`, catches `yaml.YAMLError`.
- Validates type is `dict` (not a bare scalar or list).
- Calls `GatewayConfig.model_validate(data)`, catches any exception.
- All failures raise `ConfigError` with a descriptive message.

**Registry** (`gateway/config.py: Registry`):
- `__init__()` builds `backends` (dict by name), `model_to_backends` (reverse index via `setdefault`), and `api_key_to_tenant` (resolved from env vars with warning on missing).
- `find_backend_for_model(model) -> BackendConfig | None` returns first match from the reverse index.

**Auth dependency** (`gateway/auth.py: get_current_tenant()`):
- FastAPI dependency using `Header(None)` for the `authorization` parameter.
- Three rejection paths: missing/malformed header, empty token after stripping prefix, token not in registry. All raise `HTTPException(401)`.
- Logs `auth_failed` with reason on rejection, `auth_success` with tenant_id on success.

**Admin routes** (`gateway/routes/admin.py`):
- `POST /admin/reload` — loads config, builds new registry, swaps `app.state.registry`. Returns `{"status": "reloaded", "backends": N, "tenants": N}` on success, 400 on failure.
- `GET /admin/backends` — returns list of backend summaries (name, provider, models, health="unknown").

**Chat route** (`gateway/routes/chat.py`):
- `Depends(get_current_tenant)` added to `chat_completions()` signature.
- Model authorization check: `"*" not in tenant.allowed_models and chat_request.model not in tenant.allowed_models` raises 403.
- Backend lookup: `registry.find_backend_for_model(chat_request.model)` replaces the hardcoded `OLLAMA_BASE_URL`.
- Logging enriched with `tenant_id` and `backend` name.

**Lifespan** (`gateway/main.py: lifespan()`):
- Loads config via `load_config(CONFIG_PATH)`, catches `ConfigError` and calls `sys.exit(1)`.
- Stores `config_path`, `registry`, and `http_client` on `app.state`.
- Registers `handle_sighup` signal handler for hot-reload.
- Logs `gateway_started` with backend and tenant counts.

**Config file** (`config/backends.yaml`):
- Defines one backend (`ollama-local`, provider ollama, base_url `http://ollama:11434`, models: [tinyllama]).
- Defines two tenants: `tenant-alpha` (allowed_models: [tinyllama]) and `tenant-beta` (allowed_models: ["*"] wildcard).
- API keys reference env vars `TENANT_ALPHA_KEY` and `TENANT_BETA_KEY`.

**Infrastructure** (`docker-compose.yaml`):
- Gateway service gains `CONFIG_PATH`, `TENANT_ALPHA_KEY`, `TENANT_BETA_KEY` environment variables.
- Config directory bind-mounted as read-only volume: `./config:/app/config:ro`.

### Key design decisions

1. **YAML config over database.** Configuration is version-controlled alongside the code, auditable via git history, and reviewable in pull requests. A database-backed config store adds operational complexity (schema migrations, connection management, consistency concerns) without clear benefit for a gateway with a small number of backends and tenants.

2. **Pydantic fail-fast validation.** Invalid configuration stops startup immediately via `ConfigError` caught in `lifespan()` with `sys.exit(1)`. This is deliberate: a gateway with a misconfigured backend or tenant is worse than a gateway that is down, because it would silently drop or misroute requests. The `model_validator` also rejects duplicate backend names and tenant IDs at parse time.

3. **FastAPI `Depends()` over middleware for authentication.** A middleware runs on every request including `/health` and `/admin/backends`. The dependency injection approach applies authentication only to routes that declare `tenant: TenantConfig = Depends(get_current_tenant)`. This keeps health checks unauthenticated (required for container orchestration probes) without maintaining an exclusion list.

4. **API keys as environment variable references, not inline values.** The YAML file contains `api_key_env: TENANT_ALPHA_KEY`, not the key itself. This follows twelve-factor app principles: secrets live in the environment (or a secrets manager), the config file is safe to commit to version control.

5. **Registry on `app.state` rather than a module-level singleton.** Consistent with the Phase 1 pattern for `http_client`. Avoids module-level mutable state that complicates testing (no need to reset global state between tests). FastAPI's `request.app.state` is the idiomatic location for request-scoped shared resources.

6. **Atomic swap for hot-reload.** `app.state.registry = new_registry` is a single reference assignment, which is atomic under CPython's GIL. In-flight requests that already hold a reference to the old registry continue using it safely. New requests pick up the new registry. No locking, no request draining, no downtime.

7. **Model access check in route handler, not in auth dependency.** The auth dependency answers "who is this?" (identity). The route handler answers "can they do this?" (authorization). Separating identity from authorization keeps the auth dependency reusable across routes with different authorization logic. A future `/v1/embeddings` route can use the same `get_current_tenant` with different model-access rules.

8. **`ConfigError` exception instead of `sys.exit()` in the loader.** The `load_config()` function raises `ConfigError` on any failure. The `lifespan()` function catches it and calls `sys.exit(1)` for startup. The `POST /admin/reload` handler catches it and returns 400, preserving the old registry. This separation means the loader is testable (tests catch exceptions, not process exits) and the error handling policy is owned by the caller.

### Alternatives considered

1. **Middleware vs dependency for auth.** A middleware would centralize auth in one place and guarantee it runs on every request. But it requires maintaining an exclusion list for unauthenticated paths (`/health`, `/admin/*`), which is error-prone — forgetting to exclude a path locks out monitoring. The dependency approach is opt-in: only routes that declare the dependency get authentication. The tradeoff is that a developer could forget to add the dependency to a new route, but that is caught by integration tests.

2. **Module-level singleton vs `app.state`.** A singleton `Registry` at module level would be simpler to access (import and use) but harder to test (requires monkeypatching or resetting global state). It also complicates hot-reload — replacing a module-level singleton requires care to ensure all importers see the new reference. `app.state` is scoped to the application instance, making test isolation straightforward (each test creates its own `FastAPI()` with its own state).

3. **Database-backed config vs file-based.** A database (PostgreSQL, Redis) would enable runtime config changes via API without file system access. But it adds a hard dependency on the database for startup — if the database is down, the gateway cannot start. File-based config is self-contained, version-controlled, and does not introduce a circular dependency (the gateway proxies to backends; it should not depend on a backend to know which backends exist).

4. **Fail-hard vs warn-and-skip for missing tenant env vars.** The implementation warns and skips tenants whose `api_key_env` resolves to an empty string. The alternative is to fail startup entirely. Warning-and-skip was chosen because a missing tenant key is a partial degradation (that tenant cannot authenticate) rather than a total failure (no tenants can authenticate). The startup log clearly reports the count of loaded tenants, making the omission visible.

### Failure modes and edge cases

- **Missing config file at startup.** `load_config()` raises `ConfigError("Config file not found")`, `lifespan()` catches it, logs `config_load_failed`, and calls `sys.exit(1)`. The gateway does not start.
- **Invalid YAML syntax.** `yaml.safe_load()` raises `yaml.YAMLError`, wrapped as `ConfigError("Invalid YAML")`. Same exit path as above.
- **Valid YAML, invalid schema** (e.g., empty backends list, missing required field, invalid provider). Pydantic validation raises, wrapped as `ConfigError("Config validation failed")`. Same exit path.
- **Missing tenant API key env var.** `os.environ.get(tenant.api_key_env)` returns `""`. The tenant is logged as `tenant_api_key_missing` with tenant_id and env_var name, then skipped. Other tenants load normally. The tenant cannot authenticate until the env var is set and config is reloaded.
- **Bad config on reload** (POST /admin/reload or SIGHUP). `load_config()` raises `ConfigError`. For the admin endpoint, this is caught and returned as HTTP 400 — the old registry remains active. For SIGHUP, the error is logged as `sighup_reload_failed` and the old registry remains.
- **Concurrent reload requests.** Two simultaneous `POST /admin/reload` calls both read the file, both build a new registry, both assign to `app.state.registry`. The last writer wins. Because each assignment is atomic (GIL-protected pointer swap), there is no corruption — the registry is either the old one or one of the new ones. In practice, this is harmless because both reload from the same file.
- **SIGHUP on non-Unix platforms** (Windows). `signal.signal(signal.SIGHUP, ...)` raises `AttributeError` (no `SIGHUP` constant) or `OSError`. Caught by `except (OSError, AttributeError)`, logged as `sighup_not_available`. The admin reload endpoint still works.
- **Model served by no backend.** `registry.find_backend_for_model()` returns `None`. The route handler raises `HTTPException(404, "No backend available for model: ...")`.
- **Tenant with wildcard `"*"` in allowed_models.** The check `"*" not in tenant.allowed_models` short-circuits — the tenant passes model authorization for any model string. Authorization passes, but the model still must exist in a backend's model list or the request gets 404.

### Observability

- **Startup:** `gateway_started` log event includes `backends` (count) and `tenants` (count), giving operators immediate visibility into how many backends and tenants were loaded.
- **Tenant key warnings:** `tenant_api_key_missing` log event with `tenant_id` and `env_var` fields when a tenant's API key env var is unset. This surfaces misconfiguration without blocking startup.
- **Auth events:** `auth_success` (tenant_id) on successful authentication. `auth_failed` (reason: `missing_or_invalid_header`, `empty_token`, or `invalid_key`) on rejection. All events carry the `request_id` from the middleware's contextvars binding.
- **Chat request logs:** `chat_request_received` and `chat_request_completed` now include `tenant_id` and `backend` (name), enabling per-tenant and per-backend observability. Duration, model, and token counts continue from Phase 1.
- **Config reload:** `config_reloaded` (backends count, tenants count) on successful admin reload. `config_reload_failed` (error) on failure. `config_reloaded_via_sighup` on successful SIGHUP reload. `sighup_reload_failed` (error) on SIGHUP failure.
- **Admin endpoints:** `GET /admin/backends` returns backend name, provider, models, and health status (currently hardcoded to `"unknown"` — will be populated by Phase 6 health checks).

### Testing

**Unit tests — config** (`tests/unit/test_config.py`, 21 tests):
- `TestBackendConfig` (4 tests): valid backend with defaults, missing name rejected, invalid provider rejected, empty models list rejected.
- `TestTenantConfig` (3 tests): valid tenant with defaults, missing id rejected, missing api_key_env rejected.
- `TestGatewayConfig` (5 tests): valid config, duplicate backend names rejected, duplicate tenant IDs rejected, empty backends rejected, empty tenants rejected.
- `TestLoadConfig` (4 tests): valid YAML file loads correctly, missing file raises `ConfigError`, invalid YAML raises `ConfigError`, invalid schema raises `ConfigError`.
- `TestRegistry` (5 tests): backends indexed by name, model-to-backends reverse index, `find_backend_for_model` returns correct backend or None, API key resolution from env vars, missing API key env var excludes tenant with warning.

**Unit tests — auth** (`tests/unit/test_auth.py`, 6 tests):
- `TestGetCurrentTenant`: valid key returns tenant (200), invalid key returns 401, missing header returns 401, malformed header (Basic instead of Bearer) returns 401, empty Bearer token returns 401, lowercase "bearer" returns 401 (case-sensitive check).

**Unit tests — admin** (`tests/unit/test_admin.py`, 4 tests):
- `TestListBackends`: returns all backends with name, provider, models, health.
- `TestReloadConfig`: successful reload swaps registry, bad config file returns 400 and retains old registry, missing config file returns 400.

**Integration tests — auth** (`tests/integration/test_auth.py`, 5 tests):
- `TestAuthenticatedChatFlow` using the full app with lifespan (loads real config, creates real registry): valid key returns 200 with mocked backend response, bad key returns 401, missing auth returns 401, disallowed model returns 403 (tenant-alpha cannot access gpt-4o), wildcard tenant passes auth but gets 404 for model with no backend.

**E2E (Docker):**
- `make up` starts the full stack. `curl` with `Authorization: Bearer test-alpha-key` to `POST /v1/chat/completions` verifies the end-to-end flow through config loading, auth, model access check, backend lookup, and Ollama translation.

### Production gaps

- **Admin endpoints are unauthenticated.** `POST /admin/reload` and `GET /admin/backends` have no auth requirement. In production, these would need either a separate admin API key, IP allowlisting, or placement on an internal-only network interface.
- **No config file watch.** Changes to `backends.yaml` require an explicit `POST /admin/reload` or `SIGHUP`. A file watcher (inotify/fsevents) or periodic poll would enable automatic reload, but adds complexity and a potential race condition with partial file writes.
- **Health always "unknown".** `GET /admin/backends` returns `"health": "unknown"` for every backend. Active health checking is planned for Phase 6 with circuit breakers.
- **First-match routing only.** `find_backend_for_model()` returns the first backend in the list that serves the model. There is no load balancing, no weight-based selection, and no failover to a second backend. The `weight` field on `BackendConfig` is defined but unused until Phase 5 (weighted load balancing).
- **No per-model auth granularity beyond the allowed list.** A tenant either has access to a model or does not. There is no rate limiting per model, no token budget enforcement, and no cost tracking. The `rate_limit_rps`, `rate_limit_rpm`, and `token_budget_daily` fields on `TenantConfig` are defined but not enforced until later phases.
- **No API key rotation.** Changing a tenant's API key requires updating the environment variable and reloading config. There is no support for multiple active keys per tenant (for zero-downtime rotation).
- **SIGHUP handler runs synchronously in a signal context.** The `handle_sighup` function performs file I/O and object construction inside a signal handler, which is technically unsafe for non-reentrant operations. In practice, CPython's signal handling defers execution to the main thread between bytecode instructions, making this safe for typical usage. A production system would use `asyncio.get_event_loop().call_soon_threadsafe()` to schedule the reload on the event loop.

### Interview talking points

- **Config-as-code with environment-separated secrets.** The YAML file is version-controlled and auditable via git. Secrets (API keys) are referenced by env var name, never stored in the file. This enables the config to be committed, reviewed, and diffed while keeping secrets in the deployment environment (Docker Compose env vars, Kubernetes secrets, Vault).
- **Atomic pointer swap for zero-downtime reload.** `app.state.registry = new_registry` is a single reference assignment, atomic under CPython's GIL. In-flight requests holding the old registry complete safely. No locking, no request draining, no coordination required. The worst case is that two concurrent reloads race and the last write wins — both produce valid registries from the same file.
- **Dependency injection over middleware for auth.** FastAPI's `Depends()` applies auth only where declared, avoiding the maintenance burden of a middleware exclusion list. Health checks stay unauthenticated without any special-case code. The auth dependency is a pure function of the request headers and registry state, making it trivially testable with a minimal FastAPI app and `httpx.ASGITransport`.

### Likely interview questions

**Q: "How do you prevent dropping in-flight requests during a config reload?"**
**A:** The reload performs an atomic pointer swap: `app.state.registry = new_registry`. Under CPython's GIL, this is a single reference assignment. Any request handler that already retrieved the old registry via `request.app.state.registry` holds its own reference — Python's reference counting ensures the old `Registry` object stays alive until all references are released. New requests pick up the new registry. There is no lock, no pause, and no request draining needed. The old and new registries can coexist simultaneously for the duration of in-flight requests.

**Q: "Why not store API keys directly in the YAML config file?"**
**A:** Twelve-factor app methodology: config files are committed to version control, and secrets must never appear in git history. The YAML contains `api_key_env: TENANT_ALPHA_KEY` — a reference to an environment variable, not the key value. The `Registry.__init__()` method resolves these references at runtime via `os.environ.get()`. This means the same config file works across environments (dev, staging, production) with different API keys injected through Docker Compose env vars, Kubernetes secrets, or a secrets manager like Vault.

**Q: "What happens if someone pushes a bad config and triggers a reload?"**
**A:** The reload path in `POST /admin/reload` (`gateway/routes/admin.py`) wraps the entire load-validate-build sequence in a try/except for `ConfigError`. If `load_config()` fails (bad YAML, schema violation), or `Registry()` construction fails, the exception is caught, the error is logged as `config_reload_failed`, and the endpoint returns HTTP 400 with the error message. Critically, `app.state.registry` is never assigned — the old, working registry remains active. The same pattern applies to SIGHUP reload in `gateway/main.py`: the error is logged as `sighup_reload_failed` and the old registry is retained. The gateway continues serving requests with the last known-good configuration.

**Q: "Why separate identity (auth dependency) from authorization (route handler)?"**
**A:** The `get_current_tenant()` dependency in `gateway/auth.py` answers "who is making this request?" by resolving a Bearer token to a `TenantConfig`. The model access check in `gateway/routes/chat.py` answers "is this tenant allowed to use this model?" This separation means the same auth dependency can be reused for future routes (embeddings, completions, admin) that have different authorization rules. It also makes each piece independently testable: auth tests do not need to know about model access, and model access tests can inject a pre-authenticated tenant directly.
