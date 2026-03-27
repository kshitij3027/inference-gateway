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

## Request/Response Translation — Multi-Backend

### Why this exists

Phase 1 built translation for a single backend: Ollama. Clients that needed to use OpenAI or Anthropic models had no path through the gateway — they would need to call those providers directly, bypassing all gateway features (auth, logging, unified response format). Each provider has its own API format: OpenAI uses `/v1/chat/completions` with `Authorization: Bearer` headers, Anthropic uses `/v1/messages` with `x-api-key` headers and a fundamentally different request/response schema (top-level `system` field, content blocks instead of a string, `stop_reason` instead of `finish_reason`, `input_tokens`/`output_tokens` instead of `prompt_tokens`/`completion_tokens`). Without translators for each provider, the gateway cannot fulfill its core promise: clients speak one protocol (OpenAI-compatible), and the gateway handles the rest.

### How it works

1. Client sends `POST /v1/chat/completions` with an OpenAI-format JSON body, same as Phase 1.
2. `get_current_tenant()` authenticates the request via Bearer token (Phase 2).
3. The route handler in `gateway/routes/chat.py` calls `registry.find_backend_for_model()`, which returns a `BackendConfig`. The `BackendConfig.provider` field (one of `ollama`, `openai`, `anthropic`, `vllm`) identifies which translator to use.
4. The `TRANSLATORS` dict in `gateway/routes/chat.py` maps provider strings to `chat_completion` functions: `{"ollama": ollama.chat_completion, "openai": openai_backend.chat_completion, "anthropic": anthropic_backend.chat_completion}`. If the provider is not in the dict, the handler returns 400.
5. The selected translator runs its three-step pipeline:
   - **`translate_request()`** converts the OpenAI-format `ChatCompletionRequest` into the provider's native format and builds provider-specific headers.
   - **HTTP POST** sends the translated request to the backend's `base_url` plus the provider-specific path (`/api/chat` for Ollama, `/v1/chat/completions` for OpenAI, `/v1/messages` for Anthropic).
   - **`translate_response()`** converts the provider's native response back into a `ChatCompletionResponse` with a gateway-generated `chatcmpl-{uuid}` ID.
6. The route handler logs the completed request with provider, backend name, tenant, token counts, and duration, then returns the unified `ChatCompletionResponse`.

### Implementation

**Anthropic translator** (`gateway/backends/anthropic.py`):
- `translate_request(request: ChatCompletionRequest, backend: BackendConfig) -> tuple[dict, dict]` — Extracts system messages from the messages list and hoists them into a top-level `system` field (Anthropic requires system content outside the messages array). Non-system messages are converted to `AnthropicMessage` objects with `AnthropicContentBlock` content. Sets `max_tokens` to `request.max_tokens or 4096` (Anthropic requires this field; 4096 is the default). Maps `stop` (string or list) to `stop_sequences` (always a list). Builds headers with `x-api-key` (resolved from `backend.api_key_env` via `os.environ.get()`), `anthropic-version: 2023-06-01`, and `content-type: application/json`.
- `translate_response(data: dict, model: str) -> ChatCompletionResponse` — Validates the raw dict through `AnthropicResponse.model_validate(data)`. Iterates content blocks, concatenating text from blocks where `type == "text"`. Maps `stop_reason` to `finish_reason` via `STOP_REASON_MAP`: `end_turn` -> `stop`, `max_tokens` -> `length`, `stop_sequence` -> `stop`, `tool_use` -> `tool_calls`. Maps `usage.input_tokens` to `prompt_tokens` and `usage.output_tokens` to `completion_tokens`.
- `chat_completion(client, backend, request)` — POSTs to `{backend.base_url}/v1/messages`. Handles `TimeoutException` (504), `HTTPStatusError` (propagated status code), and `ValidationError`/`ValueError`/`KeyError` on response parsing (502). Logs `anthropic_completion` with model, token counts, and duration.

**OpenAI translator** (`gateway/backends/openai.py`):
- `translate_request(request: ChatCompletionRequest, backend: BackendConfig) -> tuple[dict, dict]` — Near-passthrough: calls `request.model_dump(exclude_none=True)` and sets `stream=False`. Builds headers with `Authorization: Bearer {api_key}` (only if key is non-empty) and `Content-Type: application/json`.
- `translate_response(data: dict, model: str) -> ChatCompletionResponse` — Extracts `choices[0].message.content`, `finish_reason`, and usage fields via dict access. Raises `ValueError` on malformed structure (missing keys, empty choices). Does not use Pydantic validation for the response — the format is already OpenAI-compatible, so dict parsing is sufficient.
- `chat_completion(client, backend, request)` — POSTs to `{backend.base_url}/v1/chat/completions`. Same error handling pattern as Anthropic: `TimeoutException` (504), `HTTPStatusError` (propagated), `ValueError`/`KeyError` (502). Logs `openai_completion`.

**Ollama translator** (`gateway/backends/ollama.py`):
- Refactored signature from `chat_completion(client, base_url, request)` to `chat_completion(client: httpx.AsyncClient, backend: BackendConfig, request: ChatCompletionRequest)` for consistency with the other translators. The `base_url` is now read from `backend.base_url` instead of a separate parameter. Internal `translate_request` and `translate_response` are unchanged from Phase 1.

**Route dispatch** (`gateway/routes/chat.py`):
- `TRANSLATORS` dict at module level maps provider strings to `chat_completion` callables. The route handler calls `TRANSLATORS.get(backend.provider)` and raises `HTTPException(400, "Unsupported provider: ...")` if the provider is not found.
- All three translators are called with the same signature: `translator(client=request.app.state.http_client, backend=backend, request=chat_request)`.

**Anthropic Pydantic models** (`gateway/models.py`):
- `AnthropicContentBlock` — `type: str = "text"`, `text: str`. Represents a single content block in an Anthropic message.
- `AnthropicMessage` — `role: str`, `content: list[AnthropicContentBlock]`. Used for building the translated request body.
- `AnthropicRequest` — `model`, `messages`, optional `system`, `max_tokens` (default 4096), optional `temperature`/`top_p`/`stop_sequences`. Documents the Anthropic API contract.
- `AnthropicUsage` — `input_tokens: int`, `output_tokens: int`. Anthropic's usage field naming differs from OpenAI's.
- `AnthropicResponse` — `id`, `type`, `role`, `content` (list of `AnthropicContentBlock`), `model`, optional `stop_reason`, `usage` (`AnthropicUsage`). Uses `ConfigDict(extra="allow")` to tolerate additional fields returned by the real API or mock servers without validation errors.

**Token counting utility** (`gateway/token_counting.py`):
- `count_tokens(text: str, model: str) -> int` — Tries `tiktoken.encoding_for_model(model)` first for accurate token counts on OpenAI models. Falls back to `len(text) // 4` (character-based approximation) when tiktoken is unavailable or the model is not recognized. Prepared for use in future phases (caching, rate limiting).

**Mock backends** (`docker-compose.yaml`):
- 8 new services using `zerob13/mock-openai-api:latest`: `mock-openai-1` through `mock-openai-5` (ports 9001-9005) and `mock-anthropic-1` through `mock-anthropic-3` (ports 9006-9008). All use the same image, which serves both OpenAI-compatible and Anthropic-compatible endpoints.
- Gateway `depends_on` updated to include `mock-openai-1` and `mock-anthropic-1`.
- Gateway environment gains `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` (test values).

**Backend configuration** (`config/backends.yaml`):
- Three backends: `ollama-local` (provider: ollama, models: [tinyllama]), `mock-openai-1` (provider: openai, base_url: `http://mock-openai-1:3000`, models: [mock-gpt-markdown]), `mock-anthropic-1` (provider: anthropic, base_url: `http://mock-anthropic-1:3000/anthropic`, models: [mock-claude-markdown]).
- Note: `mock-anthropic-1` has base_url ending in `/anthropic` so the translator's `{base_url}/v1/messages` resolves to the mock server's Anthropic-compatible endpoint path.
- Tenants updated: `tenant-alpha` allowed_models now includes `mock-gpt-markdown` and `mock-claude-markdown`.

### Key design decisions

1. **Function registry dict over abstract base class.** Three providers with identical call signatures do not need an inheritance hierarchy. A dict mapping `{"ollama": fn, "openai": fn, "anthropic": fn}` is simpler to reason about, simpler to test (mock one function, not a class), and simpler to extend (add one entry). An ABC would add boilerplate (`class BaseTranslator(ABC)`, `@abstractmethod`, concrete subclasses) without providing any benefit — there is no shared state across calls and no polymorphic behavior beyond dispatch.

2. **Unified `(client, backend, request)` signature for all translators.** Every `chat_completion` function takes the same three arguments. This enables the route handler to call any translator with the same code path: `translator(client=..., backend=..., request=...)`. Ollama was refactored from `(client, base_url, request)` to `(client, backend, request)` to match. The `backend: BackendConfig` parameter gives each translator access to `base_url`, `api_key_env`, `timeout_ms`, and any other backend-specific config without expanding the function signature.

3. **Anthropic system message extraction.** OpenAI treats system messages as regular entries in the messages array with `role: "system"`. Anthropic requires system content as a separate top-level `system` field, outside the messages array. The translator iterates the messages list, separates system-role messages, concatenates their content with double newlines, and places the result in the `system` field. Non-system messages are converted to `AnthropicMessage` objects. This is a semantic mapping, not just a format conversion — the two APIs model system instructions differently.

4. **`max_tokens` defaults to 4096 for Anthropic.** Anthropic's Messages API requires `max_tokens` as a mandatory field. OpenAI's API treats it as optional (defaults to model-dependent limits). Rather than rejecting requests that omit `max_tokens`, the translator defaults to 4096 — a reasonable upper bound that does not silently truncate most responses. The client can always override this by including `max_tokens` in their request.

5. **Gateway-generated IDs for all providers.** `ChatCompletionResponse` generates `chatcmpl-{uuid.uuid4().hex[:24]}` via a `Field(default_factory=...)`. This means every response from every provider carries the same ID format. OpenAI's real API returns its own IDs, but the gateway overrides them to maintain a consistent contract. This decision was made in Phase 1 for Ollama (which returns no ID) and is now enforced uniformly across all providers.

6. **Mock Anthropic `base_url` includes `/anthropic` path prefix.** The `zerob13/mock-openai-api` image serves Anthropic endpoints under a `/anthropic` prefix. Rather than adding path-routing logic to the translator, the `base_url` in `config/backends.yaml` is set to `http://mock-anthropic-1:3000/anthropic`. The translator always appends `/v1/messages`, so the final URL is `http://mock-anthropic-1:3000/anthropic/v1/messages`. This keeps the translator code clean — it does not need to know about mock server routing. In production, the `base_url` would be `https://api.anthropic.com` and the final URL would be `https://api.anthropic.com/v1/messages`.

7. **Same Docker image for all mock services.** `zerob13/mock-openai-api` serves both OpenAI and Anthropic-compatible endpoints. Using one image for 8 containers (5 OpenAI, 3 Anthropic) reduces Docker pull time and disk usage. The extra mock instances (beyond the one per provider used in Phase 3) are pre-provisioned for Phase 5 (load balancing) and Phase 6 (failover), avoiding Docker Compose changes in later phases.

8. **`ConfigDict(extra="allow")` on `AnthropicResponse`.** The real Anthropic API and mock servers may return fields not modeled in the Pydantic class (e.g., `stop_sequence`, `type` variants for tool use). Strict validation would reject these responses. `extra="allow"` tolerates unknown fields, extracting only the fields the translator needs. This mirrors the `extra="allow"` decision on `ChatCompletionRequest` from Phase 1 — be liberal in what you accept.

### Alternatives considered

1. **Abstract base class with subclasses vs function dict.** An ABC like `class BackendTranslator(ABC)` with `translate_request()`, `translate_response()`, and `chat_completion()` abstract methods would enforce the interface at the type level. However, the translators share no state, no constructor logic, and no common implementation code. The function dict achieves the same dispatch with less code. If a fourth provider required shared logic (e.g., retry with backoff), that logic could be added as a decorator or utility function without refactoring to classes.

2. **Per-provider Pydantic models vs dict parsing for OpenAI passthrough.** The Anthropic translator uses Pydantic models (`AnthropicResponse`) for response validation because the Anthropic format differs significantly from OpenAI and needs structural validation. The OpenAI translator uses raw dict access for the response because the format is already OpenAI-compatible — adding Pydantic models would be redundant validation of a format the gateway already understands. The tradeoff is weaker type safety on the OpenAI path, but the response is immediately wrapped in `ChatCompletionResponse` (which is Pydantic-validated), so malformed data is still caught.

3. **Inline token counting vs utility module.** Token counting logic (`tiktoken` with `chars/4` fallback) could live inside each translator. Extracting it to `gateway/token_counting.py` makes it reusable across future phases: semantic caching (Phase 4) needs token counts for cache key generation, rate limiting (Phase 7) needs token counts for budget enforcement. The utility module avoids duplicating the `try: tiktoken / except: fallback` pattern in every translator.

4. **Single mock image vs provider-specific mock images.** Using separate mock images per provider (e.g., one for OpenAI, one for Anthropic) would provide more realistic API simulation. However, `zerob13/mock-openai-api` serves both protocols from the same image. Using one image simplifies the Docker Compose file, reduces image pull overhead, and is sufficient for testing the translation layer — the goal is to verify that translators correctly map request/response formats, not to simulate provider-specific edge cases like rate limiting or content filtering.

### Failure modes and edge cases

- **Unsupported provider in registry.** If a `BackendConfig` has a `provider` value not in the `TRANSLATORS` dict (e.g., `vllm` is defined in the `Literal` type but has no translator yet), the route handler returns `HTTPException(400, "Unsupported provider: vllm")`. The request never reaches a backend.
- **Anthropic request missing `max_tokens`.** The translator defaults to 4096 via `request.max_tokens or 4096`. This is graceful degradation — the request proceeds rather than failing. Clients that need a different limit must specify it explicitly.
- **Mock container not ready.** If a mock service has not finished starting when the gateway sends a request, `httpx` raises a connection error, which propagates as `HTTPStatusError` or a lower-level `httpx.ConnectError`, resulting in a 502 to the client. The `depends_on: service_started` condition in Docker Compose only ensures the container process is running, not that the HTTP server is accepting connections.
- **Malformed backend response.** For Anthropic: `AnthropicResponse.model_validate(data)` raises `ValidationError`, caught as 502. For OpenAI: missing `choices` key or empty `choices` list raises `KeyError`/`IndexError` wrapped as `ValueError`, caught as 502. For Ollama: `OllamaResponse.model_validate()` raises `ValidationError`, caught as 502.
- **Empty content blocks in Anthropic response.** If the Anthropic response contains content blocks but none have `type == "text"`, the `text_parts` list is empty and `"".join(text_parts)` produces an empty string. The response is valid but has empty content — no crash, no error.
- **OpenAI response with `content: null` (tool calls).** When an OpenAI-compatible backend returns a tool_calls response, `message.content` is `null`. The translator calls `message.get("content", "")` which returns `None` (not the default), and this `None` propagates into `ChatMessageResponse(content=None)`, which fails Pydantic validation because `content` is typed as `str`. This surfaces as a `ValueError` caught by the 502 handler. Full tool_calls support is a future enhancement.
- **Missing API key environment variable.** If `backend.api_key_env` references an env var that is not set, `os.environ.get(backend.api_key_env or "", "")` returns an empty string. For Anthropic, this sends an empty `x-api-key` header — the real API would reject it with 401. For OpenAI, the empty key check skips the `Authorization` header entirely. Against mock servers, this is irrelevant because they do not validate keys.
- **Stop field type mismatch for Anthropic.** OpenAI accepts `stop` as either a string or a list. Anthropic requires `stop_sequences` as a list. The translator checks `isinstance(request.stop, str)` and wraps a single string in a list: `[request.stop]`. Lists pass through directly.

### Observability

**Per-provider log events:**
- Anthropic: `anthropic_completion` (model, prompt_tokens, completion_tokens, duration_ms), `anthropic_timeout` (base_url, model), `anthropic_error` (base_url, model, status_code), `anthropic_invalid_response` (error).
- OpenAI: `openai_completion` (model, prompt_tokens, completion_tokens, duration_ms), `openai_timeout` (base_url, model), `openai_error` (base_url, model, status_code), `openai_invalid_response` (error).
- Ollama: `ollama_completion`, `ollama_timeout`, `ollama_error`, `ollama_invalid_response` (unchanged from Phase 1).

**Route-level logging enriched with provider:**
- `chat_request_received` now includes `provider` field alongside `model`, `tenant_id`, `backend`, and `message_count`.
- `chat_request_completed` includes `provider` alongside `model`, `tenant_id`, `backend`, token counts, and `duration_ms`.

**Token counting:** `gateway/token_counting.py` logs no events currently — it is a pure utility. When integrated into the request path in future phases, token counts will feed into per-tenant usage tracking.

### Testing

**Unit tests — Anthropic translator** (`tests/unit/test_anthropic_translator.py`, 15 tests):
- `translate_request`: system message extraction (single, multiple concatenated), non-system messages converted to `AnthropicContentBlock`, `max_tokens` defaults to 4096 when absent, `max_tokens` passthrough when present, `stop` string wrapped in list, `stop` list passthrough, temperature and top_p mapped, headers include `x-api-key` and `anthropic-version`.
- `translate_response`: text content concatenation from multiple blocks, `stop_reason` mapping (`end_turn` -> `stop`, `max_tokens` -> `length`), usage field mapping (`input_tokens` -> `prompt_tokens`).
- `chat_completion`: async happy path (200 with valid response), timeout returns 504, backend error propagates status code, malformed response returns 502.

**Unit tests — OpenAI translator** (`tests/unit/test_openai_translator.py`, 13 tests):
- `translate_request`: passthrough via `model_dump`, `stream` forced to `False`, `Authorization: Bearer` header set from env var, no auth header when key is empty.
- `translate_response`: content extraction from `choices[0].message.content`, usage field mapping, gateway ID replacement (original ID discarded), missing usage defaults to zeros, empty choices raises `ValueError`.
- `chat_completion`: async happy path, timeout returns 504, backend error propagates, malformed response returns 502.

**Unit tests — token counting** (`tests/unit/test_token_counting.py`, 5 tests):
- tiktoken path for known OpenAI models, fallback `chars/4` for unknown models, fallback when tiktoken is not installed, empty string returns 0, consistency check (tiktoken count is within reasonable range of char estimate).

**Unit tests — Ollama translator** (`tests/unit/test_ollama_translator.py`, updated to 4 tests):
- Refactored to use `BackendConfig` instead of raw `base_url` string. Existing test coverage for `translate_request`, `translate_response`, and `chat_completion` preserved with updated function signatures.

**Test totals:** 91 tests across all unit and integration test files.

**E2E (Docker):**
- `make up` starts the full stack including mock-openai-1 and mock-anthropic-1.
- Three provider paths verified: `curl` with `model: tinyllama` routes through Ollama, `model: mock-gpt-markdown` routes through OpenAI mock, `model: mock-claude-markdown` routes through Anthropic mock. All return 200 with valid `ChatCompletionResponse` including `chatcmpl-` IDs, choices array, and usage tokens.

### Production gaps

- **Mock backends vs real API keys.** Tests run against `zerob13/mock-openai-api`, which returns canned responses. Real OpenAI and Anthropic APIs require valid API keys, enforce rate limits, and may return different response structures (e.g., content filtering, function calls, streaming chunks). The translators have not been validated against production APIs.
- **No streaming support.** All translators set `stream=False`. OpenAI and Anthropic both support streaming via SSE. Clients that set `stream: true` will receive a non-streaming response, which violates their expectation of chunked transfer encoding. Streaming is planned for a later phase.
- **No retry or failover.** If a backend returns an error, the gateway propagates it immediately. There is no retry with backoff, no circuit breaker, and no failover to an alternate backend serving the same model. Phase 5 adds load balancing and Phase 6 adds circuit breakers.
- **Single backend per model.** `registry.find_backend_for_model()` returns the first match. If multiple backends serve the same model, only the first is used. No load distribution, no health-aware routing.
- **No health checking of mock containers.** The gateway has no startup probes or readiness checks for mock services. If a mock crashes, requests to its models fail until the container restarts.
- **Token counting utility not integrated into request path.** `gateway/token_counting.py` exists but is not called during chat completions. Token counts come from backend responses. The utility is prepared for future phases (caching, rate limiting) where the gateway needs to count tokens before forwarding.
- **No tool_calls or function_calling support.** OpenAI and Anthropic responses that contain tool use (where `content` is null) will cause a 502. The translators assume text-only responses.
- **Anthropic content block types limited to text.** The translator only processes blocks where `type == "text"`. Image blocks, tool_use blocks, and tool_result blocks are silently ignored.

### Interview talking points

- **Why function registry over abstract class for provider dispatch.** The dispatch problem is "given a string, call the right function." A dict solves this directly. An ABC adds inheritance hierarchy, constructor boilerplate, and no shared behavior — it is abstraction without benefit for three stateless functions with identical signatures. If shared behavior emerged (e.g., retry logic common to all providers), it could be added as a decorator or wrapper function without refactoring to classes.
- **Anthropic system message extraction — different API design philosophies.** OpenAI treats system instructions as messages in the conversation (role: system). Anthropic separates them as a top-level `system` field, distinct from the conversation. The translator bridges this philosophical difference by partitioning the messages list at the gateway layer. This is a semantic translation, not just a structural one — the gateway understands the domain meaning of system messages.
- **Why gateway generates its own IDs for cross-provider consistency.** OpenAI returns `chatcmpl-` IDs, Anthropic returns `msg_` IDs, Ollama returns no ID. If the gateway passed through backend IDs, clients would see inconsistent formats depending on which backend handled the request. Generating `chatcmpl-{uuid}` at the gateway layer normalizes this — the client sees a consistent contract regardless of the backend. It also means the gateway can correlate its logs with client-visible IDs without depending on backend ID formats.

### Likely interview questions

**Q: "How do you add a new provider (e.g., Google Gemini)?"**
**A:** Three steps. First, write a new module `gateway/backends/gemini.py` with `translate_request()` (convert `ChatCompletionRequest` to Gemini's format), `translate_response()` (convert Gemini's response to `ChatCompletionResponse`), and `chat_completion()` (orchestrate the translate-POST-translate pipeline with error handling). Second, add `"gemini": gemini.chat_completion` to the `TRANSLATORS` dict in `gateway/routes/chat.py`. Third, add `"gemini"` to the `provider` Literal type in `BackendConfig` in `gateway/config.py`. No changes to the route handler logic, auth, or any other translator. If the new provider has a different response model, add Pydantic classes to `gateway/models.py` following the pattern of `AnthropicResponse`.

**Q: "Why not use the OpenAI Python SDK to talk to OpenAI-compatible backends?"**
**A:** The gateway is an HTTP proxy — it receives raw HTTP requests, translates them, and forwards them as raw HTTP. The OpenAI SDK adds an unnecessary abstraction layer that hides HTTP details the gateway needs to control: custom headers (for auth), timeouts (per-backend from config), error response bodies (for logging and propagation), and connection pooling (shared `httpx.AsyncClient`). The SDK also adds a dependency with its own versioning, breaking changes, and opinionated error handling. Using `httpx` directly gives full control over the HTTP interaction, which is exactly what a proxy needs.

**Q: "What happens when Anthropic adds a new field to their API response?"**
**A:** `AnthropicResponse` uses `ConfigDict(extra="allow")`, so unknown fields are accepted without validation errors. The translator extracts only the fields it uses (`content`, `stop_reason`, `usage`). New fields are silently ignored. If the new field is something the gateway should surface (e.g., a new token count category), the `AnthropicResponse` model is updated to include it, and `translate_response()` maps it to the appropriate OpenAI-compatible field. The `extra="allow"` policy means the gateway does not break when the upstream API evolves — it degrades gracefully by ignoring what it does not understand.

## Streaming Normalization

### Why this exists

LLM inference is slow — seconds to tens of seconds per response. Without streaming, clients wait for the entire response before seeing any output, which creates a poor user experience and prevents use cases like real-time chat UIs, progressive rendering, and typewriter effects. Streaming sends tokens as they are generated, but each provider uses a different streaming protocol: Ollama emits newline-delimited JSON (NDJSON), OpenAI emits Server-Sent Events (SSE) with `data:` prefixes, and Anthropic emits event-typed SSE with a state machine of `message_start`, `content_block_delta`, `message_delta`, and `message_stop` events. Without a normalization layer, clients would need three different stream parsers. The gateway absorbs this complexity by translating all three protocols into OpenAI-compatible SSE format, so clients write one parser regardless of which backend handles the request.

### How it works

1. Client sends `POST /v1/chat/completions` with `"stream": true` in the `ChatCompletionRequest` body.
2. The route handler in `gateway/routes/chat.py` detects the `stream` flag before dispatch. It looks up the provider in the `STREAM_TRANSLATORS` dict (separate from the non-streaming `TRANSLATORS` dict) to get the provider-specific `stream_chat_completion` async generator function.
3. The selected `stream_chat_completion` generator opens an httpx streaming connection to the backend via `client.stream("POST", ...)`, which holds the TCP connection open during the response.
4. The generator yields a role announcement chunk first: `{"choices": [{"delta": {"role": "assistant"}}]}` — this matches OpenAI's behavior where the first chunk announces the assistant role before content begins.
5. Per provider, the generator parses the backend's native streaming format:
   - **Ollama**: Iterates NDJSON lines via `response.aiter_lines()`. Each line is `json.loads`-parsed. `message.content` is extracted, empty content lines are skipped, and `done: true` triggers the final chunk with `finish_reason: "stop"`.
   - **OpenAI**: Near-passthrough SSE. Lines not starting with `"data: "` are skipped. `"data: [DONE]"` terminates the stream. JSON chunks are parsed, and the backend's `id` and `created` fields are replaced with gateway-generated values.
   - **Anthropic**: Inline state machine dispatching on the `type` field of each parsed SSE data line. `content_block_delta` with `delta.type == "text_delta"` extracts the text. `message_delta` extracts `stop_reason` and maps it via `STOP_REASON_MAP` (`end_turn` -> `stop`, `max_tokens` -> `length`). `message_stop` breaks the loop. Non-text delta types (e.g., `input_json_delta` for tool use) are silently skipped.
6. Each extracted content token is wrapped in a `ChatCompletionChunk` Pydantic model with a gateway-generated `chatcmpl-{uuid}` ID (shared across all chunks in the stream), the current Unix timestamp, and the model name.
7. Chunks are serialized as SSE-formatted strings: `data: {json}\n\n`.
8. After the stream loop ends (normal completion or error), the generator yields `data: [DONE]\n\n`.
9. FastAPI wraps the async generator in a `StreamingResponse` with `media_type="text/event-stream"`, which sends chunks to the client as they are yielded.

### Implementation

**Streaming chunk models** (`gateway/models.py`):
- `ChunkDelta` — `role: str | None = None`, `content: str | None = None`. Represents the delta payload in a streaming chunk. Role is set only in the first chunk; content is set in subsequent chunks.
- `ChunkChoice` — `index: int = 0`, `delta: ChunkDelta`, `finish_reason: str | None = None`. Parallels `Choice` from the non-streaming path but uses `delta` instead of `message`.
- `ChatCompletionChunk` — `id: str`, `object: Literal["chat.completion.chunk"]`, `created: int`, `model: str`, `choices: list[ChunkChoice]`. The streaming equivalent of `ChatCompletionResponse`. Unlike the non-streaming model, `id` and `created` are not auto-generated via `Field(default_factory=...)` — they are explicitly set by the generator so all chunks in a stream share the same values.

**Ollama streaming translator** (`gateway/backends/ollama.py: stream_chat_completion()`):
- Signature: `async def stream_chat_completion(client: httpx.AsyncClient, backend: BackendConfig, request: ChatCompletionRequest) -> AsyncGenerator[str, None]`.
- Reuses `translate_request()` from the non-streaming path, then overrides `stream = True` on the resulting `OllamaRequest`.
- Generates `chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"` and `created = int(time.time())` once, shared across all chunks.
- Yields role announcement chunk, then enters the streaming loop via `async with client.stream("POST", f"{backend.base_url}/api/chat", ...)`.
- Iterates `response.aiter_lines()`. Each line is parsed with `json.loads(line)`. Content is extracted from `data["message"]["content"]`. Empty content strings are skipped. When `data["done"]` is `True`, a final chunk with `finish_reason="stop"` is yielded and the loop breaks.
- Error handling: `httpx.HTTPStatusError` and generic `Exception` are caught. Both yield an SSE error event (`data: {"error": {"message": "...", "type": "stream_error"}}\n\n`) before the `[DONE]` sentinel.
- The `yield "data: [DONE]\n\n"` is outside the try/except block, ensuring it is always sent.

**OpenAI streaming translator** (`gateway/backends/openai.py: stream_chat_completion()`):
- Reuses `translate_request()` from the non-streaming path, then overrides `body["stream"] = True`.
- Does not yield a separate role announcement chunk — the OpenAI backend already includes one in its SSE stream. The generator passes through backend chunks with ID replacement.
- Iterates `response.aiter_lines()`. Empty lines and lines not starting with `"data: "` are skipped. `"data: [DONE]"` breaks the loop. JSON chunks are parsed, and `data["id"]` and `data["created"]` are replaced with gateway-generated values before re-serialization.
- `json.JSONDecodeError` on a malformed line causes a `continue` (skip the line), not an error event.
- Same error handling and `[DONE]` guarantee as Ollama.

**Anthropic streaming translator** (`gateway/backends/anthropic.py: stream_chat_completion()`):
- Reuses `translate_request()` from the non-streaming path, then sets `body["stream"] = True`.
- Yields role announcement chunk before entering the streaming loop.
- Initializes `finish_reason = "stop"` as default, updated if `message_delta` provides a `stop_reason`.
- Iterates `response.aiter_lines()`. Lines that are empty or do not start with `"data: "` are skipped (this filters out `event:` lines from the SSE protocol). Each data line is JSON-parsed; `json.JSONDecodeError` lines are skipped.
- Dispatches on `data["type"]`:
  - `"content_block_delta"`: checks `delta.type == "text_delta"`, extracts `delta.text`, yields a content chunk.
  - `"message_delta"`: extracts `delta.stop_reason`, maps it via `STOP_REASON_MAP` to update `finish_reason`.
  - `"message_stop"`: breaks the loop.
  - All other event types (e.g., `message_start`, `content_block_start`, `content_block_stop`) are implicitly ignored.
- After the loop, yields a final chunk with the accumulated `finish_reason` and empty delta, then `[DONE]`.
- Same error handling pattern as the other providers.

**Route dispatch** (`gateway/routes/chat.py`):
- `STREAM_TRANSLATORS` dict at module level: `{"ollama": ollama.stream_chat_completion, "openai": openai_backend.stream_chat_completion, "anthropic": anthropic_backend.stream_chat_completion}`.
- The streaming branch in `chat_completions()` checks `chat_request.stream`, looks up the provider in `STREAM_TRANSLATORS`, logs `chat_request_received` with `streaming=True`, and returns `StreamingResponse(stream_translator(client=..., backend=..., request=...), media_type="text/event-stream")`.
- The `response_model=ChatCompletionResponse` decorator on the route does not interfere with streaming — FastAPI passes `StreamingResponse` through without validation when it is returned directly.

### Key design decisions

1. **Separate `STREAM_TRANSLATORS` dict parallel to `TRANSLATORS`.** Streaming and non-streaming have fundamentally different return types: `AsyncGenerator[str, None]` vs `ChatCompletionResponse`. Trying to unify them in a single dispatch dict would require runtime type checking or union return types that make the code harder to reason about. Two dicts with identical keys but different value types keeps the dispatch clear and type-safe.

2. **Hand-formatted SSE via `StreamingResponse` instead of `EventSourceResponse`.** Libraries like `sse-starlette` provide `EventSourceResponse` that handles SSE framing. However, OpenAI's SSE format has specific conventions (e.g., `data: [DONE]\n\n` as a sentinel, no `event:` or `id:` fields on content chunks) that an abstraction might not match exactly. Using `StreamingResponse` with `media_type="text/event-stream"` and hand-formatted `f"data: {json}\n\n"` strings gives precise control over the wire format, ensuring clients that parse OpenAI SSE see exactly what they expect.

3. **Gateway-generated chunk IDs shared across the stream.** Each stream generates a single `chatcmpl-{uuid}` ID used for all chunks. This is consistent with OpenAI's behavior and the non-streaming path's ID generation. It enables clients to group chunks by ID and ensures the ID format is uniform regardless of the backend. The `created` timestamp is also generated once per stream.

4. **Inline Anthropic state machine without buffering.** The Anthropic translator processes each SSE event as it arrives and yields the translated chunk immediately. There is no buffering of events to reconstruct a complete message before yielding. This preserves low time-to-first-token (TTFT) — the client sees the first content chunk as soon as the first `content_block_delta` arrives from Anthropic, without waiting for `message_stop`.

5. **`httpx client.stream()` with `aiter_lines()` for lazy iteration.** The streaming context manager holds the backend TCP connection open and yields lines one at a time. This keeps memory consumption constant regardless of response length — the gateway never holds the entire response in memory. The context manager's `__aexit__` ensures the backend connection is closed even if the generator is abandoned mid-stream.

6. **Role announcement chunk before the streaming loop.** Ollama and Anthropic translators yield an explicit `{"delta": {"role": "assistant"}}` chunk before entering the backend streaming loop. This matches OpenAI's behavior where the first SSE chunk announces the role. The OpenAI translator does not emit a separate role chunk because the backend already includes one in its stream.

7. **Error SSE events for mid-stream failures.** Once `StreamingResponse` is returned, HTTP 200 is already committed — the gateway cannot change the status code. If the backend fails mid-stream, the generator yields a JSON error payload as a `data:` line (`{"error": {"message": "...", "type": "stream_error"}}`) before sending `[DONE]`. This gives clients a structured way to detect errors in the stream, even though the HTTP status is 200.

8. **`response_model=ChatCompletionResponse` stays on the route decorator.** This is not removed for streaming because FastAPI only applies response model validation when the handler returns a dict or Pydantic model. When `StreamingResponse` is returned directly, FastAPI passes it through without validation. Keeping the decorator documents the non-streaming contract and enables OpenAPI schema generation for the non-streaming response format.

### Alternatives considered

1. **FastAPI `EventSourceResponse` (sse-starlette) vs raw `StreamingResponse`.** `EventSourceResponse` would handle SSE framing automatically, but it adds a dependency and its default behavior (adding `id:` fields, handling reconnection via `Last-Event-ID`) does not match OpenAI's SSE format. The OpenAI format uses bare `data:` lines with no `event:` or `id:` fields on content chunks. Raw `StreamingResponse` gives exact control over the wire format with no dependency.

2. **Buffering Anthropic events to reconstruct full messages vs inline translation.** Buffering all events until `message_stop` and then yielding a single response would simplify the translator but eliminate the streaming benefit entirely — the client would wait for the full response just like the non-streaming path. Inline translation preserves the low-latency token-by-token delivery that makes streaming valuable.

3. **Single unified return type (union of `ChatCompletionResponse | AsyncGenerator`) vs separate translator dicts.** A single `TRANSLATORS` dict returning a union type would avoid the parallel dict pattern, but it pushes type discrimination into the route handler: `if isinstance(result, AsyncGenerator): return StreamingResponse(...)`. This is fragile and loses the clarity of separate dispatch paths. Two dicts with clear type contracts are simpler.

4. **`httpx-sse` library for SSE parsing vs manual line parsing.** The `httpx-sse` library provides a typed SSE event parser. However, the gateway only needs to handle three well-known formats (Ollama NDJSON, OpenAI SSE, Anthropic SSE), and the parsing logic for each is under 20 lines. Adding a library dependency for minimal parsing gains does not justify the added dependency surface.

### Failure modes and edge cases

- **Backend disconnect mid-stream.** If the backend closes the TCP connection while the generator is iterating `aiter_lines()`, httpx raises an exception. The generator's `except Exception` block catches it, yields an error SSE event with the exception message, and then yields `[DONE]`. The client receives a partial stream followed by an error event.
- **Malformed JSON chunk from backend.** For OpenAI and Anthropic translators: `json.JSONDecodeError` is caught and the line is skipped via `continue`. The stream continues with the next line. For Ollama: `json.loads(line)` raises `JSONDecodeError` which is caught by the generic `except Exception` block, yielding an error event and terminating the stream. This difference reflects the reliability assumptions: OpenAI and Anthropic SSE may include non-JSON lines (event types, comments), while Ollama NDJSON should be strictly JSON.
- **Connection refused before stream starts.** If the backend is unreachable, `client.stream()` raises an `httpx.ConnectError` inside the `async with` block. This is caught by the `except Exception` handler, which yields an error SSE event and then `[DONE]`. Because `StreamingResponse` is returned before the generator runs, HTTP 200 is already committed — the client sees a 200 response with an error event in the body.
- **Client disconnect mid-stream.** When the client closes the connection, Starlette raises `ClientDisconnect` when attempting to write the next chunk. The `StreamingResponse` handler catches this and stops iterating the generator. The async generator is garbage-collected, and the `async with client.stream(...)` context manager's `__aexit__` closes the backend TCP connection. No manual cleanup is needed.
- **Ollama returns `done: true` with empty content.** The empty content check (`if content:`) skips the content chunk. The `if done:` check then yields a final chunk with `finish_reason="stop"` and breaks. This is the correct behavior — the final NDJSON line from Ollama often has empty content alongside the done flag.
- **Anthropic `message_delta` with unknown `stop_reason`.** `STOP_REASON_MAP.get(stop_reason, "stop")` defaults to `"stop"` for any unrecognized stop reason. The stream does not fail on an unknown reason — it degrades gracefully to the most common finish reason.
- **Concurrent streaming requests.** Each streaming request gets its own async generator instance, its own httpx streaming context manager, and its own chunk ID. There is no shared mutable state between streams. httpx's connection pool manages the backend connections.

### Observability

- **Pre-stream logging:** `chat_request_received` is logged with `streaming=True`, model, tenant_id, backend name, provider, and message_count before the `StreamingResponse` is returned.
- **No post-stream logging:** `chat_request_completed` is not logged for streaming requests. The route handler returns `StreamingResponse` immediately — the generator runs asynchronously as Starlette writes chunks to the client. There is no hook to log after the stream finishes. Phase 16 will add TTFT (time to first token) and ITL (inter-token latency) metrics via a callback mechanism on the generator.
- **Per-provider error logs:** `ollama_stream_error`, `openai_stream_error`, and `anthropic_stream_error` are logged when exceptions occur during streaming. These include the error message or HTTP status code.
- **Request ID propagation:** The `request_id_middleware` in `gateway/main.py` sets the request ID before the streaming generator runs, so error logs within the generator carry the correct request ID via structlog contextvars.

### Testing

**Unit tests — streaming chunk models** (`tests/unit/test_models.py`, 8 streaming-specific tests):
- `ChunkDelta` serialization with role only, content only, and both fields. `ChunkChoice` with and without `finish_reason`. `ChatCompletionChunk` round-trip with `object: "chat.completion.chunk"`.

**Unit tests — Ollama streaming** (`tests/unit/test_ollama_streaming.py`, 5 tests):
- Happy path: role announcement chunk, two content chunks ("Hello", " world"), final chunk with `finish_reason="stop"`, `[DONE]` terminator.
- Shared IDs: all non-`[DONE]` chunks share the same `chatcmpl-` prefixed ID.
- Empty content skip: NDJSON lines with `content: ""` do not produce content chunks.
- Done terminator: stream always ends with `data: [DONE]\n\n`.
- SSE format: every yielded string starts with `data: ` and ends with `\n\n`.

**Unit tests — OpenAI streaming** (`tests/unit/test_openai_streaming.py`, 5 tests):
- Happy path: role chunk passthrough, content chunks ("Hello", " world"), final chunk with `finish_reason="stop"`, `[DONE]` terminator.
- ID replacement: backend's `chatcmpl-backend` ID is replaced with a gateway-generated `chatcmpl-` ID.
- Empty line skip: blank lines between SSE events do not produce output chunks.
- Done terminator: stream always ends with `data: [DONE]\n\n`.
- SSE format: every yielded string starts with `data: ` and ends with `\n\n`.

**Unit tests — Anthropic streaming** (`tests/unit/test_anthropic_streaming.py`, 6 tests):
- Happy path: role chunk, content chunks from `text_delta` events, final chunk with `finish_reason="stop"` (mapped from `end_turn`), `[DONE]` terminator.
- Stop reason mapping: `max_tokens` maps to `finish_reason="length"`.
- Non-text delta skip: `input_json_delta` events do not produce content chunks.
- Shared IDs: all chunks share the same gateway-generated `chatcmpl-` ID.
- Event line skip: `event:` lines from Anthropic SSE are filtered out, only `data:` lines are processed.
- Done terminator: stream always ends with `data: [DONE]\n\n`.

**Integration tests — streaming dispatch** (`tests/integration/test_streaming.py`, 4 tests):
- SSE content type: streaming response has `text/event-stream` in `Content-Type` header.
- Chunk format: response body contains `data:` lines with `chat.completion.chunk` objects and `data: [DONE]`.
- Done termination: last non-empty line in the response body is `data: [DONE]`.
- Non-streaming regression: requests with `stream: false` still return JSON `chat.completion` objects, not SSE.

**Test infrastructure:**
- All unit tests use a custom `_make_mock_stream_client()` helper that creates a mock httpx client with an `AsyncMock` context manager. The mock's `aiter_lines()` returns an async iterator over a list of pre-defined lines, enabling deterministic testing of each provider's parsing logic without network I/O.
- Integration tests use `patch.dict` on `STREAM_TRANSLATORS` to inject a mock async generator, testing the route dispatch and `StreamingResponse` wrapping without requiring a real backend.

**E2E (Docker):**
- `make up` starts the full stack. All three providers (Ollama, OpenAI mock, Anthropic mock) stream correctly via `curl` with `"stream": true`. Responses arrive as chunked `text/event-stream` data ending with `data: [DONE]`.

### Production gaps

- **No TTFT or ITL metrics.** Time-to-first-token and inter-token latency are critical streaming performance indicators. Currently there is no instrumentation to measure when the first chunk is yielded or the spacing between chunks. Phase 16 adds these metrics.
- **No cache integration with streams.** The semantic caching layer (Phase 8) cannot cache streaming responses. A tee buffer pattern (duplicate the stream into a cache writer and the client writer) is prepared but not connected.
- **No usage or token tracking for streaming requests.** Non-streaming requests log `prompt_tokens` and `completion_tokens` from the backend response. Streaming requests have no equivalent — token counts are not available until after the stream completes, and there is no post-stream callback to collect them.
- **No `chat_request_completed` log for streaming.** Duration, token counts, and final status are not logged because the route handler returns `StreamingResponse` before the generator runs. The completion log only fires for non-streaming requests.
- **No request timeout on streaming connections.** The httpx client has a 120-second timeout for non-streaming requests, but streaming connections can remain open indefinitely. A slow backend that sends one token per minute would hold the connection open without limit.
- **No max response size limit for streaming.** A backend that streams an unbounded response (e.g., a model in a loop) would be forwarded to the client without limit. There is no byte or token cap on streaming responses.
- **No backpressure handling.** If the client reads slowly, Starlette buffers the generator's output. There is no mechanism to signal the backend to slow down if the client cannot keep up.

### Interview talking points

- **Why normalize to OpenAI SSE format.** Clients write one SSE parser regardless of whether the backend is Ollama, OpenAI, or Anthropic. The gateway absorbs three different streaming protocols (NDJSON, SSE passthrough, event-typed SSE state machine) and presents a uniform interface. Adding a fourth provider requires one new translator, not changes to every client.
- **Anthropic inline state machine for low TTFT.** The Anthropic translator dispatches on each event's `type` field as it arrives, yielding translated chunks immediately. No buffering means the first token reaches the client as soon as Anthropic sends its first `content_block_delta`. Buffering until `message_stop` would defeat the purpose of streaming.
- **httpx streaming context manager for constant memory.** `client.stream()` with `aiter_lines()` processes one line at a time. The gateway never holds the entire response in memory, regardless of response length. The context manager's `__aexit__` guarantees the backend connection is closed even if the generator is interrupted (client disconnect, error, garbage collection).

### Likely interview questions

**Q: "How do you handle a client disconnecting mid-stream?"**
**A:** When the client closes the TCP connection, Starlette's `StreamingResponse` detects the closed socket on its next write attempt and raises `ClientDisconnect`. This stops the iteration of the async generator. Python's garbage collector then finalizes the generator, which triggers the `async with client.stream(...)` context manager's `__aexit__`, closing the backend HTTP connection. No manual cleanup code is needed — the combination of Python's generator finalization and httpx's context manager protocol handles it automatically. The backend stops receiving reads on its connection and can clean up on its side.

**Q: "Why can't you log token counts for streaming requests?"**
**A:** The route handler returns the `StreamingResponse` object immediately — at that point, the async generator has not started running yet. The generator executes asynchronously as Starlette writes chunks to the client. When the stream finishes, execution is inside Starlette's response writer, not in the route handler. There is no return value and no hook to run post-stream logic. Phase 16 addresses this by wrapping the generator in a metrics-collecting wrapper that tracks TTFT, ITL, and total chunks, then fires a callback when the generator is exhausted.

**Q: "How do you ensure `[DONE]` is always sent?"**
**A:** In all three translators, the `yield "data: [DONE]\n\n"` statement is placed outside the `try/except` block, after both the normal stream loop and the error handlers. Whether the stream completes successfully, the backend returns an error, or an unexpected exception occurs, the generator always reaches the `[DONE]` yield. The only scenario where `[DONE]` is not sent is if the client disconnects first — but in that case, no one is listening for it anyway.

**Q: "What is the tradeoff of returning HTTP 200 before the stream starts?"**
**A:** `StreamingResponse` commits HTTP 200 and the `text/event-stream` headers as soon as the route handler returns — before the generator yields its first chunk. If the backend is unreachable or returns an error, the client has already received a 200 status code. The gateway cannot retroactively change it to 502. Instead, it yields a JSON error payload as a `data:` line in the SSE stream. This means clients must parse the stream body for errors, not just check the HTTP status. This is the same tradeoff OpenAI's own API makes — it is inherent to HTTP streaming, not a gateway-specific limitation.

## Consistent Hash Router

### Why this exists

Phase 3 added multiple backends per model (5 OpenAI mock instances serving `mock-gpt-markdown`, 3 Anthropic mock instances serving `mock-claude-markdown`, 2 Ollama instances serving `tinyllama`) but `find_backend_for_model()` used first-match routing — it always returned the first backend in the list for a given model. This meant `mock-openai-1` handled 100% of `mock-gpt-markdown` traffic while `mock-openai-2` through `mock-openai-5` sat idle. There was no load distribution, no cache locality, and no way to weight backends differently (e.g., a beefy GPU node vs a small CPU node).

Consistent hashing solves this by distributing requests across backends proportionally to their configured weights while keeping the same tenant+model combination pinned to the same backend. This tenant affinity enables cache locality — when semantic caching is added in a later phase, repeated queries from the same tenant hit the same backend, increasing cache hit rates. The consistent hashing property also means adding or removing a backend only remaps approximately 1/N of the keys, not all of them, minimizing disruption during scaling events.

### How it works

1. On startup (or config reload), `Registry.__init__()` in `gateway/config.py` iterates `model_to_backends` and builds a `ConsistentHashRing` per model. Each backend is added to the ring as a `(name, weight)` tuple. The ring for `mock-gpt-markdown` contains 5 nodes (mock-openai-1 through mock-openai-5, each weight 1); the ring for `mock-claude-markdown` contains 3 nodes; the ring for `tinyllama` contains 2 nodes.
2. Each backend gets `weight * 150` virtual nodes on the ring. A weight-1 backend gets 150 virtual nodes; a weight-3 backend gets 450. Virtual node keys are formatted as `"{backend_name}:{i}"` for `i` in `range(num_vnodes)`. Each key is hashed via MD5 to a 32-bit integer position on the ring. The `(position, backend_name)` tuples are sorted by position to form the ring.
3. When a chat request arrives, the route handler in `gateway/routes/chat.py` constructs `routing_key = f"{tenant.id}:{chat_request.model}"` — e.g., `"tenant-alpha:mock-gpt-markdown"`.
4. `registry.find_backend_for_model(model, routing_key=routing_key)` looks up the per-model ring and calls `ring.get_node(routing_key)`.
5. `get_node()` hashes the routing key via `MD5(routing_key.encode()).digest()[:4]` to a 32-bit position, then uses `bisect.bisect_right()` on the sorted positions array to find the clockwise-nearest virtual node. The backend name at that position is returned.
6. The selected `BackendConfig` is retrieved from `registry.backends[node_name]` and used for the rest of the request (translation, HTTP call, response mapping).
7. `request.state.backend_name = backend.name` is set in the route handler. The `request_id_middleware` in `gateway/main.py` checks `getattr(request.state, "backend_name", None)` after `call_next` and sets the `X-Backend` response header if present. This works for both streaming and non-streaming responses because the middleware wraps the entire request lifecycle.

### Implementation

**ConsistentHashRing** (`gateway/routing.py`, ~80 lines):
- `__init__(nodes: list[tuple[str, int]], vnodes_per_unit: int = 150)` — builds the ring by generating `weight * vnodes_per_unit` virtual nodes per backend. Each virtual node key `"{name}:{i}"` is hashed to a 32-bit position via `_hash()`. The `(position, name)` pairs are sorted, and a parallel `_positions` list is built for binary search.
- `_hash(key: str) -> int` — static method. `hashlib.md5(key.encode()).digest()[:4]` converted to a big-endian unsigned 32-bit integer via `int.from_bytes(digest[:4], "big")`. The `# noqa: S324` suppresses the Bandit security warning — MD5 is not used cryptographically here, only for distribution.
- `get_node(key: str, exclude: frozenset[str] = frozenset()) -> str | None` — hashes the key, uses `bisect.bisect_right()` to find the insertion point in the sorted positions array, wraps around via `% len(self._ring)`, then walks clockwise through the ring skipping any nodes in `exclude`. Returns `None` if the ring is empty or all nodes are excluded. The walk is bounded by `len(self._ring)` to avoid infinite loops.
- `node_count` / `vnode_count` — properties returning the number of distinct backends and total virtual nodes.
- `get_distribution() -> dict[str, int]` — counts virtual nodes per backend. Used by `ring_state()` for the admin endpoint.

**Registry integration** (`gateway/config.py`):
- `model_rings: dict[str, ConsistentHashRing]` — built in `__init__()` by iterating `model_to_backends` and constructing a ring per model with `[(b.name, b.weight) for b in backends]`.
- `find_backend_for_model(model: str, routing_key: str | None = None) -> BackendConfig | None` — if `routing_key` is provided, looks up the model's ring and calls `ring.get_node(routing_key)`. Maps the returned node name back to a `BackendConfig` via `self.backends.get(node_name)`. Falls back to first-match if `routing_key is None` (backward compatibility for any callers that predate the routing key addition).
- `ring_state() -> dict` — returns a dict keyed by model name, each value containing `backends` (list of backend names), `total_vnodes` (int), and `distribution` (dict of backend name to vnode count). Used by the admin endpoint.

**Chat route** (`gateway/routes/chat.py`):
- `routing_key = f"{tenant.id}:{chat_request.model}"` — constructed after tenant authentication and model access check.
- `backend = registry.find_backend_for_model(chat_request.model, routing_key=routing_key)` — passes the routing key into the registry lookup.
- `request.state.backend_name = backend.name` — set after backend selection, read by the middleware for the `X-Backend` header.

**Middleware** (`gateway/main.py: request_id_middleware`):
- After `call_next`, reads `backend_name = getattr(request.state, "backend_name", None)`. If present, sets `response.headers["X-Backend"] = backend_name`. This placement after `call_next` ensures the header is set even if the response is a `StreamingResponse` — the middleware wraps the entire response object.

**Admin endpoint** (`gateway/routes/admin.py`):
- `GET /admin/ring` — calls `registry.ring_state()` and returns the JSON dict. Each model entry shows which backends serve it, the total virtual node count, and the distribution of virtual nodes per backend.

**Backend configuration** (`config/backends.yaml`):
- 10 backends total: `ollama-1` and `ollama-2` (both serving `tinyllama`, weight 1), `mock-openai-1` through `mock-openai-5` (all serving `mock-gpt-markdown`, weight 1), `mock-anthropic-1` through `mock-anthropic-3` (all serving `mock-claude-markdown`, weight 1).
- All backends within a model group have equal weight (1), so traffic distributes evenly. Changing a backend's `weight` to 3 would give it 3x the virtual nodes and approximately 3x the traffic share.

### Key design decisions

1. **Custom implementation from scratch (< 100 lines) over a library like `uhashring`.** The consistent hash ring is a core architectural component of this project and a likely interview discussion topic. Implementing it from scratch ensures the author can explain every line: the MD5 hashing, the virtual node generation, the binary search lookup, and the clockwise walk with exclusion. A library dependency would obscure these internals. The implementation is under 100 lines including docstrings — the complexity does not justify a dependency.

2. **MD5 for hashing.** MD5 provides excellent distribution across the 32-bit keyspace, which is what matters for load balancing. It is not used cryptographically — there is no security requirement for collision resistance or preimage resistance. SHA-256 would produce equally good distribution but is slightly slower (irrelevant given LLM inference latency dominates). CRC32 would be faster but has known distribution weaknesses for structured input. MD5 is the standard choice for consistent hashing in industry (Dynamo, Cassandra, Memcached).

3. **Virtual nodes at `weight * 150` per backend.** Without virtual nodes, each backend would occupy a single point on the ring. With only 3-5 backends, the arc lengths between adjacent points would vary wildly, leading to uneven load. Virtual nodes spread each backend across 150 points per unit of weight, smoothing the distribution. The `150` multiplier was empirically validated: with 5 equal-weight backends and 10,000 sample keys, each backend receives 18-22% of traffic (within 5% of the ideal 20%). Lower multipliers (e.g., 50) showed up to 10% skew.

4. **Per-model rings.** Each model has its own independent hash ring containing only the backends that serve that model. The `mock-gpt-markdown` ring has 5 backends; the `tinyllama` ring has 2. This ensures that a backend's weight in one ring does not affect distribution in another. It also means adding a new backend for one model does not cause any key remapping for other models.

5. **Routing key `tenant_id:model` for cache locality.** The routing key determines which backend a request lands on. Using `tenant_id:model` means the same tenant always hits the same backend for a given model. This creates cache affinity — when semantic caching is added, repeated similar queries from the same tenant are more likely to hit a warm cache on the same backend. Using `request_id` as the routing key would produce uniform distribution but destroy tenant affinity. Using only `tenant_id` would pin a tenant to one backend across all models, which is unnecessarily rigid.

6. **`get_node(key, exclude=frozenset())` for forward-compatible failover.** The `exclude` parameter allows Phase 6's circuit breaker to exclude unhealthy backends from the ring without rebuilding it. When a backend trips its circuit breaker, the caller passes it in the `exclude` set, and `get_node()` walks clockwise past it to the next healthy backend. This was designed into the ring from the start to avoid a Phase 6 refactor. The `frozenset` type for `exclude` is immutable and hashable, suitable for use in future caching of routing decisions.

7. **`X-Backend` via `request.state` + middleware instead of setting it in the route handler.** The route handler cannot set response headers on a `StreamingResponse` after returning it — the response is already committed. By storing the backend name on `request.state` and reading it in the middleware (which wraps `call_next`), the header is set on the `Response` object that the middleware returns. This works for both streaming and non-streaming responses because the middleware operates on the final `Response` object.

8. **`routing_key=None` backward compatibility.** `find_backend_for_model(model, routing_key=None)` falls back to first-match behavior. This ensures that any code path or test that calls `find_backend_for_model` without a routing key (e.g., older tests, admin endpoints that just need to check if a model exists) continues to work without modification.

### Alternatives considered

1. **Round-robin routing.** Simple and fair, but stateless — the same tenant+model would hit a different backend on every request, destroying cache locality. In a system with semantic caching, round-robin would reduce cache hit rates by a factor of N (number of backends). Round-robin also requires shared state (a counter) that must be coordinated across multiple gateway instances in a horizontally-scaled deployment. Consistent hashing is stateless — any gateway instance produces the same routing decision for the same key.

2. **Consistent hashing library (`uhashring`, `hash_ring`).** Would save ~80 lines of code but would make it impossible to explain the internals during an interview. "I used a library" is a weak answer to "how does your load balancer work?" The library also may not support the `exclude` parameter needed for Phase 6 failover without subclassing or wrapping.

3. **`request_id`-based routing key.** Using the unique request ID as the routing key would produce perfectly uniform distribution across backends. However, it would eliminate tenant affinity — the same tenant would hit different backends on every request. This trades cache locality for distribution uniformity. Since the consistent hash ring already provides good distribution (within 5% of ideal with 150 vnodes per weight unit), the cache locality benefit of tenant affinity outweighs the marginal improvement in distribution uniformity.

4. **Modular hashing (`hash(key) % N`).** Simple and fast, but catastrophic on topology changes. Adding a 6th backend changes `hash % 5` to `hash % 6`, remapping approximately 80% of keys (only keys where `hash % 5 == hash % 6` stay put). Consistent hashing remaps only ~1/N keys (approximately 17% when adding a 6th backend to a 5-backend ring). This property is critical for maintaining cache locality during scaling events.

### Failure modes and edge cases

- **Empty ring (no backends for a model).** If a model has no backends in the config, `model_rings` has no entry for it. `find_backend_for_model()` falls through to the fallback path, where `model_to_backends.get(model, [])` returns an empty list, and `backends[0] if backends else None` returns `None`. The route handler raises `HTTPException(404, "No backend available for model: ...")`.
- **All backends excluded via `exclude` parameter.** `get_node()` walks the entire ring (bounded by `len(self._ring)` iterations) without finding a non-excluded node and returns `None`. Phase 6 interprets this as all circuit breakers open and applies its fallback logic (e.g., half-open attempt or 503).
- **Weight 0.** A backend with `weight: 0` generates `0 * 150 = 0` virtual nodes and is effectively absent from the ring. The backend still exists in `registry.backends` but receives no traffic via the hash ring. This can be used as a soft-disable mechanism without removing the backend from config.
- **Ring walk with all nodes excluded.** The `for offset in range(len(self._ring))` loop is bounded by the total number of virtual nodes. Even if every vnode is checked, the loop terminates and returns `None`. There is no risk of an infinite loop.
- **Hash collision on virtual node positions.** Two virtual nodes landing on the same 32-bit position is possible but harmless — both appear in the sorted ring, and `bisect_right` picks the one immediately after the key's position. The "losing" vnode is shadowed (never selected as the nearest), slightly reducing its backend's effective weight. With 150 vnodes per unit, the probability of collision affecting distribution measurably is negligible.
- **Config reload rebuilds all rings.** When `POST /admin/reload` or SIGHUP triggers a config reload, a new `Registry` is constructed from scratch, including new `ConsistentHashRing` instances. The old registry (and its rings) remain in use by in-flight requests that already hold a reference. The swap is atomic (GIL-protected pointer assignment). There is no incremental ring update — the entire ring is rebuilt. For 10 backends with 1,500 total vnodes, this takes microseconds.
- **Single backend for a model.** A model served by only one backend has a ring with one node. `get_node()` always returns that node regardless of the routing key. The ring degenerates to a no-op lookup, which is correct — there is only one choice.

### Observability

- **`X-Backend` response header.** Set on every chat completion response (both streaming and non-streaming) via the `request_id_middleware`. Clients and load balancers can inspect this header to verify which backend handled a request. Useful for debugging routing issues and validating that consistent hashing is working (same tenant+model should consistently show the same `X-Backend` value).
- **`GET /admin/ring` endpoint.** Returns the full ring state per model as JSON: which backends are in each ring, the total virtual node count, and the distribution of virtual nodes per backend. Example response: `{"mock-gpt-markdown": {"backends": ["mock-openai-1", ..., "mock-openai-5"], "total_vnodes": 750, "distribution": {"mock-openai-1": 150, ...}}}`. Operators can use this to verify that weight changes took effect and that the ring is balanced.
- **Chat request logs.** `chat_request_received` and `chat_request_completed` log events include `backend=backend.name`, enabling per-backend log filtering and traffic analysis. Combined with `tenant_id`, operators can verify that tenant affinity is working correctly.

### Testing

**Unit tests — hash ring** (`tests/unit/test_hash_ring.py`, 12 tests):
- `test_deterministic` — same key returns same node across 100 lookups. Validates that the ring produces stable, reproducible routing decisions.
- `test_different_keys_hit_multiple_nodes` — 100 different keys hit at least 2 of 3 nodes. Validates that the ring distributes traffic across backends rather than collapsing to a single node.
- `test_single_node_always_returns_it` — ring with one node returns it for all keys. Validates degenerate case.
- `test_empty_ring_returns_none` — ring with no nodes returns `None`. Validates the empty guard.
- `test_weight_proportional_distribution` — weight-3 node receives 70-80% of 10,000 keys in a 2-node ring (1:3 weight ratio). Validates that virtual node count correctly translates to proportional traffic.
- `test_add_backend_redistributes_less_than_1_over_n` — adding a 4th node to a 3-node ring changes fewer than `ceil(1000/4) + 5%` keys. Validates the consistent hashing redistribution property.
- `test_remove_backend_only_redistributes_that_backends_keys` — removing a node only remaps keys that were on that node; all other keys stay on their original nodes. Validates minimal disruption.
- `test_wrap_around` — keys resolve correctly even when the hash position is near the end of the 32-bit range. Validates the modulo wrap in `bisect_right() % len(ring)`.
- `test_exclude_skips_node` — excluding one node in a 2-node ring routes all keys to the other. Validates the clockwise walk with exclusion.
- `test_exclude_all_returns_none` — excluding all nodes returns `None`. Validates the termination condition.
- `test_node_count` / `test_vnode_count` — property accessors return correct counts.
- `test_get_distribution` — virtual node counts match `weight * vnodes_per_unit` for each backend.

**Unit tests — Registry routing** (`tests/unit/test_config.py`, updated):
- `find_backend_for_model` with `routing_key` returns a valid backend from the model's ring.
- `find_backend_for_model` with `routing_key=None` falls back to first-match (backward compat).
- `ring_state()` returns correct structure with backends, total_vnodes, and distribution per model.
- `model_rings` is populated for each model in the config.

**E2E (Docker):**
- Consistent routing: 10 identical requests with the same tenant+model produce the same `X-Backend` header value across all 10 responses.
- Different tenants, different backends: requests from `tenant-alpha` and `tenant-beta` for the same model may return different `X-Backend` values, demonstrating that the routing key includes tenant identity.
- `GET /admin/ring` returns valid JSON with ring state for all three model groups (`tinyllama`, `mock-gpt-markdown`, `mock-claude-markdown`), showing correct backend lists and vnode distributions.

### Production gaps

- **No failover.** Phase 6 adds circuit breakers that pass unhealthy backends to `get_node(key, exclude={...})`. Currently, if the selected backend is down, the request fails without trying another backend. The `exclude` parameter is wired into the ring but not yet called with a non-empty set.
- **No health-aware exclusion.** The ring routes to backends without checking their health status. A backend that is returning 500s or timing out will continue receiving its share of traffic until a circuit breaker (Phase 6) is added.
- **No latency-aware routing.** The ring distributes traffic based on weight, not on observed backend latency. A backend that is slower than its peers will receive the same traffic share. Latency-weighted routing is planned for Phase 12.
- **Ring rebuilt on every config reload, not incremental.** A config reload constructs an entirely new `ConsistentHashRing` for every model, even if only one backend changed. For the current scale (10 backends, ~1,500 vnodes total), this is negligible. At much larger scale (hundreds of backends, tens of thousands of vnodes), an incremental ring update that only adds/removes affected vnodes would be more efficient.
- **No ring persistence or cross-instance consistency.** Each gateway instance builds its own ring from config. In a multi-instance deployment, all instances must read the same config to produce the same ring and the same routing decisions. There is no shared state or consensus mechanism — consistency is achieved by deploying the same config file to all instances.

### Interview talking points

- **Why consistent hashing over round-robin for a caching gateway.** The gateway's value proposition includes semantic caching (later phase). Round-robin would spread the same tenant's requests across all backends, reducing cache hit rates by N. Consistent hashing pins `tenant:model` to a specific backend, creating cache affinity. If the tenant repeats a similar query, it hits the same backend with the warm cache. This is the same principle behind Memcached and Redis Cluster's hash slot routing.
- **Virtual nodes prevent hotspots.** Without virtual nodes, 5 backends would have 5 points on a circle. The arc lengths between adjacent points vary wildly due to hash randomness, so one backend might own 40% of the keyspace while another owns 10%. With 150 virtual nodes per weight unit, each backend is spread across 150 points, and the statistical variance drops to under 5%. The math: standard deviation of arc length decreases as `1/sqrt(k)` where `k` is the number of vnodes per backend.
- **Redistribution property for safe scaling.** When adding a 6th backend to a 5-backend ring, only approximately 1/6 (~17%) of keys remap. The remaining 5/6 stay on their current backend. This means 83% of cached entries remain valid after a scaling event. With modular hashing (`hash % N`), adding a backend would invalidate approximately 80% of cached entries — a cache stampede that could overwhelm backends.

### Likely interview questions

**Q: "What happens when a backend fails?"**
**A:** Currently, the request fails and returns the backend's error to the client. Phase 6 adds a circuit breaker that tracks backend failures. When a backend trips its circuit breaker, the routing layer calls `ring.get_node(routing_key, exclude={failed_backend})`, which walks clockwise past the failed backend's virtual nodes to the next healthy backend. The key insight is that only the ~1/N of traffic that was going to the failed backend gets redistributed — traffic to healthy backends is unaffected. The `exclude` parameter was designed into the ring from the start specifically for this use case.

**Q: "Why MD5 and not a faster hash?"**
**A:** MD5 gives excellent distribution across the 32-bit keyspace, which is the only property that matters for load balancing. There is no cryptographic requirement — we do not need collision resistance or preimage resistance. The hash function's execution time (~200 nanoseconds for MD5) is negligible compared to LLM inference latency (seconds to tens of seconds). SHA-256 would work equally well. CRC32 would be faster but has known distribution weaknesses for structured keys (e.g., sequential tenant IDs). The choice of MD5 is pragmatic: good distribution, widely understood, no dependencies beyond Python's standard library.

**Q: "How does the ring handle weighted backends?"**
**A:** Weight maps directly to virtual node count. A weight-1 backend gets 150 virtual nodes; a weight-3 backend gets 450. Because virtual nodes are uniformly distributed by the hash function, a backend with 3x the virtual nodes occupies approximately 3x the ring arc and receives approximately 3x the traffic. This is validated in the unit test `test_weight_proportional_distribution`, which verifies that a weight-3 node in a `(1, 3)` ring receives 70-80% of 10,000 sample keys (expected: 75%).

## Circuit Breaker & Failover

### Why this exists

Without circuit breakers, a dead backend causes repeated failures until it is manually removed from the configuration. Every request that the hash ring routes to the unhealthy backend fails, and the gateway dutifully forwards the error to the client. With consistent hashing, the problem is worse: the routing key `tenant_id:model` is deterministic, so the same tenant always hits the same backend. If that backend is down, that specific tenant experiences 100% failure rate while other tenants on healthy backends are fine. The gateway has no mechanism to detect that a backend is unhealthy, exclude it from routing, or resume traffic once it recovers.

Circuit breakers solve this by tracking per-backend failure rates in a rolling window, tripping to OPEN when the error rate exceeds a threshold, excluding OPEN backends from the hash ring via the `exclude` parameter designed into Phase 5, probing for recovery after a cooldown period with exponential backoff, and resuming traffic automatically when a probe succeeds. Combined with a retry loop that accumulates an exclude set per request, the gateway can fail over to the next clockwise node on the ring within the same request -- the client sees a successful response even when one backend is down.

### How it works

1. Each backend has a `CircuitBreaker` instance with three states: `CLOSED`, `OPEN`, and `HALF_OPEN`.
2. **CLOSED** (normal operation): All requests flow through to the backend. Every request outcome (success or failure) is recorded in a rolling window -- a `deque` of `(timestamp, success_bool)` tuples. Entries older than 60 seconds are pruned via `popleft()` on each `_should_trip()` check.
3. **Trip condition**: When the failure rate reaches 50% or higher (`failure_threshold=0.5`) with at least 10 requests in the window (`min_requests=10`), the breaker transitions from `CLOSED` to `OPEN`. The `min_requests` guard prevents a single failure from tripping the breaker on low-traffic backends.
4. **OPEN** (backend excluded): The `CircuitBreakerRegistry.get_open_backends()` method calls `allow_request()` on each breaker and returns the names of those that deny requests. This `frozenset` is passed to `registry.find_backend_for_model(model, routing_key=routing_key, exclude=exclude)`, which calls `ring.get_node(routing_key, exclude=exclude)`. The hash ring walks clockwise past the excluded backend's virtual nodes to the next healthy backend, so traffic that was destined for the failed backend shifts to the next clockwise neighbor.
5. **Cooldown with exponential backoff**: After the initial cooldown period (30 seconds), the breaker transitions from `OPEN` to `HALF_OPEN` on the next `allow_request()` call. If the probe fails, the cooldown doubles: 30s, 60s, 120s, 240s, capping at 300s (`max_cooldown`). If the probe succeeds, the cooldown resets to the initial 30s.
6. **HALF_OPEN** (one probe allowed): Exactly one request is permitted through to the backend. The `_half_open_probe_sent` flag prevents concurrent probes. If the probe succeeds (`record_success()`), the breaker transitions back to `CLOSED` and the cooldown resets. If it fails (`record_failure()`), the breaker transitions back to `OPEN` with a doubled cooldown.
7. **Non-streaming retry loop**: The route handler in `gateway/routes/chat.py` runs a `for attempt in range(3)` loop. On each iteration, it calls `find_backend_for_model` with the current `exclude` set. If the backend returns a 5xx or raises `httpx.ConnectError`, the failure is recorded on the circuit breaker, the backend name is added to `exclude` via `exclude = exclude | {backend.name}`, and the loop continues. The next iteration routes to a different backend because the failed one is now in the exclude set. If all attempts exhaust or no backend is available, the last error is raised (or 404 if no backend was found).
8. **Streaming path**: Streaming requests do not use the retry loop (retrying a partially-consumed stream is not feasible). Instead, the route handler wraps the async generator with `_wrap_stream_with_circuit_breaker()`, which monitors the stream for errors. If the stream contains `"stream_error"` in any SSE data line, or if the generator raises an exception, `record_failure()` is called in the `finally` block. Otherwise, `record_success()` is called. This ensures the circuit breaker learns from streaming outcomes.
9. **All backends OPEN**: If `find_backend_for_model` returns `None` (the hash ring has no non-excluded nodes), the route handler raises `HTTPException(503, "All backends unavailable for this model")`.

### Implementation

**CircuitBreaker class** (`gateway/circuit_breaker.py: CircuitBreaker`):
- Constructor parameters: `backend_name`, `window_size=60.0` (seconds), `failure_threshold=0.5` (50%), `min_requests=10`, `cooldown=30.0` (initial), `max_cooldown=300.0` (cap).
- `_requests: deque[tuple[float, bool]]` -- rolling window of `(monotonic_timestamp, is_success)` tuples. Uses `deque` for O(1) `popleft()` during pruning.
- `allow_request() -> bool` -- returns `True` if CLOSED, `True` for one probe if OPEN and cooldown expired (triggers HALF_OPEN transition), `True` for one probe if HALF_OPEN and `_half_open_probe_sent` is `False`, `False` otherwise.
- `record_success()` -- appends `(time.monotonic(), True)` to the window. If HALF_OPEN, transitions to CLOSED and resets cooldown to `_initial_cooldown`.
- `record_failure()` -- appends `(time.monotonic(), False)`. If HALF_OPEN, doubles `_current_cooldown` (capped at `max_cooldown`), resets `_half_open_probe_sent`, and transitions to OPEN. If CLOSED, calls `_should_trip()`.
- `_should_trip() -> bool` -- calls `_prune_window()`, checks `len(self._requests) >= min_requests`, then computes `failures / total >= failure_threshold`.
- `_prune_window()` -- pops entries where `timestamp <= time.monotonic() - window_size` from the left of the deque.
- `_transition(new_state)` -- sets `self.state`, records `_opened_at = time.monotonic()` if transitioning to OPEN, logs `circuit_state_change` with old state, new state, and current cooldown.
- `snapshot() -> dict` -- prunes the window and returns `{"state", "error_rate", "requests_in_window", "current_cooldown_s"}` for the admin endpoint.

**CircuitBreakerRegistry class** (`gateway/circuit_breaker.py: CircuitBreakerRegistry`):
- `__init__(backend_names, **kwargs)` -- creates a `CircuitBreaker` for each backend name. `kwargs` are forwarded to the `CircuitBreaker` constructor (used in tests to override thresholds).
- `get(backend_name) -> CircuitBreaker | None` -- returns the breaker for a specific backend.
- `get_open_backends() -> frozenset[str]` -- iterates all breakers, calls `allow_request()` on each, returns names of those that return `False`. The side effect of `allow_request()` is intentional: calling it on an OPEN breaker past its cooldown triggers the HALF_OPEN transition (passive probing).
- `get_all_snapshots() -> dict[str, dict]` -- returns `snapshot()` for every breaker, used by the admin endpoint.
- `sync_backends(backend_names)` -- adds breakers for new backends, removes breakers for removed backends, preserves state for existing backends. Called on config reload (SIGHUP and `POST /admin/reload`).

**Non-streaming retry loop** (`gateway/routes/chat.py: chat_completions()`):
- Lines 112-205: initializes `exclude = cb_registry.get_open_backends()` and `last_error = None`.
- `for attempt in range(3)`: calls `find_backend_for_model(model, routing_key=routing_key, exclude=exclude)`. If `None`, breaks out of the loop.
- On `HTTPException` with `status_code >= 500`: calls `cb.record_failure()`, adds backend to exclude via `exclude = exclude | {backend.name}`, stores `last_error`, logs `backend_failed_trying_next`, and `continue`s.
- On `httpx.ConnectError`: same pattern -- `record_failure()`, extend exclude, store `last_error` as a 502, log `backend_connect_failed`, and `continue`.
- On 4xx `HTTPException`: re-raised immediately (not a backend failure).
- After the loop: raises `last_error` if set, otherwise raises 404.

**Streaming circuit breaker wrapper** (`gateway/routes/chat.py: _wrap_stream_with_circuit_breaker()`):
- Wraps an async generator. Iterates through all chunks, checking each for `'"stream_error"'` substring. In the `finally` block, calls `record_failure()` if any error was detected, otherwise `record_success()`.
- The route handler passes the raw streaming generator and the `CircuitBreaker` instance to this wrapper, then wraps the result in `StreamingResponse`.

**Lifespan initialization** (`gateway/main.py: lifespan()`):
- Line 36-38: `app.state.circuit_breakers = CircuitBreakerRegistry(list(app.state.registry.backends.keys()))` -- creates breakers for all backends at startup.
- Line 46-47: `app.state.circuit_breakers.sync_backends(list(app.state.registry.backends.keys()))` -- called inside `handle_sighup()` on config reload.

**Admin endpoint** (`gateway/routes/admin.py: list_backends()`):
- Line 48-49: retrieves `cb_registry` from `app.state`, calls `get_all_snapshots()`.
- Lines 51-60: each backend in the response includes `"health": cb_snapshots.get(b.name, {}).get("state", "unknown")` and `"circuit_breaker": cb_snapshots.get(b.name, {})` containing `state`, `error_rate`, `requests_in_window`, and `current_cooldown_s`.

**Config reload** (`gateway/routes/admin.py: reload_config()` and `gateway/main.py: handle_sighup()`):
- Both call `app.state.circuit_breakers.sync_backends(list(new_registry.backends.keys()))` after swapping the registry. This preserves circuit breaker state for existing backends while adding breakers for new backends (starting CLOSED) and removing breakers for deleted backends.

### Key design decisions

1. **In-memory state, not Redis.** Each gateway instance maintains its own circuit breaker state. In a multi-instance deployment, instances have independent views of backend health. This is acceptable because a backend that is truly down will fail on all instances -- they all trip independently, with slight timing divergence. The alternative (Redis-shared state) adds latency to every request (reading and writing CB state) and introduces a dependency on Redis availability for the core routing path. If Redis is down, the gateway cannot make routing decisions. In-memory state has zero additional latency and no external dependencies.

2. **Rolling window with deque for efficient pruning.** The `_requests` deque stores timestamped outcomes. Since entries are appended in monotonic order, expired entries are always at the front. `_prune_window()` calls `popleft()` in a `while` loop -- O(k) where k is the number of expired entries, typically a small number per call. A fixed-size counter (e.g., "last 10 requests") would lose temporal information -- 10 failures over 60 seconds is different from 10 failures over 1 second. The rolling window captures both the rate and recency of failures.

3. **Exponential backoff on probe failure (30s to 300s cap).** A backend that fails its first probe is unlikely to recover in another 30 seconds. Doubling the cooldown (30s, 60s, 120s, 240s, 300s cap) reduces probe frequency for persistently unhealthy backends, preventing the gateway from repeatedly sending traffic to a backend that is clearly not recovering. The cap at 300 seconds (5 minutes) ensures that even the worst case recovers eventually -- the backend is never permanently excluded. When a probe succeeds, the cooldown resets to 30s, so a recovered backend quickly returns to full service.

4. **Passive probing, no background thread.** The breaker transitions from OPEN to HALF_OPEN when the next real request calls `allow_request()` after the cooldown expires. There is no background health check thread. This is simpler (no thread management, no additional backend load from health probes) and sufficient for the gateway's traffic patterns -- if a backend is in the routing path, there will be requests that trigger the probe. The tradeoff is that a backend with zero traffic (e.g., serving a model nobody is currently requesting) stays in whatever state it is in until a request arrives.

5. **`time.monotonic()` for all timestamps.** `time.monotonic()` is immune to system clock adjustments (NTP corrections, manual changes, daylight saving). Using `time.time()` would risk incorrect cooldown calculations if the system clock jumps backward (cooldown appears to not have elapsed) or forward (cooldown expires prematurely). All timestamps in the circuit breaker -- window entries, `_opened_at`, and pruning cutoffs -- use `monotonic()`.

6. **Retry loop with exclude accumulation for within-request failover.** The non-streaming path retries up to 3 times, accumulating failed backends in the `exclude` set on each iteration. This means a single client request can transparently fail over through up to 3 backends. The exclude set starts with `cb_registry.get_open_backends()` (already-tripped breakers) and grows as new failures are discovered. This combines proactive exclusion (known-bad backends) with reactive exclusion (just-failed backends) in the same request.

7. **`sync_backends()` preserves state on config reload.** When config is reloaded, `sync_backends()` compares the current breaker set with the new backend names. Existing backends keep their `CircuitBreaker` instances (and their state -- OPEN, CLOSED, window data, cooldown). New backends get fresh breakers starting in CLOSED. Removed backends have their breakers deleted. This prevents a config reload from resetting a tripped circuit breaker, which would immediately send traffic back to a backend that was just failing.

8. **`_half_open_probe_sent` flag prevents concurrent probes.** In HALF_OPEN state, only one request should probe the backend. Without this flag, multiple concurrent `allow_request()` calls could all see HALF_OPEN and all return `True`, sending multiple probes simultaneously. The flag is set to `True` on the first `allow_request()` that returns `True` in HALF_OPEN, and subsequent calls return `False`. The flag is reset on `record_success()` (transition to CLOSED) or `record_failure()` (transition to OPEN).

9. **Streaming error detection via `"stream_error"` substring check.** The `_wrap_stream_with_circuit_breaker()` wrapper checks each SSE chunk for the `'"stream_error"'` substring. This matches the error event format established in Phase 4 (streaming normalization), where backend errors during streaming are emitted as `data: {"error": {"message": "...", "type": "stream_error"}}`. The substring check is deliberately simple -- it avoids JSON parsing every chunk for performance.

### Alternatives considered

1. **Active health checks (background thread) vs passive probing.** Active health checks would send periodic requests to a health endpoint on each backend (e.g., `GET /health`). This detects failures faster (no waiting for a real request to trigger the probe) and can detect unhealthy backends that are not currently receiving traffic. However, it adds complexity (thread management, health endpoint configuration per provider, additional backend load from probes) and is unnecessary for the gateway's use case -- backends in the routing path always have traffic to trigger passive probes. Active health checks also require knowledge of each provider's health endpoint, which varies (Ollama: `GET /`, OpenAI: none, Anthropic: none).

2. **Redis-shared circuit breaker state vs in-memory.** Sharing CB state across gateway instances via Redis would ensure all instances have the same view of backend health, eliminating the window where one instance has tripped a breaker but others have not yet. However, this adds a Redis read/write on every request (to check and update CB state), increasing latency. It also makes the circuit breaker dependent on Redis availability -- if Redis is down, the gateway cannot determine backend health. Since a truly failing backend will trip breakers on all instances independently (just with slight timing differences), the convergence time is acceptable. The slight divergence is a feature, not a bug: it provides natural load spreading during transitions.

3. **Simple retry without circuit breaker state.** A retry loop without circuit breaker state would try the next backend on failure, but it would not remember which backends are unhealthy across requests. Every request would start by trying the same (potentially dead) backend, fail, then retry. The circuit breaker's OPEN state means subsequent requests skip the known-bad backend entirely, avoiding the wasted first attempt. This is the difference between O(1) routing to a healthy backend and O(k) retries where k is the number of unhealthy backends.

4. **Fixed cooldown vs exponential backoff.** A fixed 30-second cooldown would probe the backend at a constant rate regardless of how many probes have failed. If the backend is down for 10 minutes, it would receive 20 probe requests (one every 30 seconds), each of which fails and adds latency for one client. Exponential backoff with a 300-second cap means the same 10-minute outage receives approximately 7 probes (at 30s, 60s, 120s, 240s, 300s, 300s, 300s). Fewer probes mean fewer clients experience the probe-failure latency.

### Failure modes and edge cases

- **All backends OPEN for a model.** `find_backend_for_model` returns `None` because the hash ring's `get_node()` cannot find a non-excluded node. The non-streaming path raises `HTTPException(503, "All backends unavailable for this model")`. The streaming path raises the same 503 before returning `StreamingResponse`, so the client receives a proper HTTP error status (not a 200 with an error in the body).
- **HALF_OPEN probe fails.** `record_failure()` doubles `_current_cooldown` (up to `max_cooldown=300`), resets `_half_open_probe_sent`, and transitions back to OPEN. The next probe attempt occurs after the doubled cooldown. The client that served as the probe receives the error (5xx or ConnectError) and, in the non-streaming path, the retry loop tries the next backend.
- **`httpx.ConnectError` before any response bytes.** Caught in the retry loop's `except httpx.ConnectError` block. `record_failure()` is called, the backend is added to the exclude set, and the loop continues with the next attempt. The client is unaware of the failed attempt if the next backend succeeds.
- **Mid-stream error.** The `_wrap_stream_with_circuit_breaker()` wrapper detects the error (via `"stream_error"` substring or raised exception) and calls `record_failure()` in the `finally` block. The client receives the partial stream followed by the error SSE event. There is no retry for streaming -- retrying would require re-sending the entire request and discarding the partial stream the client already received.
- **Config reload while backends are OPEN.** `sync_backends()` preserves existing breaker instances. An OPEN breaker for `mock-openai-1` stays OPEN through the reload. If the backend is removed from config, its breaker is deleted (the backend no longer exists, so no breaker is needed). If a new backend is added, it starts CLOSED.
- **Split-brain across gateway instances.** In a multi-instance deployment, each instance has independent CB state. Instance A may have `mock-openai-1` OPEN while Instance B still has it CLOSED (because B received fewer failures or started later). This is acceptable: Instance B will trip its own breaker after accumulating enough failures. The convergence time is bounded by the rolling window size (60 seconds) and the min_requests threshold (10 requests). During the divergence window, some traffic from Instance B still hits the unhealthy backend, but the retry loop provides per-request failover.
- **Low-traffic backend never reaches `min_requests`.** If a backend receives fewer than 10 requests in the 60-second window, the breaker never trips regardless of the failure rate. This prevents false positives on backends with sporadic traffic -- 2 failures out of 3 requests is a 67% failure rate, but it could be transient. The `min_requests` threshold ensures that the breaker only acts on statistically meaningful sample sizes.
- **Clock skew with `time.monotonic()`.** Not an issue -- `time.monotonic()` is per-process and monotonically increasing. It cannot jump backward or be affected by NTP. The only scenario where it misbehaves is if the process is suspended (e.g., `SIGSTOP`) for longer than the window size, in which case old entries would still be in the deque but would be pruned on the next `_prune_window()` call.

### Observability

**Structured log events:**
- `circuit_state_change` -- logged on every state transition with `backend`, `old_state`, `new_state`, and `current_cooldown`. Enables operators to track breaker state changes across backends and correlate them with incident timelines.
- `backend_failed_trying_next` -- logged in the retry loop when a backend returns 5xx. Includes `backend`, `status_code`, and `attempt` number.
- `backend_connect_failed` -- logged when `httpx.ConnectError` occurs. Includes `backend` and `attempt` number.

**Admin endpoint** (`GET /admin/backends`):
- Each backend includes a `circuit_breaker` object with `state` (CLOSED/OPEN/HALF_OPEN), `error_rate` (float, 0.0-1.0), `requests_in_window` (int), and `current_cooldown_s` (float). The top-level `health` field mirrors the `state` value for quick scanning.
- Operators can poll this endpoint to monitor backend health across the fleet. Example: `curl /admin/backends | jq '.[] | select(.health != "CLOSED")'` shows all unhealthy backends.

**Request-level logging:**
- `chat_request_received` and `chat_request_completed` log events include `backend` name, enabling per-backend traffic analysis and failure correlation.
- The `X-Backend` response header (from Phase 5) shows which backend ultimately handled the request, including after failover.

### Testing

**Unit tests -- circuit breaker** (`tests/unit/test_circuit_breaker.py`, 19 tests):
- `TestCircuitBreakerTransitions` (9 tests): starts CLOSED, stays CLOSED below threshold (4 failures out of 10 = 40%), trips to OPEN at threshold (5 failures out of 10 = 50%), does not trip below `min_requests` (5 failures out of 5), `allow_request()` returns `False` when OPEN, transitions to HALF_OPEN after cooldown (mocked `time.monotonic`), HALF_OPEN success transitions to CLOSED, HALF_OPEN failure transitions back to OPEN, HALF_OPEN allows only one probe (`_half_open_probe_sent` flag).
- `TestExponentialBackoff` (3 tests): cooldown doubles on repeated HALF_OPEN failure (30 -> 60 -> 120), cooldown caps at `max_cooldown` (30 -> 60 -> 120 -> 240 -> 300 -> 300), cooldown resets to initial on HALF_OPEN success (60 -> 30).
- `TestRollingWindow` (1 test): entries older than `window_size` are pruned by `_prune_window()` (10 entries at t=0, all gone at t=61).
- `TestCircuitBreakerRegistry` (6 tests): `get_open_backends` returns names of tripped breakers, `get_all_snapshots` returns snapshot for every breaker, `sync_backends` adds new backends, `sync_backends` removes stale backends, `sync_backends` preserves existing state, `snapshot` includes state/error_rate/requests_in_window/current_cooldown_s.

**Integration tests -- circuit breaker** (`tests/integration/test_circuit_breaker.py`, 7 tests):
- `TestNonStreamingFailover` (3 tests): failover to next backend on 5xx (first call raises 500, second succeeds, client sees 200), failover on `httpx.ConnectError` (same pattern), 503 when all backends fail (always_fail mock, response status >= 500).
- `TestStreamingCircuitBreaker` (2 tests): streaming records success on circuit breaker (mock stream completes normally, response is 200 with `text/event-stream`), streaming error records failure (mock stream emits `stream_error` event, circuit breaker `record_failure` is called).
- `TestAdminCircuitState` (2 tests): `GET /admin/backends` includes `health` and `circuit_breaker` fields with correct structure, circuit breaker state reflects failures after making requests that fail.

**E2E (Docker):**
- `docker compose stop mock-openai-2` followed by requests to `mock-gpt-markdown` from a tenant that normally routes to `mock-openai-2`. After enough failures to trip the breaker, subsequent requests return the `X-Backend` header showing a different backend (e.g., `mock-openai-3`). `docker compose start mock-openai-2` followed by waiting for cooldown expiry shows the breaker transitioning back to CLOSED and traffic returning to `mock-openai-2`.

### Production gaps

- **No distributed circuit breaker state.** Each gateway instance maintains independent CB state. In a multi-instance deployment, there is a convergence window where instances disagree on backend health. A centralized CB state store (Redis, etcd) would eliminate this divergence but adds latency and an availability dependency.
- **No active health probes.** The circuit breaker relies on real traffic to detect failures. A backend that serves a low-traffic model could stay in OPEN state for a long time before a request triggers the HALF_OPEN probe. Active health checks (background thread pinging each backend) would detect recovery faster.
- **No Prometheus metrics.** Circuit breaker state changes, error rates, and failover events are logged but not exported as Prometheus metrics. Phase 10 adds `circuit_breaker_state` gauge, `circuit_breaker_trip_total` counter, and `failover_total` counter.
- **No admin endpoint to manually open/close circuits.** Operators cannot force a backend into OPEN state (for maintenance) or force it back to CLOSED (to override a false positive). A `POST /admin/backends/{name}/circuit` endpoint with `{"state": "OPEN"}` or `{"state": "CLOSED"}` would enable manual intervention.
- **No per-backend timeout configuration.** The circuit breaker uses the same `window_size`, `failure_threshold`, `min_requests`, and `cooldown` for all backends. In production, different backends may have different reliability profiles -- a flaky backend might need a lower threshold while a stable one could tolerate a higher one. Per-backend CB configuration in `backends.yaml` would address this.
- **Streaming does not retry.** If the first backend fails during streaming, the client receives a partial stream with an error event. There is no mechanism to retry the stream on a different backend. This is inherent to HTTP streaming -- once bytes are sent, they cannot be unsent.

### Interview talking points

- **Why circuit breaker over simple retry.** Simple retry without state means every request starts by trying the same dead backend, wastes an attempt, then falls through to a healthy backend. The circuit breaker's OPEN state remembers that the backend is unhealthy, so subsequent requests skip it entirely -- O(1) routing to a healthy backend instead of O(k) retries through k dead backends. The breaker also gives the backend time to recover by reducing probe frequency via exponential backoff, rather than hammering it with retries.
- **Three-state model and why HALF_OPEN exists.** HALF_OPEN is the safe path back to healthy. Without it, the breaker would have to choose between staying OPEN forever (never recovering) or jumping directly from OPEN to CLOSED (sending full traffic to a possibly-still-broken backend). HALF_OPEN allows exactly one probe request. If it succeeds, the backend is likely healthy and full traffic resumes. If it fails, the backend stays excluded with a longer cooldown. This single-probe approach avoids the thundering herd problem where a recovered backend is suddenly hit with 100% of its traffic before it has fully warmed up.
- **Interaction with consistent hashing.** The circuit breaker and hash ring are composed, not coupled. The breaker produces an exclude set; the ring's `get_node(key, exclude=...)` walks clockwise past excluded nodes to the next healthy backend. The ring does not know about circuit breakers, and the breaker does not know about hash rings. This separation means either component can be replaced or modified independently. The `exclude` parameter was designed into the ring in Phase 5 specifically to enable this composition.

### Likely interview questions

**Q: "What happens during the cooldown period?"**
**A:** All requests to that model that would have been routed to the failed backend are instead routed to the next clockwise node on the hash ring. The `get_open_backends()` call returns the failed backend's name, which is passed to `find_backend_for_model(exclude=...)`, and the ring's `get_node()` skips past its virtual nodes. The failed backend receives zero traffic until the cooldown expires and a probe request is allowed through. If the probe succeeds, the breaker closes and traffic resumes. If it fails, the cooldown doubles and the backend stays excluded for longer.

**Q: "Why in-memory and not distributed state?"**
**A:** Each gateway instance has its own view of backend health. A backend that is truly down will fail on all instances, so they all trip their breakers independently -- the timing differs by at most the window size (60 seconds) plus the min_requests threshold (10 requests at whatever the request rate is). The slight divergence in timing is acceptable because the retry loop provides per-request failover even before the breaker trips. The alternative -- Redis-shared state -- adds a Redis read/write on the critical path of every request and makes the routing layer dependent on Redis availability. If Redis is down, the gateway cannot determine which backends are healthy. In-memory state has zero additional latency, zero external dependencies, and convergence time measured in seconds.

**Q: "How does the exponential backoff prevent thundering herd on recovery?"**
**A:** When a backend recovers, the first probe request in HALF_OPEN succeeds and the breaker closes. Traffic resumes to that backend gradually because only requests whose routing key maps to that backend (approximately 1/N of total traffic) start flowing to it. There is no sudden redirect of all traffic. The exponential backoff during the outage means fewer probes hit the recovering backend (7 probes over 10 minutes instead of 20), giving it more time to stabilize. When the breaker closes, the cooldown resets to 30 seconds, so if the backend fails again, the probing starts at a reasonable frequency rather than at the escalated interval.

**Q: "What happens if a circuit breaker is stuck OPEN due to a transient issue that resolved?"**
**A:** The circuit breaker cannot get permanently stuck. After the current cooldown period expires (at most 300 seconds / 5 minutes), `allow_request()` transitions to HALF_OPEN and allows one probe. If the issue has resolved, the probe succeeds, the breaker closes, and traffic resumes. The maximum time a backend can be excluded after a transient issue is the current cooldown period -- in the worst case, 300 seconds. There is no mechanism for permanent exclusion; the exponential backoff always caps and the probe always fires eventually.
