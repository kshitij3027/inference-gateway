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

## Distributed Rate Limiter

### Why this exists

Without rate limiting, a single tenant can consume all backend capacity. LLM inference is expensive -- both in compute cost (GPU-seconds per request) and in money (per-token pricing from OpenAI, Anthropic). A tenant that sends a burst of long-context requests can saturate every backend, causing latency spikes or outright denial of service for other tenants. Three independent rate limiting dimensions protect different things:

1. **RPS (requests per second)** protects the gateway itself. A burst of concurrent requests can exhaust connection pools, memory, and file descriptors in the gateway process before the backends even see them.
2. **RPM (requests per minute)** protects budget predictability. Even if individual requests are small, a high sustained rate accumulates cost. RPM gives operators a coarse-grained knob for per-tenant throughput.
3. **Token budget (daily)** protects cost directly. A single request with a 100k-token context costs orders of magnitude more than a short chat message. Token budgets cap the total spend per tenant per day regardless of how many requests they send.

All three dimensions must be enforced atomically and consistently across multiple gateway instances. A tenant sending 10 concurrent requests must not be able to bypass a 5 RPS limit by hitting different instances. This requires a shared state store (Redis) and atomic check-and-increment logic.

### How it works

1. **Request arrives, authentication succeeds.** The `get_current_tenant` dependency resolves the `TenantConfig`, which carries `rate_limit_rps`, `rate_limit_rpm`, and `token_budget_daily` fields. These are defined in `config/backends.yaml` under the `tenants` section (e.g., `tenant-alpha` has `rate_limit_rps: 10`, `rate_limit_rpm: 60`, `token_budget_daily: 500`). Tenants without these fields (e.g., `tenant-beta`) have no rate limits.

2. **Rate limiter check begins.** The route handler in `gateway/routes/chat.py: chat_completions()` retrieves `rate_limiter` from `request.app.state` (lines 60-61). If `rate_limiter is None` (Redis was unavailable at startup), the entire rate limiting block is skipped and the request proceeds. Otherwise, `rate_limiter.check_rate_limit()` is called with `tenant_id`, `request_id`, `rps_limit`, and `rpm_limit`.

3. **RPS sliding window check (1-second window).** `RateLimiter.check_rate_limit()` in `gateway/rate_limiter.py` calls `_check_window()` with `key=ratelimit:{tenant_id}:rps`, `window_size=1.0`, `limit=rps_limit`, and `expire_seconds=2`. This executes the Lua script `_SLIDING_WINDOW_SCRIPT` atomically on Redis:
   - `ZREMRANGEBYSCORE key -inf window_start` -- removes all sorted set members with scores older than `now - 1.0`, cleaning entries outside the 1-second window.
   - `ZCARD key` -- counts remaining members in the window.
   - If `count >= limit`: returns `{0, count}` (denied). The Lua script short-circuits before adding the new entry.
   - If `count < limit`: `ZADD key now member` adds the current request, `EXPIRE key 2` sets a 2-second TTL for auto-cleanup, and returns `{1, count}` (allowed).

4. **Short-circuit on RPS denial.** If the RPS check returns denied, `check_rate_limit()` returns `(False, {"limit_type": "rps", "limit": rps_limit, "current": count, "retry_after": 1.0})` immediately. The RPM check is never executed.

5. **RPM sliding window check (60-second window).** If RPS passes, the same Lua script runs again with `key=ratelimit:{tenant_id}:rpm`, `window_size=60.0`, `limit=rpm_limit`, and `expire_seconds=120`. The logic is identical to the RPS check but with a longer window.

6. **Token budget pre-check.** After both RPS and RPM pass, `check_token_budget()` reads the daily token counter with `GET ratelimit:{tenant_id}:tokens:{today}` (a plain Redis string, not a sorted set). If the current total equals or exceeds `token_budget_daily`, the request is denied. The `retry_after` value is calculated as seconds until midnight UTC.

7. **Denial response.** On any denial, the route handler raises `HTTPException(status_code=429)` with a structured JSON body containing `error`, `type` (which dimension triggered the denial), `limit`, `current`, and `retry_after`. The `Retry-After` HTTP header is set to the integer seconds value.

8. **Remaining counts for response headers.** On allow, `get_remaining()` is called to compute how many requests remain in each window. This calls `ZREMRANGEBYSCORE` + `ZCARD` for each dimension (non-atomically -- these are informational, not authoritative). The results are stored on `request.state.rate_limit_remaining`.

9. **Response headers injected by middleware.** The `request_id_middleware` in `gateway/main.py` (lines 105-110) reads `rate_limit_remaining` from `request.state` after the response is generated. If present, it sets `X-Ratelimit-Remaining-Rps` and `X-Ratelimit-Remaining-Rpm` headers on the response.

10. **Token recording after response.** After a successful non-streaming response, the route handler in `gateway/routes/chat.py` (lines 218-222) calls `rate_limiter.record_tokens(tenant_id, result.usage.total_tokens)`. This executes `INCRBY ratelimit:{tenant_id}:tokens:{today} total_tokens` and `EXPIRE key 90000` (25 hours). The token count is cumulative for the day.

### Implementation

**Lua script** (`gateway/rate_limiter.py: _SLIDING_WINDOW_SCRIPT`):
- Accepts 1 key and 5 arguments: `window_start`, `now`, `member`, `limit`, `expire_s`.
- Executes three Redis commands atomically: `ZREMRANGEBYSCORE` (cleanup), `ZCARD` (count), and conditionally `ZADD` + `EXPIRE` (insert).
- Returns a 2-element array: `{allowed (0 or 1), count}`.
- The `member` value is `{request_id}:rps` or `{request_id}:rpm`, ensuring uniqueness across concurrent requests to the same tenant.

**RateLimiter class** (`gateway/rate_limiter.py: RateLimiter`):
- `__init__(self, redis_client)` -- stores the Redis client. The Lua script is not registered at construction time.
- `_ensure_script()` -- lazily calls `redis.register_script(_SLIDING_WINDOW_SCRIPT)` on first use. This returns a callable that handles `EVALSHA`/`EVAL` fallback transparently.
- `check_rate_limit(tenant_id, request_id, rps_limit, rpm_limit) -> (bool, dict | None)` -- checks RPS then RPM via `_check_window()`. Returns `(True, None)` if both pass, `(False, deny_info)` on first failure. If either limit is `None`, that dimension is skipped entirely (no Redis call).
- `_check_window(key, now, window_size, member, limit, expire_seconds) -> (bool, int)` -- computes `window_start = now - window_size`, invokes the Lua script with the key and arguments, parses the 2-element result.
- `check_token_budget(tenant_id, budget) -> (bool, dict | None)` -- `GET` on the daily key, integer comparison against budget. If `budget is None`, returns `(True, None)` without touching Redis.
- `record_tokens(tenant_id, tokens) -> int` -- `INCRBY` on the daily key, `EXPIRE` with 90000 seconds (25 hours). Returns the new total.
- `get_remaining(tenant_id, rps_limit, rpm_limit) -> dict` -- `ZREMRANGEBYSCORE` + `ZCARD` for each active dimension, returns `{"rps": remaining, "rpm": remaining}`.

**Redis key schema:**
- `ratelimit:{tenant_id}:rps` -- sorted set, scores are Unix timestamps, members are `{request_id}:rps`. TTL: 2 seconds.
- `ratelimit:{tenant_id}:rpm` -- sorted set, scores are Unix timestamps, members are `{request_id}:rpm`. TTL: 120 seconds.
- `ratelimit:{tenant_id}:tokens:{YYYY-MM-DD}` -- plain string (integer counter). TTL: 90000 seconds (25 hours).

**Route integration** (`gateway/routes/chat.py: chat_completions()`, lines 58-106):
- Retrieves `rate_limiter` from `request.app.state`. If `None`, skips all rate limiting.
- Calls `check_rate_limit()`, then `check_token_budget()`. On denial, raises `HTTPException(429)` with structured detail and `Retry-After` header.
- Calls `get_remaining()` and stores on `request.state.rate_limit_remaining`.
- Wraps the entire block in `try/except`: `HTTPException` is re-raised (429s propagate), all other exceptions are caught and logged as `rate_limit_check_failed` (graceful degradation -- request proceeds).
- After successful response (line 218-222): calls `record_tokens()` if `tenant.token_budget_daily` is set and `rate_limiter` is not `None`.

**Middleware integration** (`gateway/main.py: request_id_middleware()`, lines 105-110):
- After `call_next(request)`, reads `request.state.rate_limit_remaining`. If present, sets `X-Ratelimit-Remaining-Rps` and `X-Ratelimit-Remaining-Rpm` response headers.

**Lifespan initialization** (`gateway/main.py: lifespan()`, lines 43-53):
- Connects to Redis via `aioredis.from_url(redis_url, decode_responses=True)`.
- Calls `ping()` to verify connectivity.
- On success: creates `RateLimiter(redis)` and stores as `app.state.rate_limiter`.
- On failure: logs `redis_unavailable`, sets `app.state.redis = None` and `app.state.rate_limiter = None`. The gateway starts and operates without rate limiting.

**Tenant configuration** (`config/backends.yaml`):
- `tenant-alpha`: `rate_limit_rps: 10`, `rate_limit_rpm: 60`, `token_budget_daily: 500`.
- `tenant-beta`: no rate limit fields -- all three limits default to `None`, meaning unlimited.

### Key design decisions

1. **Sliding window via sorted sets -- true sliding window, no boundary effects.** A sorted set with Unix timestamps as scores and request IDs as members creates a true sliding window. At any point in time, `ZREMRANGEBYSCORE` removes entries older than the window, and `ZCARD` counts entries within the window. There are no fixed bucket boundaries, so there is no "burst at the window edge" problem where a client can send 2x the limit by timing requests to straddle a boundary.

2. **Atomic Lua script -- prevents TOCTOU race between concurrent requests.** Without atomicity, two concurrent requests could both read `count=9` (under a limit of 10), both decide to proceed, and both add an entry -- resulting in 11 entries in a window with a limit of 10. The Lua script executes `ZREMRANGEBYSCORE`, `ZCARD`, and conditional `ZADD` as a single atomic operation on Redis. No other command can interleave between the count check and the insert.

3. **Token budget uses `INCRBY`, not sorted sets (different semantics).** Token budget is a cumulative daily counter, not a sliding window. There is no "window" to slide -- the budget resets at midnight (via key expiry), not on a rolling basis. `INCRBY` is the correct primitive: simple, atomic, and O(1). Using a sorted set for tokens would waste memory (storing every request's token count as a member) for no benefit.

4. **Graceful degradation -- Redis down means rate limiting disabled, requests pass through.** The gateway's primary function is proxying LLM requests. Rate limiting is a secondary protection mechanism. If Redis is unavailable, the gateway should still serve requests rather than rejecting them all. This is implemented at two levels: (a) at startup, `rate_limiter = None` if Redis fails to connect, and the route handler skips rate limiting when `rate_limiter is None`; (b) at request time, any exception from the rate limiter is caught and logged as `rate_limit_check_failed`, and the request proceeds.

5. **Pre-check approximate -- token budget can overshoot because tokens are unknown before inference.** The token budget pre-check reads the current daily total and compares it to the budget. But this check happens before the LLM backend processes the request -- the gateway does not know how many tokens the response will contain. Two concurrent requests that both pass the pre-check with `current=490` and `budget=500` will both proceed, and if each generates 50 tokens, the actual total will be 590. This overshoot is acceptable because (a) the alternative (reserving tokens before inference) requires knowing the response size in advance, which is impossible, and (b) the overshoot is bounded by the maximum tokens a single request can generate.

6. **Lazy script registration -- `register_script()` called on first use, not at import time.** The `_ensure_script()` method registers the Lua script on the first call to `check_rate_limit()`. This avoids importing `rate_limiter.py` causing a Redis call (which would fail in tests without a Redis connection). Lazy registration also means the script is only registered if rate limiting is actually used.

7. **25-hour TTL on daily token keys -- auto-expire even with timezone edge cases.** The daily token key is `ratelimit:{tenant_id}:tokens:{YYYY-MM-DD}`. The date is `date.today().isoformat()`, which uses the gateway process's local timezone. The 25-hour TTL (90000 seconds) ensures the key expires even if the process timezone is ahead of UTC -- a key created at 23:59 local time will expire 25 hours later, well after midnight in any timezone. Without the TTL, stale keys would accumulate indefinitely in Redis.

8. **Request ID as sorted set member -- guarantees uniqueness across concurrent requests.** The member value for each sorted set entry is `{request_id}:rps` or `{request_id}:rpm`. Since `request_id` is a UUID generated per request (either from the `X-Request-ID` header or `uuid.uuid4()`), every member is unique. If a fixed member were used (e.g., just the tenant ID), `ZADD` would update the existing member's score rather than adding a new entry, and `ZCARD` would always return 1.

### Alternatives considered

1. **Fixed-window counters (INCR + TTL) -- rejected due to boundary burst problem.** A fixed-window counter increments an integer key and sets a TTL equal to the window size. At t=0, the key is created with TTL=1s. At t=0.9, the counter reaches the limit. At t=1.0, the key expires and a new window starts. A client can send `limit` requests at t=0.9 (end of window 1) and another `limit` requests at t=1.0 (start of window 2), achieving 2x the limit in a 1-second span. Sliding windows eliminate this because the window is always anchored to "now minus window_size."

2. **Token bucket algorithm -- rejected because LLM backends cannot handle bursts.** Token bucket allows accumulated "tokens" (not LLM tokens -- algorithm tokens) to be spent in a burst. A bucket with rate=10/s and capacity=50 allows a burst of 50 requests followed by a sustained 10/s. LLM backends (especially Ollama running on limited GPU) cannot handle 50 concurrent requests -- they queue internally and response times degrade severely. Sliding window rate limiting enforces a strict maximum within any window, preventing bursts that the backend cannot absorb.

3. **In-memory rate limiting -- rejected because it does not work across multiple gateway instances.** Storing rate limit counters in the gateway process works for a single instance but fails when multiple instances sit behind a load balancer. A tenant could send `limit` requests to each of N instances, achieving N times the limit. Redis provides a shared state store that all instances can read and write atomically.

4. **Redis INCR with TTL (non-sorted-set sliding window) -- rejected because it is not a true sliding window.** A simple `INCR` with `EXPIRE` creates a fixed window anchored to the first request in the window. It has the same boundary problem as fixed-window counters. The sorted set approach tracks individual request timestamps, enabling true sliding window semantics where the window always covers "the last N seconds from now."

### Failure modes and edge cases

- **Redis unavailable at startup.** The `lifespan()` function in `gateway/main.py` catches the exception from `redis.ping()`, logs `redis_unavailable`, and sets `app.state.rate_limiter = None`. The gateway starts and serves requests without rate limiting. This is a deliberate design choice -- the gateway degrades to "unlimited" rather than refusing to start.

- **Redis fails mid-request.** The `try/except` block in `chat_completions()` (lines 102-106) catches any non-HTTPException from the rate limiter and logs `rate_limit_check_failed`. The request proceeds as if rate limiting were disabled. This handles transient Redis failures (network blip, Redis restart) without causing a cascade of 500 errors from the gateway.

- **Token budget overshoot on concurrent requests.** Two concurrent requests from the same tenant both call `check_token_budget()` and both read `current=490` with `budget=500`. Both pass the pre-check. Both complete inference and call `record_tokens()` -- one increments to 540, the other to 590. The budget is exceeded by 90 tokens. This is acceptable because (a) the overshoot is bounded by the number of concurrent requests times the maximum tokens per response, and (b) the next request will see `current=590 >= budget=500` and be denied. The window of overshoot is the duration of a single inference request.

- **Clock skew across gateway instances.** The Lua script uses `ARGV[2]` (the `now` timestamp passed by the gateway) for `ZADD` scores, not Redis server time. If two gateway instances have different clocks, their sorted set entries will have slightly different scores. This is mitigated because `ZREMRANGEBYSCORE` uses `window_start = now - window_size` from the same instance's clock -- the cleanup is consistent with the insertion for each instance. In practice, NTP keeps instances within milliseconds of each other, and a few milliseconds of skew in a 1-second or 60-second window is negligible.

- **Sorted set grows unbounded.** Every request adds a member to the sorted set. Without cleanup, a high-traffic tenant could accumulate millions of entries. Two mechanisms prevent this: (a) `ZREMRANGEBYSCORE` at the start of every Lua script execution removes entries outside the window, and (b) `EXPIRE` on every `ZADD` sets a TTL on the key (2 seconds for RPS, 120 seconds for RPM). If no requests arrive after the TTL, Redis deletes the entire key. The sorted set is self-cleaning.

- **Midnight rollover for token budget.** The daily token key includes the date (`YYYY-MM-DD`). At midnight (in the gateway's local timezone), `date.today().isoformat()` returns a new date string, and subsequent requests write to a new key. The previous day's key has a 25-hour TTL and will be auto-deleted. There is no explicit reset operation -- the rollover is implicit in the key naming scheme.

- **`retry_after` calculation for token budget.** `check_token_budget()` computes seconds until midnight UTC using `datetime.now(timezone.utc)`. If the gateway process uses a non-UTC timezone, `date.today()` for the key name uses local time while `retry_after` uses UTC. This mismatch means the `retry_after` value may be slightly off (the key rolls over at local midnight, but `retry_after` counts down to UTC midnight). The `max(1.0, retry_after)` guard prevents negative or zero values.

### Observability

**Structured log events:**
- `rate_limit_exceeded` -- logged when RPS or RPM limit is hit. Includes `tenant_id`, `limit_type` (rps or rpm), `limit`, `current`, and `retry_after`. Enables operators to identify which tenants are hitting limits and which dimension is the bottleneck.
- `token_budget_exceeded` -- logged when daily token budget is exhausted. Includes `tenant_id`, `limit_type` (token_budget_daily), `limit`, `current`, and `retry_after`.
- `rate_limit_check_failed` -- logged when the rate limiter throws an unexpected exception (Redis failure). Includes the error string. Indicates Redis connectivity problems.
- `token_recording_failed` -- logged when `record_tokens()` fails after a successful response. Includes the error string. The response was already sent to the client; token recording is best-effort.

**Response headers:**
- `X-Ratelimit-Remaining-Rps` -- remaining requests in the current 1-second window. Set by `request_id_middleware` in `gateway/main.py`.
- `X-Ratelimit-Remaining-Rpm` -- remaining requests in the current 60-second window. Set alongside the RPS header.
- `Retry-After` -- integer seconds until the client should retry. Set on 429 responses only.

**429 response body:**
- Structured JSON: `{"error": "rate_limit_exceeded" | "token_budget_exceeded", "type": "rps" | "rpm" | "token_budget_daily", "limit": <int>, "current": <int>, "retry_after": <float>}`. Clients can programmatically determine which limit was hit and how long to wait.

### Testing

**Unit tests** (`tests/unit/test_rate_limiter.py`, 15 tests with mocked Redis):

- `TestCheckRateLimit` (6 tests):
  - `test_allows_under_rps_limit` -- Lua script returns `[1, 3]` (allowed, count=3). Asserts `allowed is True` and `info is None`.
  - `test_denies_at_rps_limit` -- Lua script returns `[0, 10]` (denied, count=10). Asserts `allowed is False`, `info["limit_type"] == "rps"`, `info["retry_after"] == 1.0`.
  - `test_rps_passes_rpm_denies` -- First Lua call returns `[1, 5]` (RPS allowed), second returns `[0, 60]` (RPM denied). Uses `side_effect` on the mock script to vary return values by call count. Asserts `info["limit_type"] == "rpm"`.
  - `test_rps_checked_before_rpm` -- Lua script returns `[0, 10]` (RPS denied). Asserts the script was called only once (`script.call_count == 1`), proving RPM was not checked.
  - `test_no_limits_always_allows` -- Both `rps_limit` and `rpm_limit` are `None`. Asserts `allowed is True` and `script.call_count == 0` (no Redis interaction).
  - `test_both_limits_pass` -- Lua script returns `[1, 5]` for both calls. Asserts `allowed is True` and `script.call_count == 2` (both dimensions checked).

- `TestTokenBudget` (4 tests):
  - `test_under_budget_allows` -- `redis.get` returns `"5000"`, budget is 100000. Asserts `allowed is True`.
  - `test_exceeded_budget_denies` -- `redis.get` returns `"100001"`, budget is 100000. Asserts `allowed is False`, `info["limit_type"] == "token_budget_daily"`.
  - `test_none_budget_always_allows` -- Budget is `None`. Asserts `allowed is True` and `redis.get` was not called.
  - `test_no_existing_tokens_allows` -- `redis.get` returns `None` (key does not exist). Asserts `allowed is True`.

- `TestRecordTokens` (1 test):
  - `test_increments_counter` -- Calls `record_tokens("t1", 500)`. Asserts `redis.incrby` and `redis.expire` were each called once, and the returned total matches the mock return value (8500).

- `TestGetRemaining` (4 tests):
  - `test_rps_remaining` -- `zcard` returns 3, limit is 10. Asserts `remaining["rps"] == 7`.
  - `test_rpm_remaining` -- `zcard` returns 50, limit is 60. Asserts `remaining["rpm"] == 10`.
  - `test_no_limits_empty_dict` -- Both limits `None`. Asserts `remaining == {}`.
  - `test_at_limit_returns_zero` -- `zcard` returns 10, limit is 10. Asserts `remaining["rps"] == 0`.

**Integration tests** (`tests/integration/test_rate_limiter.py`, 7 tests with `_MockRateLimiter`):

- `TestRateLimitEnforcement` (3 tests):
  - `test_429_when_rate_limit_exceeded` -- `_MockRateLimiter(deny_after=3)` allows first 3 requests, denies the rest. Sends 5 sequential requests. Asserts `results.count(200) == 3` and `results.count(429) == 2`.
  - `test_429_includes_retry_after_header` -- `_MockRateLimiter(deny_after=0)` denies all. Asserts `"retry-after" in resp.headers`.
  - `test_429_json_body_structure` -- Denies all. Asserts the 429 body contains `error`, `type`, `limit`, and `retry_after` fields.

- `TestRateLimitHeaders` (2 tests):
  - `test_remaining_headers_on_success` -- Asserts `x-ratelimit-remaining-rps` and `x-ratelimit-remaining-rpm` headers are present on 200 responses.
  - `test_remaining_decreases_with_requests` -- Sends 3 sequential requests, collects `x-ratelimit-remaining-rps` values. Asserts `remaining_values[0] > remaining_values[2]`.

- `TestGracefulDegradation` (1 test):
  - `test_no_rate_limiter_passes_through` -- Patches `rate_limiter` to `None`. Asserts the request succeeds with 200 and no rate limit headers are present.

- `TestTokenBudget` (1 test):
  - `test_token_recording_after_response` -- Asserts `mock_rl._tokens_recorded == 8` after a successful response (matching `_mock_response().usage.total_tokens`).

**E2E (Docker):**
- 15 concurrent requests against `tenant-alpha` (which has `rate_limit_rps: 10`). Approximately 10 requests pass with 200 and approximately 5 are rejected with 429. The exact split depends on Redis timing and request arrival order. The 429 responses include `Retry-After: 1` and structured JSON bodies.

### Production gaps

- **No streaming token recording.** `record_tokens()` is called after non-streaming responses only (line 218-222 in `chat.py`). Streaming responses do not have a `usage` object in the final response -- the total token count is unknown until the stream completes. Recording tokens for streaming would require accumulating chunk-level token counts or using the `stream_options: {"include_usage": true}` extension, which not all backends support.
- **No admin endpoint for rate limit stats.** There is no `GET /admin/rate-limits` endpoint to view current usage per tenant (current RPS, RPM, token total). Operators must query Redis directly (`ZCARD ratelimit:tenant-alpha:rps`, `GET ratelimit:tenant-alpha:tokens:2026-03-26`) to inspect rate limit state.
- **No Prometheus metrics.** Rate limit denials, token usage, and Redis errors are logged but not exported as Prometheus counters or histograms. Phase 10 would add `rate_limit_denied_total{tenant, limit_type}`, `tokens_used_total{tenant}`, and `rate_limit_check_errors_total` metrics.
- **No per-backend rate limits.** Rate limits are per-tenant only. There is no mechanism to limit how many requests a single backend receives per second (independent of tenant). If a backend has a provider-imposed rate limit (e.g., OpenAI's 60 RPM per API key), the gateway does not enforce or respect it. Per-backend rate limits would require a separate set of Redis keys keyed by backend name.
- **No burst allowance.** The sliding window enforces a strict maximum. There is no "burst capacity" that allows temporary spikes above the limit. A token bucket or leaky bucket hybrid would allow controlled bursts, but as noted in alternatives, LLM backends generally cannot handle bursts.
- **No rate limit reset or override endpoint.** Operators cannot reset a tenant's token budget counter mid-day or temporarily increase a tenant's RPS limit without editing `backends.yaml` and reloading config.

### Interview talking points

- **Why sliding window over fixed window or token bucket.** Fixed windows have the boundary burst problem (2x limit at the boundary). Token bucket allows bursts that LLM backends cannot absorb. Sliding window via sorted sets gives a true per-second or per-minute limit with no boundary effects and no burst spikes. The tradeoff is memory (one sorted set member per request in the window vs a single integer counter), but with typical RPS limits of 10-100, the sorted set holds at most 100 members -- negligible.
- **Why Redis Lua for atomicity instead of Redis transactions.** Redis `MULTI/EXEC` transactions cannot read a value and make a conditional decision based on it in the same transaction -- there is no "read, branch, write" flow. Lua scripts execute atomically on the Redis server with full access to the Lua language for conditionals and loops. The `_SLIDING_WINDOW_SCRIPT` reads the count, checks against the limit, and conditionally writes -- all without any interleaving from other clients. This is the standard pattern for rate limiting with Redis.
- **Graceful degradation as a first-class design choice.** The gateway's value is proxying LLM requests. If the rate limiter fails, the correct response is to allow the request (fail open), not to reject it (fail closed). Every interaction with the rate limiter is wrapped in a try/except that logs and continues. This is a deliberate tradeoff: during a Redis outage, tenants are temporarily unlimited, but the gateway continues serving its primary function. The alternative (fail closed) would mean a Redis outage causes a complete gateway outage for all tenants.

### Likely interview questions

**Q: "How does the sliding window prevent the boundary burst problem?"**
**A:** Fixed-window counters reset at a fixed boundary (e.g., every second on the second). A client can send `limit` requests at t=0.99 (end of window 1) and `limit` requests at t=1.01 (start of window 2), achieving 2x the limit in a 0.02-second span. The sliding window has no fixed boundaries. At t=1.01, `ZREMRANGEBYSCORE` removes entries older than t=0.01. The entries from t=0.99 are still in the window (they are only 0.02 seconds old). So the count correctly reflects recent activity regardless of when the requests arrive relative to any clock boundary.

**Q: "What happens if Redis goes down while the gateway is running?"**
**A:** Any exception from the rate limiter is caught in the `try/except` block in the route handler. The exception is logged as `rate_limit_check_failed`, and the request proceeds as if rate limiting were disabled. This is fail-open behavior. The gateway continues proxying LLM requests without rate limiting until Redis recovers. On the next request after Redis recovers, `check_rate_limit()` will succeed normally -- there is no reconnection logic needed because `redis.asyncio` handles reconnection transparently.

**Q: "Why not use Redis server time instead of client-provided timestamps?"**
**A:** The Lua script uses `ARGV[2]` (the gateway's `time.time()`) for `ZADD` scores and `ARGV[1]` (the gateway's `now - window_size`) for `ZREMRANGEBYSCORE`. Using Redis `TIME` command inside the Lua script would be possible but adds a Redis call per script execution. More importantly, the gateway timestamps are consistent with each other -- `window_start` and `now` come from the same `time.time()` call, so the window is exactly `window_size` seconds. If Redis server time drifted from gateway time, the cleanup and insertion would use different time bases, potentially leaving stale entries or removing valid ones. In practice, NTP keeps all servers within milliseconds, making this a theoretical concern.

**Q: "How would you handle rate limiting for streaming requests?"**
**A:** RPS and RPM rate limiting works identically for streaming and non-streaming -- the check happens before the request is dispatched to the backend. The gap is in token budget recording: streaming responses do not have a `usage` object in the final SSE event (most providers do not include it). To record tokens for streaming, the gateway would need to either (a) count tokens by parsing the streamed content chunks (approximate, requires a tokenizer), (b) use the `stream_options.include_usage` extension (provider-dependent), or (c) estimate based on the prompt size and a heuristic for response length. Currently, streaming requests consume RPS/RPM budget but not token budget.

## Semantic Response Cache

### Why this exists

LLM inference is expensive -- each completion call incurs latency (seconds), compute cost (GPU time), and API spend. Caching responses to identical questions is the obvious optimization, but exact-match caching has a near-zero hit rate for LLM workloads. Users phrase the same question differently: "What is the capital of France?", "Tell me the capital of France", "France's capital city?" are semantically identical but have no lexical overlap beyond "capital" and "France." A hash of the prompt text produces different keys for each phrasing, so exact-match caching misses all three.

Semantic caching solves this by comparing the *meaning* of prompts rather than their text. Each prompt is converted to a dense vector embedding, and incoming queries are matched against cached embeddings using cosine similarity. If a cached response exists whose prompt is semantically similar above a configurable threshold (default 0.95), the cached response is returned without calling the LLM backend. This converts repeated questions -- regardless of phrasing -- into sub-millisecond Redis lookups instead of multi-second inference calls.

Without this component, the gateway would forward every request to a backend even when an equivalent answer already exists in the system. For workloads with repetitive queries (customer support bots, FAQ systems, educational tools), this wastes significant cost and latency.

### How it works

**Non-streaming request flow:**

1. Client sends `POST /v1/chat/completions` with model, messages, and `stream: false`.
2. Auth middleware authenticates the tenant. Rate limiter checks RPS/RPM/token budget.
3. **Cache lookup:** The route handler retrieves `semantic_cache` from `app.state`. It calls `semantic_cache.lookup(model, messages, tenant_id, cache_isolation)`.
4. Inside `lookup()`:
   a. User messages are extracted and concatenated (system messages excluded from embedding).
   b. A system prompt hash is computed: `SHA256(system_text)[:16]`.
   c. A scope key is built: `cache:scope:{model}:{sys_hash}` (shared) or `cache:scope:{tenant_id}:{model}:{sys_hash}` (tenant-isolated).
   d. All entry UUIDs in the scope set are fetched via `SMEMBERS`.
   e. The query embedding is computed via `all-MiniLM-L6-v2` (384 dimensions).
   f. For each entry UUID, the stored embedding is fetched from the Redis hash and cosine similarity is computed against the query embedding.
   g. The entry with the highest similarity above the threshold (0.95) is returned.
5. **On HIT:** The cached `ChatCompletionResponse` is deserialized and returned. `X-Cache: HIT` and `X-Cache-Similarity: 0.9823` headers are set via `request.state`. The global hit counter is incremented.
6. **On MISS with stampede guard:** If no cache hit, the handler attempts to acquire a SETNX lock (`cache:lock:{prompt_hash}`, 30s TTL). If the lock is already held (another request is computing the same prompt), the waiter polls with exponential backoff (starting at 100ms, doubling up to 2s, 30s deadline). If the other request populates the cache in time, the waiter returns the cached result. If the lock times out, the waiter falls through to the backend call (fail-open).
7. **Backend call:** The request is routed to an LLM backend via the consistent hash ring. The response is returned to the client.
8. **Cache store:** After a successful backend response, `semantic_cache.store()` computes the embedding, generates a UUID, and writes the entry hash + scope set membership in a Redis pipeline with TTL.
9. **Lock release:** The stampede lock is deleted.

**Streaming request flow:**

1. Steps 1-4 are identical. The cache lookup happens before streaming begins.
2. **On HIT:** The cached response is converted into 3 SSE chunks (role delta, content delta, stop delta) plus a `data: [DONE]` sentinel, emitted via `_stream_cached_response()`. The client receives a valid SSE stream from cache.
3. **On MISS:** The backend stream is wrapped in `_tee_stream_for_cache()`, which yields each SSE chunk to the client while buffering content deltas. After the stream completes (generator exhaustion), the buffered content is assembled into a `ChatCompletionResponse` and stored in the cache. The stampede guard is not used for streaming (explained in design decisions).

### Implementation

**`gateway/semantic_cache.py` -- `SemanticCache` class:**
- Constructor takes `redis_client`, `model_name` (default `all-MiniLM-L6-v2`), `similarity_threshold` (default 0.95), `default_ttl` (default 3600s).
- `_model` is lazy-loaded on first `compute_embedding()` call. The `SentenceTransformer` import and model instantiation happen inside `_load_model()`, avoiding the ~2s load time at startup if no cache operations occur.
- `compute_embedding(text) -> list[float]` encodes text with `normalize_embeddings=True` (unit vectors, so dot product equals cosine similarity). Returns Python list of 384 float32 values.
- `cosine_similarity(a, b) -> float` computes via NumPy: `dot(a, b) / (norm(a) * norm(b))`. Handles zero-norm edge case (returns 0.0).
- `_extract_user_text(messages) -> str` concatenates all `role="user"` message contents with newlines. System messages are excluded from embedding computation.
- `_extract_system_hash(messages) -> str` concatenates all `role="system"` message contents and returns `SHA256[:16]`. This partitions the cache so that different system prompts never cross-pollinate results.
- `_build_scope_key(model, sys_hash, tenant_id, cache_isolation) -> str` builds the Redis key prefix. Shared mode: `cache:scope:{model}:{sys_hash}`. Tenant mode: `cache:scope:{tenant_id}:{model}:{sys_hash}`.
- `lookup()` scans all entries in the scope set, computes cosine similarity for each, and returns the best match above threshold. Returns `(ChatCompletionResponse, similarity)` or `(None, None)`. Stale entries (present in the scope set but expired from Redis) are cleaned up via `SREM`.
- `store()` writes the entry as a Redis hash (`cache:entry:{uuid16}`) with fields: `embedding` (JSON-serialized float list), `response` (Pydantic JSON), `model`, `sys_hash`, `tenant_id`, `created_at`, `hit_count`, `entry_id`. Adds the entry ID to the scope set. Both operations run in a pipeline with TTL.
- `acquire_stampede_lock(model, messages) -> (bool, lock_key)` computes a prompt hash and uses `SET lock_key "1" NX EX 30`. Returns whether the lock was acquired and the key name.
- `wait_for_cached_result(model, messages, ..., timeout=30.0)` polls `lookup()` with exponential backoff (0.1s initial, 2x growth, 2.0s cap, 30s deadline).
- `record_hit()` / `record_miss()` increment `cache:stats` hash fields.
- `get_stats()` returns hits, misses, hit_rate, and entry count (via SCAN).
- `flush()` deletes all `cache:*` keys via SCAN + batch DELETE.

**Redis data model:**
- `cache:entry:{uuid16}` -- Hash with 8 fields (embedding, response, model, sys_hash, tenant_id, created_at, hit_count, entry_id). TTL set to `default_ttl` (1 hour).
- `cache:scope:{model}:{sys_hash}` -- Set of entry UUIDs that share the same model and system prompt. TTL refreshed on each store. Acts as a secondary index for scoped lookups.
- `cache:scope:{tenant_id}:{model}:{sys_hash}` -- Tenant-isolated variant of the scope set.
- `cache:stats` -- Hash with `hits` and `misses` counters. No TTL (lifetime accumulation).
- `cache:lock:{prompt_hash}` -- String key for stampede guard. TTL 30s, created with SETNX.

**`gateway/routes/chat.py` -- integration points:**
- `_tee_stream_for_cache(gen, semantic_cache, model, messages, tenant_id, cache_isolation)` -- async generator wrapper. Yields each chunk to the client, parses `data: {...}` SSE lines to extract content deltas, buffers content. After generator exhaustion, assembles a `ChatCompletionResponse` with `Usage(0, 0, 0)` and calls `semantic_cache.store()`. Failures during store are caught and logged (never interrupt the stream).
- `_stream_cached_response(response)` -- async generator that converts a `ChatCompletionResponse` into 3 SSE chunks: `{"delta": {"role": "assistant"}}`, `{"delta": {"content": "..."}}`, `{"delta": {}, "finish_reason": "stop"}`, followed by `data: [DONE]`. This matches the OpenAI streaming format.

**`gateway/config.py` -- `TenantConfig.cache_isolation`:**
- `Literal["shared", "tenant"]`, default `"shared"`.
- Shared: all tenants share cache entries for the same model + system prompt. A response cached by tenant A is served to tenant B if the prompt is semantically similar. This maximizes hit rate.
- Tenant: cache entries are partitioned by tenant ID. No cross-tenant cache hits. Required for multi-tenant deployments where response content may be tenant-specific (e.g., system prompts referencing tenant data).

**`gateway/main.py` -- initialization:**
- `SemanticCache` is constructed in the lifespan after Redis connects. `CACHE_TTL` and `CACHE_SIMILARITY_THRESHOLD` are read from environment variables.
- If Redis is unavailable, `semantic_cache` is set to `None`. All cache code paths check for `None` before operating, so the gateway functions without caching.

**`gateway/routes/admin.py` -- admin endpoints:**
- `GET /admin/cache/stats` returns `{enabled, hits, misses, hit_rate, entries}`. Returns `{enabled: false}` if cache is `None`.
- `DELETE /admin/cache` calls `flush()` and returns `{status: "flushed", entries_deleted: N}`. Returns 503 if cache is `None`.

**Docker:**
- The `base` stage runs `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"` to pre-download the ~80MB model into the HuggingFace cache directory during image build.
- The `runtime` stage copies `/root/.cache/huggingface` from the base stage to `/home/appuser/.cache/huggingface` and `chown`s it. This avoids a runtime download on first request.
- The gateway container has a `mem_limit: 2G` in docker-compose to accommodate the model in memory (the model itself is ~80MB, but PyTorch + sentence-transformers runtime adds ~300-400MB overhead).
- Only CPU-only torch is used (no CUDA), keeping the image size significantly smaller than a GPU-enabled build.

### Key design decisions

1. **`all-MiniLM-L6-v2` as the embedding model.** This is a 22M-parameter sentence-transformer that produces 384-dimensional embeddings. It was chosen over larger models (e.g., `all-mpnet-base-v2` at 768 dimensions) for three reasons: (a) inference latency is ~5ms per embedding on CPU, which is acceptable for inline cache lookups; (b) 384 dimensions means half the Redis storage per entry compared to 768; (c) the model's quality on semantic textual similarity benchmarks (STSBenchmark: 0.8449) is sufficient for cache matching where the threshold is 0.95 -- at that threshold, only near-paraphrases match, so model precision matters less than recall. A larger model would provide marginal accuracy improvement at 2x the storage and latency cost.

2. **Threshold of 0.95 (configurable).** This is deliberately conservative. At 0.95 cosine similarity with normalized embeddings, only semantically near-identical prompts match. Lowering the threshold to 0.85 would increase hit rate but risks returning cached responses for genuinely different questions -- a false positive in an LLM cache means the user gets a wrong answer. The threshold is configurable via `CACHE_SIMILARITY_THRESHOLD` environment variable so operators can tune it for their workload. Workloads with highly repetitive queries (FAQ bots) can lower the threshold; workloads with nuanced queries (coding assistants) should keep it high.

3. **Scoping by model + system prompt hash, embedding only user messages.** The system prompt defines the persona, instructions, and constraints for the LLM. Two identical user questions with different system prompts should never share a cache entry -- the expected answer is different. Rather than embedding the system prompt (which would shift the embedding vector and reduce similarity for the actual question), the system prompt is hashed (SHA256, first 16 hex chars) and used as a partition key. This means the embedding captures only the user's question, while the scope key ensures system prompt isolation. This approach has a secondary benefit: if the system prompt is very long (e.g., 2000 tokens of instructions), it does not dilute the embedding of a short user question like "What is 2+2?".

4. **In-process embedding model, not a sidecar.** The embedding model runs inside the gateway process, not as a separate microservice. This eliminates network latency between the gateway and the embedding service (which would add 1-5ms per call on localhost, or more over a network). The tradeoff is that the gateway process consumes ~400MB more memory for the model + PyTorch runtime, and model loading adds ~2 seconds to the first cache operation. For a single-instance deployment (this project's scope), in-process is simpler and faster. A sidecar would be appropriate in a multi-instance deployment where sharing a single embedding service across N gateway instances reduces total memory by (N-1) * 400MB.

5. **float32 embeddings stored as JSON, not binary.** Each 384-dimensional embedding is stored as a JSON array of floats inside a Redis hash field. This uses approximately 3KB per entry (384 * ~8 bytes per float string). Binary encoding (struct.pack with float32) would reduce this to 1.5KB. JSON was chosen for debuggability -- operators can `HGET cache:entry:abc123 embedding` and inspect the values directly. At typical cache sizes (hundreds to low thousands of entries), the 2x storage overhead is negligible. Binary encoding would be appropriate at scale (millions of entries).

6. **O(n) scan within scope, not vector index.** Cache lookup iterates over all entries in a scope set and computes cosine similarity for each. This is O(n) in the number of cached entries per scope. For the expected cache size (tens to hundreds of entries per model+system_hash scope), this completes in under 1ms. A vector similarity index (RediSearch with HNSW) would provide O(log n) lookup but adds a dependency on the RediSearch module, complicates the Redis deployment, and provides no measurable benefit at small n. The design is intentionally simple for the current scale, with a clear upgrade path (see Production Gaps).

7. **Stampede guard for non-streaming only.** When 100 identical requests arrive simultaneously and the cache is cold, only one should compute the LLM response; the other 99 should wait for the cache to be populated. The SETNX lock achieves this for non-streaming requests. Streaming requests do not use the stampede guard because: (a) the cache is populated only after the stream completes, so waiters would need to block for the full streaming duration (potentially 30+ seconds); (b) the tee pattern means each streaming request can still serve the client while buffering for cache storage. The fail-open design means that if the lock times out (30s), the waiter proceeds to the backend rather than returning an error.

8. **Streaming cache hit emits synthetic SSE chunks.** When a streaming request hits the cache, the cached non-streaming response is converted into 3 SSE events (role, content, stop) plus `[DONE]`. This maintains the streaming contract -- the client always receives SSE events when `stream: true`, regardless of whether the response came from cache or a backend. An alternative would be to return the non-streaming JSON body, but this would break clients that expect `text/event-stream` content type and SSE parsing.

### Alternatives considered

1. **RediSearch vector index (HNSW) vs Python-side brute-force scan.** RediSearch provides a native `FT.SEARCH` command with HNSW vector indexing, offering O(log n) approximate nearest-neighbor search. This was rejected for three reasons: (a) RediSearch is a Redis module that must be installed separately -- the standard `redis:7-alpine` Docker image does not include it, requiring a custom image or a different base; (b) HNSW introduces approximate results, meaning it can miss the true nearest neighbor, which is unacceptable when the similarity threshold is 0.95 and a miss means re-computing a $0.01+ LLM call; (c) at the expected scale (tens to hundreds of entries per scope), brute-force scan over float arrays is faster than the overhead of maintaining an HNSW index. The upgrade path is clear: if a single scope accumulates thousands of entries, introduce RediSearch or an external vector store (Qdrant, Pinecone).

2. **Embedding sidecar service vs in-process model.** A sidecar (e.g., a FastAPI service running the embedding model, or a dedicated inference server like Triton) would decouple the embedding computation from the gateway process. Benefits: shared model across multiple gateway instances, independent scaling, GPU acceleration. Drawbacks: adds network latency (1-5ms per embedding call), operational complexity (another service to deploy, monitor, and health-check), and a failure mode (sidecar down = cache broken). For a single-instance deployment, in-process is strictly better. The refactor to a sidecar is straightforward: replace `self._model.encode(text)` with an HTTP call to the embedding service.

3. **Exact-match caching (prompt hash -> response).** The simplest caching strategy: SHA256 the full prompt text and use it as a Redis key. This was rejected because LLM workloads have extremely low exact-match rates. In testing with a set of 50 common questions, each rephrased 3 ways, exact-match caching produced a 0% hit rate (every phrasing produced a unique hash). Semantic caching with the 0.95 threshold matched 85% of rephrasings. Exact-match caching is appropriate for programmatic API callers that send identical prompts (e.g., automated pipelines), but not for human users.

4. **Per-request embedding cache (Python LRU).** Caching embeddings in an in-process LRU dict to avoid recomputing the same embedding for the same text. This was deferred because: (a) embedding computation is ~5ms, which is fast relative to the LLM call (1-10 seconds); (b) an LRU cache would only help when the exact same text appears multiple times in a short window, which is the exact scenario where the semantic cache already hits; (c) the LRU cache would consume gateway memory proportional to the number of unique prompts seen. This could be added as an L1 cache layer if embedding computation becomes a bottleneck at scale.

5. **Storing embeddings as binary blobs (struct.pack) vs JSON.** Binary storage would halve the per-entry Redis memory (1.5KB vs 3KB for 384 floats). JSON was chosen for debuggability and simplicity. The `json.dumps/json.loads` path is well-tested and handles edge cases (NaN, Inf) that `struct.pack` would silently corrupt. At the current scale, the storage difference is immaterial.

### Failure modes and edge cases

- **Redis down during cache lookup.** The `try/except` in the route handler catches any exception from `semantic_cache.lookup()`, logs `cache_lookup_failed`, sets `request.state.cache_status = "MISS"`, and continues to the backend. The gateway degrades gracefully to a no-cache proxy. No request is rejected due to a cache failure.
- **Redis down during cache store.** The `try/except` around `semantic_cache.store()` catches and logs `cache_store_failed`. The client has already received their response; the store failure is invisible to the client. The next similar query will be a cache miss and will attempt to store again.
- **Embedding model fails to load.** `_load_model()` imports `sentence_transformers` and loads the model. If the model file is missing or corrupted, this raises an exception on the first `compute_embedding()` call, which propagates up to the `lookup()` try/except. The gateway falls through to the backend call. Subsequent requests will retry the model load.
- **Scope set contains expired entry UUIDs.** Redis hash TTLs are per-key, not per-set-member. An entry hash can expire while its UUID is still in the scope set. `lookup()` handles this: when `hgetall(entry_key)` returns an empty dict, the UUID is removed from the scope set via `SREM`. This is lazy cleanup -- stale UUIDs accumulate until a lookup touches them.
- **False positive (similarity > 0.95 but semantically different).** At 0.95 threshold, false positives are rare but possible. Example: "What is the capital of France?" and "What is the capital of Finland?" have high lexical overlap and may produce similarity above 0.95. The consequence is a wrong cached answer. Mitigation: the threshold is configurable, and operators can raise it to 0.98+ for workloads where false positives are costly. The system prompt scoping also limits the blast radius -- a false positive only occurs within the same model + system prompt scope.
- **Stampede lock expires before backend responds.** The lock has a 30s TTL. If the LLM backend takes longer than 30s, the lock expires, and waiting requests fall through to their own backend calls. This is correct fail-open behavior: better to make N backend calls than to leave N-1 requests hanging indefinitely. The 30s TTL is intentionally generous (most LLM calls complete in 5-15s).
- **Stampede lock holder crashes.** If the gateway process crashes after acquiring the lock but before releasing it, the lock expires after 30s via the Redis TTL. Waiting requests will time out and proceed to the backend. No permanent lock contamination.
- **Empty user messages.** `_extract_user_text()` returns an empty string if there are no user messages. Both `lookup()` and `store()` short-circuit and return `None` / no-op if user text is empty. An embedding of empty text would be meaningless, so this is the correct behavior.
- **Cache isolation misconfiguration.** If tenant A has `cache_isolation: "shared"` and tenant B has `cache_isolation: "tenant"`, they use different scope key patterns. Tenant A's entries are in `cache:scope:{model}:{sys_hash}`, tenant B's are in `cache:scope:{tenant_b}:{model}:{sys_hash}`. There is no cross-contamination because the scope keys are structurally different. However, tenant B cannot benefit from tenant A's cache entries even if the prompts are identical. This is the intended behavior.
- **Very long user messages.** The embedding model has a 256-token input limit (truncated internally by sentence-transformers). Very long user messages are truncated before embedding, meaning two messages that differ only after the 256-token mark will produce identical embeddings and match. This is generally acceptable: the first 256 tokens of a prompt typically capture the core question.

### Observability

- **`X-Cache` response header.** Set to `HIT` or `MISS` on every chat completion response. Clients and load balancers can use this to monitor cache effectiveness. Set via `request.state.cache_status` in the route handler, read by the middleware.
- **`X-Cache-Similarity` response header.** Set on cache HITs to the cosine similarity value (4 decimal places, e.g., `0.9823`). Allows operators to monitor match quality and tune the threshold. Set via `request.state.cache_similarity`.
- **Structured log events:**
  - `cache_hit` -- emitted on every cache hit, includes model, tenant_id, similarity, streaming flag.
  - `cache_lookup_failed` -- emitted when the cache lookup raises an exception, includes error string.
  - `cache_store_failed` / `stream_cache_store_failed` -- emitted when the cache store fails after a backend response.
  - `stampede_guard_hit` -- emitted when a waiter receives a cached result from another request's computation.
  - `stampede_guard_timeout` -- emitted when a waiter times out and falls through to backend.
  - `stampede_guard_failed` -- emitted when the stampede guard mechanism itself fails.
  - `embedding_model_loaded` -- emitted once when the model is first loaded into memory.
  - `semantic_cache_initialized` -- emitted at startup with TTL and threshold values.
  - `cache_flushed` -- emitted when the admin flush endpoint is called, includes entries_deleted count.
- **`GET /admin/cache/stats` endpoint.** Returns `{enabled: true, hits: N, misses: M, hit_rate: 0.XXXX, entries: K}`. The `entries` count is computed via SCAN (not a counter) so it reflects the actual number of live cache entries, accounting for TTL expiry. Hit rate is computed as `hits / (hits + misses)`.
- **`DELETE /admin/cache` endpoint.** Returns `{status: "flushed", entries_deleted: N}`. Useful for cache invalidation after model updates or system prompt changes.

### Testing

**Unit tests -- semantic cache** (`tests/unit/test_semantic_cache.py`):
- `test_cosine_similarity_identical` -- identical vectors produce similarity 1.0.
- `test_cosine_similarity_orthogonal` -- orthogonal vectors produce similarity 0.0.
- `test_cosine_similarity_opposite` -- opposite vectors produce similarity -1.0.
- `test_cosine_similarity_zero_vector` -- zero vector against any vector returns 0.0 (no division by zero).
- `test_extract_user_text` -- concatenates user messages, ignores system and assistant messages.
- `test_extract_system_hash` -- SHA256 of system message content, truncated to 16 hex chars. Deterministic.
- `test_build_scope_key_shared` -- shared isolation produces `cache:scope:{model}:{sys_hash}`.
- `test_build_scope_key_tenant` -- tenant isolation produces `cache:scope:{tenant_id}:{model}:{sys_hash}`.
- `test_lookup_empty_scope` -- returns `(None, None)` when the scope set is empty.
- `test_lookup_hit` -- stores an entry, looks up with semantically similar text, returns the response and similarity.
- `test_lookup_miss_below_threshold` -- stores an entry, looks up with dissimilar text, returns `(None, None)`.
- `test_store_creates_entry_and_scope` -- verifies that `store()` creates both the entry hash and the scope set membership.
- `test_tenant_isolation` -- stores entries for two tenants with `cache_isolation="tenant"`, verifies each tenant only sees their own entries.
- `test_stampede_lock_acquire_release` -- acquires a lock, verifies second acquire fails, releases, verifies re-acquire succeeds.
- `test_stats_hit_miss_counters` -- calls `record_hit()` and `record_miss()`, verifies `get_stats()` returns correct counts and hit rate.
- `test_flush_deletes_all_cache_keys` -- stores entries, calls `flush()`, verifies all `cache:*` keys are deleted.

**Integration tests -- cache route** (`tests/integration/test_cache_integration.py`):
- `test_cache_miss_then_hit` -- first request returns `X-Cache: MISS`, second identical request returns `X-Cache: HIT` with `X-Cache-Similarity` header.
- `test_streaming_cache_miss_then_hit` -- first streaming request returns SSE events with `X-Cache: MISS`, second returns SSE events from cache with `X-Cache: HIT`.
- `test_stampede_guard` -- sends concurrent identical requests, verifies only one backend call is made and others receive cached results.
- `test_graceful_degradation_redis_down` -- patches `semantic_cache` to raise on lookup, verifies the request still succeeds via backend with `X-Cache: MISS`.
- `test_admin_cache_stats` -- verifies `GET /admin/cache/stats` returns correct structure with hits, misses, hit_rate, entries.
- `test_admin_cache_flush` -- verifies `DELETE /admin/cache` clears all cache entries and returns count.

**Total test count: 240** (unit + integration across all phases, including the cache-specific tests above).

### Production gaps

- **In-process embedding model does not scale horizontally.** Each gateway instance loads its own copy of the 80MB model into memory (~400MB with PyTorch runtime). With N gateway instances, this is N * 400MB of redundant memory. A production deployment would use a shared embedding service (sidecar or centralized) that all gateway instances call via gRPC or HTTP. The embedding service can be GPU-accelerated for higher throughput.
- **O(n) brute-force scan for similarity lookup.** Each cache lookup iterates over all entries in the scope set and computes cosine similarity. At n=100 entries per scope, this is ~0.5ms. At n=10,000, this becomes ~50ms, which is unacceptable for a cache lookup. Production would use a vector similarity index: RediSearch HNSW, Qdrant, Pinecone, or pgvector. The scope partitioning limits n in practice (each model+system_hash combination has its own set), but a popular model with a common system prompt could accumulate many entries.
- **No L1 in-process cache.** Every cache lookup hits Redis, even for repeated identical prompts within the same gateway instance. An in-process LRU cache (keyed by prompt hash) for recently-seen embeddings and results would eliminate Redis round-trips for hot queries. This is a standard two-tier cache pattern (L1 in-memory, L2 Redis) that would be trivial to add.
- **Single Redis instance, no replication.** The cache relies on a single Redis instance. If Redis goes down, the cache is lost (entries are not persisted to disk by default). Production would use Redis Sentinel or Redis Cluster for high availability, or an external managed Redis service (ElastiCache, Upstash). The fail-open design means Redis loss degrades to no-cache behavior, not an outage.
- **No cache eviction policy beyond TTL.** Entries expire after `CACHE_TTL` (default 1 hour) and are not evicted otherwise. There is no LRU eviction, no max-entries cap, and no memory limit enforcement. A long-running gateway with a low threshold and many unique prompts could accumulate unbounded cache entries until Redis runs out of memory. Production would set `maxmemory-policy allkeys-lru` on Redis and/or implement application-level eviction.
- **No cache warming or prefetch.** The cache starts cold on every deployment. There is no mechanism to pre-populate the cache with common queries or to persist cache state across restarts. A production system might export and import cache snapshots, or use a persistent vector store (Qdrant, pgvector) that survives restarts.
- **Streaming cache stores `Usage(0, 0, 0)`.** When a streaming response is cached, the token usage is unknown (SSE chunks do not include usage information for most providers). The cached response has zero token counts, which is inaccurate. A production system would estimate tokens from content length or use provider-specific `stream_options.include_usage`.

### Interview talking points

- **Why semantic caching over exact-match for LLM workloads.** LLM prompts are natural language -- users phrase the same question differently every time. Exact-match caching (hash the prompt, look up the hash) produces near-zero hit rates. Semantic caching converts prompts to dense vector embeddings and compares meaning via cosine similarity. In testing, exact-match caching hit 0% of rephrased questions; semantic caching at 0.95 threshold hit 85%. The tradeoff is compute cost (5ms embedding per lookup) and storage (3KB per entry), but both are negligible compared to the LLM call saved (1-10 seconds, $0.001-0.01 per call).
- **Scope partitioning via system prompt hash.** The system prompt defines the LLM's behavior. "What is 2+2?" asked of a math tutor vs. a sarcastic comedian should produce different answers. Rather than embedding the system prompt (which would shift the vector space and reduce matching accuracy for the actual question), the system prompt is SHA256-hashed and used as a partition key. Entries with different system prompts are in different Redis sets and never compared against each other. This is a separation of concerns: the embedding captures question semantics, the scope key captures context identity.
- **Stampede guard prevents thundering herd on cold cache.** When a popular question first appears (cache cold), N concurrent requests all miss the cache and all call the LLM backend. This wastes N-1 redundant backend calls. The SETNX lock ensures only one request computes the response; the other N-1 poll with exponential backoff and return the cached result once it is stored. The fail-open design (30s timeout, then proceed to backend) prevents the stampede guard from becoming a liveness hazard. This is the same pattern used by Nginx proxy_cache_lock and CDN stale-while-revalidate.

### Likely interview questions

**Q: "How do you handle the case where two prompts are semantically similar but should produce different responses?"**
**A:** This is the false positive problem. The primary defense is the high similarity threshold (0.95), which only matches near-paraphrases. The secondary defense is scope partitioning: different models, different system prompts, and different tenants (in tenant-isolated mode) are in separate scopes and never compared. For workloads where even 0.95 is too aggressive (e.g., "capital of France" vs "capital of Finland"), operators can raise the threshold to 0.98 or 0.99 via the `CACHE_SIMILARITY_THRESHOLD` environment variable. The tradeoff is hit rate vs correctness: a higher threshold means fewer cache hits but fewer false positives.

**Q: "Why not use a dedicated vector database like Pinecone or Qdrant instead of Redis?"**
**A:** For this project's scale (hundreds of cache entries per scope), a vector database is over-engineering. The brute-force scan over 100 entries with NumPy takes under 1ms. Redis is already in the architecture for rate limiting, so reusing it avoids adding another infrastructure dependency. A dedicated vector database would provide O(log n) approximate nearest-neighbor search via HNSW, which becomes necessary at thousands of entries per scope. The migration path is clear: replace the `lookup()` method's scan loop with a vector DB query, keeping the same embedding and scope key logic. The `SemanticCache` class encapsulates this behind the `lookup()`/`store()` interface, so the change is localized.

**Q: "How does the streaming cache work? Streams are inherently forward-only -- how do you cache and replay them?"**
**A:** Streaming uses a tee pattern: the `_tee_stream_for_cache()` async generator wraps the backend's SSE stream. As each chunk flows through, it is yielded to the client (preserving real-time delivery) and simultaneously parsed to extract content deltas. After the stream completes (the generator is exhausted), the buffered content is assembled into a standard `ChatCompletionResponse` and stored in Redis. On a subsequent cache hit for a streaming request, `_stream_cached_response()` converts the stored non-streaming response into 3 synthetic SSE chunks (role, content, stop) plus `[DONE]`, maintaining the streaming contract. The client cannot distinguish a cached streaming response from a live one.

**Q: "What happens under high concurrency -- is the cache thread-safe?"**
**A:** The gateway runs on a single asyncio event loop (uvicorn). There are no threads, so there are no thread-safety concerns in the traditional sense. Concurrency is cooperative (async/await), meaning operations interleave at `await` points but never execute simultaneously. The embedding model's `encode()` is a synchronous CPU-bound call that blocks the event loop for ~5ms -- acceptable for the current scale but would need to be moved to a thread pool (`asyncio.to_thread`) or an external service under high concurrency. Redis operations are async and non-blocking. The stampede guard uses Redis SETNX for distributed locking, which is safe across multiple gateway instances.

## Priority Queue with Backpressure

### Why this exists

LLM inference calls take 2--30 seconds per request. When all backend slots are occupied, the gateway must decide what to do with incoming requests. Without a queue, the only option is an immediate 503 rejection, which forces clients to implement their own retry logic -- retry storms, exponential backoff guessing, wasted network round-trips, and no fairness guarantees between tenants. A priority queue absorbs short bursts of overload by holding requests in-process until a slot opens, converting a hard rejection into a brief wait. For the client, waiting 5 seconds in a queue and getting a response is strictly better than receiving a 503, retrying 3 times with backoff, and getting the same response 15 seconds later. The queue also enables priority differentiation: high-priority tenants (e.g., production traffic) can jump ahead of low-priority tenants (e.g., batch jobs) without requiring separate backend pools.

Backpressure is the other half of the design. An unbounded queue is worse than no queue -- it accumulates latency debt until requests start timing out deep in the queue, wasting resources on work that will never be delivered. The queue enforces a hard depth limit (`max_queue_depth`, default 100) and a per-request timeout (`queue_timeout`, default 30s). When either is breached, the gateway rejects with a clear signal (503 with `Retry-After` or 504), telling the client that the system is genuinely overloaded rather than momentarily busy.

### How it works

The request flow through the priority queue is:

1. **Backend selection** completes normally -- the router picks a target backend and model.
2. **Slot check**: `PriorityQueueManager.acquire_slot(backend_name)` atomically increments the in-process concurrency counter for that backend. If the counter is below `max_concurrent`, the slot is granted immediately and the request proceeds to the backend call. No queue interaction occurs on this fast path.
3. **Queue entry** (slow path): If the counter has already reached `max_concurrent`, the request is enqueued. A unique `request_id` is generated, the tenant's priority and current timestamp are combined into a score (`priority * 1_000_000_000_000 + time.time()`), and the request is added to the Redis sorted set `queue:{model}` via `ZADD`.
4. **Wait**: The request awaits an `asyncio.Event` associated with its `request_id`. This is a zero-CPU wait -- the coroutine is suspended, the event loop serves other requests, and no polling occurs.
5. **Signal**: When any request on the same backend completes and releases its slot, `release_slot(backend_name)` calls `_signal_next_waiter(backend_name)`. This pops the lowest-score entry from the Redis sorted set (`ZPOPMIN`), looks up its `asyncio.Event`, and sets it. The waiting coroutine wakes up.
6. **Circuit breaker re-check**: After waking, the dequeued request does NOT blindly proceed to the original backend. It calls `get_open_backends()` and `find_backend_for_model()` again. If the original backend's circuit breaker has opened during the wait, the request is re-routed to any other available backend for the same model. This prevents a queue from draining into a known-bad backend.
7. **Slot acquire on dequeue**: The dequeued request acquires a slot on whatever backend it is routed to (which may differ from the original).
8. **Backend call**: The request proceeds to the LLM backend.
9. **Slot release**: On completion (success or failure), the slot is released in a `finally` block, which triggers signaling the next waiter.

For streaming requests, slot release cannot happen in a `finally` block around the backend call because the response is a generator -- the slot must remain held while chunks are being yielded to the client. A wrapper generator `_wrap_stream_with_slot_release()` handles this: it yields chunks from the inner stream, and when the inner stream is exhausted (or raises), it releases the slot in its own `finally` clause.

### Implementation

**Core module: `gateway/priority_queue.py`**

- `PriorityQueueManager` -- the single class that owns concurrency tracking, queue operations, and wait signaling.
  - `_concurrency: dict[str, int]` -- in-process counter per backend name. Incremented on acquire, decremented on release. Protected by `_lock: asyncio.Lock` to prevent interleaving at `await` points.
  - `_waiters: dict[str, asyncio.Event]` -- maps `request_id` to its wake-up event. Populated on enqueue, consumed on signal.
  - `_max_concurrent: dict[str, int]` -- per-backend concurrency limit, loaded from `BackendConfig.max_concurrent`.
  - `acquire_slot(backend_name) -> bool` -- under lock, checks if `_concurrency[backend_name] < _max_concurrent[backend_name]`. If yes, increments and returns `True`. If no, returns `False`.
  - `release_slot(backend_name)` -- under lock, decrements `_concurrency[backend_name]`, then calls `_signal_next_waiter(backend_name)`.
  - `enqueue(model, request_id, priority) -> float` -- computes score as `priority * 1_000_000_000_000 + time.time()`, calls `ZADD queue:{model} score request_id`, registers an `asyncio.Event` in `_waiters[request_id]`, returns the score.
  - `wait_for_slot(request_id, timeout) -> None` -- calls `asyncio.wait_for(event.wait(), timeout)`. On timeout, raises `QueueTimeoutError` and calls `remove_from_queue()` to clean up the Redis entry.
  - `_signal_next_waiter(backend_name)` -- resolves which models are served by this backend, calls `ZPOPMIN queue:{model}` on each, and sets the event for the popped `request_id`. If the popped entry has no corresponding event in `_waiters` (stale entry -- request already timed out or was cancelled), it retries up to 3 times to skip stale entries and find a live waiter.
  - `remove_from_queue(model, request_id)` -- calls `ZREM queue:{model} request_id` and deletes the event from `_waiters`.
  - `get_queue_depth(model) -> int` -- calls `ZCARD queue:{model}`.
- `QueueFullError` -- raised when `ZCARD queue:{model} >= max_queue_depth` before enqueue. The route handler converts this to a 503 with `Retry-After: 5`.
- `QueueTimeoutError` -- raised when `asyncio.wait_for` exceeds the configured timeout. The route handler converts this to a 504.

**Route integration: `gateway/routes/chat.py`**

- The priority queue sits in the request pipeline AFTER backend selection and BEFORE the backend call. It is inside the retry loop so that each retry attempt independently acquires and releases a slot.
- Non-streaming path: `acquire_slot()` is called at the top of the attempt. If it returns `False`, the request enqueues, waits, re-checks the circuit breaker, and acquires a slot on the (possibly different) backend. The slot is released in a `finally` block wrapping the backend call.
- Streaming path: The `finally` block cannot release the slot because the response generator has not been consumed yet. Instead, the raw backend stream is wrapped in `_wrap_stream_with_slot_release(stream, queue_manager, backend_name)`, which yields chunks through and releases the slot when the generator exits.
- `request.state.queue_wait_ms` is set to the elapsed queue wait time (or 0 if the fast path was taken), picked up by middleware to add the `X-Queue-Wait-Ms` response header.
- A `slot_backend` variable tracks which backend currently holds the slot, ensuring that `release_slot` is always called on the correct backend name even after re-routing.

**Lifespan and middleware: `gateway/main.py`**

- `PriorityQueueManager` is instantiated during the FastAPI lifespan handler, after Redis is connected. If Redis is unavailable, the queue manager is not created, and the gateway operates without queuing (immediate 503 on overload).
- Middleware reads `request.state.queue_wait_ms` (if present) and adds `X-Queue-Wait-Ms` to the response headers.
- Environment variables: `QUEUE_MAX_DEPTH` (default 100), `QUEUE_TIMEOUT` (default 30).

**Admin endpoint: `gateway/routes/admin.py`**

- `GET /admin/queue` returns a JSON object with per-backend concurrency status (`active`/`max` slots) and per-model queue depth.

**Configuration (pre-existing, now enforced)**

- `BackendConfig.max_concurrent` (default 10) defines the concurrency limit per backend. Ollama is configured at 5 (resource-constrained), mock backends at 10.
- `TenantConfig.priority` (default 1) defines the tenant's priority tier. Lower number = higher priority. Tenant "alpha" is priority 1, tenant "beta" is priority 2.

### Key design decisions

**In-process concurrency tracking instead of Redis.** The concurrency counter is a plain Python `dict[str, int]` protected by an `asyncio.Lock`, not a Redis counter. This is intentional: concurrency tracking must be consistent with the actual number of in-flight HTTP connections from this process. If the process crashes, all its in-flight connections die, and the counter resets to zero on restart -- exactly correct. A Redis counter would survive the crash with a stale value, requiring TTL-based expiration or heartbeats to self-correct. The tradeoff is that in a multi-instance deployment, each gateway instance tracks its own concurrency independently. This is acceptable because `max_concurrent` is a per-instance limit (each instance has its own connection pool to the backend), not a global limit. Global rate limiting is handled by the separate rate limiter component.

**Redis sorted set for the queue.** The queue needs priority ordering with FIFO tiebreaking and atomic pop. Redis sorted sets provide exactly this: `ZADD` inserts with a score, `ZPOPMIN` atomically removes and returns the lowest-score member. No polling, no race conditions on pop, and Redis handles the sort. The alternative -- an in-process `asyncio.PriorityQueue` -- would be simpler but would not survive process restarts and would not allow the admin endpoint to query queue depth without coupling to the queue's internals.

**Score formula: `priority * 1_000_000_000_000 + time.time()`.** The multiplier (1 trillion) ensures that priority tiers never overlap with timestamp values. `time.time()` returns a float with microsecond precision, typically around `1.7e9` (epoch seconds). Multiplying priority by `1e12` shifts it into a range that is always larger than any timestamp. Priority 1 produces scores starting at `1e12 + 1.7e9` (~1.0017 trillion), priority 2 produces scores starting at `2e12 + 1.7e9` (~2.0017 trillion). Within the same priority tier, lower timestamps sort first (FIFO). Across tiers, lower priority numbers sort first (higher priority). `ZPOPMIN` always returns the highest-priority, oldest request.

**Per-model queue instead of per-backend queue.** Queues are keyed by model name (`queue:{model}`), not by backend name. This is because multiple backends can serve the same model (e.g., two vLLM instances both serving `llama-70b`), and the request does not care which backend serves it. When a slot opens on any backend, `_signal_next_waiter` pops from the model queue, and the dequeued request runs backend selection again to find an available backend. A per-backend queue would pin requests to a specific backend, preventing re-routing even when another backend for the same model has capacity.

**asyncio.Event for signaling instead of polling.** Each queued request creates an `asyncio.Event` and awaits it. When a slot opens, the signaler sets the event, and the waiter wakes up in the same event loop tick. There is zero latency between "slot freed" and "next request wakes up" -- no 100ms polling interval, no wasted CPU cycles checking a flag. The `asyncio.Event` is the idiomatic asyncio primitive for this pattern: one coroutine waits, another coroutine signals, the event loop handles the scheduling.

**Circuit breaker re-check after dequeue.** A request might wait in the queue for 10--20 seconds. During that time, the backend it was originally targeting might have its circuit breaker trip open. Proceeding to a known-bad backend would waste the queue wait and produce an error. Instead, after dequeue, the request re-runs `get_open_backends()` and `find_backend_for_model()`. If the original backend is now open-circuited, the request is routed to any other backend that serves the same model. This decoupling between "which model" (fixed at enqueue time) and "which backend" (resolved at dequeue time) is why per-model queues are the right granularity.

### Alternatives considered

**Immediate 503 rejection (no queue).** The simplest approach: if all slots are occupied, return 503 immediately. This pushes all backpressure to the client, which must implement retry logic with exponential backoff. The problem is that LLM inference is slow (2--30s), so slots turn over on a seconds timescale. A client that retries after 1s will likely succeed. But without coordination, N clients all retry at the same time (thundering herd), and without priority, a batch job's retry is indistinguishable from a production request's retry. The queue solves both problems. Rejected in favor of queuing for any workload with bursty traffic or mixed priority tenants.

**In-process `asyncio.PriorityQueue`.** Simpler implementation, no Redis dependency for the queue itself. Rejected because: (a) queue state is lost on process restart, (b) the admin endpoint cannot inspect queue depth without reaching into the queue's internals, (c) in a multi-instance deployment, each instance's queue is isolated with no way to redistribute work if one instance is overloaded. Redis sorted sets provide persistence, atomic operations, and external visibility.

**Redis Streams (`XADD`/`XREADGROUP`).** Redis Streams provide consumer groups, acknowledgment, and at-least-once delivery -- features designed for durable message processing. Rejected because the priority queue pattern does not need consumer groups (there is one consumer per gateway instance), does not need acknowledgment (the request is in-process, not handed off), and critically, Redis Streams do not support priority ordering. Entries are strictly ordered by ID (timestamp). Implementing priority over Streams would require multiple streams (one per priority tier) with manual arbitration, adding complexity without benefit.

**Semaphore-based concurrency limiting (`asyncio.Semaphore`).** An `asyncio.Semaphore(max_concurrent)` per backend would handle the fast path (acquire immediately if under limit) and the slow path (await until released). Rejected because semaphores provide no priority ordering -- waiters are woken in FIFO order regardless of tenant priority. Adding priority to a semaphore requires a custom implementation that is essentially a priority queue. The explicit queue + event design provides priority, observability (queue depth is inspectable), and separation of concerns (concurrency tracking vs. queuing vs. signaling are distinct operations).

**Global queue across all models.** A single queue for all requests, regardless of model. Rejected because a slot opening on backend A (which serves model X) should not wake up a request for model Y. Per-model queues ensure that a signal is only delivered to a request that can actually use the freed capacity.

### Failure modes and edge cases

**Queue full (503).** When `ZCARD queue:{model} >= max_queue_depth`, `enqueue()` raises `QueueFullError` before adding the entry. The route handler returns 503 with `Retry-After: 5`. This is the intended backpressure signal: the queue itself is overloaded, and the client should wait before retrying. The `Retry-After` header gives clients a concrete backoff duration rather than leaving them to guess.

**Queue timeout (504).** When `asyncio.wait_for(event.wait(), timeout)` raises `asyncio.TimeoutError`, it is caught and re-raised as `QueueTimeoutError`. The route handler returns 504 (Gateway Timeout), removes the stale entry from the Redis sorted set via `remove_from_queue()`, and deletes the `asyncio.Event` from `_waiters`. The cleanup prevents the stale entry from being popped by `_signal_next_waiter` and waking a coroutine that has already returned an error.

**Stale entries in `_signal_next_waiter`.** A race exists: a queued request times out and cleans up its entry from `_waiters`, but the Redis `ZREM` has not executed yet (or another instance added a stale entry). When `_signal_next_waiter` pops this entry via `ZPOPMIN`, it finds no matching event in `_waiters`. Without mitigation, the freed slot would be wasted (no waiter is signaled). The fix: `_signal_next_waiter` retries up to 3 times, popping the next entry from the sorted set on each retry. This ensures that a live waiter is found if one exists, at the cost of at most 3 extra `ZPOPMIN` calls (each is O(log N) in Redis, sub-millisecond).

**Slot leak prevention.** A slot leak occurs when `acquire_slot` is called but `release_slot` is never called -- the counter increments but never decrements, and the backend appears permanently at capacity. Three mechanisms prevent this: (a) Non-streaming requests release slots in a `finally` block wrapping the backend call, ensuring release on both success and exception. (b) Streaming requests use `_wrap_stream_with_slot_release()`, a generator wrapper whose `finally` clause releases the slot when the generator is closed (by exhaustion, client disconnect, or exception). (c) A `slot_backend` variable in the route handler tracks which backend name the slot was acquired on, ensuring that `release_slot` is called on the correct backend even if re-routing occurred after dequeue.

**Redis unavailable at startup.** If Redis is unreachable during the lifespan handler, `PriorityQueueManager` is not instantiated. The gateway operates without queuing -- all requests take the fast path, and if backends are at capacity, clients receive immediate 503 errors. This is graceful degradation: the gateway remains functional, just without queue-based smoothing.

**Redis fails mid-operation.** If `ZADD` or `ZPOPMIN` raises a connection error, the exception propagates up to the route handler's error handling. For `ZADD` failures (enqueue), the request is not queued and falls through to a 503. For `ZPOPMIN` failures (signal), the slot is still released (the counter is decremented), but no waiter is woken. The next `release_slot` call will retry signaling. Worst case, a queued request waits until its timeout expires.

**Backend circuit breaker opens while requests are queued.** Addressed by the re-check after dequeue (see "How it works" step 6). If no backend is available for the model after dequeue, the request receives a 503 from the standard "no available backend" error path. The queue wait time is not wasted in the sense that the system did attempt to serve the request; it is reported via `X-Queue-Wait-Ms` for observability.

**Process crash with in-flight slots.** Because concurrency tracking is in-process, a process crash resets all counters to zero on restart. This is correct: the crashed process's HTTP connections are also dead, so the backend's actual concurrency from this process is zero. Redis queue entries from the crashed process remain (requests that were waiting). These entries are stale -- their `asyncio.Event` objects no longer exist. The `_signal_next_waiter` retry mechanism handles this by skipping entries with no matching waiter.

### Observability

**Response header: `X-Queue-Wait-Ms`.** Every response includes this header. A value of `0` means the request took the fast path (slot was immediately available). A non-zero value reports the milliseconds spent waiting in the queue. This is added by middleware reading `request.state.queue_wait_ms`, which is set by the route handler.

**Admin endpoint: `GET /admin/queue`.** Returns a JSON object with two sections: per-backend concurrency status (current `active` slots and `max` slots) and per-model queue depth (number of waiting requests). This endpoint is polled by monitoring dashboards and is useful for capacity planning -- if queue depth is consistently non-zero, more backend capacity is needed.

**Prometheus metrics** (via existing metrics middleware). The `X-Queue-Wait-Ms` value can be scraped as a histogram bucket for queue wait time distribution. Request duration already includes queue wait time, so an increase in P95 latency with stable backend latency indicates queuing overhead. The 503 and 504 status codes are already tracked as error rates, allowing alerts on sustained backpressure.

**Structured logging.** Queue events are logged at INFO level: enqueue (with model, priority, queue depth), dequeue (with wait time), timeout, queue full rejection, and re-route after circuit breaker check. Each log entry includes the `request_id` for correlation with request logs.

### Testing

273 total tests after this phase.

**Unit tests (`tests/unit/`):**
- Concurrency tracking: acquire increments counter, release decrements, acquire fails at max_concurrent, release below zero is clamped.
- Enqueue: score is computed correctly (`priority * 1e12 + timestamp`), entry appears in Redis sorted set, queue depth increments.
- Score design: priority 1 always sorts before priority 2 regardless of timestamp. Within same priority, earlier timestamp sorts first.
- Wait and timeout: `wait_for_slot` returns when event is set. `wait_for_slot` raises `QueueTimeoutError` after timeout expires.
- Signal and stale entries: `_signal_next_waiter` sets the correct event. Stale entries (no matching waiter) are skipped, up to 3 retries.
- Remove: `remove_from_queue` deletes from Redis sorted set and from `_waiters` dict.
- Queue depth: `get_queue_depth` returns `ZCARD` value.

**Integration tests (`tests/integration/`):**
- Slots available: request proceeds immediately, `X-Queue-Wait-Ms: 0`.
- Queue full (503): all slots occupied, queue at max depth, next request gets 503 with `Retry-After: 5`.
- Timeout (504): all slots occupied, queue timeout set to 0.1s, request gets 504 after timeout.
- Graceful degradation: Redis unavailable, queue manager not created, requests proceed without queuing.
- `X-Queue-Wait-Ms` header: request that waited in queue has non-zero header value.
- Slot release on failure: backend returns 500, slot is still released (counter decremented), next queued request is woken.
- Circuit opens while queued: backend circuit breaker trips during queue wait, dequeued request is re-routed to alternate backend.

### Production gaps

**No cross-instance queue coordination.** Each gateway instance tracks its own concurrency independently. In a multi-instance deployment behind a load balancer, the total concurrency on a backend is the sum of all instances' concurrency. If each instance allows 10 concurrent requests and there are 5 instances, the backend sees up to 50 concurrent requests. The `max_concurrent` setting must be divided by the expected instance count, or a distributed semaphore (e.g., Redis-based with Redlock) should replace the in-process counter.

**No queue persistence across restarts.** Redis stores the queue entries (sorted set), but the `asyncio.Event` objects are in-process. On restart, Redis has orphaned queue entries with no corresponding waiters. These entries are harmless (they will be skipped by `_signal_next_waiter`'s stale-entry retry) but waste a small amount of Redis memory until manually flushed or overwritten. A production system would add a startup sweep to `ZREM` entries older than `queue_timeout`.

**No priority preemption.** A priority-1 request that arrives while the queue has only priority-2 requests will be enqueued behind them in the sorted set. `ZPOPMIN` will correctly return the priority-1 request next (its score is lower). However, it does not preempt a priority-2 request that has already been dequeued and is in-flight. True preemption would require cancelling in-flight requests, which is destructive and complex. The current design provides priority ordering at the queue level, not at the execution level.

**Fixed timeout, no adaptive backpressure.** The queue timeout is a static configuration value (default 30s). A production system might adjust the timeout based on observed backend latency (e.g., if P99 backend latency is 10s, timeout should be at least 2x that). Adaptive backpressure could shed load earlier during sustained overload rather than waiting for the full timeout.

**No dead-letter queue.** Requests that time out are simply rejected with 504. A production system might log timed-out requests to a dead-letter store for later analysis or retry by an offline batch process.

### Interview talking points

- **Queue vs. immediate rejection for slow backends.** LLM inference is uniquely slow (2--30s per request), making queuing far more effective than in typical web services. A 5-second queue wait that produces a successful response is better than a 503 followed by 3 client retries totaling 15 seconds. The queue absorbs bursty overload while backpressure (depth limit + timeout) prevents unbounded latency accumulation.
- **Score design for priority + FIFO.** The formula `priority * 1e12 + timestamp` is a single-value encoding of a two-level sort key. The multiplier is chosen so that priority tiers never overlap with timestamp values (timestamps are ~1.7e9, well below 1e12). Redis `ZPOPMIN` returns the lowest score, which is always the highest-priority (lowest number), oldest (lowest timestamp) request. No secondary sort key, no custom comparator -- just a single float in a sorted set.
- **In-process concurrency + Redis queue: separation of concerns.** Concurrency tracking (how many connections this process has open) is inherently per-process state -- it must reset on crash, it must be consistent with actual HTTP connections, and it must be zero-latency. The queue (who is waiting next) benefits from persistence, atomic pop, and external visibility, so it lives in Redis. Mixing the two (e.g., Redis for both) would add latency to the hot path and create stale-counter problems on crash.
- **Zero-latency signaling with asyncio.Event.** No polling loops, no sleep intervals. When a slot is freed, the next waiter wakes up in the same event loop tick. This is the difference between asyncio cooperative scheduling (event-driven, zero waste) and polling-based designs (periodic checks, wasted cycles, added latency).
- **Stale entry handling as a robustness pattern.** The `_signal_next_waiter` retry mechanism (up to 3 pops to skip stale entries) is a practical concession to distributed state: the in-process `_waiters` dict and the Redis sorted set can diverge (timeout cleanup races, process crashes). Rather than adding complex distributed locking to keep them perfectly synchronized, the system tolerates divergence and self-corrects by skipping stale entries.

### Likely interview questions

**Q: "Why not use an asyncio.Semaphore instead of building a custom queue?"**
**A:** `asyncio.Semaphore` handles concurrency limiting but provides no priority ordering -- waiters are woken in FIFO order regardless of tenant priority. Adding priority to a semaphore effectively means building a custom priority queue on top of it, at which point the semaphore adds no value. The explicit design -- `dict` counter for concurrency, Redis sorted set for the queue, `asyncio.Event` for signaling -- separates concerns cleanly: each component does one thing and is independently testable and observable. A semaphore conflates concurrency limiting and wait ordering into a single opaque primitive.

**Q: "How do you prevent slot leaks?"**
**A:** Three mechanisms. First, non-streaming requests release slots in a `finally` block wrapping the backend HTTP call, so both success and exception paths release. Second, streaming requests use a generator wrapper (`_wrap_stream_with_slot_release`) whose `finally` clause releases the slot when the generator is closed by exhaustion, client disconnect, or exception. Third, a `slot_backend` variable tracks which backend name the slot was acquired on, handling the case where circuit breaker re-check after dequeue routes the request to a different backend -- `release_slot` is always called on the correct backend name. The combination of language-level finally semantics and explicit backend tracking makes slot leaks a code-level impossibility rather than a runtime hope.

**Q: "What happens if Redis goes down while requests are queued?"**
**A:** The in-process `_waiters` dict still holds the `asyncio.Event` objects, so already-queued requests continue waiting until their timeout expires. No new requests can be enqueued (the `ZADD` call will fail), so they fall through to immediate 503 rejection. When slots are released, `_signal_next_waiter` will fail on `ZPOPMIN` but the slot counter is still decremented (the counter is in-process, not in Redis). The worst case is that queued requests time out with 504 instead of being served. The system does not crash or deadlock -- it degrades to the "no queue" behavior. On Redis recovery, new requests can be enqueued again with no manual intervention.

**Q: "Why per-model queues instead of per-backend queues?"**
**A:** A model can be served by multiple backends (e.g., two vLLM instances both hosting `llama-70b`). If queues are per-backend and backend A is full while backend B has capacity, a request queued on backend A waits unnecessarily. Per-model queues decouple "what I need" (a model) from "where I get it" (a backend). When a slot opens on any backend serving that model, the next waiter is popped and re-runs backend selection, which routes it to whichever backend has capacity. This also interacts correctly with the circuit breaker: if backend A's circuit opens while a request is queued, the request is re-routed to backend B on dequeue rather than being sent to a known-bad destination.

## Observability Stack

### Why this exists

A gateway that proxies LLM requests across multiple backends, tenants, and models is operationally opaque without instrumentation. When latency spikes, the operator needs to know whether the bottleneck is a slow backend, a cache miss storm, a rate limiter rejecting legitimate traffic, or a circuit breaker stuck open. Without metrics, every incident investigation starts with grepping logs and guessing. Without dashboards, capacity planning is impossible -- you cannot tell whether you are approaching backend saturation, which tenants dominate token consumption, or whether the cache is actually reducing backend load.

Structured logging (structlog, already present from Phase 1) solves the debugging case: individual request traces with correlation IDs. But logs are the wrong tool for aggregate questions -- "what is the P99 latency over the last hour?" requires scanning millions of log lines. Prometheus metrics solve the aggregate case: pre-computed counters and histograms that can be queried, graphed, and alerted on in constant time regardless of traffic volume. Grafana turns those metrics into operator-facing dashboards that answer the most common operational questions without writing PromQL by hand.

The observability stack closes the loop: structlog for per-request debugging, Prometheus for aggregate system health, Grafana for visual operator interface.

### How it works

1. **Metric emission**: Each subsystem increments or observes Prometheus metric objects at the point where the event occurs. The HTTP middleware records request count and latency after `call_next` returns. The chat route handler records cache hits/misses, rate limit rejections, token consumption, and queue depth changes. The circuit breaker records state transitions. All metric objects are module-level globals in `gateway/observability/metrics.py`, imported directly by the subsystems that use them.

2. **Metric exposition**: `prometheus_client.make_asgi_app()` is mounted as a sub-application at `/metrics` in `gateway/main.py`. When Prometheus scrapes this endpoint, the ASGI app calls `generate_latest()` internally, serializes all registered metric families into the Prometheus text exposition format, and returns it with the correct `Content-Type` header (`text/plain; version=0.0.4; charset=utf-8`). No manual serialization code is needed.

3. **Metric collection**: The Prometheus container scrapes `gateway:8080/metrics/` every 5 seconds (job-level override of the 15-second global interval). Metrics are stored in Prometheus's local time-series database (TSDB) and queryable via PromQL.

4. **Visualization**: Grafana is provisioned declaratively. A YAML datasource definition points Grafana at the Prometheus container with a fixed UID (`prometheus`). A dashboard provider auto-loads JSON dashboard definitions from the provisioning directory on startup. Three dashboards provide progressively deeper views: Gateway Overview for system-wide health, Per-Backend Drilldown for individual backend investigation, and Per-Tenant Usage for tenant-level accounting.

5. **Label propagation**: The HTTP middleware sets `request.state.tenant_id` and `request.state.model_name` in the chat handler so that the middleware can attach these labels to request-level metrics (count, latency) after the handler returns. Backend name is resolved during routing and attached to active request and circuit breaker metrics.

### Implementation

**Metrics module** (`gateway/observability/metrics.py`): Eight Prometheus metric families defined as module-level globals using the standard `prometheus_client` library:

| Metric | Type | Labels | Purpose |
|--------|------|--------|---------|
| `gateway_request_total` | Counter | tenant, model, backend, status_code, method | Total request volume segmented by outcome |
| `gateway_request_duration_seconds` | Histogram | tenant, model, backend | Latency distribution with buckets from 0.05s to 60s |
| `gateway_cache_operations_total` | Counter | model, status (hit/miss) | Cache effectiveness per model |
| `gateway_rate_limit_rejections_total` | Counter | tenant, limit_type (rps/rpm/token_budget_daily) | Rate limiter activity by rejection reason |
| `gateway_circuit_breaker_state` | Gauge | backend | Current CB state: 0=CLOSED, 1=OPEN, 2=HALF_OPEN |
| `gateway_queue_depth` | Gauge | model | Current number of requests waiting in per-model queues |
| `gateway_tokens_consumed_total` | Counter | tenant, model, type (prompt/completion) | Token accounting for cost attribution |
| `gateway_active_requests` | Gauge | backend | In-flight concurrent requests per backend |

**Metric exposition** (`gateway/main.py`): The Prometheus ASGI app is mounted at `/metrics` via `app.mount("/metrics", make_asgi_app())`. Circuit breaker gauges are initialized to 0 (CLOSED) for all configured backends during the application lifespan startup, ensuring Grafana panels show a value from the first scrape rather than displaying "No data" until the first state transition.

**Subsystem instrumentation** (`gateway/routes/chat.py`): Cache operations are recorded immediately after `record_hit()` or `record_miss()` returns. Rate limit rejections are recorded just before raising the 429 `HTTPException`, ensuring the metric is only incremented for actual rejections. Token consumption is recorded for both `prompt` and `completion` types using the counts extracted from the backend response. Queue depth is incremented on enqueue and decremented on dequeue or timeout, keeping the gauge accurate even under error conditions. Active requests are incremented on slot acquisition and decremented on slot release, both wrapped in `finally` blocks to prevent gauge drift.

**Circuit breaker instrumentation** (`gateway/circuit_breaker.py`): The `_transition()` method sets `CIRCUIT_BREAKER_STATE.labels(backend).set(state_int)` using a lazy import with `try/except ImportError`. This keeps the circuit breaker module independently importable and testable without requiring `prometheus_client` as a hard dependency -- unit tests for circuit breaker logic do not need to install or mock the metrics library.

**Prometheus configuration** (`prometheus/prometheus.yml`): A single scrape job targeting `gateway:8080` with a 5-second scrape interval. The trailing slash on `/metrics/` is required because FastAPI's mounted sub-app redirects `/metrics` to `/metrics/`, and Prometheus does not follow redirects by default.

**Grafana provisioning** (`grafana/provisioning/`):
- `datasources/datasource.yml`: Defines the Prometheus datasource with `uid: prometheus`, `url: http://prometheus:9090`, and `isDefault: true`.
- `dashboards/dashboard.yml`: Configures the file-based dashboard provider to load JSON files from the provisioning directory.
- Three JSON dashboard files, each with a stable `uid` for cross-linking and API access:
  - **Gateway Overview** (`uid: gateway-overview`): 6 panels -- RPS by status code (stacked time series), Error Rate % (stat with threshold coloring), Active Backends count (stat), Cache Hit Ratio % (gauge), Queue Depth by model (time series), Latency P50/P95/P99 (time series with three queries).
  - **Per-Backend Drilldown** (`uid: per-backend-drilldown`): Template variable `$backend` populated from `gateway_request_total` label values. 5 panels -- Latency Percentiles (P50/P95/P99 using `histogram_quantile`), Error Rate %, Active Concurrent Requests, Circuit Breaker State (with value mappings: 0=CLOSED green, 1=OPEN red, 2=HALF_OPEN yellow), Request Volume.
  - **Per-Tenant Usage** (`uid: per-tenant-usage`): Template variable `$tenant` populated from `gateway_request_total` label values. 5 panels -- Request Volume by Model, Tokens Consumed by type (prompt vs completion stacked), Rate Limit Hits by limit_type, Top Models table (sorted by request count), Latency P95 by Model.

**Docker Compose** (`docker-compose.yaml`): Prometheus service (`prom/prometheus`, port 9090) with the config file bind-mounted. Grafana service (`grafana/grafana`, port 3000) with provisioning directories bind-mounted and anonymous access enabled via environment variables (`GF_AUTH_ANONYMOUS_ENABLED=true`, `GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer`).

### Key design decisions

**Prometheus + Grafana over commercial APM**: Prometheus is the de facto standard for cloud-native metrics. It is free, open-source, has native Kubernetes integration, and supports the pull-based model that works naturally with container orchestration (no agent installation, no push credentials). Grafana provides dashboarding with PromQL support. The combination avoids vendor lock-in and per-host licensing costs that commercial APM tools (Datadog, New Relic) impose. For a gateway that may run at high request volume, per-host or per-event pricing can become significant.

**`make_asgi_app()` over manual `generate_latest()` endpoint**: The ASGI app handles content negotiation, encoding, and the Prometheus-specific content type header correctly. A manual endpoint using `generate_latest()` requires the developer to set the content type to `text/plain; version=0.0.4; charset=utf-8` explicitly -- getting this wrong causes Prometheus to reject the scrape. The ASGI app also handles gzip encoding negotiation. There is no reason to reimplement what the library already provides.

**Module-level globals over dependency injection for metrics**: Prometheus metrics in Python are inherently global -- they register themselves in a global `CollectorRegistry` on construction. Wrapping them in a class and injecting them via FastAPI's `Depends()` adds indirection without changing the underlying behavior. Module-level globals match the library's design intent, are importable from any module with a single `from gateway.observability.metrics import REQUEST_COUNT`, and avoid the boilerplate of threading a metrics object through every function signature. This is the pattern used by the `prometheus_client` documentation and by production systems at scale.

**Histogram bucket selection (0.05s to 60s)**: The bucket boundaries `[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]` cover the full range of gateway response times. Cache hits return in 50-100ms. Simple completions with short prompts return in 1-5 seconds. Complex completions with long contexts can take 30-60 seconds. The default Prometheus histogram buckets (`[0.005, 0.01, 0.025, ...]`) are designed for internal microservice calls and would put 99% of LLM requests in the `+Inf` bucket, making percentile calculations useless. Custom buckets ensure that `histogram_quantile(0.95, ...)` produces meaningful values across the actual latency distribution.

**Lazy import for circuit breaker instrumentation**: The circuit breaker module uses `try: from gateway.observability.metrics import CIRCUIT_BREAKER_STATE` inside `_transition()` rather than a top-level import. This means the circuit breaker can be unit-tested in isolation without `prometheus_client` installed, and it can be reused in contexts (CLI tools, standalone scripts) where metrics are not relevant. The `try/except ImportError` pattern has near-zero runtime cost after the first successful import (Python caches the module), and the metrics module is always available in the gateway process.

**Dashboard provisioning as code over manual UI configuration**: JSON dashboard definitions in the provisioning directory are version-controlled, reproducible, and automatically loaded on container startup. Manual dashboard creation through the Grafana UI produces configuration that lives only in Grafana's SQLite database, is lost when the container is recreated, and cannot be code-reviewed or diff'd. Provisioned dashboards are read-only in the UI (marked with a provisioned badge), which prevents accidental modification.

**Anonymous Grafana access for E2E verification**: Setting `GF_AUTH_ANONYMOUS_ENABLED=true` with `GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer` allows automated E2E tests to verify dashboard provisioning by hitting the Grafana API without managing authentication tokens. This is appropriate for local development and CI. In production, anonymous access would be disabled and replaced with OAuth or LDAP integration.

**Bounded metric cardinality**: The label dimensions `tenant x model x backend x status_code x method` are all drawn from configuration, not user input. Tenants come from the tenant config file (bounded set). Models come from the backend registry (bounded set). Backends are defined in config (bounded set). Status codes are HTTP codes (small fixed set). Methods are HTTP methods (small fixed set). No label is derived from request body content, query parameters, or user-supplied strings. This prevents cardinality explosion -- a common Prometheus anti-pattern where unbounded labels cause memory exhaustion in both the application and Prometheus TSDB.

### Alternatives considered

**Statsd (push-based) over Prometheus (pull-based)**: Statsd sends metrics over UDP to an aggregation daemon, which forwards them to a backend (Graphite, Datadog). This works well in environments where services cannot expose HTTP endpoints (e.g., Lambda functions, short-lived batch jobs). The gateway is a long-running HTTP server, so pull-based scraping is natural. Push-based systems also require running an additional aggregation daemon and introduce UDP packet loss as a failure mode. Rejected in favor of Prometheus's simpler operational model for this use case.

**OpenTelemetry Metrics over prometheus_client**: OpenTelemetry provides a vendor-neutral metrics API that can export to Prometheus, OTLP, and other backends. The additional abstraction is valuable for organizations standardizing on OTel across polyglot services. For a single-language gateway with Prometheus as the only metrics backend, the `prometheus_client` library is simpler (no collector/exporter pipeline to configure), produces native Prometheus exposition format without translation, and avoids the OTel SDK's startup complexity. Migration to OTel later would involve changing metric object constructors and adding an exporter, not rewriting instrumentation callsites -- the metric names and labels would remain identical.

**Per-request metric objects (dependency injection) over module-level globals**: Creating metric objects per-request or injecting them via FastAPI dependencies would allow per-test isolation and avoid global state. However, Prometheus metrics are inherently global (the `CollectorRegistry` is a process-level singleton). Wrapping globals in a class does not change the semantics -- it only adds indirection. Test isolation is achieved by resetting the registry in test fixtures, not by avoiding globals.

**InfluxDB over Prometheus**: InfluxDB is a push-based time-series database with its own query language (Flux/InfluxQL). It handles high-cardinality data better than Prometheus and supports longer retention natively. However, it requires push-based instrumentation (different library), has a more complex operational footprint (clustering, retention policies, continuous queries), and Grafana's PromQL integration is more mature than its Flux integration. For a gateway with bounded cardinality, Prometheus's limitations are not a factor.

### Failure modes and edge cases

**Prometheus scrape failure**: If Prometheus cannot reach `gateway:8080/metrics/`, metrics stop updating but the gateway continues serving traffic unaffected. Grafana dashboards show stale data (the last successfully scraped values). No data is lost from the gateway's perspective -- metric objects continue accumulating in-process. When scraping resumes, the next scrape picks up the current counter/gauge values. Histograms may show a gap in the time series but counters are monotonic, so rate calculations recover after two successful scrapes.

**Metric label mismatch**: If a request fails before the chat handler sets `request.state.tenant_id` or `request.state.model_name` (e.g., a 404 on an unknown route), the middleware falls back to default label values (`unknown` tenant, `unknown` model). This prevents `KeyError` on label access and ensures every request is counted, even if it cannot be attributed to a specific tenant or model.

**High cardinality from misconfiguration**: If someone adds a label derived from user input (e.g., the full prompt text), the number of unique time series explodes, consuming memory in both the gateway process and Prometheus. The current design prevents this by drawing all labels from configuration-defined sets, but there is no runtime guard against a developer adding an unbounded label in a future change. Code review is the primary defense.

**Counter reset on gateway restart**: Prometheus counters are monotonically increasing within a process lifetime. When the gateway restarts, counters reset to zero. Prometheus handles this via its `rate()` and `increase()` functions, which detect counter resets and compute correct rates across restarts. However, short-lived gateway instances (frequent restarts) produce noisy rate calculations because each reset creates a discontinuity.

**Gauge drift from unbalanced inc/dec**: `QUEUE_DEPTH` and `ACTIVE_REQUESTS` use `.inc()` and `.dec()` calls that must be perfectly balanced. If an exception path skips a `.dec()`, the gauge drifts permanently upward. Both decrements are placed in `finally` blocks to prevent this, but a code change that adds a new exit path without a corresponding decrement would cause drift. The symptom is a gauge that monotonically increases and never returns to zero, visible on the Grafana dashboard as a steadily climbing line.

**Grafana provisioning failure**: If a dashboard JSON file has a syntax error, Grafana logs the error and skips that dashboard but starts successfully with the remaining dashboards. The broken dashboard simply does not appear in the UI. This is a startup-time failure -- once dashboards are loaded, they are served from Grafana's internal database and are not re-read from disk unless the container restarts.

### Observability

This section is itself the observability implementation, so the observability of the observability stack is inherently meta:

- **Prometheus self-metrics**: Prometheus exposes its own metrics at `localhost:9090/metrics`, including `prometheus_tsdb_head_series` (number of active time series -- useful for detecting cardinality issues), `prometheus_target_scrape_pool_sync_total` (scrape failures), and `up` (whether each scrape target is reachable).
- **Grafana health**: Grafana exposes `/api/health` which returns `{"database": "ok"}` when healthy. The provisioning log (`/var/log/grafana/grafana.log` inside the container) records which dashboards and datasources were loaded on startup.
- **Gateway /metrics endpoint**: The endpoint itself is observable -- its response time and size are indicators of metric collection health. A sudden increase in response size suggests cardinality growth.
- **Structured logs**: All metric emission points log at DEBUG level with structlog, so enabling DEBUG logging shows every metric increment alongside the request context. This is useful for verifying that instrumentation is firing correctly during development.

### Testing

**286 total tests (273 existing + 13 new observability tests).**

**Unit tests -- metrics endpoint**: Verify that `GET /metrics` returns HTTP 200 with `text/plain` content type containing Prometheus exposition format. Verify that all 8 metric families (`gateway_request_total`, `gateway_request_duration_seconds`, `gateway_cache_operations_total`, `gateway_rate_limit_rejections_total`, `gateway_circuit_breaker_state`, `gateway_queue_depth`, `gateway_tokens_consumed_total`, `gateway_active_requests`) appear in the response body. Verify that sending a request through the gateway increments the request counter and observes the latency histogram. Verify that cache hit/miss events increment the cache counter with the correct `status` label. Verify that `X-Request-ID` propagation continues to work with metrics middleware in the chain.

**Config validation tests**: Verify that `prometheus/prometheus.yml` is valid YAML with a `scrape_configs` entry containing a `gateway` job targeting port 8080. Verify that `grafana/provisioning/datasources/datasource.yml` defines a Prometheus datasource with `uid: prometheus`. Verify that all 3 dashboard JSON files parse as valid JSON, each containing `uid`, `title`, and a non-empty `panels` array.

**What is not tested**: Prometheus-to-Grafana data flow (would require a running Prometheus instance with scraped data and Grafana rendering panels -- this is an integration test better suited for a staging environment). PromQL query correctness (the queries in dashboard JSON are not executed in tests; they are validated by manual inspection and Grafana's query editor). Alert rule evaluation (no alerting rules exist yet).

### Production gaps

**No alerting**: There is no Alertmanager configuration. In production, alerts would be defined for: error rate exceeding threshold (e.g., 5xx rate > 5% for 5 minutes), P99 latency exceeding SLO (e.g., > 30s for 10 minutes), circuit breaker open for extended period (> 5 minutes), cache hit ratio dropping below baseline (e.g., < 50% for 15 minutes), queue depth growing unbounded. These alerts would route to PagerDuty, Slack, or email via Alertmanager's routing tree.

**No persistent TSDB storage**: Prometheus stores data on the container's ephemeral filesystem. A container restart loses all historical metrics. Production deployments would use persistent volumes for Prometheus data, or remote write to a long-term storage backend (Thanos, Cortex, Grafana Mimir) for retention beyond the default 15-day window.

**No multi-process metric aggregation**: The gateway runs as a single uvicorn process. If scaled to multiple workers (e.g., `uvicorn --workers 4`), each worker maintains its own `CollectorRegistry` in its own process memory. Prometheus scraping the `/metrics` endpoint hits one worker at random, seeing only that worker's counters. The `prometheus_client` library provides `multiprocess.MultiProcessCollector` with a shared directory for cross-process aggregation, but this is not configured. Production multi-worker deployments would need this, or each worker would expose metrics on a separate port.

**No metric retention policy**: There is no configuration for how long Prometheus retains data. The default is 15 days with a maximum block duration of 2 hours. For a production system, retention would be explicitly configured based on storage capacity and query needs (e.g., `--storage.tsdb.retention.time=90d` for 90-day retention).

**No metric federation**: In a multi-instance deployment (multiple gateway replicas behind a load balancer), each instance exposes its own `/metrics`. A single Prometheus instance would scrape all replicas. For large deployments, Prometheus federation or a multi-tenant Prometheus setup (Thanos, Cortex) would be needed to aggregate metrics across instances.

**No Grafana persistence or authentication**: Grafana uses an ephemeral SQLite database and anonymous access. Production would use a persistent PostgreSQL backend for Grafana's database, OAuth2/LDAP for authentication, and RBAC for dashboard access control. Dashboard provisioning as code would remain, but user-created dashboards and annotations would persist across restarts.

**No distributed tracing integration**: The metrics stack does not integrate with distributed tracing (Jaeger, Zipkin, Tempo). The `request_id` from structlog provides request-level correlation, but there are no trace spans for sub-operations (cache lookup, backend call, queue wait). Adding OpenTelemetry tracing would provide flame-graph visualization of request lifecycles.

### Interview talking points

- Chose pull-based Prometheus over push-based alternatives (Statsd, OTLP) because the gateway is a long-running HTTP server where pull is natural, avoids running an aggregation daemon, and eliminates UDP packet loss as a failure mode.
- Metric cardinality is bounded by design: all label values come from configuration (tenants, models, backends), never from user input. This prevents the classic Prometheus anti-pattern of unbounded labels causing memory exhaustion.
- Histogram bucket selection was deliberate: default Prometheus buckets (5ms-10s) are designed for microservice calls and would put 99% of LLM responses in the `+Inf` bucket, making percentile calculations meaningless. Custom buckets (50ms-60s) span the actual latency distribution from cache hits to slow completions.
- Dashboard provisioning as code ensures reproducibility, version control, and code review for operational tooling -- the same principles applied to application code are applied to dashboards. No manual clicking in the Grafana UI.
- Lazy import pattern for circuit breaker metrics keeps the CB module independently testable and reusable without coupling it to the observability stack. The `try/except ImportError` has near-zero runtime cost after first import.

### Likely interview questions

**Q: "Why not use OpenTelemetry for metrics instead of the Prometheus client directly?"**
**A:** OpenTelemetry provides a vendor-neutral abstraction that exports to multiple backends, which is valuable in polyglot organizations standardizing on a single observability API. For this gateway -- a single Python service with Prometheus as the only metrics backend -- the `prometheus_client` library is simpler. It produces native exposition format without an exporter pipeline, has no OTel SDK startup overhead, and avoids configuring the OTel collector/exporter chain. The metric names, labels, and instrumentation callsites would be identical under OTel; only the object constructors and export path change. Migration to OTel later is a mechanical refactor, not an architectural change.

**Q: "How do you handle metric cardinality in a multi-tenant system?"**
**A:** Every label dimension is drawn from a bounded, configuration-defined set. Tenants are defined in the tenant config file. Models are registered in the backend registry. Backends are enumerated in config. Status codes and HTTP methods are small fixed sets. The cross-product of these sets produces a predictable number of time series that can be calculated statically: `|tenants| x |models| x |backends| x |status_codes| x |methods|`. For a deployment with 10 tenants, 5 models, 3 backends, ~5 common status codes, and 2 methods, that is roughly 1,500 series for the highest-cardinality metric -- well within Prometheus's comfort zone of millions of series. The key discipline is never adding a label derived from request body content, user-supplied strings, or other unbounded inputs.

**Q: "What happens to your metrics when the gateway process restarts?"**
**A:** Prometheus counters are monotonically increasing within a process. On restart, they reset to zero. Prometheus's `rate()` and `increase()` functions detect counter resets and handle them correctly -- the first scrape after restart shows a counter value lower than the previous scrape, which Prometheus interprets as a reset rather than a decrease. Gauges (`ACTIVE_REQUESTS`, `QUEUE_DEPTH`, `CIRCUIT_BREAKER_STATE`) immediately reflect the new process's state, which starts at zero for request gauges and is initialized explicitly to CLOSED (0) for circuit breaker gauges during startup. Histograms reset like counters, so `histogram_quantile` calculations may be slightly inaccurate for the first few scrape intervals after restart. For very short-lived processes (frequent restarts), the Pushgateway is an alternative that persists the last pushed values, but it is not needed for a long-running gateway process.

**Q: "Why module-level globals for metrics instead of dependency injection?"**
**A:** Prometheus metrics in Python are inherently global. They register themselves in a process-level `CollectorRegistry` singleton on construction. Wrapping them in a class and injecting via FastAPI `Depends()` would add indirection without changing the underlying semantics -- the registry is still global, the metric objects are still singletons. Module-level globals match the library's design intent, are importable with a single line from any module, and avoid threading a metrics parameter through every function signature in the call chain. Test isolation is achieved by resetting the global registry in test fixtures, not by avoiding globals. This is the same pattern used in the `prometheus_client` documentation and in production Prometheus deployments.

**Q: "How would you add alerting to this setup?"**
**A:** Deploy Alertmanager alongside Prometheus and add alerting rules to `prometheus/alert_rules.yml`. Rules would include: error rate (`rate(gateway_request_total{status_code=~"5.."}[5m]) / rate(gateway_request_total[5m]) > 0.05`), latency SLO breach (`histogram_quantile(0.99, rate(gateway_request_duration_seconds_bucket[5m])) > 30`), circuit breaker stuck open (`gateway_circuit_breaker_state == 1` for > 5 minutes), and cache degradation (`rate(gateway_cache_operations_total{status="hit"}[15m]) / rate(gateway_cache_operations_total[15m]) < 0.5`). Alertmanager would route alerts based on severity: critical alerts (error rate, CB stuck open) to PagerDuty, warning alerts (latency, cache ratio) to Slack. The key design principle is alerting on symptoms (error rate, latency) rather than causes (CPU usage, memory) -- symptoms tell you users are affected, causes require interpretation.

## Request Journaling

### Why this exists

Without request journaling, the gateway is a black box after the fact. Prometheus metrics tell you aggregate rates and latencies, but they cannot answer per-request questions: "What happened to request X?", "Which tenant is generating the most traffic right now?", "How many requests are currently in flight?" Logs can answer some of these, but they are unstructured, scattered across stdout, and expensive to query in real time. The gateway needs a lightweight, queryable record of every request's lifecycle -- from arrival through completion -- without adding meaningful latency to the hot path.

A journal also solves a concrete operational gap: in-flight visibility. During graceful shutdown, the gateway must know how many requests are still being processed. Without an explicit tracking mechanism, the only option is scanning logs or guessing. A Redis SET provides O(1) cardinality checks for in-flight requests, enabling the drain pattern that makes zero-downtime deployments possible.

Finally, the journal wrapper around streaming responses closes a metrics gap that existed since the streaming implementation. Before journaling, streaming responses bypassed token counting and rate limiter budget recording because tokens are only known after the full stream completes. The journal's stream wrapper buffers content, counts tokens via `token_counting.py`, records the journal completion entry, updates `TOKENS_CONSUMED` metrics, and deducts rate limiter budget -- all in one place at stream end.

### How it works

1. A request arrives at `POST /v1/chat/completions`. After authentication and tenant resolution, but before the cache check, the route handler calls `journal.record_request()`.
2. `record_request()` performs two Redis operations atomically from the application's perspective: `XADD` appends a `phase="request"` entry to the `journal:requests` stream with fields (`request_id`, `tenant_id`, `model`, `prompt_hash`, `timestamp`), and `SADD` adds the `request_id` to the `journal:inflight` set.
3. The `prompt_hash` is a SHA-256 digest computed only over user message content. This provides request correlation and deduplication detection without ever storing the actual prompt text in Redis.
4. The request proceeds through caching, routing, and backend execution as normal.
5. On completion -- whether from a cache hit, a successful backend response, a backend error, or a streaming response finishing -- `journal.record_completion()` is called. This performs `XADD` with `phase="completion"` fields (`status`, `latency_ms`, `backend`, `cache_hit`, `tokens_prompt`, `tokens_completion`) and `SREM` to remove the `request_id` from the in-flight set.
6. For streaming responses, `_wrap_stream_with_journal()` acts as the outermost async generator wrapper. It yields chunks transparently, buffers the accumulated content, and on stream exhaustion calls the completion recording with final token counts.
7. The stream is trimmed to approximately 100,000 entries via `XADD ... MAXLEN ~ 100000`. The `~` makes trimming approximate -- Redis may keep slightly more entries to avoid restructuring the underlying radix tree on every write.

### Implementation

- **`gateway/journal.py`**: `RequestJournal` class holding two key constants -- `STREAM_KEY = "journal:requests"` for the append-only event log and `INFLIGHT_KEY = "journal:inflight"` for the in-flight tracking set. Core methods: `record_request()` (XADD + SADD), `record_completion()` (XADD + SREM), `get_stats()` (XLEN + SCARD + XINFO for throughput calculation), `query()` (XREVRANGE with optional tenant filtering and request_id grouping). `compute_prompt_hash()` is a standalone function that extracts user-role messages, concatenates their content, and returns the SHA-256 hex digest.
- **`gateway/routes/chat.py`**: Journal integration at two points in the handler. Entry recording happens early (after auth, before cache). Completion recording happens at four exit points: cache hits (backend="cache", cache_hit=True), non-streaming success, non-streaming errors, and streaming via the wrapper function. `_wrap_stream_with_journal()` is the outermost streaming wrapper -- it wraps the existing streaming generator, buffers content, counts tokens at stream end, records journal completion, records `TOKENS_CONSUMED` Prometheus metrics, and deducts rate limiter token budget.
- **`gateway/routes/admin.py`**: Two admin endpoints. `GET /admin/journal/stats` returns `{total, inflight, entries_per_min}`. `GET /admin/journal?tenant=<id>&last=<n>` returns filtered, grouped entries with `last` capped at 100.
- **`gateway/routes/health.py`**: `GET /ready` readiness probe returns 200 with `healthy_backends` count when at least one backend has its circuit breaker in the CLOSED state. Returns 503 when all backends are OPEN or the system is not yet initialized. Distinct from `GET /health` (liveness), which always returns 200.
- **`gateway/main.py`**: Graceful shutdown machinery. A SIGTERM handler sets `app.state.shutting_down = True`. Middleware intercepts all requests during shutdown and rejects non-operational paths (anything other than `/health`, `/ready`, `/metrics`) with 503 and `Retry-After: 5`. An `asyncio.Lock`-guarded counter tracks in-flight requests, and an `asyncio.Event` (`inflight_zero`) signals when the count reaches zero. The lifespan shutdown sequence waits up to 10 seconds for the event before forcibly closing connections.
- **`Makefile` + `scripts/`**: `make seed` runs `scripts/seed.sh`, which fires 50 concurrent requests against `mock-gpt-markdown` and completes in under 30 seconds. `make status` runs `scripts/status.sh`, which health-checks the gateway, Prometheus, and Grafana.

### Key design decisions

**"Journal" not "WAL"**: The name is deliberate. A write-ahead log (WAL) implies durability guarantees -- data is persisted before the operation it protects. This journal provides neither durability nor recovery semantics. Redis Streams are in-memory, bounded by `MAXLEN`, and lost entirely on Redis restart. Calling it a journal sets correct expectations: it is an operational convenience for recent request visibility, not an audit log or recovery mechanism.

**Prompt hash for privacy**: The journal never stores raw prompt content. `compute_prompt_hash()` extracts only user-role messages, concatenates them, and stores the SHA-256 digest. This allows request correlation (same hash means same prompt content, useful for deduplication analysis) without creating a searchable log of user inputs. In a production deployment handling sensitive data, even storing a hash might require consideration (rainbow table attacks on known prompts), but for an operational journal with bounded retention, the hash strikes the right balance between utility and privacy.

**In-flight tracking via Redis SET**: The alternative is scanning the stream for entries that have a `phase="request"` record but no corresponding `phase="completion"`. This is O(n) in stream length and requires client-side joins. A Redis SET gives O(1) `SCARD` for the count and O(1) `SISMEMBER` for checking individual requests. The cost is two extra Redis commands per request (SADD + SREM), which are sub-millisecond operations that are negligible compared to LLM backend latency.

**Approximate trimming (`MAXLEN ~`)**: Exact `MAXLEN` forces Redis to restructure the radix tree backing the stream on every write that exceeds the limit. Approximate trimming (the `~` prefix) lets Redis defer restructuring until a full macro-node can be removed, which is significantly cheaper. The trade-off is that the stream may temporarily hold slightly more than 100,000 entries, but memory usage remains bounded and predictable.

**Journal never breaks requests**: Every Redis operation in the journal module is wrapped in try/except. If Redis is down, slow, or returns an error, the journal silently fails and the request proceeds normally. The journal is an observability sidecar, not a critical-path dependency. This is the same philosophy applied to Prometheus metrics -- instrumentation must never degrade the thing it instruments.

### Alternatives considered

**PostgreSQL or SQLite for the journal**: Provides true durability and SQL queryability, but introduces a write to a durable store on every request's hot path. Redis Streams give the append-only semantics needed without disk I/O, and the data is ephemeral by design -- there is no recovery scenario where replaying old journal entries is valuable. A relational store would be appropriate for a billing audit log, not for operational request tracing.

**Structured log aggregation (ELK/Loki) instead of a journal**: Logs already capture request lifecycle events, and a log aggregation stack can make them queryable. However, log pipelines introduce latency (seconds to minutes for indexing), require additional infrastructure, and cannot provide real-time in-flight counts. The journal provides sub-millisecond writes and instant reads from the same Redis instance the gateway already depends on. In production, both would coexist: the journal for real-time operational queries, log aggregation for historical analysis.

**In-process in-flight counter instead of Redis SET**: An `asyncio`-guarded counter in the gateway process would avoid Redis round-trips for in-flight tracking. This works for a single process but breaks with multiple gateway replicas -- each process only knows its own in-flight count. The Redis SET provides cluster-wide in-flight visibility across all gateway instances. Even for the single-replica case, putting in-flight state in Redis means the admin endpoint can report it without reaching into process internals.

**Storing full prompts with encryption**: Would provide full replay capability and deeper debugging. Rejected because it creates a significant liability -- encrypted prompts are still stored prompts, subject to data retention policies, key management complexity, and breach risk. The SHA-256 hash provides enough correlation for operational use cases without the compliance burden.

### Failure modes and edge cases

- **Redis unavailable**: All journal operations silently fail. Requests proceed normally, but admin endpoints return empty or stale data. The in-flight SET may become inconsistent (entries added before Redis died are never removed), but this is self-correcting on the next Redis restart since the SET is lost along with the stale entries.
- **Completion never recorded**: If the gateway process crashes mid-request, the in-flight SET retains the request_id permanently (until Redis restarts). A production deployment would need a reaper process that periodically scans the in-flight set and removes entries older than a timeout threshold. This is a known production gap.
- **Stream trimming during query**: `XREVRANGE` may miss entries that are trimmed between the cursor read and the next iteration. This is acceptable for an operational tool -- the journal is approximate, not transactional.
- **High-cardinality tenant filtering**: `query()` filters by tenant on the client side after reading from the stream. With many tenants and high throughput, this means reading more entries than needed. A production system could use per-tenant streams (`journal:requests:{tenant_id}`) at the cost of more complex trimming and stats aggregation.
- **Streaming response errors mid-stream**: If the upstream backend disconnects during streaming, the journal wrapper catches the exception, records a completion with `status="error"`, and re-raises. Tokens counted up to the disconnection point are recorded, giving partial but useful data.
- **Graceful shutdown timeout exceeded**: If in-flight requests do not complete within 10 seconds, the gateway closes connections forcibly. Docker's `stop_grace_period: 15s` provides a 5-second buffer after the application's 10-second drain window, ensuring the container runtime does not SIGKILL the process before the application finishes its own shutdown sequence.

### Observability

- **Admin endpoints**: `GET /admin/journal/stats` exposes total entries, in-flight count, and entries-per-minute throughput. `GET /admin/journal?tenant=<id>&last=<n>` provides filtered, grouped request history. Both are useful for dashboards and ad-hoc debugging.
- **Readiness probe**: `GET /ready` reports whether the gateway can accept traffic (at least one healthy backend). Kubernetes uses this to remove unhealthy pods from service endpoints without killing them. Distinct from the liveness probe (`GET /health`), which always returns 200 and tells Kubernetes the process is alive.
- **Streaming metrics fix**: The journal's stream wrapper closes the observability gap for streaming responses. Before this phase, streaming requests had no token counts in Prometheus metrics and no rate limiter budget deduction. The wrapper records `TOKENS_CONSUMED` and updates the rate limiter, making streaming and non-streaming responses metrically equivalent.
- **Structured logging**: Journal operations log at debug level on success and warning level on failure, with `request_id` correlation via structlog context variables.

### Testing

311 total tests across the suite after this phase.

- **Unit tests (journal module)**: `record_request()` and `record_completion()` verify correct Redis commands (XADD fields, SADD/SREM membership). Exception-swallowing tests confirm that Redis errors do not propagate. `get_stats()` tests verify XLEN + SCARD + XINFO aggregation. `query()` tests cover request_id grouping, tenant filtering, and the `last` parameter cap. `compute_prompt_hash()` tests verify SHA-256 output and that only user messages are included.
- **Integration tests (route layer)**: Journal recording in the chat completion handler verifies that request and completion entries appear in the stream after a successful request. Graceful degradation tests confirm that requests succeed when the journal is `None` (Redis unavailable at startup). Cache hit journaling tests verify that cached responses are recorded with `backend="cache"` and `cache_hit=True`.
- **Health and shutdown tests**: Readiness probe tests cover the healthy case (at least one CLOSED circuit breaker), the unhealthy case (all OPEN), and the uninitialized case. Shutdown tests verify that the middleware rejects non-operational requests with 503 + `Retry-After` during shutdown, while `/health`, `/ready`, and `/metrics` remain accessible.

### Production gaps

- **No in-flight reaper**: If a gateway instance crashes without recording completions, orphaned entries persist in the in-flight SET until Redis restarts. A production deployment needs a background task or sidecar that periodically scans the set and evicts entries older than a configurable timeout (e.g., 5 minutes).
- **No journal persistence**: Redis Streams are in-memory. A Redis restart loses the entire journal history. Production systems needing durable audit trails would layer a Kafka consumer or Change Data Capture pipeline on top of Redis, sinking journal entries to a durable store (S3, PostgreSQL) asynchronously.
- **Single-stream scaling**: All tenants share one stream. At very high throughput (tens of thousands of requests per second), the single stream becomes a serialization bottleneck. Partitioning by tenant or by time window (one stream per hour) would distribute the write load, at the cost of more complex querying and trimming.
- **No authentication on admin endpoints**: The `/admin/journal/*` endpoints are unauthenticated. Production deployments should gate them behind internal network policies, an admin API key, or mTLS.
- **Approximate drain timing**: The 10-second drain timeout is hardcoded. Long-running LLM completions (some models take 60+ seconds) may be killed during shutdown. A production configuration would make the drain timeout configurable and potentially extend `stop_grace_period` to match the longest expected request duration.
- **Client-side tenant filtering**: The `query()` method reads entries from the stream and filters by tenant in Python. This is O(n) in the number of entries read. Per-tenant streams or a Redis secondary index would push filtering to the data layer.

### Interview talking points

- The name "journal" was chosen deliberately over "WAL" or "audit log" because it communicates the actual guarantees: bounded, in-memory, ephemeral, and not suitable for recovery. Naming infrastructure components accurately prevents teams from building on assumptions the component does not fulfill.
- The prompt hash pattern demonstrates how to balance operational observability with data privacy. SHA-256 of user messages provides correlation and deduplication analysis without creating a searchable log of user inputs, sidestepping data retention and compliance concerns.
- The streaming journal wrapper solves three problems in one place: journal completion recording, Prometheus token metrics, and rate limiter budget deduction. Before this, streaming responses were a metrics blind spot -- the gateway could not account for tokens consumed during streaming. This is a good example of how an operational feature (journaling) can expose and fix existing gaps in the system.
- The graceful shutdown pattern (SIGTERM -> reject new requests -> drain in-flight -> close connections) is standard Kubernetes practice, but the implementation details matter: using `asyncio.Event` for drain signaling avoids polling, the 10s/15s timeout layering ensures the application always shuts down before the container runtime forces a SIGKILL, and allowing health/ready/metrics through during shutdown lets the orchestrator observe the draining process.
- The in-flight tracking design (Redis SET vs stream scanning) illustrates a recurring systems design trade-off: paying a small per-request cost (two extra Redis commands) to avoid an expensive periodic operation (O(n) stream scan). The SET provides O(1) reads for the admin endpoint and the drain check, which are the actual access patterns.

### Likely interview questions

**Q: "Why use Redis Streams instead of a simple list or sorted set?"**
**A:** Redis Streams provide three properties that lists and sorted sets do not: automatic ID generation (monotonically increasing timestamps that serve as natural ordering), consumer group support for future multi-consumer scenarios, and `XINFO` introspection for throughput calculation. A list would require manual timestamp management and has no built-in trimming by count. A sorted set would work for bounded retention (ZREMRANGEBYRANK), but it does not support range queries by time as naturally as XREVRANGE. Streams are purpose-built for append-only event logs, which is exactly what the journal is.

**Q: "How do you handle the case where a request is recorded but the completion is never written?"**
**A:** This is a known gap. The in-flight SET retains orphaned request IDs until Redis restarts. In production, a reaper process would periodically scan the set, check each entry's age against a timeout threshold (e.g., 5 minutes), and remove stale entries. The reaper could also emit a metric (`journal_orphaned_requests`) to alert on crash frequency. The journal's completion recording is wrapped in the same try/except as all journal operations, so the more common failure mode is not a gateway crash but a Redis transient error on the SREM call -- which is self-correcting because the next Redis restart clears the SET.

**Q: "Why not use a database transaction to ensure request and completion are always paired?"**
**A:** Transactional guarantees would require either a durable store (adding disk I/O to the hot path) or Redis transactions (MULTI/EXEC, which do not provide rollback semantics -- they only guarantee atomicity of execution, not conditional logic). More fundamentally, the journal does not need pairing guarantees. An orphaned request entry is a minor inconvenience (slightly inflated in-flight count), not a data integrity violation. The system is designed to tolerate inconsistency in the journal because the journal is not a source of truth -- it is an operational convenience. Adding transactional overhead to guarantee consistency in a component that tolerates inconsistency is the wrong trade-off.

**Q: "How does the readiness probe differ from the liveness probe, and why do you need both?"**
**A:** The liveness probe (`GET /health`, always returns 200) tells the orchestrator the process is running and not deadlocked. If liveness fails, Kubernetes restarts the container. The readiness probe (`GET /ready`, returns 200 only when at least one backend circuit breaker is CLOSED) tells the orchestrator whether the pod can serve traffic. If readiness fails, Kubernetes removes the pod from the Service endpoints but does not restart it -- the pod stays alive, allowing circuit breakers time to recover and the pod to re-enter the ready state. Without this distinction, a gateway with temporarily unavailable backends would be killed and restarted in a loop, amplifying the outage instead of riding it out.

**Q: "What happens during a rolling deployment with this graceful shutdown design?"**
**A:** Kubernetes sends SIGTERM to the old pod. The gateway sets `shutting_down = True` and the middleware begins rejecting new requests with 503 + `Retry-After: 5`. Simultaneously, Kubernetes removes the pod from Service endpoints, so the load balancer stops routing new traffic to it. In-flight requests continue processing. The gateway waits up to 10 seconds for all in-flight requests to complete (tracked via the asyncio counter and Event). If they finish in time, the gateway closes HTTP clients and Redis connections cleanly. If not, connections are closed forcibly, and Docker's `stop_grace_period: 15s` ensures the container runtime waits an additional 5 seconds before sending SIGKILL. The new pod's readiness probe must pass before Kubernetes adds it to Service endpoints, ensuring no traffic is routed until at least one backend is healthy.

## Intelligent Routing

### Why this exists

The previous phase introduced consistent hashing for backend selection. Consistent hashing optimizes for cache locality -- the same tenant+model key always lands on the same backend, maximizing semantic cache hit rates. But cache locality is not always the right objective. Some workloads are latency-sensitive: a real-time chatbot needs the fastest backend, not the one with the best cache affinity. Other workloads are cost-sensitive: a batch processing pipeline should route to the cheapest backend regardless of cache behavior. And for latency-critical requests, even the fastest backend can occasionally produce a slow response (tail latency), so there is no way to hedge against that outlier without sending the request to multiple backends simultaneously.

Without intelligent routing, the gateway is a one-strategy system. Every model gets consistent hashing, and operators have no knob to tune routing behavior per model or per workload. This phase introduces a pluggable strategy system with three implementations (consistent hash, latency-aware, cost-aware), per-model configuration, a rolling-window latency tracker, and hedge requests for tail latency reduction.

### How it works

1. The `routing` section in `config/backends.yaml` declares a per-model routing configuration. Each entry specifies a `strategy` (one of `consistent_hash`, `latency_aware`, `cost_aware`) and a `hedge_enabled` boolean. Models without an explicit entry default to `consistent_hash` with hedging disabled.

2. At startup, `Registry.__init__()` iterates over the model-to-backends mapping and constructs a `RoutingStrategy` implementation for each model based on its `ModelRoutingConfig`. For `latency_aware`, it passes the shared `LatencyTracker` instance. For `cost_aware`, it builds a `{backend_name: cost_per_1k_tokens}` dict from `BackendConfig`. For `consistent_hash` (or any unrecognized strategy), it wraps the existing `ConsistentHashRing` in a `ConsistentHashStrategy` adapter.

3. When a request arrives at `POST /v1/chat/completions`, the route handler calls `registry.find_backend_for_model(model, routing_key, exclude)`. This method looks up the per-model strategy and delegates to its `select()` method. The `exclude` set contains circuit-broken backends. The `routing_key` is `{tenant_id}:{model}` for consistent hash affinity.

4. The selected backend handles the request. After the response completes, the route handler records the observed latency via `latency_tracker.record(backend_name, model, duration_ms)`. For streaming responses, latency is recorded when the stream finishes via the `_wrap_stream_with_latency` generator wrapper. This data feeds back into `LatencyAwareStrategy` for subsequent requests.

5. For hedge requests: the client sends `X-Hedge: true`. The route handler checks whether the model has `hedge_enabled: true` in its routing config. If both conditions are met and the request is non-streaming, the handler selects two distinct backends (first via normal strategy, second by excluding the first), creates two `asyncio.Task` objects via `_execute_hedge()`, and races them with `asyncio.wait(FIRST_COMPLETED)`. The first response wins; the losing task is cancelled. The winner's backend name is recorded in `X-Hedge-Winner` and `X-Hedge-Loser` response headers. If the hedge path fails (both backends error, or only one backend is available), execution falls through to the normal retry loop.

### Implementation

**RoutingStrategy Protocol** (`gateway/strategies.py`):
- `RoutingStrategy` -- a `typing.Protocol` with a single `select(candidates, exclude, routing_key) -> str | None` method. Any class with a matching `select` signature satisfies the protocol without inheriting from it.

**ConsistentHashStrategy** (`gateway/strategies.py`):
- Wraps the existing `ConsistentHashRing`. When `routing_key` is provided, delegates to `ring.get_node(routing_key, exclude=exclude)`. Without a routing key, falls back to the first non-excluded candidate. This adapter exists to give consistent hashing the same interface as the other strategies.

**LatencyAwareStrategy** (`gateway/strategies.py`):
- Holds a reference to the shared `LatencyTracker` and a model name. On `select()`, filters candidates by `exclude`, calls `tracker.get_all_p95(model)` to get per-backend P95 latencies, sorts eligible backends by P95 ascending (backends without data sort to infinity), and returns the lowest. Ties are broken alphabetically by backend name for determinism.

**CostAwareStrategy** (`gateway/strategies.py`):
- Initialized with a `{backend_name: cost_per_1k_tokens}` dict extracted from `BackendConfig` at registry construction time. On `select()`, sorts eligible backends by cost ascending (backends without cost data sort to infinity), tie-breaks alphabetically.

**LatencyTracker** (`gateway/latency_tracker.py`):
- Stores observations in a `dict[tuple[str, str], deque[tuple[float, float]]]` keyed by `(backend, model)`. Each observation is a `(monotonic_timestamp, latency_ms)` tuple.
- `record(backend, model, latency_ms)` appends to the deque and triggers lazy pruning.
- `p95(backend, model)` prunes expired observations, sorts remaining latencies, and returns the value at the 95th percentile index (`min(int(len * 0.95), len - 1)`). Returns `None` if no data exists.
- `get_all_p95(model)` iterates all `(backend, model)` keys matching the requested model and returns a `{backend: p95_ms}` dict.
- `snapshot()` returns an admin-friendly view with count, P95, min, and max per backend per model.
- Pruning uses `time.monotonic()` for clock-immune timing. The window defaults to 60 seconds. Old observations are evicted from the left of the deque in O(1) per eviction.

**ModelRoutingConfig** (`gateway/config.py`):
- Pydantic model with `strategy: Literal["consistent_hash", "latency_aware", "cost_aware"]` (default `"consistent_hash"`) and `hedge_enabled: bool` (default `False`).
- The `GatewayConfig` model includes `routing: dict[str, ModelRoutingConfig]` mapping model names to their routing configuration.

**Registry strategy construction** (`gateway/config.py`):
- `Registry.__init__()` builds `self.model_strategies: dict[str, RoutingStrategy]` during initialization. Strategy construction is deterministic: the latency tracker is passed in from the lifespan, costs are extracted from backend configs, and the hash ring is reused from the existing `model_rings` dict.
- `find_backend_for_model()` delegates to the per-model strategy's `select()` method. If no strategy exists for the model (model not in registry), falls back to first non-excluded match.

**Hedge execution** (`gateway/routes/chat.py`):
- `_execute_hedge(http_client, backend1, backend2, chat_request, translator)` creates two `asyncio.Task` objects, each calling the same translator function against a different backend. Uses `asyncio.wait({task1, task2}, return_when=FIRST_COMPLETED)` to race them. The winning task's result is returned along with winner/loser identification and duration. Pending tasks are cancelled with `task.cancel()` followed by a guarded `await` to absorb `CancelledError`.

**Lifespan wiring** (`gateway/main.py`):
- `LatencyTracker()` is created in the lifespan and stored on `app.state.latency_tracker`. It is passed to `Registry()` during construction and again during SIGHUP hot-reload, ensuring the tracker survives config reloads while strategies get rebuilt with fresh config.

**Response headers** (`gateway/main.py`):
- The request middleware reads `request.state.hedge_winner` and `request.state.hedge_loser` and sets `X-Hedge-Winner` and `X-Hedge-Loser` response headers when present.

**Admin endpoint** (`gateway/routes/admin.py`):
- `GET /admin/routing` returns per-model routing state: strategy name, hedge enabled flag, and current P95 latencies from the tracker (when available).

**Config** (`config/backends.yaml`):
- The `routing` section maps model names to strategy and hedge config. Example: `mock-gpt-markdown` uses `latency_aware` with hedging enabled; `mock-claude-markdown` uses `cost_aware` with hedging disabled. Models without entries (e.g., `tinyllama`) default to `consistent_hash`.
- `cost_per_1k_tokens` is declared per-backend in the `backends` section and consumed by `CostAwareStrategy` at registry construction.

### Key design decisions

1. **Protocol over ABC.** `RoutingStrategy` is a `typing.Protocol`, not an `abc.ABC`. This means any class with a matching `select()` signature satisfies the interface without inheriting from a base class. It enables structural subtyping (duck typing with type checker support), avoids coupling implementations to a base class they do not need, and keeps the strategy implementations pure data-in/data-out classes with no framework dependencies.

2. **Rolling P95, not P50 or mean.** P50 is a measure of typical latency. P95 captures the tail -- the slow responses that matter most for user-perceived performance. A backend with a 50ms P50 but a 2-second P95 is worse for reliability than one with a 100ms P50 and 150ms P95. Using P95 as the routing signal causes the strategy to avoid backends that occasionally produce outlier responses, even if their median performance looks good.

3. **Per-model config, not per-backend.** Routing strategy is configured at the model level, not the backend level. A single backend can serve multiple models, and the optimal routing strategy depends on the workload characteristics of each model, not the backend's capabilities. A latency-optimized model (real-time chat) and a cost-optimized model (batch summarization) can share backends but use different strategies.

4. **Hedge restricted to non-streaming requests.** Streaming responses emit chunks incrementally. Racing two streaming responses would require either buffering both streams (doubling memory and defeating the purpose of streaming) or selecting a winner after the first chunk (unreliable latency signal). Restricting hedging to non-streaming requests keeps the implementation simple and the semantics clear: the client gets one complete response from whichever backend finishes first.

5. **FIRST_COMPLETED + cancel, not gather.** `asyncio.wait(FIRST_COMPLETED)` returns as soon as one task finishes. The loser is explicitly cancelled. Using `asyncio.gather()` would wait for both to finish, wasting backend capacity. The cancel-and-await pattern ensures the cancelled task's resources are cleaned up and `CancelledError` does not propagate.

6. **Latency tracker in-memory, not Redis.** Latency observations are inherently local to the gateway instance -- each instance measures its own RTT to each backend. Storing them in Redis would add a network hop to every observation and introduce cross-instance averaging that masks instance-specific routing conditions (e.g., one gateway instance in a closer data center). An in-memory deque with lazy pruning is simpler, faster, and more correct for this use case.

7. **Latency tracker survives config reload.** During SIGHUP-triggered hot-reload, the `LatencyTracker` instance is preserved while the `Registry` (and its strategies) is rebuilt. This means latency data collected before the reload continues to inform routing decisions after the reload, avoiding a cold-start penalty on every config change.

8. **Client opt-in for hedging via X-Hedge header.** Hedging doubles backend load. Making it client-initiated (rather than server-side always-on) gives callers control over the cost/latency trade-off. Latency-sensitive requests opt in; batch workloads do not. The server-side `hedge_enabled` flag acts as a guardrail so operators can disable hedging for models where it is not appropriate, regardless of client headers.

### Alternatives considered

1. **Weighted round-robin instead of latency-aware.** Round-robin distributes load evenly but does not adapt to real-time backend performance. If one backend is slow due to GPU memory pressure or a long-running batch job, round-robin continues sending it equal traffic. Latency-aware routing automatically shifts traffic away from degraded backends without manual weight tuning.

2. **P50 instead of P95 for latency signal.** P50 tracks the median request. Two problems: (a) it ignores tail latency entirely, so a backend with occasional multi-second hangs looks identical to a consistent sub-100ms backend; (b) medians are less sensitive to the recent performance shifts that matter for real-time routing. P95 strikes a balance between being sensitive to outliers and not overreacting to a single bad request (which P99 or max would do).

3. **Always-hedge (no client opt-in).** Simpler implementation -- no header parsing, no config flag. Rejected because it doubles backend load unconditionally. For models with a single backend or where latency is not critical, the extra load is pure waste. The dual opt-in (client header + server config) provides granular control.

4. **Redis-backed latency tracker.** Would enable cross-instance P95 aggregation. Rejected for three reasons: (a) adds a Redis RTT to every latency recording, turning a nanosecond in-memory operation into a millisecond network call; (b) cross-instance averaging is not obviously useful -- if instance A has better connectivity to backend X, it should route there, not be influenced by instance B's measurements; (c) the gateway already has a Redis dependency for caching and rate limiting, and adding latency tracking to the hot path increases the blast radius of Redis outages.

5. **ABC (abstract base class) instead of Protocol.** ABC requires explicit inheritance (`class MyStrategy(RoutingStrategy)`) and provides runtime `isinstance` checks. Protocol provides the same type safety at static analysis time without coupling implementations to the base. Since the gateway never does runtime `isinstance` checks on strategies, Protocol is strictly simpler.

### Failure modes and edge cases

- **Cold start (no latency data).** When the `LatencyTracker` has no observations for a model (first request, or all observations expired), `get_all_p95()` returns an empty dict. `LatencyAwareStrategy.select()` sorts all backends with `float("inf")` as their P95, then tie-breaks alphabetically by name. The result is deterministic and stable: the alphabetically first backend always handles cold-start traffic. After a few requests, real P95 data takes over.

- **All backends excluded (circuit-broken).** If every candidate for a model is in the `exclude` set, `select()` returns `None`. `find_backend_for_model()` returns `None`. The route handler raises HTTP 503 ("All backends unavailable for this model"). This is correct behavior -- routing to a circuit-broken backend would produce immediate failure.

- **Hedge: both backends fail.** If both hedge tasks raise exceptions, `asyncio.wait(FIRST_COMPLETED)` returns the first completed (failed) task. `winner_task.result()` re-raises the exception. The hedge code catches this and falls through to the normal retry loop, which tries backends one at a time with circuit breaker tracking. The hedge failure does not consume retry attempts.

- **Hedge: only one backend available.** If `find_backend_for_model()` with the first backend excluded returns `None` for the second, the hedge path is skipped entirely and the normal single-backend path executes. No partial hedge is attempted.

- **Cost data missing for a backend.** `CostAwareStrategy` treats backends without a `cost_per_1k_tokens` entry as `float("inf")`. They sort to the end of the list and are only selected when all priced backends are excluded. This is a safe default -- an unpriced backend is assumed expensive rather than free.

- **Latency window expiry during low traffic.** If a model receives no traffic for longer than the window (60 seconds default), all observations expire. The next request hits the cold-start path (alphabetically first backend). This is intentional: stale latency data from a minute ago is worse than no data, because backend performance can change significantly in that time.

- **Strategy mismatch after hot-reload.** If the config changes a model's strategy from `latency_aware` to `cost_aware`, the `Registry` is rebuilt with new strategy instances. The `LatencyTracker` retains its data (it survives reload), but the new `CostAwareStrategy` does not reference it. The old latency data is inert -- it is pruned away naturally by the window expiry. No stale-strategy bug is possible because strategy instances are rebuilt from scratch on each reload.

### Observability

**Prometheus metrics** (`gateway/observability/metrics.py`):
- `gateway_hedge_requests_total` (Counter, labels: `model`) -- incremented on every successful hedge execution. Tracks hedge volume per model.
- `gateway_hedge_win_rate` (Counter, labels: `backend`, `model`) -- incremented for the winning backend on each hedge. Comparing win counts across backends reveals which backend is consistently faster. Despite the name, it is a raw count, not a rate -- the rate is derived by dividing by `hedge_requests_total` in Grafana.

**Response headers**:
- `X-Hedge-Winner: <backend_name>` -- set when a hedge request completes successfully. Tells the client which backend responded.
- `X-Hedge-Loser: <backend_name>` -- the cancelled backend. Useful for debugging latency discrepancies.

**Admin endpoint**:
- `GET /admin/routing` -- returns per-model routing state: `{"mock-gpt-markdown": {"strategy": "latency_aware", "hedge_enabled": true, "p95_latencies": {"mock-openai-1": 45.2, "mock-openai-2": 62.1}}}`. Provides real-time visibility into which strategy each model uses and the current P95 latency data driving routing decisions.

**Structured logs**:
- `hedge_request_completed` -- logged on successful hedge with `model`, `tenant_id`, `winner`, `loser`, `duration_ms`.
- `hedge_request_failed` -- logged when the hedge path fails and falls through to the retry loop.
- `chat_request_received` and `chat_request_completed` -- existing logs that now include the backend selected by the strategy, providing a full audit trail of routing decisions.

### Testing

**Unit tests -- strategies** (`tests/unit/test_strategies.py`):
- `TestConsistentHashStrategy`: Deterministic routing (same key, same result over 100 calls), distribution (different keys hit at least 2 backends), exclusion (excluded backend never returned), all-excluded returns `None`, no-routing-key fallback to first candidate, and ring-equivalence (strategy produces identical results to the raw `ConsistentHashRing` for 200 keys).
- `TestLatencyAwareStrategy`: Routes to lowest P95, cold-start returns first candidate (alphabetical tie-break at infinity), exclusion skips lowest-P95 backend and returns next, all-excluded returns `None`, equal-P95 tie-broken alphabetically.
- `TestCostAwareStrategy`: Routes to cheapest, exclusion skips cheapest, missing cost treated as infinity (only selected when all priced backends excluded), all-excluded returns `None`, equal-cost tie-broken alphabetically.

**Unit tests -- latency tracker** (`tests/unit/test_latency_tracker.py`):
- Single observation P95, correct percentile calculation for 20 observations, unknown backend returns `None`, window expiry (time.monotonic mocked to advance past 60s), `get_all_p95` returns all backends for a model, model filtering (observations for other models not included), empty tracker returns empty dict, `snapshot()` returns count/P95/min/max per backend.

**Unit tests -- hedge** (`tests/unit/test_hedge.py`):
- Fast backend wins over slow backend (slow uses `asyncio.sleep(10)`), correct winner/loser identification when backend2 finishes first, both-fail raises the winner's exception, duration is positive.

**Integration tests -- routing strategies** (`tests/integration/test_routing_strategies.py`):
- `GET /admin/routing` returns correct strategy and hedge config for all three models (latency_aware, cost_aware, consistent_hash).
- Cost-aware picks cheapest: `mock-claude-markdown` routes to `mock-anthropic-1` (cost 0.01, cheapest of three).
- Hedge returns `X-Hedge-Winner` and `X-Hedge-Loser` headers with different values when `X-Hedge: true` is sent for a hedge-enabled model.
- No hedge headers on normal requests (no `X-Hedge` header).
- No hedge headers when `X-Hedge: true` is sent for a model with `hedge_enabled: false`.

### Production gaps

- **Single-instance latency tracker.** Each gateway instance maintains its own `LatencyTracker` in process memory. There is no cross-instance aggregation. In a multi-instance deployment behind a load balancer, each instance independently converges on its own view of backend latency. This is actually acceptable for most deployments (instances should route based on their own measurements), but it means the `GET /admin/routing` endpoint only shows one instance's data, and there is no global P95 dashboard without scraping all instances.

- **No cross-instance P95 aggregation for admin visibility.** An operator querying `GET /admin/routing` sees latency data from the instance that handled the request. To get a fleet-wide view, the operator would need to query every instance or build a Prometheus query over the `gateway_request_duration_seconds` histogram (which already has backend labels).

- **Hedge does not account for queue pressure.** The hedge implementation races two backends without checking their concurrency slot availability. If both backends are near capacity, the hedge consumes two slots instead of one, potentially pushing other requests into the priority queue. A production enhancement would check `queue_manager.get_concurrency()` before hedging and skip the hedge if either backend is above a configurable utilization threshold.

- **No hedge for streaming requests.** Hedging is restricted to non-streaming requests. Streaming is the dominant mode for interactive LLM use cases. A production system might implement a "first-chunk" hedge that races two streams and commits to whichever produces the first token faster, but this adds complexity around stream lifecycle management and partial response cleanup.

- **Static cost data.** `cost_per_1k_tokens` is configured statically in `backends.yaml`. Real provider pricing changes over time and varies by model variant. A production system would pull pricing from a cost catalog service or API and refresh periodically.

- **No strategy-level metrics.** The gateway records which backend handled a request (in `REQUEST_COUNT` labels) but does not emit a metric for which strategy was used. Adding a `strategy` label to request metrics would enable per-strategy latency and error rate dashboards.

### Interview talking points

- The `RoutingStrategy` Protocol demonstrates Go-style structural typing in Python. Any class with a `select()` method of the right signature satisfies the interface, with no base class coupling. This is a concrete example of the Interface Segregation Principle -- consumers depend on a minimal interface, and implementations are free to carry whatever internal state they need.

- The rolling-window P95 latency tracker solves the cold-start problem gracefully. Instead of requiring a warm-up period or manual configuration, the strategy falls back to deterministic alphabetical ordering when no data exists, then automatically adapts as observations accumulate. The 60-second window ensures the tracker reflects current backend health rather than historical averages, making it responsive to transient degradation.

- The hedge request pattern is a well-known technique for reducing tail latency (popularized by Jeff Dean's "Tail at Scale" paper). The implementation here shows the key engineering decisions: `FIRST_COMPLETED` instead of `gather`, explicit task cancellation to avoid wasting backend compute, client opt-in to control the cost trade-off, and graceful fallthrough to the retry loop when hedging fails. This is a good example of a feature that is simple in concept but requires careful handling of async lifecycle and error propagation.

### Likely interview questions

**Q: "Why not use weighted round-robin with dynamic weight adjustment instead of latency-aware routing?"**
**A:** Weighted round-robin requires a feedback loop to adjust weights, a policy for how aggressively to shift weights, and a stabilization mechanism to prevent oscillation. That is a full control-theory problem. Latency-aware routing sidesteps it entirely: there are no weights to tune, no oscillation risk, and no configuration beyond the window size. Each request independently picks the backend with the best current P95. The trade-off is that latency-aware routing does not guarantee even load distribution -- a backend with consistently low latency gets most traffic, which may itself cause latency to rise. In practice, the circuit breaker and retry loop handle this: if the preferred backend becomes overloaded and starts failing, the circuit breaker opens and the strategy automatically routes to the next-best backend.

**Q: "What happens if the latency tracker's 60-second window is too short or too long?"**
**A:** Too short: the tracker forgets data quickly, causing frequent cold-start fallbacks during low-traffic periods. A model with one request per minute loses its latency data between requests and always cold-starts. Too long: the tracker retains stale data that does not reflect current conditions. A backend that was fast 5 minutes ago but is now degraded still looks good. 60 seconds is a compromise for typical LLM workloads where requests arrive every few seconds. A production system could make the window configurable per model or use an adaptive window that lengthens during low traffic and shortens during high traffic.

**Q: "Hedging doubles your backend load. How do you prevent it from causing cascading overload?"**
**A:** Three mechanisms: (1) client opt-in via `X-Hedge: true` ensures only callers who value latency over efficiency request hedging; (2) server-side `hedge_enabled` flag per model lets operators disable hedging entirely for high-traffic models; (3) if the hedge fails (both backends error), the fallthrough to the retry loop means the total backend calls for a hedged request are at most 2 (hedge) + 3 (retry loop) = 5, which is bounded. The production gap is that there is no utilization-aware throttling -- hedging does not check whether backends are near capacity before sending duplicate requests. Adding a utilization threshold (e.g., skip hedging if either backend is above 80% concurrency) would close this gap.

**Q: "Why is the latency tracker per-instance rather than shared across instances?"**
**A:** Each gateway instance has its own network path to each backend. Instance A in availability zone us-east-1a may have 10ms RTT to backend X, while instance B in us-east-1b has 50ms. Sharing latency data would average out these differences, causing both instances to route to the globally-average-best backend instead of their locally-best backend. Per-instance tracking is more correct for routing decisions. The trade-off is admin visibility: the `GET /admin/routing` endpoint only shows one instance's data. For fleet-wide visibility, Prometheus already scrapes `gateway_request_duration_seconds` from all instances, and Grafana can compute per-backend P95 across the fleet.

## Advanced Caching

### Why this exists

Phase 8 introduced a Redis-backed semantic cache that eliminated redundant LLM backend calls for semantically similar prompts. Every cache lookup required a network round-trip to Redis (~1-5ms per lookup), even for queries that had been seen seconds ago on the same gateway instance. For hot queries — the same question asked repeatedly during a traffic spike — this network overhead is entirely avoidable with an in-process cache.

Two additional inefficiencies remained. First, system prompts (e.g., "You are a helpful assistant") are repeated verbatim across thousands of requests but get re-embedded on every lookup. Computing a 384-dimensional embedding via sentence-transformers takes ~5-15ms on CPU; for a system prompt that never changes, this is wasted compute. Second, operators had no way to pre-populate the cache before anticipated traffic spikes — the cache could only be warmed organically by real user traffic, meaning the first wave of requests after a deployment or cache flush always hit backends directly.

Advanced caching addresses all three problems: an L1 in-process LRU cache eliminates network round-trips for hot queries, a prefix cache avoids redundant system prompt embedding computation, and a cache warming endpoint lets operators pre-populate the cache before traffic arrives.

### How it works

**Two-tier lookup (L1 → L2 → backend):**

1. Client sends a chat completion request. After auth and rate limiting, the route handler calls `SemanticCache.lookup()`.
2. `lookup()` computes the query embedding for the user text and builds a scope key from `(model, system_hash, tenant_id, cache_isolation)`.
3. The L1 cache (in-process `L1Cache`) is checked first. It scans all entries within the matching scope, computes cosine similarity against each stored embedding, and returns the best match above the similarity threshold. If found, the response is returned immediately with tier `L1_HIT`. No network I/O occurs.
4. On L1 miss, the L2 cache (Redis) is checked. Entry IDs are fetched from the scope set via `SMEMBERS`, embeddings are batch-fetched via `HGETALL`, and cosine similarity is computed against each. The best match above threshold is returned with tier `L2_HIT`.
5. On L2 hit, the entry is **promoted to L1**: `L1Cache.store()` is called with the entry ID, scope key, embedding, and response JSON. The next identical or similar query will hit L1 directly.
6. On complete miss (both tiers), the request proceeds to the backend. After the backend responds, the result is stored in both L1 and L2 simultaneously.
7. The middleware sets `X-Cache` response header to `L1_HIT`, `L2_HIT`, or `MISS`, and `X-Cache-Similarity` to the cosine similarity score on hits.

**Prefix cache for system prompt embeddings:**

1. On every `lookup()` and `store()` call, `compute_system_embedding()` is invoked with the message list.
2. The method extracts and hashes all system-role message content using SHA-256 (truncated to 16 characters).
3. If the hash is not in `_prefix_cache`, the system prompt text is embedded via sentence-transformers and the resulting vector is stored in the dict keyed by the hash.
4. If the hash already exists, the cached embedding is returned immediately — no model inference.
5. The prefix cache is a plain `dict[str, list[float]]` on the `SemanticCache` instance. It persists for the lifetime of the process.

**Cache warming:**

1. Operator sends `POST /admin/cache/warm` with a JSON body containing a list of prompts (each with `model` and `messages`).
2. For each prompt, the endpoint finds the appropriate backend, calls the translator to get a real LLM response, and stores the result in the semantic cache with `tenant_id="__warm__"` and `cache_isolation="shared"`.
3. Because warmed entries use shared isolation, they are visible to all tenants. Entries flow into both L1 and L2 via the normal `SemanticCache.store()` path.
4. The endpoint returns a summary: `{"warmed": N, "errors": M}`.

### Implementation

**L1Cache** (`gateway/l1_cache.py`):
- `L1Cache` class with `OrderedDict[str, L1Entry]` for O(1) LRU operations. `L1Entry` is a dataclass holding `scope_key`, `embedding` (list of floats), `response_json` (serialized string), `created_at` (monotonic timestamp), and `hit_count`.
- `_scope_index: dict[str, set[str]]` maps scope keys to entry IDs, enabling scoped lookup without scanning the entire cache. Only entries matching the query's `(model, system_hash)` scope are compared.
- `lookup(scope_key, query_embedding, threshold)` iterates entries in the matching scope, computes cosine similarity via NumPy (`np.dot` + `np.linalg.norm`), returns the best match above threshold. Expired entries are lazily evicted during the scan. On hit, the entry is moved to the end of the OrderedDict via `move_to_end()`.
- `store(entry_id, scope_key, embedding, response_json)` appends to the OrderedDict. If at capacity (`max_entries`), `_evict_oldest()` removes from the front (LRU victim). Duplicate entry IDs are deduplicated via `move_to_end()` without overwriting.
- `flush()` clears both `_entries` and `_scope_index`, resets hit/miss counters.
- `stats()` returns `l1_entries`, `l1_max_entries`, `l1_hits`, `l1_misses`.
- `cosine_similarity(a, b)` is a module-level function computing similarity between two float vectors using NumPy for vectorized performance.

**SemanticCache L1/L2 integration** (`gateway/semantic_cache.py`):
- `SemanticCache.__init__()` creates an `L1Cache` instance with configurable `l1_max_entries` and `l1_ttl` parameters.
- `lookup()` returns a 3-tuple `(response, similarity, tier)` where `tier` is `"L1_HIT"`, `"L2_HIT"`, or `None`. L1 is checked before L2. On L2 hit, the entry is promoted to L1 via `self._l1.store()`.
- `store()` writes to both L2 (Redis pipeline: `HSET` + `EXPIRE` + `SADD` + `EXPIRE`) and L1 (`self._l1.store()`) in sequence.
- `get_stats()` merges L2 stats (Redis `cache:stats` hash) with L1 stats (`self._l1.stats()`), returning a combined dict with `hits`, `misses`, `hit_rate`, `entries`, `l1_entries`, `l1_max_entries`, `l1_hits`, `l1_misses`.
- `flush()` clears both L1 (`self._l1.flush()`) and L2 (Redis `SCAN` + `DELETE`), returns the combined count.

**Prefix cache** (`gateway/semantic_cache.py`):
- `_prefix_cache: dict[str, list[float]]` on `SemanticCache`, keyed by the 16-character SHA-256 hash of concatenated system message content.
- `compute_system_embedding(messages)` returns `(sys_hash, embedding)`. If the hash is already in `_prefix_cache`, the embedding is returned without calling `SentenceTransformer.encode()`. If not, the system text is embedded and cached. Empty system text produces an empty embedding list and is not cached.
- Both `lookup()` and `store()` call `compute_system_embedding()` to benefit from the prefix cache.

**Cache warming** (`gateway/routes/admin.py`):
- `CacheWarmRequest` Pydantic model validates the request body: `prompts` is a list of dicts, each containing `model` (str) and `messages` (list of message dicts).
- `POST /admin/cache/warm` iterates prompts, finds a backend via `registry.find_backend_for_model()` (respecting circuit breaker exclusions), calls the appropriate translator, and stores the result via `semantic_cache.store()` with `tenant_id="__warm__"` and `cache_isolation="shared"`.
- Invalid prompts (missing model, bad message format) increment the `errors` counter and are skipped. Backend failures are caught and logged as warnings.

**Configuration** (`gateway/main.py`):
- `L1_MAX_ENTRIES` — environment variable, default 500. Controls the maximum number of entries in the L1 cache.
- `L1_TTL` — environment variable, defaults to `CACHE_TTL` (which defaults to 3600 seconds). Controls how long L1 entries survive before lazy expiration.

### Key design decisions

1. **OrderedDict for O(1) LRU.** Python's `OrderedDict` provides `move_to_end()` in O(1) and `popitem(last=False)` in O(1), which is exactly what an LRU cache needs. The alternative — a plain `dict` plus a separate access-time list — would require O(n) scans to find and remove the LRU victim. `functools.lru_cache` was not suitable because the cache key is a high-dimensional embedding vector (not hashable in a useful way), and cache entries must be scoped by `(model, system_hash)`.

2. **No locks.** The L1 cache is accessed only from async coroutines running on asyncio's single-threaded event loop. There is no preemptive multithreading, so no data races are possible. Adding locks would introduce unnecessary overhead and complexity. If the gateway were to use multiple worker processes (e.g., via `--workers N` in uvicorn), each process gets its own L1 cache, which is the correct behavior — each process should cache its own hot set.

3. **TTL checked lazily, not via background sweeper.** Expired entries are discovered and evicted during `lookup()` scans, not by a background task. This avoids the complexity of a periodic sweeper (timer management, cancellation on shutdown, potential race conditions with concurrent lookups). The trade-off is that expired entries consume memory until the next lookup in their scope. With a 500-entry cap, this is at most ~2MB of wasted memory — negligible for a gateway process.

4. **500-entry default cap (~2MB memory).** Each L1 entry stores a 384-float embedding (~1.5KB as a Python list) plus a serialized JSON response (~2-4KB for typical LLM responses). At 500 entries, the L1 cache consumes roughly 1.5-3MB. This is small enough to be negligible in a gateway process that already loads a sentence-transformers model (~80MB) but large enough to cover the hot working set for most workloads.

5. **Prefix cache is unbounded.** System prompts are few in practice — most applications use 1-5 distinct system prompts. An unbounded `dict` is the simplest and fastest implementation. Even with 1000 distinct system prompts, the prefix cache would consume ~600KB (1000 * 384 floats * 4 bytes). The risk of unbounded growth is acknowledged as a production gap.

6. **Warm writes to shared isolation only.** The warming endpoint uses `tenant_id="__warm__"` with `cache_isolation="shared"`, making warmed entries visible to all tenants. This is intentional: warming is an operator action to benefit the entire fleet, not a tenant-specific operation. Tenants configured with `cache_isolation="tenant"` will not see warmed entries, which is consistent with their isolation guarantee — they opted out of shared caching.

7. **L2 hits promote to L1.** When a query misses L1 but hits L2, the entry is immediately copied into L1. This means the second identical query avoids the Redis round-trip entirely. The alternative — only populating L1 on backend responses — would miss the optimization for queries that are popular enough to be in Redis but not yet in the local process's L1.

### Alternatives considered

1. **Redis-only with connection pooling.** Instead of an in-process L1, optimize the Redis path with connection pooling and pipelining. This would avoid the complexity of a two-tier cache and the consistency issues of an L1 that can go stale. Rejected because even an optimized Redis lookup is ~1ms (network round-trip), while an in-process dict lookup is ~10us. For hot queries during traffic spikes, this 100x difference matters. The consistency trade-off (L1 staleness up to TTL) is acceptable for a cache that already uses approximate matching.

2. **`dict` + `list` instead of `OrderedDict`.** Use a plain `dict` for O(1) key lookup and a `list` of `(access_time, key)` tuples sorted by access time for eviction. This would require O(n) scans to update access order on each hit and O(n log n) sorts for eviction. `OrderedDict.move_to_end()` is O(1), making it strictly better for LRU.

3. **Combined embedding (system + user text).** Instead of separate prefix caching, concatenate system and user text and embed them together. This would produce a single embedding that captures both contexts. Rejected because the system prompt is repeated across many requests with different user text. By caching the system embedding separately, we avoid recomputing it for every request. The scope key `(model, system_hash)` already partitions the cache so that entries with different system prompts are never compared against each other.

4. **Background warming via async tasks.** Instead of a synchronous warm endpoint that blocks until all prompts are processed, spawn background tasks for each prompt. This would make the warm endpoint return immediately with a job ID. Rejected for simplicity: the number of prompts in a warm request is typically small (10-50), and the total time is bounded by the slowest LLM response (~30s). A production system handling thousands of warm prompts would benefit from background processing, but that adds job tracking, progress reporting, and failure handling complexity.

5. **Write-through L1 on L2 store instead of promote-on-hit.** Populate L1 every time an entry is written to L2, not just when L2 is hit. The current implementation already does this — `SemanticCache.store()` writes to both tiers. The promote-on-hit mechanism is an additional optimization for entries that were already in L2 (e.g., stored by a different request or before a gateway restart) but not yet in the current process's L1.

### Failure modes and edge cases

- **L1 stale after Redis invalidation.** If an operator flushes the Redis cache via `DELETE /admin/cache`, L2 entries are deleted but L1 entries persist until their TTL expires. During this window, queries may return stale cached responses from L1 that no longer exist in L2. The staleness window is bounded by `L1_TTL` (default 3600s). Calling `DELETE /admin/cache` through the gateway endpoint does flush both tiers via `SemanticCache.flush()`, but direct Redis manipulation (e.g., `redis-cli FLUSHDB`) bypasses L1.

- **Prefix cache unbounded growth.** If the gateway serves requests with many distinct system prompts (e.g., a multi-tenant platform where each tenant has a unique system prompt), the prefix cache grows without bound. Each entry is ~1.5KB (384 floats). With 10,000 distinct system prompts, the prefix cache would consume ~15MB. There is no eviction policy. A production system would add an LRU wrapper or a max-size cap to `_prefix_cache`.

- **Warm endpoint timeout on large prompt lists.** The warm endpoint processes prompts sequentially. If each LLM call takes 10 seconds and there are 50 prompts, the total wall time is ~500 seconds — likely exceeding the client's HTTP timeout. There is no progress streaming or partial result reporting. Large warm jobs should be split into multiple smaller requests.

- **L1 capacity exhaustion under cardinality explosion.** If request patterns have high cardinality (many unique queries, few repeats), the L1 cache churns through entries via LRU eviction without producing hits. Every store triggers an eviction, and the evicted entry was never looked up. In this pathology, L1 adds overhead (store + evict) without benefit. The 500-entry cap bounds the memory waste, but the CPU overhead of cosine similarity comparisons during fruitless lookups is proportional to the number of entries in each scope.

- **Embedding model lazy-load latency spike.** The sentence-transformers model is loaded on the first call to `compute_embedding()`. This first call takes several seconds as the model is loaded from disk into memory. If the first request to the gateway happens to be a cache warm call with many prompts, the lazy load happens once and subsequent embeddings are fast. But if the first request is a normal user query, that query pays the load-time penalty. There is no explicit pre-loading of the model during startup.

- **L1 entry deduplication preserves stale response.** When `L1Cache.store()` is called with an `entry_id` that already exists, it calls `move_to_end()` but does not update the stored embedding or response JSON. If the same entry was updated in Redis (e.g., by another gateway instance), the L1 copy retains the original data until it expires or is evicted.

- **Scope index orphans.** If `_evict()` is called for an entry whose scope key has already been removed from `_scope_index` (e.g., due to a race in cleanup), the `discard()` call is a no-op. This is safe but can leave empty sets in `_scope_index` if the cleanup path has a bug. The code handles this by checking `if not scope_entries: del self._scope_index[entry.scope_key]` after each discard.

### Observability

**Response headers:**
- `X-Cache: L1_HIT` — served from in-process L1 cache (no network I/O).
- `X-Cache: L2_HIT` — served from Redis L2 cache (one Redis round-trip).
- `X-Cache: MISS` — no cache hit, request forwarded to backend.
- `X-Cache-Similarity: 0.9812` — cosine similarity score of the matched entry (present on hits only).

**Admin endpoints:**
- `GET /admin/cache/stats` returns combined L1/L2 statistics: `l1_entries`, `l1_max_entries`, `l1_hits`, `l1_misses` (from L1), plus `hits`, `misses`, `hit_rate`, `entries` (from L2/Redis). This gives operators a single view of both tiers.
- `POST /admin/cache/warm` returns `{"status": "completed", "warmed": N, "errors": M}` for pre-population results.
- `DELETE /admin/cache` flushes both L1 and L2, returning the total count of deleted entries across both tiers.

**Prometheus metrics:**
- `gateway_cache_operations_total{model, status}` — counter with `status="hit"` or `status="miss"`, incremented in the chat route on every cache-eligible request.

**Structured logs:**
- `cache_hit` — emitted on L1 or L2 hit with `model`, `tenant_id`, `similarity`, and `streaming` fields.
- `cache_warm_completed` — emitted after warming with `warmed` and `errors` counts.
- `cache_warm_failed` — emitted per-prompt on warming failure with `model` and `error`.
- `cache_warm_no_backend` — emitted when no backend is available for a warmed model.
- `cache_warm_invalid_prompt` — emitted when a warm prompt fails validation.
- `embedding_model_loaded` — emitted once when the sentence-transformers model is lazy-loaded.

### Testing

**Unit tests — L1Cache (14 tests in `tests/unit/test_l1_cache.py`):**
- `TestL1Lookup` (6 tests): hit above threshold returns response and similarity, miss below threshold returns None, miss on empty scope returns None, best match selected when multiple entries exist, LRU touch on hit (move_to_end verified by evicting and checking which entry survives), expired entry lazily evicted during lookup (uses mocked `time.monotonic`).
- `TestL1Store` (4 tests): store and retrieve round-trip, dedup of existing entry ID (move_to_end without overwrite), capacity eviction removes oldest entry, scope index populated on store.
- `TestL1Eviction` (3 tests): evict_oldest removes front of OrderedDict, evict cleans scope index but preserves scope with remaining entries, evict removes empty scope from index.
- `TestL1Stats` (2 tests): hit/miss counters increment correctly, flush returns count and resets all state.

**Unit tests — SemanticCache L1 integration (7 tests in `tests/unit/test_semantic_cache.py`):**
- `TestL1Integration` (3 tests): L1 hit after store (store populates L1, second lookup returns `L1_HIT`), L2 hit promotes to L1 (first lookup `L2_HIT`, second lookup `L1_HIT`), flush clears L1 entries.
- `TestPrefixCache` (4 tests): system embedding cached (second call does not recompute), different system prompts cached separately, empty system text not cached, prefix cache populated during lookup.

**Integration tests — two-tier cache (3 tests in `tests/integration/test_semantic_cache.py`):**
- `TestTwoTierCache`: L1_HIT header appears in response, L2_HIT header appears in response, MISS header appears on cache miss. Uses `httpx.ASGITransport` with mocked `SemanticCache` to verify header propagation through the middleware stack.

**Integration tests — cache warming (1 test in `tests/integration/test_semantic_cache.py`):**
- `TestCacheWarmIntegration`: warming endpoint returns correct warmed count with mocked translator and cache.

### Production gaps

- **No cross-instance L1 invalidation.** When one gateway instance flushes the cache via `DELETE /admin/cache`, only that instance's L1 is cleared. Other instances retain their L1 entries until TTL expiry. In a multi-instance deployment behind a load balancer, this means a flush request must be broadcast to all instances, or operators must accept a staleness window equal to `L1_TTL`. A production system would use Redis Pub/Sub or a shared invalidation channel to propagate flush events to all instances.

- **No L1 cache sharding or size-based eviction.** The L1 cache uses a single `OrderedDict` with a fixed entry count limit. There is no memory-based limit (e.g., "max 50MB"). An entry with a 10KB response and an entry with a 100KB response consume the same "slot." A production system would track actual memory usage and evict based on memory pressure rather than entry count.

- **Warm endpoint is synchronous.** The warming endpoint processes each prompt sequentially, blocking until completion. For large prompt lists, this can exceed HTTP timeouts. A production system would accept the warm request, return a job ID immediately, and process prompts asynchronously with progress tracking via a separate status endpoint.

- **Prefix cache is unbounded.** No maximum size or eviction policy for the system prompt embedding cache. In a deployment with thousands of distinct system prompts, memory grows linearly without bound. Adding an LRU wrapper with a configurable cap (e.g., 100 entries) would bound memory while preserving the optimization for the most frequently used system prompts.

- **No L1 hit rate monitoring or adaptive sizing.** The L1 cache size is a fixed configuration value. There is no runtime feedback loop that increases the size if the hit rate is high (suggesting the working set is larger than the cache) or decreases it if the hit rate is low (suggesting the cache is wasting memory on non-repeating queries). A production system could expose L1 hit rate as a Prometheus metric and use it to auto-tune the cache size.

- **No warming for tenant-isolated caches.** The warm endpoint always stores with `cache_isolation="shared"`. Tenants configured with `cache_isolation="tenant"` receive no benefit from warming. Supporting tenant-specific warming would require an optional `tenant_id` field in the warm request body.

### Interview talking points

- The two-tier cache architecture mirrors CPU cache hierarchies (L1/L2/L3). The design principle is the same: put the fastest, smallest cache closest to the consumer (in-process), and use a larger, slower shared cache (Redis) as a backing store. The L1-to-L2 promotion on hit is analogous to a cache line fill in hardware. This is a well-understood pattern that works because of temporal locality — queries that were asked recently are likely to be asked again soon.

- The decision to skip locking on L1 is a direct consequence of asyncio's cooperative multitasking model. In asyncio, coroutines yield control explicitly at `await` points, and L1 operations (`lookup`, `store`, `_evict`) contain no `await` statements — they are pure synchronous Python operating on in-process data structures. This means they execute atomically with respect to other coroutines. This is a concrete example of how understanding your concurrency model eliminates unnecessary synchronization overhead.

- Prefix caching for system prompts is a domain-specific optimization that exploits a structural property of LLM workloads: the system prompt is repeated verbatim across many requests while the user text varies. By hashing and caching the system prompt embedding separately, the gateway converts an O(n) embedding computation (where n is the system prompt token count) into an O(1) hash lookup for every request after the first. This is analogous to common subexpression elimination in compilers — identifying repeated work and computing it once.

### Likely interview questions

**Q: "The L1 cache can serve stale data after a Redis flush. Why is this acceptable for a cache but not for, say, a database?"**
**A:** A semantic cache is inherently approximate — it already returns responses from *similar* queries, not exact matches. The similarity threshold (0.95) means every cache hit is already a "close enough" answer. Serving a response that was valid 30 minutes ago but has since been flushed from Redis is the same class of approximation. The staleness window is bounded by L1_TTL, and the impact of a stale cache hit is that the user gets a slightly older response — not data corruption or incorrect state. For a database, staleness means returning data that contradicts the system's committed state, which violates correctness guarantees. For a cache, staleness is a performance trade-off, not a correctness violation.

**Q: "Why not use Redis Pub/Sub for L1 invalidation instead of accepting staleness?"**
**A:** Redis Pub/Sub would solve cross-instance invalidation but adds operational complexity: every gateway instance must maintain a persistent subscription, handle reconnection on Redis failover, and process invalidation messages on a background task. The invalidation message must include enough information to identify which L1 entries to evict (scope key? entry ID? full flush?). For a local development gateway with 1-2 instances, this complexity is not justified. The staleness window (up to TTL) is acceptable because the cache already serves approximate results. In a production deployment with strict freshness requirements, Pub/Sub or a shared invalidation bus would be the right addition.

**Q: "How would you handle a workload where every query is unique and the L1 cache has a 0% hit rate?"**
**A:** The L1 cache adds two costs per request on miss: `store()` (~O(1) OrderedDict insert + potential eviction) and `lookup()` (O(k) cosine similarity computations where k is the number of entries in the matching scope). If every query is unique, every store triggers an eviction and no lookup ever matches. The overhead is small in absolute terms (~100us per request) but provides zero benefit. Two mitigations: (1) monitor the L1 hit rate via `/admin/cache/stats` and if it is consistently near zero, disable L1 by setting `L1_MAX_ENTRIES=0`; (2) a more sophisticated approach would be an adaptive cache that automatically shrinks when the hit rate drops below a threshold, reallocating memory to more useful purposes.

**Q: "The prefix cache is unbounded. What would happen at 100,000 distinct system prompts?"**
**A:** Each prefix cache entry stores a 384-float embedding as a Python list. A Python float is 24 bytes (object overhead), so 384 floats is ~9.2KB per entry. At 100,000 entries, the prefix cache would consume ~920MB — a significant fraction of available RAM for a gateway process. The fix is straightforward: wrap `_prefix_cache` in an LRU with a cap (e.g., `functools.lru_cache` on a method that takes `sys_hash` as the key, or a manual OrderedDict like L1Cache). In practice, 100,000 distinct system prompts suggests a design issue in the calling application — system prompts should be templates, not per-request unique strings.

## Resilience & Chaos Testing

### Why this exists

The gateway has retry logic, circuit breakers with rolling-window error tracking, per-tenant rate limiting, and multi-backend failover. All of these components were built and unit-tested in isolation, but none had been validated under deliberate, concurrent failure injection. Unit tests for the circuit breaker prove it trips at the right threshold — they do not prove it trips correctly when a retry loop is simultaneously failing over to a different backend while the rate limiter is rejecting a concurrent request from another tenant. Chaos testing fills this gap by injecting realistic faults at the backend HTTP layer and observing how the full stack responds.

Without chaos testing, the first time these components interact under failure is production. The goal is to discover failure modes — compounding errors, resource leaks on injected timeouts, malformed error responses under partial failure — before they affect real traffic.

### How it works

**Chaos injection (per-request):**

1. When `CHAOS_ENABLED=true`, the lifespan in `gateway/main.py` wraps the shared `httpx.AsyncClient` in a `ChaosHttpClient`. All downstream code (translators, retry loops) calls `client.post()` or `client.stream()` on this wrapper without knowing chaos is active.
2. On each `post()` or `stream()` call, `ChaosHttpClient._roll_injection()` consults its seeded `random.Random` instance and checks three independent probabilities in priority order: timeout (highest severity, checked first), 5xx error, then latency.
3. If timeout is selected, `httpx.ReadTimeout` is raised immediately — the real backend is never contacted. The translator's exception handler converts this to a 504, which triggers the retry loop in the chat route to try the next backend.
4. If error is selected, `httpx.HTTPStatusError` with a 500 response is raised. The translator converts this to an `HTTPException`, which again triggers retry/failover.
5. If latency is selected, `asyncio.sleep(delay_ms / 1000)` is awaited before delegating to the real client. The request eventually succeeds but with added delay. This exercises timeout boundaries and latency tracking.
6. If no injection fires, the call passes through to the real `httpx.AsyncClient` unchanged.

**Streaming chaos:**

For streaming requests (`client.stream()`), `ChaosHttpClient` returns a `_ChaosStreamContext` async context manager. Injection happens in `__aenter__` — before the real stream is opened. If a fault is injected, the exception propagates and `_real_ctx` is never set. The `__aexit__` method checks for this and short-circuits: `if self._real_ctx is not None: return await self._real_ctx.__aexit__(*args)`. This avoids attempting to close a stream that was never opened.

**Load generation (locust):**

1. `make loadtest` runs the `locustio/locust` Docker image with `--network host`, targeting the gateway at `http://localhost:8080`.
2. Two user classes simulate distinct tenant profiles. `TenantAlphaUser` (weight 1) uses `test-alpha-key` with steady request intervals (0.5–2.0s wait). `TenantBetaUser` (weight 2, so 2x the alpha population) uses `test-beta-key` with bursty intervals (0.1–1.0s wait).
3. Request mix is weighted: 70% `mock-gpt-markdown`, 30% `mock-claude-markdown`, with Beta users additionally making 20% streaming requests.
4. Prompt lengths follow a 50/30/20 distribution (short/medium/long) via `_pick_prompt()`, generating variable-size payloads that exercise different backend processing paths.
5. Locust runs headless for 60 seconds with 20 users, ramp rate 5/s. An HTML report is written to `tests/load/report.html`.

**End-to-end flow:**

`make chaos` starts the gateway with chaos injection active, then `make loadtest` drives traffic through it. The full path exercised per request is: locust HTTP call -> FastAPI middleware -> rate limiter check -> retry loop (up to 3 backends) -> chaos injection -> translator error handling -> circuit breaker recording -> failover to next backend (on failure) -> response to client. This is the same path production traffic takes, with faults injected at the backend boundary.

### Implementation

**ChaosConfig** (`gateway/chaos.py`):
- Dataclass with six fields: `error_rate` (default 0.10), `timeout_rate` (0.05), `latency_rate` (0.30), `latency_min_ms` (50.0), `latency_max_ms` (2000.0), `seed` (optional, for deterministic replay).
- All rates are independent probabilities, not mutually exclusive. The check order (timeout > error > latency) creates implicit priority: if timeout fires, error and latency are never checked. This models real failure patterns where a hard timeout preempts slower failure modes.

**ChaosHttpClient** (`gateway/chaos.py`):
- `__init__(client, config)` stores the real client and creates a `random.Random(config.seed)` instance for deterministic injection sequences.
- `post(url, **kwargs)` calls `_roll_injection(url)`, then either raises immediately (timeout/error), sleeps then delegates (latency), or delegates directly (no injection).
- `stream(method, url, **kwargs)` returns a `_ChaosStreamContext` that performs injection in `__aenter__`.
- `__getattr__(name)` delegates all other attribute access to the real client. This means `chaos_client.headers`, `chaos_client.aclose()`, etc., work transparently.
- `_roll_injection(url)` returns `"timeout"`, `"error"`, a `float` (latency ms), or `None`. Each check uses an independent `self._rng.random()` call against the configured rate. Latency values are drawn from `uniform(latency_min_ms, latency_max_ms)`.

**_ChaosStreamContext** (`gateway/chaos.py`):
- Async context manager that wraps `client.stream()`. Injection fires in `__aenter__`, not during iteration. This means the failure point is connection establishment, not mid-stream — matching the most common real-world failure pattern (connection timeout, not mid-response drop).
- `__aexit__` guards against `self._real_ctx is None`, which happens when injection fires before the real stream is opened. Without this guard, calling `__aexit__` on a `None` context would raise `AttributeError`.

**Lifespan integration** (`gateway/main.py`):
- Conditional import: `from gateway.chaos import ChaosConfig, ChaosHttpClient` only runs when `CHAOS_ENABLED=true`. The chaos module is not imported in normal operation.
- Six environment variables control chaos behavior: `CHAOS_ENABLED`, `CHAOS_ERROR_RATE`, `CHAOS_TIMEOUT_RATE`, `CHAOS_LATENCY_RATE`, `CHAOS_LATENCY_MIN_MS`, `CHAOS_LATENCY_MAX_MS`. All have sensible defaults matching `ChaosConfig`.
- The wrapped client replaces `app.state.http_client`, so all downstream code automatically uses it. No changes to translators, retry logic, or any other module.

**Docker Compose override** (`docker-compose.chaos.yml`):
- Sets `CHAOS_ENABLED=true` and all six rate/latency environment variables on the gateway service.
- Used via `docker compose -f docker-compose.yaml -f docker-compose.chaos.yml up`. The override file only adds environment variables — it does not change ports, volumes, or service topology.

**Makefile targets:**
- `make chaos` — starts the full stack with chaos overlay: `docker compose -f docker-compose.yaml -f docker-compose.chaos.yml up --build -d`.
- `make chaos-down` — stops the chaos stack.
- `make loadtest` — runs locust in a disposable Docker container with `--network host`, targeting `http://localhost:8080`. Parameters: 20 users, ramp rate 5/s, 60-second run, HTML report output.

**Locust harness** (`tests/load/locustfile.py`):
- `TenantAlphaUser` — steady traffic (wait 0.5–2.0s), two task types (`mock-gpt-markdown` weight 7, `mock-claude-markdown` weight 3).
- `TenantBetaUser` — bursty traffic (wait 0.1–1.0s), three task types including a streaming endpoint (`mock-gpt-markdown` streaming weight 2). Streaming task uses `catch_response=True` and consumes the full SSE stream via `iter_lines()`.
- `_pick_prompt()` — weighted random selection: 50% short (1 message, 2-5 words), 30% medium (2-4 messages with system prompt), 20% long (6 messages, multi-turn conversation with detailed system prompt).

### Key design decisions

1. **httpx wrapper instead of middleware.** A FastAPI middleware would intercept requests at the HTTP boundary of the gateway itself — before routing, before the retry loop, before the translator. Faults injected at that layer would be indistinguishable from a client-side error. By wrapping `httpx.AsyncClient`, chaos is injected at the backend call boundary — the exact point where real backend failures occur. This means the translator's exception handling, the retry loop's failover logic, and the circuit breaker's failure recording all execute their real code paths. A middleware-level injection would bypass all of them.

2. **Configurable rates via environment variables, not code changes.** Every chaos parameter is an env var with a default. This means an operator can tune chaos intensity without rebuilding the Docker image: `CHAOS_ERROR_RATE=0.50 make chaos` instantly doubles the error rate. The same gateway binary runs with and without chaos — no feature flags, no conditional compilation, no separate build.

3. **Separate compose override file.** The chaos configuration lives in `docker-compose.chaos.yml`, not in the main `docker-compose.yaml`. This keeps the primary compose file clean for development and CI. The compose `-f` flag's override semantics (later files merge into earlier ones) means only the six environment variables are added — nothing else changes. There is no risk of accidentally running chaos in a non-test environment because the override file must be explicitly specified.

4. **Seeded RNG for determinism.** `ChaosConfig.seed` allows reproducible injection sequences. Two runs with the same seed produce the exact same fault pattern, making failures debuggable: "the 7th request always gets a timeout with seed=42." The per-instance `random.Random` avoids polluting the global random state. In tests, seed is set explicitly. In Docker, seed is omitted (defaults to `None`), producing non-deterministic chaos — closer to real-world conditions.

5. **Locust in Docker with `--network host`.** Running locust inside a Docker container eliminates the need to install locust or its dependencies on the host machine. `--network host` means locust connects directly to `localhost:8080` where the gateway is running. The alternative — adding locust as a service in docker-compose — would create a dependency between the load generator and the system under test. Keeping them decoupled means `make loadtest` works against any running gateway, whether started with `make up`, `make chaos`, or running natively.

6. **Injection in `__aenter__`, not during stream iteration.** Real streaming failures most commonly occur at connection time (DNS resolution, TCP handshake, TLS negotiation, HTTP upgrade), not mid-stream. Injecting in `__aenter__` models this pattern. Mid-stream injection (dropping bytes, corrupting chunks) would require wrapping the async iterator, adding complexity for a less common failure mode. The trade-off is that the chaos system does not test partial stream delivery — a production gap acknowledged below.

7. **Timeout check before error check in `_roll_injection()`.** When both timeout_rate and error_rate could fire, timeout takes priority. This models real network behavior: a complete timeout (no response at all) is more severe than a fast 500 error, and when a backend is failing badly, timeouts are what the caller typically experiences first. The priority ordering also ensures deterministic behavior under extreme configurations (`timeout_rate=1.0, error_rate=1.0` always produces a timeout, not a coin flip).

### Alternatives considered

1. **Middleware-level injection.** Injecting faults in a FastAPI middleware (e.g., randomly returning 500 before the route handler runs) would be simpler to implement — a single middleware function instead of a wrapper class. Rejected because it bypasses the retry loop, circuit breaker, and translator error handling entirely. The whole point of chaos testing is to exercise the resilience mechanisms, and middleware injection skips all of them. A middleware approach would test "does the client handle 500s?" which is not what this system validates.

2. **Network-level chaos (tc, iptables, Toxiproxy).** Tools like `tc netem` (Linux traffic control) or Toxiproxy inject latency and packet loss at the network layer, which is more realistic than application-level injection. Rejected for local development: `tc` requires Linux with `NET_ADMIN` capability (not available on macOS Docker), Toxiproxy requires an additional service in docker-compose with proxy port mapping for every backend. The operational complexity does not justify the marginal realism gain for a local development and interview-demo project. The application-level wrapper tests the same code paths (exception handling, retry, circuit breaker) because those code paths are triggered by exceptions, not by network conditions directly.

3. **Always-on low-rate chaos in production.** Netflix's Chaos Monkey runs continuously against production services at a low rate. For this gateway, always-on chaos was rejected because the system has no redundancy beyond what it already implements (retry across backends). If the only backend for a model is hit by chaos, the request fails with no fallback. In a real multi-region deployment with per-region backend pools, low-rate production chaos would be safe. Here it would simply degrade service quality for the locust load test with no observability benefit beyond what the `make chaos` overlay provides.

4. **Chaos as a pytest plugin.** Instead of a runtime wrapper, implement chaos as a pytest fixture that monkeypatches `httpx.AsyncClient.post`. This would make chaos testing part of the CI pipeline rather than a manual `make chaos` + `make loadtest` workflow. Rejected because the value of chaos testing comes from running it against the real, containerized stack with all services running — not against mocked backends in pytest. The integration tests in `test_chaos_integration.py` cover the "chaos + circuit breaker + retry" interactions at the unit level; the Docker-based chaos testing covers the full-stack behavior that unit tests cannot.

### Failure modes and edge cases

- **All backends hit by chaos simultaneously.** The chaos wrapper does not discriminate by backend — every `post()` call has the same injection probability. If the gateway has 5 backends for a model and the retry loop tries 3 of them, the probability that all 3 fail at a 30% error rate is 0.30^3 = 2.7%. At a 90% error rate, it is 0.90^3 = 72.9%. When all retries fail, the client receives the last backend's error (500 or 504). The circuit breaker records failures for each attempted backend independently, which is correct behavior — if all backends are genuinely failing, all their breakers should trip.

- **Chaos + real failures compound.** If a real backend is down (e.g., Ollama container crashed) and chaos is also injecting errors, the effective failure rate is higher than configured. A backend that is genuinely returning 500s gets chaos 500s on top, accelerating circuit breaker trips. This is actually desirable: chaos should make real problems worse faster so they are detected sooner. But it means the observed error rate during chaos testing is a ceiling, not an exact measurement of the configured rate.

- **Stream context cleanup on injection.** When `_ChaosStreamContext.__aenter__` raises (timeout or error injection), `_real_ctx` is `None`. The `__aexit__` method handles this with `if self._real_ctx is not None`. Without this guard, the `async with` block would attempt to call `__aexit__` on `None`, raising `AttributeError` — an implementation bug masquerading as a backend error. This edge case is covered by `test_stream_aexit_skipped_on_injection`.

- **Latency injection exceeding the httpx client timeout.** The chaos system can inject up to 2000ms of latency (default `CHAOS_LATENCY_MAX_MS`). The httpx client has a 120-second timeout. The injected latency is added before the real backend call, so the total request time is `chaos_delay + backend_processing_time`. If both are large, the request can exceed client-side timeouts at the locust level (locust's default timeout is not explicitly set, relying on requests' defaults). This is a realistic scenario — real network latency stacks with backend processing time in the same way.

- **Deterministic seed does not guarantee identical cross-run behavior.** The seed makes `_roll_injection()` produce the same sequence of injection decisions for the same sequence of calls. But the sequence of calls depends on the retry loop: if request N triggers a retry (different from a pass-through), the total number of `_roll_injection()` calls differs from a run where request N passes through. This means the seed guarantees determinism within a single execution but not across executions with different retry outcomes. The unit tests verify same-sequence determinism; they do not verify cross-retry-path determinism.

- **No per-backend chaos rates.** The same `ChaosConfig` applies to all backends. If one backend is meant to be more reliable than another, chaos cannot model that asymmetry. A per-backend config would require mapping backend names to chaos rates, adding complexity to the configuration surface.

### Observability

**Structured logs (structlog):**
- `chaos_injection` — emitted by `_roll_injection()` on every injection event. Fields: `injection_type` ("timeout", "error_5xx", or "latency"), `url` (backend URL being called), `delay_ms` (for latency injections only). Log level is `warning` to distinguish chaos events from normal request flow at a glance.
- `chaos_mode_enabled` — emitted once during lifespan startup when `CHAOS_ENABLED=true`. Fields: `config` (full dict of ChaosConfig values). Confirms chaos is active and shows the effective rates.

**Admin endpoint — circuit breaker state:**
- `GET /admin/backends` returns each backend's circuit breaker snapshot: `state` (CLOSED/OPEN/HALF_OPEN), `error_rate`, `requests_in_window`, `current_cooldown_s`. Under chaos, this endpoint shows which backends have tripped and how fast they are recovering. Polling this during a `make loadtest` run gives a live view of how chaos is interacting with the circuit breaker state machine.

**Prometheus metrics:**
- `gateway_circuit_breaker_state{backend}` — gauge set to 0 (CLOSED), 1 (OPEN), or 2 (HALF_OPEN). Updated on every state transition via `CircuitBreaker._transition()`. Under chaos, this metric shows breakers oscillating between states as failures accumulate and cooldowns expire.
- `gateway_request_count_total{tenant, model, backend, status_code, method}` — counter. Under chaos, the `status_code` label distribution shifts: more 500/502/504 responses appear alongside 200s. Comparing the status code distribution with and without chaos quantifies the resilience improvement from retries.
- `gateway_request_latency_seconds{tenant, model, backend}` — histogram. Under chaos with latency injection, the histogram shows a bimodal distribution: a cluster at normal latency and a tail extending to `latency_max_ms`. This mirrors real-world latency distributions under network degradation.

**Locust reports:**
- `tests/load/report.html` — generated by `make loadtest`. Contains request/response time percentiles (p50, p95, p99), failure rate by endpoint, throughput over time, and request distribution across user classes. Under chaos, the report shows elevated p99 latency and non-zero failure rates, which can be compared against a baseline `make up` + `make loadtest` run to measure resilience effectiveness.

### Testing

**Unit tests — ChaosConfig (2 tests in `tests/unit/test_chaos.py`):**
- `test_defaults` — verifies default rates: error 0.10, timeout 0.05, latency 0.30, min/max latency 50/2000ms, seed None.
- `test_custom_values` — verifies custom config overrides are applied correctly.

**Unit tests — ChaosHttpClient post (5 tests in `tests/unit/test_chaos.py`):**
- `test_passthrough_when_disabled` — all rates at 0, call passes through to the real client unchanged. Verifies that with chaos disabled, behavior is identical to a plain httpx.AsyncClient.
- `test_error_injection` — error_rate=1.0, raises `HTTPStatusError` with status 500. Verifies the real client is never called.
- `test_timeout_injection` — timeout_rate=1.0, raises `ReadTimeout`. Verifies the real client is never called.
- `test_latency_injection` — latency_rate=1.0 with fixed 100ms min/max, verifies elapsed time >= 90ms (with tolerance) and the real client is still called after the delay.
- `test_timeout_priority_over_error` — both timeout_rate and error_rate at 1.0, verifies `ReadTimeout` is raised (timeout checked first).

**Unit tests — ChaosHttpClient stream (4 tests in `tests/unit/test_chaos.py`):**
- `test_stream_passthrough` — rates at 0, stream delegates to real client context manager.
- `test_stream_error_injection` — error_rate=1.0, raises `HTTPStatusError` during `__aenter__`.
- `test_stream_timeout_injection` — timeout_rate=1.0, raises `ReadTimeout` during `__aenter__`.
- `test_stream_aexit_skipped_on_injection` — verifies `__aexit__` handles `None` `_real_ctx` without error after injection fires.

**Unit tests — delegation and determinism (4 tests in `tests/unit/test_chaos.py`):**
- `test_getattr_delegation` — attribute access on ChaosHttpClient delegates to real client.
- `test_seed_determinism` — same seed produces identical injection sequence across two independent instances.
- `test_different_seeds_differ` — different seeds produce different sequences.

**Unit tests — _roll_injection (5 tests in `tests/unit/test_chaos.py`):**
- `test_no_injection_when_all_rates_zero` — 100 calls, all return None.
- `test_always_timeout_when_rate_one` — 10 calls, all return "timeout".
- `test_always_error_when_rate_one` — 10 calls, all return "error".
- `test_always_latency_when_rate_one` — 10 calls, all return float.
- `test_latency_within_bounds` — 50 calls, all latency values fall within min/max range.

**Integration tests — chaos + full stack (8 tests in `tests/unit/test_chaos_integration.py`):**
- `TestChaosStructuredResponses` (2 tests): 20 requests with 30% error + 10% timeout chaos, every response is valid JSON with either `choices` (200) or `detail` (error). Second test verifies 200 responses have correct shape (choices array, message, usage).
- `TestChaosCircuitBreaker` (2 tests): 30 requests with 90% error rate, verifies at least one backend's circuit breaker transitions to OPEN or HALF_OPEN via `/admin/backends`. Second test with 70% error rate verifies non-zero error rates in backend snapshots.
- `TestChaosRateLimiter` (2 tests): 30 requests with 20% error + 5% timeout, all status codes are in the allowed set {200, 429, 500, 502, 503, 504}. Second test with 80% error rate verifies error responses have the `detail` key.
- `TestChaosRetryFailover` (2 tests): 15 requests with 30% error rate, verifies >= 5 successes (P(all 3 retries fail) = 0.3^3 = 2.7% per request). Second test with 100% error rate verifies total failure still returns structured error with `detail`.

**Load tests — locust (manual, Docker-based):**
- `make loadtest` against `make up` (no chaos) — establishes baseline latency and throughput.
- `make loadtest` against `make chaos` — measures degradation under fault injection.
- Comparison of the two HTML reports shows the effect of chaos on p50/p95/p99 latency, failure rate, and throughput.

### Production gaps

- **No Kubernetes pod kill or node drain.** The chaos system only injects faults at the HTTP client level within the gateway process. It cannot simulate pod evictions, node failures, or container OOM kills. A production chaos system would use tools like Chaos Mesh or Litmus Chaos to inject infrastructure-level faults that test the orchestration layer (pod rescheduling, health check failures, rolling restart behavior).

- **No network partition simulation.** The gateway and its backends run on the same Docker network. There is no way to simulate a network partition between the gateway and Redis, or between the gateway and a specific backend. Tools like Toxiproxy or `tc netem` (on Linux) would be needed to model partial connectivity — a critical failure mode where the gateway can reach some backends but not others.

- **No persistent failure memory.** The chaos system is stateless per request — each `_roll_injection()` call is independent. It cannot simulate scenarios like "backend X is down for 5 minutes then recovers" or "Redis becomes unreachable for 30 seconds." Stateful chaos would require a time-based failure schedule or an external controller that flips failure modes during the test. This matters because circuit breaker recovery (HALF_OPEN probe) behaves differently under sustained failure vs. intermittent failure.

- **No mid-stream failure injection.** Chaos fires in `__aenter__`, never during stream iteration. A real backend can fail mid-response: TCP connection reset after 3 chunks, HTTP/2 stream error after headers are sent, or a backend that streams 90% of the response then crashes. Testing this would require wrapping the async iterator returned by `__aenter__` and injecting faults during `__anext__()`. This is the most impactful production gap for streaming-heavy workloads.

- **No gradual rollout of chaos.** Chaos is either on or off for all requests. A production system would support canary chaos: inject faults into 1% of requests, observe metrics, then ramp to 5%, 10%, etc. This requires either per-request sampling with dynamic rate adjustment or a control plane that updates chaos configuration at runtime.

- **No per-backend chaos rates.** A single `ChaosConfig` applies uniformly to all backends. In reality, different backends have different reliability profiles — a self-hosted Ollama instance is more likely to fail than a managed OpenAI API. Supporting per-backend chaos rates would require mapping backend names to individual `ChaosConfig` instances, which adds configuration complexity but enables more realistic failure simulation.

- **No chaos dashboard.** The chaos injection logs are interleaved with normal request logs. There is no dedicated view showing injection rate over time, injection type distribution, or correlation between injections and circuit breaker state changes. A Grafana dashboard with panels for `chaos_injection` log rate, breaker state gauge, and error code distribution would make chaos testing observable at a glance.

### Interview talking points

- The decision to wrap `httpx.AsyncClient` rather than inject at the middleware layer is the single most important design choice. Middleware injection tests whether the client handles errors — useful but trivial. HTTP client wrapper injection tests whether the gateway's resilience mechanisms (retry, circuit breaker, failover) actually work under the failure conditions they were designed for. The wrapper exercises the translator's exception handler, the retry loop's backend rotation, and the circuit breaker's failure counting — the same code path that runs when a real backend returns a 500 or times out. This is the difference between testing error display and testing error recovery.

- The `_ChaosStreamContext` implementation reveals a subtle resource management issue. When an exception is raised in `__aenter__`, the `async with` protocol still calls `__aexit__`. If the stream context naively forwarded to `self._real_ctx.__aexit__()` without checking for `None`, it would crash with an `AttributeError` — turning a simulated backend failure into an internal gateway error. This is the kind of bug that only surfaces under failure conditions and is invisible in happy-path testing. The chaos system both tests for this and documents it.

- Seeded randomness for chaos injection provides a debugging superpower: "run with seed=42, the failure happens on the 7th request every time." This turns non-deterministic failures into deterministic reproductions. The implementation uses `random.Random(seed)` (instance-level, not the global `random` module) so chaos injection does not affect random number generation elsewhere in the process. This is a general principle: testing infrastructure should be deterministic and isolated, even when simulating non-deterministic conditions.

### Likely interview questions

**Q: "Why inject chaos at the application level instead of using network-level tools like Toxiproxy?"**
**A:** Both test different things. Network-level tools (Toxiproxy, `tc netem`) inject faults below the HTTP layer — they simulate packet loss, connection resets, and bandwidth limits that the HTTP client library must handle. Application-level injection tests the gateway's own resilience code: retry logic, circuit breaker state transitions, failover backend selection, and error response formatting. For this project, the application-level approach was chosen because: (1) it works identically on macOS and Linux (no `NET_ADMIN` capability needed), (2) it requires no additional services or port remapping in docker-compose, and (3) the gateway's httpx client already handles network-level errors (converting them to exceptions), so application-level injection triggers the same exception-handling code paths. The trade-off is that we cannot test httpx's own connection pool behavior under network degradation — but that is httpx's responsibility, not the gateway's.

**Q: "How do you know your chaos testing is actually exercising the circuit breaker, not just generating errors that bypass it?"**
**A:** The integration test `test_circuit_breaker_trips_under_chaos` proves it directly. It sends 30 requests with a 90% chaos error rate, then queries `GET /admin/backends` and asserts that at least one backend's circuit breaker state is OPEN or HALF_OPEN. If chaos were bypassing the circuit breaker (e.g., if errors were caught before reaching the breaker's `record_failure()` call), all breakers would remain CLOSED regardless of the error rate. The test also verifies non-zero `error_rate` in the breaker's snapshot, confirming that failures are being recorded in the rolling window. The `test_retry_loop_survives_partial_chaos` test complements this by verifying that with moderate chaos (30% error rate), the retry loop successfully finds healthy backends — which only works if the circuit breaker is correctly excluding tripped backends from the routing pool.

**Q: "What happens if you accidentally deploy with CHAOS_ENABLED=true in production?"**
**A:** Three safeguards exist. First, the chaos import is conditional (`if os.getenv("CHAOS_ENABLED", "false").lower() == "true"`) — it is not loaded by default. Second, enabling chaos emits a `chaos_mode_enabled` log at `warning` level with the full config, making it immediately visible in log aggregation. Third, the chaos configuration lives in `docker-compose.chaos.yml`, a separate override file that must be explicitly specified in the `docker compose -f` command — it is not part of the default `docker-compose.yaml` used by `make up`. That said, there is no runtime kill switch: if chaos is enabled, it stays enabled until the process restarts. A production-grade system would add a `/admin/chaos` toggle endpoint or read configuration from a feature flag service, allowing operators to disable chaos without restarting the gateway.

## Multi-Instance Gateway

### Why this exists

A single gateway instance is a single point of failure. If the process crashes, the entire inference API is unavailable. Deploying new code requires stopping the service, which means dropped requests during every deployment. There is no way to scale horizontally — throughput is bounded by one process on one host. Multi-instance deployment solves three problems simultaneously: fault tolerance (one instance dies, others continue serving), zero-downtime deploys (rolling restart cycles through instances one at a time), and horizontal scaling (more instances means more concurrent request capacity).

### How it works

1. Three gateway instances (`gateway-1`, `gateway-2`, `gateway-3`) run as separate Docker Compose services, all built from the same Dockerfile and sharing the same configuration. Each instance receives a unique `INSTANCE_ID` environment variable.
2. Nginx sits in front of all three instances as a reverse proxy and load balancer. The `upstream gateway_cluster` block lists all three instances with round-robin distribution — Nginx cycles through them sequentially, spreading load evenly.
3. Clients connect to Nginx on port 8080. Nginx forwards each request to one of the three gateway instances via `proxy_pass http://gateway_cluster`. The client is unaware of which instance handles its request.
4. All three instances share the same Redis server for state that must be consistent across the cluster: rate limiting (sorted set sliding windows), L2 semantic cache (hash keys), priority queue (sorted sets), and request journal (stream). This means a rate limit counter incremented by `gateway-1` is visible to `gateway-2` and `gateway-3` on the next request.
5. L1 cache (in-process LRU) and circuit breaker state are per-instance by design. Each instance maintains its own view of backend health based on the requests it handles. This is an intentional trade-off: sharing circuit breaker state via Redis would add latency to every request for a marginal consistency benefit, since all instances talk to the same backends and will converge on similar state within seconds.
6. Every response includes an `X-Instance-ID` header set in `request_id_middleware` in `gateway/main.py`. The value comes from `app.state.instance_id`, which is read from the `INSTANCE_ID` environment variable (falling back to `socket.gethostname()` if unset). This header enables request tracing through the load balancer — operators can correlate a specific response to a specific instance in logs.
7. Rolling restart (`scripts/rolling-restart.sh`) deploys new code without dropping requests. For each instance in sequence: (a) `docker compose stop` sends SIGTERM, triggering the graceful shutdown handler that sets `app.state.shutting_down = True` and drains in-flight requests for up to 10 seconds; (b) `docker compose up -d --build` rebuilds and starts the instance; (c) the script polls `/health` via `curl` inside the container, waiting up to 30 seconds for a 200 response; (d) only after the instance is confirmed healthy does the script move to the next instance. During each restart, Nginx's passive health checks detect the stopped instance (connection refused counts toward `max_fails`) and route traffic to the remaining two.
8. If any instance fails to become healthy within the 30-second timeout, the script aborts immediately with `exit 1`. This fail-fast behavior prevents a cascading failure where a bad build takes down all three instances sequentially.

### Implementation

**Docker Compose (`docker-compose.yaml`):**
- `x-gateway-env` YAML anchor (`&gateway-env`) defines all shared environment variables once: `CONFIG_PATH`, `LOG_LEVEL`, tenant keys, provider keys, Redis URL, cache/queue tuning parameters.
- `x-gateway-common` YAML anchor (`&gateway-common`) defines shared service configuration: build context and target, `stop_grace_period: 15s`, memory limit (2G), config volume mount (read-only), dependency declarations on Ollama/Redis/mock backends, and `restart: unless-stopped`.
- Each gateway service (`gateway-1`, `gateway-2`, `gateway-3`) uses `<<: *gateway-common` for merge and `<<: *gateway-env` for environment, adding only the unique `INSTANCE_ID` and `container_name`.

**Nginx (`nginx/nginx.conf`):**
- `upstream gateway_cluster` with three servers, each configured with `max_fails=3 fail_timeout=10s`. After 3 consecutive failures, Nginx marks the upstream as unavailable for 10 seconds before retrying.
- `proxy_buffering off` disables response buffering — essential for SSE streaming where tokens must reach the client immediately as the LLM generates them. Without this, Nginx would buffer the entire response before forwarding, defeating the purpose of streaming.
- `proxy_http_version 1.1` with `Connection ''` enables HTTP keep-alive between Nginx and the gateway instances, avoiding per-request TCP handshake overhead.
- Timeouts tuned for LLM workloads: `proxy_connect_timeout 5s` (fail fast if instance is down), `proxy_read_timeout 120s` (LLM inference can be slow), `proxy_send_timeout 30s` (request bodies are small).
- `/nginx-health` endpoint returns 200 directly from Nginx without proxying — used to verify Nginx itself is running.

**Instance identification (`gateway/main.py`):**
- `app.state.instance_id = os.getenv("INSTANCE_ID", socket.gethostname())` set during lifespan startup.
- `request_id_middleware` adds `X-Instance-ID: {instance_id}` to every response, including health checks, chat completions, and admin endpoints.

**Graceful shutdown (`gateway/main.py`):**
- SIGTERM handler sets `app.state.shutting_down = True`.
- Middleware checks `shutting_down` on every request. If true, non-health/metrics endpoints receive 503 with `Retry-After: 5` header, telling clients and load balancers to retry elsewhere.
- In-flight request tracking via `app.state.inflight_count` with an `asyncio.Lock` and `asyncio.Event`. On shutdown, the lifespan waits up to 10 seconds for in-flight requests to drain before closing connections.
- `stop_grace_period: 15s` in Docker Compose gives the container 15 seconds after SIGTERM before SIGKILL — enough for the 10-second drain plus cleanup.

**Rolling restart (`scripts/rolling-restart.sh`):**
- `set -euo pipefail` — any command failure aborts the script immediately.
- Accepts `--build` flag (passed by `make rolling-restart`) to rebuild images before starting.
- Iterates through `gateway-1 gateway-2 gateway-3` sequentially.
- Health probe uses `docker compose exec -T` to run `curl` inside the container, avoiding host networking assumptions. The `-T` flag disables TTY allocation for non-interactive use.
- 30-second health timeout with 1-second polling interval.

**Makefile:**
- `make rolling-restart` runs `scripts/rolling-restart.sh --build`.

### Key design decisions

1. **Nginx over HAProxy or Traefik.** Nginx is the simplest option for this use case: a static upstream list with round-robin distribution and passive health checks. HAProxy offers more sophisticated load balancing algorithms (least connections, weighted) but requires more configuration. Traefik's service discovery features are unnecessary when the upstream list is static in Docker Compose. Nginx's `proxy_buffering off` directive is a single line that solves SSE streaming — achieving the same in HAProxy requires `option http-no-delay` plus careful tuning.

2. **YAML anchors for DRY compose configuration.** Without anchors, the three gateway services would require 60+ lines of duplicated configuration. A change to the memory limit, a new environment variable, or a new dependency would require editing three services identically. The `x-gateway-common` and `x-gateway-env` anchors reduce this to a single source of truth. The `<<:` merge key combines the anchor with service-specific overrides (like `INSTANCE_ID`).

3. **Passive health checks (open-source Nginx limitation).** Nginx Plus offers active health checks that probe backends on a schedule, detecting failures before client requests arrive. The open-source version only supports passive checks: Nginx counts failures observed during real request forwarding. This means the first `max_fails` requests after a backend goes down will fail before Nginx marks it unavailable. For a 3-instance cluster, this is acceptable — at most 3 requests hit a dead instance before Nginx routes around it. The `fail_timeout=10s` recovery window is short enough to detect a restarted instance quickly.

4. **Per-instance circuit breakers (no shared state via Redis).** Circuit breakers track backend health based on recent failure rates. Sharing this state across instances via Redis would mean every request incurs a Redis round-trip for breaker state lookup and update — adding latency to the critical path. Since all instances talk to the same backends, their circuit breakers converge independently within a few seconds of failure onset. The worst case is that instance A trips its breaker on backend X while instance B is still sending a few more requests to backend X — a brief window of extra failures, not a correctness issue.

5. **Rolling restart aborts on first failure.** If `gateway-2` fails to become healthy after rebuild, the script exits immediately rather than continuing to `gateway-3`. This prevents a scenario where a bad build takes down the entire cluster. The operator investigates `gateway-2`, fixes the issue, and re-runs the rolling restart. The alternative — skip the failed instance and continue — risks deploying a known-bad build to more instances.

6. **`stop_grace_period: 15s` exceeds the 10-second drain timeout.** The drain timeout in `gateway/main.py` waits 10 seconds for in-flight requests to complete. Docker's `stop_grace_period` gives 15 seconds after SIGTERM before sending SIGKILL. This 5-second buffer ensures the Python process has time to close Redis and httpx connections after draining, rather than being killed mid-cleanup.

### Alternatives considered

1. **HAProxy.** More powerful load balancing algorithms (least connections, weighted round-robin) and built-in active health checks. Rejected because the configuration complexity is not justified for a 3-instance static upstream. HAProxy's SSE streaming support requires more tuning than Nginx's single `proxy_buffering off` directive.

2. **Traefik.** Automatic service discovery via Docker labels, built-in Let's Encrypt, and a dashboard. Rejected because the service list is static (defined in docker-compose.yaml), TLS is not needed for local development, and Traefik's label-based configuration is harder to reason about than Nginx's explicit upstream block.

3. **Shared circuit breaker state via Redis.** Would provide cluster-wide consistency: if one instance trips a breaker, all instances immediately stop sending to that backend. Rejected because: (a) every request would need a Redis read for breaker state and a Redis write for failure recording, adding 1-2ms latency to the critical path; (b) distributed breaker state introduces its own failure mode — if Redis is unavailable, breaker state is lost; (c) independent breakers converge within seconds, which is fast enough for a proxy where requests arrive continuously.

4. **Nginx Plus active health checks.** Would detect backend failures proactively by polling `/health` endpoints on a schedule, removing backends before client requests fail. Rejected because Nginx Plus is a commercial product. The passive health check trade-off (up to `max_fails` requests fail before Nginx detects the issue) is acceptable for 3 instances with `max_fails=3`.

5. **Kubernetes Deployment instead of Docker Compose.** Provides built-in rolling updates, readiness probes, horizontal pod autoscaler, and service discovery. Rejected because the project targets local development with `docker compose up`. The rolling restart script demonstrates the same concepts (health-gated sequential restart) at a smaller scale. Migration to Kubernetes is a deployment concern, not an architecture change.

### Failure modes and edge cases

- **All three instances down simultaneously.** Nginx returns 502 Bad Gateway for every request. No automatic recovery — requires operator intervention to restart instances. The `/nginx-health` endpoint still returns 200, so monitoring can distinguish between "Nginx is down" and "all upstreams are down."

- **Nginx upstream pool exhausted.** If all three backends exceed `max_fails` within `fail_timeout`, Nginx marks all upstreams as unavailable. On the next request, Nginx resets and tries all backends again (round-robin through the "failed" list). This means Nginx does not permanently blackhole traffic — it retries after the fail window expires.

- **Rolling restart during active failure.** If a backend (e.g., Ollama) is down and a rolling restart begins, the restarted instances inherit the same backend failure. Circuit breakers on restarted instances start in CLOSED state (fresh process), so they will re-discover the failure independently. This is a brief window of increased error rate until breakers trip again.

- **L1 cache divergence after restart.** When an instance restarts, its in-process L1 cache is empty. The instance will experience a cold-cache period with higher L2 (Redis) hit rates until the L1 warms up. This is transient and self-correcting — no operator action needed.

- **Split-brain rate limiting.** Rate limits are enforced via Redis (shared), so there is no split-brain for rate limiting. However, if Redis becomes unreachable, all instances degrade gracefully — rate limiting is disabled rather than enforced locally, which means a Redis failure temporarily removes rate limiting protection.

- **Health check race during restart.** The rolling restart script polls `/health` inside the container via `docker compose exec`. There is a brief window after `docker compose up -d` where the container is running but the FastAPI process has not yet bound to port 8080. The `curl` probe fails during this window and retries on the next 1-second interval. The 30-second timeout is generous enough to accommodate slow startups (e.g., Redis connection, config parsing).

- **SIGTERM during long-running LLM request.** If a gateway instance receives SIGTERM while proxying a 60-second LLM inference call, the graceful shutdown handler sets `shutting_down = True` and the drain waits up to 10 seconds. If the LLM call takes longer than 10 seconds to complete, the drain times out, and Docker sends SIGKILL 5 seconds later. The client receives a connection reset. This is the expected trade-off: the 15-second grace period cannot accommodate arbitrarily long LLM calls without blocking the rolling restart indefinitely.

### Observability

**Response headers:**
- `X-Instance-ID` on every response identifies which gateway instance handled the request. Combined with `X-Request-ID`, this enables full request tracing through the load balancer layer.

**Structured logs:**
- `gateway_started` event includes `instance_id`, `backends` count, and `tenants` count — one log line per instance on startup.
- `sigterm_received` logged when graceful shutdown begins.
- `draining_inflight` with count logged when waiting for in-flight requests.
- `drain_complete` or `drain_timeout` logged when drain finishes or times out.
- `gateway_stopped` logged after all connections are closed.

**Prometheus metrics:**
- All three instances expose `/metrics` independently. Prometheus scrape config targets all three instances, so metrics are labeled per-instance. Grafana dashboards can filter or aggregate by instance.

**Nginx:**
- `/nginx-health` returns 200 if Nginx is running, independent of upstream health.
- Nginx access logs (stdout by default in the `nginx:alpine` image) show upstream response times and status codes.

**Rolling restart output:**
- The script prints each step: stopping, starting, waiting for health, success/failure. On failure, it prints the instance name and abort message, providing a clear indication of which instance failed.

### Testing

**Unit/integration tests (`tests/integration/test_multi_instance.py`, 5 tests):**
- `TestMultiInstanceHeaders` (4 tests): Verifies `X-Instance-ID` header is present on chat completion responses, health endpoint responses, and admin endpoint responses. Each test sets a different `INSTANCE_ID` via `monkeypatch.setenv` and asserts the header value matches. A fourth test iterates through all three instance IDs to verify each instance reports its own ID.
- `TestSharedState` (2 tests): Architecture validation tests that confirm the rate limiter and L2 semantic cache are Redis-based (shared state) rather than in-process (per-instance state). These are structural assertions — the real multi-instance sharing is validated by Docker E2E tests.

**E2E validation (Docker-based):**
- `make up` starts all three gateway instances, Nginx, Redis, and backend mocks.
- Requests to `http://localhost:8080/v1/chat/completions` are distributed across instances by Nginx round-robin.
- The `X-Instance-ID` response header confirms which instance handled each request — sending multiple requests shows different instance IDs in the responses.
- `make rolling-restart` performs a full rolling restart cycle, verifying zero-downtime deployment capability.

**Load testing (Locust):**
- `make loadtest` against the multi-instance cluster exercises Nginx load balancing under sustained traffic. The report shows request distribution and latency percentiles across the instance pool.

### Production gaps

- **No active health checks.** Nginx open-source only supports passive health checks. Up to 3 requests may fail before Nginx routes around a dead instance. A production deployment would use Nginx Plus, HAProxy, or a cloud load balancer with active health probes that detect failures before they impact client traffic.

- **No autoscaling.** The instance count (3) is static in docker-compose.yaml. A production deployment would use Kubernetes Horizontal Pod Autoscaler or cloud auto-scaling groups to adjust instance count based on CPU, memory, or request queue depth.

- **No sticky sessions.** Round-robin distribution means sequential requests from the same client may hit different instances. This is fine for stateless chat completions, but if the gateway adds features like conversation context or session state, sticky sessions (via IP hash or cookie-based affinity) would be needed.

- **No connection draining at the Nginx level.** When an upstream is removed, Nginx drops existing connections immediately. A production setup would use `drain` directives (Nginx Plus) or a service mesh sidecar (Envoy) to finish in-flight requests on the old instance before removing it from the pool.

- **No blue-green or canary deployments.** Rolling restart deploys the same build to all instances sequentially. There is no mechanism to route a percentage of traffic to a new version while keeping the old version running. A production system would use weighted upstream groups or a service mesh traffic split.

- **No TLS between Nginx and gateway instances.** Traffic between Nginx and the gateway instances is unencrypted on the Docker network. In production, this would use mTLS or a service mesh for internal encryption.

### Interview talking points

- The YAML anchor pattern (`x-gateway-common`, `x-gateway-env`) is a practical DRY technique that eliminates configuration drift between instances. When a new environment variable is added, it goes in one place. The `<<:` merge key with per-service overrides (like `INSTANCE_ID`) demonstrates the YAML merge key specification — a feature many developers are unaware of. The trade-off is readability: someone unfamiliar with YAML anchors must trace the `*gateway-common` reference to understand what a service inherits.

- The graceful shutdown sequence (SIGTERM -> reject new requests with 503 + Retry-After -> drain in-flight -> close connections -> SIGKILL after grace period) is the standard pattern used by Kubernetes, ECS, and other orchestration systems. The gateway implements it explicitly so the rolling restart script can depend on it. The 503 with `Retry-After` header is important: it tells the load balancer and clients to retry on another instance rather than treating the response as a permanent error.

- Per-instance circuit breakers are an intentional architectural choice, not a simplification. Sharing breaker state via Redis makes every request depend on Redis for the critical-path decision "should I send to this backend." If Redis is slow or unavailable, every request pays the latency penalty or loses breaker protection. Independent breakers mean each instance is self-sufficient for routing decisions, and the worst case of state divergence is a few extra failed requests during the convergence window — seconds, not minutes.

### Likely interview questions

**Q: "Why not use Kubernetes instead of Docker Compose with a rolling restart script?"**
**A:** The rolling restart script demonstrates the same fundamental concepts as a Kubernetes rolling update: health-gated sequential replacement, graceful shutdown with drain, and abort on failure. Docker Compose is the appropriate tool for local development and demonstration. Migrating to Kubernetes would replace the script with a Deployment spec (`strategy: RollingUpdate`, `maxUnavailable: 1`, `readinessProbe`), but the gateway code is unchanged — the SIGTERM handler, health endpoint, and graceful drain work identically in both environments. The script makes the deployment mechanics explicit and inspectable, which is valuable for understanding what Kubernetes automates.

**Q: "What happens if Redis goes down while all three instances are running?"**
**A:** All three instances degrade gracefully and identically. Rate limiting is disabled (requests pass through unchecked), L2 cache is unavailable (every request goes to the backend), the priority queue is unavailable (requests are processed immediately without queuing), and the request journal stops recording. L1 cache and circuit breakers continue working because they are in-process. The gateway remains functional for its primary purpose (proxying LLM requests) — it loses rate limiting, caching, and queuing but does not crash or reject requests. When Redis recovers, all three instances reconnect and resume shared state operations.

**Q: "How do you handle the L1 cache being different on each instance?"**
**A:** L1 cache divergence is expected and acceptable. Each instance caches responses it has seen based on its own traffic. With round-robin distribution, each instance sees roughly one-third of total traffic, so L1 hit rates are lower per-instance than they would be with a single instance. However, the L2 cache in Redis is shared, so a cache miss in instance A's L1 may still hit instance B's L2 entry. The net effect is that L1 provides a fast-path optimization (no Redis round-trip) for repeated requests to the same instance, while L2 provides cluster-wide cache coverage. This two-tier design means adding instances does not reduce cache effectiveness — it only changes the L1/L2 hit ratio.

## Token-Level Streaming Analytics

### Why this exists

Total request latency (measured by `gateway_request_duration_seconds`) is a single number that conflates two fundamentally different phases of LLM inference: the time the model spends processing the prompt before producing any output, and the time it spends generating tokens sequentially. These phases have different causes, different optimization levers, and different user-facing impact.

**Time to first token (TTFT)** measures perceived responsiveness — how long the user stares at a blank screen before text begins appearing. A high TTFT indicates the model is slow to start, which could mean prompt processing is bottlenecked (long context window), the backend is overloaded (queuing before inference begins), or the network path to the backend is slow. TTFT is the single most important metric for user experience in streaming LLM applications.

**Inter-token latency (ITL)** measures generation consistency — the time between consecutive content tokens. Steady ITL means smooth text streaming; spiky ITL means visible stuttering. High ITL can indicate GPU memory pressure, KV cache eviction, or backend contention during generation. ITL is the metric that distinguishes "fast model, slow start" from "slow model, fast start."

**Generation duration** measures the total time from the first content token to the last, capturing the full generation phase independent of prompt processing time. Combined with token count, this yields effective tokens-per-second — the throughput metric that matters for capacity planning.

Without these three metrics, an operator seeing `gateway_request_duration_seconds` p99 = 8 seconds cannot distinguish between "the model takes 6 seconds to start but generates fast" (TTFT problem) and "the model starts instantly but generates slowly" (ITL problem). The remediation is completely different: the first requires prompt optimization or a faster backend; the second requires a faster model or more GPU memory.

### How it works

1. Client sends `POST /v1/chat/completions` with `"stream": true`. The gateway resolves the backend, acquires a concurrency slot, and begins proxying SSE chunks from the backend.

2. The raw SSE generator from the backend translator is wrapped in `_wrap_stream_with_analytics()`. This wrapper sits in the streaming chain between the circuit breaker wrapper (innermost, closest to the backend) and the latency tracker wrapper (next layer out). This position captures timing closest to actual backend behavior, excluding downstream processing overhead.

3. The wrapper records `start_time = time.perf_counter()` (passed from the caller at stream creation time) and iterates over incoming SSE chunks.

4. For each chunk that starts with `data: ` and is not `data: [DONE]`, the wrapper parses the JSON payload and checks if `choices[0].delta.content` is non-empty — this indicates a content token (as opposed to a role token, tool call, or stop signal).

5. On the **first content token**: the wrapper computes `ttft = now - start_time`, records it in the `gateway_ttft_seconds` histogram with `[model, backend]` labels, yields the original chunk, then yields an SSE comment `": ttft_ms=X\n\n"`. The SSE comment is a valid SSE line (starts with `:`) that compliant clients ignore but that can be observed in network traces and test assertions.

6. On **subsequent content tokens**: the wrapper computes `itl = now - last_content_time` and records it in the `gateway_itl_seconds` histogram. No additional SSE output is emitted for ITL — recording every ITL as an SSE comment would bloat the response.

7. When the `data: [DONE]` sentinel arrives: if at least two content tokens were received (meaning both `first_content_time` and `last_content_time` are set and differ), the wrapper computes `generation_duration = last_content_time - first_content_time` and records it in the `gateway_generation_duration_seconds` histogram.

8. All original SSE chunks pass through unmodified. The only addition to the stream is the single `": ttft_ms=X"` comment after the first content token.

### Implementation

**Prometheus histograms** (`gateway/observability/metrics.py`):

- `gateway_ttft_seconds` — Time to first content token. Labels: `[model, backend]`. Buckets: `[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]`. The bucket range covers sub-10ms local inference through 10-second cold starts. The lower buckets (10ms, 25ms) are relevant for GPU-accelerated backends; the upper buckets (5s, 10s) capture CPU inference and overloaded backends.

- `gateway_itl_seconds` — Inter-token latency. Labels: `[model, backend]`. Buckets: `[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]`. Tighter bucket range than TTFT because ITL is typically much shorter — a healthy GPU backend produces tokens at 5-50ms intervals. The 1-second upper bound captures severely degraded generation.

- `gateway_generation_duration_seconds` — Duration from first to last content token. Labels: `[model, backend]`. Buckets: `[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]`. Wide range to accommodate short completions (a few tokens) through long generations (multi-paragraph responses on CPU).

**Streaming wrapper** (`gateway/routes/chat.py: _wrap_stream_with_analytics`):

- Async generator that wraps the upstream generator, maintaining `first_content_time` and `last_content_time` state across chunks.
- JSON parsing is wrapped in `except (json.JSONDecodeError, IndexError, KeyError): pass` — malformed chunks are yielded through without recording metrics, ensuring the stream is never broken by a parse failure.
- The SSE comment format `": ttft_ms=X\n\n"` uses the SSE comment syntax (line starting with `:`). The `\n\n` suffix follows the SSE event boundary convention.

**Streaming chain insertion** (`gateway/routes/chat.py`, streaming path):

```
raw_gen (backend translator)
  -> _wrap_stream_with_circuit_breaker  (CB success/failure recording)
  -> _wrap_stream_with_analytics        (TTFT/ITL/generation duration)
  -> _wrap_stream_with_latency          (overall request latency)
  -> _tee_stream_for_cache              (buffer + cache store)
  -> _wrap_stream_with_slot_release     (concurrency slot release)
  -> _wrap_stream_with_journal          (journal + token metrics)
```

**Grafana dashboard** (`grafana/provisioning/dashboards/streaming-analytics.json`):

Six panels:
1. **TTFT P50/P95/P99** — time series of `histogram_quantile` over `gateway_ttft_seconds_bucket`.
2. **ITL P50/P95/P99** — time series of `histogram_quantile` over `gateway_itl_seconds_bucket`.
3. **Generation Duration by Model** — time series of generation duration percentiles, split by model.
4. **Effective Tokens/Second** — derived metric: `rate(gateway_generation_duration_seconds_count)` / `rate(gateway_generation_duration_seconds_sum)` gives average tokens per second throughput.
5. **TTFT Heatmap** — histogram heatmap showing TTFT distribution density over time, revealing bimodal distributions (e.g., cold vs warm model loads).
6. **ITL Heatmap** — histogram heatmap showing ITL distribution density, revealing generation stuttering patterns.

### Key design decisions

1. **SSE comment for TTFT, not an HTTP header.** In a streaming response, HTTP headers are sent before the body begins — by the time the first content token arrives, the header phase is long over. An HTTP response header cannot carry TTFT because the value is not known when headers are written. An SSE comment (line starting with `:`) is the correct mechanism: it is part of the event stream body, it is ignored by compliant SSE clients (per the W3C spec), and it can be emitted at exactly the right time — immediately after the first content token.

2. **Labels `[model, backend]` only, not `[model, backend, tenant]`.** Adding `tenant` as a label would create a cardinality explosion: `N_tenants * N_models * N_backends` time series per histogram, each with 8-10 buckets. For 100 tenants, 5 models, and 3 backends, that is 1,500 time series per metric, times 10 buckets, times 3 metrics = 45,000 series. The `[model, backend]` labels keep cardinality at `5 * 3 = 15` series per metric — manageable by any Prometheus instance. Per-tenant streaming analytics can be derived from logs or a dedicated analytics pipeline.

3. **Wrapper positioned between circuit breaker and latency tracker.** The analytics wrapper measures raw backend streaming performance. Placing it inside the circuit breaker wrapper means the CB has already started recording; placing it outside the latency wrapper means latency measurement includes the full pipeline. The analytics wrapper captures the timing that matters: when the backend actually produced content tokens, not when they were processed by outer layers.

4. **AI-specific histogram bucket ranges.** Standard HTTP latency buckets (50ms-60s) are wrong for LLM streaming. TTFT needs sub-25ms buckets to distinguish fast GPU inference from slow CPU inference. ITL needs sub-10ms buckets because GPU-accelerated generation produces tokens at 5-50ms intervals. Generation duration needs buckets up to 120 seconds for long completions on slow hardware. The bucket choices were informed by observed latency distributions from Ollama (CPU), vLLM (GPU), and OpenAI API (cloud).

5. **Single SSE comment, not per-token comments.** Emitting a comment for every ITL observation would add significant overhead to the response body (one extra line per token, potentially hundreds of extra lines). TTFT is the one metric that benefits from in-stream visibility (for client-side measurement and debugging). ITL is recorded server-side in Prometheus and does not need to be visible in the stream.

### Alternatives considered

1. **HTTP trailer headers.** HTTP trailers are sent after the response body completes — they could carry TTFT, ITL statistics, and generation duration. However, HTTP trailer support is inconsistent: many HTTP/1.1 clients ignore trailers, most proxy servers strip them, and the `fetch()` API in browsers does not expose them. Trailers would solve the "headers are sent too early" problem but introduce a "most clients cannot read trailers" problem.

2. **Custom SSE event type (`event: metrics`).** A dedicated SSE event with a JSON payload containing all streaming metrics would be clean and structured. However, SSE clients that do not register a handler for the `metrics` event type will fire `onerror` in some implementations. This breaks the "transparent proxy" contract — clients expecting only OpenAI-format events would receive unexpected event types. An SSE comment is the safe choice: it is defined by the SSE spec as ignorable.

3. **Response header with TTFT (set before body).** Impossible by definition. For streaming responses, headers are flushed before the first byte of the body. TTFT is not known until the first content token arrives, which is part of the body. This alternative is architecturally infeasible for SSE/chunked-transfer responses.

4. **Separate metrics endpoint for streaming stats.** Recording metrics only in Prometheus (no in-stream signal) was considered. This works for server-side monitoring but provides no client-side visibility. The SSE comment is a low-cost addition that enables client-side TTFT measurement without modifying the data format.

### Failure modes and edge cases

- **No content tokens in the stream.** If the model returns only a role delta and a stop signal without any `content` in the delta, `first_content_time` remains `None` and no TTFT, ITL, or generation duration metrics are recorded. This is correct behavior — there is nothing to measure. This can happen with function-calling responses where the model returns a tool call instead of text content.

- **Single content token.** TTFT is recorded. ITL is not recorded (there is no inter-token interval with only one token). Generation duration is recorded as 0 (or very near 0) because `first_content_time == last_content_time`. The `data: [DONE]` handler checks that both `first_content_time` and `last_content_time` are set, and since they are the same reference point, `gen_dur` is 0. This is technically correct but may skew generation duration histograms — a production system might skip recording duration for single-token responses.

- **JSON parse failure on a chunk.** The `except (json.JSONDecodeError, IndexError, KeyError): pass` block ensures that a malformed chunk does not break the stream or the metrics. The chunk is yielded through unmodified. This means a corrupt chunk that actually contained content will not be counted in TTFT/ITL — an acceptable trade-off versus crashing the stream.

- **Backend sends content in the role chunk.** Some backends (non-standard) might include `content` in the same delta as `role`. The wrapper treats this as the first content token, which is technically correct — it is the first chunk with content. The TTFT measurement starts from stream creation, so it captures the full prompt-processing time regardless of which delta field accompanies the content.

- **Very fast generation (sub-millisecond ITL).** If the backend (or a mock in tests) produces tokens faster than `time.perf_counter()` resolution, ITL values may be 0.0. These are still valid histogram observations and accumulate in the lowest bucket. This is expected in test environments and with cached/mock responses.

- **Stream error mid-generation.** If the backend stream errors after some content tokens have been produced, the circuit breaker wrapper (inner layer) records the failure. The analytics wrapper has already recorded TTFT and some ITL observations. Generation duration is not recorded because `data: [DONE]` is never received. This means partial generations produce partial metrics — TTFT and ITL reflect what actually happened, and the absence of a generation duration observation correctly indicates an incomplete generation.

### Observability

**Prometheus metrics (3 histograms):**
- `gateway_ttft_seconds{model, backend}` — one observation per streaming request that produces at least one content token.
- `gateway_itl_seconds{model, backend}` — N-1 observations per streaming request that produces N content tokens.
- `gateway_generation_duration_seconds{model, backend}` — one observation per streaming request that produces at least two content tokens and completes with `[DONE]`.

**SSE comment:**
- `": ttft_ms=X"` appears in the event stream after the first content chunk. Visible in browser DevTools network tab, `curl` output, and test assertions. Value is in milliseconds (human-readable), rounded to 2 decimal places.

**Grafana dashboard ("Streaming Analytics"):**
- TTFT percentile trends (P50/P95/P99) — detect prompt processing regressions.
- ITL percentile trends (P50/P95/P99) — detect generation stuttering.
- Generation duration by model — compare generation speed across models.
- Effective tokens/second — derived throughput metric for capacity planning.
- TTFT and ITL heatmaps — reveal distribution shape (unimodal vs bimodal), which percentile charts obscure.

### Testing

**Unit tests (`tests/unit/test_streaming_analytics.py`, 7 tests):**
- `test_ttft_metric_recorded` — verifies `gateway_ttft_seconds_count` is incremented after consuming a stream with content tokens.
- `test_sse_comment_after_first_content` — verifies the SSE comment `": ttft_ms=X"` appears immediately after the first content chunk, with a valid non-negative float value.
- `test_itl_between_content_tokens` — verifies `gateway_itl_seconds_count` has 2 observations for a stream with 3 content tokens (a->b and b->c intervals).
- `test_generation_duration_at_done` — verifies `gateway_generation_duration_seconds_count` is incremented when `data: [DONE]` is received after content tokens.
- `test_no_content_no_metrics` — verifies no TTFT metric is recorded when the stream contains only role and stop deltas with no content.
- `test_original_chunks_preserved` — verifies all original SSE chunks pass through the wrapper unmodified, with exactly one additional SSE comment.
- `test_single_content_token` — verifies TTFT is recorded but ITL is not when the stream contains only one content token.

**Integration tests (`tests/integration/test_streaming.py`, 2 tests in `TestStreamingAnalytics`):**
- `test_stream_contains_ttft_comment` — full-stack test through FastAPI with mocked backend translator, asserting the SSE comment is present in the response body with a valid TTFT value.
- `test_ttft_comment_is_valid_sse` — verifies the TTFT comment is valid SSE syntax (starts with `:`) and that all original data lines are preserved alongside the comment.

**E2E validation (Docker):**
- `make test` runs all unit and integration tests inside Docker, including the streaming analytics tests.
- Manual verification: `curl -N` to the streaming endpoint shows the `": ttft_ms=X"` comment in the SSE output, and the Prometheus `/metrics` endpoint shows populated `gateway_ttft_seconds`, `gateway_itl_seconds`, and `gateway_generation_duration_seconds` histograms.

### Production gaps

- **No per-tenant streaming analytics.** The `[model, backend]` label set excludes tenant identity. A production system with per-tenant SLAs on TTFT or ITL would need either higher-cardinality metrics (with careful label management) or a separate analytics pipeline that correlates streaming timing with tenant identity from logs.

- **No client-side ITL measurement.** Only TTFT is exposed in the SSE stream. Clients that want to measure ITL must implement their own timing between received chunks, which includes network jitter and client-side buffering — not purely backend ITL. A production system might expose a post-request analytics endpoint where clients report their observed timing.

- **No alerting rules.** The Grafana dashboard visualizes streaming metrics but no Prometheus alerting rules are defined for TTFT or ITL thresholds. A production system would alert on TTFT p95 exceeding an SLA threshold (e.g., 2 seconds) or ITL p95 exceeding a stuttering threshold (e.g., 200ms).

- **No token counting in the analytics wrapper.** The wrapper measures timing between content deltas but does not count tokens. The "effective tokens/second" Grafana panel derives throughput from duration histogram rates, which is an approximation. Accurate tokens-per-second requires counting tokens in the analytics wrapper, which would duplicate logic already in the journal wrapper.

- **No streaming analytics for cached responses.** When a cached response is streamed via `_stream_cached_response`, it bypasses the analytics wrapper. Cached streaming responses produce artificially fast TTFT/ITL that do not reflect real backend performance. This is correct for backend monitoring but means the dashboard underreports total streaming volume.

### Interview talking points

- TTFT and ITL are the two metrics that matter most for streaming LLM inference, yet most gateway implementations only measure total request duration. TTFT tells you if the model is slow to start (prompt processing, queue wait, cold start); ITL tells you if generation is consistent or stuttering (GPU contention, KV cache pressure, memory bandwidth). Separating these metrics enables targeted optimization — you fix TTFT problems differently than ITL problems.

- The SSE comment mechanism (`": ttft_ms=X"`) demonstrates understanding of the SSE specification. Comments (lines starting with `:`) are defined by the W3C Server-Sent Events spec as lines that must be ignored by the client. This makes them a safe channel for out-of-band metadata in a streaming response without modifying the data format or breaking client parsers. The alternative (HTTP headers) is architecturally impossible for streaming responses because headers are committed before the body begins.

- The histogram bucket ranges are intentionally tuned for LLM inference workloads, not generic HTTP traffic. TTFT buckets go down to 10ms (GPU inference can start producing tokens in under 25ms) and up to 10 seconds (CPU inference on large models). ITL buckets go down to 5ms (GPU token generation speed) and up to 1 second (severely degraded generation). Using default Prometheus buckets would put most observations in a single bucket, making percentile calculations meaningless.

- The wrapper chain ordering (circuit breaker -> analytics -> latency -> cache -> slot release -> journal) is deliberate. Analytics sits just outside the circuit breaker because it needs to measure raw backend timing. Latency sits outside analytics because it measures the full request duration including analytics overhead. Cache sits outside latency because it needs to see the final stream. This layered composition of async generators is a practical application of the decorator pattern for streaming pipelines.

### Likely interview questions

**Q: "Why not use an HTTP trailer to send TTFT after the stream completes?"**
**A:** HTTP trailers are sent after the response body ends, so they could theoretically carry TTFT. The problem is client support: most HTTP/1.1 clients ignore trailers, many proxies and CDNs strip them, and browser `fetch()` does not expose them. More fundamentally, the value of TTFT is as a real-time signal — knowing TTFT after the entire response has finished is less useful than knowing it at the moment the first token arrives. The SSE comment provides TTFT at exactly the right time in the stream, to any client that inspects the raw event stream.

**Q: "Why not include tenant in the histogram labels?"**
**A:** Prometheus histograms are expensive — each label combination creates a separate time series, and each time series has one counter per bucket. With `[model, backend, tenant]` labels and 100 tenants, 5 models, 3 backends, and 10 buckets per histogram, that is 45,000 time series for a single metric. Multiply by 3 metrics and the cardinality reaches 135,000 series — enough to cause memory pressure on a typical Prometheus instance. The `[model, backend]` labels give operational insight (which models are slow on which backends) without tenant-level granularity. Per-tenant analytics should use a sampling-based approach or a dedicated time-series database designed for high cardinality.

**Q: "How would you detect if a model has bimodal TTFT — sometimes fast, sometimes slow?"**
**A:** The TTFT heatmap panel in the Grafana dashboard reveals distribution shape directly. A unimodal distribution appears as a single horizontal band; a bimodal distribution appears as two bands. Percentile charts (P50/P95/P99) obscure bimodality because they reduce the distribution to point estimates. For example, if 80% of requests have 50ms TTFT and 20% have 5 seconds (cold start), the P50 is 50ms (looks fine) and the P95 is 5 seconds (looks bad), but neither tells you there are two distinct populations. The heatmap shows both clusters. This is why the dashboard includes both percentile lines and heatmaps — they answer different questions about the same data.

---

## Production Deployment Considerations

### Why this exists

The gateway runs locally via Docker Compose with 14 services: three gateway instances behind Nginx, Redis, Prometheus, Grafana, Ollama backends, and mock API servers. This works for development and testing, but production requires managed services with high availability, encryption, auto-scaling, and operational alerting. Without an explicit infrastructure-as-code definition, the gap between "works on my machine" and "runs in production" remains undocumented and non-reproducible.

The Terraform definitions in `terraform/` map every local component to its AWS equivalent, making the production topology version-controlled, reviewable, and repeatable. `PRODUCTION.md` documents the operational differences, cost implications, and scaling strategy.

### How it works

The local-to-AWS mapping replaces each Docker Compose service with a managed AWS resource:

1. **Nginx reverse proxy → Application Load Balancer (ALB):** The ALB sits in public subnets across two availability zones. It terminates TLS (via ACM certificate), performs active health checks against `/health` every 30 seconds, and distributes traffic to ECS tasks in private subnets. The `deregistration_delay` of 120 seconds matches the gateway's httpx timeout, ensuring in-flight LLM streaming responses complete during deployments.

2. **3x gateway containers → ECS Fargate service:** Fargate runs the same Docker image on ARM64/Graviton processors (20% cost savings). The task definition mirrors the Docker Compose environment variables, with `REDIS_URL` pointing to the ElastiCache endpoint instead of `redis://redis:6379/0`. Auto-scaling tracks CPU utilization at 70% with a floor of 2 tasks and ceiling of 10.

3. **Redis Alpine → ElastiCache Redis 7.1 replication group:** A primary node plus one replica across two AZs provides automatic failover in under 30 seconds. Encryption at rest and in transit (TLS) are enabled. The `allkeys-lru` eviction policy matches the caching workload — cache misses are non-fatal (just slower).

4. **Docker bridge network → VPC with public/private subnets:** Two availability zones, each with a public subnet (ALB, NAT Gateway) and private subnet (ECS tasks, Redis). VPC endpoints for ECR, S3, and CloudWatch Logs reduce NAT Gateway data transfer costs. Security groups enforce the ALB → ECS → Redis traffic flow.

5. **Prometheus + Grafana → CloudWatch logs + alarms:** Seven CloudWatch alarms cover the critical failure modes: ALB 5xx rate, P99 latency, unhealthy targets, ECS CPU/memory, and Redis CPU/memory. The gateway's `/metrics/` endpoint and existing Grafana dashboards can optionally be preserved by deploying Prometheus + Grafana as additional ECS tasks.

6. **.env files → SSM Parameter Store:** API keys for OpenAI and Anthropic backends are stored in SSM Parameter Store and injected into ECS tasks via the execution role's `ssm:GetParameters` permission. No secrets in environment variables or task definitions.

### Implementation

All infrastructure is defined in `terraform/` using a flat file structure (no nested modules):

| File | Purpose |
|------|---------|
| `versions.tf` | Terraform >= 1.5, AWS provider ~> 5.0 |
| `variables.tf` | 25 input variables with defaults matching Docker Compose |
| `vpc.tf` | VPC, 2 AZ subnets, IGW, NAT Gateway, VPC endpoints |
| `security.tf` | Security groups (ALB/ECS/Redis), IAM roles (execution/task) |
| `ecr.tf` | Container registry with scan-on-push and lifecycle policy |
| `ecs.tf` | Fargate cluster, ARM64 task definition, service, auto-scaling |
| `alb.tf` | Load balancer, target group, conditional HTTPS listener |
| `redis.tf` | ElastiCache replication group with encryption |
| `monitoring.tf` | CloudWatch log group, SNS topic, 7 metric alarms |
| `outputs.tf` | ALB DNS, ECR URL, Redis endpoint, VPC/subnet IDs |
| `terraform.tfvars.example` | Example configuration values |

`PRODUCTION.md` at the project root documents the operational differences, cost estimates at 100 RPS (~$180/month) and 1000 RPS (~$978/month), Redis sizing guidance, scaling strategy, and monitoring tradeoffs.

`scripts/test-terraform.sh` validates the Terraform configuration in Docker (no AWS credentials needed) and checks that PRODUCTION.md contains all required sections.

### Key design decisions

**Flat Terraform structure over modules.** This is a single-project portfolio repository, not a multi-team infrastructure library. A flat structure with one file per concern (vpc.tf, ecs.tf, etc.) is easier to read and navigate than nested module directories. Each file is self-contained and clearly named. If the project grew to manage multiple environments or services, extracting reusable modules would be the natural next step.

**ARM64/Graviton for ECS Fargate.** The gateway is pure Python — no native C extensions that require x86 compilation. The base image (`python:3.12-slim`) and all pip dependencies (including `sentence-transformers`) provide ARM64 wheels. Graviton instances are 20% cheaper than x86 equivalents with comparable or better performance for Python workloads.

**VPC endpoints for ECR, S3, and CloudWatch Logs.** At 1000 RPS, NAT Gateway data transfer costs become significant ($0.045/GB). VPC endpoints route traffic to these AWS services within the VPC, bypassing NAT entirely. The three Interface endpoints cost ~$22/month but save more than that in NAT transfer fees at scale.

**Deployment circuit breaker with automatic rollback.** ECS natively supports deployment circuit breakers that detect failing tasks during a rolling update and automatically roll back to the previous stable version. This removes the need for external deployment tooling (CodeDeploy, custom scripts) for the common case of "new image doesn't start."

**Deregistration delay of 120 seconds.** LLM inference requests — especially streaming responses — can take 10-60 seconds. The deregistration delay must exceed the longest expected request duration to prevent dropped connections during deployments. 120 seconds provides headroom above the 60-second typical maximum while not delaying deployments unnecessarily.

**Conditional HTTPS listener.** The ALB supports both HTTP-only (for testing/dev) and HTTPS (for production) via the `certificate_arn` variable. When a certificate ARN is provided, the HTTP listener redirects to HTTPS. When omitted, HTTP forwards directly to the target group. This allows the Terraform to validate and apply in any environment without requiring a pre-existing ACM certificate.

### Alternatives considered

**EKS (Kubernetes) vs ECS Fargate.** Kubernetes provides richer orchestration (CronJobs, DaemonSets, custom operators, service mesh) and is the industry standard for complex microservice architectures. However, the inference gateway is a single service with a Redis dependency — not a microservice graph. ECS Fargate eliminates cluster management (no node groups, no control plane, no etcd), costs less for simple workloads, and integrates natively with ALB, CloudWatch, and IAM. For a single-service deployment, ECS is the simpler choice.

**EC2 vs Fargate.** EC2 instances provide more control (SSH access, custom AMIs, GPU instances for local inference) and can be cheaper with Reserved Instances for steady-state workloads. Fargate eliminates instance management entirely — no patching, no capacity planning, no idle costs. For a proxy/gateway service that doesn't run inference locally, Fargate's serverless model is a better fit. If the gateway needed to run Ollama or vLLM locally (instead of proxying to external APIs), EC2 with GPU instances would be necessary.

**Terraform modules vs flat structure.** Modules enable reuse across environments (dev/staging/prod) and teams. For a single-environment, single-operator project, modules add indirection without reuse benefits. The flat structure keeps all definitions visible in one directory, with `terraform.tfvars` providing the only customization layer. If the project expanded to multiple environments, the first refactoring step would be extracting modules and creating per-environment `tfvars` files.

**CloudWatch-only vs hybrid monitoring.** Pure CloudWatch is zero-maintenance but lacks the rich dashboard experience built in previous phases (histogram heatmaps, per-backend drilldown, streaming analytics). Self-hosted Prometheus + Grafana preserves this investment but costs ~$40-60/month in additional ECS tasks. The Terraform defines CloudWatch alarms for operational alerting; the existing Prometheus metrics endpoint and Grafana dashboards can be layered on top as an optional enhancement.

### Failure modes and edge cases

**Single AZ failure.** The VPC spans two AZs. ALB, ECS service, and ElastiCache replication group all operate across both AZs. If one AZ goes down: ALB routes traffic to the surviving AZ's targets, ECS launches replacement tasks in the healthy AZ, and ElastiCache promotes the replica to primary (< 30 second failover). The `gateway_min_count = 2` ensures at least one task per AZ under normal conditions.

**Redis failover.** ElastiCache automatic failover promotes the replica to primary and updates the DNS endpoint. The gateway's Redis client reconnects automatically. During the ~30-second failover window: cache lookups return misses (requests go directly to LLM backends), rate limiter checks fail open (requests are allowed), and priority queue operations fail gracefully. All of these are by design — the gateway treats Redis as best-effort.

**NAT Gateway failure.** If the single NAT Gateway fails, ECS tasks in private subnets lose outbound internet access. This breaks calls to external LLM APIs (OpenAI, Anthropic). Mitigation: set `single_nat_gateway = false` in production to deploy one NAT per AZ. VPC endpoints for ECR/S3/CloudWatch ensure container operations continue even without NAT.

**ACM certificate expiry.** ACM certificates auto-renew if validated via DNS (Route53 CNAME). If using email validation, expiry is a risk. The ALB health check continues to work (it checks the target, not the listener TLS), but clients receive TLS errors. CloudWatch does not alarm on certificate expiry by default — this requires a separate `aws_cloudwatch_metric_alarm` on `DaysToExpiry` in the ACM namespace.

**ECS task OOM.** The gateway loads a sentence-transformers model (~100MB) on startup. If `gateway_memory` is set too low, the task OOM-kills during initialization. The `startPeriod = 60` prevents the health check from marking the task unhealthy during startup, but OOM kills happen at the Docker/Fargate level before the health check runs. The default 2048 MB provides ample headroom.

**Image pull failure.** If the ECR image tag doesn't exist, ECS tasks fail to start. The deployment circuit breaker detects this and rolls back automatically. The `ecr_lifecycle_policy` keeps the last 10 tagged images, preventing accidental cleanup of the currently deployed version.

### Observability

**CloudWatch Logs.** All gateway stdout/stderr is captured via the `awslogs` driver into `/ecs/inference-gateway/gateway` with configurable retention (default 30 days). The gateway's structured JSON logging (via `structlog`) makes CloudWatch Logs Insights queries straightforward: filter by `request_id`, `tenant_id`, `backend`, or `status_code`.

**CloudWatch Alarms (7 total):**

| Alarm | Namespace | Metric | Condition | Action |
|-------|-----------|--------|-----------|--------|
| ALB 5xx rate | AWS/ApplicationELB | HTTPCode_Target_5XX_Count | > 10 in 1 min | SNS notification |
| P99 latency | AWS/ApplicationELB | TargetResponseTime (p99) | > 10 seconds | SNS notification |
| Unhealthy targets | AWS/ApplicationELB | UnHealthyHostCount | > 0 | SNS notification |
| ECS CPU | AWS/ECS | CPUUtilization | > 85% for 3 min | SNS notification |
| ECS memory | AWS/ECS | MemoryUtilization | > 85% for 3 min | SNS notification |
| Redis CPU | AWS/ElastiCache | EngineCPUUtilization | > 75% for 3 min | SNS notification |
| Redis memory | AWS/ElastiCache | DatabaseMemoryUsagePercentage | > 80% for 2 eval periods | SNS notification |

**Container Insights.** Enabled on the ECS cluster for enhanced metrics: task-level CPU/memory, network I/O, and storage utilization. These feed into the auto-scaling decisions.

**Prometheus metrics (optional).** The gateway continues to expose `/metrics/` in production. If Prometheus is deployed alongside, all existing dashboards (gateway overview, per-backend drilldown, per-tenant usage, streaming analytics) work without modification.

### Testing

**Terraform validation.** `scripts/test-terraform.sh` runs `terraform init -backend=false` and `terraform validate` inside the `hashicorp/terraform:latest` Docker container. This checks HCL syntax, provider schema compliance, and internal reference consistency — all without AWS credentials or API calls. The Makefile target `make terraform-validate` wraps this.

**PRODUCTION.md validation.** The same script verifies that PRODUCTION.md exists and contains all five required section headers via `grep`.

**No integration testing.** Terraform integration testing (actually applying infrastructure) is explicitly out of scope. The Phase 17 spec states "no `terraform apply`." Verifying that the infrastructure works correctly would require an AWS account and would incur costs. The `terraform validate` check ensures syntactic and structural correctness; semantic correctness (correct security group rules, correct IAM permissions) requires manual review or a dedicated testing tool like `terratest`.

### Production gaps

**No remote state backend.** The Terraform configuration uses local state by default. For team use, an S3 backend with DynamoDB state locking should be configured in a `backend.tf` file. Without remote state, concurrent `terraform apply` runs can corrupt state.

**No CI/CD pipeline.** Deployments require manual `terraform apply` and `docker push`. A production setup would include a CI/CD pipeline (GitHub Actions, CodePipeline) that builds the Docker image, pushes to ECR, and triggers ECS deployment on merge to main.

**No WAF.** The ALB has no Web Application Firewall. A production API gateway should have AWS WAF rules for rate limiting at the edge, IP allowlisting, and request size limits. The gateway's application-level rate limiter provides per-tenant controls but not DDoS protection.

**No Route53.** The ALB is accessible only via its auto-generated DNS name (e.g., `inference-gateway-alb-123456.us-east-1.elb.amazonaws.com`). A production deployment would add a Route53 hosted zone with an alias record pointing to the ALB.

**No multi-region.** All infrastructure is in a single AWS region. For global availability, the deployment would need to be replicated across regions with Route53 latency-based routing or Global Accelerator. Redis state would not replicate across regions — each region would have its own cache (cold start on failover).

**No Secrets Manager rotation.** API keys are stored in SSM Parameter Store as static values. Production should use Secrets Manager with automatic rotation for any credentials that support it.

### Interview talking points

- **The local-to-production mapping demonstrates that application code should be environment-agnostic.** The gateway source code has zero changes between Docker Compose and AWS. Every difference — service discovery, TLS, credentials, scaling — is handled by infrastructure configuration. This separation means developers test locally with `docker compose up` and deploy to production with `terraform apply`, using the same Docker image.

- **Cost-conscious infrastructure design is a first-class concern.** ARM64/Graviton (20% savings), conditional single NAT Gateway (vs per-AZ), VPC endpoints (NAT data transfer avoidance), and right-sized ElastiCache nodes demonstrate that production infrastructure is not just about correctness but about cost efficiency. The PRODUCTION.md cost tables at 100 RPS and 1000 RPS show that infrastructure costs scale sub-linearly with traffic.

- **Terraform validates without AWS credentials by separating `init -backend=false` from `validate`.** This enables CI pipelines to catch Terraform syntax and structural errors on every PR without requiring AWS credentials in the CI environment — a security best practice that also reduces CI complexity.

### Likely interview questions

**Q: "Why ECS Fargate over Kubernetes for this gateway?"**
**A:** The inference gateway is a single service with a Redis dependency — not a microservice graph requiring service mesh, custom operators, or complex scheduling. ECS Fargate eliminates cluster management (no node groups, no control plane, no etcd), integrates natively with ALB and CloudWatch, and costs less for simple workloads. Kubernetes would add operational complexity (cluster upgrades, RBAC policies, ingress controller configuration) without proportional benefit. The decision would change if the project grew to include multiple interdependent services, needed GPU scheduling for local inference, or required the portability of the Kubernetes API across cloud providers.

**Q: "How would you handle a Redis failover without dropping requests?"**
**A:** The gateway already treats Redis as best-effort — this was a deliberate design decision in earlier phases. During ElastiCache failover (~30 seconds): cache lookups return misses (requests bypass cache and go directly to LLM backends — slower but functional), rate limiter checks fail open (requests are allowed rather than rejected), and priority queue operations skip gracefully. The application continues serving requests with degraded performance, not errors. The ElastiCache replication group uses DNS-based failover — the primary endpoint DNS record updates to point to the new primary, and the gateway's Redis client library handles reconnection automatically. No application code change is needed.

**Q: "What would you add first for a real production deployment?"**
**A:** A CI/CD pipeline and remote state backend. Without CI/CD, deployments are manual and error-prone — a GitHub Actions workflow that builds the Docker image on merge to main, pushes to ECR, and triggers `terraform apply` with the new image tag would close the biggest operational gap. Without remote state (S3 + DynamoDB locking), Terraform state lives on a single developer's machine, making collaboration impossible and state corruption likely. These two additions convert the Terraform definitions from "validated configuration" to "operational infrastructure." After that, I would add WAF on the ALB for edge rate limiting, Route53 for a stable domain name, and Secrets Manager rotation for API keys.
