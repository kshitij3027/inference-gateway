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
