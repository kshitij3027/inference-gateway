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
