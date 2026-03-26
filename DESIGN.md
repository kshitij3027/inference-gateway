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
