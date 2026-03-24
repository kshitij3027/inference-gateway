# Inference Gateway

A high-performance proxy that sits in front of multiple LLM backends, intelligently routing, caching, rate-limiting, and load-balancing inference requests — with a real-time observability dashboard.

Think of it as a miniature [Cloudflare AI Gateway](https://developers.cloudflare.com/ai-gateway/) or [Portkey](https://portkey.ai/) built from distributed systems primitives.

---

## Features

- **Unified OpenAI-compatible API** — Send standard `/v1/chat/completions` requests; the gateway handles the rest.
- **Multi-backend routing** — OpenAI, Anthropic, Ollama, vLLM, and any OpenAI-compatible endpoint.
- **Intelligent load balancing** — Round-robin, least-connections, and latency-aware strategies.
- **Semantic caching** — Redis-backed response cache with configurable TTL to cut costs and latency.
- **Rate limiting** — Per-key and global token-bucket rate limiting via Redis.
- **Automatic failover** — If a backend is down or returns errors, requests are retried on healthy backends.
- **Streaming support** — Full SSE streaming pass-through for chat completions.
- **Real-time dashboard** — Live metrics, request logs, and backend health via WebSocket-powered UI.
- **Observability** — Prometheus metrics, structured JSON logs, distributed request tracing.

## Architecture

```
                         ┌─────────────────────────────────────┐
                         │          Inference Gateway           │
                         │                                     │
  Client ──► /v1/chat/   │  ┌───────────┐   ┌──────────────┐  │
  completions            │  │ Rate      │──►│ Router /     │  │
                         │  │ Limiter   │   │ Load Balancer│  │
                         │  └───────────┘   └──────┬───────┘  │
                         │        │                 │          │
                         │  ┌─────▼─────┐    ┌─────▼───────┐  │
                         │  │  Cache     │    │  Backend    │  │
                         │  │  (Redis)   │    │  Pool       │  │
                         │  └───────────┘    └──────┬──────┘  │
                         │                          │         │
                         └──────────────────────────┼─────────┘
                                                    │
                    ┌───────────┬───────────┬───────┴────┐
                    ▼           ▼           ▼            ▼
                 OpenAI    Anthropic     Ollama        vLLM
```

## Quick Start

```bash
# Clone and start everything
git clone <repo-url> && cd inference-gateway
make up

# Send a request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-gateway-key" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Open dashboard
open http://localhost:3000

# Tear down
make down
```

## How It Runs

A multi-service application launched via `docker compose up`. Users send standard OpenAI-compatible API requests to the gateway, which handles routing, caching, rate limiting, and failover transparently. A companion dashboard shows live metrics. The entire system is bootstrapped with a single `make up` command and torn down with `make down`.

## Tech Stack

| Component         | Technology                     |
|-------------------|--------------------------------|
| API Server        | FastAPI + Uvicorn              |
| HTTP Client       | httpx (async)                  |
| Cache / Rate Limit| Redis                          |
| Metrics           | Prometheus + prometheus-client |
| Logging           | structlog (JSON)               |
| Token Counting    | tiktoken                       |
| Retry / Failover  | tenacity                       |
| Dashboard         | WebSocket + lightweight frontend |
| Orchestration     | Docker Compose                 |

## Project Structure (Planned)

```
inference-gateway/
├── gateway/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entrypoint
│   ├── config.py            # Settings via pydantic-settings
│   ├── routes/
│   │   ├── chat.py          # /v1/chat/completions
│   │   └── health.py        # /health, /readiness
│   ├── backends/
│   │   ├── base.py          # Abstract backend interface
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── ollama.py
│   │   └── vllm.py
│   ├── routing/
│   │   ├── balancer.py      # Load-balancing strategies
│   │   └── failover.py      # Health checks + retry logic
│   ├── middleware/
│   │   ├── rate_limiter.py  # Token-bucket via Redis
│   │   └── cache.py         # Semantic response cache
│   └── observability/
│       ├── metrics.py       # Prometheus counters/histograms
│       └── logging.py       # Structured logging setup
├── dashboard/
│   ├── app.py               # WebSocket server for live metrics
│   └── static/              # Lightweight frontend
├── tests/
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## License

MIT
