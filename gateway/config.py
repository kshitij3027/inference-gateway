from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import structlog
import yaml
from pydantic import BaseModel, Field, model_validator

from gateway.routing import ConsistentHashRing
from gateway.strategies import (
    ConsistentHashStrategy,
    CostAwareStrategy,
    LatencyAwareStrategy,
    RoutingStrategy,
)

logger = structlog.get_logger()


class ConfigError(Exception):
    """Raised when config loading or validation fails."""

    pass


class BackendConfig(BaseModel):
    name: str
    provider: Literal["ollama", "openai", "anthropic", "vllm"]
    base_url: str
    api_key_env: str | None = None
    models: list[str] = Field(..., min_length=1)
    weight: int = 1
    max_concurrent: int = 10
    timeout_ms: int = 120000
    cost_per_1k_tokens: float | None = None


class TenantConfig(BaseModel):
    id: str
    name: str | None = None
    api_key_env: str
    allowed_models: list[str] = Field(..., min_length=1)
    priority: int = 1
    rate_limit_rps: int | None = None
    rate_limit_rpm: int | None = None
    token_budget_daily: int | None = None
    cache_isolation: Literal["shared", "tenant"] = "shared"


class ModelRoutingConfig(BaseModel):
    """Per-model routing configuration."""

    strategy: Literal["consistent_hash", "latency_aware", "cost_aware"] = (
        "consistent_hash"
    )
    hedge_enabled: bool = False


class GatewayConfig(BaseModel):
    backends: list[BackendConfig] = Field(..., min_length=1)
    tenants: list[TenantConfig] = Field(..., min_length=1)
    routing: dict[str, ModelRoutingConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_uniqueness(self) -> GatewayConfig:
        backend_names = [b.name for b in self.backends]
        if len(backend_names) != len(set(backend_names)):
            raise ValueError("Backend names must be unique")
        tenant_ids = [t.id for t in self.tenants]
        if len(tenant_ids) != len(set(tenant_ids)):
            raise ValueError("Tenant IDs must be unique")
        return self


def load_config(path: str | Path) -> GatewayConfig:
    """Load and validate gateway config from a YAML file.

    Raises ConfigError on any failure (file not found, bad YAML, validation error).
    """
    path = Path(path)
    try:
        raw = path.read_text()
    except FileNotFoundError:
        raise ConfigError(f"Config file not found: {path}")
    except OSError as e:
        raise ConfigError(f"Cannot read config file {path}: {e}")

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")

    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a YAML mapping, got {type(data).__name__}")

    try:
        return GatewayConfig.model_validate(data)
    except Exception as e:
        raise ConfigError(f"Config validation failed: {e}")


class Registry:
    """In-memory registry of backends and tenants, built from config."""

    def __init__(self, config: GatewayConfig, latency_tracker=None) -> None:
        self.backends: dict[str, BackendConfig] = {b.name: b for b in config.backends}

        # Build model -> backends reverse index
        self.model_to_backends: dict[str, list[BackendConfig]] = {}
        for backend in config.backends:
            for model in backend.models:
                self.model_to_backends.setdefault(model, []).append(backend)

        # Resolve tenant API keys from environment
        self.api_key_to_tenant: dict[str, TenantConfig] = {}
        for tenant in config.tenants:
            api_key = os.environ.get(tenant.api_key_env, "")
            if not api_key:
                logger.warning(
                    "tenant_api_key_missing",
                    tenant_id=tenant.id,
                    env_var=tenant.api_key_env,
                )
                continue
            self.api_key_to_tenant[api_key] = tenant

        # Build per-model consistent hash rings
        self.model_rings: dict[str, ConsistentHashRing] = {}
        for model, backends in self.model_to_backends.items():
            self.model_rings[model] = ConsistentHashRing(
                [(b.name, b.weight) for b in backends]
            )

        # Build per-model routing strategies
        self.model_routing_config: dict[str, ModelRoutingConfig] = dict(config.routing)
        self.model_strategies: dict[str, RoutingStrategy] = {}
        for model, backends in self.model_to_backends.items():
            routing_cfg = config.routing.get(model, ModelRoutingConfig())
            ring = self.model_rings[model]

            if routing_cfg.strategy == "latency_aware" and latency_tracker is not None:
                self.model_strategies[model] = LatencyAwareStrategy(
                    latency_tracker, model
                )
            elif routing_cfg.strategy == "cost_aware":
                costs = {
                    b.name: b.cost_per_1k_tokens
                    for b in backends
                    if b.cost_per_1k_tokens is not None
                }
                self.model_strategies[model] = CostAwareStrategy(costs)
            else:
                self.model_strategies[model] = ConsistentHashStrategy(ring)

    def find_backend_for_model(
        self,
        model: str,
        routing_key: str | None = None,
        exclude: frozenset[str] = frozenset(),
    ) -> BackendConfig | None:
        """Find a backend for the given model using the configured strategy.

        Delegates to the per-model RoutingStrategy. Falls back to first match
        if no strategy is configured (model not in registry).

        Args:
            model: The model name to look up.
            routing_key: Optional key for consistent hash routing.
            exclude: Set of backend names to skip (for failover).
        """
        strategy = self.model_strategies.get(model)
        if strategy is not None:
            candidates = [b.name for b in self.model_to_backends.get(model, [])]
            node_name = strategy.select(candidates, exclude, routing_key)
            return self.backends.get(node_name) if node_name else None

        # Fallback: first match (model not in registry at all)
        backends = [
            b for b in self.model_to_backends.get(model, []) if b.name not in exclude
        ]
        return backends[0] if backends else None

    def ring_state(self) -> dict:
        """Return hash ring state per model for admin endpoint."""
        state = {}
        for model, ring in self.model_rings.items():
            backends_for_model = self.model_to_backends.get(model, [])
            state[model] = {
                "backends": [b.name for b in backends_for_model],
                "total_vnodes": ring.vnode_count,
                "distribution": ring.get_distribution(),
            }
        return state
