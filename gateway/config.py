from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import structlog
import yaml
from pydantic import BaseModel, Field, model_validator

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


class TenantConfig(BaseModel):
    id: str
    name: str | None = None
    api_key_env: str
    allowed_models: list[str] = Field(..., min_length=1)
    priority: int = 1
    rate_limit_rps: int | None = None
    rate_limit_rpm: int | None = None
    token_budget_daily: int | None = None


class GatewayConfig(BaseModel):
    backends: list[BackendConfig] = Field(..., min_length=1)
    tenants: list[TenantConfig] = Field(..., min_length=1)

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

    def __init__(self, config: GatewayConfig) -> None:
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

    def find_backend_for_model(self, model: str) -> BackendConfig | None:
        """Return the first backend that serves the given model, or None."""
        backends = self.model_to_backends.get(model, [])
        return backends[0] if backends else None
