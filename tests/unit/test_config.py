import os

import pytest
import yaml

from gateway.config import (
    BackendConfig,
    ConfigError,
    GatewayConfig,
    Registry,
    TenantConfig,
    load_config,
)


class TestBackendConfig:
    def test_valid_backend(self):
        b = BackendConfig(
            name="test-backend",
            provider="ollama",
            base_url="http://localhost:11434",
            models=["tinyllama"],
        )
        assert b.name == "test-backend"
        assert b.weight == 1
        assert b.max_concurrent == 10
        assert b.timeout_ms == 120000
        assert b.api_key_env is None

    def test_backend_requires_name(self):
        with pytest.raises(Exception):
            BackendConfig(
                provider="ollama",
                base_url="http://localhost:11434",
                models=["tinyllama"],
            )

    def test_invalid_provider_rejected(self):
        with pytest.raises(Exception):
            BackendConfig(
                name="bad",
                provider="invalid",
                base_url="http://localhost:11434",
                models=["tinyllama"],
            )

    def test_empty_models_rejected(self):
        with pytest.raises(Exception):
            BackendConfig(
                name="bad",
                provider="ollama",
                base_url="http://localhost:11434",
                models=[],
            )


class TestTenantConfig:
    def test_valid_tenant(self):
        t = TenantConfig(
            id="t1",
            api_key_env="TEST_KEY",
            allowed_models=["tinyllama"],
        )
        assert t.id == "t1"
        assert t.priority == 1
        assert t.name is None
        assert t.rate_limit_rps is None

    def test_tenant_requires_id(self):
        with pytest.raises(Exception):
            TenantConfig(
                api_key_env="TEST_KEY",
                allowed_models=["tinyllama"],
            )

    def test_tenant_requires_api_key_env(self):
        with pytest.raises(Exception):
            TenantConfig(
                id="t1",
                allowed_models=["tinyllama"],
            )


class TestGatewayConfig:
    def _make_backend(self, name="b1"):
        return {
            "name": name,
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "models": ["tinyllama"],
        }

    def _make_tenant(self, id="t1"):
        return {
            "id": id,
            "api_key_env": "TEST_KEY",
            "allowed_models": ["tinyllama"],
        }

    def test_valid_config(self):
        cfg = GatewayConfig.model_validate(
            {
                "backends": [self._make_backend()],
                "tenants": [self._make_tenant()],
            }
        )
        assert len(cfg.backends) == 1
        assert len(cfg.tenants) == 1

    def test_duplicate_backend_names_rejected(self):
        with pytest.raises(Exception):
            GatewayConfig.model_validate(
                {
                    "backends": [self._make_backend("b1"), self._make_backend("b1")],
                    "tenants": [self._make_tenant()],
                }
            )

    def test_duplicate_tenant_ids_rejected(self):
        with pytest.raises(Exception):
            GatewayConfig.model_validate(
                {
                    "backends": [self._make_backend()],
                    "tenants": [self._make_tenant("t1"), self._make_tenant("t1")],
                }
            )

    def test_empty_backends_rejected(self):
        with pytest.raises(Exception):
            GatewayConfig.model_validate(
                {
                    "backends": [],
                    "tenants": [self._make_tenant()],
                }
            )

    def test_empty_tenants_rejected(self):
        with pytest.raises(Exception):
            GatewayConfig.model_validate(
                {
                    "backends": [self._make_backend()],
                    "tenants": [],
                }
            )


class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "backends": [
                        {
                            "name": "test",
                            "provider": "ollama",
                            "base_url": "http://localhost:11434",
                            "models": ["tinyllama"],
                        }
                    ],
                    "tenants": [
                        {
                            "id": "t1",
                            "api_key_env": "TEST_KEY",
                            "allowed_models": ["tinyllama"],
                        }
                    ],
                }
            )
        )
        cfg = load_config(config_file)
        assert len(cfg.backends) == 1
        assert cfg.backends[0].name == "test"

    def test_load_missing_file_raises_config_error(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config("/nonexistent/path.yaml")

    def test_load_invalid_yaml_raises_config_error(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("{{invalid yaml")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(config_file)

    def test_load_invalid_schema_raises_config_error(self, tmp_path):
        config_file = tmp_path / "bad_schema.yaml"
        config_file.write_text(yaml.dump({"backends": [], "tenants": []}))
        with pytest.raises(ConfigError, match="validation failed"):
            load_config(config_file)


class TestRegistry:
    def _make_config(self):
        return GatewayConfig.model_validate(
            {
                "backends": [
                    {
                        "name": "ollama-1",
                        "provider": "ollama",
                        "base_url": "http://ollama:11434",
                        "models": ["tinyllama", "llama2"],
                    },
                    {
                        "name": "openai-1",
                        "provider": "openai",
                        "base_url": "http://mock-openai:9001",
                        "models": ["gpt-4o-mini"],
                    },
                ],
                "tenants": [
                    {
                        "id": "t1",
                        "api_key_env": "T1_KEY",
                        "allowed_models": ["tinyllama"],
                    },
                    {
                        "id": "t2",
                        "api_key_env": "T2_KEY",
                        "allowed_models": ["*"],
                    },
                ],
            }
        )

    def test_backends_indexed_by_name(self):
        config = self._make_config()
        os.environ["T1_KEY"] = "key1"
        os.environ["T2_KEY"] = "key2"
        try:
            reg = Registry(config)
            assert "ollama-1" in reg.backends
            assert "openai-1" in reg.backends
        finally:
            del os.environ["T1_KEY"]
            del os.environ["T2_KEY"]

    def test_model_to_backends_index(self, monkeypatch):
        monkeypatch.setenv("T1_KEY", "key1")
        monkeypatch.setenv("T2_KEY", "key2")
        config = self._make_config()
        reg = Registry(config)
        assert len(reg.model_to_backends["tinyllama"]) == 1
        assert reg.model_to_backends["tinyllama"][0].name == "ollama-1"
        assert len(reg.model_to_backends["gpt-4o-mini"]) == 1

    def test_find_backend_for_model(self, monkeypatch):
        monkeypatch.setenv("T1_KEY", "key1")
        monkeypatch.setenv("T2_KEY", "key2")
        config = self._make_config()
        reg = Registry(config)
        assert reg.find_backend_for_model("tinyllama").name == "ollama-1"
        assert reg.find_backend_for_model("gpt-4o-mini").name == "openai-1"
        assert reg.find_backend_for_model("nonexistent") is None

    def test_api_key_resolution(self, monkeypatch):
        monkeypatch.setenv("T1_KEY", "secret-key-1")
        monkeypatch.setenv("T2_KEY", "secret-key-2")
        config = self._make_config()
        reg = Registry(config)
        assert reg.api_key_to_tenant["secret-key-1"].id == "t1"
        assert reg.api_key_to_tenant["secret-key-2"].id == "t2"

    def test_missing_api_key_env_excluded(self, monkeypatch):
        monkeypatch.setenv("T1_KEY", "key1")
        # T2_KEY not set
        monkeypatch.delenv("T2_KEY", raising=False)
        config = self._make_config()
        reg = Registry(config)
        assert "key1" in reg.api_key_to_tenant
        assert len(reg.api_key_to_tenant) == 1  # t2 excluded

    def test_find_backend_with_routing_key(self, monkeypatch):
        """Same routing key always returns same backend."""
        monkeypatch.setenv("T1_KEY", "key1")
        config = GatewayConfig.model_validate({
            "backends": [
                {
                    "name": "ollama-1",
                    "provider": "ollama",
                    "base_url": "http://ollama-1:11434",
                    "models": ["tinyllama"],
                },
                {
                    "name": "ollama-2",
                    "provider": "ollama",
                    "base_url": "http://ollama-2:11434",
                    "models": ["tinyllama"],
                },
            ],
            "tenants": [{
                "id": "t1",
                "api_key_env": "T1_KEY",
                "allowed_models": ["tinyllama"],
            }],
        })
        reg = Registry(config)
        # Same key -> same backend (deterministic)
        results = {reg.find_backend_for_model("tinyllama", routing_key="tenant-a:tinyllama").name for _ in range(10)}
        assert len(results) == 1

    def test_find_backend_routing_key_none_falls_back(self, monkeypatch):
        """routing_key=None falls back to first match."""
        monkeypatch.setenv("T1_KEY", "key1")
        config = GatewayConfig.model_validate({
            "backends": [
                {
                    "name": "first",
                    "provider": "ollama",
                    "base_url": "http://first:11434",
                    "models": ["tinyllama"],
                },
                {
                    "name": "second",
                    "provider": "ollama",
                    "base_url": "http://second:11434",
                    "models": ["tinyllama"],
                },
            ],
            "tenants": [{
                "id": "t1",
                "api_key_env": "T1_KEY",
                "allowed_models": ["tinyllama"],
            }],
        })
        reg = Registry(config)
        result = reg.find_backend_for_model("tinyllama", routing_key=None)
        assert result.name == "first"

    def test_ring_state_returns_correct_structure(self, monkeypatch):
        monkeypatch.setenv("T1_KEY", "key1")
        monkeypatch.setenv("T2_KEY", "key2")
        config = self._make_config()
        reg = Registry(config)
        state = reg.ring_state()
        # Should have entries for each model
        assert "tinyllama" in state
        assert "gpt-4o-mini" in state
        assert "backends" in state["tinyllama"]
        assert "total_vnodes" in state["tinyllama"]
        assert "distribution" in state["tinyllama"]
        assert state["tinyllama"]["total_vnodes"] > 0

    def test_model_rings_built_for_each_model(self, monkeypatch):
        monkeypatch.setenv("T1_KEY", "key1")
        monkeypatch.setenv("T2_KEY", "key2")
        config = self._make_config()
        reg = Registry(config)
        # Should have a ring for each model in the config
        for model in reg.model_to_backends:
            assert model in reg.model_rings

    def test_find_backend_with_exclude(self, monkeypatch):
        """Excluded backends are skipped."""
        monkeypatch.setenv("T1_KEY", "key1")
        config = GatewayConfig.model_validate({
            "backends": [
                {"name": "a", "provider": "ollama", "base_url": "http://a:11434", "models": ["tinyllama"]},
                {"name": "b", "provider": "ollama", "base_url": "http://b:11434", "models": ["tinyllama"]},
            ],
            "tenants": [{"id": "t1", "api_key_env": "T1_KEY", "allowed_models": ["tinyllama"]}],
        })
        reg = Registry(config)
        # With routing_key and exclude
        result = reg.find_backend_for_model("tinyllama", routing_key="t1:tinyllama", exclude=frozenset({"a"}))
        assert result is not None
        assert result.name == "b"

    def test_find_backend_exclude_all_returns_none(self, monkeypatch):
        """Excluding all backends returns None."""
        monkeypatch.setenv("T1_KEY", "key1")
        config = GatewayConfig.model_validate({
            "backends": [
                {"name": "a", "provider": "ollama", "base_url": "http://a:11434", "models": ["tinyllama"]},
                {"name": "b", "provider": "ollama", "base_url": "http://b:11434", "models": ["tinyllama"]},
            ],
            "tenants": [{"id": "t1", "api_key_env": "T1_KEY", "allowed_models": ["tinyllama"]}],
        })
        reg = Registry(config)
        result = reg.find_backend_for_model("tinyllama", routing_key="t1:tinyllama", exclude=frozenset({"a", "b"}))
        assert result is None

    def test_find_backend_fallback_with_exclude(self, monkeypatch):
        """Fallback path (no routing_key) also respects exclude."""
        monkeypatch.setenv("T1_KEY", "key1")
        config = GatewayConfig.model_validate({
            "backends": [
                {"name": "first", "provider": "ollama", "base_url": "http://first:11434", "models": ["tinyllama"]},
                {"name": "second", "provider": "ollama", "base_url": "http://second:11434", "models": ["tinyllama"]},
            ],
            "tenants": [{"id": "t1", "api_key_env": "T1_KEY", "allowed_models": ["tinyllama"]}],
        })
        reg = Registry(config)
        result = reg.find_backend_for_model("tinyllama", exclude=frozenset({"first"}))
        assert result.name == "second"
