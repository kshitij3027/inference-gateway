"""Tests for observability configuration files (Prometheus, Grafana)."""

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestPrometheusConfig:
    def test_config_valid_yaml(self):
        config_path = PROJECT_ROOT / "prometheus" / "prometheus.yml"
        data = yaml.safe_load(config_path.read_text())
        assert "scrape_configs" in data

    def test_gateway_job_exists(self):
        config_path = PROJECT_ROOT / "prometheus" / "prometheus.yml"
        data = yaml.safe_load(config_path.read_text())
        job_names = [job["job_name"] for job in data["scrape_configs"]]
        assert "gateway" in job_names

    def test_gateway_target(self):
        config_path = PROJECT_ROOT / "prometheus" / "prometheus.yml"
        data = yaml.safe_load(config_path.read_text())
        gateway_job = next(j for j in data["scrape_configs"] if j["job_name"] == "gateway")
        targets = gateway_job["static_configs"][0]["targets"]
        assert "gateway:8080" in targets


class TestGrafanaDatasource:
    def test_datasource_valid_yaml(self):
        ds_path = PROJECT_ROOT / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
        data = yaml.safe_load(ds_path.read_text())
        assert "datasources" in data

    def test_prometheus_datasource(self):
        ds_path = PROJECT_ROOT / "grafana" / "provisioning" / "datasources" / "prometheus.yml"
        data = yaml.safe_load(ds_path.read_text())
        ds = data["datasources"][0]
        assert ds["name"] == "Prometheus"
        assert ds["type"] == "prometheus"
        assert "prometheus" in ds["url"]


class TestGrafanaDashboardProvider:
    def test_provider_valid_yaml(self):
        provider_path = PROJECT_ROOT / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
        data = yaml.safe_load(provider_path.read_text())
        assert "providers" in data
        assert len(data["providers"]) >= 1
