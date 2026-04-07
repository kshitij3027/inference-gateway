"""Unit tests for the igw CLI tool."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from cli.main import cli


@pytest.fixture
def runner():
    return CliRunner()


def _mock_response(json_data, status_code=200):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


class TestStatusCommand:
    def test_shows_healthy(self, runner):
        with patch("cli.main.httpx.get") as mock_get:
            mock_get.side_effect = [
                _mock_response({"status": "ok"}),
                _mock_response({"status": "ready", "healthy_backends": 10, "total_backends": 10}),
            ]
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "ok" in result.output
        assert "10" in result.output


class TestBackendsCommand:
    def test_shows_backends_table(self, runner):
        with patch("cli.main.httpx.get") as mock_get:
            mock_get.return_value = _mock_response([
                {
                    "name": "mock-openai-1",
                    "provider": "openai",
                    "models": ["gpt-4"],
                    "health": "CLOSED",
                    "circuit_breaker": {"state": "CLOSED", "error_rate": 0.0, "requests_in_window": 5},
                },
                {
                    "name": "ollama-1",
                    "provider": "ollama",
                    "models": ["tinyllama"],
                    "health": "OPEN",
                    "circuit_breaker": {"state": "OPEN", "error_rate": 0.75, "requests_in_window": 20},
                },
            ])
            result = runner.invoke(cli, ["backends"])
        assert result.exit_code == 0
        assert "mock-openai-1" in result.output
        assert "ollama-1" in result.output
        assert "CLOSED" in result.output


class TestTenantsCommand:
    def test_shows_tenants(self, runner):
        with patch("cli.main.httpx.get") as mock_get:
            mock_get.return_value = _mock_response([
                {"id": "tenant-alpha", "allowed_models": ["gpt-4"], "priority": 1,
                 "rate_limit_rps": 10, "rate_limit_rpm": 60, "token_budget_daily": 500},
            ])
            result = runner.invoke(cli, ["tenants"])
        assert result.exit_code == 0
        assert "tenant-alpha" in result.output
        assert "10" in result.output


class TestCacheStatsCommand:
    def test_shows_stats(self, runner):
        with patch("cli.main.httpx.get") as mock_get:
            mock_get.return_value = _mock_response({
                "enabled": True, "hits": 42, "misses": 100, "hit_rate": 0.296,
                "entries": 80, "l1_hits": 10, "l1_misses": 32,
            })
            result = runner.invoke(cli, ["cache", "stats"])
        assert result.exit_code == 0
        assert "42" in result.output
        assert "29.6%" in result.output


class TestCostCommand:
    def test_shows_all_tenants(self, runner):
        with patch("cli.main.httpx.get") as mock_get:
            mock_get.return_value = _mock_response({
                "enabled": True,
                "tenants": [
                    {"tenant_id": "t1", "today": 0.05, "costs_by_date": {"2025-01-15": 0.05}},
                    {"tenant_id": "t2", "today": 0.12, "costs_by_date": {"2025-01-15": 0.12}},
                ],
            })
            result = runner.invoke(cli, ["cost"])
        assert result.exit_code == 0
        assert "t1" in result.output
        assert "t2" in result.output
        assert "0.05" in result.output

    def test_shows_single_tenant(self, runner):
        with patch("cli.main.httpx.get") as mock_get:
            mock_get.return_value = _mock_response({
                "enabled": True, "tenant_id": "t1", "today": 0.05,
                "costs_by_date": {"2025-01-15": 0.05, "2025-01-14": 0.03},
            })
            result = runner.invoke(cli, ["cost", "--tenant", "t1"])
        assert result.exit_code == 0
        assert "t1" in result.output
        assert "2025-01-15" in result.output


class TestConnectionError:
    def test_graceful_error(self, runner):
        with patch("cli.main.httpx.get", side_effect=ConnectionError("refused")):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code != 0


class TestGatewayOption:
    def test_custom_url(self, runner):
        with patch("cli.main.httpx.get") as mock_get:
            mock_get.side_effect = [
                _mock_response({"status": "ok"}),
                _mock_response({"status": "ready", "healthy_backends": 5, "total_backends": 5}),
            ]
            result = runner.invoke(cli, ["-g", "http://custom:9090", "status"])
        assert result.exit_code == 0
        call_url = mock_get.call_args_list[0][0][0]
        assert "custom:9090" in call_url
