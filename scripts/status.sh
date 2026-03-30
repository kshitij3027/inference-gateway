#!/usr/bin/env bash
set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"

PASS=0
FAIL=0

check() {
    local name="$1"
    local cmd="$2"
    if eval "$cmd" > /dev/null 2>&1; then
        echo "  PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL  $name"
        FAIL=$((FAIL + 1))
    fi
}

echo "Checking services..."
check "Gateway /health"    "curl -sf $GATEWAY_URL/health"
check "Gateway /ready"     "curl -sf $GATEWAY_URL/ready"
check "Gateway /metrics"   "curl -sf $GATEWAY_URL/metrics/"
check "Prometheus"         "curl -sf $PROMETHEUS_URL/-/ready"
check "Grafana"            "curl -sf $GRAFANA_URL/api/health"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
