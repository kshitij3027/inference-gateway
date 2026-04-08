#!/usr/bin/env bash
set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
JAEGER_URL="${JAEGER_URL:-http://localhost:16686}"

echo "=== Tracing E2E Test ==="
echo "Gateway: $GATEWAY_URL"
echo "Jaeger:  $JAEGER_URL"
echo ""

# 1. Send a request, capture X-Trace-ID header
echo "Sending chat completion request..."
TRACE_ID=$(curl -s -D - -o /dev/null -X POST "$GATEWAY_URL/v1/chat/completions" \
  -H "Authorization: Bearer test-beta-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"mock-gpt-markdown","messages":[{"role":"user","content":"trace test"}]}' \
  | grep -i x-trace-id | tr -d '\r' | awk '{print $2}')

if [ -z "$TRACE_ID" ]; then
  echo "FAIL: No X-Trace-ID header in response"
  exit 1
fi
echo "Got trace ID: $TRACE_ID"

# 2. Wait for Jaeger to ingest (batch export delay)
echo "Waiting for Jaeger ingestion..."
sleep 5

# 3. Query Jaeger API for the trace
echo "Querying Jaeger for trace..."
TRACE_JSON=$(curl -s "$JAEGER_URL/api/traces/$TRACE_ID")

if echo "$TRACE_JSON" | grep -q '"errors"'; then
  echo "FAIL: Jaeger returned errors"
  echo "$TRACE_JSON" | head -5
  exit 1
fi

# 4. Verify expected spans exist in the trace
PASS_COUNT=0
FAIL_COUNT=0

for SPAN_NAME in "gateway.auth" "gateway.rate_limit" "gateway.cache.lookup" "gateway.router" "gateway.translator.request" "gateway.circuit_breaker"; do
  if echo "$TRACE_JSON" | grep -q "\"$SPAN_NAME\""; then
    echo "PASS: Found span '$SPAN_NAME'"
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    echo "FAIL: Missing span '$SPAN_NAME'"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done

echo ""
echo "Results: $PASS_COUNT passed, $FAIL_COUNT failed"

if [ "$FAIL_COUNT" -gt 0 ]; then
  echo "FAIL: Not all expected spans found in Jaeger"
  exit 1
fi

echo "PASS: All expected tracing spans found in Jaeger"
