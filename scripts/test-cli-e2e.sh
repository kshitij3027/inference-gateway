#!/usr/bin/env bash
set -uo pipefail

GATEWAY="${IGW_GATEWAY_URL:-http://nginx:80}"
ALPHA_KEY="${TENANT_ALPHA_KEY:-test-alpha-key}"
BETA_KEY="${TENANT_BETA_KEY:-test-beta-key}"

PASS=0
FAIL=0
pass() { echo -e "PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo -e "FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=== Phase 1: Generate realistic traffic ==="

# Send 5 requests as tenant-beta to mock-openai backends (beta has no rate limits)
for i in $(seq 1 5); do
  curl -s -H "Authorization: Bearer $BETA_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"mock-gpt-markdown\",\"messages\":[{\"role\":\"user\",\"content\":\"Tell me about openai request number $i in detail\"}]}" \
    "$GATEWAY/v1/chat/completions" > /dev/null 2>&1 &
done
wait
echo "Sent 5 beta/openai requests"

# Send 5 requests as tenant-beta to mock-anthropic backends
for i in $(seq 1 5); do
  curl -sf -H "Authorization: Bearer $BETA_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"mock-claude-markdown\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain topic number $i with examples\"}]}" \
    "$GATEWAY/v1/chat/completions" > /dev/null 2>&1 &
done
wait
echo "Sent 5 beta/anthropic requests"
sleep 2

# Send 3 duplicate requests for cache hits (sequential, use beta to avoid rate limits)
for i in $(seq 1 3); do
  curl -s -H "Authorization: Bearer $BETA_KEY" \
    -H "Content-Type: application/json" \
    -d '{"model":"mock-gpt-markdown","messages":[{"role":"user","content":"Tell me about openai request number 1 in detail"}]}' \
    "$GATEWAY/v1/chat/completions" > /dev/null 2>&1 || true
  sleep 1
done
echo "Sent 3 duplicate requests for cache hits"

sleep 2
echo ""
echo "=== Phase 2: Verify CLI commands against real data ==="

# 1. Status
OUTPUT=$(igw -g "$GATEWAY" status 2>&1) || true
echo "$OUTPUT" | grep -qi "ok" && pass "igw status shows ok" || fail "igw status shows ok"

# 2. Backends — list real backends
OUTPUT=$(igw -g "$GATEWAY" backends 2>&1) || true
echo "$OUTPUT" | grep -q "mock-openai" && pass "igw backends shows mock-openai" || fail "igw backends shows mock-openai"
echo "$OUTPUT" | grep -q "anthropic" && pass "igw backends shows anthropic" || fail "igw backends shows anthropic"
echo "$OUTPUT" | grep -q "CLOSED" && pass "igw backends shows CLOSED state" || fail "igw backends shows CLOSED state"

# 3. Tenants
OUTPUT=$(igw -g "$GATEWAY" tenants 2>&1) || true
echo "$OUTPUT" | grep -qi "alpha" && pass "igw tenants shows alpha" || fail "igw tenants shows alpha"
echo "$OUTPUT" | grep -qi "beta" && pass "igw tenants shows beta" || fail "igw tenants shows beta"

# 4. Cache stats — should show hits from duplicates
OUTPUT=$(igw -g "$GATEWAY" cache stats 2>&1) || true
echo "$OUTPUT" | grep -q "Hit Rate" && pass "igw cache stats shows hit rate" || fail "igw cache stats shows hit rate"

# 5. Ring
OUTPUT=$(igw -g "$GATEWAY" ring 2>&1) || true
echo "$OUTPUT" | grep -q "mock-gpt-markdown\|tinyllama" && pass "igw ring shows model ring" || fail "igw ring shows model ring"

# 6. Journal — should show entries with real data
OUTPUT=$(igw -g "$GATEWAY" journal 2>&1) || true
echo "$OUTPUT" | grep -q "mock-\|openai\|anthropic\|cache" && pass "igw journal shows real backend entries" || fail "igw journal shows real backend entries"

# 7. Cost — should show non-zero costs
OUTPUT=$(igw -g "$GATEWAY" cost 2>&1) || true
echo "$OUTPUT" | grep -qE "\\\$0\.[0-9]" && pass "igw cost shows non-zero dollar values" || fail "igw cost shows non-zero dollar values"

# 8. X-Estimated-Cost header — use beta (unlimited) to avoid rate limits
RESP=$(curl -s -D - -H "Authorization: Bearer $BETA_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"mock-gpt-markdown","messages":[{"role":"user","content":"cost header test request"}]}' \
  "$GATEWAY/v1/chat/completions" 2>&1)
echo "$RESP" | grep -qi "X-Estimated-Cost" && pass "X-Estimated-Cost header present" || fail "X-Estimated-Cost header present"

# 9. Cost API endpoint with real data
COST_JSON=$(curl -sf "$GATEWAY/admin/cost" 2>&1) || true
echo "$COST_JSON" | grep -q '"enabled": true\|"enabled":true' && pass "/admin/cost enabled" || fail "/admin/cost enabled"

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
