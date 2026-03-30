#!/usr/bin/env bash
set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
CONCURRENCY=50

echo "Seeding $CONCURRENCY concurrent requests to $GATEWAY_URL..."
start=$(date +%s)

# Use tenant-beta (no rate limits) with mock-gpt-markdown (instant response)
for i in $(seq 1 $CONCURRENCY); do
    curl -s -o /dev/null -w "req=%s status=%{http_code} time=%{time_total}s\n" \
        -X POST "$GATEWAY_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer test-beta-key" \
        -d "{\"model\": \"mock-gpt-markdown\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello request $i\"}]}" &
done

wait
end=$(date +%s)
elapsed=$((end - start))

echo ""
echo "Completed $CONCURRENCY requests in ${elapsed}s"
if [ "$elapsed" -gt 30 ]; then
    echo "FAIL: Seed took longer than 30s"
    exit 1
fi
echo "PASS: Seed completed within 30s"
