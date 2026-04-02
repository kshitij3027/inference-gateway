#!/usr/bin/env bash
set -euo pipefail

# Rolling restart: stop/start one gateway at a time with zero dropped requests.
# Usage: ./scripts/rolling-restart.sh [--build]
#
# The script:
#   1. Stops one instance (SIGTERM -> graceful drain up to 15s via stop_grace_period)
#   2. Rebuilds and starts it
#   3. Waits for /health to return 200
#   4. Moves to the next instance
#
# During restart, Nginx passive health checks route traffic to remaining instances.

BUILD_FLAG=""
if [[ "${1:-}" == "--build" ]]; then
    BUILD_FLAG="--build"
fi

INSTANCES="gateway-1 gateway-2 gateway-3"
HEALTH_TIMEOUT=30
HEALTH_INTERVAL=1

echo "Starting rolling restart..."

for instance in $INSTANCES; do
    echo ""
    echo "=== Restarting $instance ==="

    # Step 1: Stop the instance (sends SIGTERM, waits stop_grace_period)
    echo "  Stopping $instance..."
    docker compose stop "$instance"

    # Step 2: Start (optionally rebuild)
    echo "  Starting $instance..."
    docker compose up -d $BUILD_FLAG "$instance"

    # Step 3: Wait for healthy
    echo "  Waiting for $instance to become healthy..."
    healthy=false
    for i in $(seq 1 "$HEALTH_TIMEOUT"); do
        if docker compose exec -T "$instance" curl -sf http://localhost:8080/health > /dev/null 2>&1; then
            echo "  $instance is healthy (${i}s)"
            healthy=true
            break
        fi
        sleep "$HEALTH_INTERVAL"
    done

    if [ "$healthy" = false ]; then
        echo "  FAIL: $instance did not become healthy within ${HEALTH_TIMEOUT}s"
        echo "  Aborting rolling restart to prevent further disruption."
        exit 1
    fi

    echo "  $instance restarted successfully"
done

echo ""
echo "Rolling restart complete. All instances healthy."
