#!/usr/bin/env bash
set -euo pipefail

# Phase 17 validation: Terraform + PRODUCTION.md
# Runs terraform validate in Docker and checks PRODUCTION.md sections

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

PASS=0
FAIL=0

pass() { echo -e "${GREEN}PASS${NC}: $1"; PASS=$((PASS + 1)); }
fail() { echo -e "${RED}FAIL${NC}: $1"; FAIL=$((FAIL + 1)); }

echo "=== Terraform Validation ==="

# Run terraform init + validate in a single Docker container
# Uses entrypoint override to run both commands sequentially
VALIDATE_OUTPUT=$(docker run --rm \
  -v "${PROJECT_ROOT}/terraform:/workspace" \
  -w /workspace \
  --entrypoint sh \
  hashicorp/terraform:latest \
  -c "terraform init -backend=false -no-color 2>&1 && terraform validate -no-color 2>&1" 2>&1) || true

if echo "$VALIDATE_OUTPUT" | grep -q "successfully initialized"; then
  pass "terraform init"
else
  fail "terraform init"
  echo "$VALIDATE_OUTPUT"
fi

if echo "$VALIDATE_OUTPUT" | grep -q "Success"; then
  pass "terraform validate"
else
  fail "terraform validate"
  echo "$VALIDATE_OUTPUT"
fi

# Clean up .terraform directory created by init on the mounted volume
rm -rf "${PROJECT_ROOT}/terraform/.terraform" "${PROJECT_ROOT}/terraform/.terraform.lock.hcl"

echo ""
echo "=== PRODUCTION.md Validation ==="

PROD_MD="${PROJECT_ROOT}/PRODUCTION.md"

if [ -f "$PROD_MD" ]; then
  pass "PRODUCTION.md exists"
else
  fail "PRODUCTION.md not found"
  echo ""
  echo "Results: ${PASS} passed, ${FAIL} failed"
  exit 1
fi

# Check required sections
REQUIRED_SECTIONS=(
  "## Local vs Production Comparison"
  "## Cost Estimates"
  "## Redis Cluster Guidance"
  "## Scaling Strategy"
  "## Monitoring in Production"
)

for section in "${REQUIRED_SECTIONS[@]}"; do
  if grep -q "$section" "$PROD_MD"; then
    pass "Section found: $section"
  else
    fail "Section missing: $section"
  fi
done

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] || exit 1
