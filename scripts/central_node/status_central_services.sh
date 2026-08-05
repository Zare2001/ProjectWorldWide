#!/bin/bash
# Status and health check for central node services (DARL on 29510, Flower on 29511)

set -euo pipefail

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PWW_SITE="${PWW_SITE:-central}"
source "${PWW_ROOT}/env.sh"

DARL_PORT="${DARL_PORT:-29510}"
FLOWER_PORT="${FLOWER_PORT:-29511}"
STATE_DIR="${PWW_OUTPUT_DIR:-${PWW_ROOT}/runs}/central"

echo "=== DARL Coordinator Status (Port ${DARL_PORT}) ==="
"${PWW_ROOT}/scripts/darl_coordinator.sh" status || true

echo ""
echo "=== DARL Health Endpoint Check ==="
curl -sS "http://127.0.0.1:${DARL_PORT}/health" || echo "DARL HTTP endpoint unreachable on port ${DARL_PORT}"

echo ""
echo "=== Flower Aggregator Server Status (Port ${FLOWER_PORT}) ==="
FLOWER_PID_FILE="${STATE_DIR}/flower.pid"
if [[ -f "${FLOWER_PID_FILE}" ]] && kill -0 "$(cat "${FLOWER_PID_FILE}")" 2>/dev/null; then
    echo "Flower server is RUNNING (PID $(cat "${FLOWER_PID_FILE}"))."
else
    echo "Flower server is STOPPED."
fi

echo ""
echo "=== Open Port Listener Check ==="
ss -tulpn 2>/dev/null | grep -E "29510|29511" || true
