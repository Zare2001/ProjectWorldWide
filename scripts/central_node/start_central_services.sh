#!/bin/bash
# Start central node services: DARL Coordinator (port 29510) and Flower Aggregator (port 29511)
# Run on the central Ubuntu cloud VM (192.168.2.176).

set -euo pipefail

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    exit 1
fi

source "${PWW_ROOT}/env.sh"

DARL_PORT="${DARL_PORT:-29510}"
FLOWER_PORT="${FLOWER_PORT:-29511}"
STATE_DIR="${PWW_OUTPUT_DIR:-${PWW_ROOT}/runs}/central"

mkdir -p "${STATE_DIR}"

DARL_PID_FILE="${STATE_DIR}/darl.pid"
FLOWER_PID_FILE="${STATE_DIR}/flower.pid"

echo "========================================================="
echo " Starting Central Node Aggregator Services"
echo " Central Node Host: $(hostname -f 2>/dev/null || hostname)"
echo " Central Node Public IP: 145.38.206.143"
echo " DARL Port:   ${DARL_PORT}"
echo " Flower Port: ${FLOWER_PORT}"
echo " State Dir:   ${STATE_DIR}"
echo "========================================================="

# 1. Start DARL Coordinator
if [[ -f "${DARL_PID_FILE}" ]] && kill -0 "$(cat "${DARL_PID_FILE}")" 2>/dev/null; then
    echo "DARL coordinator already running (PID $(cat "${DARL_PID_FILE}"))."
else
    echo "Starting DARL Lease Coordinator on port ${DARL_PORT}..."
    DARL_PORT="${DARL_PORT}" "${PWW_ROOT}/scripts/darl_coordinator.sh" start \
        --num-samples 1000000 \
        --block-size 10000 > "${STATE_DIR}/darl.log" 2>&1 &
    echo "DARL started."
fi

# 2. Start Flower Aggregator Server
if [[ -f "${FLOWER_PID_FILE}" ]] && kill -0 "$(cat "${FLOWER_PID_FILE}")" 2>/dev/null; then
    echo "Flower server already running (PID $(cat "${FLOWER_PID_FILE}"))."
else
    echo "Starting Flower Aggregator Server on port ${FLOWER_PORT}..."
    nohup python3 -m pww.central.server \
        --config "${PWW_ROOT}/configs/central_aggregator.yaml" \
        --port "${FLOWER_PORT}" > "${STATE_DIR}/flower.log" 2>&1 &
    echo $! > "${FLOWER_PID_FILE}"
    echo "Flower server started (PID $(cat "${FLOWER_PID_FILE}"))."
fi

echo ""
echo "Central node services successfully launched!"
echo "Check status anytime with: ./scripts/central_node/status_central_services.sh"
