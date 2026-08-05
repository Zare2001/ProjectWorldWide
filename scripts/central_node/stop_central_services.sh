#!/bin/bash
# Stop central node services gracefully

set -euo pipefail

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PWW_SITE="${PWW_SITE:-central}"
source "${PWW_ROOT}/env.sh"

STATE_DIR="${PWW_OUTPUT_DIR:-${PWW_ROOT}/runs}/central"
FLOWER_PID_FILE="${STATE_DIR}/flower.pid"

echo "Stopping DARL Lease Coordinator..."
"${PWW_ROOT}/scripts/darl_coordinator.sh" stop || true

echo "Stopping Flower Aggregator Server..."
if [[ -f "${FLOWER_PID_FILE}" ]]; then
    PID="$(cat "${FLOWER_PID_FILE}")"
    if kill -0 "${PID}" 2>/dev/null; then
        kill "${PID}"
        echo "Stopped Flower server PID ${PID}."
    fi
    rm -f "${FLOWER_PID_FILE}"
fi

echo "Central node services stopped."
