#!/bin/bash
# Stop central node services gracefully

set -euo pipefail

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PWW_SITE="${PWW_SITE:-central}"
source "${PWW_ROOT}/env.sh"

STATE_DIR="${PWW_OUTPUT_DIR:-${PWW_ROOT}/runs}/central"
FLOWER_PID_FILE="${STATE_DIR}/flower.pid"
BLOB_PID_FILE="${STATE_DIR}/blob.pid"

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

if [[ -f "${BLOB_PID_FILE}" ]]; then
    PID="$(cat "${BLOB_PID_FILE}")"
    if kill -0 "${PID}" 2>/dev/null; then
        echo "Stopping blob store (PID ${PID})..."
        kill "${PID}" 2>/dev/null || true
    fi
    rm -f "${BLOB_PID_FILE}"
fi

# Ensure the coordinator, aggregator and blob store ports are completely freed.
# Note what is NOT deleted: runs/central/global holds the durable global model,
# momentum buffer and membership record, and a restart resumes from it. Remove that
# directory only when you intend to start a new run from scratch.
fuser -k 29510/tcp 2>/dev/null || true
fuser -k 29511/tcp 2>/dev/null || true
fuser -k 29512/tcp 2>/dev/null || true

echo "Central node services stopped."
