#!/bin/bash
# Stop central node services gracefully

set -euo pipefail

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PWW_SITE="${PWW_SITE:-central}"
source "${PWW_ROOT}/env.sh"

STATE_DIR="${PWW_OUTPUT_DIR:-${PWW_ROOT}/runs}/central"
FLOWER_PID_FILE="${STATE_DIR}/flower.pid"
BLOB_PID_FILE="${STATE_DIR}/blob.pid"

# The same three variables start_central_services.sh reads, because this script must
# stop THAT stack and no other. These used to be hardcoded below, which was invisible
# with one stack -- and with parallel stacks (RUNBOOK.md "Parallel experiments") it made
# `PWW_OUTPUT_DIR=...runs-churn ./stop_central_services.sh` kill the default stack's
# daemons on 29510-29512 as a side effect of stopping the churn one.
#
# Moving them to env vars was not enough: a stop invoked with PWW_OUTPUT_DIR but
# WITHOUT the port variables still swept 29510-29512 through the fuser -k below --
# observed killing the full arm's live aggregator mid-run while stopping the dclt
# stack. So the stack's own launch.env (written by start_central_services.sh at
# every successful launch) is the second source: explicit env wins, the recorded
# launch is next, and the bare defaults are last -- correct only for the one stack
# that actually runs on them.
LAUNCH_FILE="${STATE_DIR}/launch.env"
_recorded() { grep -oE "^$1=[0-9]+$" "${LAUNCH_FILE}" 2>/dev/null | cut -d= -f2; }
DARL_PORT="${DARL_PORT:-$(_recorded DARL_PORT || true)}"
FLOWER_PORT="${FLOWER_PORT:-$(_recorded FLOWER_PORT || true)}"
BLOB_PORT="${BLOB_PORT:-$(_recorded BLOB_PORT || true)}"
DARL_PORT="${DARL_PORT:-29510}"
FLOWER_PORT="${FLOWER_PORT:-29511}"
BLOB_PORT="${BLOB_PORT:-29512}"
echo "Ports to free for this stack: darl=${DARL_PORT} flower=${FLOWER_PORT} blob=${BLOB_PORT} (state: ${STATE_DIR})"

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
fuser -k "${DARL_PORT}/tcp" 2>/dev/null || true
fuser -k "${FLOWER_PORT}/tcp" 2>/dev/null || true
fuser -k "${BLOB_PORT}/tcp" 2>/dev/null || true

echo "Central node services stopped."
