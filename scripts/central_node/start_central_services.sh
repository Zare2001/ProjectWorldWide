#!/bin/bash
# Start central node services: DARL Coordinator (port 29510) and Flower Aggregator (port 29511)
# Run on the central Ubuntu cloud VM.

set -euo pipefail

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    exit 1
fi

export PWW_SITE="${PWW_SITE:-central}"
source "${PWW_ROOT}/env.sh"

DARL_PORT="${DARL_PORT:-29510}"
FLOWER_PORT="${FLOWER_PORT:-29512}"
STATE_DIR="${PWW_OUTPUT_DIR:-${PWW_ROOT}/runs}/central"

mkdir -p "${STATE_DIR}"

DARL_PID_FILE="${STATE_DIR}/darl.pid"
FLOWER_PID_FILE="${STATE_DIR}/flower.pid"
VENV_DIR="${STATE_DIR}/.venv"
FLOWER_REPO="${FLOWER_REPO:-git+https://github.com/Zare2001/flower.git@fedmom-strategy#subdirectory=framework}"

echo "========================================================="
echo " Starting Central Node Aggregator Services"
echo " Central Node Host: $(hostname -f 2>/dev/null || hostname)"
echo " Central Node Public IP: 145.38.206.143"
echo " DARL Port:   ${DARL_PORT}"
echo " Flower Port: ${FLOWER_PORT}"
echo " State Dir:   ${STATE_DIR}"
echo "========================================================="

# 0. Setup Environment: uv -> venv -> lightweight venv fallback
PYTHON_BIN="python3"

if ! "${VENV_DIR}/bin/python3" -c "import flwr, torch" 2>/dev/null; then
    echo "Installing Flower & PyTorch into ${VENV_DIR}..."
    if command -v uv >/dev/null 2>&1; then
        echo "Using uv..."
        uv pip install --system "${FLOWER_REPO}" torch --extra-index-url https://download.pytorch.org/whl/cpu 2>/dev/null || true
    elif python3 -m venv "${VENV_DIR}" >/dev/null 2>&1 && [[ -x "${VENV_DIR}/bin/pip" ]]; then
        echo "Using python3 venv..."
        "${VENV_DIR}/bin/pip" install "${FLOWER_REPO}" torch --extra-index-url https://download.pytorch.org/whl/cpu
        PYTHON_BIN="${VENV_DIR}/bin/python3"
    else
        echo "Creating lightweight venv and installing Flower & PyTorch into ${VENV_DIR}..."
        python3 -m venv --without-pip "${VENV_DIR}" >/dev/null 2>&1 || true
        PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        SITE_PKGS="${VENV_DIR}/lib/python${PY_VER}/site-packages"
        mkdir -p "${SITE_PKGS}"
        python3 -m pip install --target="${SITE_PKGS}" --break-system-packages "${FLOWER_REPO}" torch --extra-index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
        python3 -m pip install --target="${SITE_PKGS}" "${FLOWER_REPO}" torch --extra-index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
        pip install --target="${SITE_PKGS}" "${FLOWER_REPO}" torch --extra-index-url https://download.pytorch.org/whl/cpu
        PYTHON_BIN="${VENV_DIR}/bin/python3"
    fi
elif [[ -x "${VENV_DIR}/bin/python3" ]]; then
    PYTHON_BIN="${VENV_DIR}/bin/python3"
fi

NUM_SAMPLES="${NUM_SAMPLES:-50000}"
BLOCK_SIZE="${BLOCK_SIZE:-1000}"

# 1. Start DARL Coordinator
if [[ -f "${DARL_PID_FILE}" ]] && kill -0 "$(cat "${DARL_PID_FILE}")" 2>/dev/null; then
    echo "DARL coordinator already running (PID $(cat "${DARL_PID_FILE}"))."
else
    echo "Starting DARL Lease Coordinator on port ${DARL_PORT} (samples: ${NUM_SAMPLES}, block_size: ${BLOCK_SIZE})..."
    DARL_PORT="${DARL_PORT}" DARL_TOKEN="" "${PWW_ROOT}/scripts/darl_coordinator.sh" start \
        --num-samples "${NUM_SAMPLES}" \
        --block-size "${BLOCK_SIZE}" \
        --fresh > "${STATE_DIR}/darl.log" 2>&1 &
    echo "DARL started."
fi

# 2. Start Flower Aggregator Server
if [[ -f "${FLOWER_PID_FILE}" ]] && kill -0 "$(cat "${FLOWER_PID_FILE}")" 2>/dev/null; then
    echo "Flower server already running (PID $(cat "${FLOWER_PID_FILE}"))."
else
    echo "Starting Flower Aggregator Server on port ${FLOWER_PORT}..."
    nohup "${PYTHON_BIN}" -m pww.central.server \
        --config "${PWW_ROOT}/configs/central_aggregator.yaml" \
        --port "${FLOWER_PORT}" > "${STATE_DIR}/flower.log" 2>&1 &
    echo $! > "${FLOWER_PID_FILE}"
    echo "Flower server started (PID $(cat "${FLOWER_PID_FILE}"))."
fi

echo ""
echo "Central node services successfully launched!"
echo "Check status anytime with: ./scripts/central_node/status_central_services.sh"
