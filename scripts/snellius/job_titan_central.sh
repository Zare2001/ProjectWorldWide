#!/bin/bash
#SBATCH --job-name=pww-snellius-titan-central
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Snellius single-node central baseline run (Qwen3 0.6B) matching DARL data sequence.
#
#   CONFIG=configs/titan/qwen3_0.6b_c4_central.toml sbatch scripts/snellius/job_titan_central.sh
#   sbatch scripts/snellius/job_titan_central.sh
#
# Unlike the distributed DiLoCo run (job_titan_diloco.sh), this script runs a continuous
# non-distributed training loop (flower.enable = false) on 1 node (4 GPUs).
# It uses DARL dataset leasing (pww_tokens) with space_seed = 42 so that training
# processes tokens in the exact same pseudo-random block-permuted sequence as the DiLoCo run.
#
# If CENTRAL_IP is set and reachable, it leases from the central coordinator.
# Otherwise (or if LOCAL_DARL=1), it automatically spawns a temporary local DARL coordinator
# on 127.0.0.1 during the job.

set -euo pipefail

: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${PWW_ROOT}/env.sh"

export PYTHONNOUSERSITE=0

# Snellius' XALT wrapper injects an OpenSSL 3 shared library that conflicts with
# the one torch links against.
module unload XALT 2>/dev/null || true
unset LD_PRELOAD
if [[ -d "/sw/arch/RHEL9/EB_production/2024/software/OpenSSL/3/lib64" ]]; then
    export LD_LIBRARY_PATH="/sw/arch/RHEL9/EB_production/2024/software/OpenSSL/3/lib64:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v "/opt/xalt" | paste -sd:)
fi

PWW_TITAN_VENV="${PWW_TITAN_VENV:-${HOME}/venvs/pww-titan-snellius}"
if [[ -r "${PWW_TITAN_VENV}/bin/activate" ]]; then
    source "${PWW_TITAN_VENV}/bin/activate"
    export PWW_VENV="${PWW_TITAN_VENV}"
    echo "activated torchtitan venv: ${PWW_TITAN_VENV}"
else
    echo "ERROR: no torchtitan venv at ${PWW_TITAN_VENV}" >&2
    echo "build it once from a login node: ./scripts/titan/setup_venv_snellius.sh" >&2
    exit 1
fi

# Always run a local DARL coordinator on 127.0.0.1 for local standalone Snellius execution.
# Probe an unused local port to prevent collisions with any concurrent jobs.
DARL_PORT="${PWW_DARL_PORT:-$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo "29510")}"
CENTRAL_IP="127.0.0.1"

if [[ -z "${DARL_TOKEN:-}" ]]; then
    export DARL_TOKEN="$(head -c 24 /dev/urandom | base64 | tr -d '/+=')"
fi

# Extract num_samples from shards manifest
NUM_SAMPLES=$(python3 -c "from pww.titan.shards import read_manifest; print(read_manifest('${SHARDS}').num_samples)" 2>/dev/null || echo "1000000")

LOCAL_STATE_DIR="${PWW_ROOT}/outputs/darl_local_central_${SLURM_JOB_ID:-0}"
mkdir -p "${LOCAL_STATE_DIR}"

echo "darl: launching local coordinator on http://127.0.0.1:${DARL_PORT} (num_samples=${NUM_SAMPLES}, seed=42)..."
PYTHONPATH="${PWW_ROOT}/src:${PYTHONPATH:-}" python3 -m pww.darl.server \
    --num-samples "${NUM_SAMPLES}" \
    --block-size 1024 \
    --seed 42 \
    --port "${DARL_PORT}" \
    --token "${DARL_TOKEN}" \
    --state-dir "${LOCAL_STATE_DIR}" \
    --fresh > "${LOCAL_STATE_DIR}/coordinator.log" 2>&1 &
LOCAL_DARL_PID=$!
sleep 2

if ! kill -0 "${LOCAL_DARL_PID}" 2>/dev/null; then
    echo "ERROR: Local DARL coordinator failed to start. Last lines of log:" >&2
    tail -n 20 "${LOCAL_STATE_DIR}/coordinator.log" >&2
    exit 1
fi
echo "darl: local coordinator running (PID ${LOCAL_DARL_PID})"

cleanup() {
    echo "cleaning up central job process..."
    if [[ -n "${TRAIN_PID:-}" ]]; then
        kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    fi
    if [[ -n "${LOCAL_DARL_PID:-}" ]] && kill -0 "${LOCAL_DARL_PID}" 2>/dev/null; then
        echo "stopping local DARL coordinator (PID ${LOCAL_DARL_PID})..."
        kill -TERM "${LOCAL_DARL_PID}" 2>/dev/null || true
    fi
}
trap cleanup TERM INT EXIT

"${PWW_ROOT}/scripts/titan/run_train.sh" \
    --config "${CONFIG}" \
    --tokenizer "${TOKENIZER}" \
    --shards "${SHARDS}" \
    --central "${CENTRAL_IP}" \
    --darl-port "${DARL_PORT}" \
    --site snellius \
    --nproc "${SLURM_GPUS_PER_NODE:-4}" \
    -- "$@" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
