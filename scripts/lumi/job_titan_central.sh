#!/bin/bash
#SBATCH --job-name=pww-lumi-titan-central
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# LUMI single-node central baseline run (Qwen3 0.6B) matching DARL data sequence.
#
#   CONFIG=configs/titan/qwen3_0.6b_c4_central.toml sbatch scripts/lumi/job_titan_central.sh
#   sbatch -A $PWW_ACCOUNT scripts/lumi/job_titan_central.sh
#
# Unlike the distributed DiLoCo run (job_titan_diloco.sh), this script runs a continuous
# non-distributed training loop (flower.enable = false) on 1 full node (8 GCDs).
# It uses DARL dataset leasing (pww_tokens) with space_seed = 42 so that training
# processes tokens in the exact same pseudo-random block-permuted sequence as the DiLoCo run.
#
# Automatically spawns a temporary local DARL coordinator on 127.0.0.1 during the job.

set -euo pipefail

: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    exit 1
fi
source "${PWW_ROOT}/env.sh"

TITAN_SIF="${PWW_TITAN_SIF:-${PWW_SCRATCH}/containers/pww-titan.sif}"
if [[ ! -r "${TITAN_SIF}" ]]; then
    echo "ERROR: no torchtitan container at ${TITAN_SIF}" >&2
    echo "build it once: sbatch scripts/lumi/build_titan_container.sh" >&2
    exit 1
fi

export PWW_CONTAINER="${TITAN_SIF}"

export MIOPEN_USER_DB_PATH="${PWW_TMPDIR}/miopen-${SLURM_JOB_ID:-0}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
export SINGULARITYENV_MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH}"
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_DISABLE_HOST_REGISTER=1
for var in NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL FI_CXI_DISABLE_HOST_REGISTER; do
    export "SINGULARITYENV_${var}=${!var}"
done

CONFIG="${CONFIG:-${PWW_ROOT}/configs/titan/qwen3_0.6b_c4_central.toml}"
TOKENIZER="${TOKENIZER:-${PWW_DATA_DIR}/tokenizers/tokenizer-128k}"
SHARDS="${SHARDS:-${PWW_DATA_DIR}/c4-tokenizer-128k-2048}"

DARL_PORT="${PWW_DARL_PORT:-$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo "29510")}"
CENTRAL_IP="127.0.0.1"

if [[ -z "${DARL_TOKEN:-}" ]]; then
    export DARL_TOKEN="$(head -c 24 /dev/urandom | base64 | tr -d '/+=')"
fi
export SINGULARITYENV_DARL_TOKEN="${DARL_TOKEN}"

NUM_SAMPLES=$(python3 - "${SHARDS}" <<'EOF'
import json, pathlib, sys
raw = json.loads((pathlib.Path(sys.argv[1]) / "manifest.json").read_text())
print(int(raw["num_windows"]))
EOF
) || {
    echo "ERROR: could not read num_windows from ${SHARDS}/manifest.json." >&2
    echo "The baseline must use the same block space as the DiLoCo run, so this is fatal" >&2
    echo "rather than defaulted -- a different sample count is a different data sequence." >&2
    exit 1
}
echo "darl: block space from ${SHARDS}/manifest.json -> ${NUM_SAMPLES} windows"

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
    --site lumi \
    --nproc "${SLURM_GPUS_PER_NODE:-8}" \
    -- "$@" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
