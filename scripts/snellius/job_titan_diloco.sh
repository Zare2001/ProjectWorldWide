#!/bin/bash
#SBATCH --job-name=pww-snellius-titan
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# torchtitan (Qwen3) + DARL + Flower/FedMom on Snellius (NVIDIA H100).
#
#   CONFIG=configs/titan/qwen3_0.6b_smoke.toml sbatch scripts/snellius/job_titan_diloco.sh
#   sbatch scripts/snellius/job_titan_diloco.sh                # the C4 DiLoCo run
#
# Unlike scripts/snellius/job_flower_diloco_llm.sh this asks for ONE task per node
# and lets torchrun fork the four ranks. torchtitan expects to own the process
# topology -- LOCAL_RANK, the rendezvous, the device mesh -- and srun spawning
# four independent tasks that each then try to be rank 0 of their own torchrun is
# the classic way to get four one-GPU jobs that never form a mesh.

set -euo pipefail

: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    exit 1
fi
source "${PWW_ROOT}/env.sh"

export PYTHONNOUSERSITE=0

# Snellius' XALT wrapper injects an OpenSSL 3 shared library that conflicts with
# the one torch links against, and LD_PRELOAD from the host does not resolve in
# every environment. Same treatment as the existing job scripts.
module unload XALT 2>/dev/null || true
unset LD_PRELOAD
if [[ -d "/sw/arch/RHEL9/EB_production/2024/software/OpenSSL/3/lib64" ]]; then
    export LD_LIBRARY_PATH="/sw/arch/RHEL9/EB_production/2024/software/OpenSSL/3/lib64:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v "/opt/xalt" | paste -sd:)
fi

# The torchtitan venv, NOT the torch 2.7.1 one env.sh activated. torchtitan needs
# torch >= 2.9; see scripts/titan/README.md for why both exist.
PWW_TITAN_VENV="${PWW_TITAN_VENV:-${HOME}/venvs/pww-titan-snellius}"
if [[ -r "${PWW_TITAN_VENV}/bin/activate" ]]; then
    source "${PWW_TITAN_VENV}/bin/activate"
    echo "activated torchtitan venv: ${PWW_TITAN_VENV}"
else
    echo "ERROR: no torchtitan venv at ${PWW_TITAN_VENV}" >&2
    echo "build it once from a login node: ./scripts/titan/setup_venv_snellius.sh" >&2
    exit 1
fi

CENTRAL_IP="${CENTRAL_IP:-145.38.206.143}"
CONFIG="${CONFIG:-${PWW_ROOT}/configs/titan/qwen3_0.6b_c4_diloco.toml}"
TOKENIZER="${TOKENIZER:-${PWW_DATA_DIR}/tokenizers/tokenizer-128k}"
SHARDS="${SHARDS:-${PWW_DATA_DIR}/c4-tokenizer-128k-2048}"

if [[ -z "${DARL_TOKEN:-}" ]]; then
    for candidate in "${PWW_ROOT}/runs/darl/token" "${PWW_ROOT}/runs/central/darl/token"; do
        [[ -s "${candidate}" ]] && export DARL_TOKEN="$(cat "${candidate}")" && break
    done
fi

# Releasing the DARL leases on SIGTERM returns the tail to the pool in
# milliseconds instead of after a full TTL, which is the difference between the
# other site idling for a quarter of an hour at every walltime kill and not idling
# at all. Slurm sends SIGTERM before SIGKILL, so forward it to torchrun's group.
trap 'echo "SIGTERM -- releasing DARL leases"; kill -TERM ${TRAIN_PID:-0} 2>/dev/null' TERM

"${PWW_ROOT}/scripts/titan/run_train.sh" \
    --config "${CONFIG}" \
    --tokenizer "${TOKENIZER}" \
    --shards "${SHARDS}" \
    --central "${CENTRAL_IP}" \
    --site snellius \
    --nproc "${SLURM_GPUS_PER_NODE:-4}" \
    -- "$@" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
