#!/bin/bash
#SBATCH --job-name=pww-lumi-titan
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# torchtitan (Qwen3) + DARL + Flower/FedMom on LUMI (AMD MI250X, ROCm/RCCL).
#
#   CONFIG=configs/titan/qwen3_0.6b_smoke.toml sbatch scripts/lumi/job_titan_diloco.sh
#   sbatch -A $PWW_ACCOUNT scripts/lumi/job_titan_diloco.sh
#
# One task per node, torchrun forking 8 ranks -- one per GCD. LUMI's MI250X is two
# GCDs per physical card and torch addresses each as its own device, so 4 cards is
# 8 ranks, not 4.
#
# The container here is NOT LUMI's maintained one: that ships torch 2.7.1 and
# torchtitan needs >= 2.9. See scripts/titan/README.md and
# containers/titan-lumi.def.

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

# Replaces the PWW_LAUNCH env.sh set up for the torch 2.7.1 container.
PWW_LAUNCH=(singularity exec --rocm --bind "${PWW_SCRATCH}" "${TITAN_SIF}")
export PWW_LAUNCH

# MIOpen writes a kernel cache and defaults to $HOME, which on LUMI is a small
# quota and shared between concurrent jobs -- two jobs racing on the same cache
# database is a known hang. One cache per job, on scratch.
export MIOPEN_USER_DB_PATH="${PWW_TMPDIR}/miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
export SINGULARITYENV_MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH}"
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"

# Slingshot 11: RCCL needs to be told which interfaces to use, and host
# registration has to be off or large transfers fail on the CXI provider. Values
# carried over from the existing LUMI job scripts.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_DISABLE_HOST_REGISTER=1
for var in NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL FI_CXI_DISABLE_HOST_REGISTER; do
    export "SINGULARITYENV_${var}=${!var}"
done

CENTRAL_IP="${CENTRAL_IP:-145.38.206.143}"
CONFIG="${CONFIG:-${PWW_ROOT}/configs/titan/qwen3_0.6b_c4_diloco.toml}"
TOKENIZER="${TOKENIZER:-${PWW_DATA_DIR}/tokenizers/tokenizer-128k}"
SHARDS="${SHARDS:-${PWW_DATA_DIR}/c4-tokenizer-128k-2048}"

if [[ -z "${DARL_TOKEN:-}" ]]; then
    for candidate in "${PWW_ROOT}/runs/darl/token" "${PWW_ROOT}/runs/central/darl/token"; do
        [[ -s "${candidate}" ]] && export DARL_TOKEN="$(cat "${candidate}")" && break
    done
fi
export SINGULARITYENV_DARL_TOKEN="${DARL_TOKEN:-}"

# LUMI kills jobs at walltime, which for a long DiLoCo run is the normal way a job
# ends rather than an exception. Forwarding SIGTERM lets the DARL session release
# its uncommitted spans immediately instead of the other site waiting out a full
# lease TTL before it can pick them up.
trap 'echo "SIGTERM -- releasing DARL leases"; kill -TERM ${TRAIN_PID:-0} 2>/dev/null' TERM

"${PWW_ROOT}/scripts/titan/run_train.sh" \
    --config "${CONFIG}" \
    --tokenizer "${TOKENIZER}" \
    --shards "${SHARDS}" \
    --central "${CENTRAL_IP}" \
    --site lumi \
    --nproc "${SLURM_GPUS_PER_NODE:-8}" \
    -- "$@" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
