#!/bin/bash
#SBATCH --job-name=pww-snellius-flower
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Federated DiLoCo training on Snellius (NVIDIA H100) using Flower + FedMom and DARL data partitioning.

set -euo pipefail

: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    exit 1
fi
source "${PWW_ROOT}/env.sh"

export PYTHONNOUSERSITE=0

# Fix Snellius XALT module OpenSSL 3 shared library conflict
module unload XALT 2>/dev/null || true
unset LD_PRELOAD
if [[ -d "/sw/arch/RHEL9/EB_production/2024/software/OpenSSL/3/lib64" ]]; then
    export LD_LIBRARY_PATH="/sw/arch/RHEL9/EB_production/2024/software/OpenSSL/3/lib64:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v "/opt/xalt" | paste -sd:)
fi

# Ensure forked Flower repository (fedmom-strategy branch) is installed
FLOWER_REPO="${FLOWER_REPO:-git+https://github.com/Zare2001/flower.git@fedmom-strategy#subdirectory=framework}"
if ! pww_run python3 -c "import flwr" 2>/dev/null; then
    pww_run python3 -m pip install --user "${FLOWER_REPO}"
fi

CENTRAL_IP="${CENTRAL_IP:-145.38.206.143}"
DARL_PORT="${DARL_PORT:-29510}"
FLOWER_PORT="${FLOWER_PORT:-29511}"

# Resolve DARL_TOKEN if not explicitly set
if [[ -z "${DARL_TOKEN:-}" ]]; then
    if [[ -s "${PWW_ROOT}/runs/darl/token" ]]; then
        export DARL_TOKEN="$(cat "${PWW_ROOT}/runs/darl/token")"
    elif [[ -s "${PWW_ROOT}/runs/central/darl/token" ]]; then
        export DARL_TOKEN="$(cat "${PWW_ROOT}/runs/central/darl/token")"
    fi
fi

echo "Starting Snellius Flower Client -> Central Node IP ${CENTRAL_IP} (DARL: ${DARL_PORT}, Flower: ${FLOWER_PORT})"

srun "${PWW_ROOT}/scripts/task_wrapper.sh" \
    python3 -m pww.train_flower \
        --central-ip "${CENTRAL_IP}" \
        --darl-port "${DARL_PORT}" \
        ${DARL_TOKEN:+--darl-token "${DARL_TOKEN}"} \
        --flower-port "${FLOWER_PORT}" \
        --cluster-id "snellius" \
        --config "${PWW_ROOT}/configs/cifar10_resnet18_diloco.yaml" \
        "$@"
