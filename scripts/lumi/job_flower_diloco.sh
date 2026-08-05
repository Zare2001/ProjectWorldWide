#!/bin/bash
#SBATCH --job-name=pww-lumi-flower
#SBATCH --account=project_462000226
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Federated DiLoCo training on LUMI (AMD MI250X) using Flower + FedMom and DARL data partitioning.

set -euo pipefail

: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    exit 1
fi
source "${PWW_ROOT}/env.sh"

export PYTHONNOUSERSITE=0
export SINGULARITYENV_PYTHONNOUSERSITE=0

export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DEFAULT_CQ_SIZE=131072

# Ensure forked Flower repository (fedmom-strategy branch) is installed inside container
FLOWER_REPO="${FLOWER_REPO:-git+https://github.com/Zare2001/flower.git@fedmom-strategy#subdirectory=framework}"
if ! pww_run python3 -c "import flwr" 2>/dev/null; then
    pww_run python3 -m pip install --user "${FLOWER_REPO}"
fi

CENTRAL_IP="${CENTRAL_IP:-145.38.206.143}"
DARL_PORT="${DARL_PORT:-29510}"
FLOWER_PORT="${FLOWER_PORT:-29512}"

# Resolve DARL_TOKEN if not explicitly set
if [[ -z "${DARL_TOKEN:-}" ]]; then
    if [[ -s "${PWW_ROOT}/runs/darl/token" ]]; then
        export DARL_TOKEN="$(cat "${PWW_ROOT}/runs/darl/token")"
    elif [[ -s "${PWW_ROOT}/runs/central/darl/token" ]]; then
        export DARL_TOKEN="$(cat "${PWW_ROOT}/runs/central/darl/token")"
    fi
fi
export SINGULARITYENV_DARL_TOKEN="${DARL_TOKEN:-}"

echo "Starting LUMI Flower Client -> Central Node IP ${CENTRAL_IP} (DARL: ${DARL_PORT}, Flower: ${FLOWER_PORT})"

srun --cpu-bind="$(pww_cpu_bind)" \
    "${PWW_LAUNCH[@]}" \
        "${PWW_ROOT}/scripts/task_wrapper.sh" \
            python3 -m pww.train_flower \
                --central-ip "${CENTRAL_IP}" \
                --darl-port "${DARL_PORT}" \
                ${DARL_TOKEN:+--darl-token "${DARL_TOKEN}"} \
                --flower-port "${FLOWER_PORT}" \
                --cluster-id "lumi" \
                --config "${PWW_ROOT}/configs/cifar10_resnet18_diloco.yaml" \
                "$@"
