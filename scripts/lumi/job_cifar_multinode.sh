#!/bin/bash
#SBATCH --job-name=pww-cifar-multi
#SBATCH --account=project_462000226
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Same training run scaled across nodes -- 2 nodes x 8 GCDs = 16 ranks.
#
#   sbatch scripts/job_cifar_multinode.sh
#   sbatch --nodes=4 scripts/job_cifar_multinode.sh
#
# CIFAR-10 is far too small to benefit from this (50k images across 16 ranks is
# ~390 images per rank per epoch, so communication dominates). It exists to
# prove the multi-node path works before it matters for LLM training.

set -euo pipefail

# Submitted from the repo root, so SLURM_SUBMIT_DIR is it; PWW_ROOT overrides.
: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    echo "  submit from the repo root, or export PWW_ROOT=/path/to/ProjectWorldWide" >&2
    exit 1
fi
source "${PWW_ROOT}/env.sh"

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29500

# Cross-node specifics. Inside a node RCCL uses Infinity Fabric and none of this
# matters; across nodes it must go through libfabric on the Slingshot NICs.
export NCCL_SOCKET_IFNAME=hsn          # 4x 200 Gb/s NICs, not the mgmt interface
export NCCL_NET_GDR_LEVEL=PHB          # GPU-direct RDMA where the topology allows
export FI_CXI_DEFAULT_CQ_SIZE=131072   # default completion queue overflows at scale

echo "nodes: ${SLURM_JOB_NUM_NODES} | ranks: ${SLURM_NTASKS} | master: ${MASTER_ADDR}"

srun --cpu-bind="$(pww_cpu_bind)" \
    "${PWW_LAUNCH[@]}" \
        "${PWW_ROOT}/scripts/task_wrapper.sh" \
            python3 -m pww.train_cifar \
                --run-name "cifar10-${SLURM_JOB_NUM_NODES}node-${SLURM_JOB_ID}" \
                --epochs 30 \
                --batch-size 64 \
                --parallel ddp \
                "$@"
