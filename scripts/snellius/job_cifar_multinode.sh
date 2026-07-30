#!/bin/bash
#SBATCH --job-name=pww-cifar-multi
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Multi-node on Snellius: 2 nodes x 4 GPUs = 8 ranks.
#
#   sbatch scripts/snellius/job_cifar_multinode.sh
#   sbatch --nodes=4 scripts/snellius/job_cifar_multinode.sh
#
# !! UNVERIFIED -- run ./scripts/siteinfo.sh and confirm the values in the
#    job_smoke.sh header before trusting results.

set -euo pipefail

: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    echo "  submit from the repo root, or export PWW_ROOT=/path/to/ProjectWorldWide" >&2
    exit 1
fi
source "${PWW_ROOT}/env.sh"

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29500

# Cross-node collectives go over InfiniBand here, not Slingshot. If job_smoke.sh
# reports single-digit GB/s across nodes, NCCL fell back to TCP -- set the
# interface explicitly and check that the IB stack is visible in the job.
# export NCCL_SOCKET_IFNAME=ib0

echo "nodes: ${SLURM_JOB_NUM_NODES} | ranks: ${SLURM_NTASKS} | master: ${MASTER_ADDR}"

srun --cpu-bind="$(pww_cpu_bind)" \
    "${PWW_LAUNCH[@]}" \
        "${PWW_ROOT}/scripts/task_wrapper.sh" \
            python3 -m pww.train_cifar \
                --run-name "cifar10-snellius-${SLURM_JOB_NUM_NODES}node-${SLURM_JOB_ID}" \
                --epochs 30 \
                --batch-size 128 \
                --parallel ddp \
                "$@"
