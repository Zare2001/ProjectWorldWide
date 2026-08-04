#!/bin/bash
#SBATCH --job-name=pww-cifar-1node
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
#
# ResNet-18 on CIFAR-10 across one full LUMI-G node (8 GCDs).
#
#   sbatch scripts/job_cifar_1node.sh
#   sbatch scripts/job_cifar_1node.sh --model resnet50 --epochs 50
#
# Anything passed after the script name is forwarded to train_cifar.
#
# Note on partitions: small-g is billed per-GPU and allows node sharing, so it
# usually queues faster than standard-g for single-node work. standard-g bills
# whole nodes -- use it only for multi-node runs.

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
# Derived from the job id rather than fixed: when two of your own jobs share a
# node -- routine on Snellius, where partial single-node allocations get packed
# together -- a fixed port makes the second job die in the TCPStore rendezvous
# with "address already in use". Kept below the ephemeral range (32768+) so it
# cannot clash with an outgoing connection either.
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 20000))

srun --cpu-bind="$(pww_cpu_bind)" \
    "${PWW_LAUNCH[@]}" \
        "${PWW_ROOT}/scripts/task_wrapper.sh" \
            python3 -m pww.train_cifar \
                --run-name "cifar10-1node-${SLURM_JOB_ID}" \
                --epochs 30 \
                --batch-size 128 \
                --parallel ddp \
                "$@"
