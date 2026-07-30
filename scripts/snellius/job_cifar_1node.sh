#!/bin/bash
#SBATCH --job-name=pww-cifar-1node
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --distribution=block:block
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# ResNet on CIFAR-10 across one Snellius GPU node (4 GPUs).
#
#   sbatch scripts/snellius/job_cifar_1node.sh --config configs/cifar10_resnet18.yaml
#
# On gpu_a100 instead (72 cores rather than 64, so 18 per rank):
#
#   sbatch -p gpu_a100 --cpus-per-task=18 scripts/snellius/job_cifar_1node.sh
#
# On accuracy comparability across sites: this node has 4 ranks where LUMI has 8,
# so at the same --batch-size the global batch is halved and train_cifar scales
# the LR accordingly. That is the correct behaviour, but it means results are
# only directly comparable to a LUMI run if you match the GLOBAL batch, e.g.
# --batch-size 256 here against --batch-size 128 on LUMI.

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

srun --cpu-bind="$(pww_cpu_bind)" \
    "${PWW_LAUNCH[@]}" \
        "${PWW_ROOT}/scripts/task_wrapper.sh" \
            python3 -m pww.train_cifar \
                --run-name "cifar10-snellius-1node-${SLURM_JOB_ID}" \
                --epochs 30 \
                --batch-size 256 \
                --parallel ddp \
                "$@"
