#!/bin/bash
#SBATCH --job-name=pww-cifar-debug
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Single GCD, 2 short epochs. The fastest way to check that a code change runs
# at all -- dev-g normally starts in well under a minute.
#
#   sbatch scripts/job_cifar_debug.sh

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
                --run-name "cifar10-debug-${SLURM_JOB_ID}" \
                --parallel single \
                --epochs 2 \
                --max-steps-per-epoch 20 \
                --save-every 1 \
                --log-every 5 \
                "$@"
