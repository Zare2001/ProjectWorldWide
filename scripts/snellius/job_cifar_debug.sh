#!/bin/bash
#SBATCH --job-name=pww-cifar-debug
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Single GPU, 2 short epochs. The fastest way to check that a code change runs
# at all.
#
#   sbatch scripts/snellius/job_cifar_debug.sh
#
# Why this exists on a cluster that is usually 100% allocated: Snellius lets
# SINGLE-node jobs take a fraction of a node, and the smallest unit here is
# 1 GPU + 16 cores + 180 GiB. That fraction is almost always free even when
# every node is busy -- at the time of writing, 23 of 88 H100 nodes could have
# started this immediately while ZERO could start a 4-GPU job.
#
# The same trick does NOT work for multi-node jobs. Snellius rejects a
# multi-node GPU job that asks for fewer than all 4 GPUs per node:
#
#   "You've requested less than the maximum amount of GPUs for your multi-node
#    job. If that's intentional, use --exclusive. You will be charged for all
#    GPUs, including the ones that you don't use."
#
# So there is no cheap multi-node run; scripts/snellius/job_cifar_multinode.sh
# takes whole nodes because that is the only shape allowed.
#
# Note --parallel single: with one rank there is no process group to form, which
# keeps this a test of the training code rather than of the interconnect. Use
# job_smoke.sh for the collectives.

set -euo pipefail

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
