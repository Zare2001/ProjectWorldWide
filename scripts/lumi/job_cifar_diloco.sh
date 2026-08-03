#!/bin/bash
#SBATCH --job-name=pww-cifar-diloco
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
# DiLoCo on CIFAR-10: k model replicas that only talk every H steps.
#
#   sbatch scripts/lumi/job_cifar_diloco.sh                       # k=2 x 4 GCDs
#   sbatch scripts/lumi/job_cifar_diloco.sh --diloco-replicas 4   # k=4 x 2 GCDs
#   sbatch scripts/lumi/job_cifar_diloco.sh --diloco-inner-steps 200
#
# Anything after the script name is forwarded to train_cifar.
#
# Choosing k on one node: k must divide the 8 GCDs. k=2 gives two replicas of 4
# GCDs each; within a replica gradients go over Infinity Fabric every step, and
# the two replicas exchange outer gradients every H steps. k=8 is one GCD per
# replica -- the most communication-avoidant layout, and the one where DiLoCo's
# accuracy cost is largest.
#
# For the layout DiLoCo is actually for -- replicas that are far apart -- run it
# multi-node with one replica per node, so *no* inter-node traffic happens except
# once every H steps:
#
#   sbatch --nodes=4 --partition=standard-g \
#       scripts/lumi/job_cifar_diloco.sh --diloco-replicas 4 --diloco-inner-steps 100
#
# CIFAR-10 is too small for DiLoCo to pay off (50k images means an epoch is a few
# hundred steps, so H=100 is most of an epoch). It exists to prove the mechanism
# before it matters for LLM training, where H in the hundreds is cheap.

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

# Needed only once the job spans nodes; harmless on one.
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=PHB
export FI_CXI_DEFAULT_CQ_SIZE=131072

echo "nodes: ${SLURM_JOB_NUM_NODES} | ranks: ${SLURM_NTASKS} | master: ${MASTER_ADDR}"

srun --cpu-bind="$(pww_cpu_bind)" \
    "${PWW_LAUNCH[@]}" \
        "${PWW_ROOT}/scripts/task_wrapper.sh" \
            python3 -m pww.train_cifar \
                --run-name "cifar10-diloco-${SLURM_JOB_ID}" \
                --config "${PWW_ROOT}/configs/cifar10_resnet18_diloco.yaml" \
                "$@"
