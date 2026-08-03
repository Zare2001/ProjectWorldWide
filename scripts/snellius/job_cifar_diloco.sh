#!/bin/bash
#SBATCH --job-name=pww-cifar-diloco
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# DiLoCo on CIFAR-10: k model replicas that only talk every H steps.
#
#   sbatch scripts/snellius/job_cifar_diloco.sh                       # k=2 x 2 GPUs
#   sbatch scripts/snellius/job_cifar_diloco.sh --diloco-replicas 4   # k=4 x 1 GPU
#
# !! UNVERIFIED -- like every other script under scripts/snellius/, this was
#    written from SURF documentation. Run ./scripts/siteinfo.sh and correct
#    sites/snellius.sh first. The DiLoCo code itself is site-independent and is
#    verified on LUMI plus by tests/test_diloco_gloo.py, so the risk here is the
#    partition/GPU/core values above, not the algorithm.
#
# k must divide the number of ranks, which is 4 here against LUMI's 8 -- so the
# usable values are k=1,2,4 rather than k=1,2,4,8. To compare a run against LUMI,
# match k *and* the per-replica global batch: k=2 here gives 2 ranks per replica
# where k=2 on LUMI gives 4, so double --batch-size to keep the replica batch
# equal.
#
# One replica per node is the layout DiLoCo is for, since it removes inter-node
# traffic except once every H steps:
#
#   sbatch --nodes=4 scripts/snellius/job_cifar_diloco.sh --diloco-replicas 4

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

# [VERIFY] InfiniBand interface name. `ibstat`/`ip link` on a GPU node will say;
# NCCL usually picks it correctly on its own, so this is commented out until a
# multi-node run shows single-digit GB/s in the smoke test.
#export NCCL_SOCKET_IFNAME=ib

echo "nodes: ${SLURM_JOB_NUM_NODES} | ranks: ${SLURM_NTASKS} | master: ${MASTER_ADDR}"

srun --cpu-bind="$(pww_cpu_bind)" \
    "${PWW_LAUNCH[@]}" \
        "${PWW_ROOT}/scripts/task_wrapper.sh" \
            python3 -m pww.train_cifar \
                --run-name "cifar10-diloco-snellius-${SLURM_JOB_ID}" \
                --config "${PWW_ROOT}/configs/cifar10_resnet18_diloco.yaml" \
                --batch-size 256 \
                "$@"
