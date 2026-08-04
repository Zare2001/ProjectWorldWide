#!/bin/bash
#SBATCH --job-name=pww-cifar-diloco
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
# DiLoCo on CIFAR-10: k model replicas that only talk every H steps.
#
#   sbatch scripts/snellius/job_cifar_diloco.sh                       # k=2 x 2 GPUs
#   sbatch scripts/snellius/job_cifar_diloco.sh --diloco-replicas 4   # k=4 x 1 GPU
#
# The Snellius path itself is verified end to end; the headers above match
# job_cifar_1node.sh. What has NOT been run is DiLoCo on this site specifically --
# it is verified on LUMI (92.41% against 93.35% for DDP) and by
# tests/test_diloco_gloo.py, and nothing in diloco.py is site-dependent, so this
# is expected to work rather than known to.
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

# NCCL found InfiniBand unaided on the verified multi-node runs (133.1 GB/s across
# two nodes), so this stays off. If the smoke test ever reports single-digit GB/s
# across nodes it fell back to TCP -- the interfaces here are ibp*/mlx5, not ib0.
#export NCCL_SOCKET_IFNAME=ibp

echo "nodes: ${SLURM_JOB_NUM_NODES} | ranks: ${SLURM_NTASKS} | master: ${MASTER_ADDR}"

srun --cpu-bind="$(pww_cpu_bind)" \
    "${PWW_LAUNCH[@]}" \
        "${PWW_ROOT}/scripts/task_wrapper.sh" \
            python3 -m pww.train_cifar \
                --run-name "cifar10-diloco-snellius-${SLURM_JOB_ID}" \
                --config "${PWW_ROOT}/configs/cifar10_resnet18_diloco.yaml" \
                --batch-size 256 \
                "$@"
