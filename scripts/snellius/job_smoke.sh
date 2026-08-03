#!/bin/bash
#SBATCH --job-name=pww-smoke
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --distribution=block:block
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Validate the distributed stack on one Snellius GPU node.
#
#   sbatch scripts/snellius/job_smoke.sh
#   sbatch scripts/snellius/job_smoke.sh --diloco-replicas 2   # also check DiLoCo
#
# On gpu_a100 instead (72 cores rather than 64, so 18 per rank):
#
#   sbatch -p gpu_a100 --cpus-per-task=18 scripts/snellius/job_smoke.sh
#
# Differences from the LUMI equivalent, all handled by sites/snellius.sh rather
# than by the training code:
#   * 4 ranks per node, not 8 (no GCD split on NVIDIA)
#   * no --mem=0; Snellius gives memory in proportion to the GPUs requested
#   * environment via a pip venv rather than a container (scripts/snellius/setup_venv.sh)
#   * no MIOpen cache workaround; that is a ROCm problem only

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

# InfiniBand rather than LUMI's Slingshot. NCCL autodetects it correctly here --
# the smoke test measures bandwidth, so a fallback to TCP shows up as
# single-digit GB/s rather than as silent slowness. Only set this if that
# happens, and note the interfaces are ibp*/mlx5 on these nodes, not ib0.
# export NCCL_SOCKET_IFNAME=ibp

srun --cpu-bind="$(pww_cpu_bind)" \
    "${PWW_LAUNCH[@]}" \
        "${PWW_ROOT}/scripts/task_wrapper.sh" \
            python3 -m pww.smoke "$@"
