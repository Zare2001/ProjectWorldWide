#!/bin/bash
#SBATCH --job-name=pww-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Validate the distributed stack on one Snellius GPU node.
#
#   sbatch scripts/snellius/job_smoke.sh
#
# !! UNVERIFIED -- run ./scripts/siteinfo.sh first and correct:
#   --partition   gpu | gpu_a100 | gpu_h100
#   4 / 18        GPUs per node and cores per GPU for that partition
#   --account     add "#SBATCH --account=..." if your project requires one
#
# Differences from the LUMI equivalent, all of which are handled by
# sites/snellius.sh rather than by the training code:
#   * 4 ranks per node, not 8 (no GCD split on NVIDIA)
#   * no --mem=0; Snellius allocates memory proportionally to requested GPUs
#   * environment via modules rather than a container

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

# InfiniBand rather than LUMI's Slingshot. Leave unset to let NCCL autodetect;
# set explicitly (e.g. ib0) if the smoke test reports single-digit GB/s, which
# means it fell back to TCP over the management network.
# export NCCL_SOCKET_IFNAME=ib0

srun --cpu-bind="$(pww_cpu_bind)" \
    "${PWW_LAUNCH[@]}" \
        "${PWW_ROOT}/scripts/task_wrapper.sh" \
            python3 -m pww.smoke
