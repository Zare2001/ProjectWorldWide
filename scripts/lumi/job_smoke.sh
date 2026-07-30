#!/bin/bash
#SBATCH --job-name=pww-smoke
#SBATCH --account=project_462000226
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Validate the distributed stack on 8 GCDs. Run this after any environment
# change, before spending GPU hours on real training.
#
#   sbatch scripts/job_smoke.sh
#
# --mem=0 requests all memory on the node; the default per-node share is far too
# small for 8 dataloader-heavy ranks.

set -euo pipefail

# Submitted from the repo root, so SLURM_SUBMIT_DIR is it; PWW_ROOT overrides.
: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    echo "  submit from the repo root, or export PWW_ROOT=/path/to/ProjectWorldWide" >&2
    exit 1
fi
source "${PWW_ROOT}/env.sh"

# All ranks must agree on the rendezvous point, so it is computed once here
# rather than per-task.
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=29500

srun --cpu-bind="$(pww_cpu_bind)" \
    "${PWW_LAUNCH[@]}" \
        "${PWW_ROOT}/scripts/task_wrapper.sh" \
            python3 -m pww.smoke
