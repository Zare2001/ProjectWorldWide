# Snellius (SURF, Netherlands) -- NVIDIA GPUs, CUDA.
#
# Verified on Snellius (RHEL9, Slurm 'snellius' cluster) -- every value below was
# read off the machine with ./scripts/siteinfo.sh, not from documentation.
#
# The one structural difference from LUMI: LUMI has a maintained container with
# the whole AI stack in it, and Snellius has nothing equivalent. Its only
# PyTorch module is far too old for this codebase (see scripts/snellius/setup_venv.sh
# for the specifics), so the environment here is a pip venv that this repo
# builds and owns. Run setup_venv.sh once before anything else.

# Snellius accounts by group; no -A is required for a personal association.
# Set it if your project needs to be charged explicitly.
export PWW_ACCOUNT="${PWW_ACCOUNT:-}"

# --- Partition and node shape -----------------------------------------------
#   partition  nodes  GPUs/node        cores  RAM GiB  min alloc        SBU/GPU-h
#   gpu_h100     88   4x H100             64      720  16c + 1 GPU + 180      192
#   gpu_a100     63   4x A100             72      480  18c + 1 GPU + 120      128
#   gpu_vis      63   4x A100 (24h max)   72      480  18c + 1 GPU + 120      128
#   gpu_mig       4   8x A100 MIG slice   72      480   9c + 1 MIG +  60       64
#
# There is no partition called "gpu" -- that name in the original template was a
# guess and jobs using it are rejected at submit time.
#
# Unlike LUMI there is no GCD split: 4 GPUs = 4 ranks, not 8.
#
# Two scheduling notes that matter more here than on LUMI. Both GPU partitions
# routinely show zero idle nodes, but jobs asking for <= 1 h of walltime are
# routed to a reserved pool of short-job nodes -- so keep debug runs at or under
# --time=01:00:00 and they queue far faster. And H100 is billed 1.5x A100 per
# GPU-hour, so a run that is not H100-bound is cheaper on gpu_a100.
export PWW_PARTITION="${PWW_PARTITION:-gpu_h100}"

case "${PWW_PARTITION}" in
    gpu_h100)          _pww_gpus=4 ; _pww_cores=16 ;;
    gpu_a100|gpu_vis)  _pww_gpus=4 ; _pww_cores=18 ;;
    gpu_mig)           _pww_gpus=8 ; _pww_cores=9  ;;
    *)                 _pww_gpus=4 ; _pww_cores=16 ;;
esac
export PWW_GPUS_PER_NODE="${PWW_GPUS_PER_NODE:-${_pww_gpus}}"
export PWW_CPUS_PER_TASK="${PWW_CPUS_PER_TASK:-${_pww_cores}}"
unset _pww_gpus _pww_cores

export PWW_ACCELERATOR=cuda
export PWW_GPU_VISIBLE_VAR=CUDA_VISIBLE_DEVICES

# --- Storage ----------------------------------------------------------------
# /scratch-shared is large (8 TiB quota) but PURGED ON FILE AGE, around 14 days.
# That is fine for datasets you can re-download and for run output you will look
# at this week; it is not fine for anything you want to keep. Point PWW_SCRATCH
# at one of your /projects spaces for work that has to survive:
#     export PWW_SCRATCH=/projects/<your-project>/$USER/projectworldwide
export PWW_SCRATCH="${PWW_SCRATCH:-/scratch-shared/${USER}/projectworldwide}"

# --- Environment: module-provided Python + a pip venv ------------------------
# Not the PyTorch module. The 2023 tree's PyTorch/2.1.2 predates every
# distributed API this codebase uses, the 2024 tree's PyTorch directory is
# empty, and 2025 ships no ai modules at all. setup_venv.sh installs torch
# 2.7.1 to match the LUMI container version for version.
export PWW_MODULE_YEAR="${PWW_MODULE_YEAR:-2024}"
export PWW_PYTHON_MODULE="${PWW_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
export PWW_VENV="${PWW_VENV:-${HOME}/venvs/pww-snellius}"

# Lmod is normally already initialised, but not in every non-interactive shell.
if ! command -v module >/dev/null 2>&1; then
    for _init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash /sw/lmod/lmod/init/bash; do
        # shellcheck source=/dev/null
        [[ -r "${_init}" ]] && source "${_init}" && break
    done
    unset _init
fi

# Guarded with || true throughout: job scripts run under `set -e`, and Lmod
# returns non-zero for things as harmless as an already-loaded module.
if command -v module >/dev/null 2>&1; then
    module load "${PWW_MODULE_YEAR}" >/dev/null 2>&1 || true
    module load "${PWW_PYTHON_MODULE}" >/dev/null 2>&1 \
        || echo "ProjectWorldWide: could not load ${PWW_PYTHON_MODULE}" >&2
fi

if [[ -r "${PWW_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${PWW_VENV}/bin/activate"
else
    echo "ProjectWorldWide: no venv at ${PWW_VENV}" >&2
    echo "  build it once from a login node: ./scripts/snellius/setup_venv.sh" >&2
fi

# The venv puts python on PATH directly -- no container to enter.
PWW_LAUNCH=()

# --- CPU binding ------------------------------------------------------------
# Both GPU node types are 4 sockets with one GPU attached to each, so the
# NUMA-correct placement is simply "rank N on socket N". Asking SLURM for
# --cpus-per-task=<cores per socket> and binding to cores gets that, provided
# the job also uses block distribution (the job scripts pass
# --distribution=block:block). This is why no hand-written mask is needed here,
# unlike LUMI where the GCD-to-core map is not something SLURM can infer.
pww_cpu_bind() {
    echo "cores"
}
