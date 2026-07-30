# Snellius (SURF, Netherlands) -- NVIDIA GPUs, CUDA.
#
# !! UNVERIFIED !!
# This file was written from SURF documentation, not from a Snellius login. The
# values marked [VERIFY] below must be confirmed on the machine before trusting
# a run. Do that in one step:
#
#     ./scripts/siteinfo.sh          # prints what every [VERIFY] value should be
#
# Everything else in the codebase is site-independent and already verified on
# LUMI, so this file plus scripts/snellius/ should be the only things to correct.

# [VERIFY] Snellius usually accounts by group rather than requiring --account.
# Leave empty to omit -A from sbatch; set it if your project needs one.
export PWW_ACCOUNT="${PWW_ACCOUNT:-}"

# [VERIFY] Snellius has several options and they differ in lifetime:
#   /scratch-shared/$USER   large, cluster-wide, PURGED after ~14 days
#   /projects/0/<project>   persistent project space
#   $TMPDIR (/scratch-local) node-local, fastest, erased at job end
# Default to scratch-shared, but move to /projects for anything you want to keep.
export PWW_SCRATCH="${PWW_SCRATCH:-/scratch-shared/${USER}/projectworldwide}"

# [VERIFY] A100 nodes: 4x A100, 72 cores -> 18 cores/rank.
#          H100 nodes: 4x H100, 64 cores -> 16 cores/rank.
# Unlike LUMI there is no GCD split, so 4 GPUs = 4 ranks.
export PWW_GPUS_PER_NODE="${PWW_GPUS_PER_NODE:-4}"
export PWW_CPUS_PER_TASK="${PWW_CPUS_PER_TASK:-18}"

export PWW_ACCELERATOR=cuda
export PWW_GPU_VISIBLE_VAR=CUDA_VISIBLE_DEVICES

# --- Environment: EasyBuild modules -----------------------------------------
# Snellius has no equivalent of LUMI's prebuilt AI containers, so use the module
# tree. [VERIFY] the exact module names with `module load 2024; module avail PyTorch`.
#
# Note this is the one real behavioural difference between the sites: the LUMI
# container ships transformers/tokenizers/datasets/flash-attn, whereas here they
# are separate modules or a venv. If a module is missing, the documented SURF
# approach is a venv layered on the PyTorch module:
#     python -m venv --system-site-packages $PWW_SCRATCH/venv
export PWW_MODULE_YEAR="${PWW_MODULE_YEAR:-2024}"
export PWW_TORCH_MODULE="${PWW_TORCH_MODULE:-PyTorch/2.1.2-foss-2023a-CUDA-12.1.1}"

# Lmod is normally already initialised, but not in every non-interactive shell.
if ! command -v module >/dev/null 2>&1; then
    for _init in /usr/share/lmod/lmod/init/bash /sw/lmod/lmod/init/bash; do
        # shellcheck source=/dev/null
        [[ -r "${_init}" ]] && source "${_init}" && break
    done
    unset _init
fi

if command -v module >/dev/null 2>&1; then
    module load "${PWW_MODULE_YEAR}" 2>/dev/null || true
    module load "${PWW_TORCH_MODULE}" 2>/dev/null \
        || echo "ProjectWorldWide: could not load ${PWW_TORCH_MODULE} -- run ./scripts/siteinfo.sh" >&2
fi

# Optional venv layered on the module, for packages the module lacks.
if [[ -n "${PWW_VENV:-}" && -r "${PWW_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${PWW_VENV}/bin/activate"
fi

# Modules put python on PATH directly -- no container to enter.
PWW_LAUNCH=()

# --- CPU binding ------------------------------------------------------------
# Unlike LUMI, there is no published GPU-to-core mask to reproduce here, and
# --cpus-per-task already gives SLURM enough information to place ranks
# sensibly. If profiling later shows NUMA imbalance, this is the place to add an
# explicit mask_cpu list.
pww_cpu_bind() {
    echo "cores"
}
