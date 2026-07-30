# ProjectWorldWide -- site-agnostic environment.
#
#   source ~/ProjectWorldWide/env.sh
#
# Detects which machine it is on and sources sites/<site>.sh for everything
# machine-specific. Nothing below this line should contain a LUMI- or
# Snellius-specific value; if you are tempted to add one, it belongs in the site
# file instead.
#
# Override detection with PWW_SITE=lumi|snellius.

# --- Site detection ---------------------------------------------------------
pww_detect_site() {
    if [[ -n "${PWW_SITE:-}" ]]; then
        echo "${PWW_SITE}"
    elif [[ -d /appl/local/containers/sif-images ]]; then
        echo lumi
    elif [[ -d /sw/arch ]] || [[ "$(hostname -f 2>/dev/null)" == *snellius* ]]; then
        echo snellius
    else
        echo unknown
    fi
}

export PWW_ROOT="${PWW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export PWW_SITE="$(pww_detect_site)"

if [[ "${PWW_SITE}" == "unknown" ]]; then
    echo "ProjectWorldWide: could not detect the site." >&2
    echo "  Set PWW_SITE=lumi or PWW_SITE=snellius, or add sites/<name>.sh" >&2
    return 1 2>/dev/null || exit 1
fi

# --- Site-independent defaults ----------------------------------------------
# Site files may override any of these; they are set first so a site file can
# build on them (e.g. deriving PWW_DATA_DIR from its own PWW_SCRATCH).
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# --- Site specifics ---------------------------------------------------------
# Each sites/<site>.sh must define:
#   PWW_ACCOUNT           accounting project for sbatch
#   PWW_SCRATCH           writable, large, node-visible base directory
#   PWW_GPUS_PER_NODE     ranks per full node
#   PWW_CPUS_PER_TASK     cores per rank
#   PWW_ACCELERATOR       "rocm" | "cuda"
#   PWW_GPU_VISIBLE_VAR   env var used to pin one device per rank
#   pww_cpu_bind()        echoes the --cpu-bind value for this allocation
#   PWW_LAUNCH            bash array: command prefix to enter the environment
#                         (container exec, or empty when using modules)
# shellcheck source=/dev/null
source "${PWW_ROOT}/sites/${PWW_SITE}.sh"

# --- Derived paths ----------------------------------------------------------
export PWW_DATA_DIR="${PWW_DATA_DIR:-${PWW_SCRATCH}/data}"
export PWW_OUTPUT_DIR="${PWW_OUTPUT_DIR:-${PWW_SCRATCH}/runs}"
export PWW_TMPDIR="${PWW_TMPDIR:-${PWW_SCRATCH}/tmp}"
export PWW_CACHE_DIR="${PWW_CACHE_DIR:-${PWW_SCRATCH}/cache}"

export PYTHONPATH="${PWW_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Some containers set PYTHONPATH in their own /.singularity.d/env scripts, which
# then wins over the host value and makes `pww` unimportable inside the job.
# SINGULARITYENV_/APPTAINERENV_ are applied with higher precedence, so set both.
# Safe even when the container ships a venv: a venv resolves its own
# site-packages from sys.prefix, not from PYTHONPATH.
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

# Keep framework caches off small, inode-limited home directories.
export HF_HOME="${PWW_CACHE_DIR}/huggingface"
export TORCH_HOME="${PWW_CACHE_DIR}/torch"
export TRITON_CACHE_DIR="${PWW_CACHE_DIR}/triton"

# --- Helpers ----------------------------------------------------------------
# Run a command in the project environment on whichever site we are on.
#   pww_run python3 -c 'import torch; print(torch.__version__)'
pww_run() {
    if [[ ${#PWW_LAUNCH[@]} -eq 0 ]]; then
        "$@"
    else
        "${PWW_LAUNCH[@]}" "$@"
    fi
}

pww_summary() {
    cat <<EOF
site        : ${PWW_SITE}
accelerator : ${PWW_ACCELERATOR}
account     : ${PWW_ACCOUNT}
ranks/node  : ${PWW_GPUS_PER_NODE}  (cores/rank: ${PWW_CPUS_PER_TASK})
repo        : ${PWW_ROOT}
scratch     : ${PWW_SCRATCH}
launch      : ${PWW_LAUNCH[*]:-<native / modules>}
EOF
}
