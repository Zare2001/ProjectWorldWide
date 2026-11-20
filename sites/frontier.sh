# Frontier (OLCF, USA) -- AMD MI250X, ROCm.
#
# NOT verified on the machine from this repository. The values are taken from the
# header of the August 2026 Frontier job logs (8 ranks per node, 7 cores per rank,
# native launch, HTTP proxy) and from LUMI, which has the same node type. Check
# the summary that every job prints against the machine before a long run.

export PWW_ACCOUNT="${PWW_ACCOUNT:-csc752}"
export PWW_SCRATCH="${PWW_SCRATCH:-/lustre/orion/${PWW_ACCOUNT}/scratch/${USER}}"

# Same node as LUMI-G: 4 MI250X cards, each two GCDs, and ROCm treats a GCD as one
# GPU. 64 cores with one per L3 group left to the OS: 7 per rank.
export PWW_GPUS_PER_NODE=8
export PWW_CPUS_PER_TASK=7

export PWW_ACCELERATOR=rocm
export PWW_GPU_VISIBLE_VAR=ROCR_VISIBLE_DEVICES

# --- Network ------------------------------------------------------------------
# Compute nodes have no direct route to the internet. Outgoing HTTP goes through
# the OLCF proxy, which is why Frontier runs use the blob transport and the HTTP
# round protocol (--darl.use_proxy --flower.transport blob --flower.protocol http).
export PWW_HTTP_PROXY="${PWW_HTTP_PROXY:-http://proxy.ccs.ornl.gov:3128}"
export PWW_NO_PROXY="${PWW_NO_PROXY:-localhost,127.0.0.1,::1,.olcf.ornl.gov,.ornl.gov,.ccs.ornl.gov,frontier,10.0.0.0/8}"

# --- Environment: module-provided Python + a pip venv -------------------------
# The torch ROCm wheel carries its own ROCm libraries, so only a Python is needed
# from the module tree. Build the venv once with scripts/titan/setup_venv_frontier.sh.
export PWW_PYTHON_MODULE="${PWW_PYTHON_MODULE:-cray-python}"
export PWW_TITAN_VENV="${PWW_TITAN_VENV:-${PWW_SCRATCH}/venvs/pww-titan-frontier}"
export PWW_VENV="${PWW_VENV:-${PWW_TITAN_VENV}}"

if command -v module >/dev/null 2>&1; then
    module load "${PWW_PYTHON_MODULE}" >/dev/null 2>&1 \
        || echo "ProjectWorldWide: could not load ${PWW_PYTHON_MODULE}" >&2
fi

if [[ -r "${PWW_VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${PWW_VENV}/bin/activate"
else
    echo "ProjectWorldWide: no venv at ${PWW_VENV}" >&2
    echo "  build it once from a login node: ./scripts/titan/setup_venv_frontier.sh" >&2
fi

# The venv puts python on PATH directly -- no container to enter.
PWW_LAUNCH=()

# The titan job runs one task that forks the ranks itself, so no per-rank mask.
pww_cpu_bind() {
    echo "cores"
}

pww_titan_env() {
    printf 'venv\t%s\t./scripts/titan/setup_venv_frontier.sh\n' "${PWW_TITAN_VENV}"
}
