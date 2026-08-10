#!/bin/bash
#SBATCH --job-name=pww-lumi-titan
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# torchtitan (Qwen3) + DARL + Flower/FedMom on LUMI (AMD MI250X, ROCm/RCCL).
#
#   CONFIG=configs/titan/qwen3_0.6b_smoke.toml sbatch scripts/lumi/job_titan_diloco.sh
#   sbatch -A $PWW_ACCOUNT scripts/lumi/job_titan_diloco.sh
#
# One task per node, torchrun forking 8 ranks -- one per GCD. LUMI's MI250X is two
# GCDs per physical card and torch addresses each as its own device, so 4 cards is
# 8 ranks, not 4.
#
# The container here is NOT LUMI's maintained one: that ships torch 2.7.1 and
# torchtitan needs >= 2.9. See scripts/titan/README.md and
# containers/titan-lumi.def.

set -euo pipefail

: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    exit 1
fi
source "${PWW_ROOT}/env.sh"

TITAN_SIF="${PWW_TITAN_SIF:-${PWW_SCRATCH}/containers/pww-titan.sif}"
if [[ ! -r "${TITAN_SIF}" ]]; then
    echo "ERROR: no torchtitan container at ${TITAN_SIF}" >&2
    echo "build it once: sbatch scripts/lumi/build_titan_container.sh" >&2
    exit 1
fi

# Replaces the 2.7.1 container env.sh selects with the torch 2.9 one, for the CHILD
# process that does the training.
#
# It has to travel as PWW_CONTAINER, not as PWW_LAUNCH. `export PWW_LAUNCH` looks
# like it works and cannot: bash does not export arrays, so run_train.sh -- a separate
# process that sources env.sh itself (scripts/titan/run_train.sh:67) -- rebuilt
# PWW_LAUNCH from PWW_CONTAINER and silently got the 2.7.1 image. The symptom was
# eight ranks dying on `ModuleNotFoundError: No module named 'tyro'` while the header
# printed the 2.7.1 path, which is the only place it was visible.
#
# env.sh reads PWW_CONTAINER while building PWW_LAUNCH, so exporting it here is
# consumed correctly by the child.
export PWW_CONTAINER="${TITAN_SIF}"

# NO --rocm, deliberately. It binds the host's ROCm over the container's, and this
# image ships its own complete ROCm 6.4 while LUMI's host stack is 6.2.4. The older
# librccl then shadows the newer one that libtorch_hip.so was linked against:
#
#   ImportError: .../libtorch_hip.so: undefined symbol: ncclGroupSimulateEnd
#
# which is a link-time failure at `import torch`, before any GPU is touched -- so it
# reproduces on a login node and does not need an allocation to test. Device access
# does not need the flag: /dev/kfd and /dev/dri are visible without it, which is why
# no other job script here uses it either.
#
# The scratch bind is likewise unnecessary: sites/lumi.sh lists /scratch in
# SINGULARITY_BIND alongside the rest of the AI bindings.
PWW_LAUNCH=(singularity exec "${TITAN_SIF}")

# MIOpen writes a kernel cache and defaults to $HOME, which on LUMI is a small
# quota and shared between concurrent jobs -- two jobs racing on the same cache
# database is a known hang. One cache per job, on scratch.
export MIOPEN_USER_DB_PATH="${PWW_TMPDIR}/miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"
export SINGULARITYENV_MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH}"
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"

# Slingshot 11: RCCL needs to be told which interfaces to use, and host
# registration has to be off or large transfers fail on the CXI provider. Values
# carried over from the existing LUMI job scripts.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_DISABLE_HOST_REGISTER=1
for var in NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL FI_CXI_DISABLE_HOST_REGISTER; do
    export "SINGULARITYENV_${var}=${!var}"
done

CENTRAL_IP="${CENTRAL_IP:-145.38.206.143}"
CONFIG="${CONFIG:-${PWW_ROOT}/configs/titan/qwen3_0.6b_c4_diloco.toml}"
TOKENIZER="${TOKENIZER:-${PWW_DATA_DIR}/tokenizers/tokenizer-128k}"
SHARDS="${SHARDS:-${PWW_DATA_DIR}/c4-tokenizer-128k-2048}"

# The token, and it has to be here rather than discovered later.
#
# These candidate paths only ever exist on the central VM, which is where the file is
# written. At a site, $PWW_ROOT/runs points into that site's own scratch, so the loop
# finds nothing and leaves DARL_TOKEN empty. That used to be harmless because the
# coordinator ran without a token and authorised everything; now that it enforces one, an
# empty value means every request is refused. Failing here costs a second. Failing at
# registration costs the queue wait plus the allocation.
if [[ -z "${DARL_TOKEN:-}" ]]; then
    for candidate in "${PWW_ROOT}/runs/darl/token" "${PWW_ROOT}/runs/central/darl/token"; do
        [[ -s "${candidate}" ]] && export DARL_TOKEN="$(cat "${candidate}")" && break
    done
fi
if [[ -z "${DARL_TOKEN:-}" ]]; then
    cat >&2 <<EOF
ERROR: DARL_TOKEN is empty and the coordinator enforces it, so every request from this
job would be refused with 401.

Pass it explicitly at submit time -- the fallback paths this script checks exist only on
the central VM, not here:

    DARL_TOKEN="\$(<the token from the central node>)" sbatch $0

On the central VM the value is in runs/darl/token.
EOF
    exit 1
fi
export SINGULARITYENV_DARL_TOKEN="${DARL_TOKEN}"

# Reachability and auth, before srun claims eight GCDs. A wrong token or an unreachable
# coordinator otherwise surfaces well into the run, after the allocation is already spent.
# Run outside the container on purpose: this is about the node's route to the WAN.
if command -v curl >/dev/null 2>&1; then
    darl_probe="$(curl -sS -m 15 -o /dev/null -w '%{http_code}' \
        -H "X-DARL-Token: ${DARL_TOKEN}" \
        "http://${CENTRAL_IP}:${PWW_DARL_PORT:-29510}/health" 2>/dev/null)" || true
    # curl writes 000 through -w *and* exits non-zero when it cannot connect, so a
    # fallback echo inside the substitution would append a second 000 and miss the case
    # below. Normalise instead of guessing.
    [[ "${darl_probe}" =~ ^[0-9]{3}$ ]] || darl_probe=000
    case "${darl_probe}" in
        200) echo "darl: coordinator reachable at ${CENTRAL_IP}, token accepted" ;;
        401) echo "ERROR: the coordinator rejected this DARL_TOKEN. Re-copy it from" \
                  "runs/darl/token on the central node." >&2; exit 1 ;;
        # Not fatal: LUMI compute nodes reach the outside world through a proxy, so a
        # failed probe here does not prove the client cannot connect.
        000) echo "WARNING: could not reach http://${CENTRAL_IP}:${PWW_DARL_PORT:-29510}/health" \
                  "from this node -- continuing, but expect the client to fail if the" \
                  "central node is not up." >&2 ;;
        *)   echo "WARNING: coordinator answered HTTP ${darl_probe} on /health." >&2 ;;
    esac
fi

# LUMI kills jobs at walltime, which for a long DiLoCo run is the normal way a job
# ends rather than an exception. Forwarding SIGTERM lets the DARL session release
# its uncommitted spans immediately instead of the other site waiting out a full
# lease TTL before it can pick them up.
# The shell only forwards the signal; the release itself happens in the client, which
# installs a SIGTERM handler on the leader rank (darl_dataloader._release_on_sigterm) and
# logs how many blocks it actually returned. This message used to claim the release
# outright, before any handler existed -- so read the client's line, not this one.
trap 'echo "SIGTERM -- forwarding to torchrun so the leader can release its DARL leases"; kill -TERM ${TRAIN_PID:-0} 2>/dev/null' TERM

"${PWW_ROOT}/scripts/titan/run_train.sh" \
    --config "${CONFIG}" \
    --tokenizer "${TOKENIZER}" \
    --shards "${SHARDS}" \
    --central "${CENTRAL_IP}" \
    --site lumi \
    --nproc "${SLURM_GPUS_PER_NODE:-8}" \
    -- "$@" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
