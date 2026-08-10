#!/bin/bash
#SBATCH --job-name=pww-snellius-titan
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# torchtitan (Qwen3) + DARL + Flower/FedMom on Snellius (NVIDIA H100).
#
#   CONFIG=configs/titan/qwen3_0.6b_smoke.toml sbatch scripts/snellius/job_titan_diloco.sh
#   sbatch scripts/snellius/job_titan_diloco.sh                # the C4 DiLoCo run
#
# Unlike scripts/snellius/job_flower_diloco_llm.sh this asks for ONE task per node
# and lets torchrun fork the four ranks. torchtitan expects to own the process
# topology -- LOCAL_RANK, the rendezvous, the device mesh -- and srun spawning
# four independent tasks that each then try to be rank 0 of their own torchrun is
# the classic way to get four one-GPU jobs that never form a mesh.

set -euo pipefail

: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    exit 1
fi
source "${PWW_ROOT}/env.sh"

export PYTHONNOUSERSITE=0

# Snellius' XALT wrapper injects an OpenSSL 3 shared library that conflicts with
# the one torch links against, and LD_PRELOAD from the host does not resolve in
# every environment. Same treatment as the existing job scripts.
module unload XALT 2>/dev/null || true
unset LD_PRELOAD
if [[ -d "/sw/arch/RHEL9/EB_production/2024/software/OpenSSL/3/lib64" ]]; then
    export LD_LIBRARY_PATH="/sw/arch/RHEL9/EB_production/2024/software/OpenSSL/3/lib64:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v "/opt/xalt" | paste -sd:)
fi

# The torchtitan venv, NOT the torch 2.7.1 one env.sh activated. torchtitan needs
# torch >= 2.9; see scripts/titan/README.md for why both exist.
PWW_TITAN_VENV="${PWW_TITAN_VENV:-${HOME}/venvs/pww-titan-snellius}"
if [[ -r "${PWW_TITAN_VENV}/bin/activate" ]]; then
    source "${PWW_TITAN_VENV}/bin/activate"
    # run_train.sh sources env.sh again in its own shell, and sites/snellius.sh
    # activates PWW_VENV unconditionally -- which would put the 2.7.1 venv back on
    # PATH and launch the ranks with a torch that has no FSDP2 and no tyro. Point
    # PWW_VENV at the titan venv so that re-source is a no-op instead of a silent
    # downgrade.
    export PWW_VENV="${PWW_TITAN_VENV}"
    echo "activated torchtitan venv: ${PWW_TITAN_VENV}"
else
    echo "ERROR: no torchtitan venv at ${PWW_TITAN_VENV}" >&2
    echo "build it once from a login node: ./scripts/titan/setup_venv_snellius.sh" >&2
    exit 1
fi

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

# Reachability and auth, before torchrun claims the GPUs. A wrong token or an unreachable
# coordinator otherwise surfaces well into the run, after the allocation is already spent.
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
        # Not fatal: a proxy or a missing curl should not block a run that would work.
        # The client retries and reports its own error if the coordinator really is down.
        000) echo "WARNING: could not reach http://${CENTRAL_IP}:${PWW_DARL_PORT:-29510}/health" \
                  "from this node -- continuing, but expect the client to fail if the" \
                  "central node is not up." >&2 ;;
        *)   echo "WARNING: coordinator answered HTTP ${darl_probe} on /health." >&2 ;;
    esac
fi

# Releasing the DARL leases on SIGTERM returns the tail to the pool in
# milliseconds instead of after a full TTL, which is the difference between the
# other site idling for a quarter of an hour at every walltime kill and not idling
# at all. Slurm sends SIGTERM before SIGKILL, so forward it to torchrun's group.
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
    --site snellius \
    --nproc "${SLURM_GPUS_PER_NODE:-4}" \
    -- "$@" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
