#!/bin/bash
# Start central node services: DARL Coordinator (port 29510) and Flower Aggregator (port 29511)
# Run on the central Ubuntu cloud VM.

set -euo pipefail

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    exit 1
fi

export PWW_SITE="${PWW_SITE:-central}"
source "${PWW_ROOT}/env.sh"

DARL_PORT="${DARL_PORT:-29510}"
FLOWER_PORT="${FLOWER_PORT:-29511}"
BLOB_PORT="${BLOB_PORT:-29512}"
STATE_DIR="${PWW_OUTPUT_DIR:-${PWW_ROOT}/runs}/central"

# What this script was last launched with, so a restart does not quietly become a
# different run. Two files, each next to the state it has to stay consistent with:
# the block geometry lives with the lease table it partitions (see SPACE_FILE below),
# the run flavour with the global model it produced.
#
# Read the comment on SPACE_FILE for why this is not optional. The aggregator config is
# the sharper edge of the two: omitting AGGREGATOR_CONFIG falls back to the ResNet
# defaults, which turn FedMom off (momentum 0.0 is algebraically FedAvg), set
# min-clients to 2 so the run blocks until both sites are out of the queue, and cut
# num-rounds to 50 with a 300s timeout. Only the momentum change warns; the rest are
# silent.
LAUNCH_FILE="${STATE_DIR}/launch.env"
_remember() {  # $1=file $2=key $3=ERE the value must match entirely
    [[ -r "$1" ]] || return 1
    local value
    value="$(grep -m1 -E "^$2=$3$" "$1" | cut -d= -f2-)" || return 1
    [[ -n "${value}" ]] || return 1
    printf '%s' "${value}"
}

# TRANSPORT=blob moves weights out of band over HTTP instead of inside the Flower
# gRPC message, which is required above roughly 1B parameters (gRPC caps a single
# message at 2 GiB and no setting raises it). It adds a third daemon and needs
# real disk: the global model and the momentum buffer are resident, plus one
# delta per site transiently. See scripts/titan/README.md for the arithmetic.
TRANSPORT="${TRANSPORT:-$(_remember "${LAUNCH_FILE}" TRANSPORT '[a-z]+' || echo inline)}"
# Both under PWW_OUTPUT_DIR by default, and deliberately on the same filesystem:
# publishing a merged global model into the blob store is then a hard link rather
# than a copy of up to hundreds of gigabytes.
GLOBAL_STATE_DIR="${GLOBAL_STATE_DIR:-${STATE_DIR}/global}"
BLOB_ROOT="${BLOB_ROOT:-${STATE_DIR}/blobs}"
# The URL the CLUSTERS use, so it must be this VM's routable address, not localhost.
BLOB_HOST="${BLOB_HOST:-145.38.206.143}"
BLOB_URL="${BLOB_URL:-http://${BLOB_HOST}:${BLOB_PORT}}"
RUN_ID="${RUN_ID:-pww}"

mkdir -p "${STATE_DIR}"

DARL_PID_FILE="${STATE_DIR}/darl.pid"
FLOWER_PID_FILE="${STATE_DIR}/flower.pid"
BLOB_PID_FILE="${STATE_DIR}/blob.pid"
VENV_DIR="${STATE_DIR}/.venv"
FLOWER_REPO="${FLOWER_REPO:-git+https://github.com/Zare2001/flower.git@fedmom-strategy#subdirectory=framework}"

echo "========================================================="
echo " Starting Central Node Aggregator Services"
echo " Central Node Host: $(hostname -f 2>/dev/null || hostname)"
echo " Central Node Public IP: 145.38.206.143"
echo " DARL Port:   ${DARL_PORT}"
echo " Flower Port: ${FLOWER_PORT}"
echo " Transport:   ${TRANSPORT}"
if [[ "${TRANSPORT}" == "blob" ]]; then
echo " Blob Port:   ${BLOB_PORT}  (${BLOB_URL})"
echo " Blob Root:   ${BLOB_ROOT}"
echo " Global Dir:  ${GLOBAL_STATE_DIR}"
fi
echo " State Dir:   ${STATE_DIR}"
echo "========================================================="

# 0. Setup Environment: uv -> venv -> lightweight venv fallback
PYTHON_BIN="python3"

if ! "${VENV_DIR}/bin/python3" -c "import flwr, torch" 2>/dev/null; then
    echo "Installing Flower & PyTorch into ${VENV_DIR}..."
    if command -v uv >/dev/null 2>&1; then
        echo "Using uv..."
        uv pip install --system "${FLOWER_REPO}" torch --extra-index-url https://download.pytorch.org/whl/cpu 2>/dev/null || true
    elif python3 -m venv "${VENV_DIR}" >/dev/null 2>&1 && [[ -x "${VENV_DIR}/bin/pip" ]]; then
        echo "Using python3 venv..."
        "${VENV_DIR}/bin/pip" install "${FLOWER_REPO}" torch --extra-index-url https://download.pytorch.org/whl/cpu
        PYTHON_BIN="${VENV_DIR}/bin/python3"
    else
        echo "Creating lightweight venv and installing Flower & PyTorch into ${VENV_DIR}..."
        python3 -m venv --without-pip "${VENV_DIR}" >/dev/null 2>&1 || true
        PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        SITE_PKGS="${VENV_DIR}/lib/python${PY_VER}/site-packages"
        mkdir -p "${SITE_PKGS}"
        python3 -m pip install --target="${SITE_PKGS}" --break-system-packages "${FLOWER_REPO}" torch --extra-index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
        python3 -m pip install --target="${SITE_PKGS}" "${FLOWER_REPO}" torch --extra-index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
        pip install --target="${SITE_PKGS}" "${FLOWER_REPO}" torch --extra-index-url https://download.pytorch.org/whl/cpu
        PYTHON_BIN="${VENV_DIR}/bin/python3"
    fi
elif [[ -x "${VENV_DIR}/bin/python3" ]]; then
    PYTHON_BIN="${VENV_DIR}/bin/python3"
fi

# NUM_SAMPLES read from a manifest instead of retyped.
#
#   MANIFEST=/path/to/c4-tokenizer-128k-2048/manifest.json DARL_FRESH=1 \
#       ./scripts/central_node/start_central_services.sh
#
# The window count is the one number that has to agree between this node and every site,
# and the block-space digest catches a disagreement only at registration -- correct, but
# by then a site has already spent its queue wait. The central node never reads the
# corpus, so copying just this one small file across is enough to take the transcription
# step out of the loop entirely.
if [[ -n "${MANIFEST:-}" && -z "${NUM_SAMPLES:-}" ]]; then
    if ! NUM_SAMPLES="$(python3 -c 'import json,sys
raw = json.load(open(sys.argv[1]))
print(int(raw["num_windows"]))' "${MANIFEST}" 2>&1)"; then
        echo "ERROR: could not read num_windows from ${MANIFEST}: ${NUM_SAMPLES}" >&2
        exit 1
    fi
    echo "num_samples ${NUM_SAMPLES} read from ${MANIFEST}"
fi

# The block-space parameters, remembered across restarts.
#
# These three define the partitioning, and the coordinator refuses to resume a snapshot
# that describes a different one. That check is correct, but it made a bare
# `./start_central_services.sh` unusable after any real run: the constants below are not
# any run's real geometry, so the defaults silently disagreed with the snapshot and the
# restart died inside Coordinator.load.
#
# They cannot be recovered from the snapshot -- it stores num_blocks and the digest, and a
# digest is a hash -- so this script records what it launched with and reuses it. An
# explicit environment variable always wins, so switching corpus is still just
# NUM_SAMPLES=... DARL_FRESH=1, and the file is inside the DARL state dir on purpose:
# delete that state and the memory of it goes too.
SPACE_FILE="${PWW_OUTPUT_DIR:-${PWW_ROOT}/runs}/darl/space.env"
_remembered() { _remember "${SPACE_FILE}" "$1" '[0-9]+'; }   # geometry is numeric-only
SPACE_SOURCE="explicit"
if [[ -z "${NUM_SAMPLES:-}${BLOCK_SIZE:-}${SEED:-}" ]] && [[ -r "${SPACE_FILE}" ]]; then
    SPACE_SOURCE="resumed from ${SPACE_FILE}"
fi
NUM_SAMPLES="${NUM_SAMPLES:-$(_remembered NUM_SAMPLES || echo 50000)}"
BLOCK_SIZE="${BLOCK_SIZE:-$(_remembered BLOCK_SIZE || echo 1000)}"
SEED="${SEED:-$(_remembered SEED || echo 42)}"
# Default DARL epochs set to 1 for single-pass LLM pre-training over tokenized corpora.
DARL_EPOCHS="${DARL_EPOCHS:-1}"

# Resume by default; --fresh has to be asked for by name.
#
# --fresh makes the coordinator ignore the snapshot in its state dir and start a new
# epoch with every block free again. Passing it unconditionally made restarting this
# script hand out spans that were already trained -- and silently, which is the worst
# part: the Flower server has no --fresh and DOES resume from its own --state-dir, so
# the merge round picks up where it left off and the restart looks clean while the
# exactly-once partitioning that is the entire point of the lease table is gone.
#
#   DARL_FRESH=1 ./scripts/central_node/start_central_services.sh   # new run, new corpus
#
# Note that changing NUM_SAMPLES/BLOCK_SIZE/SEED does not need this flag: the restore
# checks the block-space digest and refuses a snapshot that describes a different
# partitioning, which is a loud failure rather than a quiet one.
DARL_FRESH="${DARL_FRESH:-0}"
DARL_EXTRA=()
if [[ "${DARL_FRESH}" == "1" ]]; then
    DARL_EXTRA+=(--fresh)
fi

# 1. Start DARL Coordinator
if [[ -f "${DARL_PID_FILE}" ]] && kill -0 "$(cat "${DARL_PID_FILE}")" 2>/dev/null; then
    echo "DARL coordinator already running (PID $(cat "${DARL_PID_FILE}"))."
else
    echo "Starting DARL Lease Coordinator on port ${DARL_PORT} (samples: ${NUM_SAMPLES}, block_size: ${BLOCK_SIZE}, epochs: ${DARL_EPOCHS}, seed: ${SEED}, fresh: ${DARL_FRESH})"
    echo "  block space: ${SPACE_SOURCE}"
    # Not backgrounded. darl_coordinator.sh already nohups the server and returns after
    # a liveness check, so the `&` that used to be here threw that check away -- a
    # coordinator that died on a port collision still reported "DARL started." -- and
    # left step 2 racing the creation of the token file it reads.
    if ! DARL_PORT="${DARL_PORT}" "${PWW_ROOT}/scripts/darl_coordinator.sh" start \
        --num-samples "${NUM_SAMPLES}" \
        --block-size "${BLOCK_SIZE}" \
        --epochs "${DARL_EPOCHS}" \
        --seed "${SEED}" \
        ${DARL_EXTRA[@]+"${DARL_EXTRA[@]}"} > "${STATE_DIR}/darl.log" 2>&1; then
        echo "ERROR: DARL coordinator failed to start. Last lines of ${STATE_DIR}/darl.log:" >&2
        tail -n 20 "${STATE_DIR}/darl.log" >&2
        if grep -q "holds a .*-block epoch but this coordinator was started for" \
                "${PWW_OUTPUT_DIR:-${PWW_ROOT}/runs}/darl/coordinator.log" 2>/dev/null; then
            cat >&2 <<EOF

That is the resume check refusing a snapshot for a different partitioning. Either pass the
geometry the snapshot was built with, or start a new epoch on purpose:

    NUM_SAMPLES=<windows> BLOCK_SIZE=<n> SEED=<n> $0
    DARL_FRESH=1 NUM_SAMPLES=<windows> BLOCK_SIZE=<n> SEED=<n> $0
EOF
        fi
        exit 1
    fi
    # Only after a start that survived its liveness check, so a failed launch cannot
    # overwrite the geometry of the run still recorded in the snapshot.
    printf 'NUM_SAMPLES=%s\nBLOCK_SIZE=%s\nSEED=%s\n' \
        "${NUM_SAMPLES}" "${BLOCK_SIZE}" "${SEED}" > "${SPACE_FILE}"
    echo "DARL started."
fi

# The shared secret, for the blob store below and for the sites. Previously this
# script forced DARL_TOKEN="" into the coordinator, and because darl_coordinator.sh
# expands it with `-` rather than `:-`, that empty value survived the token-file
# fallback: the token was generated, written and never enforced. DarlHandler treats an
# empty token as "no auth configured" and authorises everything, so every endpoint on
# this port was open, not just /health.
#
# Setting DARL_TOKEN in the environment still wins, including to opt out explicitly:
#   DARL_TOKEN="" ./scripts/central_node/start_central_services.sh   # no auth, deliberate
DARL_TOKEN_FILE="${PWW_OUTPUT_DIR:-${PWW_ROOT}/runs}/darl/token"
if [[ -z "${DARL_TOKEN+set}" && -s "${DARL_TOKEN_FILE}" ]]; then
    DARL_TOKEN="$(cat "${DARL_TOKEN_FILE}")"
fi
export DARL_TOKEN
if [[ -n "${DARL_TOKEN:-}" ]]; then
    echo "DARL token enforced (${DARL_TOKEN_FILE}) -- the sites need this value."
else
    echo "WARNING: no DARL token: any process that can reach port ${DARL_PORT} can lease," \
         "commit and release blocks in this run." >&2
fi

# 2. Start the blob store (blob transport only)
SERVER_EXTRA=(--transport "${TRANSPORT}" --run-id "${RUN_ID}")
if [[ "${TRANSPORT}" == "blob" ]]; then
    mkdir -p "${BLOB_ROOT}" "${GLOBAL_STATE_DIR}"
    if [[ -f "${BLOB_PID_FILE}" ]] && kill -0 "$(cat "${BLOB_PID_FILE}")" 2>/dev/null; then
        echo "Blob store already running (PID $(cat "${BLOB_PID_FILE}"))."
    else
        echo "Starting blob store on port ${BLOB_PORT} (root ${BLOB_ROOT})..."
        nohup "${PYTHON_BIN}" -m pww.central.blobstore \
            --port "${BLOB_PORT}" \
            --root "${BLOB_ROOT}" \
            --token "${DARL_TOKEN:-}" > "${STATE_DIR}/blob.log" 2>&1 &
        echo $! > "${BLOB_PID_FILE}"
        echo "Blob store started (PID $(cat "${BLOB_PID_FILE}"))."
        sleep 1
    fi
    SERVER_EXTRA+=(--state-dir "${GLOBAL_STATE_DIR}" --blob-root "${BLOB_ROOT}" --blob-url "${BLOB_URL}")
else
    # Durable state is useful even inline: it is what lets this process restart, and
    # lets the server start before any cluster is out of the Slurm queue.
    mkdir -p "${GLOBAL_STATE_DIR}"
    SERVER_EXTRA+=(--state-dir "${GLOBAL_STATE_DIR}")
fi

# 3. Start Flower Aggregator Server
if [[ -f "${FLOWER_PID_FILE}" ]] && kill -0 "$(cat "${FLOWER_PID_FILE}")" 2>/dev/null; then
    echo "Flower server already running (PID $(cat "${FLOWER_PID_FILE}"))."
else
    # Overridable so a torchtitan/Qwen3 run can use its own strategy settings --
    # notably server-momentum, which configs/central_aggregator.yaml pins to 0.0
    # for ResNet's BatchNorm statistics and which an LLM run wants at 0.9.
    AGGREGATOR_CONFIG="${AGGREGATOR_CONFIG:-$(_remember "${LAUNCH_FILE}" AGGREGATOR_CONFIG \
        '[A-Za-z0-9._/-]+' || echo "${PWW_ROOT}/configs/central_aggregator.yaml")}"
    if [[ ! -r "${AGGREGATOR_CONFIG}" ]]; then
        echo "ERROR: aggregator config not readable: ${AGGREGATOR_CONFIG}" >&2
        exit 1
    fi
    echo "Starting Flower Aggregator Server on port ${FLOWER_PORT} (${AGGREGATOR_CONFIG})..."
    nohup "${PYTHON_BIN}" -m pww.central.server \
        --config "${AGGREGATOR_CONFIG}" \
        --port "${FLOWER_PORT}" \
        "${SERVER_EXTRA[@]}" > "${STATE_DIR}/flower.log" 2>&1 &
    echo $! > "${FLOWER_PID_FILE}"
    sleep 2
    if ! kill -0 "$(cat "${FLOWER_PID_FILE}")" 2>/dev/null; then
        echo "ERROR: Flower server exited immediately. Last lines of ${STATE_DIR}/flower.log:" >&2
        tail -n 20 "${STATE_DIR}/flower.log" >&2
        exit 1
    fi
    echo "Flower server started (PID $(cat "${FLOWER_PID_FILE}"))."
    # Same rule as the geometry: only record a launch that survived its liveness check.
    printf 'TRANSPORT=%s\nAGGREGATOR_CONFIG=%s\n' \
        "${TRANSPORT}" "${AGGREGATOR_CONFIG}" > "${LAUNCH_FILE}"
fi

echo ""
echo "Central node services successfully launched!"
echo "Check status anytime with: ./scripts/central_node/status_central_services.sh"
