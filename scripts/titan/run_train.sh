#!/usr/bin/env bash
# Launch a torchtitan + DARL (+ Flower) run. Site- and topology-agnostic.
#
# Runs INSIDE the site environment -- the container on LUMI, the venv on Snellius
# -- which env.sh arranges via PWW_LAUNCH. Called either directly on a login node
# for a CPU-side check, or from scripts/{lumi,snellius}/job_titan_diloco.sh inside
# an allocation.
#
#   scripts/titan/run_train.sh --config configs/titan/qwen3_0.6b_smoke.toml
#
# Running TWO jobs at the same site at the same time: pass --replica a and --replica b.
# Without it both register with the coordinator under the same cluster id and silently
# corrupt each other's leases and deltas. See the --replica block below.
#
#   scripts/titan/run_train.sh \
#       --config configs/titan/qwen3_0.6b_c4_diloco.toml \
#       --shards $PWW_DATA_DIR/c4-tokenizer-128k-2048 \
#       --tokenizer $PWW_DATA_DIR/tokenizers/tokenizer-128k \
#       --central 145.38.206.143
#
# Everything after `--` is passed to torchtitan verbatim, so any upstream flag
# works without this script knowing about it:
#   scripts/titan/run_train.sh --config ... -- --training.steps 50 --metrics.log_freq 1
#
# Why the paths are CLI overrides rather than values in the TOML: TOML does not
# expand environment variables, and the tokenizer/shard/dump paths are exactly the
# things that differ between LUMI's /scratch/project_* and Snellius'
# /scratch-shared. Baking either site's layout into a config is what sites/*.sh
# exists to prevent.
set -euo pipefail

CONFIG=""
SHARDS=""
TOKENIZER=""
CENTRAL="${PWW_CENTRAL_IP:-145.38.206.143}"
DARL_PORT="${PWW_DARL_PORT:-29510}"
FLOWER_PORT="${PWW_FLOWER_PORT:-29511}"
NPROC=""
NNODES="${SLURM_NNODES:-1}"
DUMP=""
SITE_OVERRIDE=""
# Distinguishes concurrent jobs at ONE site. Empty means "the only job here", which
# is the common case; see the block that consumes it below for why it matters.
REPLICA="${REPLICA:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --shards) SHARDS="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --central) CENTRAL="$2"; shift 2 ;;
        --darl-port) DARL_PORT="$2"; shift 2 ;;
        --flower-port) FLOWER_PORT="$2"; shift 2 ;;
        --nproc) NPROC="$2"; shift 2 ;;
        --dump) DUMP="$2"; shift 2 ;;
        --site) SITE_OVERRIDE="$2"; shift 2 ;;
        --replica) REPLICA="$2"; shift 2 ;;
        --) shift; break ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (use -- to pass flags to torchtitan)" >&2; exit 1 ;;
    esac
done
EXTRA=("$@")

PWW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PWW_ROOT}/env.sh"
cd "${PWW_ROOT}"

[[ -n "${CONFIG}" ]] || { echo "--config is required" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "no such config: ${CONFIG}" >&2; exit 1; }

if [[ ! -f third_party/torchtitan/torchtitan/train.py ]]; then
    echo "third_party/torchtitan is empty -- initialise the submodule:" >&2
    echo "  git submodule update --init --recursive" >&2
    exit 1
fi

# torchtitan is a submodule and never pip-installed, so it reaches the interpreter
# via PYTHONPATH. env.sh already put src/ there and mirrored it into the
# SINGULARITYENV_/APPTAINERENV_ forms that survive entering a container.
export PYTHONPATH="${PWW_ROOT}/src:${PWW_ROOT}/third_party/torchtitan${PYTHONPATH:+:${PYTHONPATH}}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

# One rank per GPU, from the site file rather than assumed -- LUMI exposes 8 GCDs
# per node and Snellius 4 H100s, and getting this wrong either idles hardware or
# oversubscribes it.
NPROC="${NPROC:-${SLURM_GPUS_PER_NODE:-${PWW_GPUS_PER_NODE}}}"
SITE="${SITE_OVERRIDE:-${PWW_SITE}}"

# The torchrun rendezvous port. Two jobs of this repo landing on the same node is
# routine -- both sites allow partial-node allocations, so submitting two 2-GPU jobs
# to one facility can put them on the same host -- and a shared port makes the second
# job's rendezvous fail, or worse, join the first job's.
#
# Multi-node and single-node need different answers, because every node running
# torchrun must agree on the port:
#
#   NNODES > 1   it has to be *derived*, identically on every node, so probing is not
#                an option. Derived from SLURM_JOB_ID, which is unique among
#                concurrent jobs, over a wide range: a collision needs two live jobs
#                whose ids differ by exactly the modulus. This used to be `% 400`,
#                which is only about 1 in 400 per pair of co-resident jobs -- small,
#                but these are multi-hour queue waits to lose.
#   NNODES = 1   ask the kernel for a free port, which cannot collide at all. Safe
#                here because this script runs once per job (the shipped job scripts
#                are --nodes=1 --ntasks-per-node=1 and call it directly, not via srun).
JOB_ID="${SLURM_JOB_ID:-0}"
MASTER_PORT=$(( 29600 + (JOB_ID % 20000) ))
if [[ "${NNODES}" -gt 1 ]]; then
    HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
    RDZV_ENDPOINT="${HEAD_NODE}:${MASTER_PORT}"
else
    # Bind port 0, read what the kernel picked, release it. There is a small window
    # between releasing and torchrun binding; the derived port is the fallback if the
    # probe itself fails (no python on PATH, a hostile sandbox).
    PROBED=$(python3 -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()' 2>/dev/null || true)
    [[ -n "${PROBED}" ]] && MASTER_PORT="${PROBED}"
    RDZV_ENDPOINT="localhost:${MASTER_PORT}"
fi
echo "rendezvous  : ${RDZV_ENDPOINT}"

overrides=()
[[ -n "${TOKENIZER}" ]] && overrides+=(--model.hf_assets_path "${TOKENIZER}")
[[ -n "${SHARDS}" ]] && overrides+=(--training.dataset_path "${SHARDS}")
[[ -n "${DUMP}" ]] && overrides+=(--job.dump_folder "${DUMP}")

# --- validation, made identical across sites --------------------------------
#
# The held-out loss is the only cross-site check there is: every cluster evaluates the
# *same* global weights, so a disagreement is a bug rather than variance. That only holds
# if the sites score the same data, and two things stopped them.
#
#   dataset_path  relative in the TOML ("./third_party/..."), so it resolved only because
#                 both jobs happened to run with the repo as their working directory. A
#                 different CWD, or a container without the submodule visible, and
#                 validation reads nothing or reads something else.
#
#   steps         total windows scored = steps x local_batch_size x nproc, so a fixed
#                 `steps` scales with the rank count. At 20 steps and batch 8 that is 640
#                 windows on 4 ranks and 1,280 on 8 -- against a fixture of ~543 windows,
#                 so the sites re-looped it 1.2x and 2.4x and weighted the partial tail
#                 differently. Observed as 1,310,720 vs 2,621,440 eval tokens.
#
# Fixing the window TOTAL instead of the step count makes coverage identical: the loader
# shards round-robin, so taking the first k windows on each of n ranks covers windows
# 0..(n*k - 1) for any n. Same union, same mean, comparable number.
VAL_DATA="${PWW_VAL_DATA:-${PWW_ROOT}/third_party/torchtitan/tests/assets/c4_test}"
if [[ -d "${VAL_DATA}" ]]; then
    overrides+=(--validation.dataset_path "${VAL_DATA}")
else
    echo "WARNING: no validation data at ${VAL_DATA}; leaving validation.dataset_path as" \
         "configured, which is relative and depends on the working directory" >&2
fi

VAL_WINDOWS="${PWW_VAL_WINDOWS:-512}"
if [[ "${VAL_WINDOWS}" != "0" ]]; then
    val_batch="$(grep -m1 -E '^[[:space:]]*local_batch_size[[:space:]]*=' "${CONFIG}" \
                 | cut -d= -f2 | tr -d ' ')"
    if [[ "${val_batch}" =~ ^[1-9][0-9]*$ ]]; then
        per_step=$(( val_batch * NPROC ))
        val_steps=$(( VAL_WINDOWS / per_step ))
        (( val_steps < 1 )) && val_steps=1
        if (( val_steps * per_step != VAL_WINDOWS )); then
            echo "NOTE: PWW_VAL_WINDOWS=${VAL_WINDOWS} is not a multiple of ${per_step}" \
                 "(local_batch_size ${val_batch} x ${NPROC} ranks), so this site will" \
                 "score $(( val_steps * per_step )) windows. Sites whose rank counts give" \
                 "a different remainder will not be comparable -- choose a multiple of" \
                 "every site's local_batch_size x nproc." >&2
        fi
        overrides+=(--validation.steps "${val_steps}")
        echo "validation  : ${val_steps} steps x ${val_batch} x ${NPROC} ranks =" \
             "$(( val_steps * per_step )) windows (PWW_VAL_WINDOWS=${VAL_WINDOWS})"
    else
        echo "WARNING: could not read local_batch_size from ${CONFIG}; leaving" \
             "validation.steps as configured. The held-out loss will not be comparable" \
             "between sites with different rank counts." >&2
    fi
fi

# The federation endpoints. Only added when the config actually asks for DARL, so
# a plain single-site torchtitan run through this launcher needs no coordinator.
if grep -qE '^[[:space:]]*dataset[[:space:]]*=[[:space:]]*"pww_tokens"' "${CONFIG}"; then
    overrides+=(--darl.url "http://${CENTRAL}:${DARL_PORT}" --darl.site "${SITE}")
    # --replica is REQUIRED when you run more than one job at the same site at the
    # same time, and it is not a performance knob -- it is a correctness one.
    #
    # The DARL cluster id defaults to the site name alone ("lumi"), deliberately, so
    # that a job killed at walltime and requeued is recognised as the same cluster and
    # keeps its measured throughput -- which is what sizes its grants. The cost is
    # that two *concurrent* jobs at one site both call themselves "lumi", and the
    # coordinator cannot tell them apart. Two things then go wrong silently:
    #
    #   * /release with no lease id releases every lease held by that cluster id, so
    #     the first job to finish hands back the second job's live leases. The second
    #     keeps training blocks that are back in the free pool -- duplicate work, and
    #     no assertion anywhere catches it.
    #   * the delta blob is named by (run, round, cluster), so both jobs upload to the
    #     same object each round and one silently overwrites the other. A whole round
    #     of one job's work disappears.
    #
    # Passing --replica a makes the id "lumi-a", which is unique per job and still
    # stable across that job's own requeues.
    [[ -n "${REPLICA}" ]] && overrides+=(--darl.cluster_id "${SITE}-${REPLICA}")
fi
if grep -qE '^[[:space:]]*enable[[:space:]]*=[[:space:]]*true' <(sed -n '/^\[flower\]/,/^\[/p' "${CONFIG}"); then
    overrides+=(--flower.server_address "${CENTRAL}:${FLOWER_PORT}")
fi

echo "=============================================================="
pww_summary
cat <<EOF
config      : ${CONFIG}
ranks       : ${NPROC} x ${NNODES} node(s)
rendezvous  : ${RDZV_ENDPOINT}
tokenizer   : ${TOKENIZER:-<from config>}
shards      : ${SHARDS:-<from config>}
central     : ${CENTRAL} (darl ${DARL_PORT}, flower ${FLOWER_PORT})
overrides   : ${overrides[*]:-<none>} ${EXTRA[*]:-}
EOF
echo "=============================================================="

# `python3 -m torch.distributed.run` rather than the torchrun entrypoint: inside
# the LUMI container the console script is not always on PATH, and the module form
# always is.
pww_run python3 -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NPROC}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${RDZV_ENDPOINT}" \
    --role rank --tee 3 \
    -m pww.titan.train \
    --job.config-file "${CONFIG}" \
    "${overrides[@]}" \
    ${EXTRA[@]+"${EXTRA[@]}"}
