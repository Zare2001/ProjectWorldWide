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

# Derived rather than fixed, so two jobs on the same node cannot collide on the
# rendezvous port. Same trick the existing per-site job scripts use.
JOB_ID="${SLURM_JOB_ID:-0}"
MASTER_PORT=$(( 29600 + (JOB_ID % 400) ))
if [[ "${NNODES}" -gt 1 ]]; then
    HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
    RDZV_ENDPOINT="${HEAD_NODE}:${MASTER_PORT}"
else
    RDZV_ENDPOINT="localhost:${MASTER_PORT}"
fi

overrides=()
[[ -n "${TOKENIZER}" ]] && overrides+=(--model.hf_assets_path "${TOKENIZER}")
[[ -n "${SHARDS}" ]] && overrides+=(--training.dataset_path "${SHARDS}")
[[ -n "${DUMP}" ]] && overrides+=(--job.dump_folder "${DUMP}")

# The federation endpoints. Only added when the config actually asks for DARL, so
# a plain single-site torchtitan run through this launcher needs no coordinator.
if grep -qE '^[[:space:]]*dataset[[:space:]]*=[[:space:]]*"pww_tokens"' "${CONFIG}"; then
    overrides+=(--darl.url "http://${CENTRAL}:${DARL_PORT}" --darl.site "${SITE}")
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
