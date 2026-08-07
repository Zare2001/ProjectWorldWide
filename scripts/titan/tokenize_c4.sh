#!/usr/bin/env bash
# Tokenise a corpus into the memmapped shard format DARL leases over.
#
#   # bundled fixture, offline, seconds -- for the smoke config
#   scripts/titan/tokenize_c4.sh --dataset c4_test --seq-len 2048
#
#   # real C4, streamed from the hub (LOGIN NODE ONLY -- compute nodes have no
#   # internet). 32 of C4-en's 1024 train files is ~5B tokens.
#   scripts/titan/tokenize_c4.sh --dataset c4 --seq-len 2048 --max-files 32
#
#   # real C4 from shards already staged on scratch by stage_c4.sh
#   scripts/titan/tokenize_c4.sh --dataset c4_local --seq-len 2048
#
# Do this ONCE per (corpus, tokenizer, seq_len). It is the pass that makes a real
# corpus usable at all: the HuggingFace path in src/pww/pww/data/text.py builds a
# Python list of every token before making a tensor, which is ~3.5 GB of objects
# for WikiText-103 and would be several TB for C4-en.
#
# Both sites must end up with the same window count for DARL's index space to mean
# the same thing on each. Two ways to get that, and the second is cheaper and
# safer: run this identically on both (deterministic given the same inputs), or
# run it once and rsync the output directory across. The manifest digest printed
# at the end is what the coordinator checks -- if the two sites disagree,
# registration fails instead of the two clusters silently training overlapping
# tokens.
set -euo pipefail

DATASET="c4_test"
DATASET_PATH=""
SEQ_LEN=2048
TOKENIZER=""
OUT_DIR=""
MAX_FILES=0
MAX_WINDOWS=0
SPLIT="train"
DTYPE="uint32"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --dataset-path) DATASET_PATH="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        --max-files) MAX_FILES="$2"; shift 2 ;;
        --max-windows) MAX_WINDOWS="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

PWW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PWW_ROOT}/env.sh"

# torchtitan lives as a submodule and is never pip-installed, so it reaches the
# interpreter through PYTHONPATH.
export PYTHONPATH="${PWW_ROOT}/src:${PWW_ROOT}/third_party/torchtitan${PYTHONPATH:+:${PYTHONPATH}}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

[[ -n "${TOKENIZER}" ]] || TOKENIZER="${PWW_DATA_DIR}/tokenizers/tokenizer-128k"
if [[ ! -f "${TOKENIZER}/tokenizer.json" ]]; then
    echo "no tokenizer.json under ${TOKENIZER}" >&2
    echo "run scripts/titan/download_tokenizer.sh first (login node)" >&2
    exit 1
fi

# Names the shard directory after what produced it, so a tokenizer or seq_len
# change lands in a new directory instead of silently mixing incompatible shards.
if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${PWW_DATA_DIR}/${DATASET}-$(basename "${TOKENIZER}")-${SEQ_LEN}"
fi

# torchtitan registers c4_test at a path relative to its own directory, but we run
# from this repo's root, so spell it out.
if [[ "${DATASET}" == "c4_test" && -z "${DATASET_PATH}" ]]; then
    DATASET_PATH="${PWW_ROOT}/third_party/torchtitan/tests/assets/c4_test"
fi

echo "site       : ${PWW_SITE}"
echo "dataset    : ${DATASET} ${DATASET_PATH:+(${DATASET_PATH})}"
echo "tokenizer  : ${TOKENIZER}"
echo "seq_len    : ${SEQ_LEN}  (window ${SEQ_LEN} + 1)"
echo "output     : ${OUT_DIR}"
echo

args=(
    --dataset "${DATASET}"
    --split "${SPLIT}"
    --tokenizer "${TOKENIZER}"
    --seq-len "${SEQ_LEN}"
    --out "${OUT_DIR}"
    --dtype "${DTYPE}"
)
[[ -n "${DATASET_PATH}" ]] && args+=(--dataset-path "${DATASET_PATH}")
[[ "${MAX_FILES}" -gt 0 ]] && args+=(--max-files "${MAX_FILES}")
[[ "${MAX_WINDOWS}" -gt 0 ]] && args+=(--max-windows "${MAX_WINDOWS}")

cd "${PWW_ROOT}"
pww_run python3 -m pww.titan.tokenize_corpus "${args[@]}"

echo
echo "Start the DARL coordinator over this many windows (the 'windows' figure above)."
echo "These are environment variables, not flags:"
echo "  AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml \\"
echo "  NUM_SAMPLES=<windows> BLOCK_SIZE=1024 SEED=42 \\"
echo "      ./scripts/central_node/start_central_services.sh"
echo "Then train with:"
echo "  scripts/titan/run_train.sh --config configs/titan/qwen3_0.6b_c4_diloco.toml \\"
echo "      --shards ${OUT_DIR} --tokenizer ${TOKENIZER}"
