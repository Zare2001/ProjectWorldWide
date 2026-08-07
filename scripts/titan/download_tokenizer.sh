#!/usr/bin/env bash
# Fetch a HuggingFace tokenizer and save it in the fast (tokenizer.json) format
# torchtitan's build_hf_tokenizer expects.
#
#   scripts/titan/download_tokenizer.sh                       # OpenEuroLLM 128k
#   scripts/titan/download_tokenizer.sh --repo-id Qwen/Qwen3-0.6B
#
# Run this on a LOGIN node. Compute nodes on LUMI and Snellius have no internet,
# so the tokenizer -- like the corpus -- has to be staged onto scratch first.
#
# The default is openeurollm/tokenizer-128k. Note that it is *not* a Qwen
# tokenizer: it reports 131073 ids against Qwen3's own 151936. That is handled
# rather than worked around -- src/pww/titan/__init__.py's PWWQwen3ModelArgs reads the vocab size and
# EOS id back off whatever tokenizer a run points at and rebuilds the model
# accordingly, so no config has to restate them. The number this script prints is
# the one that ends up as the embedding size.
#
# Changing tokenizer invalidates any token shards built with the old one. That is
# enforced, not just documented: the shard manifest records the tokenizer's
# sha256 and src/pww/titan/shards.py refuses to train on a mismatch (every token
# id would mean a different piece of text while the index space still looked
# valid).
set -euo pipefail

REPO_ID="openeurollm/tokenizer-128k"
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-id|--repo_id) REPO_ID="$2"; shift 2 ;;
        --output-dir|--output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

PWW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PWW_ROOT}/env.sh"

# Derived from the repo id so two tokenizers can coexist on the same scratch and
# a shard directory can name which one it was built with.
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${PWW_DATA_DIR}/tokenizers/$(basename "${REPO_ID}")"
fi

mkdir -p "${OUTPUT_DIR}"
echo "site        : ${PWW_SITE}"
echo "tokenizer   : ${REPO_ID}"
echo "destination : ${OUTPUT_DIR}"

pww_run python3 - "${REPO_ID}" "${OUTPUT_DIR}" <<'PY'
import sys
from transformers import AutoTokenizer

repo_id, out_dir = sys.argv[1], sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(repo_id)
tokenizer.save_pretrained(out_dir)

# vocab_size excludes added/special tokens on some tokenizers, and the embedding
# has to cover every id the tokenizer can emit -- so report both and let the
# training-time override (which reads the fast tokenizer directly) be the
# authority.
print(f"vocab_size            : {tokenizer.vocab_size}")
print(f"len(tokenizer)        : {len(tokenizer)}")
print(f"bos / eos             : {tokenizer.bos_token!r} / {tokenizer.eos_token!r}")
PY

if [[ ! -f "${OUTPUT_DIR}/tokenizer.json" ]]; then
    echo >&2
    echo "ERROR: no tokenizer.json in ${OUTPUT_DIR}." >&2
    echo "torchtitan's HuggingFaceTokenizer needs the fast format; this repo only" >&2
    echo "ships a slow SentencePiece model. Convert it with a transformers version" >&2
    echo "that can, or pick a repo that publishes tokenizer.json." >&2
    exit 1
fi

echo
echo "Tokenizer ready. Point a run at it with:"
echo "  --model.hf_assets_path ${OUTPUT_DIR}"
echo "Then tokenise a corpus against it:"
echo "  scripts/titan/tokenize_c4.sh --tokenizer ${OUTPUT_DIR}"
