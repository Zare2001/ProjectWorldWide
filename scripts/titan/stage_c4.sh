#!/usr/bin/env bash
# Download raw allenai/c4 shards onto this site's scratch.
#
#   scripts/titan/stage_c4.sh --files 32            # ~5B tokens of C4-en train
#   scripts/titan/stage_c4.sh --files 4 --split validation
#
# LOGIN NODE ONLY -- compute nodes on LUMI and Snellius have no internet.
#
# Only needed if you want to (re)tokenise inside the facility, or to keep the raw
# text around. The shorter path for a first real run is to skip this and let
# tokenize_c4.sh stream from the hub directly:
#
#   scripts/titan/tokenize_c4.sh --dataset c4 --max-files 32
#
# Staging first is worth it when you expect to re-tokenise (a tokenizer change, a
# different seq_len), since it avoids re-downloading hundreds of GB, and it makes
# the tokenisation step reproducible from a fixed local input rather than from
# whatever the hub serves that day.
#
# C4-en is 1024 train files of ~350 MB compressed, ~156B tokens in total. Take a
# slice: 32 files is already ~5B tokens, which is more than a 0.6B model needs to
# be compute-bound rather than data-bound.
set -euo pipefail

FILES=32
SPLIT="train"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --files) FILES="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

PWW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PWW_ROOT}/env.sh"

OUT_DIR="${OUT_DIR:-${PWW_DATA_DIR}/c4/en}"
mkdir -p "${OUT_DIR}"

echo "site   : ${PWW_SITE}"
echo "split  : ${SPLIT}, ${FILES} file(s)"
echo "output : ${OUT_DIR}"
echo

pww_run python3 - "${OUT_DIR}" "${SPLIT}" "${FILES}" <<'PY'
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

out_dir, split, count = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
# C4-en's own file naming; validation has 8 shards, train has 1024.
total = 1024 if split == "train" else 8
if count > total:
    print(f"{split} has only {total} files; downloading all of them")
    count = total

for index in range(count):
    name = f"en/c4-{split}.{index:05d}-of-{total:05d}.json.gz"
    path = os.path.realpath(hf_hub_download(
        repo_id="allenai/c4", filename=name, repo_type="dataset",
        local_dir=str(out_dir.parent.parent / "c4-hub"),
    ))
    # Flattened into one directory, which is the layout datasets.py's c4_local loader
    # globs for.
    #
    # A HARD LINK, not a symlink. Both are free -- hf_hub_download already has the bytes
    # and the cache is on the same filesystem -- but a symlink does not survive being
    # copied between sites, and that failure is silent in the worst way:
    #
    #   `rsync -a` implies `-l`, so copying a staged directory from one site to the other
    #   transfers the *link*, still pointing at the source site's absolute c4-hub path.
    #   On the destination it dangles. `ls` lists a dangling symlink exactly like a real
    #   file, so the directory looks correctly staged, and the run only fails minutes
    #   later inside HuggingFace's loader with a FileNotFoundError buried in a
    #   multi-rank traceback. Observed on Snellius after rsyncing LUMI's copy across.
    #
    # A hard link is a real directory entry, so rsync copies the bytes and the
    # destination is self-contained. Falls back to a copy across filesystems.
    target = out_dir / Path(name).name
    # A previous run's dangling symlink still occupies the path, and `exists()` is False
    # for one -- so the old `if not link.exists(): link.symlink_to(...)` raised
    # FileExistsError and re-staging could not repair a broken link.
    if target.is_symlink() or target.exists():
        target.unlink()
    try:
        os.link(path, target)
        how = "hardlink"
    except OSError:
        shutil.copy2(path, target)
        how = "copy"
    print(f"[{index + 1}/{count}] {target.name} ({how})")
PY

echo
if [[ "${SPLIT}" == "validation" ]]; then
    # Do NOT tokenise this: the validator consumes raw text through the c4_local
    # loader, and run_train.sh selects it automatically from this conventional path.
    if [[ "${OUT_DIR}" == "${PWW_DATA_DIR}/c4-validation" ]]; then
        echo "Staged. run_train.sh will use it automatically (validation: c4_local)."
    else
        echo "Staged. run_train.sh auto-detects \$PWW_DATA_DIR/c4-validation; for this"
        echo "location pass PWW_VAL_DATA=${OUT_DIR} at submit time instead."
    fi
    echo "Confirm both sites staged identical bytes:"
    echo "  sha256sum ${OUT_DIR}/c4-validation.*.json.gz"
else
    echo "Staged. Tokenise it with:"
    echo "  PWW_C4_DIR=${OUT_DIR} scripts/titan/tokenize_c4.sh --dataset c4_local --seq-len 2048"
fi
