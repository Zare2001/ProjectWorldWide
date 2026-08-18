#!/bin/bash
# Deploy or update this repo on whichever machine you are on.
# Everything machine-specific lives in sites/<site>.sh; env.sh picks it by detection.
#
#   ./scripts/deploy.sh
#
# Idempotent -- this is also the normal way to update after a commit.

set -euo pipefail

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PWW_ROOT}"

git pull --ff-only
git submodule update --init --recursive

# shellcheck source=/dev/null
source "${PWW_ROOT}/env.sh"
pww_summary

mkdir -p "${PWW_DATA_DIR}" "${PWW_OUTPUT_DIR}" "${PWW_TMPDIR}" "${PWW_CACHE_DIR}" logs

# runs/ and data/ are symlinks into scratch. `ln -sfn` puts the link INSIDE an
# existing real directory, which silently gives you runs/runs and breaks every
# documented path -- so refuse that case instead.
for pair in "runs:${PWW_OUTPUT_DIR}" "data:${PWW_DATA_DIR}"; do
    name="${pair%%:*}"; target="${pair#*:}"
    if [[ -e "${name}" && ! -L "${name}" ]]; then
        echo "ERROR: ${name}/ is a real directory; move it aside: mv ${name} ${name}.bak" >&2
        exit 1
    fi
    ln -sfn "${target}" "${name}"
    echo "  ${name} -> $(readlink "${name}")"
done

# torch >= 2.9 environment. The site file says where it lives and how to build it;
# building is a separate batch job, not this script's business.
if declare -F pww_titan_env >/dev/null; then
    IFS=$'\t' read -r kind path build < <(pww_titan_env)
    if [[ "${kind}" != none && ! -e "${path}" ]]; then
        echo "  ${kind} missing -- build it once:  ${build}"
    fi
fi

echo
echo "next:"
case "${PWW_SITE}" in
    central) echo "  ./scripts/central_node/start_central_services.sh" ;;
    *)       echo "  ./scripts/titan/download_tokenizer.sh"
             echo "  ./scripts/titan/tokenize_c4.sh --dataset c4 --seq-len 2048 --max-files 32"
             echo "  DARL_TOKEN=... sbatch scripts/${PWW_SITE}/job_titan_diloco.sh" ;;
esac
