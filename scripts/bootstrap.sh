#!/bin/bash
# One-time setup: scratch directories, log dir, convenience symlinks.
#
#   ./scripts/bootstrap.sh

set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/../env.sh"

pww_summary
echo

mkdir -p "${PWW_DATA_DIR}" "${PWW_OUTPUT_DIR}" "${PWW_TMPDIR}" "${PWW_CACHE_DIR}"
mkdir -p "${PWW_ROOT}/logs"

# Symlinks so scratch is reachable from the repo without typing full paths.
# Keeping the real data out of $HOME matters: home is 20 GB with a 100k inode
# limit, and a tokenised LLM corpus will blow through both.
ln -sfn "${PWW_OUTPUT_DIR}" "${PWW_ROOT}/runs"
ln -sfn "${PWW_DATA_DIR}" "${PWW_ROOT}/data"

if [[ -n "${PWW_CONTAINER:-}" && ! -f "${PWW_CONTAINER}" ]]; then
    echo "ERROR: container not found: ${PWW_CONTAINER}" >&2
    exit 1
fi

echo "verifying environment..."
pww_run python3 -c "
import torch, torchvision, transformers
print(f'  torch        {torch.__version__}')
print(f'  torchvision  {torchvision.__version__}')
print(f'  transformers {transformers.__version__}')
import pww
print(f'  pww          {pww.__version__} (importable)')
"

echo
echo "bootstrap complete. next:"
echo "  ./scripts/download_data.sh                          # once, from a login node"
echo "  sbatch scripts/${PWW_SITE}/job_smoke.sh             # validate the distributed stack"
echo "  sbatch scripts/${PWW_SITE}/job_cifar_1node.sh"
