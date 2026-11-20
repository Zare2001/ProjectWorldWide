#!/bin/bash
# Build the torchtitan venv on Frontier. Run once, on a login node:
#
#   ./scripts/titan/setup_venv_frontier.sh
#
# NOT verified on the machine from this repository. It mirrors
# setup_venv_snellius.sh with the ROCm wheel in place of the CUDA one. torch is
# installed first and every later install is constrained to it, because pip would
# otherwise replace the ROCm build with the CUDA build from PyPI.
set -euo pipefail

TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
ROCM_CHANNEL="${ROCM_CHANNEL:-rocm6.4}"

PWW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PWW_SITE=frontier
source "${PWW_ROOT}/env.sh" 2>/dev/null || true

VENV="${PWW_TITAN_VENV}"
echo "python  : $(command -v python3) ($(python3 -V 2>&1))"
echo "torch   : ${TORCH_VERSION}+${ROCM_CHANNEL}"
echo "venv    : ${VENV}"
if [[ -d "${VENV}" ]]; then
    echo "${VENV} already exists. Remove it to rebuild: rm -rf ${VENV}" >&2
    exit 1
fi

mkdir -p "$(dirname "${VENV}")" "${PWW_CACHE_DIR}" "${PWW_TMPDIR}"
python3 -m venv "${VENV}"
source "${VENV}/bin/activate"
python3 -m pip install --quiet --upgrade pip wheel
python3 -m pip install "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${ROCM_CHANNEL}"

CONSTRAINT="$(mktemp)"
python3 -m pip list --format=freeze | grep -E '^(torch|triton|pytorch-triton|pytorch-triton-rocm)==' > "${CONSTRAINT}"
cat "${CONSTRAINT}"
python3 -m pip install -c "${CONSTRAINT}" \
    "torchdata>=0.8.0" "datasets>=3.6.0" "tokenizers>=0.15.0" "tyro>=1.0.5" \
    safetensors tensorboard tabulate fsspec "zstandard>=0.22" "sentencepiece>=0.2.0" \
    transformers "huggingface_hub[cli]" wandb "flwr>=1.20"
rm -f "${CONSTRAINT}"

python3 - <<'PY'
import torch
print(f"torch   : {torch.__version__}")
assert "rocm" in torch.__version__, "pip replaced the ROCm build of torch"
from torch.distributed.fsdp import fully_shard  # noqa: F401
print("fsdp2   : ok   (GPUs are only visible inside a job)")
PY
echo "Done: ${VENV}"
