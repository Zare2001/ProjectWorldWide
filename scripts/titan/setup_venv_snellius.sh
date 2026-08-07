#!/usr/bin/env bash
# Build the Snellius venv for the torchtitan path. Login node, once.
#
#   ./scripts/titan/setup_venv_snellius.sh
#
# This is a SECOND venv, deliberately separate from scripts/snellius/setup_venv.sh's
# torch 2.7.1 one. torchtitan at the pinned commit needs torch >= 2.9 (FSDP2/DTensor
# APIs that 2.7.1 does not have), and the 2.7.1 pin exists so LUMI and Snellius
# numbers are comparable -- so neither environment can move to accommodate the
# other. See scripts/titan/README.md.
#
# The existing CIFAR and HuggingFace-LLM paths keep using the 2.7.1 venv; nothing
# is migrated.
set -euo pipefail

# Keep in step with the LUMI container's torch (containers/titan-lumi.def). Same
# minor version on both sites is what makes a cross-site comparison mean anything,
# even though one side is ROCm and the other CUDA.
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
CUDA_CHANNEL="${CUDA_CHANNEL:-cu128}"

PWW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PWW_ROOT}/env.sh"

if [[ "${PWW_SITE}" != "snellius" ]]; then
    echo "this script is for Snellius; detected site '${PWW_SITE}'" >&2
    echo "override with PWW_SITE=snellius if you know what you are doing" >&2
    exit 1
fi

VENV="${PWW_TITAN_VENV:-${HOME}/venvs/pww-titan-snellius}"

echo "python  : $(command -v python3) ($(python3 -V 2>&1))"
echo "torch   : ${TORCH_VERSION}+${CUDA_CHANNEL}"
echo "venv    : ${VENV}"
echo

if [[ -d "${VENV}" ]]; then
    echo "${VENV} already exists. Remove it to rebuild:" >&2
    echo "  rm -rf ${VENV}" >&2
    exit 1
fi

python3 -m venv "${VENV}"
# shellcheck source=/dev/null
source "${VENV}/bin/activate"
python3 -m pip install --quiet --upgrade pip wheel

echo "installing torch ${TORCH_VERSION} (several GB, a few minutes)..."
python3 -m pip install "torch==${TORCH_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"

# Constrain torch before anything else is resolved, so no transitive dependency
# can quietly pull a different build over the one just installed.
CONSTRAINT="$(mktemp)"
python3 -m pip list --format=freeze | grep -E '^(torch|triton|pytorch-triton)==' > "${CONSTRAINT}"
cat "${CONSTRAINT}"

echo "installing torchtitan's dependencies..."
python3 -m pip install -c "${CONSTRAINT}" \
    "torchdata>=0.8.0" \
    "datasets>=3.6.0" \
    "tokenizers>=0.15.0" \
    "tyro>=1.0.5" \
    safetensors \
    tensorboard \
    tabulate \
    fsspec \
    "zstandard>=0.22" \
    "sentencepiece>=0.2.0" \
    transformers \
    "huggingface_hub[cli]"

# The Flower client half. The FedMom strategy lives on the central node, but the
# client needs flwr here.
echo "installing flwr..."
python3 -m pip install -c "${CONSTRAINT}" "flwr>=1.20"

rm -f "${CONSTRAINT}"

echo
python3 - <<'PY'
import torch
print(f"torch          : {torch.__version__}")
print(f"cuda available : {torch.cuda.is_available()}")
print(f"device count   : {torch.cuda.device_count()}")
# The APIs torchtitan needs and 2.7.1 lacks -- fail here rather than mid-job.
from torch.distributed.fsdp import fully_shard  # noqa: F401
from torch.distributed.tensor import DTensor    # noqa: F401
print("fsdp2 + dtensor: ok")
PY

cat <<EOF

Done. This venv is NOT sourced by env.sh (which activates the 2.7.1 one).
Use it by setting PWW_TITAN_VENV and letting the titan job scripts activate it:

  export PWW_TITAN_VENV=${VENV}
  sbatch scripts/snellius/job_titan_diloco.sh
EOF
