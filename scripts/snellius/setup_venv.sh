#!/bin/bash
# Build the Snellius Python environment. RUN FROM A LOGIN NODE (needs internet).
#
#   ./scripts/snellius/setup_venv.sh
#
# Why a venv at all, when LUMI needs nothing?
#
# LUMI ships a maintained container with torch 2.7.1 and the whole AI stack.
# Snellius has no equivalent. Its only PyTorch module is
# PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 in the *2023* tree (the 2024 tree's
# PyTorch directory is empty and 2025 has no ai modules at all), and 2.1.2 is
# too old for this codebase -- it lacks every one of:
#
#   torch.distributed.device_mesh.init_device_mesh   (parallel.build_mesh)
#   torch.distributed.checkpoint.state_dict          (checkpoint.py, all of it)
#   init_process_group(device_id=...)                (distributed.setup)
#   FSDP(device_mesh=...)                            (parallel.wrap_model)
#
# plus it has no torchvision/transformers/datasets in the same module. So the
# venv is not an optional extra here; it is the environment.
#
# Torch is pinned to 2.7.1 to match the LUMI container exactly. That is the
# point of the project: the same source running on both machines should not be
# running against two different torch generations.
#
# The pip wheels bundle their own CUDA runtime (nvidia-*-cu12), so no CUDA
# module is loaded and nothing here depends on the EasyBuild CUDA version.

set -euo pipefail

PWW_ROOT="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"

# Default to $HOME: persistent. Deliberately NOT /scratch-shared, which is
# purged on file age -- an environment that dissolves after two idle weeks is
# worse than no environment.
PWW_VENV="${PWW_VENV:-${HOME}/venvs/pww-snellius}"

PWW_PYTHON_MODULE="${PWW_PYTHON_MODULE:-Python/3.12.3-GCCcore-13.3.0}"
PWW_MODULE_YEAR="${PWW_MODULE_YEAR:-2024}"

# Match the LUMI container, version for version, wherever a version exists there.
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1}"

echo "venv    : ${PWW_VENV}"
echo "python  : ${PWW_MODULE_YEAR} / ${PWW_PYTHON_MODULE}"
echo "torch   : ${TORCH_VERSION} (LUMI container parity)"
echo

if ! command -v module >/dev/null 2>&1; then
    for _init in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
        # shellcheck source=/dev/null
        [[ -r "${_init}" ]] && source "${_init}" && break
    done
fi

module load "${PWW_MODULE_YEAR}"
module load "${PWW_PYTHON_MODULE}"

if [[ ! -d "${PWW_VENV}" ]]; then
    echo "creating venv..."
    # No --system-site-packages: the module tree's numpy/scipy would shadow the
    # versions the torch wheels want, and this env owns its whole stack anyway.
    python3 -m venv "${PWW_VENV}"
fi

# shellcheck source=/dev/null
source "${PWW_VENV}/bin/activate"

python3 -m pip install --upgrade pip setuptools wheel

echo
echo "installing torch ${TORCH_VERSION} (several GB, a few minutes)..."
python3 -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"

echo
echo "installing the rest of the stack..."
# Versions mirror what the LUMI container carries, so the LLM phase does not
# silently diverge between sites.
# accelerate is pinned like the rest: left unpinned it resolves to 1.x, which is a
# major version away from the LUMI container's 0.34.2 and quietly breaks the
# "same stack on both sites" property this file exists to maintain. 0.34.2 with
# transformers 4.55.3 is the combination LUMI already runs, so it is proven.
python3 -m pip install \
    "transformers==4.55.3" \
    "tokenizers==0.21.4" \
    "datasets==4.0.0" \
    "accelerate==0.34.2" \
    "pyyaml"

echo
echo "=== verification ==="
python3 - <<'EOF'
import importlib

import torch

print(f"  torch          {torch.__version__}  (cuda {torch.version.cuda})")
for m in ("torchvision", "transformers", "tokenizers", "datasets", "accelerate", "yaml", "numpy"):
    try:
        print(f"  {m:14s} {getattr(importlib.import_module(m), '__version__', '?')}")
    except Exception:
        print(f"  {m:14s} MISSING")

# The four APIs the 2.1.2 module lacks -- the reason this venv exists. If any of
# these regress, jobs fail deep inside a queued allocation instead of here.
import inspect

import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict  # noqa: F401
from torch.distributed.device_mesh import init_device_mesh  # noqa: F401
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

assert "device_id" in inspect.signature(dist.init_process_group).parameters
assert "device_mesh" in inspect.signature(FSDP.__init__).parameters
print("  required distributed APIs present")
EOF

echo
echo "done. point env.sh at it with:"
echo "  export PWW_VENV=${PWW_VENV}"
echo "(sites/snellius.sh already defaults to this path)"
