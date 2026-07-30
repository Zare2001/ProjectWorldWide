#!/bin/bash
# Pre-fetch datasets onto scratch. RUN FROM A LOGIN NODE.
#
#   ./scripts/download_data.sh
#
# Compute nodes have no direct route to the internet -- only a slow HTTP proxy --
# so downloading inside a job is slow at best and a hang at worst. Fetch once
# here; jobs then run with download=False.

set -euo pipefail

source "$(dirname "$(readlink -f "$0")")/../env.sh"

echo "data dir : ${PWW_DATA_DIR}"
echo "site     : ${PWW_SITE}"
mkdir -p "${PWW_DATA_DIR}/cifar10"

pww_run python3 -c "
from pww.data.cifar import download_cifar10
download_cifar10('${PWW_DATA_DIR}/cifar10')
print('CIFAR-10 ready')
"

du -sh "${PWW_DATA_DIR}/cifar10"
