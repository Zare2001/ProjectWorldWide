#!/bin/bash
#SBATCH --job-name=pww-frontier-titan
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# torchtitan (Qwen3) + DARL + Flower/FedMom on Frontier (AMD MI250X, 8 GCDs).
#
#   mkdir -p logs
#   DARL_TOKEN=... sbatch -A <project> --time=16:00:00 scripts/frontier/job_titan_diloco.sh
#
# NOT verified on the machine from this repository: the August 2026 script lived in
# another checkout and this one is written from the Snellius script and the header
# of the August Frontier logs. Submit a short job first and read its first lines.
#
# One task that owns the whole node; torchrun forks the eight ranks, as on Snellius.
# The ranks are started through srun so the script works whether Slurm runs it on a
# compute node or on a batch node.

set -euo pipefail

: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    exit 1
fi
export PWW_SITE=frontier
source "${PWW_ROOT}/env.sh"

if [[ ! -r "${PWW_TITAN_VENV}/bin/activate" ]]; then
    echo "ERROR: no torchtitan venv at ${PWW_TITAN_VENV}" >&2
    echo "build it once from a login node: ./scripts/titan/setup_venv_frontier.sh" >&2
    exit 1
fi

# Outgoing HTTP from a compute node goes through the OLCF proxy.
export http_proxy="${PWW_HTTP_PROXY}" https_proxy="${PWW_HTTP_PROXY}"
export HTTP_PROXY="${PWW_HTTP_PROXY}" HTTPS_PROXY="${PWW_HTTP_PROXY}"
export no_proxy="${PWW_NO_PROXY}" NO_PROXY="${PWW_NO_PROXY}"

CENTRAL_IP="${CENTRAL_IP:-145.38.206.143}"
CONFIG="${CONFIG:-${PWW_ROOT}/configs/titan/qwen3_0.6b_c4_diloco.toml}"
TOKENIZER="${TOKENIZER:-${PWW_DATA_DIR}/tokenizers/tokenizer-128k}"
SHARDS="${SHARDS:-${PWW_DATA_DIR}/c4-tokenizer-128k-2048}"
# Checkpoints go to scratch, not into the repository.
export PWW_DUMP="${PWW_DUMP:-${PWW_OUTPUT_DIR}/qwen3-0.6b-c4-diloco}"

for required in "${TOKENIZER}/tokenizer.json" "${SHARDS}"; do
    [[ -e "${required}" ]] || { echo "ERROR: missing ${required}" >&2; exit 1; }
done

if [[ -z "${DARL_TOKEN:-}" ]]; then
    echo "ERROR: DARL_TOKEN is empty and the coordinator enforces it. Pass the value" \
         "from runs/darl/token on the central node at submit time." >&2
    exit 1
fi

# Reachability and auth through the proxy, before the ranks claim the GPUs.
darl_probe="$(curl -sS -m 20 -o /dev/null -w '%{http_code}' \
    -H "X-DARL-Token: ${DARL_TOKEN}" \
    "http://${CENTRAL_IP}:${PWW_DARL_PORT:-29510}/health" 2>/dev/null)" || true
[[ "${darl_probe}" =~ ^[0-9]{3}$ ]] || darl_probe=000
case "${darl_probe}" in
    200) echo "darl: coordinator reachable from $(hostname), token accepted" ;;
    401) echo "ERROR: the coordinator rejected this DARL_TOKEN." >&2; exit 1 ;;
    *)   echo "ERROR: http://${CENTRAL_IP}:${PWW_DARL_PORT:-29510}/health answered" \
              "${darl_probe} through ${http_proxy}. The run cannot work without it." >&2
         exit 1 ;;
esac
echo "proxy: http_proxy=${http_proxy} no_proxy=${no_proxy}"

# Forward SIGTERM so the leader rank can release its DARL leases before the kill.
trap 'echo "SIGTERM -- forwarding to the training step"; kill -TERM ${TRAIN_PID:-0} 2>/dev/null' TERM

srun --nodes=1 --ntasks=1 --cpus-per-task=56 --gpus-per-node=8 --gpu-bind=none \
    "${PWW_ROOT}/scripts/titan/run_train.sh" \
    --config "${CONFIG}" \
    --tokenizer "${TOKENIZER}" \
    --shards "${SHARDS}" \
    --central "${CENTRAL_IP}" \
    --site frontier \
    --nproc "${PWW_GPUS_PER_NODE}" \
    -- --darl.use_proxy --flower.transport blob --flower.protocol http "$@" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
