#!/bin/bash
#SBATCH --job-name=pww-tokenize-c4
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Tokenise a real corpus into shards, as a batch job.
#
#   sbatch -A $PWW_ACCOUNT scripts/lumi/job_tokenize_c4.sh
#   MAX_FILES=64 sbatch -A $PWW_ACCOUNT scripts/lumi/job_tokenize_c4.sh
#
# RUNBOOK.md Part 1 says to run tokenize_c4.sh on a login node, and on Snellius
# that is fine. On LUMI it is not: 32 files of C4-en is ~11M documents at roughly
# 500 docs/s, so ~6 hours pinning a core on a shared login node, which is what
# LUMI's usage policing exists to kill. The `small` partition has internet -- the
# only reason the login node was specified at all -- so the streaming download
# works here exactly the same way.
#
# The fixture (--dataset c4_test, seconds) does NOT need this. Run that one
# directly per the runbook.

set -euo pipefail

: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
source "${PWW_ROOT}/env.sh"

# tokenize_c4.sh sources env.sh in its own shell, so exporting the container is
# what redirects it -- PWW_LAUNCH is rebuilt there and would not survive anyway.
export PWW_CONTAINER="${PWW_TITAN_SIF:-${PWW_SCRATCH}/containers/pww-titan.sif}"
if [[ ! -r "${PWW_CONTAINER}" ]]; then
    echo "ERROR: no torchtitan container at ${PWW_CONTAINER}" >&2
    echo "build it once: sbatch scripts/lumi/build_titan_container.sh" >&2
    exit 1
fi

DATASET="${DATASET:-c4}"
SEQ_LEN="${SEQ_LEN:-2048}"
MAX_FILES="${MAX_FILES:-32}"

# The Rust tokenizer parallelises across the cores requested above; left unset it
# prints a fork warning and serialises.
export TOKENIZERS_PARALLELISM=true
export SINGULARITYENV_TOKENIZERS_PARALLELISM=true

exec "${PWW_ROOT}/scripts/titan/tokenize_c4.sh" \
    --dataset "${DATASET}" \
    --seq-len "${SEQ_LEN}" \
    --max-files "${MAX_FILES}"
