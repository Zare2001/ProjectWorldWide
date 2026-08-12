#!/usr/bin/env bash
# Clear a run's state so the next submission starts from step 0 rather than resuming.
#
#   ./scripts/reset_run.sh                     # every shipped titan config, at this site
#   ./scripts/reset_run.sh --config configs/titan/qwen3_0.6b_c4_central_1k.toml
#   ./scripts/reset_run.sh --dry-run           # print what would go, touch nothing
#   ./scripts/reset_run.sh --yes               # no confirmation prompt
#   ./scripts/reset_run.sh --central           # ALSO the central VM's global model +
#                                              # lease table (run this ON the central VM)
#
# WHY THIS EXISTS
# ---------------
# Resuming is the default everywhere, deliberately: an HPC job killed at walltime and
# resubmitted has to continue, not restart. The cost is that "start over" is not the
# absence of an action, it is an action -- and it has to reach every store at once,
# because resetting some of them is worse than resetting none:
#
#   <dump>/checkpoint       model, optimiser moments, LR schedule position, dataloader
#                           state. Left behind, a "fresh" run resumes at the old global
#                           step, so a freshly seeded model skips warmup entirely and
#                           takes near-peak LR on its first step.
#   <dump>/blob-staging     one full model per staged blob. Harmless to correctness,
#                           expensive in scratch quota.
#   <dump>/tb               tensorboard event files, one directory per launch.
#   outputs/darl_local_*    per-job state dirs from the central-baseline job scripts,
#                           which spawn their own throwaway coordinator and never clean
#                           up after themselves.
#   runs/darl               (--central) the lease table: which windows have been trained
#   runs/central/global     (--central) the global model, its momentum buffer, checkpoints
#
# This script covers the site half. The central half is start_central_services.sh's
# PWW_FRESH_RUN=1, which --central runs for you.
#
# THE CENTRAL BASELINE NEEDS THIS TOO
# -----------------------------------
# It is easy to assume only the DiLoCo run has resumable state, because it is the one with
# a server. It is not: torchtitan's checkpointer does not know whether Flower is involved,
# so configs/titan/qwen3_0.6b_c4_central*.toml resume from ./outputs/qwen3-0.6b-c4-central*
# in exactly the same way. A baseline resubmitted without a reset continues the previous
# one while its throwaway coordinator hands out a block space that was just reset to epoch
# 0 -- a curve that is neither the old run nor a new one, and nothing in the log says so.
set -euo pipefail

PWW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIGS=()
DRY_RUN=0
ASSUME_YES=0
DO_CENTRAL=0

usage() { sed -n '2,44p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)  CONFIGS+=("$2"); shift 2 ;;
        --dry-run|-n) DRY_RUN=1; shift ;;
        --yes|-y)  ASSUME_YES=1; shift ;;
        --central) DO_CENTRAL=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

# Default to every config that has a dump folder, central and DiLoCo alike. Resetting one
# side of a comparison and not the other is its own kind of wrong answer, and the whole
# point of the pair is that they are otherwise identical.
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    while IFS= read -r cfg; do CONFIGS+=("${cfg}"); done < <(
        find "${PWW_ROOT}/configs/titan" -name '*.toml' | sort
    )
fi

# `dump_folder = "./outputs/..."` -> an absolute path resolved the way a job would.
# TOML has no environment expansion, so a shipped dump_folder is always repo-relative.
dump_folder_of() {
    local cfg="$1" dump
    dump="$(grep -m1 -E '^[[:space:]]*dump_folder[[:space:]]*=' "${cfg}" 2>/dev/null \
            | cut -d'"' -f2 || true)"
    [[ -z "${dump}" ]] && return 1
    [[ "${dump}" != /* ]] && dump="${PWW_ROOT}/${dump#./}"
    printf '%s' "${dump}"
}

human_size() { du -sh "$1" 2>/dev/null | cut -f1 || echo "?"; }

TARGETS=()
for cfg in "${CONFIGS[@]}"; do
    [[ -f "${cfg}" ]] || { echo "no such config: ${cfg}" >&2; exit 1; }
    dump="$(dump_folder_of "${cfg}")" || {
        echo "note: ${cfg#"${PWW_ROOT}/"} has no dump_folder; skipping"
        continue
    }
    [[ -d "${dump}" ]] || continue
    for sub in checkpoint checkpoint.superseded blob-staging tb; do
        [[ -e "${dump}/${sub}" ]] && TARGETS+=("${dump}/${sub}")
    done
done

# The throwaway coordinator state dirs left by job_titan_central.sh, one per Slurm job id.
while IFS= read -r stale; do
    TARGETS+=("${stale}")
done < <(find "${PWW_ROOT}/outputs" -maxdepth 1 -type d -name 'darl_local_central_*' \
         2>/dev/null | sort)

if [[ ${#TARGETS[@]} -eq 0 ]] && [[ "${DO_CENTRAL}" != "1" ]]; then
    echo "nothing to reset: no checkpoints, staging or local coordinator state found."
    echo "(configs checked: ${#CONFIGS[@]})"
    exit 0
fi

echo "=============================================================="
echo " reset_run.sh -- these paths will be DELETED"
echo "=============================================================="
for path in ${TARGETS[@]+"${TARGETS[@]}"}; do
    printf '  %6s  %s\n' "$(human_size "${path}")" "${path#"${PWW_ROOT}/"}"
done
if [[ "${DO_CENTRAL}" == "1" ]]; then
    echo "  ALSO: the central node's lease table and global model, via"
    echo "        PWW_FRESH_RUN=1 scripts/central_node/start_central_services.sh"
fi
echo "--------------------------------------------------------------"
echo "NOT touched (and usually should not be):"
echo "  the tokenised corpus under \$PWW_DATA_DIR -- reset it and you re-tokenise C4"
echo "  logs/ and wandb/ -- a finished run's record is the only copy of it"
if [[ "${DO_CENTRAL}" != "1" ]]; then
    echo "  the central node's global model and lease table -- pass --central, ON that VM"
fi
echo "=============================================================="

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "--dry-run: nothing was deleted."
    exit 0
fi

if [[ "${ASSUME_YES}" != "1" ]]; then
    read -r -p "Delete the above? Type 'yes' to confirm: " reply
    [[ "${reply}" == "yes" ]] || { echo "aborted; nothing was deleted."; exit 1; }
fi

for path in ${TARGETS[@]+"${TARGETS[@]}"}; do
    rm -rf "${path}"
    echo "deleted ${path#"${PWW_ROOT}/"}"
done

if [[ "${DO_CENTRAL}" == "1" ]]; then
    central="${PWW_ROOT}/scripts/central_node/start_central_services.sh"
    if [[ ! -x "${central}" ]]; then
        echo "ERROR: ${central} is not executable; run the central half by hand" >&2
        exit 1
    fi
    # Stop first: start_central_services.sh is a no-op for a daemon that is already
    # running, so --fresh-model and --fresh would never reach either process.
    "${PWW_ROOT}/scripts/central_node/stop_central_services.sh" || true
    echo "restarting central services with PWW_FRESH_RUN=1..."
    PWW_FRESH_RUN=1 "${central}"
fi

cat <<EOF

Site state cleared. The next submission starts from step 0.

Submit with PWW_FRESH_RUN=1 as well, so that a checkpoint written between this reset and
the job actually starting cannot be picked up:

    PWW_FRESH_RUN=1 DARL_TOKEN="\$DARL_TOKEN" \\
      sbatch --export=ALL,PWW_FRESH_RUN,DARL_TOKEN scripts/snellius/job_titan_diloco.sh

    PWW_FRESH_RUN=1 \\
      sbatch --export=ALL,PWW_FRESH_RUN scripts/snellius/job_titan_central.sh

Use a fresh WANDB_RUN_NAME too, or the new run overlays the old one under the same name:

    WANDB_PROJECT=pww-diloco-1k WANDB_RUN_NAME=diloco-snellius-v2 ...
EOF
