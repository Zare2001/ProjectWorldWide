#!/bin/bash
# One link of a chained lane: submit the successor, then become the training job.
#
# Submitted by `python3 -m pww.plan` (see src/pww/plan/emit.py), never by hand. It
# carries no #SBATCH directives of its own -- every shape flag is on the sbatch
# command line, copied verbatim from the probe row that measured the queue wait, so
# that the walltime a wait was measured at is the walltime that gets requested.
#
# WHY SELF-RESUBMISSION AND NOT --dependency
# ------------------------------------------
# A job held by --dependency=afterany/singleton is NOT eligible for backfill and
# accrues no eligible-time priority, which forfeits precisely the advantage that
# motivates short jobs: Slurm's backfill scheduler starts a 1 h job in a hole a 40 h
# job cannot fit. Submitting the successor from inside the predecessor, with
# --begin=now+(T-lead), leaves it pending-and-eligible from its begin time, so its
# queue wait runs concurrently with the predecessor's compute.
#
# The successor is submitted BEFORE training starts, not after, so a job that is
# killed at walltime -- or that dies -- has already queued its replacement. The same
# structure is already in production in this tree at
# slurm-scanner-main/collector/slurm_probe_loop.sh.
#
# STOPPING
# --------
#   touch logs/chain-<lane>.stop      # this link finishes, nothing more is submitted
# The sentinel is checked before every resubmission. Remove it to resume chaining.
#
# WHAT IT DELIBERATELY DOES NOT PROPAGATE
# ---------------------------------------
#   PWW_FRESH_RUN / PWW_FRESH_DELETE   they belong to the FIRST link only. Copied
#                                      onto a successor they delete the lane's
#                                      checkpoint, which is the one thing a lane
#                                      exists to keep: every later link resumes the
#                                      lane's own model, AdamW moments, LR schedule
#                                      and dataloader, and therefore pays no
#                                      cold-join transient.
#   WANDB_RUN_NAME                     left unset so run_train.sh appends the Slurm
#                                      job id and the links are distinguishable.
#   SLURM_*                            sbatch reads SLURM_* environment variables as
#                                      if they were command-line options, so a job's
#                                      own geometry would leak into the successor's
#                                      request. Unset for the sbatch call only.
set -euo pipefail

# NOT from BASH_SOURCE. Slurm stages the batch script as
# /var/spool/slurmd/job<id>/slurm_script, so dirname/../.. resolves to /var/spool and
# every relative path below breaks -- the link would exit 2 without training AND
# without submitting its successor, silently ending the lane. Same idiom as
# scripts/{snellius,lumi}/job_titan_diloco.sh: the submit directory is what Slurm
# guarantees, and BASH_SOURCE is only a fallback for a direct (non-Slurm) invocation.
: "${PWW_ROOT:=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT} (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset});" \
         "submit from the checkout or export PWW_ROOT" >&2
    exit 2
fi
export PWW_ROOT
cd "${PWW_ROOT}"

LINKS="${PWW_CHAIN_LINKS:-1}"
LANE="${PWW_CHAIN_LANE:-lane}"
SCRIPT="${PWW_CHAIN_SCRIPT:-}"
LEAD_S="${PWW_CHAIN_LEAD_S:-0}"
STOP="${PWW_CHAIN_STOP:-${PWW_ROOT}/logs/chain-${LANE}.stop}"

[[ -n "${SCRIPT}" ]] || { echo "PWW_CHAIN_SCRIPT is required" >&2; exit 2; }
[[ -x "${SCRIPT}" || -r "${SCRIPT}" ]] || { echo "no such job script: ${SCRIPT}" >&2; exit 2; }

# The sbatch flags for the successor, '|'-separated. Not space-separated: --export
# values with spaces survive the shell only by luck and not every Slurm build
# round-trips them, so the planner joins the verbatim probe args on '|' and they are
# split back here. IFS is scoped to the read.
IFS='|' read -r -a CHAIN_ARGS <<< "${PWW_CHAIN_ARGS:-}"

resubmit() {
    local remaining=$(( LINKS - 1 ))
    [[ "${remaining}" -ge 1 ]] || { echo "chain ${LANE}: last link, nothing to submit"; return 0; }
    if [[ -e "${STOP}" ]]; then
        echo "chain ${LANE}: ${STOP} exists, stopping the chain here"
        return 0
    fi
    [[ "${#CHAIN_ARGS[@]}" -gt 0 ]] || { echo "chain ${LANE}: no PWW_CHAIN_ARGS, cannot resubmit" >&2; return 0; }

    # When to start the successor. Read out of the -t we were submitted with rather
    # than from Slurm, so this works identically on both sites and in a dry run: the
    # link is T long and we are at its very beginning.
    local walltime="" i
    for (( i = 0; i < ${#CHAIN_ARGS[@]}; i++ )); do
        [[ "${CHAIN_ARGS[i]}" == "-t" || "${CHAIN_ARGS[i]}" == "--time" ]] && walltime="${CHAIN_ARGS[i+1]}"
    done
    local secs
    secs="$(awk -v t="${walltime}" 'BEGIN{
        n = split(t, d, "-"); rest = (n == 2 ? d[2] : d[1]); days = (n == 2 ? d[1] : 0)
        m = split(rest, p, ":")
        if (m == 3)      s = p[1]*3600 + p[2]*60 + p[3]
        else if (m == 2) s = (days ? p[1]*3600 + p[2]*60 : p[1]*60 + p[2])
        else             s = (days ? p[1]*3600 : p[1]*60)
        print s + days*86400 }')"
    [[ "${secs}" -gt 0 ]] 2>/dev/null || { echo "chain ${LANE}: cannot read -t from PWW_CHAIN_ARGS" >&2; return 0; }
    local begin_min=$(( (secs - LEAD_S + 59) / 60 ))
    [[ "${begin_min}" -ge 1 ]] || begin_min=1

    # ALL, so the successor inherits the submitting environment (PATH, module state,
    # PWW_ROOT) -- but the sbatch call below runs under `env -u` for every variable
    # that must NOT cross, and the two flags that would destroy the lane are ALSO
    # pinned to 0 at the end of this list, after ALL, so the later assignment wins
    # whatever the ambient environment says. Getting this wrong is not a nicety:
    # PWW_FRESH_RUN=1 reaching link 2 makes run_train.sh rm -rf the lane's DCP
    # checkpoint, so every link restarts from a random init with cold AdamW moments
    # and a restarted LR warmup -- i.e. a chained plan trains nothing.
    # The lane's identity (REPLICA, PWW_DUMP) is what makes the successor resume THIS
    # lane's checkpoint rather than another lane's.
    local exports="ALL"
    local var
    for var in DARL_TOKEN CONFIG PWW_DARL_PORT PWW_FLOWER_PORT PWW_GRAD_ACCUM \
               PWW_VAL_WINDOWS REPLICA PWW_DUMP ENABLE_WANDB WANDB_PROJECT \
               WANDB_API_KEY PWW_CHAIN_SCRIPT PWW_CHAIN_LANE PWW_CHAIN_ARGS \
               PWW_CHAIN_LEAD_S PWW_CHAIN_STOP; do
        [[ -n "${!var:-}" ]] && exports="${exports},${var}=${!var}"
    done
    exports="${exports},PWW_CHAIN_LINKS=${remaining}"
    # Explicitly OFF, and last, so it beats anything ALL carried in.
    exports="${exports},PWW_FRESH_RUN=0,PWW_FRESH_DELETE=0"

    echo "chain ${LANE}: submitting link $(( remaining )) more, --begin=now+${begin_min}minutes"
    # The checkout's copy, not "$0": under Slurm "$0" is the node-local spool copy,
    # which is not the file the operator can edit and is not guaranteed to exist on
    # the node that runs the successor.
    local self="${PWW_ROOT}/scripts/titan/job_chain_link.sh"
    [[ -r "${self}" ]] || self="$0"
    env -u SLURM_JOB_ID -u SLURM_JOBID -u SLURM_NNODES -u SLURM_NTASKS \
        -u SLURM_NTASKS_PER_NODE -u SLURM_CPUS_PER_TASK -u SLURM_GPUS_PER_NODE \
        -u SLURM_MEM_PER_CPU -u SLURM_MEM_PER_NODE -u SLURM_JOB_PARTITION \
        -u SLURM_TIMELIMIT -u SLURM_JOB_NAME -u SLURM_EXPORT_ENV \
        -u PWW_FRESH_RUN -u PWW_FRESH_DELETE -u WANDB_RUN_NAME \
        sbatch "${CHAIN_ARGS[@]}" \
               --begin="now+${begin_min}minutes" \
               --export="${exports}" \
               "${self}" || echo "chain ${LANE}: sbatch failed; this link still runs" >&2
}

resubmit
echo "chain ${LANE}: link starting (${LINKS} remaining including this one) -> ${SCRIPT}"
exec "${SCRIPT}"
