#!/usr/bin/env bash
# Launch a torchtitan + DARL (+ Flower) run. Site- and topology-agnostic.
#
# Runs INSIDE the site environment -- the container on LUMI, the venv on Snellius
# -- which env.sh arranges via PWW_LAUNCH. Called either directly on a login node
# for a CPU-side check, or from scripts/{lumi,snellius}/job_titan_diloco.sh inside
# an allocation.
#
#   scripts/titan/run_train.sh --config configs/titan/qwen3_0.6b_smoke.toml
#
# Running TWO jobs at the same site at the same time: pass --replica a and --replica b.
# Without it both register with the coordinator under the same cluster id and silently
# corrupt each other's leases and deltas. See the --replica block below.
#
#   scripts/titan/run_train.sh \
#       --config configs/titan/qwen3_0.6b_c4_diloco.toml \
#       --shards $PWW_DATA_DIR/c4-tokenizer-128k-2048 \
#       --tokenizer $PWW_DATA_DIR/tokenizers/tokenizer-128k \
#       --central 145.38.206.143
#
# Everything after `--` is passed to torchtitan verbatim, so any upstream flag
# works without this script knowing about it:
#   scripts/titan/run_train.sh --config ... -- --training.steps 50 --metrics.log_freq 1
#
# Why the paths are CLI overrides rather than values in the TOML: TOML does not
# expand environment variables, and the tokenizer/shard/dump paths are exactly the
# things that differ between LUMI's /scratch/project_* and Snellius'
# /scratch-shared. Baking either site's layout into a config is what sites/*.sh
# exists to prevent.
set -euo pipefail

CONFIG=""
SHARDS=""
TOKENIZER=""
CENTRAL="${PWW_CENTRAL_IP:-145.38.206.143}"
DARL_PORT="${PWW_DARL_PORT:-29510}"
FLOWER_PORT="${PWW_FLOWER_PORT:-29511}"
NPROC=""
NNODES="${SLURM_NNODES:-1}"
DUMP=""
SITE_OVERRIDE=""
# Distinguishes concurrent jobs at ONE site. Empty means "the only job here", which
# is the common case; see the block that consumes it below for why it matters.
REPLICA="${REPLICA:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --shards) SHARDS="$2"; shift 2 ;;
        --tokenizer) TOKENIZER="$2"; shift 2 ;;
        --central) CENTRAL="$2"; shift 2 ;;
        --darl-port) DARL_PORT="$2"; shift 2 ;;
        --flower-port) FLOWER_PORT="$2"; shift 2 ;;
        --nproc) NPROC="$2"; shift 2 ;;
        --dump) DUMP="$2"; shift 2 ;;
        --site) SITE_OVERRIDE="$2"; shift 2 ;;
        --replica) REPLICA="$2"; shift 2 ;;
        --wandb) ENABLE_WANDB=1; shift ;;
        --) shift; break ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (use -- to pass flags to torchtitan)" >&2; exit 1 ;;
    esac
done
EXTRA=("$@")

PWW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
source "${PWW_ROOT}/env.sh"
cd "${PWW_ROOT}"

[[ -n "${CONFIG}" ]] || { echo "--config is required" >&2; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "no such config: ${CONFIG}" >&2; exit 1; }

if [[ ! -f third_party/torchtitan/torchtitan/train.py ]]; then
    echo "third_party/torchtitan is empty -- initialise the submodule:" >&2
    echo "  git submodule update --init --recursive" >&2
    exit 1
fi

# torchtitan is a submodule and never pip-installed, so it reaches the interpreter
# via PYTHONPATH. env.sh already put src/ there and mirrored it into the
# SINGULARITYENV_/APPTAINERENV_ forms that survive entering a container.
export PYTHONPATH="${PWW_ROOT}/src:${PWW_ROOT}/third_party/torchtitan${PYTHONPATH:+:${PYTHONPATH}}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"

# One rank per GPU, from the site file rather than assumed -- LUMI exposes 8 GCDs
# per node and Snellius 4 H100s, and getting this wrong either idles hardware or
# oversubscribes it.
NPROC="${NPROC:-${SLURM_GPUS_PER_NODE:-${PWW_GPUS_PER_NODE}}}"
SITE="${SITE_OVERRIDE:-${PWW_SITE}}"

# The torchrun rendezvous port. Two jobs of this repo landing on the same node is
# routine -- both sites allow partial-node allocations, so submitting two 2-GPU jobs
# to one facility can put them on the same host -- and a shared port makes the second
# job's rendezvous fail, or worse, join the first job's.
#
# Multi-node and single-node need different answers, because every node running
# torchrun must agree on the port:
#
#   NNODES > 1   it has to be *derived*, identically on every node, so probing is not
#                an option. Derived from SLURM_JOB_ID, which is unique among
#                concurrent jobs, over a wide range: a collision needs two live jobs
#                whose ids differ by exactly the modulus. This used to be `% 400`,
#                which is only about 1 in 400 per pair of co-resident jobs -- small,
#                but these are multi-hour queue waits to lose.
#   NNODES = 1   ask the kernel for a free port, which cannot collide at all. Safe
#                here because this script runs once per job (the shipped job scripts
#                are --nodes=1 --ntasks-per-node=1 and call it directly, not via srun).
JOB_ID="${SLURM_JOB_ID:-0}"
MASTER_PORT=$(( 29600 + (JOB_ID % 20000) ))
if [[ "${NNODES}" -gt 1 ]]; then
    HEAD_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
    RDZV_ENDPOINT="${HEAD_NODE}:${MASTER_PORT}"
else
    # Bind port 0, read what the kernel picked, release it. There is a small window
    # between releasing and torchrun binding; the derived port is the fallback if the
    # probe itself fails (no python on PATH, a hostile sandbox).
    PROBED=$(python3 -c 'import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()' 2>/dev/null || true)
    [[ -n "${PROBED}" ]] && MASTER_PORT="${PROBED}"
    RDZV_ENDPOINT="localhost:${MASTER_PORT}"
fi
echo "rendezvous  : ${RDZV_ENDPOINT}"

# --- PWW_FRESH_RUN: the site half of a fresh start ---------------------------
#
# Same variable name as on the central node, because forgetting this side leaves a run
# that is half fresh, and the half that is stale is silent.
#
# torchtitan's checkpoint restores four things, and only the first is harmless here:
#
#   MODEL         overwritten anyway by configure_fit before the first step
#   OPTIMIZER     stale AdamW moments, now paired with completely different weights
#   LR_SCHEDULER  resumes at the old global step, so a freshly seeded model skips
#   + TRAIN_STATE warmup_steps entirely and takes near-peak LR on its first step
#   DATALOADER    the DARL client's epoch/phase_index/samples_seen, while the
#                 coordinator is on a fresh epoch with every block unassigned
#
# With checkpoint.interval == darl.inner_steps a checkpoint is written every round, so
# even a job that failed after one round leaves one behind.
#
# Applies to the DiLoCo job and the central baseline alike. The baseline resumes from its
# own dump folder exactly the same way -- torchtitan's checkpointer does not know or care
# whether a Flower server is involved -- so a baseline resubmitted without this silently
# continues the previous one at its old LR position, on a coordinator whose block space
# was just reset to epoch 0. That produces a plausible-looking curve of a run that is
# neither the old one nor a new one.
#
# Renamed, not deleted, keeping one generation -- the same policy the coordinator uses.
# PWW_FRESH_DELETE=1 removes outright instead, for when scratch quota is the constraint;
# scripts/reset_run.sh is the interactive front end for that.
if [[ "${PWW_FRESH_RUN:-0}" == "1" ]]; then
    fresh_dump="${DUMP:-}"
    if [[ -z "${fresh_dump}" ]]; then
        fresh_dump="$(grep -m1 -E '^[[:space:]]*dump_folder[[:space:]]*=' "${CONFIG}" \
                      | cut -d'"' -f2)"
    fi
    if [[ -n "${fresh_dump}" ]]; then
        # dump_folder is relative in the shipped configs, so resolve it the way the job
        # would rather than trusting the current directory.
        [[ "${fresh_dump}" != /* ]] && fresh_dump="${PWW_ROOT}/${fresh_dump#./}"
        fresh_ckpt="${fresh_dump}/checkpoint"
        if [[ -d "${fresh_ckpt}" ]]; then
            rm -rf "${fresh_ckpt}.superseded"
            if [[ "${PWW_FRESH_DELETE:-0}" == "1" ]]; then
                rm -rf "${fresh_ckpt}"
                echo "PWW_FRESH_RUN=1 PWW_FRESH_DELETE=1: deleted ${fresh_ckpt}"
            else
                mv "${fresh_ckpt}" "${fresh_ckpt}.superseded"
                echo "PWW_FRESH_RUN=1: moved ${fresh_ckpt} aside to .superseded"
            fi
        else
            echo "PWW_FRESH_RUN=1: no checkpoint at ${fresh_ckpt}, nothing to clear"
        fi
        # Staged blobs from the previous run. Named by (run, round, cluster), so a new
        # run at round 1 does not read them -- but they are one full model each and they
        # accumulate silently, which is how a site's scratch quota goes at 3 a.m.
        if [[ -d "${fresh_dump}/blob-staging" ]]; then
            staged="$(find "${fresh_dump}/blob-staging" -type f | wc -l)"
            rm -rf "${fresh_dump}/blob-staging"
            echo "PWW_FRESH_RUN=1: removed ${staged} staged blob file(s) from the previous run"
        fi
    else
        echo "WARNING: PWW_FRESH_RUN=1 but no dump_folder found in ${CONFIG}; this site" \
             "may resume a stale optimizer state and LR schedule position" >&2
    fi
fi

overrides=()
[[ -n "${TOKENIZER}" ]] && overrides+=(--model.hf_assets_path "${TOKENIZER}")
[[ -n "${SHARDS}" ]] && overrides+=(--training.dataset_path "${SHARDS}")
[[ -n "${DUMP}" ]] && overrides+=(--job.dump_folder "${DUMP}")

# --- gradient accumulation: let a fast site fill the round instead of idling -----
#
# The Flower round is a barrier: DiLoCo's outer step averages every participant's delta,
# so the round cannot close until the slowest site delivers. That barrier cannot be
# removed without changing the algorithm. What *can* be removed is the idling at it.
#
# A fast site finishes its H steps early and then waits. Measured on a real round: one
# site did 37s of work inside a 154s round and sat still for 118s -- 76% of every round,
# turning hardware sustaining 256k tok/s into a run averaging 101k.
#
# PWW_GRAD_ACCUM=N makes each of that site's H optimiser steps consume N microbatches
# instead of one. It processes N times the data in the same number of steps, so it finishes
# when the slow site does rather than early.
#
# Why this rather than raising H on the fast site, which also fills the round:
#
#   drift      unchanged. `drift = ||local - global|| / ||global||` measures how far the
#              weights moved, and they move once per *optimiser* step -- still H of them.
#              Accumulation improves each step's gradient estimate; it does not take more
#              steps or larger ones. Raising H multiplies drift, and drift was already
#              ~0.93 from a random init, where past 1 averaging destroys progress.
#   LR         unchanged. Identical H at every site means the schedule advances identically,
#              so there is no per-site H and nothing to align.
#   memory     unchanged. Microbatches run sequentially, so only one set of activations is
#              live at a time. This is the entire reason accumulation exists.
#
# Expressed as a multiplier rather than as training.global_batch_size, because the batch
# that gives a 2x accumulation depends on the site's rank count -- 64 on 4 ranks, 128 on 8 --
# so "accumulate 2x" is the site-independent way to say it.
#
# The cost, so it is not a surprise: a site running a larger effective batch at the same
# learning rate is under-using that batch (the linear-scaling rule would want a larger LR).
# That is an optimisation inefficiency, not an instability, and the way to see it is
# loss per *token* rather than per round.
#
# DARL needs no change: blocks_for_phase already takes grad_accum, so the lease grows with
# the phase.
GRAD_ACCUM="${PWW_GRAD_ACCUM:-1}"
if [[ "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]] && (( GRAD_ACCUM > 1 )); then
    ga_batch="$(grep -m1 -E '^[[:space:]]*local_batch_size[[:space:]]*=' "${CONFIG}"                 | cut -d= -f2 | tr -d ' ')"
    if [[ "${ga_batch}" =~ ^[1-9][0-9]*$ ]]; then
        global_batch=$(( GRAD_ACCUM * ga_batch * NPROC ))
        overrides+=(--training.global_batch_size "${global_batch}")
        echo "grad accum  : ${GRAD_ACCUM}x -> global_batch_size ${global_batch}"              "(${ga_batch} local x ${NPROC} ranks x ${GRAD_ACCUM}); H is unchanged, so"              "drift and the LR schedule are unaffected"
    else
        echo "WARNING: PWW_GRAD_ACCUM=${GRAD_ACCUM} but local_batch_size could not be read"              "from ${CONFIG}; leaving gradient accumulation at 1" >&2
    fi
elif [[ ! "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARNING: PWW_GRAD_ACCUM=${GRAD_ACCUM} is not a positive integer; ignoring" >&2
fi

# --- validation, made identical across sites AND actually held out ----------
#
# The held-out loss is the only cross-site check there is: every cluster evaluates the
# *same* global weights, so a disagreement is a bug rather than variance. That only holds
# if the sites score the same data, and two things stopped them.
#
#   dataset_path  relative in the TOML ("./third_party/..."), so it resolved only because
#                 both jobs happened to run with the repo as their working directory. A
#                 different CWD, or a container without the submodule visible, and
#                 validation reads nothing or reads something else.
#
#   steps         total windows scored = steps x local_batch_size x nproc, so a fixed
#                 `steps` scales with the rank count. At 20 steps and batch 8 that is 640
#                 windows on 4 ranks and 1,280 on 8 -- against a fixture of ~543 windows,
#                 so the sites re-looped it 1.2x and 2.4x and weighted the partial tail
#                 differently. Observed as 1,310,720 vs 2,621,440 eval tokens.
#
# Fixing the window TOTAL instead of the step count makes coverage identical: the loader
# shards round-robin and the dataset iterator is stateful, so every site walks the same
# window sequence in lockstep -- eval N scores the same union of windows at every site
# for any rank count. Same union, same mean, comparable number.
#
# WHICH data: the bundled c4_test fixture is NOT held out from a real C4 run. Its first
# document ("Beginners BBQ Class Taking Place in Missoula!") is the first document of
# en/c4-train.00000 -- the head of the very first file tokenize_c4.sh consumes -- so the
# fixture's windows are also training windows. Worse than a uniformly optimistic eval:
# the chance an eval window has been *trained on* grows with how much of the corpus a run
# consumes, so a fixture eval favours whichever arm of a central-vs-DiLoCo comparison
# trained more data. A staged copy of C4's real validation split (disjoint from train by
# construction) is therefore preferred whenever it is present, via the offline c4_local
# loader. Stage it once per site, login node -- both sites must stage the SAME file(s):
#
#   scripts/titan/stage_c4.sh --split validation --files 1 --out $PWW_DATA_DIR/c4-validation
#
# The fixture remains the fallback so smoke tests and standalone runs work with nothing
# staged -- but it warns, because a comparison read off it is quietly biased.
VAL_DATA="${PWW_VAL_DATA:-}"
VAL_DATASET="${PWW_VAL_DATASET:-}"
if [[ -z "${VAL_DATA}" && -d "${PWW_DATA_DIR:-/nonexistent}/c4-validation" ]]; then
    VAL_DATA="${PWW_DATA_DIR}/c4-validation"
fi
if [[ -z "${VAL_DATA}" ]]; then
    VAL_DATA="${PWW_ROOT}/third_party/torchtitan/tests/assets/c4_test"
fi
if [[ -z "${VAL_DATASET}" ]] && compgen -G "${VAL_DATA}/c4-validation.*.json*" > /dev/null; then
    VAL_DATASET="c4_local"
fi
if [[ -d "${VAL_DATA}" ]]; then
    overrides+=(--validation.dataset_path "${VAL_DATA}")
    if [[ -n "${VAL_DATASET}" ]]; then
        overrides+=(--validation.dataset "${VAL_DATASET}")
        echo "validation  : ${VAL_DATASET} at ${VAL_DATA} (held-out C4 validation split)"
    else
        echo "WARNING: evaluating on ${VAL_DATA}, which for the bundled c4_test fixture" \
             "OVERLAPS the C4 training files -- eval/loss will favour whichever run" \
             "trained more of the corpus. For a real comparison, stage the held-out" \
             "split once per site (login node):" >&2
        echo "  scripts/titan/stage_c4.sh --split validation --files 1 --out" \
             "\$PWW_DATA_DIR/c4-validation" >&2
    fi
else
    echo "WARNING: no validation data at ${VAL_DATA}; leaving validation.dataset_path as" \
         "configured, which is relative and depends on the working directory" >&2
fi

VAL_WINDOWS="${PWW_VAL_WINDOWS:-512}"
if [[ "${VAL_WINDOWS}" != "0" ]]; then
    val_batch="$(grep -m1 -E '^[[:space:]]*local_batch_size[[:space:]]*=' "${CONFIG}" \
                 | cut -d= -f2 | tr -d ' ')"
    if [[ "${val_batch}" =~ ^[1-9][0-9]*$ ]]; then
        per_step=$(( val_batch * NPROC ))
        val_steps=$(( VAL_WINDOWS / per_step ))
        (( val_steps < 1 )) && val_steps=1
        if (( val_steps * per_step != VAL_WINDOWS )); then
            echo "NOTE: PWW_VAL_WINDOWS=${VAL_WINDOWS} is not a multiple of ${per_step}" \
                 "(local_batch_size ${val_batch} x ${NPROC} ranks), so this site will" \
                 "score $(( val_steps * per_step )) windows. Sites whose rank counts give" \
                 "a different remainder will not be comparable -- choose a multiple of" \
                 "every site's local_batch_size x nproc." >&2
        fi
        overrides+=(--validation.steps "${val_steps}")
        echo "validation  : ${val_steps} steps x ${val_batch} x ${NPROC} ranks =" \
             "$(( val_steps * per_step )) windows (PWW_VAL_WINDOWS=${VAL_WINDOWS})"
    else
        echo "WARNING: could not read local_batch_size from ${CONFIG}; leaving" \
             "validation.steps as configured. The held-out loss will not be comparable" \
             "between sites with different rank counts." >&2
    fi
fi

# The federation endpoints. Only added when the config actually asks for DARL, so
# a plain single-site torchtitan run through this launcher needs no coordinator.
if grep -qE '^[[:space:]]*dataset[[:space:]]*=[[:space:]]*"pww_tokens"' "${CONFIG}"; then
    overrides+=(--darl.url "http://${CENTRAL}:${DARL_PORT}" --darl.site "${SITE}")
    # --replica is REQUIRED when you run more than one job at the same site at the
    # same time, and it is not a performance knob -- it is a correctness one.
    #
    # The DARL cluster id defaults to the site name alone ("lumi"), deliberately, so
    # that a job killed at walltime and requeued is recognised as the same cluster and
    # keeps its measured throughput -- which is what sizes its grants. The cost is
    # that two *concurrent* jobs at one site both call themselves "lumi", and the
    # coordinator cannot tell them apart. Two things then go wrong silently:
    #
    #   * /release with no lease id releases every lease held by that cluster id, so
    #     the first job to finish hands back the second job's live leases. The second
    #     keeps training blocks that are back in the free pool -- duplicate work, and
    #     no assertion anywhere catches it.
    #   * the delta blob is named by (run, round, cluster), so both jobs upload to the
    #     same object each round and one silently overwrites the other. A whole round
    #     of one job's work disappears.
    #
    # Passing --replica a makes the id "lumi-a", which is unique per job and still
    # stable across that job's own requeues.
    [[ -n "${REPLICA}" ]] && overrides+=(--darl.cluster_id "${SITE}-${REPLICA}")
fi
if grep -qE '^[[:space:]]*enable[[:space:]]*=[[:space:]]*true' <(sed -n '/^\[flower\]/,/^\[/p' "${CONFIG}"); then
    overrides+=(--flower.server_address "${CENTRAL}:${FLOWER_PORT}")
fi

# --- WandB ------------------------------------------------------------------
#
# One block, and it used to be two: an earlier one appended --metrics.enable_wandb and
# mirrored the environment into the container forms *before* WANDB_PROJECT had been
# defaulted, so the flag went on the command line twice and the container inherited an
# empty project on any run that did not set one itself.
#
# The run name is derived rather than required, because the names are what makes a chart
# readable and "run-20260812_154653-s7qqxpfs" is not:
#
#     central-<site>    the single-node baseline (flower.enable = false)
#     diloco-<site>     one participating cluster of the federated run
#     central-aggregator  the FedMom server -- set by start_central_services.sh
#
# The aggregator is the run to compare a baseline against, not either site: its
# train/loss is the token-weighted loss across all participants and its train/cum_tokens
# is the federation total, whereas a site reports only its own share.
if [[ "${ENABLE_WANDB:-0}" == "1" ]] || [[ "${PWW_WANDB:-0}" == "1" ]] || [[ -n "${WANDB_PROJECT:-}" ]]; then
    overrides+=(--metrics.enable_wandb)
    export WANDB_PROJECT="${WANDB_PROJECT:-pww-diloco-1k}"
    if [[ -z "${WANDB_RUN_NAME:-}" ]]; then
        if grep -qE '^[[:space:]]*enable[[:space:]]*=[[:space:]]*true' <(sed -n '/^\[flower\]/,/^\[/p' "${CONFIG}"); then
            export WANDB_RUN_NAME="diloco-${SITE}${REPLICA:+-${REPLICA}}"
        else
            export WANDB_RUN_NAME="central-${SITE}${REPLICA:+-${REPLICA}}"
        fi
    fi
    # torchtitan's WandBLogger reads the entity from WANDB_TEAM; pww.central.server
    # reads WANDB_ENTITY. Setting only one of them therefore puts the sites and the
    # aggregator in *different* entities, where they cannot appear on one chart at all.
    if [[ -n "${WANDB_ENTITY:-}" && -z "${WANDB_TEAM:-}" ]]; then
        export WANDB_TEAM="${WANDB_ENTITY}"
    elif [[ -n "${WANDB_TEAM:-}" && -z "${WANDB_ENTITY:-}" ]]; then
        export WANDB_ENTITY="${WANDB_TEAM}"
    fi
    echo "wandb       : ${WANDB_PROJECT} / ${WANDB_RUN_NAME}${WANDB_ENTITY:+ (${WANDB_ENTITY})}"
    # Mirrored into both container prefixes, because env.sh's PWW_LAUNCH may be a
    # singularity/apptainer exec and a plain export does not survive it.
    for var in WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY WANDB_RUN_NAME WANDB_MODE \
               WANDB_BASE_URL WANDB_TEAM; do
        if [[ -n "${!var:-}" ]]; then
            export "SINGULARITYENV_${var}=${!var}"
            export "APPTAINERENV_${var}=${!var}"
        fi
    done
fi

echo "=============================================================="
pww_summary
cat <<EOF
config      : ${CONFIG}
ranks       : ${NPROC} x ${NNODES} node(s)
rendezvous  : ${RDZV_ENDPOINT}
tokenizer   : ${TOKENIZER:-<from config>}
shards      : ${SHARDS:-<from config>}
central     : ${CENTRAL} (darl ${DARL_PORT}, flower ${FLOWER_PORT})
overrides   : ${overrides[*]:-<none>} ${EXTRA[*]:-}
EOF
echo "=============================================================="

# `python3 -m torch.distributed.run` rather than the torchrun entrypoint: inside
# the LUMI container the console script is not always on PATH, and the module form
# always is.
pww_run python3 -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NPROC}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${RDZV_ENDPOINT}" \
    --role rank --tee 3 \
    -m pww.titan.train \
    --job.config-file "${CONFIG}" \
    "${overrides[@]}" \
    ${EXTRA[@]+"${EXTRA[@]}"}
