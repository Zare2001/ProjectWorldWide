#!/usr/bin/env bash
# Measure this site's training throughput, so the federation can balance a round.
#
#   scripts/titan/calibrate_throughput.sh                    # measure and print
#   scripts/titan/calibrate_throughput.sh --write            # also update configs/site_throughput.env
#   scripts/titan/calibrate_throughput.sh --steps 60 --config configs/titan/qwen3_0.6b_c4_diloco.toml
#
# Runs a short single-site training job -- no Flower, no DARL coordinator, no
# central node -- and reports SEQUENCES PER SECOND for the whole site at its
# normal geometry. That number is what `run_train.sh --balance` divides into a
# target round duration to pick each site's gradient accumulation, so that a
# fast site fills the barrier instead of idling at it. See
# configs/site_throughput.env for why this matters and what it is worth.
#
# Measured at accumulation = 1 on purpose: it is a property of the hardware and
# the model, not of the balancing decision that will later be derived from it.
#
# The first steps of any run are unrepresentative -- compile, autotune, cache
# warm-up, and torchtitan's own first-step MFU is typically a third of steady
# state -- so the first --warmup steps are discarded and only the remainder is
# timed.
set -euo pipefail

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${PWW_ROOT}/env.sh"

CONFIG="${PWW_ROOT}/configs/titan/qwen3_0.6b_c4_diloco.toml"
STEPS=60
WARMUP=20
WRITE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --steps)  STEPS="$2";  shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --write)  WRITE=1;     shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
(( STEPS > WARMUP )) || { echo "--steps must exceed --warmup" >&2; exit 2; }

REGISTRY="${PWW_ROOT}/configs/site_throughput.env"
SITE_UC="$(echo "${PWW_SITE}" | tr '[:lower:]-' '[:upper:]_')"
LOG="${PWW_OUTPUT_DIR}/calibrate-${PWW_SITE}-$$.log"
mkdir -p "$(dirname "${LOG}")"

echo "calibrating ${PWW_SITE}: ${STEPS} steps (first ${WARMUP} discarded), config $(basename "${CONFIG}")"

# flower.enable=false and the stock dataset: this must not touch a coordinator,
# a token or another site. It is a hardware measurement, not a run.
"${PWW_ROOT}/scripts/titan/run_train.sh" \
    --config "${CONFIG}" -- \
    --training.steps "${STEPS}" \
    --flower.enable false \
    --checkpoint.enable false \
    --validation.enable false \
    --metrics.log_freq 1 \
    > "${LOG}" 2>&1 || { echo "calibration run failed; see ${LOG}" >&2; exit 1; }

# torchtitan prints per-rank tps every log_freq steps. Sum over ranks for the
# site figure, and take the median of the post-warmup steps rather than the mean
# so one slow step (a GC pause, a filesystem hiccup) cannot move the answer.
read -r SEQ_PER_S NPROC_SEEN <<EOF
$(python3 - "${LOG}" "${WARMUP}" <<'PY'
import re, sys, statistics
log, warmup = sys.argv[1], int(sys.argv[2])
per_step = {}
ranks = set()
pat = re.compile(r"\[rank(\d+)\].*?step:\s*(\d+).*?tps:\s*([\d,]+)")
for line in open(log, errors="ignore"):
    m = pat.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
    if not m:
        continue
    rank, step, tps = int(m.group(1)), int(m.group(2)), int(m.group(3).replace(",", ""))
    if step <= warmup:
        continue
    ranks.add(rank)
    per_step.setdefault(step, {})[rank] = tps
# a step counts only once every rank has reported it
full = [sum(v.values()) for v in per_step.values() if len(v) == len(ranks)]
if not full:
    print("0 0"); raise SystemExit
seq_len = 2048
print(f"{statistics.median(full) / seq_len:.1f} {len(ranks)}")
PY
)
EOF

if [[ "${SEQ_PER_S}" == "0" || -z "${SEQ_PER_S}" ]]; then
    echo "ERROR: no usable throughput lines in ${LOG} (did the run reach step ${WARMUP}?)" >&2
    exit 1
fi

# Both numbers are needed, not just throughput: what the balancer equalises is
# the time of one optimiser step, and sites run different batches per step. See
# the header of configs/site_throughput.env.
LOCAL_BATCH="$(grep -m1 -E '^[[:space:]]*local_batch_size[[:space:]]*=' "${CONFIG}" | cut -d= -f2 | tr -d ' ')"
BATCH_PER_STEP=$(( NPROC_SEEN * LOCAL_BATCH ))
STEP_TIME="$(awk -v b="${BATCH_PER_STEP}" -v t="${SEQ_PER_S}" 'BEGIN{printf "%.3f", b/t}')"

echo
echo "  site        : ${PWW_SITE}  (${NPROC_SEEN} ranks x local_batch ${LOCAL_BATCH})"
echo "  throughput  : ${SEQ_PER_S} sequences/s at accumulation 1"
echo "  batch/step  : ${BATCH_PER_STEP} sequences"
echo "  step time   : ${STEP_TIME}s  <- this is what gets equalised across sites"
echo
echo "  PWW_TPUT_${SITE_UC}=${SEQ_PER_S}"
echo "  PWW_BATCH_${SITE_UC}=${BATCH_PER_STEP}"
echo

if (( WRITE )); then
    tmp="$(mktemp)"
    grep -vE "^PWW_(TPUT|BATCH)_${SITE_UC}=" "${REGISTRY}" > "${tmp}" 2>/dev/null || true
    printf 'PWW_TPUT_%s=%s\nPWW_BATCH_%s=%s\n' \
        "${SITE_UC}" "${SEQ_PER_S}" "${SITE_UC}" "${BATCH_PER_STEP}" >> "${tmp}"
    mv "${tmp}" "${REGISTRY}"
    echo "updated ${REGISTRY} -- commit it so every site sees the same numbers"
else
    echo "add those two lines to ${REGISTRY} (or re-run with --write), then commit:"
    echo "every site reads the SAME file to work out who is the slow one."
fi
rm -f "${LOG}"
