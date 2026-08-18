#!/usr/bin/env bash
# Read a site's training throughput out of a job log, for round balancing.
#
#   scripts/titan/calibrate_throughput.sh logs/pww-snellius-titan-full-123.out
#   scripts/titan/calibrate_throughput.sh --write logs/...        # update the registry
#
# No benchmark job: torchtitan already logs per-rank tps every log_freq steps, so
# any real run of the current model/geometry has the number in it. Reads the
# median of the post-warmup steps (one GC pause must not move the answer) and
# normalises to accumulation = 1, which is what configs/site_throughput.env
# stores and what run_train.sh's PWW_BALANCE divides.
#
# Re-read a log after the hardware, container, model flavor, seq_len or
# local_batch_size changes.
set -euo pipefail

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WRITE=0
[[ "${1:-}" == "--write" ]] && { WRITE=1; shift; }
LOG="${1:-}"
[[ -r "${LOG}" ]] || { echo "usage: $0 [--write] <job-log>" >&2; exit 2; }

read -r SITE SEQ_PER_S BATCH STEP_TIME RANKS ACCUM <<EOF
$(python3 - "${LOG}" <<'PY'
import re, sys, statistics
raw = open(sys.argv[1], errors="ignore").read()
txt = re.sub(r"\x1b\[[0-9;]*m", "", raw)

m = re.search(r"^site\s+:\s*(\S+)", txt, re.M)
site = m.group(1) if m else "UNKNOWN"

m = re.search(r"local batch size (\d+), global batch size (\d+), "
              r"gradient accumulation steps (\d+), sequence length (\d+)", txt)
if not m:
    print("- 0 0 0 0 0"); raise SystemExit
local, gbs, accum, seq_len = (int(g) for g in m.groups())
ranks = gbs // (local * accum)
batch = local * ranks                      # sequences per step at accumulation 1

# per-rank tps, grouped by step; a step counts only once every rank reported it
steps = {}
for rank, step, tps in re.findall(r"\[rank(\d+)\].*?step:\s*(\d+).*?tps:\s*([\d,]+)", txt):
    steps.setdefault(int(step), {})[int(rank)] = int(tps.replace(",", ""))
full = [sum(v.values()) for s, v in sorted(steps.items()) if len(v) == ranks and s > 10]
if not full:
    print(f"{site} 0 {batch} 0 {ranks} {accum}"); raise SystemExit

tok_s = statistics.median(full)
seq_s = tok_s / seq_len * accum            # normalise to accumulation 1
print(f"{site} {seq_s:.1f} {batch} {batch/seq_s:.3f} {ranks} {accum}")
PY
)
EOF

[[ "${SEQ_PER_S}" != "0" ]] || { echo "ERROR: no usable tps lines in ${LOG}" >&2; exit 1; }
SITE_UC="$(echo "${SITE}" | tr '[:lower:]-' '[:upper:]_')"

cat <<EOF

  site       : ${SITE}  (${RANKS} ranks, log ran at accumulation ${ACCUM})
  throughput : ${SEQ_PER_S} sequences/s, normalised to accumulation 1
  batch/step : ${BATCH} sequences
  step time  : ${STEP_TIME}s   <- this is what gets equalised across sites

  PWW_TPUT_${SITE_UC}=${SEQ_PER_S}
  PWW_BATCH_${SITE_UC}=${BATCH}
EOF

REGISTRY="${PWW_ROOT}/configs/site_throughput.env"
if (( WRITE )); then
    tmp="$(mktemp)"
    grep -vE "^PWW_(TPUT|BATCH)_${SITE_UC}=" "${REGISTRY}" > "${tmp}" 2>/dev/null || true
    printf 'PWW_TPUT_%s=%s\nPWW_BATCH_%s=%s\n' "${SITE_UC}" "${SEQ_PER_S}" "${SITE_UC}" "${BATCH}" >> "${tmp}"
    mv "${tmp}" "${REGISTRY}"
    echo "  updated ${REGISTRY} -- commit it so every site reads the same numbers"
else
    echo "  add those two lines to ${REGISTRY}, or re-run with --write"
fi
