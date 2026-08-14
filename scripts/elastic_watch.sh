#!/usr/bin/env bash
# Fire a command once, when an arm's DARL coordinator crosses a committed-block count.
#
#   nohup scripts/elastic_watch.sh --url http://145.38.206.143:29520 --token "$TOK" \
#       --at-committed 470 -- scancel -n pww-lumi-titan-churn > watch-leave.log 2>&1 &
#
# Exists so the three timed events of the elasticity arms (RUNBOOK Part 3c) need no
# human watching a chart: committed blocks are the one progress counter readable from
# any login node with curl alone. Blocks-per-step changes with membership, but each
# event sits in a regime where the conversion is fixed:
#
#   event                     regime while waiting     blocks/step   threshold ~ step
#   churn leave   ~5,000      both sites live          0.09375       470  (~5,000)
#   churn rejoin  ~14,500     snellius solo after 5k   0.03125       766  (~14,500)
#   latejoin join ~9,500      snellius solo from 0     0.03125       297  (~9,500)
#
# (32 windows per step solo, 96 with both sites; 1,024 windows per block. The join
# thresholds sit a few hundred steps early on purpose: the queue wait after the
# sbatch fires is part of the join time.)
#
# The command runs once, in this shell's environment -- so launch it from the shell
# where DARL_TOKEN / WANDB_API_KEY are exported if the command is an sbatch with
# --export=ALL. A refused token is fatal (it never heals); an unreachable coordinator
# is retried forever (VM reboots heal). Login nodes occasionally reap long-lived
# processes: if the log stops, relaunch -- firing is idempotent for scancel, and a
# duplicate sbatch shows up in squeue rather than corrupting anything.
set -euo pipefail

URL=""
TOKEN="${DARL_TOKEN:-}"
AT=""
INTERVAL=120
while [[ $# -gt 0 ]]; do
    case "$1" in
        --url) URL="$2"; shift 2 ;;
        --token) TOKEN="$2"; shift 2 ;;
        --at-committed) AT="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --) shift; break ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (use -- before the command to fire)" >&2; exit 2 ;;
    esac
done
if [[ -z "${URL}" || -z "${AT}" || $# -eq 0 ]]; then
    sed -n '2,27p' "$0"
    exit 2
fi
if [[ ! "${AT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--at-committed must be a positive integer, got: ${AT}" >&2
    exit 2
fi

echo "$(date +'%F %T') watching ${URL}/status for committed >= ${AT}; will run: $*"
while :; do
    body="$(curl -sS -m 20 -H "X-DARL-Token: ${TOKEN}" "${URL}/status" 2>/dev/null)" || body=""
    if [[ -z "${body}" ]]; then
        echo "$(date +'%F %T') coordinator unreachable; retrying in ${INTERVAL}s"
    else
        committed="$(python3 -c 'import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print(-2)
    raise SystemExit
print(d["committed"] if "committed" in d else (-1 if "error" in d else -2))' <<<"${body}")"
        if [[ "${committed}" == "-1" ]]; then
            # 401 body, not a network blip: a wrong token never starts working.
            echo "$(date +'%F %T') FATAL: coordinator refused the token: ${body}" >&2
            exit 1
        elif [[ "${committed}" == "-2" ]]; then
            echo "$(date +'%F %T') unparseable /status response; retrying: ${body:0:200}"
        elif (( committed >= AT )); then
            echo "$(date +'%F %T') committed=${committed} >= ${AT}: firing"
            rc=0
            "$@" || rc=$?
            echo "$(date +'%F %T') command exited ${rc}"
            exit "${rc}"
        else
            echo "$(date +'%F %T') committed=${committed} / ${AT}"
        fi
    fi
    sleep "${INTERVAL}"
done
