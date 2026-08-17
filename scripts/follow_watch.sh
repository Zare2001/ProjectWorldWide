#!/usr/bin/env bash
# Fire a command once, when a named cluster appears in an arm's DARL status.
#
#   nohup scripts/follow_watch.sh --url http://145.38.206.143:29540 --token "$TOK" \
#       --cluster snellius -- sbatch ... scripts/lumi/job_titan_diloco.sh \
#       > watch-follow.log 2>&1 &
#
# The sibling of elastic_watch.sh: that one fires on a committed-block THRESHOLD
# (progress), this one fires on cluster MEMBERSHIP (liveness). Exists for "submit
# site B the moment site A starts": B's login node cannot see A's queue, but the
# coordinator sees A the instant its job registers -- which is also the first
# moment a companion is useful. Registration, not first commit, on purpose: the
# fired sbatch still has B's own queue wait ahead of it, so firing at the
# earliest true signal front-loads that wait.
#
# Liveness, NOT presence. The coordinator's clusters map is a durable membership
# RECORD: a coordinator resumed from its snapshot lists every cluster that ever
# registered, dead or alive. The first version of this script checked presence
# alone and, pointed at a resumed coordinator, fired the instant it was armed --
# submitting the companion while the site it was following was still hours deep
# in its own queue, which is precisely the solo-training window this script
# exists to close. So the trigger is last_seen within --fresh-within seconds
# (default 300; heartbeats arrive at least once a minute while a job lives).
# last_seen is a Unix epoch stamp, so login-node/VM timezones do not enter into
# it -- only ordinary NTP-level clock agreement is assumed.
#
# The command's arguments are expanded by YOUR shell at launch, so tokens and
# keys baked into an --export string are captured then -- launch it from the
# shell where they are exported, and check the log's echo of what it will run.
#
# Same failure contract as elastic_watch.sh: a refused token is fatal (it never
# heals); an unreachable coordinator is retried forever (VM reboots and closed
# firewall ports heal); login nodes occasionally reap long-lived processes, and
# relaunching is safe -- a duplicate sbatch shows up in squeue rather than
# corrupting anything, and the run itself is idempotent to it only if you catch
# it, so scancel the extra job if you ever see two.
set -euo pipefail

URL=""
TOKEN="${DARL_TOKEN:-}"
CLUSTER=""
INTERVAL=120
FRESH_WITHIN=300
while [[ $# -gt 0 ]]; do
    case "$1" in
        --url) URL="$2"; shift 2 ;;
        --token) TOKEN="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --fresh-within) FRESH_WITHIN="$2"; shift 2 ;;
        --) shift; break ;;
        -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (use -- before the command to fire)" >&2; exit 2 ;;
    esac
done
if [[ -z "${URL}" || -z "${CLUSTER}" || $# -eq 0 ]]; then
    sed -n '2,25p' "$0"
    exit 2
fi

echo "$(date +'%F %T') watching ${URL}/status for cluster '${CLUSTER}' live within ${FRESH_WITHIN}s; will run: $*"
while :; do
    body="$(curl -sS -m 20 -H "X-DARL-Token: ${TOKEN}" "${URL}/status" 2>/dev/null)" || body=""
    if [[ -z "${body}" ]]; then
        echo "$(date +'%F %T') coordinator unreachable; retrying in ${INTERVAL}s"
    else
        seen="$(python3 -c 'import json, sys, time
try:
    d = json.load(sys.stdin)
except Exception:
    print(-2)
    raise SystemExit
if "error" in d:
    print(-1)
elif "clusters" not in d:
    print(-2)
else:
    record = d["clusters"].get(sys.argv[1])
    if record is None:
        print(0)
    else:
        # Present is not alive: a resumed coordinator lists dead history too.
        age = time.time() - float(record.get("last_seen", 0))
        print(1 if age <= float(sys.argv[2]) else f"stale:{age:.0f}")' \
            "${CLUSTER}" "${FRESH_WITHIN}" <<<"${body}")"
        if [[ "${seen}" == "-1" ]]; then
            # 401 body, not a network blip: a wrong token never starts working.
            echo "$(date +'%F %T') FATAL: coordinator refused the token: ${body}" >&2
            exit 1
        elif [[ "${seen}" == "-2" ]]; then
            echo "$(date +'%F %T') unparseable /status response; retrying: ${body:0:200}"
        elif [[ "${seen}" == "1" ]]; then
            echo "$(date +'%F %T') cluster '${CLUSTER}' is live: firing"
            rc=0
            "$@" || rc=$?
            echo "$(date +'%F %T') command exited ${rc}"
            exit "${rc}"
        elif [[ "${seen}" == stale:* ]]; then
            echo "$(date +'%F %T') '${CLUSTER}' is on record but last seen ${seen#stale:}s ago (dead or pre-restart history); waiting for a live heartbeat"
        else
            known="$(python3 -c 'import json,sys
d=json.load(sys.stdin); print(",".join(sorted(d.get("clusters",{}))) or "none")' <<<"${body}" 2>/dev/null || echo "?")"
            echo "$(date +'%F %T') '${CLUSTER}' not registered yet (known: ${known})"
        fi
    fi
    sleep "${INTERVAL}"
done
