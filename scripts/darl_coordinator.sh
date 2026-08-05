#!/bin/bash
# The DARL lease coordinator, on a login/edge node. NOT a batch job.
#
#   ./scripts/darl_coordinator.sh start --num-samples 1000000 --block-size 10000
#   ./scripts/darl_coordinator.sh status
#   ./scripts/darl_coordinator.sh url          # what to give the other site
#   ./scripts/darl_coordinator.sh stop
#
# It has to outlive every job that leases from it, which is exactly why it does
# not run under Slurm: a coordinator inside an allocation dies at walltime and
# takes the epoch's bookkeeping with it. It costs no GPU, a few MB of RAM and
# essentially no CPU -- a handful of requests per minute even with dozens of
# clusters, because a lease covers a whole local phase.
#
# Login nodes get rebooted and processes get reaped, so the state directory is not
# optional: on restart the coordinator replays its snapshot plus journal and
# extends held leases by --restore-grace, and clusters mid-phase never notice.
#
# Reachability, which is the one thing to check before trusting this across sites:
#
#   * within a site  compute nodes reach their own login nodes directly. Nothing
#                    to do.
#   * across sites   compute nodes often reach the outside world only through a
#                    slow proxy, and inbound connections to a login node may be
#                    firewalled. Test it before submitting anything:
#                        curl -sS http://<this host>:<port>/health
#                    from a compute node at the *other* site. If that fails, an
#                    SSH tunnel from the remote login node is the usual fix, and
#                    is also how you get confidentiality:
#                        ssh -N -L 8760:<coord host>:8760 <this site>
#                    Latency does not matter here: leases are per-macro-step.

set -euo pipefail

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ ! -r "${PWW_ROOT}/env.sh" ]]; then
    echo "ERROR: no env.sh under PWW_ROOT=${PWW_ROOT}" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${PWW_ROOT}/env.sh"

DARL_PORT="${DARL_PORT:-8760}"
# Deliberately not on a purged scratch: the lease table is the only record of what
# a month-long run has already trained on. PWW_OUTPUT_DIR is where run state lives
# on both sites; point it at /projects for anything you intend to keep.
DARL_STATE_DIR="${DARL_STATE_DIR:-${PWW_OUTPUT_DIR}/darl}"
DARL_PID_FILE="${DARL_STATE_DIR}/coordinator.pid"
DARL_LOG_FILE="${DARL_STATE_DIR}/coordinator.log"
DARL_TOKEN_FILE="${DARL_STATE_DIR}/token"

command="${1:-status}"
shift || true

mkdir -p "${DARL_STATE_DIR}"

darl_host() {
    echo "${DARL_HOST:-$(hostname -f 2>/dev/null || hostname)}"
}

darl_url() {
    echo "http://$(darl_host):${DARL_PORT}"
}

darl_pid() {
    [[ -r "${DARL_PID_FILE}" ]] || return 1
    local pid
    pid="$(cat "${DARL_PID_FILE}")"
    kill -0 "${pid}" 2>/dev/null || return 1
    echo "${pid}"
}

case "${command}" in
start)
    if pid="$(darl_pid)"; then
        echo "already running: pid ${pid} at $(darl_url)"
        exit 0
    fi
    if [[ ! -s "${DARL_TOKEN_FILE}" ]]; then
        # A shared secret, not confidentiality: it stops an unrelated job -- or a
        # stale client from a previous experiment -- from committing blocks in this
        # run. Readable by you only.
        (umask 077; head -c 24 /dev/urandom | base64 | tr -d '/+=' > "${DARL_TOKEN_FILE}")
        echo "generated token -> ${DARL_TOKEN_FILE}"
    fi
    export DARL_TOKEN="${DARL_TOKEN-$(cat "${DARL_TOKEN_FILE}")}"

    # nohup needs a binary, and pww_run is a shell function, so expand the site's
    # launch prefix here instead. It is empty on Snellius (native) and a
    # `singularity exec` on LUMI.
    nohup ${PWW_LAUNCH[@]+"${PWW_LAUNCH[@]}"} python3 -m pww.darl.server \
        --port "${DARL_PORT}" \
        --state-dir "${DARL_STATE_DIR}" \
        --token "${DARL_TOKEN}" \
        "$@" >> "${DARL_LOG_FILE}" 2>&1 &
    echo $! > "${DARL_PID_FILE}"
    sleep 2
    if ! pid="$(darl_pid)"; then
        echo "ERROR: coordinator exited immediately. Last lines of ${DARL_LOG_FILE}:" >&2
        tail -n 20 "${DARL_LOG_FILE}" >&2
        exit 1
    fi
    cat <<EOF
started  pid ${pid}
url      $(darl_url)
token    ${DARL_STATE_DIR}/token
state    ${DARL_STATE_DIR}
log      ${DARL_LOG_FILE}

On every participating site, before submitting:
    export PWW_DARL_URL=$(darl_url)
    export DARL_TOKEN=\$(cat ${DARL_TOKEN_FILE})    # copy the value across sites
EOF
    ;;

stop)
    if ! pid="$(darl_pid)"; then
        echo "not running"
        exit 0
    fi
    # SIGTERM, not SIGKILL: the handler snapshots before exiting, so the epoch's
    # committed set survives.
    kill -TERM "${pid}"
    for _ in $(seq 20); do
        sleep 0.5
        darl_pid >/dev/null || break
    done
    if darl_pid >/dev/null; then
        echo "WARNING: pid ${pid} did not exit; leaving it alone rather than SIGKILL, "\
             "which would skip the final snapshot" >&2
        exit 1
    fi
    rm -f "${DARL_PID_FILE}"
    echo "stopped. state remains in ${DARL_STATE_DIR} -- start again to resume the epoch"
    ;;

status)
    if pid="$(darl_pid)"; then
        echo "running: pid ${pid} at $(darl_url)"
    else
        echo "not running (state in ${DARL_STATE_DIR})"
    fi
    if [[ -s "${DARL_TOKEN_FILE}" ]]; then
        export DARL_TOKEN="$(cat "${DARL_TOKEN_FILE}")"
        pww_run python3 -m pww.darl.client status --url "$(darl_url)" || true
    fi
    ;;

url)
    darl_url
    ;;

*)
    echo "usage: $0 {start|stop|status|url} [server args...]" >&2
    echo "  extra args are passed to pww.darl.server (--num-samples is required" >&2
    echo "  on a first start; see python3 -m pww.darl.server --help)" >&2
    exit 2
    ;;
esac
