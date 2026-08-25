#!/bin/bash
#
# Run one collection on Frontier, driven by cron.
#
#   scp slurm_probe.py slurm_probe_cron_frontier.sh frontier:~/slurm_probe/
#   ssh frontier && crontab -e
#
#     */15 * * * *  $HOME/slurm_probe/slurm_probe_cron_frontier.sh probe
#     */15 * * * *  $HOME/slurm_probe/slurm_probe_cron_frontier.sh usage
#
#   touch ~/slurm_probe/.stop         pause both, without editing the crontab
#   tail -f ~/slurm_probe/slurm_probe.log
#
# Cron, not the self-resubmitting sbatch chain the other sites use. The chain
# runs the collector inside a job, and a Frontier compute node has no route out
# except the OLCF forward proxy -- so the measurement would depend on the one
# path this project already knows is the fragile one. A login node reaches the
# server directly. Cron also fails better here: a missed tick ends nothing,
# because each run recomputes its own window from scratch, whereas one bad
# cycle silently ends a chain.
#
# Frontier crontabs are per-login-node: the entry runs only on the host you
# installed it on. Note which one that is -- `hostname` is logged every run --
# and reinstall it there after a maintenance reboot moves you elsewhere.

set -u

HERE="$HOME/slurm_probe"                 # the one path to edit
LOG="$HERE/slurm_probe.log"
MAX_LOG_BYTES=$((8 * 1024 * 1024))

# Frontier login nodes reach the internet directly; it is the compute nodes that
# OLCF closes off. Set this only if the post starts failing from wherever this
# ends up running -- the collector posts with urllib, which reads $http_proxy on
# its own, so exporting it here is the whole change.
PROXY=""                                 # e.g. http://proxy.ccs.ornl.gov:3128

# cron gives you /usr/bin:/bin and nothing else -- no profile, no modules. Slurm
# lives in /usr/bin on Frontier, so that is usually enough; set SLURM_BIN if a
# maintenance window ever moves it. Appended, not prepended, so running this by
# hand still uses the python3 the shell would have picked. The check further
# down is what turns "sbatch not found" into a line in the log rather than an
# empty payload posted as if it meant something.
SLURM_BIN=""                             # e.g. /opt/slurm/current/bin
export PATH="${SLURM_BIN:+$SLURM_BIN:}$PATH:/usr/bin:/bin"

usage() { echo "usage: $(basename "$0") {probe|usage}" >&2; exit 2; }

case "${1-}" in
    probe|usage) COMMAND="$1" ;;
    *) usage ;;
esac

cd "$HERE" || exit 1

# Rotate before writing, so a cron line left running for a year does not fill a
# home directory that Frontier quotas tightly.
if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt "$MAX_LOG_BYTES" ]; then
    mv -f "$LOG" "$LOG.1"
fi
exec >> "$LOG" 2>&1

echo "=== $(date '+%F %T')  $COMMAND on $(hostname -s)"

[ -e "$HERE/.stop" ] && { echo "stopping: .stop is present"; exit 0; }

# One lock per subcommand: `usage` is a 48-hour sacct scan of the busiest queue
# in the fleet and can outrun its own cron slot, while `probe` on the same tick
# must still go through. The sbatch chain never needed this -- a job cannot
# overlap itself, and cron happily starts a second copy.
LOCK="$HERE/.lock.$COMMAND"
if ! mkdir "$LOCK" 2>/dev/null; then
    STALE=$(cat "$LOCK/pid" 2>/dev/null)
    if [ -n "$STALE" ] && kill -0 "$STALE" 2>/dev/null; then
        echo "skipped: pid $STALE still running from an earlier tick"
        exit 0
    fi
    echo "clearing a lock left by pid ${STALE:-unknown}, which is gone"
    rm -rf "$LOCK"
    mkdir "$LOCK" 2>/dev/null || { echo "could not take $LOCK"; exit 1; }
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT

command -v sbatch >/dev/null || { echo "no sbatch on PATH ($PATH)"; exit 1; }
command -v python3 >/dev/null || { echo "no python3 on PATH ($PATH)"; exit 1; }

if [ -n "$PROXY" ]; then
    export http_proxy="$PROXY" https_proxy="$PROXY"
    echo "posting through $PROXY"
fi

# sbatch reads SLURM_* variables as if they were command-line options, so a
# surrounding job's shape would leak into the --test-only probes and change the
# estimates they come back with. Cron sets none of these; running this by hand
# from inside an salloc does.
unset SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE \
      SLURM_JOB_PARTITION SLURM_JOB_ACCOUNT SLURM_TIMELIMIT

python3 "$HERE/slurm_probe.py" "$COMMAND"
STATUS=$?
echo "--- $(date '+%F %T')  $COMMAND exit $STATUS"
exit $STATUS
