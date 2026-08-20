#!/bin/bash
#SBATCH --job-name=slurm_probe
#SBATCH --partition=staging
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm_probe.log
#SBATCH --open-mode=append
#
# Collect once, then resubmit for the next cycle.
#
#   sbatch slurm_probe_loop.sh     start the chain
#   touch .stop                    end it after the current cycle
#   tail -f slurm_probe.log        watch it
#
# Raise DELAY before leaving this running: every cycle runs one `sacct` scan
# over 48 hours, and five minutes means 288 of them a day.

HERE=$HOME/git/slurm_scanner/v2/collector   # the one path to edit
DELAY=now+15minutes

cd "$HERE" || exit 1
echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-none}"

# Resubmitted before the collector runs, so the cadence does not drift and a bad
# cycle -- server down, config wrong -- does not silently end the chain.
[ -e .stop ] && echo "stopping: .stop is present" ||
    sbatch --begin=$DELAY "$HERE/slurm_probe_loop.sh"

# sbatch reads SLURM_* variables as if they were command-line options, so this
# job's own shape would leak into the --test-only probes below and change the
# estimates they come back with.
unset SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE

# probe first: it records the queue as it is right now, and the sacct scan in
# `usage` can take minutes.
python3 slurm_probe.py probe
python3 slurm_probe.py usage
