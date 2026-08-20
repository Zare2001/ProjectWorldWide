#!/bin/bash
#SBATCH --job-name=slurm_probe
#SBATCH --account=project_462000226
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1750M
#SBATCH --time=00:10:00
#SBATCH --output=slurm_probe.log
#SBATCH --open-mode=append
#
# Collect once, then resubmit for the next cycle on LUMI.
#
#   sbatch slurm_probe_loop_lumi.sh     start the chain
#   touch .stop                         end it after the current cycle
#   tail -f slurm_probe.log             watch it
#

HERE="/users/vanderwal/slurm_probe"
SCRIPT_PATH="$HERE/slurm_probe_loop_lumi.sh"
DELAY="now+30minutes"

cd "$HERE" || exit 1
echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-none} (partition: ${SLURM_JOB_PARTITION:-debug})"

# Resubmitted before the collector runs, so the cadence does not drift and a bad
# cycle -- server down, config wrong -- does not silently end the chain.
if [ -e "$HERE/.stop" ]; then
    echo "stopping: .stop is present"
else
    sbatch --begin="$DELAY" "$SCRIPT_PATH"
fi

# sbatch reads SLURM_* variables as if they were command-line options, so this
# job's own shape would leak into the --test-only probes below and change the
# estimates they come back with.
unset SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_MEM_PER_CPU SLURM_MEM_PER_NODE SLURM_JOB_PARTITION

# probe first: it records the queue as it is right now, and the sacct scan in
# `usage` can take minutes.
python3 slurm_probe.py probe
python3 slurm_probe.py usage
