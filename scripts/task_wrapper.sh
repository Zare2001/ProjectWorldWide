#!/bin/bash
# Per-rank entrypoint, executed by each srun task.
#
# Translates SLURM's view of the job into what torch.distributed expects, pins
# the rank to one accelerator, and applies whatever per-site workarounds are
# needed. Then execs the real command.
#
# Site-agnostic: reads PWW_ACCELERATOR / PWW_GPU_VISIBLE_VAR from the sourced
# site file rather than hardcoding ROCm or CUDA.

set -euo pipefail

# --- torch.distributed rendezvous ------------------------------------------
# MASTER_ADDR/MASTER_PORT come from the sbatch script -- they must be identical
# across nodes, so they cannot be derived per-task.
export RANK="${SLURM_PROCID}"
export WORLD_SIZE="${SLURM_NTASKS}"
export LOCAL_RANK="${SLURM_LOCALID}"
export LOCAL_WORLD_SIZE="${SLURM_NTASKS_PER_NODE:-1}"

# --- Accelerator pinning ----------------------------------------------------
# Give each rank exactly one device, via whichever variable this site uses
# (ROCR_VISIBLE_DEVICES on ROCm, CUDA_VISIBLE_DEVICES on NVIDIA). Consequence:
# inside the process there is only ever device 0, which is why the training code
# keys off torch.cuda.device_count() rather than LOCAL_RANK.
#
# One subtlety, which bites on Snellius rather than LUMI: SLURM may have pinned
# the task already. With --gpus-per-node the whole node's GPUs are visible to
# every task and this assignment is what separates them. With --gpus-per-task
# SLURM hands each task exactly one device, already renumbered to 0 -- and
# overwriting that with the local id points ranks 1..N-1 at devices that do not
# exist in their cgroup. So only assign when more than one device is visible.
if [[ -z "${PWW_NO_GPU_PIN:-}" && -n "${PWW_GPU_VISIBLE_VAR:-}" ]]; then
    _pww_visible="${!PWW_GPU_VISIBLE_VAR:-}"
    _pww_n_visible=$(awk -F, 'NF{print NF}' <<<"${_pww_visible}")
    if [[ -z "${_pww_visible}" || "${_pww_n_visible:-0}" -gt 1 ]]; then
        export "${PWW_GPU_VISIBLE_VAR}"="${SLURM_LOCALID}"
    fi
    unset _pww_visible _pww_n_visible
fi

# --- ROCm: MIOpen cache -----------------------------------------------------
# MIOpen's default cache lives in $HOME and is not multi-process safe;
# concurrent ranks corrupt it and die on cryptic SQLite errors. Give every rank
# its own node-local copy. No CUDA equivalent is needed.
if [[ "${PWW_ACCELERATOR:-}" == "rocm" ]]; then
    MIOPEN_DIR="/tmp/${USER}-miopen-${SLURM_JOB_ID:-nojob}-${SLURM_LOCALID:-0}"
    mkdir -p "${MIOPEN_DIR}"
    export MIOPEN_USER_DB_PATH="${MIOPEN_DIR}"
    export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_DIR}"
fi

# --- CPU threading ----------------------------------------------------------
# Each rank owns a slice of the node's cores, shared with its dataloader
# workers. Leaving OMP unbounded makes every rank spawn one thread per core and
# thrash.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# --- Interconnect -----------------------------------------------------------
# NCCL and RCCL share this configuration surface, but the right interface name
# differs: hsn* are LUMI's Slingshot NICs, ib* is InfiniBand elsewhere. Set
# NCCL_SOCKET_IFNAME in the job script when a site needs something specific.
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
# Uncomment to debug collectives:
# export NCCL_DEBUG=INFO

exec "$@"
