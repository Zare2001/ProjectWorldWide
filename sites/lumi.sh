# LUMI (CSC, Finland) -- AMD MI250X, ROCm.
#
# Verified working: 8 GCDs on one node and 16 ranks across two nodes.

export PWW_ACCOUNT="${PWW_ACCOUNT:-project_462000226}"
export PWW_SCRATCH="${PWW_SCRATCH:-/scratch/${PWW_ACCOUNT}/${USER}}"

# A LUMI-G node has 4 MI250X cards, but each card is two GCDs and ROCm treats
# each GCD as an independent GPU -- so a full node is 8 ranks, not 4.
# 64 cores, one per L3 group reserved for the OS, leaving 7 usable per rank.
export PWW_GPUS_PER_NODE=8
export PWW_CPUS_PER_TASK=7

export PWW_ACCELERATOR=rocm
export PWW_GPU_VISIBLE_VAR=ROCR_VISIBLE_DEVICES

# --- Environment: container -------------------------------------------------
# Pick ONE of the two below by leaving exactly one uncommented. Either can also
# be overridden per job without editing this file at all:
#
#   PWW_CONTAINER=/path/to/other.sif sbatch scripts/lumi/job_smoke.sh
#
# Both were benchmarked head to head on the same jobs and are indistinguishable
# for CIFAR-scale work:
#
#                          official          laif
#   CIFAR 30ep 8 GCDs      93.35%            93.27%
#                          38,400 img/s      37,700 img/s
#   all-reduce 1 node      123 GB/s          123.4 GB/s
#   all-reduce 2 nodes     88 GB/s           87.8 GB/s
#   tests/test_local.py    17/17             17/17

# [1] DEFAULT -- LUMI-maintained, under /appl, permanent and system-supported.
# torch 2.7.1+rocm6.2.4, torchvision, transformers 4.55.3, tokenizers, datasets,
# accelerate, flash-attn 2.7.3, aws-ofi-rccl. Enough for the CIFAR phase and for
# plain FSDP LLM training. Prefer this unless you specifically need [2].
PWW_CONTAINER_DEFAULT=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif

# [2] LLM-phase alternative -- torch 2.9.1+rocm6.4.4, transformers 4.57.3,
# flash-attn 2.8.0, Transformer Engine 2.4.0, DeepSpeed 0.18.6, apex, triton 3.2.
# Verified on GPU here: FlashAttention forward, te.Linear (fp32/bf16/fp16),
# apex FusedAdam. Uncomment to use, and comment out [1] above.
#
# Worth switching for: DeepSpeed as a ZeRO alternative to FSDP, apex FusedAdam,
# newer FlashAttention. NOT worth switching for CIFAR -- see the table above.
#
# Three caveats, all verified on this hardware rather than assumed:
#   * fp8 does NOT work on LUMI. TE asserts "Device arch gfx94x or gfx95x
#     required"; MI250X is gfx90a. fp8 needs MI300X or newer -- bf16 is the
#     ceiling here, so do not adopt this image expecting fp8 speedups.
#   * Do not wrap TE layers in torch.autocast on gfx90a: te.Linear then fails
#     with "Unable to find any suitable algorithms". Set layer dtypes explicitly.
#   * It lives on purgeable scratch and is owned by another user, so it can
#     vanish and break every job referencing it. Copy it into your own space
#     before relying on it (13.5 GB against a 50 TB quota).
#PWW_CONTAINER_DEFAULT=/scratch/project_462000226/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif

# A plain assignment above rather than the ${VAR:-...} form on purpose: if you
# uncomment [2] and forget to comment out [1], the later line simply wins, which
# is what you meant. With ${VAR:-...} on both, [1] would silently win instead.
# The environment still overrides either, so per-job selection keeps working.
export PWW_CONTAINER="${PWW_CONTAINER:-${PWW_CONTAINER_DEFAULT}}"

# Bindings for Slingshot (RCCL/libfabric) + Lustre visibility. Mirrors
# `module load singularity-AI-bindings`, inlined so jobs need no Lmod.
export SINGULARITY_BIND="/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/pfs,/scratch,/projappl,/project,/flash,/appl"

# The host sets BASH_ENV to the Cray Lmod init script, which bash sources in every
# non-interactive shell -- including in the container, where there is no lua5.3.
# Harmless, but it prints a "bad interpreter" error per rank that looks exactly
# like a real failure at the top of a job log.
export SINGULARITYENV_BASH_ENV=""

PWW_LAUNCH=(singularity exec "${PWW_CONTAINER}")

# --- CPU binding ------------------------------------------------------------
# Maps task N -> the 7 cores physically closest to GCD N. Wrong ordering here
# silently costs 10-30% throughput. Valid only for a full 8-rank node.
export PWW_CPU_BIND="mask_cpu:7e000000000000,7e00000000000000,7e0000,7e000000,7e,7e00,7e00000000,7e0000000000"

pww_cpu_bind() {
    if [[ "${SLURM_NTASKS_PER_NODE:-1}" -eq "${PWW_GPUS_PER_NODE}" ]]; then
        echo "${PWW_CPU_BIND}"
    else
        # Partial node: SLURM hands out an arbitrary core subset, and forcing the
        # fixed mask fails with "CPU binding outside of job step allocation".
        echo "cores"
    fi
}

# --- Deployment ---------------------------------------------------------------
# How the torch >= 2.9 environment torchtitan needs is provided here, for
# scripts/deploy.sh. Three tab-separated fields: kind, path, build command.
#
# A container rather than a venv, because LUMI's own image is on torch 2.7.1 and
# there is no module tree offering 2.9. See scripts/titan/README.md.
pww_titan_env() {
    printf 'container\t%s\tsbatch scripts/lumi/build_titan_container.sh\n' \
        "${PWW_SCRATCH}/containers/pww-titan.sif"
}
