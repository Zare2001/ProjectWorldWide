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

# --- Environment: LUMI-maintained container ---------------------------------
# Already contains torch 2.7.1+rocm6.2.4, torchvision, transformers, tokenizers,
# datasets, accelerate, flash-attn and the aws-ofi-rccl plugin. No build needed.
export PWW_CONTAINER="${PWW_CONTAINER:-/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif}"

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
