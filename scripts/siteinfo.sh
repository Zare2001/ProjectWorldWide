#!/bin/bash
# Report everything needed to configure a new site. Run on a LOGIN NODE.
#
#   ./scripts/siteinfo.sh
#
# On Snellius, use the output to correct the [VERIFY] values in
# sites/snellius.sh. Nothing here modifies anything.

echo "=============================================================="
echo " ProjectWorldWide site report"
echo "=============================================================="
echo "hostname     : $(hostname -f 2>/dev/null || hostname)"
echo "user         : ${USER}"
echo "cluster      : ${SLURM_CLUSTER_NAME:-$(scontrol show config 2>/dev/null | awk -F= '/ClusterName/{gsub(/ /,"",$2); print $2}')}"
echo

echo "--- detected site --------------------------------------------"
if [[ -r "$(dirname "$(readlink -f "$0")")/../env.sh" ]]; then
    # shellcheck source=/dev/null
    source "$(dirname "$(readlink -f "$0")")/../env.sh" 2>/dev/null \
        && pww_summary \
        || echo "env.sh could not be sourced -- site probably not configured yet"
fi
echo

echo "--- GPU partitions (Avail/Idle/Other/Total) ------------------"
sinfo -o "%.20P %.8a %.14l %.10c %.10G %.16F" 2>/dev/null | grep -iE "PARTITION|gpu|-g" | head -20
echo

echo "--- accounts you can charge to -------------------------------"
sacctmgr -nP show assoc user="${USER}" format=Account,Partition 2>/dev/null | sort -u | head -20 \
    || echo "(sacctmgr unavailable)"
echo

echo "--- a GPU node's actual hardware -----------------------------"
# The authoritative answer for PWW_GPUS_PER_NODE and PWW_CPUS_PER_TASK:
# ranks/node = Gres gpu count, cores/rank = CPUTot / that count.
for part in gpu gpu_a100 gpu_h100 standard-g small-g dev-g; do
    node=$(sinfo -h -p "${part}" -o "%N" 2>/dev/null | head -1)
    [[ -z "${node}" ]] && continue
    first=$(scontrol show hostnames "${node}" 2>/dev/null | head -1)
    [[ -z "${first}" ]] && continue
    echo "partition ${part} (node ${first}):"
    scontrol show node "${first}" 2>/dev/null \
        | grep -oE "CPUTot=[0-9]+|Gres=[^ ]+|RealMemory=[0-9]+|Sockets=[0-9]+" \
        | sed 's/^/    /'
done
echo

echo "--- filesystems ----------------------------------------------"
for d in "${HOME}" /scratch-shared/"${USER}" /scratch-local "${TMPDIR:-}" \
         /projects /project /scratch /flash; do
    [[ -z "${d}" || ! -e "${d}" ]] && continue
    printf "  %-40s %s\n" "${d}" "$(df -h "${d}" 2>/dev/null | awk 'NR==2{print $2" total, "$4" avail"}')"
done
echo

echo "--- python / torch environment -------------------------------"
if command -v module >/dev/null 2>&1; then
    echo "Lmod present. Candidate PyTorch modules:"
    module avail PyTorch 2>&1 | grep -iE "pytorch" | head -10 | sed 's/^/    /'
else
    echo "no Lmod"
fi
echo
echo "torch as currently reachable:"
python3 -c "
import torch
print(f'    torch        {torch.__version__}')
print(f'    cuda avail   {torch.cuda.is_available()}  devices={torch.cuda.device_count()}')
print(f'    cuda version {torch.version.cuda}')
print(f'    hip version  {torch.version.hip}')
for m in ('torchvision','transformers','tokenizers','datasets','flash_attn'):
    try:
        import importlib; print(f'    {m:12s} {getattr(importlib.import_module(m), \"__version__\", \"?\")}')
    except Exception:
        print(f'    {m:12s} MISSING')
" 2>&1 | sed 's/^\(Traceback\|  \|[A-Za-z]*Error\)/    &/' || echo "    python3 has no torch on the login node (expected if modules are needed)"
echo
echo "=============================================================="
echo "On a new site, use the above to fill in the [VERIFY] values in"
echo "sites/<site>.sh, then run: sbatch scripts/<site>/job_smoke.sh"
echo "=============================================================="
