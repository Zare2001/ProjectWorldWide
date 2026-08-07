#!/bin/bash
#SBATCH --job-name=pww-build-titan-sif
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

# Build the LUMI torchtitan container from containers/titan-lumi.def.
#
#   sbatch -A $PWW_ACCOUNT scripts/lumi/build_titan_container.sh
#
# As a batch job rather than on a login node: the build runs pip installs and
# writes a multi-GB image, both of which login nodes are the wrong place for. It
# needs internet, which LUMI's small partition has and the GPU compute nodes do
# not.
#
# Three hours, not one: the LAIF base is ~13 GB, and "Creating SIF file" -- the
# squashfs pass, after %post and %test have both already passed -- is most of the
# runtime on Lustre. At one hour it is killed there, which looks like a build
# failure while actually being a nearly-finished image.

set -euo pipefail

: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"
source "${PWW_ROOT}/env.sh"

DEF="${PWW_ROOT}/containers/titan-lumi.def"
OUT="${PWW_TITAN_SIF:-${PWW_SCRATCH}/containers/pww-titan.sif}"

[[ -r "${DEF}" ]] || { echo "no definition at ${DEF}" >&2; exit 1; }
mkdir -p "$(dirname "${OUT}")"

# PRoot lets singularity build unprivileged, which is the only option on LUMI --
# there is no root and no --fakeroot user namespace setup.
module load CrayEnv PRoot 2>/dev/null || true

# env.sh exports SINGULARITY_BIND for *running* on LUMI: Slingshot, Lustre and
# /var/spool/slurmd. Singularity applies it to the %post container too, where it
# is both useless and fatal -- a bind fails outright if the destination does not
# already exist in the image, and a base image is under no obligation to carry
# LUMI's runtime paths. The maintained lumi-pytorch image happens to, which is why
# this only appears when building on a base from anywhere else:
#   FATAL: mount /var/spool/slurmd -> destination doesn't exist in container
unset SINGULARITY_BIND SINGULARITY_BINDPATH APPTAINER_BIND APPTAINER_BINDPATH

# Both on scratch: a build stages several GB of layers, and $HOME on LUMI is a
# ~20 GB quota that a single failed build fills.
export SINGULARITY_TMPDIR="${PWW_TMPDIR}/singularity-build-${SLURM_JOB_ID}"
export SINGULARITY_CACHEDIR="${PWW_CACHE_DIR}/singularity"
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

echo "definition : ${DEF}"
echo "output     : ${OUT}"
echo "tmpdir     : ${SINGULARITY_TMPDIR}"
echo

if [[ -e "${OUT}" ]]; then
    echo "${OUT} already exists; moving it aside"
    mv "${OUT}" "${OUT}.$(date +%Y%m%d%H%M%S).bak"
fi

singularity build "${OUT}" "${DEF}"

echo
echo "Built ${OUT}"
ls -lh "${OUT}"
echo
echo "Use it with:"
echo "  export PWW_TITAN_SIF=${OUT}"
echo "  sbatch scripts/lumi/job_titan_diloco.sh"
