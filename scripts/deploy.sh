#!/bin/bash
# Deploy or update this repo on whichever machine you are on.
#
#   ./scripts/deploy.sh              # pull, submodules, dirs, verify
#   ./scripts/deploy.sh --check      # report only, change nothing
#
# Site-agnostic on purpose. Everything machine-specific comes from
# sites/<site>.sh, which env.sh selects by detection -- so this same script is
# the deployment path on LUMI, on Snellius and on the central VM, and adding a
# fourth machine means writing a site file rather than another script. See
# "Adding a site" in FEDERATION_GUIDE.md.
#
# Idempotent: safe to re-run, and the normal way to update after a commit.
# It never builds the heavy environment (a container build is a 30-60 minute
# batch job) -- it reports what is missing and the one command that creates it.

set -euo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PWW_ROOT}"

step() { printf '\n=== %s\n' "$1"; }
ok()   { printf '  ok    %s\n' "$1"; }
warn() { printf '  WARN  %s\n' "$1" >&2; }
todo() { printf '  TODO  %s\n' "$1"; }

step "1. repository"
if [[ ! -d .git ]]; then
    echo "ERROR: ${PWW_ROOT} is not a git checkout. Clone it first:" >&2
    echo "  git clone <url> ~/ProjectWorldWide && cd ~/ProjectWorldWide" >&2
    exit 1
fi
branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    warn "tracked files are modified; not pulling. Commit or stash first."
    git status --short --untracked-files=no | sed 's/^/        /'
elif (( CHECK_ONLY )); then
    ok "on ${branch} at $(git rev-parse --short HEAD), clean (not pulling: --check)"
else
    git pull --ff-only
    ok "on ${branch} at $(git rev-parse --short HEAD)"
fi

step "2. submodules"
# Pinned, not floating: third_party/torchtitan is at a specific commit because the
# repo's own torch pin is 2.7.1 and torchtitan's HEAD moves.
if (( CHECK_ONLY )); then
    git submodule status | sed 's/^/  /'
else
    git submodule update --init --recursive
    git submodule status | sed 's/^/  /'
fi

step "3. environment and paths"
# shellcheck source=/dev/null
source "${PWW_ROOT}/env.sh"
pww_summary | sed 's/^/  /'

if (( CHECK_ONLY )); then
    for path in "${PWW_DATA_DIR}" "${PWW_OUTPUT_DIR}"; do
        [[ -d "${path}" ]] && ok "exists  ${path}" || todo "missing ${path}"
    done
else
    mkdir -p "${PWW_DATA_DIR}" "${PWW_OUTPUT_DIR}" "${PWW_TMPDIR}" "${PWW_CACHE_DIR}" logs
    # ln -sfn puts the link INSIDE the target when the name is an existing
    # directory, which silently produces runs/runs and breaks every documented
    # path. Refuse rather than create that.
    for pair in "runs:${PWW_OUTPUT_DIR}" "data:${PWW_DATA_DIR}"; do
        name="${pair%%:*}"; target="${pair#*:}"
        if [[ -e "${name}" && ! -L "${name}" ]]; then
            warn "${name}/ exists as a real directory, so it cannot become a symlink."
            warn "Move it aside and re-run:  mv ${name} ${name}.bak"
        else
            ln -sfn "${target}" "${name}"
            ok "${name} -> $(readlink "${name}")"
        fi
    done
fi

step "4. torchtitan environment (torch >= 2.9)"
# Provided by the site file, so this script has no per-machine knowledge.
if declare -F pww_titan_env >/dev/null; then
    IFS=$'\t' read -r kind path build < <(pww_titan_env)
    case "${kind}" in
        none) ok "not needed on this machine (${PWW_SITE})" ;;
        *)    if [[ -e "${path}" ]]; then
                  ok "${kind} present: ${path}"
              else
                  todo "${kind} missing: ${path}"
                  todo "build it once:  ${build}"
              fi ;;
    esac
else
    warn "sites/${PWW_SITE}.sh defines no pww_titan_env; cannot check the torch 2.9 setup"
fi

step "5. verify"
# test_darl is the one suite that needs neither torchtitan nor a GPU, so it is the
# cheapest signal that the checkout is coherent. Through pww_run because LUMI's
# login nodes ship python 3.6, which cannot parse the tests.
if pww_run python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PYTHONPATH="${PWW_ROOT}/src" pww_run python3 tests/test_darl.py | tail -1 | sed 's/^/  /'
else
    warn "no python >= 3.10 in this environment; skipping tests"
    warn "at a site that means the container or venv is not on PATH yet"
fi

step "next"
case "${PWW_SITE}" in
    central) echo "  ./scripts/central_node/start_central_services.sh   # see RUNBOOK.md Part 2" ;;
    *)       echo "  ./scripts/titan/download_tokenizer.sh              # see RUNBOOK.md Part 1"
             echo "  ./scripts/titan/tokenize_c4.sh --dataset c4 --seq-len 2048 --max-files 32"
             echo "  DARL_TOKEN=... sbatch scripts/${PWW_SITE}/job_titan_diloco.sh   # Part 3" ;;
esac
