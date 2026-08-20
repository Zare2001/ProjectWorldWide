#!/usr/bin/env bash
# Plan the next federated submission: which sites, what shape, how long, when.
#
#   scripts/plan_campaign.sh                              # live sources
#   scripts/plan_campaign.sh --alpha 0                    # federated rounds only
#   scripts/plan_campaign.sh --dry-run tests/fixtures/plan/two-site.json
#   scripts/plan_campaign.sh show                         # every input, with provenance
#   scripts/plan_campaign.sh sbatch                       # the commands, nothing else
#
# A thin wrapper: it sources env.sh (which puts src/ on PYTHONPATH and resolves
# PWW_OUTPUT_DIR), finds the DARL token the same way the job scripts do, and hands
# everything else through to `python3 -m pww.plan`. Run `--help` for the knobs.
#
# Why a wrapper at all: the planner needs three things that live in three different
# places on three different machines -- the scanner URL, the throughput registry and
# the DARL token -- and getting the token wrong is a silent 401 that reads as "the
# coordinator is down" rather than "you are not allowed to ask".
#
# Runs on the aggregator VM or on either login node. Standard library only, no venv,
# no GPU, no allocation: it reads numbers and prints a plan.
set -euo pipefail

PWW_ROOT="${PWW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "${PWW_ROOT}/env.sh"
cd "${PWW_ROOT}"

# The token, in the order the job scripts try it, so a token that submits a job also
# reads /status. The planner does the same search itself; doing it here too means the
# failure is reported once, by name, before any planning happens.
if [[ -z "${DARL_TOKEN:-}" ]]; then
    for candidate in "${PWW_OUTPUT_DIR:-}/darl/token" \
                     "${PWW_ROOT}/runs/darl/token" \
                     "${PWW_ROOT}/runs/central/darl/token"; do
        [[ -r "${candidate}" ]] || continue
        DARL_TOKEN="$(tr -d '[:space:]' < "${candidate}")"
        [[ -n "${DARL_TOKEN}" ]] && { export DARL_TOKEN; echo "darl token: ${candidate}" >&2; break; }
    done
fi
if [[ -z "${DARL_TOKEN:-}" ]]; then
    echo "note: no DARL token found, so GET /status will 401 and the remaining corpus" >&2
    echo "      will be ASSUMED to be the whole epoch. Pass --blocks <n>, or export" >&2
    echo "      DARL_TOKEN, or use --darl-token-file." >&2
fi

# Defaults for THIS campaign. Everything is overridable on the command line, and an
# explicit flag wins because it comes later in the argv.
exec python3 -m pww.plan \
    --root "${PWW_ROOT}" \
    --registry "${PWW_ROOT}/configs/site_throughput.env" \
    --planner-config "${PWW_ROOT}/configs/plan/federation.json" \
    --probe-config-dir "${PWW_ROOT}/configs/slurm_probe" \
    --config "${PWW_CONFIG:-configs/titan/qwen3_0.6b_c4_diloco.toml}" \
    --darl-url "http://${PWW_CENTRAL_IP:-145.38.206.143}:${PWW_DARL_PORT:-29510}" \
    --darl-port "${PWW_DARL_PORT:-29510}" \
    --flower-port "${PWW_FLOWER_PORT:-29511}" \
    ${PWW_OUTPUT_DIR:+--output-dir "${PWW_OUTPUT_DIR}"} \
    "$@"
