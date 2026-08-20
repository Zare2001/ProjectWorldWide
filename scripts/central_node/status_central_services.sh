#!/bin/bash
# Status and health check for central node services (DARL on 29510, Flower on 29511)

set -euo pipefail

: "${PWW_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PWW_SITE="${PWW_SITE:-central}"
source "${PWW_ROOT}/env.sh"

DARL_PORT="${DARL_PORT:-29510}"
FLOWER_PORT="${FLOWER_PORT:-29511}"
BLOB_PORT="${BLOB_PORT:-29512}"
STATE_DIR="${PWW_OUTPUT_DIR:-${PWW_ROOT}/runs}/central"

echo "=== DARL Coordinator Status (Port ${DARL_PORT}) ==="
DARL_HOST="127.0.0.1" DARL_PORT="${DARL_PORT}" "${PWW_ROOT}/scripts/darl_coordinator.sh" status || true

echo ""
echo "=== DARL Health Endpoint Check ==="
DARL_TOKEN_FILE="${PWW_OUTPUT_DIR:-${PWW_ROOT}/runs}/darl/token"
if [[ -s "${DARL_TOKEN_FILE}" ]]; then
    curl -sS -H "X-DARL-Token: $(cat "${DARL_TOKEN_FILE}")" "http://127.0.0.1:${DARL_PORT}/health" || echo "DARL HTTP endpoint unreachable on port ${DARL_PORT}"
else
    curl -sS "http://127.0.0.1:${DARL_PORT}/health" || echo "DARL HTTP endpoint unreachable on port ${DARL_PORT}"
fi

echo ""
echo "=== Flower Aggregator Server Status (Port ${FLOWER_PORT}) ==="
FLOWER_PID_FILE="${STATE_DIR}/flower.pid"
if [[ -f "${FLOWER_PID_FILE}" ]] && kill -0 "$(cat "${FLOWER_PID_FILE}")" 2>/dev/null; then
    echo "Flower server is RUNNING (PID $(cat "${FLOWER_PID_FILE}"))."
else
    echo "Flower server is STOPPED."
fi

echo ""
echo "=== Blob Store Status (Port ${BLOB_PORT}) ==="
BLOB_PID_FILE="${STATE_DIR}/blob.pid"
if [[ -f "${BLOB_PID_FILE}" ]] && kill -0 "$(cat "${BLOB_PID_FILE}")" 2>/dev/null; then
    echo "Blob store is RUNNING (PID $(cat "${BLOB_PID_FILE}"))."
    curl -sS "http://127.0.0.1:${BLOB_PORT}/health" 2>/dev/null | python3 -c '
import json, sys
try:
    usage = json.load(sys.stdin).get("usage", {})
except Exception:
    sys.exit()
gib = 2 ** 30
# Bound to locals first: an f-string expression cannot contain a backslash before
# Python 3.12, and this VM runs 3.10 -- the escaped quotes needed by the enclosing
# shell single-quotes made every blob line a SyntaxError, so the one transport that
# needs these counters was the one that never printed them.
blobs = usage.get("blobs", 0)
used = usage.get("bytes", 0) / gib
free = usage.get("disk_free", 0) / gib
bin_ = usage.get("bytes_in", 0) / gib
bout = usage.get("bytes_out", 0) / gib
print(f"  blobs {blobs} | {used:.2f} GiB used | {free:.2f} GiB free")
print(f"  transferred: {bin_:.2f} GiB in, {bout:.2f} GiB out")
' || true
else
    echo "Blob store is STOPPED (only needed for TRANSPORT=blob)."
fi

echo ""
echo "=== Global Model State ==="
GLOBAL_STATE_DIR="${GLOBAL_STATE_DIR:-${STATE_DIR}/global}"
if [[ -s "${GLOBAL_STATE_DIR}/meta.json" ]]; then
    # The merge counter, not Flower's round counter: this is how many times the
    # global model actually changed, and it is what deltas are validated against.
    python3 -c '
import json, sys
meta = json.load(open(sys.argv[1]))
print(f"  merge round   {meta.get(\"round\", 0)}")
print(f"  parameters    {meta.get(\"model_numel\", 0):,} in {len(meta.get(\"keys\", []))} tensors "
      f"({meta.get(\"storage_dtype\", \"?\")})")
print(f"  tokens merged {meta.get(\"total_tokens\", 0):,}")
for name, rec in sorted((meta.get("clusters") or {}).items()):
    print(f"  {name:12s} joined r{rec.get(\"first_seen_round\", 0)}, "
          f"last seen r{rec.get(\"last_seen_round\", 0)}, "
          f"{rec.get(\"rounds_contributed\", 0)} rounds, "
          f"{rec.get(\"tokens_total\", 0):,} tokens, "
          f"{rec.get(\"stale_rejected\", 0)} stale rejected")
' "${GLOBAL_STATE_DIR}/meta.json" || true
else
    echo "  no durable global state yet (cold start, or --state-dir unset)"
fi

echo ""
echo "=== Open Port Listener Check ==="
ss -tulpn 2>/dev/null | grep -E "${DARL_PORT}|${FLOWER_PORT}|${BLOB_PORT}" || true
