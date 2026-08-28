"""One poll of "what is running, pending or finished", from whatever sources are
reachable from here.

    python3 -m pww.plan.monitor --darl-url http://<agg>:29510 --darl-token-file ...
    python3 -m pww.plan.monitor --state runs/scheduler/events.jsonl --squeue
    python3 -m pww.plan.monitor --state runs/scheduler/events.jsonl --darl-url ... --squeue

Prints a table and appends one line per thing it saw to the SAME event log
`submit.py` writes to (see state.py) -- so the log is one growing record a
dashboard can tail later, not a second store that can disagree with the first.

WHY THIS IS "--once" BY DEFAULT, NOT A DAEMON
------------------------------------------------
`elastic_watch.sh`'s own header already says it: "Login nodes occasionally
reap long-lived processes." Rather than accept that risk here, this is a
single poll that exits -- run it from cron on a login node (same pattern this
repo already uses for `slurm-scanner-main/collector/slurm_probe_loop.sh`) or by
hand whenever you want a fresh read. `--loop --interval N` exists for
interactive use, with the exact same reap risk as `elastic_watch.sh` already
documents -- prefer cron for anything unattended.

TWO INDEPENDENT SOURCES, DELIBERATELY NOT ONE
-------------------------------------------------
  --darl-url    reachable from ANYWHERE with network access (the aggregator
                VM, a laptop) -- no login node needed. Tells you which
                clusters are actually LIVE and their committed-block progress,
                i.e. "is training happening", but nothing about a job still
                queued: a job that has not registered with DARL yet is
                indistinguishable from one never submitted.
  --squeue      SLURM's own PENDING/RUNNING/COMPLETED/FAILED, which is the
                only source that sees a queued job -- but only for jobs
                submitted from wherever this runs, and only `squeue`/`sacct`
                are on PATH, i.e. this must run on a login node.

Either alone is a real, honest partial view; run whichever this host can
reach and read the other source's rows as "unknown" rather than assume they
mean "not running".
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import state as state_mod
from . import inputs as io


def darl_snapshot(darl_url: str, token: str | None, *, block_size: int, seq_len: int,
                  fresh_within_s: float = 300.0) -> dict[str, dict[str, Any]]:
    """{cluster_id: {alive, blocks_committed, tokens, ranks, last_seen_age_s}}.

    A network or auth failure raises rather than returning an empty map: a
    poll that silently reports "nothing running" on a coordinator that is
    merely unreachable is a false negative a log reader would trust.
    """
    headers = {"X-DARL-Token": token} if token else {}
    payload = io.fetch_json(f"{darl_url.rstrip('/')}/status", headers=headers, timeout=15.0)
    alive = io.darl_liveness(payload, fresh_within_s=fresh_within_s)
    now = time.time()
    out = {}
    for name, record in (payload.get("clusters") or {}).items():
        blocks = int(record.get("blocks_committed", 0))
        out[name] = {
            "alive": alive.get(name, False),
            "blocks_committed": blocks,
            "tokens": blocks * block_size * seq_len,
            "ranks": int(record.get("ranks", 0)),
            "last_seen_age_s": now - float(record.get("last_seen") or 0.0),
        }
    out["_epoch"] = {"epoch": payload.get("epoch"), "max_epochs": payload.get("max_epochs"),
                     "committed": payload.get("committed"), "num_blocks": payload.get("num_blocks"),
                     "unassigned": payload.get("unassigned")}
    return out


def squeue_snapshot(*, name_prefix: str = "pww-") -> list[dict[str, str]]:
    """[{job_id, name, state}], every job of ours squeue currently sees.

    `--me` restricts to the invoking user, which is the right scope: a shared
    login node's squeue includes everyone's jobs, and this tool has no
    business reporting on jobs it did not submit.
    """
    if shutil.which("squeue") is None:
        raise RuntimeError("no squeue on this host -- --squeue must run on a login node")
    proc = subprocess.run(
        ["squeue", "--me", "--noheader", "--format=%i|%j|%T"],
        capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"squeue exited {proc.returncode}: {proc.stderr.strip()}")
    out = []
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 3:
            continue
        job_id, name, state = parts
        if name.startswith(name_prefix):
            out.append({"job_id": job_id, "name": name, "state": state})
    return out


def render(darl: dict[str, dict[str, Any]] | None, squeue_rows: list[dict[str, str]] | None,
          job_views) -> str:
    lines = []
    if darl is not None:
        epoch = darl.get("_epoch", {})
        lines.append(f"DARL  epoch {epoch.get('epoch')}/{epoch.get('max_epochs')}  "
                     f"committed {epoch.get('committed')}/{epoch.get('num_blocks')}  "
                     f"unassigned {epoch.get('unassigned')}")
        lines.append(f"  {'cluster':<20} {'alive':<6} {'blocks':>8} {'tokens':>12} "
                     f"{'ranks':>6} {'last_seen':>10}")
        for name, row in sorted(darl.items()):
            if name == "_epoch":
                continue
            lines.append(f"  {name:<20} {'yes' if row['alive'] else 'NO':<6} "
                         f"{row['blocks_committed']:>8} {row['tokens'] / 1e9:>10.2f}B "
                         f"{row['ranks']:>6} {row['last_seen_age_s']:>8.0f}s")
    if squeue_rows is not None:
        lines.append("")
        lines.append(f"SQUEUE  {len(squeue_rows)} job(s) matching this user/prefix")
        for row in squeue_rows:
            lines.append(f"  {row['job_id']:<10} {row['state']:<12} {row['name']}")
    if job_views:
        lines.append("")
        lines.append(f"TRACKED JOBS (from the submission log)")
        lines.append(f"  {'site':<10} {'lane':<14} {'job_id':<10} {'state':<12} "
                     f"{'age':>8}  reason")
        now = time.time()
        for v in job_views:
            age = f"{(now - v.last_update) / 60:.0f}m" if v.last_update else "-"
            lines.append(f"  {v.site:<10} {v.lane_id:<14} {v.job_id or '-':<10} "
                         f"{v.state:<12} {age:>8}  {v.reason}")
    return "\n".join(lines) if lines else "(nothing to report -- pass --darl-url and/or --squeue)"


def poll_once(*, darl_url: str | None, darl_token: str | None, block_size: int, seq_len: int,
             use_squeue: bool, state_path: str | None, name_prefix: str) -> str:
    darl = squeue_rows = job_views = None
    events = []

    if darl_url:
        darl = darl_snapshot(darl_url, darl_token, block_size=block_size, seq_len=seq_len)
        for name, row in darl.items():
            if name == "_epoch":
                continue
            events.append({"event": "status", "site": name.split("-", 1)[0], "lane_id": name,
                           "state": "running" if row["alive"] else "stale",
                           "source": "darl", "blocks_committed": row["blocks_committed"],
                           "tokens": row["tokens"]})

    if use_squeue:
        squeue_rows = squeue_snapshot(name_prefix=name_prefix)
        for row in squeue_rows:
            events.append({"event": "status", "job_id": row["job_id"],
                           "state": row["state"], "source": "squeue"})

    if state_path:
        for ev in events:
            state_mod.append_event(state_path, ev)
        job_views = state_mod.fold(state_mod.read_events(state_path))

    return render(darl, squeue_rows, job_views)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pww-plan-monitor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--darl-url", default=None,
                    help="poll DARL /status for live cluster/token progress. "
                         "Reachable from anywhere with network access")
    ap.add_argument("--darl-token", default=None)
    ap.add_argument("--darl-token-file", default=None)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--squeue", action="store_true",
                    help="also poll squeue for this user's pww- jobs. Must run on "
                         "a login node")
    ap.add_argument("--name-prefix", default="pww-")
    ap.add_argument("--state", default=None, metavar="EVENTS.JSONL",
                    help="append what this poll saw here, and print tracked-job "
                         "status folded from it. Default: no state file, no "
                         "TRACKED JOBS section")
    ap.add_argument("--loop", action="store_true",
                    help="repeat every --interval seconds instead of polling once "
                         "(see module docstring: prefer cron on a login node)")
    ap.add_argument("--interval", type=int, default=120)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = args.darl_token
    if not token and args.darl_token_file:
        token = Path(args.darl_token_file).expanduser().read_text().strip()
    if not args.darl_url and not args.squeue:
        print("nothing to poll: pass --darl-url and/or --squeue", file=sys.stderr)
        return 2

    def once() -> int:
        try:
            print(f"[{time.strftime('%F %T')}]")
            print(poll_once(darl_url=args.darl_url, darl_token=token,
                            block_size=args.block_size, seq_len=args.seq_len,
                            use_squeue=args.squeue, state_path=args.state,
                            name_prefix=args.name_prefix))
            print()
            return 0
        except Exception as exc:  # noqa: BLE001 -- a poll failing must not crash a loop
            print(f"  POLL FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    if not args.loop:
        return once()
    rc = 0
    while True:
        rc = once()
        time.sleep(args.interval)
    return rc  # pragma: no cover -- unreachable, satisfies linters expecting a return


if __name__ == "__main__":
    raise SystemExit(main())
