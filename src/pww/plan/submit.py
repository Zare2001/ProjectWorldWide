"""Actually run a plan's commands, from whichever of the plan's machines this is.

    python3 -m pww.plan.submit --plan plan.json --state runs/scheduler/events.jsonl
    python3 -m pww.plan.submit --plan plan.json --state ... --dry-run

Runs ONLY the commands this machine can actually execute, and prints the rest as
copy-paste commands naming which machine to run them on -- `pww.plan sbatch` has
always printed exactly these lines for a human to paste; this adds actually
running the ones reachable from wherever it is invoked, and recording what
happened.

WHY NOT ONE PROCESS THAT SUBMITS EVERYWHERE
---------------------------------------------
LUMI, Snellius and the aggregator VM are three machines with no shared
filesystem (`env.sh` detects which one it is on and sources a different site
file; `PWW_ROOT` is not the same path on any two of them -- see the site
switch in env.sh), so one process submitting "everywhere" would need SSH
config, host keys and credentials this tool has no business holding, for a
2-day scheduling experiment. The intended use is: run this exact script,
unmodified, from each of the up to 3 places a plan touches (each site's login
node for its `sbatch` lines, the aggregator VM for the `(central)` line); each
invocation does only what it locally can.

MATCHING A SUBMISSION TO "CAN I RUN THIS HERE"
-------------------------------------------------
  (central) line   runnable iff scripts/central_node/start_central_services.sh
                    exists under --root
  a site line       runnable iff `sbatch` is on PATH *and* this host's detected
                    site (PWW_SITE, or the same directory/hostname sniff env.sh
                    itself uses) equals the submission's site

Getting the second one wrong in the permissive direction -- running LUMI's
`sbatch` line from Snellius -- is not possible (there is no sbatch to find), so
the failure mode of this check is "skips a line it could have run", never
"runs a line at the wrong site".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Sequence

from . import state as state_mod
from .emit import Submission

_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")


def detect_site() -> str:
    """The same heuristic env.sh's `pww_detect_site` uses, so this file's idea of
    "which site am I on" cannot silently diverge from the shell environment every
    job script already trusts."""
    if os.environ.get("PWW_SITE"):
        return os.environ["PWW_SITE"]
    if Path("/appl/local/containers/sif-images").is_dir():
        return "lumi"
    if Path("/sw/arch").is_dir():
        return "snellius"
    try:
        hostname = socket.getfqdn()
    except OSError:
        hostname = ""
    if "snellius" in hostname:
        return "snellius"
    return "central"


def can_run_here(sub: Submission, *, root: str, site: str) -> tuple[bool, str]:
    """(runnable, reason-if-not)."""
    if sub.site == "(central)":
        script = Path(root) / "scripts" / "central_node" / "start_central_services.sh"
        if script.exists():
            return True, ""
        return False, f"{script} not found under --root {root}; run this from the " \
                      f"aggregator VM's PWW checkout"
    if shutil.which("sbatch") is None:
        return False, f"no sbatch on this host (detected site: {site}); run from " \
                      f"{sub.site}'s login node"
    if site != sub.site:
        return False, f"this host is {site}, not {sub.site}; run from {sub.site}'s " \
                      f"login node"
    return True, ""


def run_submission(sub: Submission, *, root: str, host: str,
                   env: dict[str, str] | None = None) -> dict:
    """Execute one Submission's command through bash (line continuations and
    `$DARL_TOKEN` expansion are exactly what the string was built for -- see
    emit.py's `_export` docstring on why the token is a shell reference, never a
    literal, in `sub.command`), and return the event to record.
    """
    proc = subprocess.run(
        ["bash", "-c", sub.command], cwd=root, env=env if env is not None else os.environ.copy(),
        capture_output=True, text=True, timeout=60)
    role = "central" if sub.site == "(central)" else "site"
    base = dict(site=sub.site, role=role, lane_id=sub.lane_id, link=1,
               shape=sub.args_verbatim, host=host)
    if proc.returncode != 0:
        return {**base, "event": "submit_failed",
                "reason": (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[:500]}
    match = _JOB_ID_RE.search(proc.stdout)
    if role == "site" and not match:
        return {**base, "event": "submit_failed",
                "reason": f"sbatch exited 0 but no job id in its output: "
                          f"{proc.stdout.strip()[:300]!r}"}
    event = {**base, "event": "submitted"}
    if match:
        event["job_id"] = match.group(1)
    return event


def submit_all(submissions: Sequence[Submission], *, root: str, state_path: str,
               dry_run: bool = False) -> list[dict]:
    """Run every Submission this host can run, record every attempt (including
    skips), and return the events in submission order -- schedule.py prints
    this directly rather than re-deriving it from the state file, so what the
    operator sees on screen and what got logged are the same read."""
    site = detect_site()
    results = []
    for sub in submissions:
        runnable, reason = can_run_here(sub, root=root, site=site)
        if dry_run:
            event = {"event": "would_submit" if runnable else "would_skip",
                     "site": sub.site, "role": "central" if sub.site == "(central)" else "site",
                     "lane_id": sub.lane_id, "link": 1, "reason": reason, "host": site}
        elif not runnable:
            event = state_mod.append_event(state_path, {
                "event": "submit_skipped", "site": sub.site,
                "role": "central" if sub.site == "(central)" else "site",
                "lane_id": sub.lane_id, "link": 1, "reason": reason, "host": site})
        else:
            event = state_mod.append_event(state_path, run_submission(sub, root=root, host=site))
        results.append(event)
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pww-plan-submit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, metavar="PLAN.JSON",
                    help="output of `pww.plan --json` (or pww.plan.schedule)")
    ap.add_argument("--root", default=".")
    ap.add_argument("--state", default=None, metavar="PATH",
                    help="default: <root>/runs/scheduler/events.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would run/skip on this host; write nothing")
    return ap


def submissions_from_plan_json(tree: dict) -> list[Submission]:
    """The `sbatch` block `report.as_json` writes (`dataclasses.asdict(Submission)`
    per entry), back into `Submission` objects. Deliberately narrow (only the
    fields `run_submission` needs) rather than a generic round-trip, so a
    report.py field this does not use can still change shape without this
    module needing to track it."""
    out = []
    for row in tree.get("sbatch", []):
        out.append(Submission(
            site=row["site"], lane_id=row["lane_id"], order=row.get("order", 0.0),
            begin_s=row.get("begin_s", 0.0), args_verbatim=row.get("args_verbatim", ""),
            command=row["command"], comment=row.get("comment", "")))
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tree = json.loads(Path(args.plan).expanduser().read_text())
    submissions = submissions_from_plan_json(tree)
    if not submissions:
        print("nothing to submit: the plan file has an empty `submissions` list "
              "(an empty plan, or the wrong file)")
        return 2
    state_path = args.state or str(state_mod.default_state_path(args.root))
    site = detect_site()
    print(f"host detected as: {site}" + ("  [--dry-run]" if args.dry_run else ""))
    print(f"state: {state_path}")
    results = submit_all(submissions, root=args.root, state_path=state_path,
                         dry_run=args.dry_run)
    ran = failed = skipped = 0
    for r in results:
        ev = r["event"]
        if ev in ("submitted", "would_submit"):
            ran += 1
            job = f" job_id={r['job_id']}" if r.get("job_id") else ""
            print(f"  {ev:<14} {r['site']:<10} {r['lane_id']:<14}{job}")
        elif ev in ("submit_failed",):
            failed += 1
            print(f"  {ev:<14} {r['site']:<10} {r['lane_id']:<14} {r['reason']}")
        else:
            skipped += 1
            print(f"  {ev:<14} {r['site']:<10} {r['lane_id']:<14} {r['reason']}")
    print(f"\n{ran} ran, {failed} failed, {skipped} skipped (not runnable from this host)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
