"""The scheduler's own record of what it submitted and what happened to it.

One JSON Lines file, append-only. Each line is one EVENT, not a snapshot -- so the
file itself is the log the operator reads (`tail -f`), and "current status" is
never a second structure that can drift from it: it is folded from the events on
read, the same way `pww.darl`'s journal-plus-snapshot already works for a
different piece of this system.

    {"ts": 1787..., "event": "submitted", "job_id": "12345", "site": "lumi",
     "role": "site", "lane_id": "lumi-l0", "link": 1, "shape": "1node_40h",
     "host": "lumi-login2"}
    {"ts": 1787..., "event": "submit_failed", "site": "lumi", "role": "site",
     "lane_id": "lumi-l0", "link": 1, "reason": "sbatch: error: ..."}
    {"ts": 1787..., "event": "submit_skipped", "site": "snellius", "role": "site",
     "reason": "no sbatch on this host; run from snellius's login node"}
    {"ts": 1787..., "event": "status", "job_id": "12345", "state": "RUNNING",
     "source": "squeue"}

Why not a mutable JSON snapshot instead: two different processes append to this
file from two different machines (a site's login node for its own `sbatch` calls,
the aggregator VM for its DARL-derived progress checks -- see submit.py and
monitor.py's module docstrings for why those cannot be the same process), and an
append is the one operation that is safe without file locking. A snapshot the
second writer read-modified-wrote would race the first.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def append_event(path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    """Append one event, stamped with `ts` if the caller did not set one.

    Opened in append mode and written as a single `write()` call: on a local or
    NFS-mounted filesystem a write under PIPE_BUF (4096 bytes on Linux; every
    event here is far smaller) is atomic against other appenders, which is the
    property two machines writing the same file depend on.
    """
    event = dict(event)
    event.setdefault("ts", time.time())
    line = json.dumps(event, sort_keys=True)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(line + "\n")
    return event


def read_events(path: str | Path) -> list[dict[str, Any]]:
    """Every event, oldest first. A malformed line is skipped, not fatal -- a
    concurrent writer's partial line (a crash mid-`write()`, which the PIPE_BUF
    atomicity above does not protect against a crash, only a race) must not take
    the whole log down with it."""
    path = Path(path).expanduser()
    if not path.exists():
        return []
    out = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


@dataclass(frozen=True)
class JobView:
    """Everything folded from events into one job's current state."""

    key: str  # job_id if known, else "{site}:{lane_id}:{link}" before submission returns one
    site: str
    role: str  # "central" | "site"
    lane_id: str = ""
    link: int = 0
    job_id: str | None = None
    shape: str = ""
    submitted_at: float | None = None
    state: str = "unknown"  # submitted | running | pending | completed | failed | skipped | unknown
    last_update: float = 0.0
    host: str = ""
    reason: str = ""  # why skipped/failed, if it is
    history: tuple[str, ...] = field(default_factory=tuple)  # state names, in order seen


_EVENT_STATE = {
    "submitted": "submitted",
    "submit_failed": "failed",
    "submit_skipped": "skipped",
}


def fold(events: list[dict[str, Any]]) -> list[JobView]:
    """One `JobView` per (site, lane_id, link), in first-seen order.

    A job's `key` starts as its (site, lane_id, link) triple -- known at
    submit-time, before sbatch has replied -- and becomes the real job_id once a
    "submitted" event supplies one; later "status" events are matched by
    job_id when they have one, else by the same triple, so a squeue poll that
    only knows the job_id still lands on the right row.
    """
    by_triple: dict[tuple[str, str, int], JobView] = {}
    by_job_id: dict[str, tuple[str, str, int]] = {}
    order: list[tuple[str, str, int]] = []

    def triple_of(ev: dict) -> tuple[str, str, int]:
        return (ev.get("site", ""), ev.get("lane_id", ""), int(ev.get("link", 0)))

    for ev in events:
        kind = ev.get("event", "")
        job_id = ev.get("job_id")
        triple = by_job_id.get(job_id) if job_id else None
        if triple is None:
            triple = triple_of(ev)
        if triple not in by_triple:
            by_triple[triple] = JobView(
                key=job_id or f"{triple[0]}:{triple[1]}:{triple[2]}",
                site=triple[0], role=ev.get("role", "site"), lane_id=triple[1],
                link=triple[2], shape=ev.get("shape", ""))
            order.append(triple)
        view = by_triple[triple]
        new_state = view.state
        new_job_id = view.job_id
        new_reason = view.reason
        new_host = view.host
        new_submitted = view.submitted_at

        if kind in _EVENT_STATE:
            new_state = _EVENT_STATE[kind]
            new_reason = ev.get("reason", "")
            new_host = ev.get("host", view.host)
            if kind == "submitted":
                new_submitted = ev.get("ts", view.submitted_at)
                if job_id:
                    new_job_id = str(job_id)
                    by_job_id[str(job_id)] = triple
        elif kind == "status":
            new_state = str(ev.get("state", view.state)).lower()
            if job_id:
                new_job_id = str(job_id)
                by_job_id[str(job_id)] = triple

        history = view.history if new_state == view.state else view.history + (new_state,)
        by_triple[triple] = JobView(
            key=new_job_id or view.key, site=view.site, role=view.role,
            lane_id=view.lane_id, link=view.link, job_id=new_job_id,
            shape=view.shape, submitted_at=new_submitted, state=new_state,
            last_update=ev.get("ts", view.last_update), host=new_host,
            reason=new_reason, history=history)

    return [by_triple[t] for t in order]


def default_state_path(root: str | Path = ".") -> Path:
    return Path(root).expanduser() / "runs" / "scheduler" / "events.jsonl"
