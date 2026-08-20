#!/usr/bin/env python3
"""Slurm start-time and walltime-usage collector.

Single file, standard library only. Copy to a login node, write a config, run
from cron. Everything is configured in ~/.slurm_probe/config.json (override the
path with SLURM_PROBE_CONFIG); there are no command-line options.

    slurm_probe.py probe    # sbatch --test-only for each configured shape
    slurm_probe.py usage    # sacct: elapsed vs requested, over one window

Two independent measurements, never joined here. With no "server" in the
config, the payload is printed instead of posted -- which is also how you check
a new site before pointing it anywhere.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request

VERSION = "2.0.0"
CONFIG_PATH = os.environ.get(
    "SLURM_PROBE_CONFIG", os.path.expanduser("~/.slurm_probe/config.json")
)

# States where elapsed/timelimit is a finished measurement, not a job still
# running.
TERMINAL = frozenset((
    "COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
))


def load_config():
    with open(CONFIG_PATH) as handle:
        config = json.load(handle)
    for key in ("cluster", "partitions", "shapes"):
        if not config.get(key):
            sys.exit("error: %s needs a %r entry" % (CONFIG_PATH, key))
    if config.get("token") and os.stat(CONFIG_PATH).st_mode & 0o077:
        sys.exit("error: %s holds a token but is readable by others; chmod 600 it"
                 % CONFIG_PATH)
    config.setdefault("usage_hours", 48)
    return config


def sh(cmd, timeout=120):
    """Run a command, returning stdout+stderr and ignoring the exit code.

    `sbatch --test-only` answers on stderr on some versions and exits non-zero
    for perfectly informative verdicts ("Requested node configuration is not
    available"), so the text is what matters, not the status.
    """
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
        env=dict(os.environ, LC_ALL="C", SLURM_TIME_FORMAT="standard"),
    )
    return proc.stdout.decode("utf-8", "replace")


def parse_time(text):
    """Slurm timestamp -> epoch int, or None. Slurm prints local time with no
    offset, so it is read in the collecting host's zone."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return int(time.mktime(time.strptime((text or "").strip(), fmt)))
        except ValueError:
            pass
    return None


def parse_duration(text):
    """Slurm duration -> seconds, or None. Handles "88-22:24:58", "02:31:00",
    "05:00". UNLIMITED is None, not infinity: a job with no limit has no ratio,
    and letting it through would silently zero out a partition's usage."""
    match = re.match(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)", (text or "").strip())
    if not match:
        return None
    days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


# --------------------------------------------------------------------------
# probe: what Slurm says right now
# --------------------------------------------------------------------------


def parse_test_only(text, now):
    """One `sbatch --test-only` verdict -> a record.

    A verdict we cannot parse is kept as a failure with its raw text, because
    "Slurm declined to answer" is itself signal: a drained partition or a
    request over a limit otherwise looks identical to an outage.
    """
    line = ([ln.strip() for ln in text.splitlines() if ln.strip()] or [""])[-1]
    match = re.search(r"to start at (\S+?)(?:\s|$)", text)
    start = parse_time(match.group(1)) if match else None
    placed = re.search(r"on nodes (\S+) in partition (\S+)", text)
    return {
        "ok": start is not None,
        "estimated_start": start,
        "estimated_wait_sec": max(0, start - now) if start else None,
        "placed_nodes": placed.group(1) if placed else None,
        "placed_partition": placed.group(2) if placed else None,
        "message": line,
    }


def probe(config):
    """Dry-run every configured shape.

    `--test-only` is the cleanest placement primitive an unprivileged user has:
    Slurm answers with the start time its backfill scheduler would actually
    pick, against the real queue it will not otherwise show us. The answer is
    conditioned on *this account's* priority, which is why the payload records
    who asked.
    """
    now = int(time.time())
    probes = []
    for shape in config["shapes"]:
        args = ["sbatch", "--test-only"] + shape["args"] + ["--wrap", "true"]
        record = parse_test_only(sh(args, timeout=60), now)
        record["name"] = shape["name"]
        record["args"] = " ".join(shape["args"])
        probes.append(record)
    return {
        "cluster": config["cluster"],
        "collected_at": now,
        "collector_version": VERSION,
        "probed_by_user": os.environ.get("USER", ""),
        "probes": probes,
    }


# --------------------------------------------------------------------------
# usage: how much of the requested walltime jobs actually used
# --------------------------------------------------------------------------


def usage(config):
    """One sacct pass over one window ending now, for the configured partitions.

    No cursor and no state file: the window is recomputed from scratch each
    run, so consecutive runs overlap and each row is an independent measurement
    of its own window. That is why the bounds ride along -- rows from different
    runs describe overlapping job sets and must never be summed.
    """
    now = int(time.time())
    start = now - int(config["usage_hours"] * 3600)
    text = sh([
        "sacct", "-X", "-a", "--parsable2",
        "-S", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(start)),
        "-E", time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "-r", ",".join(config["partitions"]),
        "-o", "Partition,State,End,Elapsed,Timelimit",
    ], timeout=600)

    return {
        "cluster": config["cluster"],
        "collected_at": now,
        "collector_version": VERSION,
        "window_start": start,
        "window_end": now,
        "window_hours": config["usage_hours"],
        "partitions": aggregate(text, start, now),
    }


def aggregate(text, window_start, window_end):
    """sacct output -> one usage row per partition.

    Jobs are selected on *end* time. A 24-hour job that started before the
    window and finished inside it is the most informative observation there is,
    since long jobs are the ones whose early exit moves a start-time forecast;
    selecting on start would drop every one of them.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    header = [h.strip() for h in lines[0].split("|")]

    totals = {}
    for line in lines[1:]:
        row = dict(zip(header, [c.strip() for c in line.split("|")]))
        end = parse_time(row.get("End"))
        elapsed = parse_duration(row.get("Elapsed"))
        limit = parse_duration(row.get("Timelimit"))
        state = (row.get("State") or "").split()[0]

        if (state not in TERMINAL or end is None or elapsed is None or not limit
                or not window_start <= end <= window_end):
            continue

        entry = totals.setdefault(
            row.get("Partition") or "unknown",
            {"n_jobs": 0, "n_timeout": 0, "elapsed": 0, "limit": 0},
        )
        entry["n_jobs"] += 1
        entry["n_timeout"] += state == "TIMEOUT"
        entry["elapsed"] += elapsed
        entry["limit"] += limit

    # Time-weighted, because that is what a forecast needs: a thousand
    # two-minute jobs returning 99% of their limit say nothing about the
    # 24-hour job ahead of you in the queue. The sums are kept so the ratio is
    # recomputable and a thin window is visible rather than averaged in.
    return [
        {
            "partition": name,
            "n_jobs": t["n_jobs"],
            "n_timeout": t["n_timeout"],
            "sum_elapsed_sec": t["elapsed"],
            "sum_timelimit_sec": t["limit"],
            "used_ratio": round(min(1.0, t["elapsed"] / t["limit"]), 4),
        }
        for name, t in sorted(totals.items())
    ]


# --------------------------------------------------------------------------


def post(config, kind, payload):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        config["server"].rstrip("/") + "/ingest/" + kind, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    if config.get("token"):
        request.add_header("Authorization", "Bearer " + config["token"])
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status


def main():
    commands = {"probe": probe, "usage": usage}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("usage: slurm_probe.py {probe|usage}")

    config = load_config()
    payload = commands[sys.argv[1]](config)
    if not config.get("server"):
        print(json.dumps(payload, indent=2))
        return
    print(post(config, sys.argv[1], payload))


if __name__ == "__main__":
    main()
