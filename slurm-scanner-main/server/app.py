"""Ingest and query server.

Two append-only CSV tables per cluster, under SLURM_SCANNER_DATA_DIR:

    data/<cluster>/probes.csv   one row per shape per probe run
    data/<cluster>/usage.csv    one row per partition per usage run

One directory per cluster so files stay small and dropping a site is `rm -rf`.
Nothing is aggregated on write; the query endpoints filter and the dashboard
draws.

Alongside them sits plan.json, the only configuration this server reads from
disk: per-cluster throughput for /plan. It is required at startup.

    SLURM_SCANNER_DATA_DIR   where the CSVs and plan.json live (default ./data)
    SLURM_SCANNER_TOKENS     comma-separated bearer tokens, one per cluster
"""

import csv
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from server import plan as planner

# expanduser because a literal ~ is not an error: it would become a relative
# directory named "~" under the working directory, and the server would look
# like it worked. Unquoted bash expands it, systemd Environment= does not.
DATA_DIR = Path(os.environ.get("SLURM_SCANNER_DATA_DIR", "./data")).expanduser()
TOKENS = {t for t in os.environ.get("SLURM_SCANNER_TOKENS", "").split(",") if t}
PLAN_CONFIG = planner.load_config(planner.config_path(DATA_DIR))

PROBE_COLUMNS = [
    "collected_at", "name", "args", "ok", "estimated_start",
    "estimated_wait_sec", "placed_partition", "placed_nodes", "message",
    "probed_by_user", "collector_version",
]
USAGE_COLUMNS = [
    "collected_at", "window_start", "window_end", "window_hours", "partition",
    "n_jobs", "n_timeout", "sum_elapsed_sec", "sum_timelimit_sec",
    "used_ratio", "collector_version",
]
TABLES = {"probes": PROBE_COLUMNS, "usage": USAGE_COLUMNS}

# The cluster name becomes a directory name, so it cannot be trusted verbatim.
SAFE_CLUSTER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

app = FastAPI(title="slurm_scanner")


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------


def table_path(cluster, table):
    if not SAFE_CLUSTER.match(cluster):
        raise HTTPException(400, "invalid cluster name %r" % cluster)
    return DATA_DIR / cluster / (table + ".csv")


def to_number(value):
    """CSV holds only text. Timestamps and counts come back as ints rather than
    floats, so a JSON consumer sees 1785921770 and not 1785921770.0."""
    if value in (None, "", "None"):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def append(cluster, table, rows):
    """Append rows, writing the header when the file is new."""
    path = table_path(cluster, table)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = TABLES[table]
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})
    return len(rows)


def read(cluster, table, since=None):
    """Read a table back, newest rows last, optionally cut to a time window."""
    path = table_path(cluster, table)
    if not path.exists():
        return []
    numeric = {c for c in TABLES[table] if c.endswith(("_at", "_sec", "_start",
                                                      "_end", "_jobs", "_timeout",
                                                      "_ratio", "_hours"))}
    out = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            stamp = int(row.get("collected_at") or 0)
            if since and stamp < since:
                continue
            row["cluster"] = cluster
            row["ok"] = row.get("ok") in ("True", "true", "1")
            for key in numeric:
                row[key] = to_number(row.get(key))
            out.append(row)
    return out


def clusters():
    if not DATA_DIR.exists():
        return []
    return sorted(d.name for d in DATA_DIR.iterdir()
                  if d.is_dir() and SAFE_CLUSTER.match(d.name))


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


def authorize(header):
    """Fail closed: an unconfigured server must not silently collect numbers a
    scheduler will act on."""
    if not TOKENS:
        raise HTTPException(503, "no tokens configured on this server")
    token = (header or "").removeprefix("Bearer ").strip()
    if token not in TOKENS:
        raise HTTPException(401, "unknown token")


def ingest(table, payload, authorization):
    authorize(authorization)
    cluster = payload.get("cluster")
    if not cluster:
        raise HTTPException(400, "payload has no cluster")
    key = "probes" if table == "probes" else "partitions"
    items = payload.get(key)
    if not isinstance(items, list):
        raise HTTPException(400, "payload has no %s list" % key)

    shared = {k: v for k, v in payload.items() if not isinstance(v, (list, dict))}
    rows = [dict(shared, **item) for item in items]
    return {"cluster": cluster, "rows": append(cluster, table, rows)}


@app.post("/ingest/probe")
def ingest_probe(payload: dict, authorization: str = Header(None)):
    return ingest("probes", payload, authorization)


@app.post("/ingest/usage")
def ingest_usage(payload: dict, authorization: str = Header(None)):
    return ingest("usage", payload, authorization)


# --------------------------------------------------------------------------
# query
# --------------------------------------------------------------------------


def collect(table, cluster=None, hours=None):
    since = int(time.time() - hours * 3600) if hours else None
    names = [cluster] if cluster else clusters()
    return [row for name in names for row in read(name, table, since)]


@app.get("/healthz")
def healthz():
    return {"ok": True, "clusters": clusters()}


@app.get("/clusters")
def list_clusters():
    out = []
    for name in clusters():
        probes = read(name, "probes")
        usages = read(name, "usage")
        out.append({
            "cluster": name,
            "shapes": sorted({r["name"] for r in probes}),
            "last_probe": max((r["collected_at"] for r in probes), default=None),
            "last_usage": max((r["collected_at"] for r in usages), default=None),
        })
    return {"clusters": out}


@app.get("/probes")
def get_probes(cluster: str = None, hours: float = 168):
    return {"rows": collect("probes", cluster, hours)}


@app.get("/usage")
def get_usage(cluster: str = None, hours: float = 720):
    return {"rows": collect("usage", cluster, hours)}


def newest(rows, key):
    """The most recently collected row per distinct value of `key`."""
    out = {}
    for row in rows:
        current = out.get(row[key])
        if not current or row["collected_at"] > current["collected_at"]:
            out[row[key]] = row
    return out


def latest(cluster):
    """Newest probe per shape and newest usage per partition, for one cluster.

    The shared read behind /overview and /plan, so the dashboard and the
    scheduler can never disagree about what the current numbers are.
    """
    return (newest(read(cluster, "probes"), "name"),
            newest(read(cluster, "usage"), "partition"))


@app.get("/overview")
def overview():
    """Latest estimate per (cluster, shape), with that cluster's usage ratio.

    The two numbers sit side by side and are deliberately not combined: the
    estimate assumes every queued job runs to its full limit, and the ratio
    says how wrong that assumption tends to be -- but how much of the freed
    capacity you actually get depends on what else is queued. Reporting a
    single blended number would hide that. /plan does combine them, under an
    explicit and reported `discount_strength`.
    """
    rows = []
    for name in clusters():
        latest_probes, latest_usage = latest(name)
        for shape, probe in sorted(latest_probes.items()):
            usage = latest_usage.get(probe.get("placed_partition")) or {}
            rows.append({
                "cluster": name,
                "shape": shape,
                "args": probe.get("args"),
                "collected_at": probe["collected_at"],
                "age_sec": int(time.time() - probe["collected_at"]),
                "ok": probe["ok"],
                "estimated_wait_sec": probe.get("estimated_wait_sec"),
                "estimated_start": probe.get("estimated_start"),
                "message": probe.get("message"),
                "partition": probe.get("placed_partition"),
                "used_ratio": usage.get("used_ratio"),
                "used_ratio_n_jobs": usage.get("n_jobs"),
                "used_ratio_window_hours": usage.get("window_hours"),
            })
    return {"rows": rows}


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------


@app.post("/plan")
def make_plan(payload: dict):
    """Split `units` data units into one job per cluster. See server/plan.py.

    Reads nothing but the newest probe and usage rows and writes nothing back:
    the same body twice gives the same answer unless the measurements moved.
    Only `units` is required; every other parameter falls back to the config
    file, and the response echoes the set that was used.
    """
    units = (payload or {}).get("units")
    overrides = {k: v for k, v in (payload or {}).items() if k != "units"}
    return planner.make_plan(units, overrides, latest, PLAN_CONFIG)


@app.get("/")
def dashboard():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
