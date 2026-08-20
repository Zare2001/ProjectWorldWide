"""Assemble a `PlannerInputs` from the outside world -- or from a recording of it.

`inputs.py` knows how to READ each source; this module knows which sources exist,
in what order to try them, and what to do when one is missing. The split matters
because the failure modes are the interesting part: the scanner is frequently
unreachable (the upstream deployment does not answer from the aggregator VM at all,
and PWW's own instance on :29513 is not up yet), the DARL coordinator answers only
with a token, and the throughput registry has an entry for the geometry the campaign
ran and none for the geometry a data cap would prefer.

Every resolved value carries a `Provenance` record -- where it came from and how old
it is -- and every value that could NOT be resolved becomes an `Exclusion` naming the
command that fixes it. `pww-plan show` prints both and nothing else, which is the
first thing to run when a plan looks wrong.

THE FIXTURE IS THE SAME SHAPE AS THE WIRE. A fixture file is a dict of endpoint name
-> the payload that endpoint returned, so `--record` is a straight dump and
`--dry-run` re-parses it through the identical code path. A fixture whose format
diverged from the wire would test the parser against itself.

    {"_captured_at": 1787142156,
     "probes": {"rows": [...]},      # GET /probes, every cluster, filtered here
     "usage":  {"rows": [...]},      # GET /usage
     "overview": {"rows": [...]},    # GET /overview -- the fallback when /probes is gone
     "darl":   {"url": ..., "status": {...}},        # GET /status, verbatim
     "throughput": {"snellius": {"4": {...}}},       # stands in for the registry file
     "startup_cost_s": {"snellius": 108.0}}

tests/fixtures/scanner_snapshot.json is already in this shape, which is why it works
as a fixture with no conversion.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .model import (
    Calibration,
    DarlState,
    Exclusion,
    Geometry,
    PlannerInputs,
    ShapeKey,
    SiteInput,
    SiteLimits,
)
from . import inputs as io

# The upstream deployment. NOT the default any more: it does not answer from the
# aggregator VM at all (the connection times out), and neither does the :29513
# instance the design reserved for PWW. The one scanner that is actually up is the
# one configs/slurm_probe/*.json POST to, so that is what `default_scanner_url` reads
# -- from the config rather than from a constant, because the address a probe was
# sent to is the address that has the probe.
UPSTREAM_SCANNER_URL = "http://145.38.185.196:8000"
PWW_SCANNER_URL = "http://145.38.206.143:29513"
DEFAULT_SCANNER_URL = "http://145.38.195.124:8000"
DEFAULT_DARL_URL = "http://145.38.206.143:29510"


def default_scanner_url(probe_config_dir: str | None = "configs/slurm_probe") -> str:
    """The `server` the collector configs post to, or the fallback constant.

    A planner whose default URL is not the URL its own collectors use cannot produce
    a plan from live sources, which is exactly what `scripts/plan_campaign.sh` (which
    passes no --scanner-url) did: it printed SCANNER UNREACHABLE and exited 2.
    """
    if not probe_config_dir:
        return DEFAULT_SCANNER_URL
    servers: list[str] = []
    root = Path(probe_config_dir).expanduser()
    if root.is_dir():
        for path in sorted(root.glob("*.json")):
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            server = payload.get("server")
            if isinstance(server, str) and server.strip() and server not in servers:
                servers.append(server.strip())
    return servers[0] if servers else DEFAULT_SCANNER_URL

# What the collectors sample. Long enough to see a queue turn over, short enough that
# a week-old regime does not set p90 for today.
DEFAULT_PROBE_WINDOW_H = 24.0
DEFAULT_USAGE_WINDOW_H = 720.0


@dataclass(frozen=True)
class Provenance:
    """(value, source, staleness) for one resolved input.

    Printed verbatim by `show` and included in `--json`. `staleness_s` is None when
    the value has no age -- a config file has a path, not an age -- and 0.0 means
    "read just now", which is not the same thing.
    """

    field: str
    value: str
    source: str
    staleness_s: float | None = None
    quality: str = ""

    def describe(self) -> str:
        age = ""
        if self.staleness_s is not None:
            age = f", {self.staleness_s / 60:.0f} min old" if self.staleness_s >= 60 else ", fresh"
        flag = f" [{self.quality}]" if self.quality else ""
        return f"{self.field:<24} {self.value:<34} <- {self.source}{age}{flag}"


@dataclass
class Sources:
    """Where to look. One object so `plan`, `show` and `verify` cannot drift apart."""

    scanner_url: str = DEFAULT_SCANNER_URL
    scanner_data_dir: str | None = None
    fixture: str | None = None
    darl_url: str | None = DEFAULT_DARL_URL
    darl_token: str | None = None
    darl_token_file: str | None = None
    registry: str = "configs/site_throughput.env"
    planner_config: str | None = "configs/plan/federation.json"
    probe_config_dir: str | None = "configs/slurm_probe"
    sites: tuple[str, ...] = ()
    submitter: str | None = None
    blocks: int | None = None
    probe_window_h: float = DEFAULT_PROBE_WINDOW_H
    usage_window_h: float = DEFAULT_USAGE_WINDOW_H
    discount_strength: float = 0.5
    local_batch_size: int = 8
    startup_overrides: dict[str, float] = field(default_factory=dict)
    warm_sites: tuple[str, ...] = ()
    now: float | None = None
    timeout_s: float = 10.0


@dataclass(frozen=True)
class Collected:
    inputs: PlannerInputs
    provenance: tuple[Provenance, ...]
    notes: tuple[str, ...]
    raw: dict[str, Any]  # what a --record would write; also what `verify` diffs


# --------------------------------------------------------------------------
# scanner
# --------------------------------------------------------------------------


def _overview_to_tables(rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    """GET /overview -> the probe and usage rows the rest of the code expects.

    /overview is the endpoint a human reaches for and the only one that joins the
    probe to its partition's used_ratio, but it keeps just the NEWEST row per shape,
    so a wait built from it has one sample and p90 == p50. That is a real loss of
    information -- the whole reason to read a wait as a distribution is that
    --test-only answers for the queue as it is at one instant -- so the caller marks
    such a wait `samples=1` and the report prints it.

    It also renames placed_partition -> partition, which is undone here.
    """
    probes: list[dict] = []
    usage: list[dict] = []
    for row in rows:
        probes.append({
            "collected_at": row.get("collected_at"),
            "name": row.get("shape"),
            "args": row.get("args") or "",
            "ok": bool(row.get("ok")),
            "estimated_start": row.get("estimated_start"),
            "estimated_wait_sec": row.get("estimated_wait_sec"),
            "placed_partition": row.get("partition"),
            "placed_nodes": "",
            "message": row.get("message") or "",
            "probed_by_user": row.get("probed_by_user") or "",
            "collector_version": "",
            "cluster": row.get("cluster"),
        })
        if row.get("used_ratio") is not None:
            usage.append({
                "collected_at": row.get("collected_at"),
                "partition": row.get("partition"),
                "used_ratio": row.get("used_ratio"),
                "n_jobs": row.get("used_ratio_n_jobs"),
                "window_hours": row.get("used_ratio_window_hours"),
                "cluster": row.get("cluster"),
            })
    return probes, usage


def probe_scanner(url: str, *, timeout: float = 10.0) -> tuple[bool, str]:
    """Is the scanner answering? Returns (reachable, one-line explanation).

    Separated from the fetch so the CLI can say WHICH source failed before it starts
    printing exclusions that all trace back to the same dead socket.
    """
    try:
        payload = io.fetch_json(f"{url.rstrip('/')}/healthz", timeout=timeout)
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} from {url}/healthz"
    except Exception as exc:  # URLError, socket.timeout, JSON, ...
        return False, f"{type(exc).__name__}: {exc}"
    clusters = ", ".join(payload.get("clusters") or []) or "(none)"
    return True, f"{url} ok, clusters: {clusters}"


def _scanner_tables(
    src: Sources, notes: list[str], prov: list[Provenance]
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict[str, Any]]:
    """probe rows and usage rows per cluster, from whichever source answers.

    Order is deliberate: a fixture is an explicit instruction and wins; a data dir is
    the path that works today, since the CSVs are on the aggregator's disk while the
    HTTP instance is not up; HTTP is last because it is the one that fails.
    """
    raw: dict[str, Any] = {}

    if src.fixture:
        payload = json.loads(Path(src.fixture).expanduser().read_text())
        raw = payload
        probe_rows = list((payload.get("probes") or {}).get("rows") or [])
        usage_rows = list((payload.get("usage") or {}).get("rows") or [])
        if not probe_rows and payload.get("overview"):
            probe_rows, usage_rows = _overview_to_tables(payload["overview"]["rows"])
            notes.append(
                "fixture has no /probes rows, so the wait came from /overview: one "
                "sample per shape, p90 == p50.")
        captured = payload.get("_captured_at")
        prov.append(Provenance(
            "scanner", f"{len(probe_rows)} probe rows", f"fixture {src.fixture}",
            None if captured is None else max(0.0, (src.now or time.time()) - captured)))
        return _by_cluster(probe_rows), _by_cluster(usage_rows), raw

    if src.scanner_data_dir:
        root = Path(src.scanner_data_dir).expanduser()
        clusters = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
        probes = {c: io.read_csv_table(root, c, "probes") for c in clusters}
        usage = {c: io.read_csv_table(root, c, "usage") for c in clusters}
        prov.append(Provenance(
            "scanner", f"{sum(len(v) for v in probes.values())} probe rows",
            f"csv {root}", None))
        raw = {"probes": {"rows": [r for v in probes.values() for r in v]},
               "usage": {"rows": [r for v in usage.values() for r in v]}}
        return probes, usage, raw

    reachable, why = probe_scanner(src.scanner_url, timeout=src.timeout_s)
    if not reachable:
        configured = default_scanner_url(src.probe_config_dir)
        notes.append(
            f"SCANNER UNREACHABLE: {why}. No queue wait can be read, so no shape can "
            f"be admitted and there is no plan to make. The address "
            f"configs/slurm_probe/*.json actually POST to is {configured} -- try "
            f"--scanner-url {configured}. ({UPSTREAM_SCANNER_URL} is the upstream "
            f"deployment and does not answer from this VM; {PWW_SCANNER_URL} is the "
            f"port reserved for PWW's own instance and nothing is listening on it "
            f"yet.) Otherwise read the CSVs directly with --scanner-data-dir <dir "
            f"containing <cluster>/probes.csv>, or replay a capture with --dry-run "
            f"<fixture.json>.")
        prov.append(Provenance("scanner", "UNREACHABLE", src.scanner_url, None, "missing"))
        return {}, {}, {}

    overview = io.fetch_json(f"{src.scanner_url.rstrip('/')}/overview",
                             timeout=src.timeout_s).get("rows", [])
    raw["overview"] = {"rows": overview}
    clusters = sorted({r.get("cluster") for r in overview if r.get("cluster")})
    probes: dict[str, list[dict]] = {}
    usage: dict[str, list[dict]] = {}
    degraded = False
    for cluster in clusters:
        try:
            probes[cluster] = io.fetch_table(
                src.scanner_url, "probes", cluster, src.probe_window_h)
            usage[cluster] = io.fetch_table(
                src.scanner_url, "usage", cluster, src.usage_window_h)
        except Exception as exc:
            degraded = True
            notes.append(f"{cluster}: /probes or /usage failed ({exc}); falling back to "
                         f"the /overview snapshot, which has one sample per shape")
            p, u = _overview_to_tables([r for r in overview if r.get("cluster") == cluster])
            probes[cluster], usage[cluster] = p, u
    if not degraded:
        raw["probes"] = {"rows": [r for v in probes.values() for r in v]}
        raw["usage"] = {"rows": [r for v in usage.values() for r in v]}
    raw["_captured_at"] = int(src.now or time.time())
    raw["_source"] = src.scanner_url
    prov.append(Provenance(
        "scanner", f"{sum(len(v) for v in probes.values())} probe rows over "
        f"{src.probe_window_h:g} h", src.scanner_url, 0.0))
    return probes, usage, raw


def _by_cluster(rows: Sequence[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(row.get("cluster") or "", []).append(row)
    return out


# --------------------------------------------------------------------------
# DARL
# --------------------------------------------------------------------------


def resolve_darl_token(src: Sources) -> tuple[str | None, str]:
    """$DARL_TOKEN, then --darl-token-file, then the two paths the job scripts try.

    Same order as scripts/{snellius,lumi}/job_titan_diloco.sh, so a token that works
    for a job works for the planner. GETs need it too: /status is 401 without.
    """
    if src.darl_token:
        return src.darl_token, "--darl-token"
    if os.environ.get("DARL_TOKEN"):
        return os.environ["DARL_TOKEN"], "$DARL_TOKEN"
    candidates = []
    if src.darl_token_file:
        candidates.append(Path(src.darl_token_file).expanduser())
    output = os.environ.get("PWW_OUTPUT_DIR")
    if output:
        candidates.append(Path(output) / "darl" / "token")
    root = Path(os.environ.get("PWW_ROOT", ".")).expanduser()
    candidates += [root / "runs" / "darl" / "token", root / "runs" / "central" / "darl" / "token"]
    for path in candidates:
        try:
            token = path.read_text().strip()
        except OSError:
            continue
        if token:
            return token, str(path)
    return None, "(not found)"


def _darl_state(
    src: Sources, calibration: Calibration, notes: list[str], prov: list[Provenance],
    raw: dict[str, Any],
) -> tuple[DarlState, list[Exclusion]]:
    """How much corpus is left, and from where.

    An explicit --blocks wins over everything, because the number a human read off
    /status five minutes ago is a fact and the planner should not argue with it. When
    nothing answers, the fallback is the WHOLE corpus, flagged `assumed` -- it is the
    only number available, it is wrong in the optimistic direction, and the report
    prints the flag next to the exhaustion hour it produced.
    """
    exclusions: list[Exclusion] = []

    # --blocks FIRST, because the docstring above says it wins over everything and a
    # fixture is not an exception: replaying yesterday's capture with today's
    # remaining corpus is the normal way to ask "what should I do now".
    if src.blocks is not None:
        state = DarlState(
            num_blocks=src.blocks, committed=0, leased=0, unassigned=src.blocks,
            block_size=calibration.block_size, source="--blocks")
        prov.append(Provenance("darl.unassigned", f"{src.blocks} blocks", "--blocks", None,
                               "operator-supplied"))
        return state, exclusions

    fixture_status = ((src.fixture and (raw.get("darl") or {}).get("status")) or None)
    if fixture_status:
        try:
            state = _state_from_payload(
                fixture_status, (raw.get("darl") or {}).get("url", "fixture"))
        except (KeyError, TypeError, ValueError) as exc:
            notes.append(
                f"the fixture's darl.status is not a coordinator payload "
                f"({type(exc).__name__}: {exc}); falling through to the whole-corpus "
                f"assumption below. Pass --blocks <n> to pin it.")
        else:
            prov.append(Provenance(
                "darl.unassigned", f"{state.unassigned} blocks", f"fixture {src.fixture}",
                None, "recorded"))
            return state, exclusions

    if src.darl_url and src.darl_url.lower() not in ("none", "off", ""):
        token, token_source = resolve_darl_token(src)
        try:
            payload = io.fetch_json(
                f"{src.darl_url.rstrip('/')}/status",
                headers={"X-DARL-Token": token} if token else {},
                timeout=src.timeout_s)
        except Exception as exc:
            notes.append(
                f"DARL {src.darl_url} did not answer ({type(exc).__name__}: {exc}); "
                f"token from {token_source}. Remaining corpus is UNKNOWN -- pass "
                f"--blocks <n> from GET /status, or accept the whole-corpus assumption "
                f"below. Note the port names the arm: 29510/29520/29530/29540 are four "
                f"different epochs with four different answers.")
        else:
            try:
                state = _state_from_payload(payload, src.darl_url)
            except (KeyError, TypeError, ValueError) as exc:
                # A 200 from something that is not the coordinator. This is the
                # operator error the module docstring warns about -- 29510/29520/
                # 29530/29540 are four DARL epochs, 29511 is Flower and 29513 is the
                # scanner, all on one VM -- and it must read as "wrong port", not as
                # a KeyError traceback with the planner's guts in it.
                notes.append(
                    f"{src.darl_url}/status answered HTTP 200 but the body is not a "
                    f"DARL coordinator status ({type(exc).__name__}: {exc}). That "
                    f"port is almost certainly a different service: 29510/29520/"
                    f"29530/29540 are four DARL epochs, 29511 is the Flower server "
                    f"and 29513 is the scanner. Remaining corpus is UNKNOWN -- pass "
                    f"--blocks <n>, or point --darl-url at the right port.")
                payload = None
            else:
                raw["darl"] = {"url": src.darl_url, "status": payload}
                # The digest check has to actually RUN, or a plan can be priced
                # against a different arm's corpus with nothing saying so. The
                # geometry comes from runs/darl/space.env, which is the record of
                # what THIS stack's coordinator was launched with -- num_samples
                # cannot be recovered from the toml (it is the window count printed
                # by tokenize_c4.sh) and cannot be recovered from the snapshot
                # either, which is why that file exists.
                exclusions.extend(_digest_exclusions(state, notes))
                prov.append(Provenance(
                    "darl.unassigned",
                    f"{state.unassigned} of {state.num_blocks} blocks "
                    f"({state.unassigned * state.block_size * 2049 / 1e9:.2f} G tokens)",
                    f"{src.darl_url}/status", 0.0, "live"))
                live = io.darl_liveness(payload, now=src.now)
                if any(live.values()):
                    notes.append(
                        "DARL says these clusters are LIVE right now: "
                        + ", ".join(sorted(k for k, v in live.items() if v))
                        + ". This plan assumes it is starting from nothing; a site that "
                        "is already running will refuse a second job under the same "
                        "cluster id (503 cluster_busy) unless the new one gets its own "
                        "--replica.")
                elif payload.get("clusters"):
                    notes.append(
                        "DARL lists " + ", ".join(sorted(payload["clusters"]))
                        + " but none has a fresh heartbeat: that map is a durable "
                        "membership RECORD, not a liveness list. Nothing is running.")
                return state, exclusions

    total = 2692
    state = DarlState(
        num_blocks=total, committed=0, leased=0, unassigned=total,
        block_size=calibration.block_size, source="assumed: whole corpus")
    prov.append(Provenance(
        "darl.unassigned", f"{total} blocks (WHOLE CORPUS)", "assumed", None, "assumed"))
    notes.append(
        f"Remaining corpus ASSUMED to be the whole {total}-block epoch. Every "
        f"data-bound conclusion below is optimistic by however much has already been "
        f"committed; today the real figure was 822. Fix with --blocks or a reachable "
        f"--darl-url.")
    return state, exclusions


def _space_env(paths: Sequence[Path] | None = None) -> dict[str, int] | None:
    """NUM_SAMPLES/BLOCK_SIZE/SEED from runs/darl/space.env, in the job scripts' order."""
    if paths is None:
        output = os.environ.get("PWW_OUTPUT_DIR")
        root = Path(os.environ.get("PWW_ROOT", ".")).expanduser()
        paths = ([Path(output) / "darl" / "space.env"] if output else []) + [
            root / "runs" / "darl" / "space.env",
            root / "runs" / "central" / "darl" / "space.env"]
    for path in paths:
        try:
            text = path.read_text()
        except OSError:
            continue
        values: dict[str, int] = {}
        for line in text.splitlines():
            key, _, value = line.strip().partition("=")
            if key in ("NUM_SAMPLES", "BLOCK_SIZE", "SEED"):
                try:
                    values[key] = int(value.strip().strip('"'))
                except ValueError:
                    return None
        if {"NUM_SAMPLES", "BLOCK_SIZE"} <= set(values):
            return values
    return None


def _digest_exclusions(state: DarlState, notes: list[str]) -> list[Exclusion]:
    space = _space_env()
    if space is None:
        notes.append(
            "runs/darl/space.env was not readable, so the coordinator's block digest "
            "could NOT be checked against the corpus this stack was built for. A "
            "mismatch would refuse every site at registration (HTTP 400) and the plan "
            "would describe a run that cannot start.")
        return []
    return io.check_darl_digest(
        state, num_samples=space["NUM_SAMPLES"], block_size=space["BLOCK_SIZE"],
        seed=space.get("SEED", 0))


def _state_from_payload(payload: dict, source: str) -> DarlState:
    return DarlState(
        num_blocks=int(payload["num_blocks"]),
        committed=int(payload.get("committed", 0)),
        leased=int(payload.get("leased", 0)),
        unassigned=int(payload["unassigned"]),
        quarantined=int(payload.get("quarantined", 0)),
        epoch=int(payload.get("epoch", 0)),
        max_epochs=int(payload.get("max_epochs", 1)),
        digest=str(payload.get("digest", "")),
        source=source,
        read_at=time.time(),
        observed=True,  # a coordinator answered; see DarlState.fresh_epoch
    )


# --------------------------------------------------------------------------
# the local files: throughput registry, calibration, collector configs
# --------------------------------------------------------------------------


def _throughput(
    src: Sources, raw: dict[str, Any], prov: list[Provenance]
) -> tuple[dict[str, dict[int, Geometry]], list[Exclusion]]:
    """{site: {devices: Geometry}} from the registry, or from a fixture that carries it.

    A fixture may pin throughput so a dry run is hermetic -- otherwise the answer
    would change whenever someone recalibrated a site, which defeats the point of
    replaying a capture.

    A pinned cell is a hand-written stand-in for a registry line and is read with the
    registry's own guard, not with a bare float(): unchecked, `"tput_seq_s": "89.8
    seq/s"` was a ValueError traceback with zero bytes on stdout, `0` was a
    ZeroDivisionError two modules downstream, and `"batch_seq": -32` was worse than
    either -- a full plan at exit 0 with a negative step time, -1.69 G tokens and -806
    DARL blocks, so the corpus never exhausted and the run was scored to the horizon.
    """
    override = (raw.get("throughput") if src.fixture else None) or {}
    if override:
        out: dict[str, dict[int, Geometry]] = {}
        problems: list[Exclusion] = []
        fix = (f"fix the throughput block of {src.fixture}: it stands in for "
               f"{src.registry}, so the same rule applies -- both halves of a cell "
               f"are positive finite numbers or the cell is not a measurement. Drop "
               f"the block to read the live registry instead.")
        for site, cells in sorted(override.items()):
            for gpus, cell in sorted(cells.items(), key=lambda kv: str(kv[0])):
                subject = f"{site}@{gpus} in fixture {src.fixture}"
                try:
                    devices = int(gpus)
                except (TypeError, ValueError):
                    problems.append(Exclusion(
                        "bad_registry_value", subject,
                        f"the fixture keys a {site} throughput cell on {gpus!r}, which "
                        f"is not a device count",
                        fix))
                    continue
                if not isinstance(cell, dict):
                    problems.append(Exclusion(
                        "bad_registry_value", subject,
                        f"the fixture's {site}@{gpus} cell is {cell!r}, not a "
                        f"tput_seq_s/batch_seq pair; a throughput alone cannot give a "
                        f"step time because sites do not run the same batch per step",
                        fix))
                    continue
                geometry, problem = io.geometry_cell(
                    site, devices, cell.get("tput_seq_s"), cell.get("batch_seq"),
                    source=cell.get("source", f"fixture {src.fixture}"),
                    subject=subject, tput_name=f"{site}@{gpus} tput_seq_s",
                    batch_name=f"{site}@{gpus} batch_seq", fix=fix)
                if problem is not None:
                    problems.append(problem)
                    continue
                out.setdefault(site, {})[devices] = geometry
        prov.append(Provenance(
            "throughput", ", ".join(f"{s}@{g}" for s in sorted(out) for g in sorted(out[s])),
            f"fixture {src.fixture}", None, "recorded"))
        return out, problems

    geometries, exclusions = io.load_throughput(
        src.registry, local_batch_size=src.local_batch_size)
    cells = ", ".join(
        f"{s}@{g} {geometries[s][g].tput_seq_s:g} seq/s"
        for s in sorted(geometries) for g in sorted(geometries[s]))
    prov.append(Provenance("throughput", cells or "(none)", src.registry, None, "measured"))
    return geometries, exclusions


def _startup_costs(
    src: Sources, raw_config: dict[str, Any], prov: list[Provenance]
) -> tuple[dict[str, tuple[float, str, str]], list[Exclusion]]:
    """c per site: {site: (seconds, quality, provenance)}.

    c is an INPUT, not a guess, and it is not measurable from any log in this repo --
    no job script prints a timestamp before torchrun and Slurm writes no start line
    into logs/%x-%j.out. Everything below is therefore a LOWER bound until someone
    instruments the job scripts, which is why the sensitivity pass runs c/2, c and 2c
    on every plan and why the report labels a c-sensitive chain recommendation
    provisional.

    Being an input is also why it is checked: c is a duration from allocation to the
    first training step, so a value that is not a non-negative number is not a
    measurement of anything.
    """
    raw: dict[str, tuple[Any, str, str]] = {}
    for site, entry in (raw_config.get("startup_cost_s") or {}).items():
        if site.startswith("_") or not isinstance(entry, dict):
            continue
        raw[site] = (entry.get("value_s"), entry.get("quality", "lower_bound"),
                     entry.get("provenance", ""))
    for site, seconds in src.startup_overrides.items():
        raw[site] = (seconds, "operator-supplied", "--startup-cost")

    out: dict[str, tuple[float, str, str]] = {}
    problems: list[Exclusion] = []
    for site, (value, quality, why) in raw.items():
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            seconds = float("nan")
        if not (math.isfinite(seconds) and seconds >= 0.0):
            # `startup_cost_s.snellius.value_s: -3600` printed `-3600 s (-60.0 min)` in
            # the provenance block as though it had been measured, and changed the
            # recommendation at exit 0: c pays for every chain link, so a negative one
            # makes links free and the search buys them. A string went the other way,
            # out of float() as a bare ValueError with the report unwritten.
            problems.append(Exclusion(
                "bad_startup_cost", site,
                f"startup cost c for {site} is {value!r}; c is the wall-clock from "
                f"allocation to the first training step, so it is a non-negative "
                f"number of seconds or it is not a measurement",
                f"fix startup_cost_s.{site}.value_s in "
                f"{src.planner_config or 'the planner config'}, or pass "
                f"--startup-cost {site}=<seconds>"))
            continue
        out[site] = (seconds, quality, why)
    for site, (seconds, quality, why) in sorted(out.items()):
        prov.append(Provenance(
            f"startup_cost[{site}]", f"{seconds:.0f} s ({seconds / 60:.1f} min)",
            why or "configs/plan/federation.json", None, quality))
    return out, problems


def _site_limits(raw_config: dict[str, Any], site: str) -> SiteLimits:
    entry = (raw_config.get("site_limits") or {}).get(site) or {}
    return SiteLimits(
        max_submit_jobs=entry.get("max_submit_jobs"),
        max_running_jobs=entry.get("max_running_jobs"),
        max_array_size=entry.get("max_array_size"),
        source=entry.get("source", "assumed"),
    )


def _wanted_shapes(
    src: Sources, site: str, probed: Sequence[ShapeKey]
) -> tuple[tuple[ShapeKey, ...], tuple[tuple[str, str], ...]]:
    """Shapes the collector config asks for but for which no probe row has arrived.

    Not speculation: configs/slurm_probe/<site>.json is the list somebody already
    decided to measure, so a gap between it and the probe table is a broken collector
    or a shape that was added and never restarted -- and it is precisely the set the
    planner must refuse to interpolate over. Anything here becomes an Exclusion that
    names the JSON to paste.

    Returns the keyed gaps AND the (name, why) of entries that cannot be keyed at all.
    Dropping the latter was only safe while build_shapes covered them, and it does not:
    it reports an unparseable entry when a probe ROW carries that name, so an entry
    that is both unkeyable and unprobed fell through both doors. Live, that is
    lumi's 1gpu_{4,8,24,40}h -- `--gpus=1` instead of --gpus-per-node, and no rows --
    which left four configured shapes out of the report entirely while their
    1node_* siblings produced the shape_not_probed the reader could act on.
    """
    if not src.probe_config_dir:
        return (), ()
    path = Path(src.probe_config_dir).expanduser() / f"{site}.json"
    if not path.exists():
        return (), ()
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return (), ()
    wanted: list[ShapeKey] = []
    unkeyable: list[tuple[str, str]] = []
    for shape in payload.get("shapes", []):
        name = str(shape.get("name") or "(unnamed)")
        try:
            key = io.parse_shape_args(site, " ".join(shape.get("args", [])))
        except ValueError as exc:
            unkeyable.append((name, str(exc)))
            continue
        if key not in probed and key not in wanted:
            wanted.append(key)
    return tuple(wanted), tuple(unkeyable)


# --------------------------------------------------------------------------
# the whole thing
# --------------------------------------------------------------------------


def collect(src: Sources) -> Collected:
    """Read every source and assemble a `PlannerInputs`.

    Never raises for a missing source: an absent input is an Exclusion with a fix, and
    a plan built on three of four sources is still worth printing as long as the
    report says which one was missing. The single exception is a malformed fixture,
    which is an operator error rather than an environment one.
    """
    now = src.now if src.now is not None else time.time()
    prov: list[Provenance] = []
    notes: list[str] = []
    exclusions: list[Exclusion] = []

    calibration, calib_problems = io.load_calibration(src.planner_config)
    exclusions.extend(calib_problems)
    raw_config: dict[str, Any] = {}
    if src.planner_config and Path(src.planner_config).expanduser().exists():
        raw_config = json.loads(Path(src.planner_config).expanduser().read_text())
    prov.append(Provenance(
        "calibration", f"merge {calibration.merge.value_s:g} s, "
        f"sites {', '.join(sorted(calibration.sites))}",
        src.planner_config or "built-in table", None, calibration.merge.quality))

    probe_rows, usage_rows, raw = _scanner_tables(src, notes, prov)
    if src.fixture and not src.now:
        # A recording ages. Replaying yesterday's capture against today's clock makes
        # every probe stale and the plan empty, which looks like a planner bug and is
        # not one -- so a fixture is replayed at the instant it was captured unless
        # the caller says otherwise with --now.
        captured = raw.get("_captured_at")
        if captured:
            now = float(captured)
            notes.append(
                f"replaying the fixture at its capture time "
                f"({time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}); pass "
                f"--now to age it instead")

    geometries, geo_problems = _throughput(src, raw, prov)
    exclusions.extend(geo_problems)
    startup, startup_problems = _startup_costs(src, raw_config, prov)
    exclusions.extend(startup_problems)
    darl, darl_problems = _darl_state(src, calibration, notes, prov, raw)
    exclusions.extend(darl_problems)

    clusters = sorted(set(probe_rows) | set(geometries))
    if src.sites:
        wanted = {s.lower() for s in src.sites}
        missing = wanted - set(clusters)
        for site in sorted(missing):
            exclusions.append(Exclusion(
                "site_not_in_sources", site,
                f"--sites named {site!r} but no probe rows and no throughput entry "
                f"exist for it",
                f"check the collector config configs/slurm_probe/{site}.json and "
                f"PWW_TPUT_{site.upper()} in {src.registry}"))
        clusters = [c for c in clusters if c in wanted]

    sites: list[SiteInput] = []
    for cluster in clusters:
        shapes, waits, shape_problems = io.build_shapes(
            cluster, probe_rows.get(cluster, []), usage_rows.get(cluster, []),
            discount_strength=src.discount_strength, now=now)
        exclusions.extend(shape_problems)
        c_s, c_quality, c_why = startup.get(cluster, (None, "", ""))
        if c_s is None:
            exclusions.append(Exclusion(
                "no_startup_cost", cluster,
                f"no startup cost c for {cluster}: how long from allocation to the "
                f"first training step is not known, and every duty-cycle and chaining "
                f"conclusion depends on it",
                f"add \"{cluster}\": {{\"value_s\": <s>, \"quality\": \"lower_bound\"}} to "
                f"startup_cost_s in {src.planner_config}, or pass "
                f"--startup-cost {cluster}=<seconds>. Measure it by putting "
                f"`date +%s > logs/jobstart-${{SLURM_JOB_ID}}` first in the job script "
                f"and differencing against the first [titan] line."))
            continue
        wanted_keys, unkeyable = _wanted_shapes(
            src, cluster, [s.key for s in shapes])
        sites.append(SiteInput(
            site=cluster,
            shapes=shapes,
            waits=waits,
            geometries=geometries.get(cluster, {}),
            startup_s=c_s,
            startup_quality=c_quality,
            submitter=src.submitter,
            limits=_site_limits(raw_config, cluster),
            warm_checkpoint=cluster in src.warm_sites,
            wanted_shapes=wanted_keys,
        ))
        # build_shapes already named the ones a probe row arrived for; re-reporting
        # those would print the same shape twice with two different fixes.
        seen = {e.subject for e in shape_problems if e.code == "unparseable_shape"}
        for name, why in unkeyable:
            if f"{cluster}/{name}" in seen:
                continue
            exclusions.append(Exclusion(
                "unparseable_shape", f"{cluster}/{name}", why,
                f"spell out -p/-N/--gpus-per-node/-t in the {name!r} entry of "
                f"configs/slurm_probe/{cluster}.json. No probe row carries this name "
                f"either, so the collector is not measuring it: an entry it cannot "
                f"parse is one it never submits."))
        probers = sorted({w.probed_by_user for w in waits.values() if w.probed_by_user})
        if not probers and src.submitter:
            # --require-own-probes cannot refuse what it cannot see. GET /overview has
            # no probed_by_user column, so a plan built from the degraded fallback
            # passes the check by omission rather than by satisfying it.
            notes.append(
                f"{cluster}: --require-own-probes is in force but no probe row carries "
                f"probed_by_user, so nothing could be checked. That is the /overview "
                f"fallback (it has no such column), not a clean bill of health.")
        if probers and src.submitter is None:
            notes.append(
                f"{cluster}: probes collected by {', '.join(probers)}. "
                f"`sbatch --test-only` is conditioned on the PROBING account's "
                f"fairshare, QOS and priority, so if that is not the account you "
                f"submit with, these waits describe a different queue. Pass "
                f"--require-own-probes to refuse them instead of warning.")

    if not sites:
        notes.append(
            "no site could be assembled, so there is nothing to plan. The exclusions "
            "above say why, and each carries the command that fixes it.")

    return Collected(
        inputs=PlannerInputs(
            sites=tuple(sites),
            calibration=calibration,
            darl=darl,
            exclusions=tuple(exclusions),
            warnings=tuple(notes),
        ),
        provenance=tuple(prov),
        notes=tuple(notes),
        raw=raw,
    )


def record(collected: Collected, path: str | Path) -> Path:
    """Write what was read as a fixture, so the same plan can be re-derived offline.

    This is how a plan becomes reproducible: the scanner's CSVs are append-only and
    unrotated, but /probes has a window and the coordinator's counters only move
    forward, so the inputs behind a decision are gone within hours unless they are
    written down.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(collected.raw)
    payload.setdefault("_captured_at", int(time.time()))
    payload["_note"] = (
        "Recorded by `pww-plan --record`. Replay with `pww-plan --dry-run <this file>`; "
        "it is replayed at _captured_at so the probe ages match the capture.")
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return path
