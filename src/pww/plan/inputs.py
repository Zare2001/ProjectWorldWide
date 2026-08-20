"""The only module in the package that touches the outside world.

Everything here is deliberately dumb: read, parse, coerce, hand back a frozen
dataclass. No decisions, no defaults invented to paper over a missing value -- a
field that cannot be read comes back as an `Exclusion` carrying the command that
fixes it, because a planner that silently substitutes a plausible number produces a
plausible plan.

Stdlib only (urllib, csv, json), so this runs on a login node with nothing installed,
the same constraint that keeps the DARL coordinator stdlib-only.

Three sources, three shapes of failure:

  scanner    GET /probes, /usage on PWW's own instance -- or --scanner-data-dir
             reading probes.csv/usage.csv straight off disk, which is the path that
             works TODAY because 145.38.206.143:29513 is not up yet and the upstream
             deployment at 145.38.185.196:8000 is unreachable from the aggregator VM.
  registry   configs/site_throughput.env, keyed (site, device count).
  DARL       GET /status, which needs X-DARL-Token on GETs as well as POSTs.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

from .model import (
    DERIVED,
    EXTRAPOLATED,
    IDENTIFIED,
    Calibration,
    DarlState,
    Exclusion,
    Geometry,
    MeasuredRegime,
    OverheadEntry,
    Shape,
    ShapeKey,
    SiteOverhead,
    WaitEstimate,
)
from .rounds import DEFAULT_CALIBRATION

# The scanner's own coercion rule (server/app.py:66-72), reproduced so a data-dir read
# and an HTTP read give identical types. A new numeric column named *_sec or *_hours is
# typed automatically; anything else stays a string.
_NUMERIC_SUFFIXES = ("_at", "_sec", "_start", "_end", "_jobs", "_timeout", "_ratio", "_hours")


def _to_number(text: str) -> Any:
    if text in ("", "None"):
        return None
    value = float(text)
    return int(value) if value.is_integer() else value


def _coerce(row: dict[str, str], cluster: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw in row.items():
        if key is None:
            continue  # a row longer than its header: see the schema-extension gotcha
        if key.endswith(_NUMERIC_SUFFIXES):
            try:
                out[key] = _to_number(raw)
            except (TypeError, ValueError):
                # Upstream's read() lets this raise, and one malformed cell then 500s
                # every query for that cluster. A planner that dies on one bad row is
                # worse than one that skips it and says so.
                out[key] = None
                out.setdefault("_bad_cells", []).append(key)
        else:
            out[key] = raw
    out["ok"] = row.get("ok") in ("True", "true", "1")
    out["cluster"] = cluster
    return out


def read_csv_table(data_dir: str | Path, cluster: str, table: str) -> list[dict[str, Any]]:
    """probes.csv / usage.csv straight off disk, coerced like the server would."""
    path = Path(data_dir).expanduser() / cluster / f"{table}.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [_coerce(row, cluster) for row in csv.DictReader(handle)]


def fetch_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 15.0) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def fetch_table(scanner_url: str, table: str, cluster: str, hours: float) -> list[dict[str, Any]]:
    """GET /probes or /usage.

    `hours=0` is never sent: the server's `if hours` treats 0 as falsy and returns the
    ENTIRE history rather than nothing, which is a convenient way to dump everything
    and a lethal way to ask for 'now'.
    """
    hours = hours if hours > 0 else 1e-6
    url = f"{scanner_url.rstrip('/')}/{table}?cluster={cluster}&hours={hours}"
    return fetch_json(url).get("rows", [])


# --------------------------------------------------------------------------
# shapes: (partition, devices, walltime) parsed out of the probe's own args
# --------------------------------------------------------------------------

_TIME = re.compile(r"^(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?$")


def parse_slurm_time(text: str) -> int:
    """Slurm duration syntax to seconds.

    All seven accepted forms, because the walltime is part of the shape KEY and a
    mis-parse here silently keys a wait to the wrong shape -- the precise failure the
    verbatim-args rule exists to make impossible downstream.
    """
    text = text.strip()
    if text.lower() in ("infinite", "unlimited"):
        raise ValueError("an unlimited walltime cannot be a shape key")
    match = _TIME.match(text)
    if not match:
        raise ValueError(f"unparseable Slurm duration {text!r}")
    days, a, b, c = match.groups()
    days = int(days or 0)
    if c is not None:  # [d-]HH:MM:SS
        h, m, s = int(a), int(b), int(c)
    elif b is not None:  # HH:MM when days present, else MM:SS
        h, m, s = (int(a), int(b), 0) if days else (0, int(a), int(b))
    else:  # [d-]HH when days present, else MM
        h, m, s = (int(a), 0, 0) if days else (0, int(a), 0)
    return days * 86400 + h * 3600 + m * 60 + s


def parse_shape_args(cluster: str, args: str) -> ShapeKey:
    """(-p, -N x --gpus-per-node, -t) out of the raw argument string.

    Nothing upstream parses this string -- not the collector, not the server, not the
    upstream planner, which is exactly why that planner quotes a wait measured for an
    8 h job and then tells you to submit a 12.4 h one. The device count is -N x
    --gpus-per-node with -N defaulting to 1; --cpus-per-task and --mem are NOT parsed
    because the emitter copies the whole string verbatim, so they travel with
    --gpus-per-node whether or not this function understands them.
    """
    tokens = args.split()
    flags: dict[str, str] = {}
    for idx, token in enumerate(tokens):
        if token.startswith("-") and idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
            flags[token] = tokens[idx + 1]
        elif "=" in token and token.startswith("--"):
            key, _, value = token.partition("=")
            flags[key] = value
        elif len(token) > 2 and token.startswith("-") and not token.startswith("--"):
            # Slurm accepts an attached value on a short option: -N2, -pgpu_h100,
            # -t8:00:00 are all valid and all appear in hand-written probe configs.
            # Unparsed, `-N2` fell through to the nodes=1 default and HALVED the device
            # count -- so the wait measured for a 2-node job was priced against the
            # 1-node throughput cell, silently, with nothing on the emitted line to
            # show for it (the emitter copies the string verbatim, so the JOB would
            # still be 2 nodes). Short options are single-letter, hence token[:2].
            flags[token[:2]] = token[2:]
    partition = flags.get("-p") or flags.get("--partition")
    walltime = flags.get("-t") or flags.get("--time")
    gpus_per_node = flags.get("--gpus-per-node")
    if not partition or not walltime or not gpus_per_node:
        raise ValueError(
            f"{cluster}: args {args!r} do not spell out -p, -t and --gpus-per-node, so "
            f"the shape key cannot be recovered. Add them to the collector config: the "
            f"shape NAME carries the walltime only by convention and nothing enforces it."
        )
    return ShapeKey(
        cluster=cluster,
        partition=partition,
        nodes=int(flags.get("-N") or flags.get("--nodes") or 1),
        gpus_per_node=int(gpus_per_node),
        walltime_s=parse_slurm_time(walltime),
        account=flags.get("-A") or flags.get("--account"),
    )


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. Deterministic and dependency-free; `statistics`
    would do for p50 but not for an arbitrary q on a short sample."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def build_shapes(
    cluster: str,
    probe_rows: Iterable[dict[str, Any]],
    usage_rows: Iterable[dict[str, Any]],
    *,
    discount_strength: float = 0.5,
    now: float | None = None,
) -> tuple[tuple[Shape, ...], dict[str, WaitEstimate], list[Exclusion]]:
    """Probe history -> one Shape and one WaitEstimate per shape name.

    The wait is a DISTRIBUTION over the window, not the newest scalar. `--test-only`
    answers for the queue as it is now and assumes every running job runs to its full
    walltime; the newest single reading is a 10-minute snapshot of a number whose
    spread is the whole reason p90 exists.
    """
    now = time.time() if now is None else now
    rows = sorted(probe_rows, key=lambda r: (r.get("name") or "", r.get("collected_at") or 0))
    usage_by_partition: dict[str, dict[str, Any]] = {}
    for row in usage_rows:
        part = row.get("partition")
        best = usage_by_partition.get(part)
        if best is None or (row.get("collected_at") or 0) > (best.get("collected_at") or 0):
            usage_by_partition[part] = row

    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(row.get("name") or "", []).append(row)

    shapes: list[Shape] = []
    waits: dict[str, WaitEstimate] = {}
    exclusions: list[Exclusion] = []
    for name, history in sorted(by_name.items()):
        newest = history[-1]
        try:
            key = parse_shape_args(cluster, newest.get("args") or "")
        except ValueError as exc:
            exclusions.append(Exclusion(
                "unparseable_shape", f"{cluster}/{name}", str(exc),
                f"spell out -p/-N/--gpus-per-node/-t in the {name!r} entry of "
                f"configs/slurm_probe/{cluster}.json"))
            continue
        # A shape is (partition, devices, WALLTIME), not a name. Editing the -t of an
        # entry in configs/slurm_probe/<site>.json -- which is exactly what this
        # planner's own shape_not_probed exclusions tell the operator to do -- keeps
        # the name and changes the shape, and grouping on the name alone then blends
        # waits measured at the old walltime into the new one's distribution. The
        # emitter would go on to copy the NEW args verbatim, so the plan would quote a
        # wait measured at 1 h and submit a 40 h job: precisely the failure the
        # verbatim-args rule exists to prevent, entering through the front door.
        args_now = newest.get("args") or ""
        history = [r for r in history if (r.get("args") or "") == args_now]
        dropped = len(by_name[name]) - len(history)
        if dropped:
            exclusions.append(Exclusion(
                "shape_args_changed", f"{cluster}/{name}",
                f"{dropped} of {len(by_name[name])} probe rows for {name!r} were "
                f"collected with different sbatch args and were DISCARDED, not blended: "
                f"the wait below describes {args_now!r} only",
                f"nothing to fix if you edited configs/slurm_probe/{cluster}.json on "
                f"purpose -- the distribution refills after one collector window. If "
                f"you want both shapes, give them different names."))
        samples: list[float] = []
        unreadable: list[str] = []
        for row in history:
            if not row.get("ok") or row.get("estimated_wait_sec") is None:
                continue
            try:
                sample = float(row["estimated_wait_sec"])
            except (TypeError, ValueError):
                # A cell that is PRESENT but not a number is the same failure the CSV
                # path already names `_bad_cells`, and it must reach the same
                # `no_wait_reading` refusal. Left to float() it was a bare ValueError
                # out of the CLI: exit 1, zero bytes on stdout, no plan, no exclusion
                # and no indication which cluster or shape was malformed. Rows that DO
                # parse still price the shape; only a shape with none left is refused.
                unreadable.append(f"estimated_wait_sec={row['estimated_wait_sec']!r}")
                continue
            if not (math.isfinite(sample) and sample >= 0.0):
                # Parsing as a float is not the same as being a reading, and the three
                # values that clear float() but are not readings each failed in their
                # own way: 'nan' reached search._links_to_cover as `cannot convert
                # float NaN to integer` (exit 1, nothing on stdout, no exclusion),
                # 'inf' made the shape unschedulable in silence, and a negative priced
                # h100_full_8h as having started ten hours ago -- exit 0, a plan that
                # gained a chain link and CHANGED the recommendation. The collector
                # writes max(0, start - now) (collector/slurm_probe.py:105), so none
                # of the three can come from a probe: it is an edited cell.
                unreadable.append(f"estimated_wait_sec={row['estimated_wait_sec']!r}")
                continue
            samples.append(sample)
        if not samples:
            # NEVER 0.0. A zero here reads as "starts immediately", which is the most
            # optimistic answer the planner can give and the one it has least right
            # to: it means the cell was missing or unparseable, not that the queue is
            # empty. Upstream's own planner skips such a row (server/plan.py:216);
            # substituting zero moves a site to the front of the plan for free.
            bad = sorted({c for r in history for c in (r.get("_bad_cells") or [])}
                         | set(unreadable))
            exclusions.append(Exclusion(
                "no_wait_reading", f"{cluster}/{name}",
                f"no usable estimated_wait_sec in {len(history)} probe row(s) for "
                f"{name!r}"
                + (f"; unreadable numeric cells: {', '.join(bad)}" if bad else "")
                + ". A missing wait is not a zero wait, so this shape is refused "
                  "rather than priced as starting immediately",
                f"check the collector on the {cluster} login node: a --test-only that "
                f"errors posts ok=false with the message, while an empty cell means "
                f"the row was written malformed. One collector cycle fixes it."))
            continue
        used_ratio = None
        usage = usage_by_partition.get(newest.get("placed_partition"))
        if usage is not None:
            used_ratio = usage.get("used_ratio")
        p50, p90 = _percentile(samples, 0.5), _percentile(samples, 0.9)
        wait = WaitEstimate(
            p50_raw_s=p50, p90_raw_s=p90,
            p50_eff_s=_discount(p50, used_ratio, discount_strength),
            p90_eff_s=_discount(p90, used_ratio, discount_strength),
            samples=len(samples),
            probe_age_s=max(0.0, now - float(newest.get("collected_at") or 0)),
            ok=bool(newest.get("ok")),
            used_ratio=used_ratio,
            discount_strength=discount_strength,
            probed_by_user=newest.get("probed_by_user") or None,
            message=newest.get("message") or "",
        )
        shapes.append(Shape(name=name, key=key, args=newest.get("args") or ""))
        waits[name] = wait
    return tuple(shapes), waits, exclusions


def _discount(wait_s: float, used_ratio: float | None, strength: float) -> float:
    """server/plan.py:165-179, verbatim. A no-op when used_ratio is None, <= 0 or > 1."""
    if used_ratio is None or not 0.0 < float(used_ratio) <= 1.0:
        return float(wait_s)
    return float(wait_s) * (1.0 - strength * (1.0 - float(used_ratio)))


# --------------------------------------------------------------------------
# throughput registry, keyed (site, device count)
# --------------------------------------------------------------------------

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def load_throughput(
    path: str | Path, *, local_batch_size: int = 8
) -> tuple[dict[str, dict[int, Geometry]], list[Exclusion]]:
    """configs/site_throughput.env -> {site: {devices: Geometry}}.

    Two accepted spellings. PWW_TPUT_<SITE>/PWW_BATCH_<SITE> is the site's REFERENCE
    geometry, and its device count is recovered as batch/local_batch_size -- the file's
    own definition of PWW_BATCH_ is `dp_ranks x training.local_batch_size`, so this is
    a reading of the registry, not an inference about it. PWW_TPUT_<SITE>_<N> pins the
    device count explicitly, which is what a reduced-geometry plan needs.

    A geometry that is not in the file is NOT interpolated from one that is. Step time
    is device-count invariant to within 8% on the measured pairs, but that is a finding
    about two sites on one model, not a licence.
    """
    path = Path(path).expanduser()
    exclusions: list[Exclusion] = []
    if not path.exists():
        return {}, [Exclusion(
            "no_registry", str(path), f"{path} is not readable",
            "run scripts/titan/calibrate_throughput.sh --write on a job log from each site")]
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _ENV_LINE.match(line)
        if match:
            values[match.group(1)] = match.group(2).strip().strip('"').strip("'")

    tput = {k[len("PWW_TPUT_"):]: v for k, v in values.items() if k.startswith("PWW_TPUT_")}
    out: dict[str, dict[int, Geometry]] = {}
    for suffix, raw_tput in sorted(tput.items()):
        raw_batch = values.get(f"PWW_BATCH_{suffix}")
        if raw_batch is None:
            exclusions.append(Exclusion(
                "half_calibrated", f"PWW_TPUT_{suffix}",
                f"PWW_TPUT_{suffix} has no matching PWW_BATCH_{suffix}; throughput alone "
                f"cannot give a step time because sites do not run the same batch per step",
                "re-run scripts/titan/calibrate_throughput.sh --write"))
            continue
        site, _, tail = suffix.rpartition("_")
        if site and tail.isdigit():
            gpus: int | None = int(tail)
        else:
            site, gpus = suffix, None
        geometry, problem = geometry_cell(
            site.lower().replace("_", "-"), gpus, raw_tput, raw_batch,
            source=str(path), subject=f"PWW_TPUT_{suffix}",
            tput_name=f"PWW_TPUT_{suffix}", batch_name=f"PWW_BATCH_{suffix}",
            local_batch_size=local_batch_size,
            fix=f"fix {path} by hand, or re-run "
                f"scripts/titan/calibrate_throughput.sh --write on a job log from "
                f"{site.lower()}")
        if problem is not None:
            exclusions.append(problem)
            continue
        out.setdefault(geometry.site, {})[geometry.gpus] = geometry
    return out, exclusions


def geometry_cell(
    site: str,
    gpus: int | None,
    raw_tput: Any,
    raw_batch: Any,
    *,
    source: str,
    subject: str,
    tput_name: str,
    batch_name: str,
    fix: str,
    local_batch_size: int = 8,
) -> tuple[Geometry | None, Exclusion | None]:
    """One (site, device count) cell, or the Exclusion that refuses it.

    Shared with adapter._throughput rather than kept private to the registry reader,
    because a fixture may pin the same pair and the guard below was on the registry
    door only: a hand-edited `"batch_seq": -32` in a fixture went straight into a
    Geometry and produced a full plan at exit 0 with negative tokens and negative
    DARL blocks -- the same defect this function refuses one caller above, arriving
    through the other door.

    `gpus=None` means recover the device count from batch/local_batch_size, which the
    registry's own definition of PWW_BATCH_ (`dp_ranks x training.local_batch_size`)
    licenses; a caller that knows the count passes it.
    """
    try:
        tps, batch = float(raw_tput), int(raw_batch)
    except (TypeError, ValueError):
        return None, Exclusion(
            "bad_registry_value", subject,
            f"{tput_name}={raw_tput!r} / {batch_name}={raw_batch!r} are not numbers",
            fix)
    # Both halves must be POSITIVE AND FINITE, not merely numeric. Geometry.step_s is
    # batch/tput, so a zero in either is a ZeroDivisionError traceback out of the CLI
    # with no report at all; a negative one is worse, yielding a negative step time
    # and a plausible-looking plan in which the slow site is the fast one, priced to
    # negative tokens and negative DARL blocks at exit 0 -- the corpus never exhausts,
    # so the run is scored all the way to the horizon. An infinity passes `> 0` and
    # divides one layer later instead: step_s 0.0 reached rounds.balance_accums as a
    # ZeroDivisionError, again with zero bytes on stdout. float() reads 'inf', 'nan'
    # and any 1e400 overflow, so all three come from one typo in the same file and
    # all three are refused by name, exactly as rounds.site_overhead_s already
    # refuses the throughput half.
    bad = [n for n, v in ((tput_name, tps), (batch_name, batch))
           if not (v > 0 and math.isfinite(v))]
    if bad:
        return None, Exclusion(
            "bad_registry_value", subject,
            f"{' and '.join(bad)} is not positive and finite "
            f"({tput_name}={tps!r}, {batch_name}={batch!r}); a step time is "
            f"batch/throughput, so zero divides, an infinity gives a zero step time "
            f"that divides one layer later, and a negative one prices the slow site "
            f"as the fast one",
            fix)
    if gpus is None:
        gpus = max(1, batch // max(1, local_batch_size))
    elif gpus <= 0:
        return None, Exclusion(
            "bad_registry_value", subject,
            f"{subject} is keyed to {gpus} devices; a cell is keyed on the device "
            f"count it was measured at, and nothing runs on {gpus}",
            fix)
    return Geometry(
        site=site, gpus=gpus, tput_seq_s=tps, batch_seq=batch, source=source), None


# --------------------------------------------------------------------------
# DARL: how much corpus is actually left
# --------------------------------------------------------------------------


def fetch_darl_status(url: str, token: str | None = None, *, timeout: float = 15.0) -> DarlState:
    """GET /status.

    The token is required on GETs too -- verified, 401 without it. The port is carried
    into the result and printed, because 29510/29520/29530/29540 are four different
    arms with four different epochs and four different answers to 'how much is left',
    and none of them says which one you asked.
    """
    headers = {"X-DARL-Token": token} if token else {}
    payload = fetch_json(f"{url.rstrip('/')}/status", headers=headers, timeout=timeout)
    return DarlState(
        num_blocks=int(payload["num_blocks"]),
        committed=int(payload["committed"]),
        leased=int(payload["leased"]),
        unassigned=int(payload["unassigned"]),
        quarantined=int(payload.get("quarantined", 0)),
        epoch=int(payload.get("epoch", 0)),
        max_epochs=int(payload.get("max_epochs", 1)),
        digest=str(payload.get("digest", "")),
        source=url,
        read_at=time.time(),
        observed=True,  # a coordinator answered; see DarlState.fresh_epoch
    )


def darl_liveness(status_payload: dict[str, Any], *, fresh_within_s: float = 300.0,
                  now: float | None = None) -> dict[str, bool]:
    """Which clusters are actually RUNNING, from a raw /status payload.

    The `clusters` map is a durable membership RECORD, not a liveness list: a
    coordinator resumed from its snapshot lists every cluster that ever registered,
    with a stale last_seen. scripts/follow_watch.sh was burned by exactly this and now
    gates on freshness; so does this.
    """
    now = time.time() if now is None else now
    out: dict[str, bool] = {}
    for name, record in (status_payload.get("clusters") or {}).items():
        last_seen = float(record.get("last_seen") or 0.0)
        out[name] = (now - last_seen) < fresh_within_s
    return out


def check_darl_digest(state: DarlState, *, num_samples: int, block_size: int, seed: int = 0) -> list[Exclusion]:
    """Refuse to plan against a coordinator that disagrees with the config.

    BlockSpace.digest is checked at registration, so a mismatch is not a warning: every
    site would be refused with HTTP 400 and the plan would describe a run that cannot
    start.
    """
    from ..darl.space import BlockSpace

    expected = BlockSpace(num_samples=num_samples, block_size=block_size, seed=seed)
    if state.digest and expected.digest(state.epoch) != state.digest:
        return [Exclusion(
            "darl_digest_mismatch", state.source,
            f"coordinator digest {state.digest} != {expected.digest(state.epoch)} for "
            f"num_samples={num_samples} block_size={block_size} seed={seed} epoch={state.epoch}",
            "check runs/darl/space.env: num_samples, block_size and seed cannot be "
            "recovered from the snapshot and every site is refused at registration if "
            "they disagree")]
    if state.num_blocks != expected.num_blocks:
        return [Exclusion(
            "darl_geometry_mismatch", state.source,
            f"coordinator has {state.num_blocks} blocks, the config implies "
            f"{expected.num_blocks}",
            "you are probably talking to a different arm's coordinator; check the port")]
    return []


# --------------------------------------------------------------------------
# calibration override
# --------------------------------------------------------------------------


def load_calibration(path: str | Path | None) -> tuple[Calibration, list[Exclusion]]:
    """configs/plan/federation.json over the built-in table.

    The defaults live in code, not in the file: rounds.residuals() has to be able to
    check the model against the measured regimes without reading anything, because a
    self-check that depends on an editable config checks nothing. The file overrides,
    and every entry it overrides must carry its own quality flag -- an unlabelled
    number is refused rather than assumed identified.
    """
    if not path:
        return DEFAULT_CALIBRATION, []
    p = Path(path).expanduser()
    if not p.exists():
        return DEFAULT_CALIBRATION, [Exclusion(
            "no_calibration_file", str(p), f"{p} not found; using the built-in table",
            "write it, or drop --planner-config")]
    payload = json.loads(p.read_text())
    problems: list[Exclusion] = []

    def entry(raw: Any, label: str) -> OverheadEntry | None:
        raw = raw if isinstance(raw, dict) else {}
        quality = raw.get("quality")
        if quality not in (IDENTIFIED, DERIVED, EXTRAPOLATED):
            problems.append(Exclusion(
                "unlabelled_calibration", label,
                f"{label} has quality {quality!r}; it must be one of "
                f"{IDENTIFIED}/{DERIVED}/{EXTRAPOLATED}",
                "say how the number was obtained; the report prints the flag beside "
                "every period derived from it"))
            quality = EXTRAPOLATED
        try:
            value = float(raw.get("value_s"))
        except (TypeError, ValueError):
            value = float("nan")
        if not (math.isfinite(value) and value >= 0.0):
            # An overhead is seconds of transport and evaluate at the barrier, so a
            # negative one is a round that costs less than its own inner phase: with
            # snellius.xfer at -33.5 the plan came out at exit 0 recommending a
            # different geometry entirely, with nothing in the report to say why. A
            # string was the other failure -- float() raising out of the CLI, no plan
            # and no exclusion. Refused rather than substituted, and the exclusion
            # says the file's value is NOT what the plan was priced with.
            problems.append(Exclusion(
                "bad_calibration_value", label,
                f"{label} is {raw.get('value_s')!r}; an overhead is a non-negative "
                f"number of seconds, so this entry was refused and the plan is not "
                f"priced with it",
                f"fix {label}.value_s in the file given to --planner-config, or drop "
                f"the entry to fall back on the built-in table deliberately"))
            return None
        return OverheadEntry(value, quality, raw.get("provenance", ""))

    sites = dict(DEFAULT_CALIBRATION.sites)
    for site, raw in (payload.get("sites") or {}).items():
        xfer = entry((raw or {}).get("xfer"), f"{site}.xfer")
        eval_fix = entry((raw or {}).get("eval_fix"), f"{site}.eval_fix")
        if xfer is None or eval_fix is None:
            continue  # half a site's overhead is not a site's overhead
        sites[site] = SiteOverhead(xfer=xfer, eval_fix=eval_fix)
    regimes = tuple(
        MeasuredRegime(
            label=r["label"],
            members=tuple((m[0], int(m[1])) for m in r["members"]),
            inner_steps=int(r["inner_steps"]),
            accums=tuple(int(a) for a in r["accums"]),
            measured_period_s=float(r["measured_period_s"]),
            quality=r.get("quality", EXTRAPOLATED),
            tolerance=float(r.get("tolerance", 0.05)),
            note=r.get("note", ""),
        )
        for r in payload.get("regimes", [])
    ) or DEFAULT_CALIBRATION.regimes
    merge = entry(payload["merge"], "merge") if "merge" in payload else None
    if merge is None:
        merge = DEFAULT_CALIBRATION.merge
    return (
        Calibration(
            merge=merge,
            sites=sites,
            regimes=regimes,
            val_windows=int(payload.get("val_windows", DEFAULT_CALIBRATION.val_windows)),
            tau_stall_s=float(payload.get("tau_stall_s", DEFAULT_CALIBRATION.tau_stall_s)),
            seq_len=int(payload.get("seq_len", DEFAULT_CALIBRATION.seq_len)),
            block_size=int(payload.get("block_size", DEFAULT_CALIBRATION.block_size)),
        ),
        problems,
    )
