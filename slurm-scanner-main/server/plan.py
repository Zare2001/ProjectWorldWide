"""Turn a number of data units into one job per cluster.

A waterfill on a completion horizon: by horizon T a cluster can process
(T - its expected wait) hours of work, capped by its walltime limit, at whatever
rate its smallest submittable allocation runs. Bisect T until the clusters
together cover the request. A cluster too busy to reach the horizon gets
nothing, which is what makes this a waterfill rather than a proportional split.

Allocation width is configured per shape. Each cluster declares its GPU speed
(`units_per_gpu_per_hour`), its available shapes and their GPU counts in
`shapes`, and the one `shape` it submits. Throughput is derived as
`units_per_gpu_per_hour * shapes[shape]`. Plans then scale by walltime only.

Three knobs shape the result, each defaulted in the config file and overridable
per request:

    discount_strength   how much of used_ratio to believe against the estimate
    min_units           a share under this moves to the clusters that stay
    reserve_units       spare capacity every job carries, in data units

Every cluster carries the reserve, and each converts it at its own rate:
`reserve_units / units_per_hour` hours. Ten units is an hour at 10 units/h and
100 seconds at 360 units/h -- the same spare capacity everywhere, bought with the
hours it actually costs that cluster, so a fast site is not asked to sit on
walltime it does not need to absorb the same work.

The config is `plan.json` in the data directory, read once at startup and
required: a server with no file refuses to start rather than plan against
assumed numbers. Declare `"clusters": []` to run without planning. It sits
alongside the per-cluster directories and is skipped by the cluster scan, which
only looks at subdirectories.
"""

import json
import os
import time
from pathlib import Path

from fastapi import HTTPException

# Every request parameter except `units` has a default, so the smallest useful
# call is {"units": N}. These are the floors; the config file overrides them and
# a request overrides that.
DEFAULTS = {
    "reserve_units": 0.0,
    "min_units": 0,
    "discount_strength": 0.5,
    "max_probe_age_hours": 6.0,
    "max_gpus": 0,
}

CLUSTER_REQUIRED = ("cluster", "units_per_gpu_per_hour", "shapes", "max_walltime_hours")


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def config_path(data_dir):
    """`plan.json` next to the data, so nothing has to be set for the planner to
    find its config and the answer does not depend on the working directory.
    SLURM_SCANNER_PLAN_CONFIG overrides it for deployments that keep config and
    data apart."""
    override = os.environ.get("SLURM_SCANNER_PLAN_CONFIG")
    path = Path(override) if override else Path(data_dir) / "plan.json"
    return path.expanduser()      # a literal ~ is a directory name, not an error


def load_config(path):
    """Parse and validate the config. Raises unless it is complete and sane.

    Called at startup, so anything wrong here stops the server instead of
    surfacing as a bad plan later. That includes the file being absent: a
    scheduler that silently has no clusters configured is worse than one that
    will not boot, so running without planning is spelled `"clusters": []`.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(
            "%s: no plan config. Copy plan.example.json there and set the "
            "per-cluster throughput, or give it \"clusters\": [] to run without "
            "planning." % path)

    config = json.loads(path.read_text())
    defaults = dict(DEFAULTS, **config.get("defaults", {}))
    check_params(defaults, "%s: defaults" % path)

    clusters = config.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError("%s: needs a 'clusters' list, empty if you plan nowhere"
                         % path)

    seen = set()
    for entry in clusters:
        missing = [k for k in CLUSTER_REQUIRED if entry.get(k) in (None, "")]
        if missing:
            raise ValueError("%s: cluster entry %r lacks %s"
                             % (path, entry.get("cluster"), ", ".join(missing)))
        for key in ("units_per_gpu_per_hour", "max_walltime_hours"):
            if not float(entry[key]) > 0:
                raise ValueError("%s: %s of %r must be positive"
                                 % (path, key, entry["cluster"]))
        if not isinstance(entry["shapes"], dict) or not entry["shapes"]:
            raise ValueError("%s: 'shapes' of %r must be a non-empty dict"
                             % (path, entry["cluster"]))
        for shape_name, gpus in entry["shapes"].items():
            if not (isinstance(gpus, (int, float)) and not isinstance(gpus, bool) and gpus > 0):
                raise ValueError("%s: GPU count for shape %r in %r must be positive"
                                 % (path, shape_name, entry["cluster"]))
        if entry.get("shape") is not None and entry["shape"] not in entry["shapes"]:
            raise ValueError("%s: shape %r not found in 'shapes' for %r"
                             % (path, entry["shape"], entry["cluster"]))
        if entry.get("min_units") is not None and int(entry["min_units"]) < 0:
            raise ValueError("%s: min_units of %r is negative" % (path, entry["cluster"]))
        if entry["cluster"] in seen:
            raise ValueError("%s: %r listed twice" % (path, entry["cluster"]))
        seen.add(entry["cluster"])

    return {"defaults": defaults, "clusters": clusters, "path": str(path)}


def check_params(values, where):
    """Bounds that make the waterfill meaningful, checked the same way for the
    config file and for a request override."""
    if not 0 <= values["discount_strength"] <= 1:
        raise ValueError("%s: discount_strength must be between 0 and 1" % where)
    if values["reserve_units"] < 0:
        raise ValueError("%s: reserve_units must not be negative" % where)
    if values["min_units"] < 0:
        raise ValueError("%s: min_units must not be negative" % where)
    if not values["max_probe_age_hours"] > 0:
        raise ValueError("%s: max_probe_age_hours must be positive" % where)
    if values.get("max_gpus") is not None and values["max_gpus"] < 0:
        raise ValueError("%s: max_gpus must not be negative" % where)


def merge_params(config, overrides):
    """Config defaults, then this request's overrides.

    An unknown key is an error rather than a silent no-op: a misspelled
    `discount_strenght` would otherwise return a plan the caller believes is
    discounted and is not.
    """
    out = dict(config["defaults"])
    for key, value in (overrides or {}).items():
        if key not in out:
            raise HTTPException(400, "unknown parameter %r (known: %s)"
                                % (key, ", ".join(sorted(out))))
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise HTTPException(400, "parameter %r must be a number" % key)
        out[key] = value
    try:
        check_params(out, "request")
    except ValueError as error:
        raise HTTPException(400, str(error))
    return out


# --------------------------------------------------------------------------
# the numbers per cluster
# --------------------------------------------------------------------------


def discount(wait_sec, used_ratio, strength):
    """Blend the two numbers this project otherwise keeps apart.

    `--test-only` assumes every queued job occupies its full requested walltime;
    used_ratio measures how wrong that assumption is in aggregate. How much of
    the handed-back capacity *you* get depends on what else is queued, so the
    two are not simply multiplied: `strength` says how much of the correction to
    believe -- 0 trusts the pessimistic estimate, 1 applies the ratio in full,
    and the result is the weighted average of those two waits. Both the raw and
    the effective wait stay in the response, so the choice is visible rather
    than baked in.
    """
    if used_ratio is None or not 0 < used_ratio <= 1:
        return float(wait_sec)
    return float(wait_sec) * (1 - strength) + float(wait_sec) * used_ratio * strength


def candidates(config, params, latest):
    """One site per configured cluster, evaluating available shapes.

    `latest(cluster)` returns (newest probe per shape, newest usage per
    partition) -- the storage layer stays in app.py.
    """
    sites, excluded = [], []
    now = time.time()
    max_gpus = params.get("max_gpus") or 0

    for entry in config["clusters"]:
        cluster = entry["cluster"]
        probes, usage = latest(cluster)

        # If a single shape is pinned in cluster entry, use it; otherwise evaluate all shapes in entry["shapes"]
        shape_keys = [entry["shape"]] if entry.get("shape") else list(entry["shapes"].keys())

        valid_shapes = []
        skip_reasons = []

        for shape in shape_keys:
            gpus = entry["shapes"].get(shape)
            if gpus is None:
                skip_reasons.append("shape %r not found in 'shapes'" % shape)
                continue

            if max_gpus > 0 and gpus > max_gpus:
                skip_reasons.append("shape %r needs %g GPUs (over max_gpus %g)" % (shape, gpus, max_gpus))
                continue

            probe = probes.get(shape)
            if probe is None:
                skip_reasons.append("no probe for shape %r" % shape)
                continue
            if not probe["ok"] or probe.get("estimated_wait_sec") is None:
                skip_reasons.append("last probe for %r failed: %s"
                                    % (shape, probe.get("message") or "no estimate"))
                continue

            age = now - probe["collected_at"]
            if age > params["max_probe_age_hours"] * 3600:
                skip_reasons.append("probe for %r is %.1f h old, limit is %.1f h"
                                    % (shape, age / 3600, params["max_probe_age_hours"]))
                continue

            rate = float(entry["units_per_gpu_per_hour"]) * float(gpus)
            # Spare capacity for the same number of units everywhere, costing each
            # cluster the hours that many units actually take it.
            reserve_hours = params["reserve_units"] / rate
            usable_hours = float(entry["max_walltime_hours"]) - reserve_hours
            if usable_hours <= 0:
                skip_reasons.append("a reserve of %g units needs %.1f h at %g units/h, over "
                                    "the %g h walltime limit"
                                    % (params["reserve_units"], reserve_hours, rate,
                                       entry["max_walltime_hours"]))
                continue

            ratio = (usage.get(probe.get("placed_partition")) or {}).get("used_ratio")
            wait_eff = discount(probe["estimated_wait_sec"], ratio,
                                params["discount_strength"])
            valid_shapes.append({
                "cluster": cluster,
                "shape": shape,
                "partition": probe.get("placed_partition"),
                "gpus": gpus,
                "units_per_gpu_per_hour": float(entry["units_per_gpu_per_hour"]),
                "units_per_hour": rate,
                "max_walltime_hours": float(entry["max_walltime_hours"]),
                "min_units": int(entry.get("min_units", params["min_units"])),
                "wait_raw_sec": probe["estimated_wait_sec"],
                "wait_eff_sec": wait_eff,
                "wait_hours": wait_eff / 3600,
                "used_ratio": ratio,
                "probe_age_sec": int(age),
                "reserve_hours": reserve_hours,
                "usable_hours": usable_hours,
                "cap_units": int(usable_hours * rate),
            })

        if not valid_shapes:
            excluded.append({
                "cluster": cluster,
                "reason": "; ".join(skip_reasons) if skip_reasons else "no usable shapes",
            })
            continue

        sites.append({
            "cluster": cluster,
            "min_units": int(entry.get("min_units", params["min_units"])),
            "shapes": valid_shapes,
            "cap_units": max(s["cap_units"] for s in valid_shapes),
        })

    return sites, excluded


# --------------------------------------------------------------------------
# the waterfill
# --------------------------------------------------------------------------


def shape_capacity(s, horizon_hours):
    """Units this shape finishes by `horizon_hours`, given when it expects to
    start and how long a single job may run."""
    hours = min(max(horizon_hours - s["wait_hours"], 0.0), s["usable_hours"])
    return hours * s["units_per_hour"]


def best_shape_at(site, horizon_hours):
    """Pick the shape for a cluster that achieves highest capacity at horizon_hours,
    breaking ties with fewer GPUs and shorter wait."""
    if not site.get("shapes"):
        return site
    return max(site["shapes"], key=lambda s: (
        shape_capacity(s, horizon_hours),
        -s["gpus"],
        -s["wait_hours"],
    ))


def capacity(site, horizon_hours):
    """Units this site finishes by `horizon_hours`."""
    if not site.get("shapes"):
        return shape_capacity(site, horizon_hours)
    best = best_shape_at(site, horizon_hours)
    return shape_capacity(best, horizon_hours)


def total_capacity(sites):
    return sum(s["cap_units"] for s in sites)


def horizon_for(sites, units):
    """Smallest horizon at which `sites` together cover `units`.

    Bisected rather than solved in closed form: with the walltime caps in place
    the capacity curve has a breakpoint per site per cap, and the algebra buys
    nothing over 60 halvings of a monotone function.
    """
    if not sites or units <= 0:
        return 0.0
    high = 0.0
    for s in sites:
        if s.get("shapes"):
            high = max(high, max(cand["wait_hours"] + cand["usable_hours"] for cand in s["shapes"]))
        else:
            high = max(high, s["wait_hours"] + s["usable_hours"])
    low = 0.0
    for _ in range(60):
        middle = (low + high) / 2
        if sum(capacity(s, middle) for s in sites) >= units:
            high = middle
        else:
            low = middle
    return high


def largest_remainder(shares, caps, target):
    """Round fractional shares to integers summing to exactly `target`.

    Data units are not divisible, and a split that misses the request by a unit
    means data silently not processed.
    """
    counts = [min(int(s), c) for s, c in zip(shares, caps)]
    order = sorted(range(len(shares)), key=lambda i: shares[i] - int(shares[i]),
                   reverse=True)
    while sum(counts) < target:
        moved = False
        for i in order:
            if sum(counts) >= target:
                break
            if counts[i] < caps[i]:
                counts[i] += 1
                moved = True
        if not moved:
            break
    return counts


def split(sites, units):
    """Integer unit counts for `sites`, one waterfill, no minimums applied."""
    if not sites:
        return []
    target = min(units, total_capacity(sites))
    h = horizon_for(sites, target)
    resolved = [best_shape_at(s, h) for s in sites]
    shares = [shape_capacity(s, h) for s in resolved]
    total = sum(shares)
    if total > 0:
        shares = [s * target / total for s in shares]
    return largest_remainder(shares, [s["cap_units"] for s in resolved], target)


def assign(sites, units):
    """Waterfill, then honour min_units by dropping and waterfilling again.

    A site whose share is under its minimum is removed and its units move to the
    sites that remain -- which is a longer job there, and the point of the knob:
    three units on a second centre is not worth a job. Dropping is skipped when
    the rest cannot cover the request without it, since a short job somewhere
    beats leaving data unprocessed.
    """
    active, dropped, pinned = list(sites), [], set()
    while True:
        target = min(units, total_capacity(active))
        h = horizon_for(active, target)
        resolved = [best_shape_at(s, h) for s in active]
        counts = split(active, units)
        under = [(c, s, r) for c, s, r in zip(counts, active, resolved)
                 if 0 < c < r["min_units"] and s["cluster"] not in pinned]
        if not under:
            return counts, resolved, dropped

        count, site, res = min(under, key=lambda triplet: triplet[0])
        rest = [s for s in active if s is not site]
        if rest and total_capacity(rest) >= min(units, total_capacity(active)):
            dropped.append({
                "cluster": site["cluster"],
                "reason": "share of %d units is under min_units %d; moved to the "
                          "remaining clusters" % (count, res["min_units"]),
            })
            active = rest
        else:
            pinned.add(site["cluster"])


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------


def make_plan(units, overrides, latest, config):
    if not isinstance(units, int) or isinstance(units, bool) or units < 1:
        raise HTTPException(400, "units must be a positive integer")

    params = merge_params(config, overrides)
    sites, excluded = candidates(config, params, latest)
    counts, active_resolved, dropped = assign(sites, units)
    excluded += dropped

    rows, horizon, latest_finish = [], None, None
    for count, site in zip(counts, active_resolved):
        if not count:
            excluded.append({
                "cluster": site["cluster"],
                "reason": "not needed: its %.1f h wait is past the horizon"
                          % site["wait_hours"],
            })
            continue
        # The reserve is walltime the job may not need. It costs nothing when
        # the start lands as predicted, and covers the run when it lands late or
        # when work arrives that another cluster could not get to.
        walltime = min(count / site["units_per_hour"] + site["reserve_hours"],
                       site["max_walltime_hours"])
        done = site["wait_hours"] + count / site["units_per_hour"]
        horizon = max(horizon or 0.0, done)
        latest_finish = max(latest_finish or 0.0, site["wait_hours"] + walltime)
        rows.append({
            "cluster": site["cluster"],
            "shape": site["shape"],
            "partition": site["partition"],
            "units": count,
            "gpus": site["gpus"],
            "units_per_gpu_per_hour": site["units_per_gpu_per_hour"],
            "units_per_hour": site["units_per_hour"],
            "walltime_hours": round(walltime, 3),
            "reserve_hours": round(site["reserve_hours"], 3),
            "start_hours": round(site["wait_hours"], 3),
            "finish_hours": round(done, 3),
            "wait_raw_sec": site["wait_raw_sec"],
            "wait_eff_sec": round(site["wait_eff_sec"]),
            "used_ratio": site["used_ratio"],
            "probe_age_sec": site["probe_age_sec"],
            "min_units": site["min_units"],
        })

    assigned = sum(r["units"] for r in rows)
    rows.sort(key=lambda r: -r["units"])
    return {
        "units": units,
        "assigned_units": assigned,
        "unassigned_units": units - assigned,
        "feasible": assigned == units,
        "horizon_hours": None if horizon is None else round(horizon, 3),
        "latest_finish_hours": None if latest_finish is None else round(latest_finish, 3),
        "params": params,
        "clusters": rows,
        "excluded": excluded,
    }
