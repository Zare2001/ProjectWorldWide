"""Admission, the joint solve, and the decisions the planner exists to make.

    python3 tests/test_plan_search.py

Membership, shape, lanes, chaining and accumulation are ONE decision. Adding a site
changes the barrier, which changes every incumbent's accumulation, which changes every
site's corpus burn, which changes when the run ends -- so an incumbent's best shape is
not the one it had before the partner arrived, and none of these can be settled in
sequence.

Every scenario below is built from literals and every expected number is derived in a
comment, because the point of a planner is that a scientist can disagree with it
arithmetically. The round costs the scenarios are built on are pinned in
tests/test_plan_rounds.py:

    snellius@4  step 32/89.8 = 0.356347 s   o =  59.0005 s   solo period 111.635 s
    lumi@8      step 64/38.2 = 1.675393 s   o = 103.3677 s   solo period 287.907 s
    together    period 346.9075 s, 9.375 DARL blocks, 19,660,800 tokens per round

The last section covers `inputs`, the only module that touches the outside world.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import signal
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASSED, FAILED = [], []
CHECK_TIMEOUT_S = 120


def check(name: str):
    def decorator(fn):
        def on_timeout(signum, frame):
            raise TimeoutError(f"exceeded {CHECK_TIMEOUT_S}s")

        previous = signal.signal(signal.SIGALRM, on_timeout)
        signal.alarm(CHECK_TIMEOUT_S)
        try:
            fn()
            PASSED.append(name)
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        return fn

    return decorator


from pww.plan import (  # noqa: E402
    DEFAULT_CALIBRATION as CAL,
    Candidate,
    DarlState,
    Geometry,
    Option,
    PlanConfig,
    PlannerInputs,
    Shape,
    ShapeKey,
    Score,
    SiteInput,
    SiteLimits,
    WaitEstimate,
    admit,
    chain_breakeven_c,
    detect_traps,
    duty_cycle,
    crosschecks,
    generate_options,
    make_plan,
    probe_shape_json,
    score,
    simulate,
    solve,
)
from pww.plan import inputs as io  # noqa: E402
from pww.plan.rounds import make_member, round_cost  # noqa: E402

HOUR = 3600.0
PLENTY = DarlState(num_blocks=2692, committed=0, leased=0, unassigned=1_000_000)
REAL = DarlState(num_blocks=2692, committed=1870, leased=0, unassigned=822)
EMPTY = DarlState(num_blocks=2692, committed=2692, leased=0, unassigned=0)


def wait(hours: float, **kw) -> WaitEstimate:
    """A probe reading. p90 defaults to p50 so a scenario is deterministic unless it
    deliberately makes the queue uncertain."""
    p90 = kw.pop("p90_h", hours)
    return WaitEstimate(p50_raw_s=hours * HOUR, p90_raw_s=p90 * HOUR,
                        p50_eff_s=hours * HOUR, p90_eff_s=p90 * HOUR,
                        samples=kw.pop("samples", 3), probe_age_s=kw.pop("age_s", 60.0), **kw)


def shape(site: str, partition: str, gpus: int, hours: float, account: str | None = None) -> Shape:
    h, m = int(hours), int(round((hours - int(hours)) * 60))
    args = (f"-A {account} " if account else "") + \
        f"-p {partition} -N 1 --gpus-per-node {gpus} -t {h}:{m:02d}:00"
    return Shape(f"{site}_{gpus}g_{hours:g}h", io.parse_shape_args(site, args), args)


def site(name, partition, gpus, curve, tput, batch, startup_s, *, account=None, **kw) -> SiteInput:
    """One site with a w(T) curve: `curve` is [(walltime_hours, wait_hours), ...]."""
    shapes, waits = [], {}
    for hours, wait_h in curve:
        sh = shape(name, partition, gpus, hours, account)
        shapes.append(sh)
        waits[sh.name] = wait(wait_h)
    return SiteInput(site=name, shapes=tuple(shapes), waits=waits,
                     geometries={gpus: Geometry(name, gpus, tput, batch)},
                     startup_s=startup_s, **kw)


def snellius(curve, *, tput=89.8, batch=32, startup_s=300.0, **kw) -> SiteInput:
    return site("snellius", "gpu_h100", 4, curve, tput, batch, startup_s, **kw)


def lumi(curve, *, gpus=8, tput=38.2, batch=64, startup_s=600.0, **kw) -> SiteInput:
    return site("lumi", "standard-g", gpus, curve, tput, batch, startup_s,
                account="project_462000226", **kw)


def config(**kw) -> PlanConfig:
    base = dict(horizon_s=48 * HOUR, num_rounds=1_000_000, balance="off", lanes_max=1)
    base.update(kw)
    return PlanConfig(**base)


def options(sites, cfg, prune=True):
    """admit + generate_options for a whole federation, as make_plan does it."""
    cands = {}
    excl = []
    for s in sites:
        c, e = admit(s, config=cfg, calibration=CAL)
        excl += e
        if c:
            cands[s.site] = c
    out = {}
    for s in sites:
        if s.site in cands:
            o, e = generate_options(s.site, cands[s.site], cands, cfg,
                                    warm=s.warm_checkpoint, prune=prune)
            excl += e
            out[s.site] = o
    return out, excl


def utility(selection, cfg, darl=PLENTY, balance=False):
    return score(simulate(selection, config=cfg, calibration=CAL, darl=darl,
                          balance=balance), cfg)


def close(got: float, want: float, tol: float = 1e-6) -> bool:
    return abs(got - want) <= tol


# --------------------------------------------------------------------------
# phase 0 -- admission. Never silently plan.
# --------------------------------------------------------------------------


@check("every admission failure names a reason AND the command that fixes it")
def _():
    """An exclusion a reader cannot act on is a silent drop with extra words. This is
    the whole matrix; if a path is added without a fix string, this fails."""
    cfg = config()
    good_geometry = {4: Geometry("snellius", 4, 89.8, 32)}
    sh = shape("snellius", "gpu_h100", 4, 8.0)

    def one(**over) -> SiteInput:
        base = dict(site="snellius", shapes=(sh,), waits={sh.name: wait(1.0)},
                    geometries=good_geometry, startup_s=300.0)
        base.update(over)
        return SiteInput(**base)

    cases = {
        # a shape the plan wants but nobody has probed: w(T) is never invented
        "shape_not_probed": one(wanted_shapes=(ShapeKey("snellius", "gpu_h100", 1, 1, 4 * 3600),)),
        # the probe row is missing outright
        "no_probe": one(waits={}),
        # sbatch --test-only came back with an error
        "probe_failed": one(waits={sh.name: wait(1.0, ok=False,
                                                 message="sbatch: error: invalid account")}),
        # the collector cron died: a stale wait is worse than no wait
        "probe_stale": one(waits={sh.name: wait(1.0, age_s=30 * HOUR)}),
        # --test-only is conditioned on the PROBING account's fairshare and QOS
        "probed_by_other_account": one(waits={sh.name: wait(1.0, probed_by_user="vanderwal")},
                                       submitter="zpalanciya"),
        # no measured (site, devices) cell: step time is NOT extrapolated across geometry
        "no_throughput": one(geometries={}),
        # the walltime does not even cover startup
        "walltime_under_startup": one(shapes=(shape("snellius", "gpu_h100", 4, 1.0),),
                                      waits={"snellius_4g_1h": wait(1.0)}, startup_s=2 * HOUR),
        # a site with no overhead calibration is refused, not priced off another's
        "no_overhead_calibration": SiteInput(
            site="vega", shapes=(shape("vega", "gpu", 4, 8.0),),
            waits={"vega_4g_8h": wait(1.0)},
            geometries={4: Geometry("vega", 4, 50.0, 32)}, startup_s=300.0),
    }
    for code, site_input in cases.items():
        _cands, excl = admit(site_input, config=cfg, calibration=CAL)
        found = [e for e in excl if e.code == code]
        assert found, f"{code}: got {[e.code for e in excl]}"
        for e in excl:
            assert e.reason.strip(), f"{e.code} has no reason"
            assert e.subject.strip(), f"{e.code} has no subject"
            assert e.fix.strip(), f"{e.code} has no fix -- that is a silent drop"
        # A site with nothing admitted is excluded wholesale, carrying the reasons.
        if code not in ("shape_not_probed",):
            whole = [e for e in excl if e.code == "site_unusable"]
            assert whole, [e.code for e in excl]

    # ...and the fixes are specific, not "check your configuration".
    _c, excl = admit(cases["no_throughput"], config=cfg, calibration=CAL)
    fix = next(e.fix for e in excl if e.code == "no_throughput")
    assert "PWW_TPUT_SNELLIUS_4" in fix and "calibrate_throughput.sh" in fix, fix
    _c, excl = admit(cases["probe_stale"], config=cfg, calibration=CAL)
    reason = next(e.reason for e in excl if e.code == "probe_stale")
    assert "30.0 h old, limit is 6.0 h" in reason, reason


@check("an unprobed shape comes back as the JSON to paste into the collector config")
def _():
    """The one thing the planner must never do is invent a w(T), so the only honest
    response to a wanted-but-unprobed shape is to name the shape."""
    key = ShapeKey("lumi", "standard-g", 1, 8, 1 * 3600, "project_462000226")
    rendered = probe_shape_json("lumi", key)
    entry = json.loads(rendered)          # it has to be pasteable, so it has to parse
    assert entry["name"] == "standardg_1node_1h"
    assert entry["args"] == ["-A", "project_462000226", "-p", "standard-g", "-N", "1",
                             "--gpus-per-node", "8", "-t", "1:00:00"]
    # and it must round-trip back to the very key that was asked for
    assert io.parse_shape_args("lumi", " ".join(entry["args"])) == key

    s = lumi([(24, 1.0)], wanted_shapes=(key,))
    _cands, excl = admit(s, config=config(), calibration=CAL)
    missing = next(e for e in excl if e.code == "shape_not_probed")
    assert "standard-g 8 gpu 1 h" in missing.reason
    assert "configs/slurm_probe/lumi.json" in missing.fix
    # The cost of the ask is stated with the ask: probes run serially, 60 s each.
    assert "serially" in missing.fix and "60 s" in missing.fix


@check("a site with one bad shape keeps its good ones")
def _():
    s = snellius([(2, 0.1), (8, 1.0)])
    s = dataclasses.replace(s, waits={**s.waits, "snellius_4g_2h": wait(0.1, ok=False,
                                                                        message="held")})
    cands, excl = admit(s, config=config(), calibration=CAL)
    assert [c.shape.name for c in cands] == ["snellius_4g_8h"]
    assert [e.code for e in excl] == ["probe_failed"]
    assert not any(e.code == "site_unusable" for e in excl)


@check("a site with a throughput entry but no probed shape names the collector")
def _():
    """The site_unusable reason is joined from the per-shape exclusions, and a site
    with zero shapes produces none: the message came out as an empty reason followed
    by "see the per-shape fixes above" with nothing above it. Exclusion's own rule is
    that a reader who cannot act on it is a silent drop with extra words, and this is
    the one case where the real cause -- no probe rows for this cluster in the window
    that was read -- is never visible anywhere else in the report.

    Untested until now: deleting the branch left the suite green while the exclusion
    degraded back to "0 shape(s) were read but none survived admission"."""
    mars = SiteInput(site="mars", shapes=(), waits={},
                     geometries={4: Geometry("mars", 4, 50.0, 32)}, startup_s=100.0)
    _cands, excl = admit(mars, config=config(), calibration=CAL)
    whole = [e for e in excl if e.code == "site_unusable"]
    assert len(whole) == 1, [e.code for e in excl]
    reason, fix = whole[0].reason, whole[0].fix
    # The reason states the fact, rather than pointing at exclusions that do not exist.
    assert "throughput entry but no probed shape" in reason, reason
    assert not reason.rstrip().endswith(":"), reason
    assert "see the per-shape fixes above" not in fix and "see the exclusions above" not in fix
    # ...and the fix names both halves of the actual remedy, either of which is valid.
    assert "configs/slurm_probe/mars.json" in fix and "collector" in fix, fix
    assert "PWW_TPUT_MARS" in fix, fix


# --------------------------------------------------------------------------
# is this site worth submitting at all?
# --------------------------------------------------------------------------


@check("a partner fast enough to keep the barrier is included, as a number")
def _():
    """Both sites live for the whole 12 h horizon, so the only question is the round
    rate. A LUMI at 179.6 seq/s steps in 64/179.6 = 0.356 s, exactly Snellius' pace:
        period = 100*0.356347 + 17 + 59.0005 + (98.9 + 512/(3*179.6)) = 211.49 s
        12 h / 211.49 s = 204 federated merges, against 386 solo Snellius rounds
        U(include) = 204   vs   U(exclude) = 0.25 * 386 = 96.5
    """
    cfg = config(horizon_s=12 * HOUR, alpha=0.25)
    plan = make_plan(PlannerInputs(
        sites=(snellius([(12, 0.0)], startup_s=0.0),
               lumi([(12, 0.0)], tput=179.6, startup_s=0.0)),
        calibration=CAL, darl=PLENTY), cfg)

    assert {o.site for o in plan.selection} == {"snellius", "lumi"}
    assert plan.score.federated_merges == 204 and plan.score.solo_merges == 0
    assert close(plan.score.utility, 204.0)
    entry = next(m for m in plan.marginal if m.site == "lumi")
    assert entry.included and close(entry.delta_utility, 204.0 - 96.5)
    # The ledger says WHERE the utility went, in both directions.
    assert close(entry.round_rate_before_h, 3600 / 111.635264, 1e-3)   # 32.2/h solo
    assert close(entry.round_rate_after_h, 3600 / 211.486, 1e-2)       # 17.0/h federated
    assert entry.tokens_per_round_before == 6_553_600
    assert entry.tokens_per_round_after == 19_660_800
    assert "kept at alpha=0.25" in entry.detail


@check("a partner slow enough to drag the barrier is excluded, as a number")
def _():
    """Same scenario, same probe data, one number changed: LUMI at 6.4 seq/s steps in
    64/6.4 = 10 s, so it sets the barrier for everyone.
        period = 100*10 + 17 + 59.0005 + (98.9 + 512/(3*6.4)) = 1201.57 s
        12 h / 1201.57 s = 35 federated merges against 386 solo -> U 35 vs 96.5
    """
    cfg = config(horizon_s=12 * HOUR, alpha=0.25)
    plan = make_plan(PlannerInputs(
        sites=(snellius([(12, 0.0)], startup_s=0.0),
               lumi([(12, 0.0)], tput=6.4, startup_s=0.0)),
        calibration=CAL, darl=PLENTY), cfg)

    assert {o.site for o in plan.selection} == {"snellius"}, plan.describe()
    assert plan.score.federated_merges == 0 and plan.score.solo_merges == 386
    entry = next(m for m in plan.marginal if m.site == "lumi")
    assert not entry.included
    assert close(entry.delta_utility, 35.0 - 96.5), entry.delta_utility
    assert "EXCLUDED at alpha=0.25" in entry.detail
    assert close(entry.round_rate_after_h, 3600 / 1201.57, 1e-2)       # 3.0/h
    # It would be worth adding at a low enough alpha, and the ledger says which.
    #   35 + alpha*0 vs 0 + alpha*386  ->  alpha = 35/386 = 0.0907
    assert close(entry.flip_alpha, 35 / 386, 1e-3), entry.flip_alpha
    assert "flips at alpha = 0.09" in entry.detail


# --------------------------------------------------------------------------
# the staggered start
# --------------------------------------------------------------------------


@check("the headstart decision flips with the weight on solo progress")
def _():
    """THE SAME PROBE DATA, two plans, and which one is right depends entirely on what
    the run is FOR. Snellius is queued 4 h; LUMI can start now.

        (b) headstart: LUMI trains alone from 600 s to 14,700 s -> 49 solo rounds at
            6.25 blocks = 306.25 blocks spent. 822 - 306.25 = 515.75 blocks left, at
            9.375 per federated round = 55 federated merges.
        (c) aligned:   LUMI delayed to Snellius' arrival. 6 Snellius solo rounds at
            3.125 blocks = 18.75, then 803.25 / 9.375 = 85 federated merges.

        U(b) = 55 + 49*alpha   vs   U(c) = 85 + 6*alpha
        equal at 43*alpha = 30, i.e. alpha = 30/43 = 0.698

    Below that the corpus is better spent federated; above it the headstart wins. The
    exchange rate is never buried -- it is reported, and it is adjustable.
    """
    cfg0 = config(alpha=0.0, balance="off")
    cfg1 = config(alpha=1.0, balance="off")
    s = snellius([(24, 4.0)])
    early = lumi([(24, 0.0)])
    # 4 h of queue + 300 s of startup: the instant Snellius can first contribute.
    aligned_begin = 4 * HOUR + 300.0

    opts, _ = options((s, early), cfg0)
    snel = next(o for o in opts["snellius"] if o.begin_s == 0.0)
    head = next(o for o in opts["lumi"] if o.begin_s == 0.0 and o.links_per_lane == 1)
    late = next(o for o in opts["lumi"]
                if close(o.begin_s, aligned_begin) and o.links_per_lane == 1)

    u_head = utility((snel, head), cfg0, darl=REAL)
    u_late = utility((snel, late), cfg0, darl=REAL)
    assert (u_head.federated_merges, u_head.solo_merges) == (55, 49), u_head
    assert (u_late.federated_merges, u_late.solo_merges) == (85, 6), u_late
    # blocks: 49*6.25 = 306.25 solo, then 55*9.375 = 515.625 federated
    assert close(simulate((snel, head), config=cfg0, calibration=CAL, darl=REAL,
                          balance=False).blocks_used, 49 * 6.25 + 55 * 9.375)

    # solo worthless: take the delayed start
    assert u_late.utility > u_head.utility, (u_late.utility, u_head.utility)
    assert close(u_late.utility, 85.0) and close(u_head.utility, 55.0)
    # solo as good as federated: take the headstart
    u_head1 = utility((snel, head), cfg1, darl=REAL)
    u_late1 = utility((snel, late), cfg1, darl=REAL)
    assert close(u_head1.utility, 104.0) and close(u_late1.utility, 91.0)
    assert u_head1.utility > u_late1.utility
    # ...and the crossover is where the hand arithmetic says it is
    crossover = (85 - 55) / (49 - 6)
    assert close(crossover, 30 / 43) and close(crossover, 0.697674, 1e-6)
    for alpha, winner in ((0.6, "late"), (0.8, "early")):
        cfg = config(alpha=alpha)
        got = "early" if utility((snel, head), cfg, darl=REAL).utility > \
                         utility((snel, late), cfg, darl=REAL).utility else "late"
        assert got == winner, (alpha, got)


@check("the planner reports the alpha at which its own recommendation changes")
def _():
    """alpha is never buried inside the objective. In this scenario the delayed start
    wins while solo rounds are cheap, and a Snellius-only plan takes over once they are
    worth as much as federated ones -- because a solo round costs 3.125 blocks against
    a federated round's 9.375, so at alpha = 1 the fastest-cycling single site always
    wins. That is a real property of the model, and it is reported rather than hidden.
    """
    sites = (snellius([(24, 4.0)]), lumi([(24, 0.0)]))
    cheap = make_plan(PlannerInputs(sites=sites, calibration=CAL, darl=REAL),
                      config(alpha=0.25, assume_overhead=True))
    assert {o.site for o in cheap.selection} == {"snellius", "lumi"}
    assert next(o for o in cheap.selection if o.site == "lumi").begin_s > 0.0, \
        "at alpha 0.25 the corpus is better spent federated than on a headstart"
    # It does better than either hand-built plan by delaying BOTH submissions until
    # the pair can start together: LUMI to 4 h + 300 s, Snellius by its partner's
    # 600 s of startup, so nothing is spent solo and all 822 blocks go to federated
    # rounds -- 822 / 9.375 = 87.7 -> 87 merges.
    assert cheap.score.federated_merges == 87 and cheap.score.solo_merges == 0
    assert all(o.begin_s > 0.0 for o in cheap.selection), cheap.describe()
    # alpha* is the boundary of the WINNER's own segment of the envelope, and it is
    # checked against a re-solve on either side of it rather than against a range
    # someone eyeballed: an envelope built with the concavity test the wrong way round
    # returns a plausible-looking number that no re-solve agrees with.
    assert cheap.alpha_star is not None
    star = cheap.alpha_star
    assert 0.33 < star < 0.34, star
    def winner_at(alpha):
        return make_plan(PlannerInputs(sites=sites, calibration=CAL, darl=REAL),
                         config(alpha=alpha, assume_overhead=True)).describe()
    assert winner_at(star - 0.01) == cheap.describe(), star
    assert winner_at(star + 0.01) != cheap.describe(), star

    dear = make_plan(PlannerInputs(sites=sites, calibration=CAL, darl=REAL),
                     config(alpha=1.0, assume_overhead=True))
    assert {o.site for o in dear.selection} == {"snellius"}, dear.describe()
    assert dear.score.federated_merges == 0
    # The sensitivity table re-ranks the plans the search already simulated, so the
    # flip is visible without re-running the planner.
    rows = {r.value: r for r in cheap.sensitivity if r.knob == "alpha"}
    assert set(rows) == {"0", "0.25", "0.5", "1"}, sorted(rows)
    assert not rows["0"].changed and rows["1"].changed
    assert rows["1"].winner != rows["0.25"].winner


@check("the alpha envelope is the UPPER hull, checked against brute force")
def _():
    """REGRESSION. The envelope is what alpha* is read off, and it was built with the
    concavity test inverted -- a LOWER hull. With two non-dominated plans that is
    invisible; with three it returns a non-monotone breakpoint list naming a plan that
    wins at no alpha at all, and alpha* then reports a value where nothing changes.

    The reference case is A(fed 35, solo 17), B(28, 25), C(8, 52): B never wins, because
    at the A/C crossover alpha = 27/35 = 0.7714 it scores 28 + 0.7714*25 = 47.3 against
    A's 48.1. The only breakpoint is that crossover."""
    from pww.plan.search import _Eval, alpha_breakpoints

    def ev(fed, solo, tag):
        sc = Score(utility=fed + 0.25 * solo, federated_merges=fed, solo_merges=solo,
                   tokens=0, blocks=0, gpu_s=0, compute_s=0, attempts=fed + solo,
                   alpha=0.25, beta=0.0)
        cand = Candidate("s", Shape(tag, ShapeKey("s", "p", 1, 1, 3600), "a"),
                         WaitEstimate(0, 0, 0, 0, 1, 0.0), Geometry("s", 1, 1.0, 8),
                         0.0, 1.0, "identified")
        return _Eval(score=sc, timeline=None, balance=False, traps=(), rankable=True,
                     selection=(Option("s", cand, 1, 1, "none", 0.0),))

    plans = [ev(35, 17, "A"), ev(28, 25, "B"), ev(8, 52, "C")]
    breaks = alpha_breakpoints(plans, 0.0)
    names = [(round(a, 4), s[0].candidate.shape.name) for a, s in breaks]
    assert names == [(0.0, "A"), (0.7714, "C")], names

    # Exhaustive: on random frontiers the envelope must agree with argmax at every
    # alpha, and its thresholds must strictly increase from 0.
    import random
    rng = random.Random(20260819)
    for _trial in range(200):
        pts = [(rng.randint(0, 300), rng.randint(0, 300))
               for _ in range(rng.randint(2, 8))]
        breaks = alpha_breakpoints([ev(f, s, f"{f}/{s}") for f, s in pts], 0.0)
        ths = [a for a, _ in breaks]
        assert ths[0] == 0.0 and all(b > a for a, b in zip(ths, ths[1:])), ths
        for step in range(0, 161):
            alpha = step / 40.0
            best = max(f + alpha * s for f, s in pts)
            chosen = breaks[0][1]
            for th, sel in breaks:
                if th <= alpha + 1e-12:
                    chosen = sel
            f, s = (int(v) for v in chosen[0].candidate.shape.name.split("/"))
            assert abs((f + alpha * s) - best) < 1e-9, (pts, alpha, best, (f, s))


@check("a plan that federates zero times is flagged before it is ranked")
def _():
    """Trapped plans sort below everything else regardless of utility, so the trap can
    only come back when every alternative is also trapped. Reading plan.selection
    without reading plan.traps is the mistake this ordering exists to prevent."""
    # LUMI can only be given a 20 h walltime, which expires exactly as Snellius
    # arrives -- and chaining is switched off, so it cannot buy presence that way.
    cfg = config(alpha=0.25, assume_overhead=True, chain_policies=("none",))
    sites = (snellius([(24, 20.0)]), lumi([(20, 0.0)]))
    plan = make_plan(PlannerInputs(sites=sites, calibration=CAL, darl=PLENTY), cfg)
    assert plan.rankable and not plan.traps, plan.traps
    # The way out is --begin: hold LUMI until its partner is out of the queue. The
    # naive plan gets zero federated rounds; this one gets 205.
    assert {o.site for o in plan.selection} == {"snellius", "lumi"}, plan.describe()
    delayed = next(o for o in plan.selection if o.site == "lumi")
    assert close(delayed.begin_s, 20 * HOUR + 300.0), delayed.begin_s
    assert plan.score.federated_merges > 200

    # Forced to submit both immediately, the trap is detected and named.
    opts, _ = options(sites, cfg)
    both = (next(o for o in opts["snellius"] if o.begin_s == 0.0),
            next(o for o in opts["lumi"] if o.begin_s == 0.0))
    timeline = simulate(both, config=cfg, calibration=CAL, darl=PLENTY, balance=False)
    traps = detect_traps(both, timeline, cfg)
    assert [t.code for t in traps] == ["trap_no_overlap"]
    assert score(timeline, cfg).federated_merges == 0


# --------------------------------------------------------------------------
# short jobs versus one long job -- and the asymmetry between the two sites
# --------------------------------------------------------------------------


@check("the site with the bad queue chains and the site with the good queue does not")
def _():
    """One input, two opposite policies, which is the whole point: a planner that
    applies one policy to both sites leaves most of the value on the table.

        snellius (steep):  w(2 h) = 0.05 h,  w(24 h) = 20 h
        lumi     (flat):   w(2 h) = 1 h,     w(24 h) = 1 h

    Over a 24 h horizon, one 24 h Snellius job is productive for 24 - 20 - 0.083 =
    3.9 h; twelve chained 2 h jobs are productive for ~23.9 h minus 12 startups.
    For LUMI the queue is insensitive to walltime, so chaining buys nothing and pays
    30 min of startup twelve times over: one long job is productive for 22.5 h,
    the chain for 12 x (2 h - 0.5 h) = 18 h.
    """
    cfg = config(horizon_s=24 * HOUR, max_links_per_lane=12, assume_overhead=True)
    sites = (snellius([(2, 0.05), (24, 20.0)]),
             lumi([(2, 1.0), (24, 1.0)], startup_s=1800.0))
    plan = make_plan(PlannerInputs(sites=sites, calibration=CAL, darl=PLENTY), cfg)

    chosen = {o.site: o for o in plan.selection}
    assert set(chosen) == {"snellius", "lumi"}, plan.describe()
    assert chosen["snellius"].links_per_lane == 12 and chosen["snellius"].chain == "self"
    assert chosen["snellius"].candidate.walltime_s == 2 * 3600
    assert chosen["lumi"].links_per_lane == 1 and chosen["lumi"].chain == "none"
    assert chosen["lumi"].candidate.walltime_s == 24 * 3600

    # The closed-form duty-cycle check is computed alongside and must AGREE. It is a
    # cross-check, never a second decision path -- a disagreement is a finding.
    # The check is keyed on (site, DEVICE COUNT): w(T) is a walltime curve, and
    # differencing a 1-GPU probe against a 4-GPU one is not a reading of it.
    checks = {c.name: c for c in plan.crosschecks}
    assert checks["chain_or_one_long_job[snellius@4]"].verdict == "chain"
    assert checks["chain_or_one_long_job[lumi@8]"].verdict == "one long job"
    assert all(c.agrees for c in plan.crosschecks if c.agrees is not None), plan.crosschecks
    # duty(2 h) = (2 - 0.083)/(2 + 0.05) = 0.935 against duty(24 h) = (24-0.083)/44 = 0.544
    assert "0.935 at 2 h" in checks["chain_or_one_long_job[snellius@4]"].detail
    assert close(duty_cycle(2 * HOUR, 300, 0.05 * HOUR), 0.93496, 1e-5)
    assert close(duty_cycle(24 * HOUR, 300, 20 * HOUR), 0.543561, 1e-6)


@check("the duty-cycle cross-check never differences two different device counts")
def _():
    """REGRESSION. `shapes` was keyed on the walltime alone, so a 1-GPU 1 h probe and a
    4-GPU 8 h probe landed in the same dict and c* was computed across a change of
    GEOMETRY as well as of T. The reduced-geometry shape has its own, much shorter,
    queue -- on the shipped fixture that produced c* = -456 min from a 1-GPU w(1 h)
    against a 4-GPU w(8 h) and reported it as a walltime curve."""
    cfg = config(horizon_s=24 * HOUR, assume_overhead=True)
    # 4 GPUs: one walltime only. 1 GPU: two walltimes, and a wildly different queue.
    site_in = site("snellius", "gpu_h100", 4, [(8, 12.0)], 89.8, 32, 300.0)
    small = site("snellius", "gpu_h100", 1, [(1, 0.01), (8, 0.02)], 24.4, 8, 300.0)
    merged = dataclasses.replace(
        site_in,
        shapes=site_in.shapes + small.shapes,
        waits={**site_in.waits, **small.waits},
        geometries={4: Geometry("snellius", 4, 89.8, 32),
                    1: Geometry("snellius", 1, 24.4, 8)})
    plan = make_plan(PlannerInputs(sites=(merged,), calibration=CAL, darl=PLENTY), cfg)
    gpus = plan.selection[0].candidate.gpus
    names = [c.name for c in plan.crosschecks if c.name.startswith("chain_or")]
    assert names == [f"chain_or_one_long_job[snellius@{gpus}]"], names
    check_ = next(c for c in plan.crosschecks if c.name == names[0])
    if gpus == 4:
        # only one 4-GPU walltime is probed, so there is no curve to difference
        assert check_.verdict == "not evaluable", check_.detail
    else:
        # both 1-GPU walltimes are probed, and only those two are in the arithmetic
        assert "at 1 device(s)" in check_.detail, check_.detail
        assert "12.00h" not in check_.detail, check_.detail


@check("chaining is never recommended when the queue is insensitive to walltime")
def _():
    """c* = [t*w(T) - T*w(t)] / [(T - t) + w(T) - w(t)]. With w(t) = w(T) = w the
    numerator is w*(t - T) < 0, so c* is negative and no startup cost is small enough.
    Chaining only ever buys a queue advantage; paying per-job startup for one that does
    not exist is pure loss."""
    flat = chain_breakeven_c(2 * HOUR, 24 * HOUR, 1 * HOUR, 1 * HOUR)
    assert flat < 0, flat
    assert close(flat, -3600.0)          # w*(t-T)/(T-t) = -w
    # A steep curve makes it positive and finite: c* = [2*20 - 24*0.05]/[22 + 19.95] h
    steep = chain_breakeven_c(2 * HOUR, 24 * HOUR, 0.05 * HOUR, 20 * HOUR)
    assert close(steep / 60, 55.5, 0.1), steep / 60
    # An identical shape is not a decision at all.
    assert chain_breakeven_c(2 * HOUR, 2 * HOUR, HOUR, HOUR) == float("-inf")

    # assume_overhead because a LUMI-solo round has no measured regime -- LUMI never
    # ran without Snellius in any log -- so the plan is otherwise priced, not ranked.
    cfg = config(horizon_s=24 * HOUR, max_links_per_lane=12, assume_overhead=True)
    plan = make_plan(PlannerInputs(sites=(lumi([(2, 1.0), (24, 1.0)], startup_s=1800.0),),
                                   calibration=CAL, darl=PLENTY), cfg)
    assert plan.selection[0].links_per_lane == 1, plan.describe()
    assert plan.selection[0].candidate.walltime_s == 24 * 3600


@check("the recommended job length rises with the startup cost, and chaining stops")
def _():
    """Sweeping c over the same steep w(T) curve. Presence to a 24 h horizon:
        1 h x 24 links : 23.95 h - 24c      4 h x 6 : 23.70 h - 6c
        2 h x 12 links : 23.90 h - 12c      8 h x 3 : 23.00 h - 3c
        24 h x 1       : 18.00 h - c        (the 24 h shape is quoted a 6 h wait)
    so the optimum length climbs as c grows, and past c = 4 h the single long job wins
    outright -- while the shapes shorter than c stop being admissible at all."""
    curve = [(1, 0.05), (2, 0.1), (4, 0.3), (8, 1.0), (24, 6.0)]
    chosen = []
    for minutes in (1, 5, 30, 60, 120, 240):
        cfg = config(horizon_s=24 * HOUR, max_links_per_lane=24, alpha=1.0)
        plan = make_plan(PlannerInputs(sites=(snellius(curve, startup_s=minutes * 60.0),),
                                       calibration=CAL, darl=PLENTY), cfg)
        option = plan.selection[0]
        chosen.append((minutes, option.candidate.walltime_s / HOUR, option.links_per_lane))
    lengths = [c[1] for c in chosen]
    assert lengths == sorted(lengths), chosen
    assert chosen == [(1, 4.0, 6), (5, 4.0, 6), (30, 8.0, 3),
                      (60, 8.0, 3), (120, 8.0, 3), (240, 24.0, 1)], chosen
    # Above c = 4 h chaining is not recommended at any probed length.
    assert chosen[-1][2] == 1


@check("a recommendation that moves between c/2 and 2c is reported, not hidden")
def _():
    """c is a LOWER BOUND at both sites -- no job script prints a timestamp before
    torchrun and Slurm writes no start line into logs/%x-%j.out -- so the c-sensitivity
    pass runs on every plan, and a plan that flips inside it is provisional."""
    cfg = config(horizon_s=24 * HOUR, max_links_per_lane=24, alpha=1.0)
    plan = make_plan(PlannerInputs(
        sites=(snellius([(1, 0.05), (2, 0.1), (4, 0.3), (8, 1.0), (24, 6.0)],
                        startup_s=90 * 60.0),),
        calibration=CAL, darl=PLENTY), cfg)
    rows = {r.value: r for r in plan.sensitivity if r.knob == "startup_cost"}
    assert set(rows) == {"c/2", "c", "2c"}, sorted(rows)
    assert not rows["c"].changed, "the base case cannot differ from itself"
    # c = 90 min chains 8 h jobs; c/2 = 45 min chains 4 h ones. The reader has to see it.
    assert rows["c/2"].changed or rows["2c"].changed, rows
    # Every knob the design promises is swept, every time.
    assert {r.knob for r in plan.sensitivity} == {
        "alpha", "beta", "discount_strength", "startup_cost", "h_model", "wait_quantile"}


# --------------------------------------------------------------------------
# the search itself
# --------------------------------------------------------------------------


@check("the two-site enumeration is exhaustive, checked against brute force")
def _():
    cfg = config(horizon_s=12 * HOUR, alpha=0.25)
    sites = (snellius([(6, 0.0), (12, 2.0)]), lumi([(6, 0.0), (12, 1.0)]))
    opts, _ = options(sites, cfg)
    selection, _timeline, sc, _ev, report = solve(
        opts, config=cfg, calibration=CAL, darl=REAL)
    assert report.exact_proved and report.method == "exact+greedy"

    best = None
    for combo in itertools.product([None] + opts["snellius"], [None] + opts["lumi"]):
        trial = tuple(o for o in combo if o is not None)
        timeline = simulate(trial, config=cfg, calibration=CAL, darl=REAL, balance=False)
        if timeline.quality == "extrapolated":
            continue                      # priced, but not rankable by default
        if detect_traps(trial, timeline, cfg):
            continue                      # flagged, not ranked
        u = score(timeline, cfg).utility
        if best is None or u > best:
            best = u
    assert close(sc.utility, best), (sc.utility, best)
    # The greedy pass runs anyway, because its rejection record is the answer to
    # "is this site worth submitting at all"; its gap is MEASURED, never assumed.
    assert report.optimality_gap is not None and report.optimality_gap >= 0.0


@check("option pruning removes duplicates only, and never the optimum")
def _():
    """The tempting prune -- 'y arrives no later and ends no earlier, so x is dominated'
    -- is unsound under a data cap and unsound in exactly the direction this planner
    exists to catch: arriving earlier spends corpus the federated phase then lacks, so
    it deletes the delayed-start option that is often the answer."""
    # Over a 10 h horizon a 12 h job and a 24 h job at the same queue wait are live
    # for exactly the same hours, so one of them is the other written twice.
    cfg = config(horizon_s=10 * HOUR, alpha=0.25, assume_overhead=True)
    sites = (snellius([(12, 1.0), (24, 1.0)]), lumi([(12, 0.0), (24, 0.0)]))
    pruned, excl = options(sites, cfg, prune=True)
    whole, _ = options(sites, cfg, prune=False)
    assert sum(len(v) for v in pruned.values()) < sum(len(v) for v in whole.values())
    assert any(e.code == "duplicate_option" for e in excl)

    a = solve(pruned, config=cfg, calibration=CAL, darl=REAL)[2].utility
    b = solve(whole, config=cfg, calibration=CAL, darl=REAL)[2].utility
    assert close(a, b), (a, b)


@check("a chain policy the simulator cannot expand is refused where it is priced")
def _():
    """generate_options chains anything that is not "none"; timeline.expand_links only
    knows "self" and "singleton" and breaks out of the loop for the rest. So a typo'd
    policy was PRICED at one job per lane and EMITTED as a full N-link chain -- 14 jobs
    submitted where 2 were costed, and merge counts off by 7x.

    argparse now rejects unknown --chain names, but that is the door and this is the
    cause: a PlanConfig built in code walks straight past the CLI. The check asks the
    simulator instead of keeping a second list of policy names, so a policy taught to
    expand_links cannot go missing here."""
    from pww.plan.timeline import expand_links

    cfg = config(chain_policies=("bogus",), max_links_per_lane=12, horizon_s=24 * HOUR)
    try:
        options((snellius([(2, 0.5)]),), cfg)
    except ValueError as exc:
        assert "bogus" in str(exc) and "expands only its first link" in str(exc), exc
    else:
        raise AssertionError("an unsimulable chain policy was priced as a chain")

    # The two policies the simulator does expand are untouched, and the pricing the
    # refusal protects is the one that used to diverge: N links priced, N simulated.
    for policy in ("self", "singleton"):
        cfg = config(chain_policies=(policy,), max_links_per_lane=12, horizon_s=24 * HOUR)
        opts, _ = options((snellius([(2, 0.5)]),), cfg)
        chained = max(opts["snellius"], key=lambda o: o.links_per_lane)
        assert chained.links_per_lane == 12, chained
        simulated = expand_links(chained, config=cfg, warm_checkpoint=False)
        assert len(simulated) == chained.lanes * chained.links_per_lane, len(simulated)
    # "none" is priced as one job and needs no simulator support.
    cfg = config(chain_policies=("none",), horizon_s=24 * HOUR)
    opts, _ = options((snellius([(2, 0.5)]),), cfg)
    assert all(o.links_per_lane == 1 for o in opts["snellius"])


@check("a plan priced off an unmeasured regime is not ranked unless it is asked for")
def _():
    """The additive overhead form is checked against four regimes and reproduces three.
    A cell outside that set is priced and TAGGED, and it is kept out of the ranking
    until someone says assume_overhead -- which the report then echoes.

    LUMI alone is exactly such a cell: LUMI never ran without Snellius in any log, so
    there is no measured LUMI-solo period anywhere and its two constants are a
    difference of two measurements rather than a measurement.
    """
    cfg = config(horizon_s=24 * HOUR, alpha=1.0)
    only_lumi = (lumi([(24, 1.0)]),)

    strict = make_plan(PlannerInputs(sites=only_lumi, calibration=CAL, darl=PLENTY), cfg)
    assert strict.selection == (), strict.describe()
    assert strict.score.utility == 0.0, "an unrankable plan must not be recommended"
    # A positive delta-U next to an exclusion reads as a bug unless the reason is
    # stated. When the WINNER is rankable, the marginal ledger is the only place the
    # demotion is explained, so the report has to print this clause.
    demoted = next(m for m in strict.marginal if m.site == "lumi")
    assert not demoted.included and demoted.delta_utility > 0
    assert "priced and NOT ranked" in demoted.detail
    assert "assume_overhead=True" in demoted.detail

    lax = make_plan(PlannerInputs(sites=only_lumi, calibration=CAL, darl=PLENTY),
                    dataclasses.replace(cfg, assume_overhead=True))
    assert len(lax.selection) == 1 and lax.score.solo_merges > 0
    assert lax.timeline.quality == "extrapolated"
    # Priced, ranked on request -- and still labelled, with the measurement that would
    # close it named rather than left as a shrug.
    named = [w for w in lax.warnings if "priced but NOT ranked" in w]
    assert named, lax.warnings
    assert "merge-complete to merge-complete" in named[0]
    assert "PWW_VAL_WINDOWS" in named[0]


@check("the wait is a distribution: p50 prices the plan, p90 decides feasibility")
def _():
    """'Will A still be alive when B arrives' is a feasibility question and a median is
    the wrong statistic for it. Here Snellius' queue reads 4 h at p50 and 19.9 h at
    p90, and LUMI has exactly 20 h of walltime:

        p50:  0 + 20 h >= 4 h + 300 s + one 346.9 s round      -> survives
        p90:  0 + 20 h >= 19.9 h + 300 s + 346.9 s = 20.08 h    -> TRAP

    The plan is still PRICED at p50 -- that is the best estimate of what will happen --
    but the feasibility clause is read at p90, so the risk is stated rather than
    averaged away.
    """
    site_a = dataclasses.replace(snellius([(24, 4.0)]),
                                 waits={"snellius_4g_24h": wait(4.0, p90_h=19.9)})
    cfg = config(chain_policies=("none",))
    opts, _ = options((site_a, lumi([(20, 0.0)])), cfg)
    selection = (next(o for o in opts["snellius"] if o.begin_s == 0.0),
                 next(o for o in opts["lumi"] if o.begin_s == 0.0))
    timeline = simulate(selection, config=cfg, calibration=CAL, darl=PLENTY, balance=False)
    # Priced at p50, so the plan does federate: Snellius arrives at 4 h, not 19.9 h.
    assert timeline.federated_merges > 100

    checks = {c.name: c for c in crosschecks(selection, timeline, config=cfg,
                                             calibration=CAL, options_by_site={})}
    assert checks["headstart_exists"].verdict == "yes"
    assert checks["early_site_survives_to_federate"].verdict == "TRAP"
    assert "(p90)" in checks["early_site_survives_to_federate"].detail
    # Read at p50 instead, the same plan looks safe -- which is the point of saying
    # which quantile a verdict came from.
    at_p50 = {c.name: c for c in crosschecks(
        selection, timeline, config=dataclasses.replace(cfg, feasibility_quantile="p50"),
        calibration=CAL, options_by_site={})}
    assert at_p50["early_site_survives_to_federate"].verdict == "yes"

    # When DARL binds, the headstart's second condition is a marginal-value one:
    # alpha against zeta_solo/zeta_fed = 64 / (32 + 64) = 0.667 for a LUMI headstart.
    bound = simulate(selection, config=cfg, calibration=CAL, darl=REAL, balance=False)
    assert bound.darl_exhausted_s is not None
    worth = {c.name: c for c in crosschecks(selection, bound, config=cfg, calibration=CAL,
                                            options_by_site={})}["headstart_worth_the_corpus"]
    assert worth.verdict == "no", worth.detail    # alpha 0.25 < 0.667
    assert "64/96 = 0.667" in worth.detail, worth.detail


@check("turning balancing on must not CREATE a data cap")
def _():
    """_auto_balance's SECOND clause, previously unexercised: the --balance auto check
    only drove the two cases the first clause already decides, so deleting
    `if on.darl_exhausted_s is not None: return False` left the suite green.

    The clause is the whole difference between "the extra sequences are free" and
    "they are free until they are not": a walltime-bound plan has corpus to spare, but
    balancing multiplies blocks/round by ~2.3x, and if THAT exhausts the corpus the
    run loses its tail. Both timelines have to be consulted, not just the unbalanced
    one."""
    from pww.plan.search import _auto_balance

    class T:                                  # only the two fields the rule reads
        def __init__(self, exhausted, tokens):
            self.darl_exhausted_s, self.tokens = exhausted, tokens

    # data-bound already: first clause, OFF.
    assert _auto_balance(T(1000.0, 5), T(None, 9)) is False
    # walltime-bound both ways and balancing buys tokens: ON.
    assert _auto_balance(T(None, 5), T(None, 9)) is True
    # walltime-bound UNBALANCED, but balancing exhausts the corpus: the second clause.
    assert _auto_balance(T(None, 5), T(2000.0, 9)) is False
    # and it is the exhaustion that decides it, not the token count -- same tokens,
    # opposite answers.
    assert _auto_balance(T(None, 5), T(None, 6)) is True
    assert _auto_balance(T(None, 5), T(9.0, 6)) is False


@check("PlanConfig.balance_max reaches the accumulation the plan is priced at")
def _():
    """The cap was reachable in balance_accums and dead everywhere above it: hardcoding
    it inside round_cost left all four suites green, so nothing checked that the field
    an operator sets is the ceiling the plan is actually priced and emitted at. It
    matters twice -- it is the linear multiplier on DARL burn, and the explicit
    PWW_GRAD_ACCUM the emitter writes bypasses run_train.sh's own PWW_BALANCE_MAX,
    which only caps in the derivation branch an explicit value turns off.

    Snellius wants 5x beside LUMI (1.675393/0.356347 = 4.70 -> 5). Held to 3, the
    tokens per round fall from 100*2048*(5*32+64) to 100*2048*(3*32+64), i.e. by
    exactly the ratio 224/160, and the plan is priced on the lower number.
    """
    def plan_at(cap):
        cfg = config(balance="on", balance_max=cap, horizon_s=48 * HOUR)
        return make_plan(PlannerInputs(sites=(snellius([(8, 0.0)]), lumi([(8, 0.0)])),
                                       calibration=CAL, darl=PLENTY), cfg)

    accums = lambda p: {l.site: l.accums for l in p.timeline.ledgers}
    assert accums(plan_at(8)) == {"snellius": (5,), "lumi": (1,)}, accums(plan_at(8))
    assert accums(plan_at(3)) == {"snellius": (3,), "lumi": (1,)}, accums(plan_at(3))
    # 1 is balancing switched off by arithmetic rather than by flag, and the plan is
    # priced accordingly rather than merely labelled.
    assert accums(plan_at(1)) == {"snellius": (1,), "lumi": (1,)}, accums(plan_at(1))
    assert plan_at(3).timeline.tokens < plan_at(8).timeline.tokens
    assert plan_at(1).timeline.tokens < plan_at(3).timeline.tokens


@check("equal-utility plans are ordered by fewer jobs then sooner, everywhere")
def _():
    """_rank_key's tiebreak, previously able to be zeroed without a failure. It is not
    cosmetic: alpha_breakpoints compares plans through _tiebreak, so if the two
    disagree the reported alpha* belongs to a plan the search would not have returned,
    and equal-utility results reorder run to run for no reason a diff can explain."""
    from pww.plan.search import _rank_key, _tiebreak

    class S:
        utility, federated_merges, solo_merges, tokens = 10.0, 10, 0, 0.0
        alpha, beta = 0.25, 0.0

    class E:
        score, traps, rankable = S(), (), True
        def __init__(self, sel): self.selection = sel

    def opt(links, begin):
        base = next(iter(options((snellius([(8, 0.0)]),), config())[0]["snellius"]))
        return dataclasses.replace(base, lanes=1, links_per_lane=links, begin_s=begin)

    few_soon, many_soon = E((opt(1, 0.0),)), E((opt(6, 0.0),))
    few_late = E((opt(1, 3600.0),))
    # fewer jobs wins at equal utility ...
    assert _rank_key(few_soon) > _rank_key(many_soon)
    # ... and among equal job counts, the one submitted sooner wins.
    assert _rank_key(few_soon) > _rank_key(few_late)
    # _tiebreak must agree with _rank_key's tail, or alpha* names another plan.
    for ev in (few_soon, many_soon, few_late):
        assert _tiebreak(ev.selection) == _rank_key(ev)[-2:], ev.selection


@check("cross-checks and NUM_ROUNDS use the balance the plan RESOLVED to, not the flag")
def _():
    """REGRESSION. Under the default --balance auto the decision is made per selection
    by _auto_balance, so `config.balance == "on"` is not the answer -- it is the
    QUESTION. A plan that auto-resolved to ON was cross-checked at accums (1,1) and had
    its NUM_ROUNDS sized against the unbalanced timeline, i.e. against a plan the
    planner did not choose. marginal_ledger already read the resolved value, so the
    three consumers of one fact disagreed."""
    cfg = config(balance="auto", horizon_s=48 * HOUR)
    plan = make_plan(PlannerInputs(sites=(snellius([(8, 0.0)]), lumi([(8, 0.0)])),
                                   calibration=CAL, darl=PLENTY), cfg)
    assert plan.config.balance == "auto"
    accums = {l.site: l.accums for l in plan.timeline.ledgers}
    # This scenario is only interesting because auto actually turned balancing ON.
    assert any(a > 1 for v in accums.values() for a in v), accums
    assert accums["snellius"] == (5,), accums

    # NUM_ROUNDS must be sized against the timeline the plan actually recommends.
    from pww.plan.search import _recommend_num_rounds, _resolved_balance
    assert _resolved_balance(plan.timeline) is True
    balanced = _recommend_num_rounds(plan.selection, cfg, CAL, PLENTY, False, balance=True)
    unbalanced = _recommend_num_rounds(plan.selection, cfg, CAL, PLENTY, False, balance=False)
    assert balanced != unbalanced, (balanced, unbalanced)
    assert plan.recommended_num_rounds == balanced, (plan.recommended_num_rounds, balanced)

    # The CROSS-CHECK half. NUM_ROUNDS alone left this reversible: putting
    # `config.balance == "on"` back into the crosschecks() round_cost call kept the
    # suite green, so the half that reaches the operator's report was unasserted.
    # min_federated_rounds is raised only to make the period legible at the two
    # decimals the detail prints -- the error is there at 1 round too, just under the
    # rounding. needed = w_b + c_late + n * period, and the period is the one the
    # resolved regime runs at:
    #   balanced   900 s + 100 * 357.5420 = 36654.2 s = 10.18 h
    #   unbalanced 900 s + 100 * 346.9075 = 35590.8 s =  9.89 h
    loud = config(balance="auto", horizon_s=48 * HOUR, min_federated_rounds=100)
    plan = make_plan(PlannerInputs(sites=(snellius([(8, 0.0)]), lumi([(8, 0.0)])),
                                   calibration=CAL, darl=PLENTY), loud)
    assert _resolved_balance(plan.timeline) is True
    survives = {c.name: c for c in plan.crosschecks}["early_site_survives_to_federate"]
    assert "need 10.18 h" in survives.detail, survives.detail
    assert "9.89 h" not in survives.detail, survives.detail

    # ... and an auto plan that resolved OFF is still cross-checked at accums (1,1).
    data_bound = make_plan(PlannerInputs(sites=(snellius([(8, 0.0)]), lumi([(8, 0.0)])),
                                         calibration=CAL, darl=REAL), config(balance="auto"))
    off = {l.site: l.accums for l in data_bound.timeline.ledgers}
    assert all(a == 1 for v in off.values() for a in v), off
    assert _resolved_balance(data_bound.timeline) is False


@check("the headstart's corpus price carries the accumulation the job will really run")
def _():
    """REGRESSION. PWW_GRAD_ACCUM is fixed for the whole job, so a solo round at an
    accumulating site burns accum * batch_seq sequences. zeta_fed always carried the
    multiplier and zeta_solo did not, so the ratio understated the solo cost by exactly
    the early site's accumulation -- the same defect that had the simulator pricing
    solo rounds at 1/5 of their real corpus burn, fixed there and left standing in the
    closed form PLANNER.md advertises as its counterpart.

    It bites precisely where it matters: the multiplier is >1 at the FAST site, which
    is the site with the short queue and therefore the one that takes the headstart.
    Here Snellius (batch 32, step 0.36 s) accumulates 5x against LUMI (batch 64, step
    1.68 s), so a solo Snellius round spends 160 sequences, not 32.
    """
    cfg = config(chain_policies=("none",), balance="on")
    opts, _ = options((snellius([(24, 0.0)]), lumi([(20, 6.0)])), cfg)
    selection = (next(o for o in opts["snellius"] if o.begin_s == 0.0),
                 next(o for o in opts["lumi"] if o.begin_s == 0.0))
    # The accumulation the SIMULATOR charges -- the number the cross-check must agree
    # with, since the two are supposed to answer the same question two ways.
    members = [make_member(f"{o.site}-l0", o.candidate) for o in selection]
    cost = round_cost(members, inner_steps=cfg.inner_steps, calibration=CAL,
                      balance=True, balance_max=cfg.balance_max)
    accums = dict(zip((m.site for m in members), cost.accums))
    assert accums["snellius"] == 5 and accums["lumi"] == 1, accums

    bound = simulate(selection, config=cfg, calibration=CAL, darl=REAL, balance=True)
    assert bound.darl_exhausted_s is not None, "this scenario is supposed to be data-bound"
    worth = {c.name: c for c in crosschecks(selection, bound, config=cfg, calibration=CAL,
                                            options_by_site={})}["headstart_worth_the_corpus"]
    # 5*32 = 160 solo against 5*32 + 1*64 = 224 federated, not 32/224 = 0.143.
    assert "160/224 = 0.714" in worth.detail, worth.detail
    # and the verdict is the one that number implies: alpha 0.25 < 0.714, so the
    # headstart is NOT worth the corpus. Priced at 0.143 it read "yes" -- the
    # cross-check recommended a headstart the planner's own simulator refuses.
    assert worth.verdict == "no", worth.detail


@check("the closed-form duty cycle agrees with the simulator, and disagreements surface")
def _():
    """The inequalities are CROSS-CHECKS, never a second decision path. They agree
    where their assumption holds -- a steady state in which each job's queue wait
    amortises over its own length -- and they must disagree visibly where it does not:
    over a FINITE horizon a 24 h job pays a 20 h wait once and is productive for 3.9 h,
    while a chain of 2 h jobs pays its wait once too and keeps going. The simulator is
    authoritative because it is the one that knows about the horizon.
    """
    def solve_one(w_short_h, c_min):
        cfg = config(horizon_s=24 * HOUR, max_links_per_lane=12, alpha=1.0)
        plan = make_plan(PlannerInputs(
            sites=(snellius([(2, w_short_h), (24, 20.0)], startup_s=c_min * 60.0),),
            calibration=CAL, darl=PLENTY), cfg)
        crosscheck = next(c for c in plan.crosschecks if c.name.startswith("chain_or"))
        return plan.selection[0], crosscheck

    # Steep queue, cheap startup: c = 5 min against c* = 55.5 min. Both say chain.
    option, crosscheck = solve_one(0.05, 5)
    assert option.links_per_lane == 12 and crosscheck.verdict == "chain"
    assert crosscheck.agrees is True

    # Flat-ish queue: w(2 h) = 10 h against w(24 h) = 20 h gives c* = -375 min, i.e.
    # never chain, and the simulator agrees because 10 h of queue on a 2 h job is
    # ruinous however often you repeat it.
    option, crosscheck = solve_one(10.0, 5)
    assert option.links_per_lane == 1 and crosscheck.verdict == "one long job"
    assert crosscheck.agrees is True

    # The instructive cell: c = 30 min against c* = 23.4 min, so the steady-state
    # inequality says "one long job" while the simulator still chains -- because over
    # 24 h the long job never gets out of the queue. The disagreement is REPORTED.
    option, crosscheck = solve_one(1.0, 30)
    assert option.links_per_lane == 12, "the simulator is authoritative"
    assert crosscheck.verdict == "one long job" and crosscheck.agrees is False
    # ...and both sides of it are legible, so a reader can decide which to believe.
    assert "c* = " in crosscheck.detail and "measured c of 30.0 min" in crosscheck.detail
    assert close(chain_breakeven_c(2 * HOUR, 24 * HOUR, 1 * HOUR, 20 * HOUR) / 60,
                 23.4, 0.1)


@check("the same inputs give the same plan, twice, down to the bytes")
def _():
    sites = (snellius([(12, 2.0), (24, 6.0)]), lumi([(12, 0.0), (24, 1.0)]))
    cfg = config(horizon_s=36 * HOUR, alpha=0.25, balance="auto", assume_overhead=True)
    first = make_plan(PlannerInputs(sites=sites, calibration=CAL, darl=REAL), cfg)
    second = make_plan(PlannerInputs(sites=sites, calibration=CAL, darl=REAL), cfg)
    assert dataclasses.asdict(first) == dataclasses.asdict(second)
    assert first.describe() == second.describe()
    # Site order in the inputs must not move the answer either.
    flipped = make_plan(PlannerInputs(sites=(sites[1], sites[0]), calibration=CAL,
                                      darl=REAL), cfg)
    assert flipped.describe() == first.describe()


@check("NUM_ROUNDS is recommended against an unbounded budget, not the one that bound")
def _():
    """A round attempt is consumed by every STARTED round, solo ones included. Setting
    it too high costs nothing; too low ends the run early."""
    cfg = config(horizon_s=12 * HOUR, num_rounds=10, alpha=1.0)
    plan = make_plan(PlannerInputs(sites=(snellius([(12, 0.0)], startup_s=0.0),),
                                   calibration=CAL, darl=PLENTY), cfg)
    assert plan.score.attempts == 10, "the configured budget bound this plan"
    # 12 h / 111.635 s = 386 rounds, x1.15 + one lane + 10 = 455
    assert plan.recommended_num_rounds == 455, plan.recommended_num_rounds


@check("site submission limits are echoed as assumptions, because they are")
def _():
    """MaxSubmitJobs, MaxRunningJobs and MaxArraySize are not readable from the
    aggregator VM -- there is no sbatch, sacct or scontrol on it."""
    s = snellius([(12, 0.0)])
    assert s.limits.source == "assumed" and s.limits.max_submit_jobs is None
    plan = make_plan(PlannerInputs(sites=(s,), calibration=CAL, darl=PLENTY), config())
    assert any("ASSUMED" in w and "scontrol" in w for w in plan.warnings), plan.warnings
    read = dataclasses.replace(s, limits=SiteLimits(max_submit_jobs=100, source="scontrol"))
    plan = make_plan(PlannerInputs(sites=(read,), calibration=CAL, darl=PLENTY), config())
    assert not any("ASSUMED" in w for w in plan.warnings)


# --------------------------------------------------------------------------
# degenerate inputs -- all of which are normal on an HPC queue
# --------------------------------------------------------------------------


@check("zero sites, one site and an empty corpus all give a plan rather than a crash")
def _():
    cfg = config(horizon_s=12 * HOUR, alpha=1.0)

    nothing = make_plan(PlannerInputs(sites=(), calibration=CAL, darl=PLENTY), cfg)
    assert nothing.selection == () and nothing.score.utility == 0.0
    assert nothing.marginal == () and nothing.crosschecks == ()
    assert "empty plan" in nothing.describe()

    alone = make_plan(PlannerInputs(sites=(snellius([(12, 0.0)], startup_s=0.0),),
                                    calibration=CAL, darl=PLENTY), cfg)
    assert len(alone.selection) == 1 and alone.score.federated_merges == 0
    assert alone.score.solo_merges == 386 and not alone.traps  # solo is not a trap
    # No partner, so no headstart check; and only ONE walltime is probed at this
    # geometry, so the duty-cycle check reports that it cannot be evaluated rather
    # than quietly vanishing -- "nobody probed a second walltime here" is a fact
    # about the collector config and the reason chaining cannot be argued either way.
    assert [c.name for c in alone.crosschecks] == ["chain_or_one_long_job[snellius@4]"]
    assert alone.crosschecks[0].verdict == "not evaluable"
    assert "only one walltime is probed" in alone.crosschecks[0].detail

    spent = make_plan(PlannerInputs(sites=(snellius([(12, 0.0)]), lumi([(12, 0.0)])),
                                    calibration=CAL, darl=EMPTY), cfg)
    assert spent.score.federated_merges == spent.score.solo_merges == 0
    assert spent.timeline.blocks_available == 0.0
    assert spent.timeline.darl_exhausted_s == 0.0


@check("a federation where every probe is stale plans nothing and says why")
def _():
    sites = (snellius([(12, 2.0)]), lumi([(12, 0.0)]))
    sites = tuple(dataclasses.replace(
        s, waits={k: dataclasses.replace(v, probe_age_s=30 * HOUR) for k, v in s.waits.items()})
        for s in sites)
    plan = make_plan(PlannerInputs(sites=sites, calibration=CAL, darl=REAL), config())
    assert plan.selection == ()
    codes = [e.code for e in plan.exclusions]
    assert codes.count("probe_stale") == 2 and codes.count("site_unusable") == 2, codes
    assert all("collector cron" in e.fix for e in plan.exclusions if e.code == "probe_stale")


@check("a site missing from the throughput registry is excluded, never interpolated")
def _():
    """Step time is device-count invariant to within 8% on the measured pairs, but that
    is a finding about two sites on one model, not a licence."""
    s = snellius([(12, 0.0)])
    # A 1-GPU shape at a site whose registry only holds the 4-GPU cell.
    s = dataclasses.replace(
        s, shapes=s.shapes + (shape("snellius", "gpu_h100", 1, 12.0),),
        waits={**s.waits, "snellius_1g_12h": wait(0.1)})
    cands, excl = admit(s, config=config(), calibration=CAL)
    assert [c.geometry.gpus for c in cands] == [4]
    missing = next(e for e in excl if e.code == "no_throughput")
    assert "snellius@1 devices" in missing.reason
    assert "PWW_TPUT_SNELLIUS_1" in missing.fix


# --------------------------------------------------------------------------
# inputs: the only module that touches the outside world
# --------------------------------------------------------------------------


@check("Slurm duration syntax is parsed in all seven of its forms")
def _():
    """The walltime is part of the shape KEY, so a mis-parse silently keys a wait to
    the wrong shape -- the precise failure the verbatim-args rule exists to prevent
    downstream."""
    assert io.parse_slurm_time("30") == 1800                 # MM
    assert io.parse_slurm_time("2:30") == 150                # MM:SS
    assert io.parse_slurm_time("8:00:00") == 28800           # HH:MM:SS
    assert io.parse_slurm_time("40:00:00") == 144000         # HH:MM:SS past a day
    assert io.parse_slurm_time("1-00:00:00") == 86400        # d-HH:MM:SS
    assert io.parse_slurm_time("1-12") == 129600             # d-HH
    assert io.parse_slurm_time("2-06:30") == 196200          # d-HH:MM
    for bad in ("infinite", "UNLIMITED", "soon", ""):
        try:
            io.parse_slurm_time(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} parsed as a walltime")


@check("the shape key is recovered from the probe's own args, walltime included")
def _():
    """Nothing upstream parses this string -- not the collector, not the server, not
    the upstream planner -- which is exactly why that planner quotes a wait measured
    for an 8 h job and then tells you to submit a 12.4 h one."""
    key = io.parse_shape_args(
        "snellius", "-p gpu_h100 -N 1 --gpus-per-node 4 --cpus-per-task 64 -t 40:00:00")
    assert key.partition == "gpu_h100" and key.gpus == 4 and key.walltime_s == 144000
    lumi_key = io.parse_shape_args(
        "lumi", "-A project_462000226 -p standard-g -N 2 --gpus-per-node 8 -t 8:00:00")
    assert lumi_key.account == "project_462000226" and lumi_key.gpus == 16   # -N x per node
    assert io.parse_shape_args("x", "--partition=gpu --gpus-per-node=1 --time=1:00:00").gpus == 1
    for bad in ("-p gpu_h100 -t 8:00:00", "--gpus-per-node 4 -t 8:00:00", ""):
        try:
            io.parse_shape_args("snellius", bad)
        except ValueError as exc:
            assert "shape key cannot be recovered" in str(exc), exc
            continue
        raise AssertionError(f"{bad!r} produced a shape key out of nothing")


@check("probes.csv is read as a distribution, and one bad cell does not poison the table")
def _():
    """Upstream's read() lets a malformed numeric cell raise, and one bad row then 500s
    every query for that cluster. A planner that dies on one bad row is worse than one
    that skips it."""
    now = int(time.time())
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "snellius").mkdir()
        header = ("collected_at,name,args,ok,estimated_start,estimated_wait_sec,"
                  "placed_partition,placed_nodes,message,probed_by_user,collector_version")
        args = "-p gpu_h100 -N 1 --gpus-per-node 4 -t 8:00:00"
        rows = [f"{now - 600},h100_full_8h,{args},True,{now},1260,gpu_h100,gcn82,ok,zpalanciya,2.0.0",
                f"{now - 300},h100_full_8h,{args},True,{now},3600,gpu_h100,gcn82,ok,zpalanciya,2.0.0",
                f"{now},h100_full_8h,{args},True,{now},7200,gpu_h100,gcn82,ok,zpalanciya,2.0.0",
                f"{now},wrecked,{args},True,,not-a-number,gpu_h100,,msg,zpalanciya,2.0.0"]
        (root / "snellius" / "probes.csv").write_text(header + "\n" + "\n".join(rows) + "\n")
        usage = ("collected_at,window_start,window_end,window_hours,partition,n_jobs,"
                 "n_timeout,sum_elapsed_sec,sum_timelimit_sec,used_ratio,collector_version")
        (root / "snellius" / "usage.csv").write_text(
            usage + f"\n{now},{now - 172800},{now},48,gpu_h100,240,3,43581,388400,0.1122,2.0.0\n")

        probes = io.read_csv_table(root, "snellius", "probes")
        assert len(probes) == 4 and probes[0]["estimated_wait_sec"] == 1260
        assert probes[-1]["estimated_wait_sec"] is None
        assert probes[-1]["_bad_cells"] == ["estimated_wait_sec"]
        assert probes[0]["cluster"] == "snellius"        # injected, not a CSV column
        usage_rows = io.read_csv_table(root, "snellius", "usage")
        assert usage_rows[0]["used_ratio"] == 0.1122
        assert usage_rows[0]["ok"] is False, "usage.csv has no ok column; do not branch on it"

        shapes, waits, excl = io.build_shapes("snellius", probes, usage_rows, now=now)
        got = waits["h100_full_8h"]
        # p50 of [1260, 3600, 7200] = 3600; p90 interpolates at 0.9*(3-1) = 1.8:
        # 3600 + 0.8*(7200-3600) = 6480
        assert got.p50_raw_s == 3600.0 and got.p90_raw_s == 6480.0
        # w_eff = 3600 * (1 - 0.5*(1 - 0.1122)) = 2001.96, the upstream blend verbatim
        assert close(got.p50_eff_s, 2001.96, 1e-9)
        assert close(got.eff_at(0.5, "p50"), 2001.96, 1e-9)
        assert got.eff_at(0.0, "p50") == 3600.0 and close(got.eff_at(1.0, "p50"), 403.92, 1e-9)
        assert got.samples == 3 and got.discounted and got.probed_by_user == "zpalanciya"
        assert next(s for s in shapes if s.name == "h100_full_8h").args == args
        assert excl == [] or all(e.fix for e in excl)


@check("the discount silently no-ops when there is no usage row, and says so")
def _():
    """A partition whose sacct window found no finishing jobs produces no usage row at
    all, so the plan LOOKS discounted when it is not. The only tell is used_ratio being
    null, which is why `discounted` is carried on every reading."""
    for ratio in (None, 0.0, -0.1, 1.5):
        w = wait(2.0, used_ratio=ratio)
        assert not w.discounted, ratio
        assert w.eff_at(0.5, "p50") == 2 * HOUR == w.raw("p50")
        assert io._discount(7200.0, ratio, 0.5) == 7200.0
    w = wait(2.0, used_ratio=0.1122)
    assert w.discounted and close(w.eff_at(0.5, "p50"), 7200 * (1 - 0.5 * (1 - 0.1122)))
    # p90 is a different question from p50 and must not be quietly substituted.
    spread = wait(2.0, p90_h=10.0)
    assert spread.raw("p50") == 7200.0 and spread.raw("p90") == 36000.0


@check("the throughput registry is read keyed on (site, devices) and never guessed")
def _():
    registry, excl = io.load_throughput(ROOT / "configs" / "site_throughput.env")
    assert excl == [], excl
    # PWW_BATCH_ is dp_ranks x local_batch_size, so the device count is batch/8 -- a
    # reading of the file's own definition, not an inference about it.
    assert registry["snellius"][4].tput_seq_s == 89.8 and registry["snellius"][4].batch_seq == 32
    assert registry["lumi"][8].tput_seq_s == 38.2 and registry["lumi"][8].batch_seq == 64
    assert close(registry["snellius"][4].step_s, 32 / 89.8, 1e-12)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "reg.env"
        path.write_text(
            "# a comment\n"
            "export PWW_TPUT_SNELLIUS_1=24.4\n"
            "PWW_BATCH_SNELLIUS_1=8\n"
            "PWW_TPUT_LUMI=38.2\n"          # no matching PWW_BATCH_
            "PWW_TPUT_VEGA=abc\nPWW_BATCH_VEGA=xyz\n")
        registry, excl = io.load_throughput(path)
        assert registry["snellius"][1].tput_seq_s == 24.4     # explicit device suffix
        assert close(registry["snellius"][1].step_s, 8 / 24.4, 1e-12)
        codes = sorted(e.code for e in excl)
        assert codes == ["bad_registry_value", "half_calibrated"], codes
        assert all(e.fix for e in excl)
    missing, excl = io.load_throughput(Path(tmp) / "gone.env")
    assert missing == {} and excl[0].code == "no_registry"


@check("a short option with an attached value parses, rather than silently halving -N")
def _():
    """`-N2` is valid Slurm and was read as nodes=1, because only the space-separated
    form was recognised. Nothing failed and nothing was printed: the device count used
    to pick the throughput cell was halved, so a 2-node wait was priced against the
    1-node step time -- while the emitter, which copies the string verbatim, still
    submitted a 2-node job. Same for -pPART, -t8:00:00 and -APROJ."""
    attached = io.parse_shape_args("lumi", "-N2 -pstandard-g --gpus-per-node 8 -t8:00:00")
    spaced = io.parse_shape_args("lumi", "-N 2 -p standard-g --gpus-per-node 8 -t 8:00:00")
    assert attached == spaced, (attached, spaced)
    assert attached.nodes == 2 and attached.gpus == 16, attached
    assert attached.partition == "standard-g" and attached.walltime_s == 8 * 3600
    assert io.parse_shape_args(
        "lumi", "-Aproj -p standard-g --gpus-per-node 8 -t 1:00:00").account == "proj"
    # the long forms and the space-separated forms are untouched
    long = io.parse_shape_args(
        "lumi", "--nodes=2 --partition=standard-g --gpus-per-node=8 --time=8:00:00")
    assert long.nodes == 2 and long.gpus == 16 and long.walltime_s == 8 * 3600, long


@check("a present-but-unreadable wait cell is a named refusal, not a ValueError")
def _():
    """The CSV path turned an unparseable numeric cell into a `no_wait_reading`
    exclusion; the JSON path fed it straight to float() and raised a bare ValueError
    out of the CLI -- exit 1, zero bytes on stdout, no plan, no exclusion and no
    indication of which cluster or shape was malformed. A missing wait is still never
    a zero wait: the shape is refused, not priced as starting immediately."""
    rows = [{"cluster": "snellius", "name": "s1", "ok": True,
             "args": "-p gpu_h100 -N 1 --gpus-per-node 4 -t 8:00:00",
             "collected_at": 1000, "estimated_wait_sec": bad}
            for bad in ("n/a", "", "null")]
    shapes, waits, excl = io.build_shapes("snellius", rows, [], now=2000.0,
                                          discount_strength=0.5)
    assert not shapes and not waits, (shapes, waits)
    refusal = [e for e in excl if e.code == "no_wait_reading"]
    assert len(refusal) == 1, excl
    assert "unreadable numeric cells" in refusal[0].reason, refusal[0].reason
    assert "'n/a'" in refusal[0].reason, refusal[0].reason
    assert refusal[0].fix, refusal[0]
    # A readable row among unreadable ones still prices the shape.
    rows.append({"cluster": "snellius", "name": "s1", "ok": True,
                 "args": "-p gpu_h100 -N 1 --gpus-per-node 4 -t 8:00:00",
                 "collected_at": 1500, "estimated_wait_sec": 3600})
    shapes, waits, excl = io.build_shapes("snellius", rows, [], now=2000.0,
                                          discount_strength=0.5)
    assert waits["s1"].p50_raw_s == 3600.0, waits
    assert not [e for e in excl if e.code == "no_wait_reading"], excl


@check("a throughput or batch that is not POSITIVE is refused by name, not divided by")
def _():
    """REGRESSION. Numeric was checked; positive was not, and only on one of the pair.
    Geometry.step_s is batch/tput, so PWW_BATCH_<SITE>=0 was a ZeroDivisionError out
    of the CLI with zero bytes on stdout, and PWW_BATCH_<SITE>=-32 was worse: a full
    plan at exit 0 with a negative step time, negative tokens and negative DARL
    blocks, so blocks_left never decreased and the corpus never exhausted -- the run
    was scored all the way to the horizon. Both come from one typo in
    configs/site_throughput.env, which is why rounds.site_overhead_s already refuses
    the throughput half by name; this is the sibling it was missing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        for bad, culprit in ((["PWW_TPUT_SNELLIUS_4=0", "PWW_BATCH_SNELLIUS_4=32"],
                              "PWW_TPUT_SNELLIUS_4"),
                             (["PWW_TPUT_SNELLIUS_4=89.8", "PWW_BATCH_SNELLIUS_4=0"],
                              "PWW_BATCH_SNELLIUS_4"),
                             (["PWW_TPUT_SNELLIUS_4=89.8", "PWW_BATCH_SNELLIUS_4=-32"],
                              "PWW_BATCH_SNELLIUS_4"),
                             (["PWW_TPUT_SNELLIUS_4=-89.8", "PWW_BATCH_SNELLIUS_4=32"],
                              "PWW_TPUT_SNELLIUS_4")):
            path = Path(tmp) / "reg.env"
            # A GOOD cell alongside, so the refusal is per-cell and not a whole-file
            # bail: the plan must still be able to fall back to snellius@1.
            path.write_text("PWW_TPUT_SNELLIUS_1=24.4\nPWW_BATCH_SNELLIUS_1=8\n"
                            + "\n".join(bad) + "\n")
            registry, excl = io.load_throughput(path)
            assert 4 not in registry.get("snellius", {}), (bad, registry)
            assert registry["snellius"][1].tput_seq_s == 24.4, registry
            hit = [e for e in excl if e.code == "bad_registry_value"]
            assert len(hit) == 1, (bad, excl)
            assert culprit in hit[0].reason, (culprit, hit[0].reason)
            assert "not positive" in hit[0].reason, hit[0].reason
            assert hit[0].fix, hit[0]                 # an exclusion must be actionable


@check("the balance cross-check never divides by a registry cell it did not validate")
def _():
    """REGRESSION. balance_crosscheck is handed the RAW registry, so admission's
    positive-throughput guard has not run on it. A single zeroed cell reaching
    `cells[gpus].step_s` raised ZeroDivisionError AFTER the plan had been computed:
    the CLI exited 1 with nothing at all on stdout -- no plan, no exclusion, no
    diagnosis -- while `show` and `--json` on the same inputs printed the proper
    refusal. Skipped and NAMED, because a missing comparison is not evidence that the
    planner and the shell agree."""
    from pww.plan import emit as emit_mod
    cfg = config()
    plan = make_plan(PlannerInputs(sites=(snellius([(8, 0.0)]), lumi([(8, 0.0)])),
                                   calibration=CAL, darl=PLENTY), cfg)
    raw = {"snellius": {4: Geometry(site="snellius", gpus=4, tput_seq_s=0.0, batch_seq=32)},
           "lumi": {8: Geometry(site="lumi", gpus=8, tput_seq_s=38.2, batch_seq=64)}}
    lines = emit_mod.balance_crosscheck(plan, CAL, raw)   # must not raise
    assert any("no shell comparison for snellius" in ln for ln in lines), lines
    # A batch of 0 divides just as well as a throughput of 0.
    raw["snellius"][4] = Geometry(site="snellius", gpus=4, tput_seq_s=89.8, batch_seq=0)
    assert any("no shell comparison for snellius" in ln
               for ln in emit_mod.balance_crosscheck(plan, CAL, raw))
    # ... and a registry that IS valid still produces the comparison it exists for.
    raw["snellius"][4] = Geometry(site="snellius", gpus=4, tput_seq_s=89.8, batch_seq=32)
    ok = emit_mod.balance_crosscheck(plan, CAL, raw)
    assert not any("no shell comparison" in ln for ln in ok), ok
    assert any("snellius: planner" in ln for ln in ok), ok


@check("DARL liveness is gated on last_seen, not on presence in the clusters map")
def _():
    """The `clusters` map is a durable membership RECORD. A coordinator resumed from
    its snapshot lists every cluster that ever registered, with a stale last_seen --
    scripts/follow_watch.sh was burned by exactly this."""
    now = 1_787_000_000.0
    payload = {"clusters": {"snellius": {"last_seen": now - 30},
                            "lumi": {"last_seen": now - 86400},
                            "vega": {}}}
    live = io.darl_liveness(payload, fresh_within_s=300.0, now=now)
    assert live == {"snellius": True, "lumi": False, "vega": False}, live


@check("a coordinator that disagrees with the config is refused, not planned against")
def _():
    """BlockSpace.digest is checked at registration, so a mismatch is not a warning:
    every site would be refused with HTTP 400 and the plan would describe a run that
    cannot start."""
    from pww.darl.space import BlockSpace

    space = BlockSpace(num_samples=2_756_597, block_size=1024, seed=42)
    good = DarlState(num_blocks=space.num_blocks, committed=1870, leased=0, unassigned=822,
                     digest=space.digest(0), source="http://145.38.206.143:29510")
    assert io.check_darl_digest(good, num_samples=2_756_597, block_size=1024, seed=42) == []
    wrong = dataclasses.replace(good, digest="deadbeef")
    problems = io.check_darl_digest(wrong, num_samples=2_756_597, block_size=1024, seed=42)
    assert [p.code for p in problems] == ["darl_digest_mismatch"]
    assert "space.env" in problems[0].fix
    # A different arm's coordinator on a different port answers a different question.
    other_arm = dataclasses.replace(good, num_blocks=1000, digest="")
    problems = io.check_darl_digest(other_arm, num_samples=2_756_597, block_size=1024, seed=42)
    assert [p.code for p in problems] == ["darl_geometry_mismatch"]
    assert "check the port" in problems[0].fix


@check("the shipped calibration file reproduces the built-in table exactly")
def _():
    """The defaults live in code because a self-check that depends on an editable
    config checks nothing. The file overrides -- and it must not drift from the code
    it overrides."""
    from pww.plan import residuals

    calibration, problems = io.load_calibration(ROOT / "configs" / "plan" / "federation.json")
    assert problems == [], problems
    a = [(r["label"], round(r["rel_error"], 9)) for r in residuals(calibration)]
    b = [(r["label"], round(r["rel_error"], 9)) for r in residuals(CAL)]
    assert a == b, list(zip(a, b))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cal.json"
        path.write_text(json.dumps({"sites": {
            "snellius": {"xfer": {"value_s": 33.5},                 # no quality flag
                         "eval_fix": {"value_s": 23.6, "quality": "identified"}}}}))
        calibration, problems = io.load_calibration(path)
        assert [p.code for p in problems] == ["unlabelled_calibration"]
        assert calibration.sites["snellius"].xfer.quality == "extrapolated", \
            "an unlabelled number must be assumed worst, not identified"
    missing, problems = io.load_calibration(Path(tmp) / "gone.json")
    assert missing is not None and problems[0].code == "no_calibration_file"
    assert io.load_calibration(None)[0] is CAL


# --------------------------------------------------------------------------
# reading the outside world: a missing number is never a convenient number
# --------------------------------------------------------------------------


def _probe_row(name, args, wait, *, ok=True, at=1_000_000.0):
    return {"name": name, "args": args, "ok": ok, "estimated_wait_sec": wait,
            "collected_at": at, "placed_partition": "gpu_h100",
            "probed_by_user": "douwew", "message": ""}


@check("an unreadable estimated_wait_sec is a refusal, never a zero wait")
def _():
    """REGRESSION. _coerce turns a malformed numeric cell into None and records it in
    _bad_cells, which nothing read; build_shapes then fell back to
    `float(newest.get('estimated_wait_sec') or 0.0)` -- and a 0 s queue wait is the
    most optimistic answer the planner can give. On the shipped fixture, corrupting
    the 24 h100_full_8h rows moved Snellius from a 13.3 h p50 to `starts now` with no
    exclusion, and flipped the recommendation. Upstream's own planner skips such a row
    (server/plan.py:216)."""
    args = "-p gpu_h100 --gpus-per-node 4 -t 8:00:00"
    rows = [_probe_row("h100_full_8h", args, None) for _ in range(24)]
    shapes, waits, excl = io.build_shapes("snellius", rows, [], now=1_000_000.0)
    assert shapes == () and waits == {}, (shapes, waits)
    assert [e.code for e in excl] == ["no_wait_reading"], excl
    assert "not a zero wait" in excl[0].reason, excl[0].reason
    # a readable row still works, and the bad ones are simply not sampled
    rows += [_probe_row("h100_full_8h", args, 3600.0)]
    shapes, waits, excl = io.build_shapes("snellius", rows, [], now=1_000_000.0)
    assert [s.name for s in shapes] == ["h100_full_8h"]
    assert waits["h100_full_8h"].p50_raw_s == 3600.0 and waits["h100_full_8h"].samples == 1


@check("waits measured at two different walltimes are never blended into one shape")
def _():
    """REGRESSION, and it is triggered by the edit the planner's OWN exclusions ask for.
    A shape is (partition, devices, WALLTIME); the probe history was grouped by NAME
    alone, so editing the -t of an entry in configs/slurm_probe/<site>.json kept the
    name and silently mixed the old walltime's waits into the new one's distribution --
    while the emitter went on copying the NEW args verbatim. That quotes a wait
    measured at 1 h and submits a 40 h job, which is the exact failure the verbatim-args
    rule exists to prevent."""
    short = "-p gpu_h100 --gpus-per-node 4 -t 1:00:00"
    long = "-p gpu_h100 --gpus-per-node 4 -t 40:00:00"
    rows = [_probe_row("h100_full", short, 600.0, at=100.0),
            _probe_row("h100_full", short, 700.0, at=200.0),
            _probe_row("h100_full", long, 180000.0, at=300.0),
            _probe_row("h100_full", long, 190000.0, at=400.0)]
    shapes, waits, excl = io.build_shapes("snellius", rows, [], now=1_000.0)
    assert shapes[0].key.walltime_s == 40 * HOUR and shapes[0].args == long
    wait = waits["h100_full"]
    # p50 of {180000, 190000} = 185000 s = 51.39 h. Blended with the two 1 h probes it
    # was 25.10 h -- a median dragged down by readings for a job half a day shorter.
    assert close(wait.p50_raw_s, 185000.0), wait.p50_raw_s
    assert wait.samples == 2, wait.samples
    assert [e.code for e in excl] == ["shape_args_changed"], excl
    assert "DISCARDED, not blended" in excl[0].reason


@check("a coordinator that answers 200 with the wrong body is a note, not a traceback")
def _():
    """The failure mode is specifically '--darl-url points at a live HTTP service that
    is not the coordinator', which is the operator error the module's own docstring
    warns about: 29510/29520/29530/29540 are four DARL epochs, 29511 is Flower and
    29513 is the scanner, all on one VM. A refused connection and a 404 were handled;
    a 200 with an unexpected body escaped as KeyError/ValueError."""
    from pww.plan import adapter

    for payload, kind in (({"detail": "Not Found"}, "KeyError"),
                          ({"num_blocks": "2692.0", "unassigned": 822}, "ValueError")):
        notes: list[str] = []
        raw = {"darl": {"status": payload, "url": "http://x:29511"}}
        src = adapter.Sources(fixture="f.json", darl_url=None, probe_config_dir=None)
        state, excl = adapter._darl_state(src, CAL, notes, [], raw)
        assert state.num_blocks == 2692 and state.source.startswith("assumed"), state
        assert any(kind in n for n in notes), (kind, notes)


@check("the default scanner is the one this checkout's own collectors post to")
def _():
    """REGRESSION. The shipped default was upstream's deployment, which does not answer
    from the aggregator VM, and the fix its error message named (:29513) is not up
    either -- so scripts/plan_campaign.sh, which passes no --scanner-url, could never
    produce a plan from live sources. The address that has the probes is the address
    the probes were POSTed to, and that is in the collector config."""
    from pww.plan import adapter

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "lumi.json").write_text(
            json.dumps({"server": "http://10.0.0.9:8000", "shapes": []}))
        assert adapter.default_scanner_url(tmp) == "http://10.0.0.9:8000"
    # no config dir at all -> the constant, and it is not upstream's
    assert adapter.default_scanner_url(None) == adapter.DEFAULT_SCANNER_URL
    assert adapter.DEFAULT_SCANNER_URL != adapter.UPSTREAM_SCANNER_URL
    # and the shipped configs agree with it
    assert adapter.default_scanner_url(str(ROOT / "configs" / "slurm_probe")) \
        == adapter.DEFAULT_SCANNER_URL


@check("--balance auto is decided by which budget binds, not by the objective")
def _():
    """REGRESSION. Auto kept balancing only when it raised U -- and it never can, since
    balancing can only move the barrier up (Snellius' accum 5 gives 1.782 s against
    LUMI's 1.675 s) and at the default beta = 0 the extra tokens are worth nothing. So
    `auto` was a fixed OFF policy while the flag, the help text and the report all said
    it chose from which budget binds."""
    from pww.plan.search import _Evaluator

    sites = (snellius([(12, 0.0)]), lumi([(12, 0.0)]))
    opts, _ = options(sites, config(balance="auto", horizon_s=12 * HOUR))
    pick = tuple(opts[s][0] for s in ("snellius", "lumi"))

    def decide(darl):
        ev = _Evaluator(config(balance="auto", horizon_s=12 * HOUR), CAL, darl, {})
        return ev(pick)

    capped = decide(REAL)                       # 822 blocks: the corpus binds
    assert capped.timeline.darl_exhausted_s is not None
    assert capped.balance is False, "under a data cap balancing costs merges for free"

    roomy = decide(PLENTY)                      # 1e6 blocks: walltime binds
    assert roomy.timeline.darl_exhausted_s is None
    assert roomy.balance is True, "walltime binds, so the extra sequences are free"
    assert roomy.timeline.tokens > decide(REAL).timeline.tokens


def main() -> int:
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for name, exc in FAILED:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
