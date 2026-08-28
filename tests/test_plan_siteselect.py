"""Site selection: of these sites, which gets the client shape running soonest.

    python3 tests/test_plan_siteselect.py

No pytest, no network -- every scenario is built from literals, same convention as
test_plan_search.py (this file deliberately re-defines its own tiny builders rather
than importing that one, so it stays readable standalone; a decorator that RUNS a
check at import time -- see `check` below -- means importing another test file would
silently execute its whole suite as a side effect).

THE ONE THING THIS FILE EXISTS TO PIN: the decision flips when the busy site flips.
A scheduler that always recommends the same site regardless of queue state is not
doing site selection, it is doing site preference -- and the two are indistinguishable
from a single scenario, which is why both directions are checked here.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASSED, FAILED = [], []
CHECK_TIMEOUT_S = 60


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


from pww.plan import DEFAULT_CALIBRATION as CAL, PlanConfig  # noqa: E402
from pww.plan.model import DarlState, Geometry, PlannerInputs, Shape, SiteInput, WaitEstimate  # noqa: E402
from pww.plan import inputs as io  # noqa: E402
from pww.plan.siteselect import client_candidate, rank_sites  # noqa: E402

HOUR = 3600.0
NO_DARL = DarlState(num_blocks=1, committed=0, leased=0, unassigned=1)


def wait(hours: float, **kw) -> WaitEstimate:
    p90 = kw.pop("p90_h", hours)
    return WaitEstimate(p50_raw_s=hours * HOUR, p90_raw_s=p90 * HOUR,
                        p50_eff_s=hours * HOUR, p90_eff_s=p90 * HOUR,
                        samples=kw.pop("samples", 3), probe_age_s=kw.pop("age_s", 60.0), **kw)


def shape(site: str, partition: str, gpus: int, hours: float, account: str | None = None) -> Shape:
    h, m = int(hours), int(round((hours - int(hours)) * 60))
    args = (f"-A {account} " if account else "") + \
        f"-p {partition} -N 1 --gpus-per-node {gpus} -t {h}:{m:02d}:00"
    return Shape(f"{site}_{gpus}g_{hours:g}h", io.parse_shape_args(site, args), args)


def site(name, partition, gpus, curve, *, tput=38.2, batch=64, startup_s=600.0,
        account=None, **kw) -> SiteInput:
    """One site with a w(T) curve: `curve` is [(walltime_hours, wait_hours), ...]."""
    shapes, waits = [], {}
    for hours, wait_h in curve:
        sh = shape(name, partition, gpus, hours, account)
        shapes.append(sh)
        waits[sh.name] = wait(wait_h)
    return SiteInput(site=name, shapes=tuple(shapes), waits=waits,
                     geometries={gpus: Geometry(name, gpus, tput, batch)},
                     startup_s=startup_s, **kw)


def lumi(curve, **kw) -> SiteInput:
    return site("lumi", "standard-g", 8, curve, tput=38.2, batch=64,
               account="project_462000226", **kw)


def snellius(curve, **kw) -> SiteInput:
    return site("snellius", "gpu_h100", 4, curve, tput=89.8, batch=32, **kw)


def cfg(**kw) -> PlanConfig:
    base = dict(horizon_s=48 * HOUR, num_rounds=1_000_000, lanes_max=4)
    base.update(kw)
    return PlanConfig(**base)


def inputs_of(*sites) -> PlannerInputs:
    return PlannerInputs(sites=tuple(sites), calibration=CAL, darl=NO_DARL)


# --------------------------------------------------------------------------
# client_candidate: fullest node, then longest walltime among ties
# --------------------------------------------------------------------------


@check("client_candidate picks the longest walltime at the largest device count")
def _():
    s = lumi([(1, 0.0), (4, 0.0), (40, 0.0)])
    from pww.plan.search import admit
    cands, excl = admit(s, config=cfg(), calibration=CAL)
    assert not excl, excl
    winner = client_candidate(cands)
    assert winner.shape.key.walltime_s == 40 * HOUR, winner.shape.name


@check("client_candidate prefers more devices over a longer walltime at fewer")
def _():
    # A 1-GPU 40h shape and an 8-GPU 1h shape: the full node wins regardless of
    # walltime, because scaling out is lanes-of-full-nodes, not partial ones.
    s = SiteInput(
        site="lumi",
        shapes=(shape("lumi", "small-g", 1, 40, "acct"),
               shape("lumi", "standard-g", 8, 1, "acct")),
        waits={"lumi_1g_40h": wait(0.0), "lumi_8g_1h": wait(0.0)},
        geometries={1: Geometry("lumi", 1, 4.75, 8), 8: Geometry("lumi", 8, 38.2, 64)},
        startup_s=600.0)
    from pww.plan.search import admit
    cands, excl = admit(s, config=cfg(), calibration=CAL)
    assert not excl, excl
    winner = client_candidate(cands)
    assert winner.gpus == 8, winner.gpus


# --------------------------------------------------------------------------
# rank_sites: the decision, and that it flips
# --------------------------------------------------------------------------


@check("lumi empty, snellius busy -> lumi wins")
def _():
    verdicts = rank_sites(
        inputs_of(lumi([(40, 0.0)]), snellius([(40, 13.27)])), cfg())
    assert verdicts[0].site == "lumi", verdicts
    assert verdicts[0].wait_s == 0.0
    assert verdicts[1].site == "snellius"


@check("the decision FLIPS: lumi busy, snellius empty -> snellius wins")
def _():
    verdicts = rank_sites(
        inputs_of(lumi([(40, 13.27)]), snellius([(40, 0.0)])), cfg())
    assert verdicts[0].site == "snellius", verdicts
    assert verdicts[0].wait_s == 0.0
    assert verdicts[1].site == "lumi"


@check("both sites admitted but tied -> deterministic order (site name), not a crash")
def _():
    verdicts = rank_sites(
        inputs_of(lumi([(40, 2.0)]), snellius([(40, 2.0)])), cfg())
    assert {v.wait_s for v in verdicts} == {2.0 * HOUR}
    # Order among exact ties is whatever `sorted` gives for equal keys (stable on
    # insertion order) -- pinned so a future refactor that reorders `inputs.sites`
    # is caught rather than silently changing which of two tied sites "wins".
    assert [v.site for v in verdicts] == ["lumi", "snellius"]


@check("a site with nothing admitted sorts last and reports why, not silently dropped")
def _():
    verdicts = rank_sites(
        inputs_of(lumi([(40, 0.0)]), snellius([])), cfg())
    assert verdicts[0].site == "lumi"
    assert verdicts[1].site == "snellius"
    assert verdicts[1].candidate is None
    assert verdicts[1].wait_s is None


@check("nothing admitted anywhere -> both report exclusions, no crash")
def _():
    verdicts = rank_sites(inputs_of(lumi([]), snellius([])), cfg())
    assert all(v.candidate is None for v in verdicts)
    assert all(v.exclusions for v in verdicts)


def main() -> int:
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for name, exc in FAILED:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
