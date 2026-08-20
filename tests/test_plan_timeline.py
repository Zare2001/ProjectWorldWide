"""The simulator: who is live when, and what that is worth.

    python3 tests/test_plan_timeline.py

This is the authoritative engine, so these are the checks that decide whether a plan
means anything. The closed forms in search.py are cross-checks against it, never a
second decision path -- because only a simulator that debits DARL blocks round by
round finds the result that matters most here: a solo headstart can EXHAUST THE CORPUS
before its partner arrives and yield zero federated merges, outcome-identical to the
staggered-start trap it was meant to avoid.

The scenarios are built from literals so every hour in them is hand-computable. Two
conventions used throughout:

  * a LANE is a durable identity (replica id, its own PWW_DUMP, its own DCP
    checkpoint, its own DARL cluster record); a LINK is one job inside it.
  * a round is FEDERATED when two distinct SITES contributed. Two lanes at one site
    are two Flower clients and the server does merge them, but that is not the
    measurement this campaign exists to make.
"""

from __future__ import annotations

import dataclasses
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


from pww.plan import (  # noqa: E402
    DEFAULT_CALIBRATION as CAL,
    Candidate,
    DarlState,
    Geometry,
    Option,
    PlanConfig,
    Shape,
    WaitEstimate,
    detect_traps,
    expand_links,
    make_schedule,
    simulate,
    site_overhead_s,
)
from pww.plan.inputs import parse_shape_args  # noqa: E402

HOUR = 3600.0

# The measured registry (configs/site_throughput.env), and the round costs that follow
# from it. Derived in tests/test_plan_rounds.py; repeated here as literals because
# every hour below is computed from them.
#   snellius@4  step 32/89.8 = 0.356347 s   o = 33.5+23.6+512/(3*89.8) =  59.0005 s
#   lumi@8      step 64/38.2 = 1.675393 s   o = 64.2+34.7+512/(3*38.2) = 103.3677 s
#   solo snellius period = 100*0.356347 + 17 +  59.0005            = 111.635 s
#   solo lumi     period = 100*1.675393 + 17 + 103.3677            = 287.907 s
#   two-site      period = 100*1.675393 + 17 + 59.0005 + 103.3677  = 346.908 s
SNELLIUS_SOLO_PERIOD_S = 111.635264
LUMI_SOLO_PERIOD_S = 287.906981
FEDERATED_PERIOD_S = 346.907500

# Corpus: 1 block = 1024 windows = 2,097,152 tokens (1024*2048). Unbalanced blocks/round:
#   snellius 100*32/1024 = 3.125 ; lumi 100*64/1024 = 6.25 ; together 9.375
BLOCKS_TODAY = 822  # GET /status on the live coordinator: 822 of 2692 unassigned

PLENTY = DarlState(num_blocks=2692, committed=0, leased=0, unassigned=1_000_000)
REAL = DarlState(num_blocks=2692, committed=1870, leased=0, unassigned=BLOCKS_TODAY)
EMPTY = DarlState(num_blocks=2692, committed=2692, leased=0, unassigned=0)


def shape(site: str, partition: str, gpus: int, hours: float, account: str | None = None) -> Shape:
    """A probed shape, built from a real sbatch argument string and parsed the way the
    scanner's rows are, so the tests exercise the same key recovery the planner does."""
    h, m = int(hours), int(round((hours - int(hours)) * 60))
    args = (f"-A {account} " if account else "") + \
        f"-p {partition} -N 1 --gpus-per-node {gpus} -t {h}:{m:02d}:00"
    return Shape(f"{site}_{gpus}g_{hours:g}h", parse_shape_args(site, args), args)


def option(site, gpus, hours, wait_h, tput, batch, startup_s, *, partition="p",
           begin_h=0.0, lanes=1, links=1, chain="none", account=None) -> Option:
    sh = shape(site, partition, gpus, hours, account)
    wait = WaitEstimate(p50_raw_s=wait_h * HOUR, p90_raw_s=wait_h * HOUR,
                        p50_eff_s=wait_h * HOUR, p90_eff_s=wait_h * HOUR,
                        samples=3, probe_age_s=60.0)
    overhead_s, quality = site_overhead_s(site, tput, CAL)
    candidate = Candidate(site, sh, wait, Geometry(site, gpus, tput, batch),
                          startup_s, overhead_s, quality)
    return Option(site, candidate, lanes, links, chain, begin_h * HOUR)


def snellius(hours=24.0, wait_h=0.0, startup_s=300.0, **kw) -> Option:
    return option("snellius", 4, hours, wait_h, 89.8, 32, startup_s,
                  partition="gpu_h100", **kw)


def lumi(hours=24.0, wait_h=0.0, startup_s=600.0, **kw) -> Option:
    return option("lumi", 8, hours, wait_h, 38.2, 64, startup_s,
                  partition="standard-g", account="project_462000226", **kw)


def config(**kw) -> PlanConfig:
    base = dict(horizon_s=60 * HOUR, num_rounds=1_000_000, balance="off", lanes_max=1)
    base.update(kw)
    return PlanConfig(**base)


def run(selection, *, cfg=None, darl=PLENTY, balance=False, **kw):
    cfg = cfg or config()
    return simulate(selection, config=cfg, calibration=CAL, darl=darl, balance=balance, **kw)


def ledger(timeline, site):
    return next(l for l in timeline.ledgers if l.site == site)


def close(got: float, want: float, tol: float = 1e-6) -> bool:
    return abs(got - want) <= tol


# --------------------------------------------------------------------------
# co-residency: sites must overlap IN TIME to federate at all
# --------------------------------------------------------------------------


@check("a site whose window does not overlap its partner's contributes zero federated rounds")
def _():
    """THE STAGGERED-START TRAP, in its purest form: LUMI is given a walltime equal to
    Snellius' quoted queue wait, so it trains solo for 20 h and dies exactly as its
    partner appears. This is what a naive 'start ASAP, ask for the wait you were
    quoted' rule produces, and the planner must recognise it rather than rank it."""
    # snellius: wait 20 h, startup 300 s -> productive at 20 h + 300 s = 72,300 s
    # lumi:     begins now, startup 600 s -> live [600 s, 72,000 s], i.e. dead first
    plan = (snellius(hours=24.0, wait_h=20.0), lumi(hours=20.0, wait_h=0.0))
    timeline = run(plan)

    assert timeline.federated_merges == 0, timeline.federated_merges
    assert timeline.first_federated_s is None
    for site in ("snellius", "lumi"):
        assert ledger(timeline, site).coresident_s == 0.0
        assert ledger(timeline, site).federated_merges == 0
    # Both sites did real solo work -- this is not a plan that does nothing, which is
    # exactly why it needs flagging rather than a low score.
    assert timeline.solo_merges > 900, timeline.solo_merges

    traps = detect_traps(plan, timeline, config())
    assert [t.code for t in traps] == ["trap_no_overlap"], traps
    assert "20.0 h" in traps[0].reason and "20.1 h" in traps[0].reason, traps[0].reason
    # The fix has to be named, and it is the opposite of the fix for the other trap.
    assert "longer walltime" in traps[0].reason and "--begin" in traps[0].reason


@check("a headstart that exhausts the corpus is reported as zero federated merges too")
def _():
    """The same outcome by a completely different route, and only a simulator that
    debits DARL blocks round by round can see it. With max_epochs = 1 there is no
    wraparound: `acquire` returns epoch_complete forever and the run ENDS, however much
    walltime is left. So the fix is the opposite one -- shorten the headstart.

    LUMI alone burns 6.25 blocks/round at 287.907 s: 822 / 6.25 = 131.5 -> 131 rounds
    -> 131 * 287.907 s = 37,715.8 s, plus the 600 s startup = 10.64 h. Snellius is not
    productive until 20 h + 300 s.
    """
    plan = (snellius(hours=24.0, wait_h=20.0), lumi(hours=40.0, wait_h=0.0))
    timeline = run(plan, darl=REAL)

    assert timeline.federated_merges == 0
    assert timeline.solo_merges == 131, timeline.solo_merges
    assert close(timeline.blocks_used, 131 * 6.25), timeline.blocks_used  # 818.75
    assert close(timeline.darl_exhausted_s, 600 + 131 * LUMI_SOLO_PERIOD_S, 1e-3)
    assert close(timeline.darl_exhausted_s / HOUR, 10.643, 1e-3)
    # The run stops there: the jobs exit, so the ledger must not bill 40 h of LUMI.
    assert close(timeline.run_end_s, timeline.darl_exhausted_s, 1e-6)
    assert close(ledger(timeline, "lumi").live_s / HOUR, 10.4766, 1e-3)
    assert ledger(timeline, "snellius").live_s == 0.0, "snellius never even started"

    traps = detect_traps(plan, timeline, config())
    assert [t.code for t in traps] == ["trap_corpus_exhausted"], traps
    assert "Shorten the headstart" in traps[0].reason, traps[0].reason
    # Same walltime, a corpus that is not the binding constraint: the trap disappears.
    assert run(plan, darl=PLENTY).federated_merges > 200


@check("extending the early site's walltime buys overlap, and the hours add up exactly")
def _():
    """The headstart plan the trap was hiding. LUMI at 40 h instead of 20 h:
        lumi     live [600 s, 144,000 s]        = 40 h - 600 s of startup
        snellius live [72,300 s, 158,400 s]     = 24 h - 300 s of startup
        overlap  [72,300 s, 144,000 s]          = 71,700 s = 19.9167 h
    """
    plan = (snellius(hours=24.0, wait_h=20.0), lumi(hours=40.0, wait_h=0.0))
    timeline = run(plan)
    assert timeline.federated_merges > 0
    assert close(timeline.first_federated_s / HOUR, 20.272, 1e-3)

    lumi_led, snel_led = ledger(timeline, "lumi"), ledger(timeline, "snellius")
    # LUMI: 19.9167 h alone, then 19.9167 h co-resident, then nothing -- it dies first.
    assert close(lumi_led.headstart_s, 72300 - 600), lumi_led.headstart_s
    assert close(lumi_led.coresident_s, 144000 - 72300), lumi_led.coresident_s
    assert lumi_led.tail_s == 0.0
    # Snellius: no headstart at all, then 19.9167 h together, then a 4 h solo tail.
    assert snel_led.headstart_s == 0.0
    assert close(snel_led.coresident_s, 71700.0)
    assert close(snel_led.tail_s, 158400 - 144000)   # 14,400 s = 4 h

    # THE INVARIANT: three hour columns, and they sum to the walltime that was asked
    # for minus the startup that was paid. One "walltime" number hides exactly what
    # this planner exists to expose.
    assert close(lumi_led.headstart_s + lumi_led.coresident_s + lumi_led.tail_s,
                 40 * HOUR - 600.0)
    assert close(snel_led.headstart_s + snel_led.coresident_s + snel_led.tail_s,
                 24 * HOUR - 300.0)
    for led in (lumi_led, snel_led):
        assert close(led.live_s, led.headstart_s + led.coresident_s + led.tail_s)
        assert led.gap_s == 0.0, "a single link has no gap"


@check("partial overlap is prorated, not rounded to all-or-nothing")
def _():
    """LUMI live [0, 8 h], Snellius live [4 h, 12 h]: exactly 4 h of co-residency for
    each, and 4 h alone on either side of it."""
    plan = (lumi(hours=8.0, wait_h=0.0, startup_s=0.0),
            snellius(hours=8.0, wait_h=4.0, startup_s=0.0))
    timeline = run(plan, cfg=config(horizon_s=16 * HOUR, balance="on"), balance=True)

    lumi_led, snel_led = ledger(timeline, "lumi"), ledger(timeline, "snellius")
    assert close(lumi_led.headstart_s, 4 * HOUR) and close(lumi_led.coresident_s, 4 * HOUR)
    assert lumi_led.tail_s == 0.0
    assert snel_led.headstart_s == 0.0 and close(snel_led.coresident_s, 4 * HOUR)
    assert close(snel_led.tail_s, 4 * HOUR)
    assert close(lumi_led.live_s, 8 * HOUR) and close(snel_led.live_s, 8 * HOUR)

    # Both sites contributed to every federated round, so the counts must agree.
    assert lumi_led.federated_merges == snel_led.federated_merges == timeline.federated_merges
    assert timeline.federated_merges == 38, timeline.federated_merges
    # Billed GPU-hours run over the whole live window, idle included: the hours a site
    # spends waiting at the barrier are the point, and hiding them in the denominator
    # would defeat it. 8 h x 8 GCDs = 64 GPU-h ; 8 h x 4 H100 = 32 GPU-h.
    assert close(lumi_led.gpu_s / HOUR, 64.0) and close(snel_led.gpu_s / HOUR, 32.0)
    assert 0.4 < snel_led.idle_fraction < 0.7, snel_led.idle_fraction


@check("accumulation is fixed for the life of a job, not re-derived per membership")
def _():
    """PWW_GRAD_ACCUM is resolved ONCE, before torchrun, into
    --training.global_batch_size (run_train.sh:279-329). It cannot drop back to 1 when
    the partner leaves the round, so a Snellius lane in a two-site plan runs at
    accumulation 5 for its whole walltime -- including every solo round.

    Simulating the LIVE membership's accumulation instead prices a solo headstart at
    1/5 of its corpus burn and 1/2.3 of its period, which is the difference between a
    plan and trap_corpus_exhausted. That is the bug the next check reproduces."""
    plan = (lumi(hours=8.0, wait_h=0.0, startup_s=0.0),
            snellius(hours=8.0, wait_h=4.0, startup_s=0.0))
    timeline = run(plan, cfg=config(horizon_s=16 * HOUR, balance="on"), balance=True,
                   record_rounds=True)
    # ONE accumulation per site, for the whole run.
    assert ledger(timeline, "snellius").accums == (5,), ledger(timeline, "snellius").accums
    assert ledger(timeline, "lumi").accums == (1,)

    members = [iv.members for iv in timeline.intervals]
    assert members == [("lumi-l0",), ("lumi-l0", "snellius-l0"), ("snellius-l0",)], members
    # 100*5*0.356347 = 178.174 s of phase against LUMI's 167.539 s: the rounded 5x
    # overshoots the barrier slightly, so the federated period is 357.5 s, not 346.9 s.
    assert close(timeline.intervals[1].period_s, 357.541953, 1e-3)
    # The TAIL is where it shows. Snellius alone still accumulates 5x, so its period is
    # 100*5*0.356347 + 17 + 59.0005 = 254.174 s, not the 111.635 s of an accum-1 solo
    # round -- and it spends 5*3.125 = 15.625 blocks a round, not 3.125.
    assert close(timeline.intervals[2].period_s, 254.173575, 1e-3)
    assert timeline.rounds[0].accums == (1,)          # lumi alone: its own accum is 1
    assert timeline.rounds[-1].accums == (5,), timeline.rounds[-1].accums
    assert close(timeline.rounds[-1].blocks, 15.625, 1e-9), timeline.rounds[-1].blocks
    federated = next(r for r in timeline.rounds if r.federated)
    assert federated.accums == (1, 5), federated.accums


@check("a balanced headstart that eats the corpus is a trap, not 195 free solo rounds")
def _():
    """REGRESSION, and the reason the check above changed. Snellius arrives first with
    a 5x accumulation pinned by the two-site plan it is part of; LUMI is six hours
    behind it. Priced at the live membership's accumulation the headstart looks like
    195 cheap solo rounds followed by 9 federated ones; priced at the accumulation the
    job is actually launched with, it exhausts all 822 blocks at 3.7 h -- before LUMI
    is productive at 6.1 h -- and federates zero times.

    Both readings end the run; only one of them is flagged, so the difference is
    whether the planner recommends the trap."""
    plan = (snellius(hours=40.0, wait_h=0.0, startup_s=108.0),
            lumi(hours=40.0, wait_h=6.0, startup_s=216.0))
    cfg = config(horizon_s=40 * HOUR, balance="on")
    timeline = run(plan, cfg=cfg, darl=REAL, balance=True)

    assert timeline.federated_merges == 0, timeline.federated_merges
    assert ledger(timeline, "snellius").accums == (5,)
    # 100 steps * 5 accum * 32 seq / 1024 = 15.625 blocks/round -> 52 rounds of 822.
    assert timeline.solo_merges == 52, timeline.solo_merges
    assert timeline.darl_exhausted_s is not None
    assert close(timeline.darl_exhausted_s / HOUR, 3.7014, 1e-3), timeline.darl_exhausted_s
    traps = detect_traps(plan, timeline, cfg)
    assert [t.code for t in traps] == ["trap_corpus_exhausted"], traps


@check("the emitted PWW_GRAD_ACCUM is the accumulation the plan was priced at")
def _():
    """The emitter and the simulator have to agree by CONSTRUCTION, not by coincidence:
    they diverged once, and the plan then described a run nobody could submit."""
    from pww.plan.rounds import make_member, plan_accums

    plan = (snellius(hours=8.0, wait_h=0.0, startup_s=0.0),
            lumi(hours=8.0, wait_h=0.0, startup_s=0.0))
    timeline = run(plan, cfg=config(horizon_s=8 * HOUR, balance="on"), balance=True)
    members = [make_member(f"{o.site}-l0", o.candidate) for o in plan]
    emitted = plan_accums(members, balance=True, cap=8)
    for site in ("snellius", "lumi"):
        assert ledger(timeline, site).accums == (emitted[f"{site}-l0"],), site


@check("two lanes at one site are not a federation")
def _():
    """They are two Flower clients and the server does merge them, but they share the
    hardware, the queue, the WAN link and the failure. Counting that as federated would
    let the planner 'federate' by submitting the same site twice."""
    timeline = run((snellius(hours=8.0, lanes=2),), cfg=config(horizon_s=12 * HOUR))
    assert timeline.federated_merges == 0, timeline.federated_merges
    assert timeline.solo_merges > 0
    assert {l.lane_id for l in timeline.links} == {"snellius-l0", "snellius-l1"}
    # And it costs: a second lane adds its whole overhead to every round.
    one = run((snellius(hours=8.0, lanes=1),), cfg=config(horizon_s=12 * HOUR))
    assert timeline.solo_merges < one.solo_merges, (timeline.solo_merges, one.solo_merges)


# --------------------------------------------------------------------------
# joining and leaving: both are normal events, and they cost different things
# --------------------------------------------------------------------------


@check("a lane that connects mid-round evaluates that round and fits only the next one")
def _():
    """Measured at latejoin round 98: `configure_fit: sampled 1`, then
    `configure_evaluate: sampled 2`. The joiner forfeits one fit round and still
    consumes a Flower attempt."""
    plan = (snellius(hours=8.0, wait_h=0.0, startup_s=0.0),
            lumi(hours=8.0, wait_h=1.0, startup_s=0.0))
    timeline = run(plan, cfg=config(horizon_s=8 * HOUR), record_rounds=True)
    joining = next(r for r in timeline.rounds if r.eval_only)
    assert joining.eval_only == ("lumi-l0",) and joining.fit == ("snellius-l0",)
    assert not joining.federated, "an evaluate-only lane is not in the merge"
    assert timeline.rounds[joining.index + 1].fit == ("snellius-l0", "lumi-l0")
    assert timeline.rounds[joining.index + 1].federated


@check("a first-ever cold join stalls the incumbent, and a warm lane does not")
def _():
    """One measurement (latejoin round 98): the incumbent's phase went 38 s to 416 s
    while the newcomer registered. A rejoin from the lane's OWN checkpoint showed no
    stall at all (churn round 147) -- which is the whole return on the lane
    abstraction: a self-resubmitting chain pays this once, not per link."""
    plan = (snellius(hours=8.0, wait_h=0.0, startup_s=0.0),
            lumi(hours=8.0, wait_h=1.0, startup_s=0.0))
    cold = run(plan, cfg=config(horizon_s=8 * HOUR), record_rounds=True)
    warm = run(plan, cfg=config(horizon_s=8 * HOUR), record_rounds=True,
               warm={"snellius": True, "lumi": True})

    stalled = [r for r in cold.rounds if r.stall_s]
    assert len(stalled) == 1, [r.index for r in stalled]
    assert stalled[0].stall_s == 378.0
    # 111.635 (snellius solo) + 39.168 (LUMI's evaluate half) + 378 = 528.80 s
    assert close(stalled[0].period_s, 528.803, 1e-3), stalled[0].period_s
    assert not any(r.stall_s for r in warm.rounds), "a warm rejoin pays no stall"
    # It costs about one federated round, which is what makes it worth modelling.
    assert warm.federated_merges - cold.federated_merges == 1


@check("a member whose walltime ends before the round closes forfeits it")
def _():
    """A departure between rounds produces zero failures -- the site is simply absent
    from num_available() at the next configure_fit. So no round may be charged to a
    lane that could not have finished it."""
    plan = (snellius(hours=8.0, wait_h=0.0, startup_s=0.0),
            lumi(hours=2.0, wait_h=0.0, startup_s=0.0))
    timeline = run(plan, cfg=config(horizon_s=10 * HOUR), record_rounds=True)
    spans = {}
    for link in timeline.links:
        spans[link.lane_id] = (link.productive_s, link.end_s)
    for rnd in timeline.rounds:
        for lane in rnd.fit:
            start, end = spans[lane]
            assert start <= rnd.start_s + 1e-6, (lane, rnd.index)
            assert rnd.start_s + rnd.period_s <= end + 1e-6, (
                f"round {rnd.index} was charged to {lane} but ran "
                f"{rnd.start_s + rnd.period_s - end:.1f} s past its walltime")
    # LUMI is gone after 2 h; Snellius carries on alone rather than the run ending.
    assert timeline.federated_merges > 0 and timeline.solo_merges > 0
    assert timeline.intervals[-1].members == ("snellius-l0",)


# --------------------------------------------------------------------------
# lanes and links: chaining buys queue access at the price of per-job startup
# --------------------------------------------------------------------------


@check("a self-resubmitting chain is seamless and only the first link is cold")
def _():
    """Link k+1 is submitted from inside link k, before training starts, with
    --begin=now+(T-lead) -- the pattern already in production in this repo
    (slurm_probe_loop.sh). It is NOT --dependency: a dependency-held job is not
    eligible for backfill, which forfeits exactly the advantage that motivated short
    jobs in the first place."""
    cfg = config(horizon_s=24 * HOUR)
    links = expand_links(snellius(hours=2.0, wait_h=1.0, links=3, chain="self"), config=cfg)
    # arrival_0 = 0 + 1 h = 3600; then +max(T, w) = +7200 each, because the successor
    # queues WHILE the predecessor runs.
    assert [l.arrival_s for l in links] == [3600.0, 10800.0, 18000.0]
    assert [l.end_s for l in links] == [10800.0, 18000.0, 25200.0]
    assert [l.cold for l in links] == [True, False, False]
    # ...and it is submitted before its predecessor ends, which is the whole point.
    assert [l.submit_s for l in links] == [0.0, 3600.0, 10800.0]

    # --dependency=singleton pays the queue wait again for every link: 3 links reach
    # 32,400 s instead of 25,200 s, for the same 6 h of walltime.
    held = expand_links(snellius(hours=2.0, wait_h=1.0, links=3, chain="singleton"), config=cfg)
    assert [l.arrival_s for l in held] == [3600.0, 14400.0, 25200.0]
    assert held[-1].end_s - links[-1].end_s == 7200.0

    # If Slurm defers eligibility to the begin time rather than running the wait
    # concurrently, the optimistic reading is wrong and the plan reprices to the same
    # thing as singleton. The knob exists so that assumption is testable, not buried.
    pessimistic = expand_links(
        snellius(hours=2.0, wait_h=1.0, links=3, chain="self"),
        config=config(horizon_s=24 * HOUR, chain_wait_overlap=False))
    assert [l.arrival_s for l in pessimistic] == [3600.0, 14400.0, 25200.0]


@check("a chain pays its startup cost once per link, and the gap is reported")
def _():
    """The cost side of the short-job policy: (T - c)/T of each link is useful."""
    timeline = run((snellius(hours=2.0, wait_h=1.0, startup_s=300.0, links=3, chain="self"),),
                   cfg=config(horizon_s=24 * HOUR))
    led = ledger(timeline, "snellius")
    # three links of 2 h, each losing 300 s to startup: live 3 x 6900 = 20,700 s
    assert close(led.live_s, 3 * (7200 - 300))
    # the gaps between links are the startups of links 1 and 2: 2 x 300 s
    assert close(led.gap_s, 600.0), led.gap_s
    assert close(led.startup_s, 900.0)     # 3 links x 300 s
    # Link 0 waits 1 h in the queue; links 1 and 2 are submitted a whole walltime ahead
    # and sit PENDING until their --begin, which is the point -- a pending job accrues
    # eligible time and stays backfill-eligible, unlike a dependency-held one.
    assert close(led.queued_s, 3600.0 + 2 * 7200.0), led.queued_s


@check("two links of one lane overlapping is refused, and the warning names the fix")
def _():
    """Two live jobs under one cluster id are refused outright: LeaseTable.register
    raises ClusterBusy -> HTTP 503, and the client gives up after ~13-40 s of retries
    and dies at startup. Concurrency must come from LANES."""
    timeline = run((snellius(hours=2.0, wait_h=1.0, links=3, chain="self"),),
                   cfg=config(horizon_s=24 * HOUR, chain_lead_s=1800.0))
    assert len(timeline.warnings) == 2, timeline.warnings
    assert "cluster_busy" in timeline.warnings[0]
    assert "own lane" in timeline.warnings[0] and "SIGTERM" in timeline.warnings[0]


@check("a job whose startup exceeds its walltime contributes nothing, and says so")
def _():
    timeline = run((snellius(hours=1.0, wait_h=0.0, startup_s=7200.0),),
                   cfg=config(horizon_s=24 * HOUR))
    assert timeline.solo_merges == 0 and timeline.federated_merges == 0
    assert any("exceeds the 1 h walltime" in w for w in timeline.warnings), timeline.warnings


# --------------------------------------------------------------------------
# the two budgets that bind before walltime does
# --------------------------------------------------------------------------


@check("DARL blocks are debited round by round and the exhaustion hour is reported")
def _():
    """822 blocks at 9.375 per federated round = 87.7 rounds. There is no wraparound:
    max_epochs = 1, so the run ends when the corpus does, whatever walltime is left."""
    plan = (snellius(hours=24.0, wait_h=0.0, startup_s=0.0),
            lumi(hours=24.0, wait_h=0.0, startup_s=0.0))
    timeline = run(plan, darl=REAL)
    assert timeline.federated_merges == 87, timeline.federated_merges
    assert close(timeline.blocks_used, 87 * 9.375)      # 815.625 of 822
    assert timeline.blocks_available == 822.0
    assert timeline.darl_exhausted_s is not None
    # 87 rounds x 346.9075 s = 30,180.9 s = 8.38 h, well inside the 24 h walltime.
    assert close(timeline.darl_exhausted_s, 87 * FEDERATED_PERIOD_S, 1e-3)
    assert close(timeline.darl_exhausted_s / HOUR, 8.383, 1e-3)
    assert timeline.intervals[-1].stop_cause == "darl_exhausted"
    assert close(timeline.intervals[-1].blocks_left, 822 - 87 * 9.375, 1e-6)
    # The ledger stops at the exhaustion, not at the walltime: billing 24 h of both
    # sites would flatter every long-walltime plan.
    assert close(timeline.run_end_s, timeline.darl_exhausted_s, 1e-6)
    assert close(ledger(timeline, "lumi").live_s, timeline.darl_exhausted_s, 1e-6)


@check("a plan over a corpus that is already spent runs nothing at all")
def _():
    plan = (snellius(hours=24.0), lumi(hours=24.0))
    timeline = run(plan, darl=EMPTY)
    assert timeline.federated_merges == timeline.solo_merges == 0
    assert timeline.blocks_available == 0.0 and timeline.blocks_used == 0.0
    assert timeline.darl_exhausted_s == 0.0, timeline.darl_exhausted_s
    assert timeline.tokens == 0 and timeline.attempts_used == 0
    # reserve_blocks is the same thing said in advance: hold 20 back and a 10-block
    # corpus is already spent.
    thin = DarlState(num_blocks=2692, committed=2682, leased=0, unassigned=10)
    held = run(plan, darl=thin, cfg=config(reserve_blocks=20))
    assert held.blocks_available == 0.0 and held.federated_merges == 0


@check("queued time burns no Flower attempt, but every started round does")
def _():
    """sample() blocks in wait_for on min_available_clients = 1, so a round is not
    ATTEMPTED while nobody is connected. It is consumed by every round that starts,
    solo rounds included."""
    timeline = run((snellius(hours=8.0, wait_h=5.0, startup_s=0.0),),
                   cfg=config(horizon_s=16 * HOUR, num_rounds=10))
    assert timeline.attempts_used == 10 == timeline.solo_merges
    assert timeline.attempts_exhausted_s is not None
    # All ten landed after the queue opened at 5 h; none were spent waiting for it.
    assert len(timeline.intervals) == 1
    assert close(timeline.intervals[0].t0_s, 5 * HOUR)
    assert timeline.intervals[0].stop_cause == "attempts_exhausted"
    assert close(timeline.intervals[0].t1_s, 5 * HOUR + 10 * SNELLIUS_SOLO_PERIOD_S, 1e-3)
    # With the budget raised the same plan runs to walltime instead.
    generous = run((snellius(hours=8.0, wait_h=5.0, startup_s=0.0),),
                   cfg=config(horizon_s=16 * HOUR, num_rounds=1000))
    # 8 h / 111.635264 s = 257.98 rounds, and the 258th would run past the walltime,
    # so the departing member forfeits it: 257.
    assert generous.solo_merges == 257, generous.solo_merges


@check("a gap between two links shows as a hole in the timeline and in gap_s")
def _():
    """A dependency-held successor cannot start until its predecessor is gone, so it
    pays the queue wait again and the site is simply absent in between. Nothing is
    charged to those hours -- no rounds, no attempts, no GPU-seconds."""
    timeline = run((snellius(hours=2.0, wait_h=0.0, startup_s=0.0, links=2,
                             chain="singleton"),),
                   cfg=config(horizon_s=24 * HOUR))
    # w = 0, so link 1 arrives the instant link 0 ends: no hole at all.
    assert [l.arrival_s for l in timeline.links] == [0.0, 7200.0]
    assert ledger(timeline, "snellius").gap_s == 0.0

    # A 1 h queue makes a 1 h hole: link 0 [3600, 10800], link 1 [14400, 21600].
    gapped = run((snellius(hours=2.0, wait_h=1.0, startup_s=0.0, links=2,
                           chain="singleton"),),
                 cfg=config(horizon_s=24 * HOUR))
    assert [l.arrival_s for l in gapped.links] == [3600.0, 14400.0]
    assert close(ledger(gapped, "snellius").gap_s, 3600.0)
    assert close(ledger(gapped, "snellius").live_s, 2 * 7200.0)
    # Two intervals with a hole between them, and nothing charged inside it.
    assert len(gapped.intervals) == 2, gapped.intervals
    assert gapped.intervals[0].t1_s < gapped.intervals[1].t0_s
    assert close(gapped.intervals[1].t0_s, 14400.0)
    assert gapped.solo_merges == sum(iv.rounds for iv in gapped.intervals)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


@check("the same inputs give a byte-identical timeline")
def _():
    """No RNG, no clock, no filesystem. A plan that changes between two runs over the
    same inputs cannot be argued with."""
    plan = (snellius(hours=24.0, wait_h=2.0), lumi(hours=40.0, wait_h=0.0))
    first = run(plan, darl=REAL, cfg=config(balance="on"), balance=True, record_rounds=True)
    second = run(plan, darl=REAL, cfg=config(balance="on"), balance=True, record_rounds=True)
    assert dataclasses.asdict(first) == dataclasses.asdict(second)
    # Option order must not change the answer either.
    flipped = run((plan[1], plan[0]), darl=REAL, cfg=config(balance="on"), balance=True)
    assert flipped.federated_merges == first.federated_merges
    assert close(flipped.blocks_used, first.blocks_used)


@check("recording rounds changes what is reported, never what is counted")
def _():
    """`record_rounds` is off in the search because building one dataclass per round of
    every discarded plan was the largest allocation in the profile. The counts are
    accumulated independently, so they must agree exactly."""
    plan = (snellius(hours=12.0, wait_h=1.0), lumi(hours=12.0, wait_h=0.0))
    quiet = run(plan, darl=REAL)
    loud = run(plan, darl=REAL, record_rounds=True)
    assert quiet.rounds == ()
    assert len(loud.rounds) == loud.federated_merges + loud.solo_merges
    assert (quiet.federated_merges, quiet.solo_merges, quiet.tokens) == \
           (loud.federated_merges, loud.solo_merges, loud.tokens)
    assert close(quiet.blocks_used, loud.blocks_used)
    # Every round's token and block count must reconcile with the totals.
    assert sum(r.tokens for r in loud.rounds) == loud.tokens
    assert close(sum(r.blocks for r in loud.rounds), loud.blocks_used, 1e-6)


@check("an empty plan simulates to an empty timeline rather than an exception")
def _():
    timeline = run(())
    assert timeline.links == () and timeline.intervals == () and timeline.ledgers == ()
    assert timeline.federated_merges == timeline.solo_merges == 0
    assert timeline.run_end_s == 0.0 and timeline.gpu_s == 0.0
    # An explicit schedule object is accepted too, and reset before use.
    assert run((), cfg=config(h_model="qsr"),
               schedule=make_schedule("qsr")).federated_merges == 0


# --------------------------------------------------------------------------
# the accounting a reader re-ranks by hand: hours, quality tags, GPU-hours
# --------------------------------------------------------------------------


@check("headstart, between, co-resident and tail sum to a site's real presence")
def _():
    """REGRESSION. The three-way split tells a CONTIGUOUS story -- solo, then together,
    then solo again -- and elastic membership does not oblige. Here LUMI is live
    [0,1 h] and again [6,7 h] while Snellius runs [0,8 h]: the five hours Snellius
    spends alone BETWEEN the two spells belong to none of the three columns, and
    dropping them made live_s 3 h instead of 8 and the barrier-idle fraction 0.287
    instead of 0.733 -- a headline metric wrong by 2.5x."""
    from pww.plan.timeline import _Window, _ledgers, _Accumulator
    from pww.plan.rounds import make_member

    sn = make_member("snellius-l0", snellius().candidate)
    lu = make_member("lumi-l0", lumi().candidate)
    windows = [_Window("snellius-l0", "snellius", 0.0, 8 * HOUR, True, sn),
               _Window("lumi-l0", "lumi", 0.0, 1 * HOUR, True, lu),
               _Window("lumi-l0", "lumi", 6 * HOUR, 7 * HOUR, False, lu)]

    class _Link:
        def __init__(self, site, a, b):
            self.site, self.lane_id = site, f"{site}-l0"
            self.arrival_s = self.submit_s = self.productive_s = a
            self.end_s = b

    links = [_Link("snellius", 0.0, 8 * HOUR),
             _Link("lumi", 0.0, 1 * HOUR), _Link("lumi", 6 * HOUR, 7 * HOUR)]
    acc = {"snellius": _Accumulator(), "lumi": _Accumulator()}
    acc["snellius"].compute_s = 8 * HOUR * (1 - 0.733)   # 26.7% of presence computing
    ledgers = _ledgers(links, windows, acc, config(), 8 * HOUR)

    sn_l = next(l for l in ledgers if l.site == "snellius")
    assert close(sn_l.headstart_s, 0.0) and close(sn_l.coresident_s, 2 * HOUR)
    assert close(sn_l.tail_s, 1 * HOUR) and close(sn_l.between_s, 5 * HOUR)
    assert close(sn_l.live_s, 8 * HOUR), sn_l.live_s
    assert close(sn_l.idle_fraction, 0.733, 1e-6), sn_l.idle_fraction
    # the weight the federation-wide idle fraction uses is the DEVICE COUNT, which is
    # only equal to gpu_s/live_s when the two measure the same interval -- they do not.
    assert sn_l.gpus == 4 and close(sn_l.gpu_s, 8 * HOUR * 4)


@check("an interval carries its OWN regime quality, not the worst seen so far")
def _():
    """REGRESSION, twice over. The tag used to be a module-level monotone accumulator,
    so a couple of lumi-solo rounds (an unmeasured cell) marked every LATER interval
    extrapolated -- hiding which regime is really unmeasured and making measured
    periods look untrustworthy -- and the plan-level quality was demoted with them, so
    `--overhead-model measured` refused to RANK a plan whose 79 federated rounds are
    all in a measured regime. The demotion silently changed the recommendation."""
    plan = (lumi(hours=8.0, wait_h=0.0, startup_s=200.0),
            snellius(hours=8.0, wait_h=0.0, startup_s=600.0))
    timeline = run(plan, cfg=config(horizon_s=8 * HOUR))
    tags = [(iv.members, iv.quality, iv.rounds) for iv in timeline.intervals]
    lumi_solo = [q for m, q, _ in tags if m == ("lumi-l0",)]
    both = [q for m, q, _ in tags if len(m) == 2]
    assert lumi_solo and all(q == "extrapolated" for q in lumi_solo), tags
    assert both and all(q == "derived_by_subtraction" for q in both), tags
    # DOMINANT, not worst-ever: the two unmeasured rounds do not demote the ~80 measured
    # ones. `worst_quality` would have said extrapolated.
    solo_rounds = sum(r for m, _, r in tags if m == ("lumi-l0",))
    fed_rounds = sum(r for m, _, r in tags if len(m) == 2)
    assert solo_rounds < fed_rounds, (solo_rounds, fed_rounds)
    assert timeline.quality == "derived_by_subtraction", timeline.quality


@check("the replay schedule walks the measured trace one value per round")
def _():
    """REGRESSION. next_h was drawn twice per round -- once inside the departure
    fixpoint and once again below it -- and ReplayH advances on every call, so the
    simulator replayed trace[1], trace[3], trace[5]... and then held the last value at
    half the trace length. The rise to H=113 was never simulated at all, while the
    `h_model replay` sensitivity row reported a winner-changing utility derived from
    a trajectory that never happened."""
    from pww.plan.rounds import DCLT_H_TRACE_HEAD

    sched = make_schedule("replay", inner_steps=100)
    timeline = run((snellius(hours=8.0, startup_s=60.0),),
                   cfg=config(horizon_s=8 * HOUR, h_model="replay"),
                   schedule=sched, record_rounds=True)
    got = tuple(r.inner_steps for r in timeline.rounds[:len(DCLT_H_TRACE_HEAD)])
    assert got == DCLT_H_TRACE_HEAD, got
    # ... and then it HOLDS the last value rather than reverting to h0
    assert timeline.rounds[len(DCLT_H_TRACE_HEAD)].inner_steps == DCLT_H_TRACE_HEAD[-1]


@check("GPU-hours count the allocation, which can outlast the corpus")
def _():
    """REGRESSION. gpu_s was clipped at run_end on the premise that the jobs exit when
    DARL is exhausted. They do not: FlowerClient sets self.done (flower_client.py:383,
    447) and nothing reads it, so the client keeps answering rounds with zero tokens
    until Slurm kills it at walltime. A scientist sizing an allocation off the clipped
    number under-budgets, and the plan is warned about it in so many words."""
    # 822 blocks at 3.125 blocks/round is 263 solo rounds of 111.635 s = 8.16 h, well
    # inside a 24 h job.
    timeline = run((snellius(hours=24.0, wait_h=0.0, startup_s=0.0),),
                   cfg=config(horizon_s=24 * HOUR), darl=REAL)
    assert timeline.darl_exhausted_s is not None
    assert timeline.run_end_s < 9 * HOUR, timeline.run_end_s / HOUR
    # billed: the whole 24 h allocation x 4 GPUs, not the 8.2 h that had corpus
    assert close(timeline.gpu_s, 24 * HOUR * 4, 1.0), timeline.gpu_s / HOUR
    assert any("hold their allocation" in w for w in timeline.warnings), timeline.warnings


def main() -> int:
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for name, exc in FAILED:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
