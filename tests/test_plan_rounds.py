"""What one federated round costs -- the arithmetic every other planner decision sits on.

    python3 tests/test_plan_rounds.py

No network, no cluster, no scanner, no torch. Everything here is pure arithmetic over
frozen dataclasses, so every expected number is written as a literal with its
derivation in a comment beside it.

THE ONE THING THIS FILE EXISTS TO PIN: a Flower round is a barrier, so the inner
phase is a MAX over the live sites and the non-training overhead is a SUM over them.

    period(S) = H * max_i(a_i * step_i)  +  o_merge  +  sum_i o_i(g_i)

Two mistakes that number is guarding against, both of which have already been made in
this campaign:

  * `round_wallclock = H * step_time`. The server logs ">> Round took Ns" and wandb
    records train/round_seconds, and BOTH are max(metrics["seconds"]) -- the slowest
    client's inner phase only. They exclude three 1.32 GiB WAN crossings per site per
    round, the fp32 merge, the 5.29 GiB checkpoint write and the whole evaluate
    barrier. Measured overhead is ~180 s against a 170 s two-site phase.
  * a membership-count-only overhead. Half of the overhead is validation COMPUTE, so
    it grows sharply at reduced geometry -- which is exactly the geometry a data cap
    recommends, so a site-blind model under-prices its own recommendation.

The calibration is checked against the four regimes differenced out of all_logs/.
Three are reproduced within 5%; the fourth is NOT reproducible by any additive form
and must come back tagged `extrapolated` rather than quietly priced.
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


def expect_raises(exc_type, fn, *, contains: str = ""):
    """Assert `fn` raises, and that the message names the fix.

    The `contains` half matters as much as the type: a refusal that does not say what
    to do about it is a silent drop with extra words.
    """
    try:
        fn()
    except exc_type as exc:
        if contains and contains not in str(exc):
            raise AssertionError(
                f"raised {exc_type.__name__} but message lacked {contains!r}: {exc}"
            ) from None
        return
    raise AssertionError(f"expected {exc_type.__name__}, nothing raised")


from pww.plan import (  # noqa: E402
    DEFAULT_CALIBRATION as CAL,
    Calibration,
    Candidate,
    Geometry,
    Member,
    MeasuredRegime,
    OverheadEntry,
    Shape,
    SiteOverhead,
    WaitEstimate,
    balance_accums,
    blocks_at_risk,
    make_member,
    make_schedule,
    residuals,
    round_cost,
    site_overhead_s,
)
from pww.plan.inputs import parse_shape_args  # noqa: E402
from pww.plan.model import DERIVED, EXTRAPOLATED, IDENTIFIED, worst_quality  # noqa: E402
from pww.plan.rounds import (  # noqa: E402
    DCLT_H_TRACE_HEAD,
    EVAL_FORWARD_SPEEDUP,
    FixedH,
    JensenController,
    LRSchedule,
    QsrSchedule,
    ReplayH,
)

# The measured registry, verbatim from configs/site_throughput.env. Both numbers are
# needed and neither is derivable from the other: the barrier equalises the time of one
# optimiser STEP, and the two sites do not run the same batch per step.
#   snellius   4 x H100  89.8 seq/s, 32 seq/step -> step 32/89.8   = 0.356347 s
#   lumi       8 x MI250X 38.2 seq/s, 64 seq/step -> step 64/38.2   = 1.675393 s
SNELLIUS_TPUT, SNELLIUS_BATCH = 89.8, 32
LUMI_TPUT, LUMI_BATCH = 38.2, 64
SNELLIUS_STEP_S = 32 / 89.8
LUMI_STEP_S = 64 / 38.2

# o_i(g) = xfer_i + eval_fix_i + V / (speedup * tput_i(g)), V = 512 validation windows.
#   snellius  33.5 + 23.6 + 512/(3*89.8) = 57.1 + 1.900519 =  59.000520 s
#   lumi      64.2 + 34.7 + 512/(3*38.2) = 98.9 + 4.467714 = 103.367714 s
O_SNELLIUS_S = 33.5 + 23.6 + 512 / (3 * 89.8)
O_LUMI_S = 64.2 + 34.7 + 512 / (3 * 38.2)
O_MERGE_S = 17.0


def member(site: str, gpus: int, tput: float, batch: int, lane: str | None = None) -> Member:
    overhead_s, quality = site_overhead_s(site, tput, CAL)
    return Member(
        lane_id=lane or f"{site}-l0",
        site=site,
        gpus=gpus,
        step_s=batch / tput,
        batch_seq=batch,
        overhead_s=overhead_s,
        quality=quality,
    )


def snellius(lane: str = "snellius-l0") -> Member:
    return member("snellius", 4, SNELLIUS_TPUT, SNELLIUS_BATCH, lane)


def lumi(lane: str = "lumi-l0") -> Member:
    return member("lumi", 8, LUMI_TPUT, LUMI_BATCH, lane)


def close(got: float, want: float, tol: float = 1e-6) -> bool:
    return abs(got - want) <= tol


# --------------------------------------------------------------------------
# the calibration itself
# --------------------------------------------------------------------------


@check("the round model reproduces the three identified regimes within 5%")
def _():
    """The whole basis for trusting a plan. Fitted on the two Snellius@4 rows; the
    other two are genuinely out of sample."""
    got = {r["label"]: r for r in residuals(CAL)}
    for label, measured in (
        ("snellius@4 solo", 113.0),          # 111-117 s over 198 solo rounds
        ("snellius@4 + lumi@8", 353.0),      # 348-358 s over 397 two-site rounds
        ("snellius@1 solo, accum 5", 248.0),  # balanced-LIVE, out of sample
    ):
        row = got[label]
        assert row["measured_s"] == measured, row
        assert row["quality"] == IDENTIFIED, row
        assert row["within_tolerance"], (
            f"{label}: predicted {row['predicted_s']:.1f} s against a measured "
            f"{measured} s, {row['rel_error'] * 100:+.2f}% -- outside the 5% the plan "
            f"is only trustworthy inside"
        )
        assert abs(row["rel_error"]) < 0.02, row  # all three land inside 2%


@check("the two-site one-device regime is tagged extrapolated, not silently priced")
def _():
    """No additive form reproduces it: LUMI@1's predicted evaluate compute alone
    approaches the whole measured evaluate segment, so either that barrier is partly a
    MAX rather than a SUM, or validation.local_batch_size differs at reduced geometry.
    n = 2 rounds; it is not resolvable from any existing log.

    The requirement is NOT that the model gets it right. It is that the model says so.
    """
    row = next(r for r in residuals(CAL) if r["label"] == "snellius@1 + lumi@1")
    assert row["quality"] == EXTRAPOLATED, row
    # 100*1.684211 + 17 + 64.094536 + 134.829825 = 384.35 s against a measured 363 s.
    assert close(row["predicted_s"], 384.345, 0.01), row
    assert not row["within_tolerance"], "a +5.9% miss must not read as reproduced"
    assert 0.05 < row["rel_error"] < 0.07, row
    # And it must name the measurement that closes it, or the tag is just a shrug.
    assert "merge-complete to merge-complete" in row["note"], row["note"]


@check("a cell outside the measured set is priced but carries the extrapolated tag")
def _():
    # snellius@4 alone is measured; snellius@4 + snellius@4 (two lanes) is not.
    solo = round_cost([snellius()], inner_steps=100, calibration=CAL, balance=False)
    assert solo.quality == IDENTIFIED, solo.quality
    two_lanes = round_cost([snellius("snellius-l0"), snellius("snellius-l1")],
                           inner_steps=100, calibration=CAL, balance=False)
    assert two_lanes.quality == EXTRAPOLATED, two_lanes.quality
    # LUMI's constants are a difference of two measurements -- no LUMI-solo round
    # exists in any log -- so anything involving LUMI is at best `derived`.
    federated = round_cost([snellius(), lumi()], inner_steps=100, calibration=CAL,
                           balance=False)
    assert federated.quality == DERIVED, federated.quality
    assert CAL.sites["lumi"].xfer.quality == DERIVED
    assert "no LUMI-solo round exists" in CAL.sites["lumi"].xfer.provenance
    assert worst_quality(IDENTIFIED, DERIVED, EXTRAPOLATED) == EXTRAPOLATED


@check("the evaluate half of the overhead scales with geometry and the transfer half does not")
def _():
    """This is why the overhead is per-site rather than per-member-count. Snellius'
    measured evaluate segment goes 25.5 s at 4 GPUs to 55 s at 1 GPU, and the 1-device
    shape is precisely what a DARL cap recommends."""
    four, _ = site_overhead_s("snellius", 89.8, CAL)   # 57.1 + 512/(3*89.8) =  59.0005
    one, _ = site_overhead_s("snellius", 24.4, CAL)    # 57.1 + 512/(3*24.4) =  64.0945
    assert close(four, 59.000520, 1e-5), four
    assert close(one, 64.094536, 1e-5), one
    # The fixed half is identical; the whole difference is validation compute.
    assert close(one - four, 512 / (3 * 24.4) - 512 / (3 * 89.8), 1e-9)
    assert EVAL_FORWARD_SPEEDUP == 3.0, "forward-only validation runs at ~3x training"


@check("a site with no overhead calibration is refused, never priced off another site's constants")
def _():
    expect_raises(
        KeyError,
        lambda: site_overhead_s("vega", 50.0, CAL),
        contains="no overhead calibration for site",
    )


# --------------------------------------------------------------------------
# the barrier: MAX over sites for the phase, SUM over sites for the overhead
# --------------------------------------------------------------------------


@check("the round's phase is the SLOWEST site's, not the mean of the sites'")
def _():
    """The single most dangerous averaging error available here. Snellius steps in
    0.356 s and LUMI in 1.675 s; a mean would say 1.016 s and a two-site round would
    look 40% cheaper than it is."""
    cost = round_cost([snellius(), lumi()], inner_steps=100, calibration=CAL,
                      balance=False, explain=True)
    # phase = 100 * max(0.356347, 1.675393) = 167.5393 s
    assert close(cost.phase_s, 100 * LUMI_STEP_S, 1e-9), cost.phase_s
    assert close(cost.phase_s, 167.539267, 1e-5), cost.phase_s
    mean_phase = 100 * (SNELLIUS_STEP_S + LUMI_STEP_S) / 2   # 101.59 s -- the wrong answer
    assert cost.phase_s > 1.6 * mean_phase, (cost.phase_s, mean_phase)
    # ...and the site order must not change it.
    flipped = round_cost([lumi(), snellius()], inner_steps=100, calibration=CAL, balance=False)
    assert close(flipped.phase_s, cost.phase_s, 1e-12)
    # The arithmetic is printed on demand so a scientist can recompute it by hand.
    assert "100*1.675 [lumi-l0]" in cost.arithmetic, cost.arithmetic
    assert cost.arithmetic.endswith("= 346.9 s"), cost.arithmetic


@check("the overhead is a SUM over live sites on top of one site-independent merge")
def _():
    solo = round_cost([snellius()], inner_steps=100, calibration=CAL, balance=False)
    both = round_cost([snellius(), lumi()], inner_steps=100, calibration=CAL, balance=False)
    # 17 + 59.000520                = 76.000520
    assert close(solo.overhead_s, O_MERGE_S + O_SNELLIUS_S, 1e-9), solo.overhead_s
    # 17 + 59.000520 + 103.367714   = 179.368234
    assert close(both.overhead_s, O_MERGE_S + O_SNELLIUS_S + O_LUMI_S, 1e-9), both.overhead_s
    # period = phase + overhead: 35.634744 + 76.000520 = 111.635264
    assert close(solo.period_s, 111.635264, 1e-5), solo.period_s
    # 167.539267 + 179.368234 = 346.907500
    assert close(both.period_s, 346.907500, 1e-5), both.period_s
    # The merge term is the one genuinely site-independent constant (15-19 s in every
    # log and every membership), so it must appear exactly once however many sites run.
    assert CAL.merge.value_s == O_MERGE_S and CAL.merge.quality == IDENTIFIED


@check("adding a site costs its own overhead plus H times the increase in the barrier")
def _():
    solo = round_cost([snellius()], inner_steps=100, calibration=CAL, balance=False)
    both = round_cost([snellius(), lumi()], inner_steps=100, calibration=CAL, balance=False)
    # 103.367714 (LUMI's own overhead) + 100*(1.675393 - 0.356347) = 103.367714 + 131.904523
    expected = O_LUMI_S + 100 * (LUMI_STEP_S - SNELLIUS_STEP_S)
    assert close(both.period_s - solo.period_s, expected, 1e-9), (both.period_s, solo.period_s)
    assert close(both.period_s - solo.period_s, 235.272236, 1e-5)

    # A second lane at the SAME site does not move the barrier at all, so it costs
    # exactly its own overhead -- which is the whole argument for pricing the overhead
    # per site rather than per member count.
    two_lanes = round_cost([snellius("snellius-l0"), snellius("snellius-l1")],
                           inner_steps=100, calibration=CAL, balance=False)
    assert close(two_lanes.phase_s, solo.phase_s, 1e-12)
    assert close(two_lanes.period_s - solo.period_s, O_SNELLIUS_S, 1e-9)


@check("a lane that joined mid-round pays the evaluate barrier and nothing else")
def _():
    """Flower's configure_evaluate samples a newcomer before configure_fit ever does,
    so its first act is an EVALUATE and it contributes nothing to that merge. Billing
    it for a fit transport it never did would overstate the cost of a late join."""
    base = round_cost([snellius()], inner_steps=100, calibration=CAL, balance=False)
    joining = round_cost([snellius()], inner_steps=100, calibration=CAL, balance=False,
                         eval_only=[lumi()])
    # LUMI's evaluate half only: 103.367714 - 64.2 (xfer) = 39.167714 s
    assert close(joining.period_s - base.period_s, O_LUMI_S - 64.2, 1e-9)
    assert close(joining.period_s - base.period_s, 39.167714, 1e-5)
    # It is not in the merge, so it moves neither the tokens nor the corpus burn.
    assert joining.tokens == base.tokens
    assert joining.blocks == base.blocks
    # And the stall a first-ever cold join charges to the incumbent is on the phase.
    stalled = round_cost([snellius()], inner_steps=100, calibration=CAL, balance=False,
                         eval_only=[lumi()], stall_s=CAL.tau_stall_s)
    assert close(stalled.period_s - joining.period_s, 378.0, 1e-9)
    assert CAL.tau_stall_s == 378.0  # measured once: 416 s against a normal 38 s


@check("a round with nobody fitting is refused rather than priced as free")
def _():
    expect_raises(ValueError, lambda: round_cost([], inner_steps=100, calibration=CAL,
                                                 balance=False),
                  contains="at least one fitting member")


# --------------------------------------------------------------------------
# balancing: accumulation is a function of the membership, not of the site
# --------------------------------------------------------------------------


@check("accumulation is the measured 5x on Snellius and 1x on LUMI")
def _():
    """Hand-computed from the registry, mirroring scripts/titan/run_train.sh:305-308:
        step_snellius = 32/89.8 = 0.356347 s ; step_lumi = 64/38.2 = 1.675393 s
        a_snellius = int(1.675393/0.356347 + 0.5) = int(4.701 + 0.5) = 5
        a_lumi     = int(1.675393/1.675393 + 0.5) = int(1.5)         = 1
    """
    assert balance_accums([snellius(), lumi()], balance=True) == (5, 1)
    assert balance_accums([lumi(), snellius()], balance=True) == (1, 5)
    assert balance_accums([snellius(), lumi()], balance=False) == (1, 1)


@check("accumulation is recomputed when membership changes, not fixed per site")
def _():
    """The same Snellius lane gets 5x beside LUMI and 1x alone. A planner that resolves
    accumulation once per site cannot express the barrier it is meant to fill."""
    s = snellius()
    assert balance_accums([s], balance=True) == (1,)              # nobody to wait for
    assert balance_accums([s, lumi()], balance=True) == (5, 1)    # LUMI sets the barrier
    assert balance_accums([s, s], balance=True) == (1, 1)         # two identical lanes
    # A third, slower member moves the incumbent's accumulation again: a member at
    # 3.0 s/step makes Snellius int(3.0/0.356347 + 0.5) = int(8.42 + 0.5) = 8 and LUMI
    # int(3.0/1.675393 + 0.5) = int(1.79 + 0.5) = 2.
    slow = Member("slow-l0", "lumi", 8, 3.0, 64, O_LUMI_S, DERIVED)
    assert balance_accums([s, lumi(), slow], balance=True) == (8, 2, 1)


@check("accumulation uses awk's int(x + 0.5) rounding and honours PWW_BALANCE_MAX")
def _():
    """Python's round() is banker's rounding and disagrees with the awk that actually
    runs on the cluster on exact halves: round(2.5) is 2, int(2.5 + 0.5) is 3."""
    fast = Member("a", "snellius", 4, 1.0, 32, O_SNELLIUS_S, IDENTIFIED)
    slow = Member("b", "lumi", 8, 2.5, 64, O_LUMI_S, DERIVED)
    assert balance_accums([fast, slow], balance=True) == (3, 1), "int(2.5+0.5) = 3, not 2"
    # The cap is the shell's PWW_BALANCE_MAX (default 8), applied after the rounding.
    very_slow = Member("c", "lumi", 8, 100.0, 64, O_LUMI_S, DERIVED)
    assert balance_accums([fast, very_slow], balance=True) == (8, 1)
    assert balance_accums([fast, very_slow], balance=True, cap=4) == (4, 1)
    # Floored at 1: a member faster than the barrier never accumulates less than once.
    assert balance_accums([slow, fast], balance=True, cap=8) == (1, 3)


@check("balancing raises tokens and corpus burn together, and nudges the period up")
def _():
    """Accumulation fills the barrier without touching H, drift, the LR schedule or
    peak memory. What it costs is DARL blocks, linearly -- which is why balancing has
    to be solved jointly with membership rather than switched on by policy.

    It is not free in wall-clock either: the rounded 5x slightly OVERSHOOTS the
    barrier (5 * 0.356347 = 1.781737 s against LUMI's 1.675393 s), so the phase grows.
    """
    off = round_cost([snellius(), lumi()], inner_steps=100, calibration=CAL, balance=False)
    on = round_cost([snellius(), lumi()], inner_steps=100, calibration=CAL, balance=True)
    assert on.accums == (5, 1)
    # 100 * 5 * 0.356347 = 178.173719 s, up from 167.539267
    assert close(on.phase_s, 100 * 5 * SNELLIUS_STEP_S, 1e-9), on.phase_s
    assert close(on.period_s, 357.541953, 1e-5), on.period_s
    # tokens: 100 * 2048 * (5*32 + 64) = 100 * 2048 * 224 = 45,875,200
    assert on.tokens == 45_875_200, on.tokens
    # blocks: 100 * 224 / 1024 = 21.875 against 9.375 -- 2.333x the corpus per round
    assert close(on.blocks, 21.875), on.blocks
    assert close(on.blocks / off.blocks, 7 / 3, 1e-9), (on.blocks, off.blocks)
    # 822 free blocks today: 87.7 federated merges unbalanced, 37.6 balanced.
    assert close(822 / off.blocks, 87.68, 0.01)
    assert close(822 / on.blocks, 37.58, 0.01)


@check("round_cost prices at the CAP it was handed, not at the uncapped ratio")
def _():
    """balance_accums honours `cap` and is checked for it directly, but round_cost's
    forwarding of PlanConfig.balance_max was not: hardcoding `cap=8` at rounds.py:278
    left every suite green, so the only knob that limits accumulation was untested
    everywhere it is actually reached from. It is the knob that decides how much DARL
    a balanced plan burns, and the emitted PWW_GRAD_ACCUM bypasses run_train.sh's own
    PWW_BALANCE_MAX (the shell caps only in the branch an explicit value disables), so
    a planner that ignores the cap is the last thing standing between the operator and
    an accumulation nobody asked for.

    At cap 3 Snellius is held to 3x instead of the 5x the ratio wants, and LUMI is
    once again the slowest member, so the barrier -- and the period with it -- falls
    back to the unbalanced one while the corpus burn stays raised.
    """
    capped = round_cost([snellius(), lumi()], inner_steps=100, calibration=CAL,
                        balance=True, balance_max=3)
    assert capped.accums == (3, 1), capped.accums
    # 3 * 0.356347 = 1.069 s against LUMI's 1.675393 s, so LUMI sets the barrier again
    assert close(capped.phase_s, 100 * LUMI_STEP_S, 1e-9), capped.phase_s
    assert close(capped.period_s, 346.907505, 1e-5), capped.period_s
    # tokens: 100 * 2048 * (3*32 + 64) = 100 * 2048 * 160 = 32,768,000
    assert capped.tokens == 32_768_000, capped.tokens
    # blocks: 100 * 160 / 1024 = 15.625, against 21.875 uncapped
    assert close(capped.blocks, 15.625), capped.blocks

    # A cap above the ratio changes nothing: it is a ceiling, not a setting.
    assert round_cost([snellius(), lumi()], inner_steps=100, calibration=CAL,
                      balance=True, balance_max=64).accums == (5, 1)
    # ...and 1 is balancing turned off by arithmetic rather than by flag.
    flat = round_cost([snellius(), lumi()], inner_steps=100, calibration=CAL,
                      balance=True, balance_max=1)
    assert flat.accums == (1, 1) and flat.tokens == 19_660_800, flat.accums


# --------------------------------------------------------------------------
# tokens and DARL blocks, against the numbers in the real logs
# --------------------------------------------------------------------------


@check("tokens per round match the figure the central log actually reported")
def _():
    cost = round_cost([snellius(), lumi()], inner_steps=100, calibration=CAL, balance=False)
    # snellius 100*4*8*1*2048 = 6,553,600 ; lumi 100*8*8*1*2048 = 13,107,200
    assert cost.tokens == 6_553_600 + 13_107_200 == 19_660_800, cost.tokens
    solo = round_cost([snellius()], inner_steps=100, calibration=CAL, balance=False)
    assert solo.tokens == 6_553_600, solo.tokens
    # H scales it exactly: the DCLT arm's H=250 Snellius-alone merge logged 16,384,000.
    at_250 = round_cost([snellius()], inner_steps=250, calibration=CAL, balance=False)
    assert at_250.tokens == 250 * 32 * 2048 == 16_384_000, at_250.tokens


@check("blocks per round match the 4 and 7 block leases in the real logs")
def _():
    """blocks_for_phase = ceil(H*batch*ranks*accum / 1024); the round RATE is the
    unrounded quantity, because the ceil only sets acquisition granularity and
    DARLDataSource._carry rides the remainder into the next phase."""
    both = round_cost([snellius(), lumi()], inner_steps=100, calibration=CAL, balance=False)
    # 100 * (32 + 64) / 1024 = 9.375 blocks per federated round
    assert close(both.blocks, 9.375), both.blocks
    assert close(round_cost([snellius()], inner_steps=100, calibration=CAL,
                            balance=False).blocks, 3.125)
    assert close(round_cost([lumi()], inner_steps=100, calibration=CAL,
                            balance=False).blocks, 6.25)
    # Data at risk when a job dies is exactly one phase's LEASE, which is the ceil.
    # The live coordinator recorded blocks_lost 4 (snellius) and 7 (lumi): one phase each.
    assert blocks_at_risk(snellius(), inner_steps=100, accum=1) == 4   # ceil(3200/1024)
    assert blocks_at_risk(lumi(), inner_steps=100, accum=1) == 7       # ceil(6400/1024)
    # Under balancing Snellius' exposure quadruples: ceil(100*32*5/1024) = ceil(15.625)
    assert blocks_at_risk(snellius(), inner_steps=100, accum=5) == 16
    # 4 blocks of a 2,692-block corpus is 0.15%: the cost of a short-job policy, and
    # the reason that policy is affordable at all.
    assert close(4 / 2692, 0.001486, 1e-6)


@check("make_member carries the price admission resolved, so the round loop stays hashable")
def _():
    shape = Shape("h100_full_8h", parse_shape_args(
        "snellius", "-p gpu_h100 -N 1 --gpus-per-node 4 -t 8:00:00"),
        "-p gpu_h100 -N 1 --gpus-per-node 4 -t 8:00:00")
    wait = WaitEstimate(p50_raw_s=0.0, p90_raw_s=0.0, p50_eff_s=0.0, p90_eff_s=0.0,
                        samples=1, probe_age_s=60.0)
    geometry = Geometry("snellius", 4, SNELLIUS_TPUT, SNELLIUS_BATCH)
    overhead_s, quality = site_overhead_s("snellius", SNELLIUS_TPUT, CAL)
    candidate = Candidate("snellius", shape, wait, geometry, 300.0, overhead_s, quality)
    m = make_member("snellius-l0", candidate)
    assert close(m.step_s, SNELLIUS_STEP_S, 1e-12) and m.batch_seq == 32
    assert close(m.overhead_s, O_SNELLIUS_S, 1e-9) and m.quality == IDENTIFIED
    # Members and candidates are frozen dataclasses of builtins, which is what lets the
    # simulator memoise on them; a Mapping anywhere in the key would break the search.
    assert hash(m) and hash(candidate) and hash(shape.key)


# --------------------------------------------------------------------------
# H: constant on the full arm, a controller on the DCLT arm, 5x apart
# --------------------------------------------------------------------------


@check("the full arm's H is constant, because it has no qsr-h0 key at all")
def _():
    """configs/central_aggregator_titan.yaml never sets --qsr-h0, so it defaults to 0
    and QSR is OFF: H is darl.inner_steps = 100 for the whole run. Applying the QSR
    rule to that arm is wrong by up to 5x per round."""
    fixed = make_schedule("fixed", inner_steps=100)
    assert isinstance(fixed, FixedH) and fixed.constant
    assert [fixed.next_h(global_step=s, n_fit=2) for s in (0, 5000, 19999)] == [100, 100, 100]
    # `constant` is what licenses the simulator's closed-form fast path.
    assert not make_schedule("qsr").constant
    assert not make_schedule("replay").constant


@check("the QSR ceiling is qsr_max/2 because the cap precedes the Jensen multiplier")
def _():
    """strategy.py:410 applies min(h, qsr_max) BEFORE multiplying, and the multiplier
    floors at 0.5 -- so the configured qsr-max of 500 is really 250. The measured
    maximum across 213 merges was 400, and the last ~20 merges sat at 250-331."""
    schedule = QsrSchedule(h0=100, qsr_max=500, warmup_steps=300)
    # At step 20000 the cosine decay is at min_lr_factor = 0.05, so
    # (lr_ref/lr_last)^2 = (1/0.05)^2 = 400 and h = 100*400 = 40000, capped to 500.
    assert close(schedule.lr.at(20000), 4.5e-4 * 0.05, 1e-12)
    assert schedule.next_h(global_step=20000, n_fit=2) == 500
    # Drive the multiplier to its floor: 5 warmup observations, then 0.7 per round.
    # 1.0 -> 0.7 -> 0.49, clamped to 0.5.
    for _ in range(7):
        schedule.observe(n_fit=2, jensen_gap=+0.1)
    assert close(schedule.jensen.multiplier, 0.5, 1e-12), schedule.jensen.multiplier
    assert schedule.next_h(global_step=20000, n_fit=2) == 250, "500 was never reachable"


@check("the Jensen multiplier does not move on a solo round")
def _():
    """The controller needs two clusters reporting a finite local_eval_loss
    (strategy.py:599), so a federation that is solo most of the time runs the DCLT arm
    with a stale multiplier: H does not adapt while a site is queued. That is a
    scheduling consequence of a scheduling decision, so the planner has to model it."""
    controller = JensenController()
    for _ in range(50):
        controller.observe(+0.1, n_clusters=1)
    assert controller.multiplier == 1.0 and controller.gauge_rounds == 0
    controller.observe(None, n_clusters=2)          # no gap reported: also a freeze
    assert controller.gauge_rounds == 0
    # Two clusters with a gap: observed for jensen_warmup_rounds, then acted on.
    for _ in range(5):
        controller.observe(+0.1, n_clusters=2)
    assert controller.multiplier == 1.0, "warmup must observe without acting"
    controller.observe(+0.1, n_clusters=2)
    assert close(controller.multiplier, 0.7, 1e-12), controller.multiplier


@check("the QSR floor is qsr_h0//2 and warmup holds H at h0")
def _():
    schedule = QsrSchedule(h0=100, qsr_max=500, warmup_steps=300)
    assert schedule.next_h(global_step=0, n_fit=2) == 100      # before qsr_warmup_steps
    assert schedule.next_h(global_step=299, n_fit=2) == 100
    for _ in range(7):
        schedule.observe(n_fit=2, jensen_gap=+0.1)             # multiplier -> 0.5
    # In the stable phase lr_ref/lr_last is 1, so h = 100*0.5 = 50 = the floor.
    assert schedule.next_h(global_step=5000, n_fit=2) == 50
    assert max(1, 100 // 2) == 50


@check("the replay schedule reproduces the measured DCLT trajectory and then holds")
def _():
    """The trajectory matters more than any single value: the controller spent ~100
    rounds BELOW h0, so QSR did not reduce the merge count in the one real run -- 213
    merges for a 20,000-step budget against 200 at a fixed H=100."""
    replay = make_schedule("replay")
    assert isinstance(replay, ReplayH)
    first = [replay.next_h(global_step=0, n_fit=2) for _ in range(5)]
    assert first == [100, 70, 50, 55, 60], first
    assert DCLT_H_TRACE_HEAD[:3] == (100, 70, 50)
    # 100 -> x0.7 = 70 -> x0.7 = 49, clamped up to the floor of 50, then the in-band
    # relaxation 0.55/0.595/... gives 55, 60, 64, 67.
    for _ in range(100):
        held = replay.next_h(global_step=0, n_fit=2)
    assert held == DCLT_H_TRACE_HEAD[-1], "a trace that runs out must hold, not reset to h0"
    replay.reset()
    assert replay.next_h(global_step=0, n_fit=2) == 100
    expect_raises(ValueError, lambda: ReplayH([]), contains="at least one H value")
    expect_raises(ValueError, lambda: make_schedule("magic"), contains="unknown h_model")


@check("the LR schedule matches torchtitan's, since QSR squares any error in it")
def _():
    """linear_warmup_stable_decay with the dclt toml's numbers: lr 4.5e-4, warmup 300,
    20000 steps, decay_ratio 0.5, cosine, min_lr_factor 0.05. A half-step error here is
    invisible in the LR and squared into H."""
    lr = LRSchedule()
    assert close(lr.at(0), 4.5e-4 * (1 / 300), 1e-12)      # 0-indexed +1 adjustment
    assert close(lr.at(299), 4.5e-4, 1e-12)                # warmup ends at the peak
    assert close(lr.at(5000), 4.5e-4, 1e-12)               # stable
    assert close(lr.at(20000), 4.5e-4 * 0.05, 1e-12)       # floor at the end
    # decay = round(20000*0.5) = 10000, stable = 20001 - 300 - 10000 = 9701, so the
    # cosine starts at step 10001 and is exactly halfway down at the midpoint.
    mid = 300 + 9701 + 5000 - 1
    assert close(lr.factor(mid), 0.05 + 0.95 * 0.5, 1e-9), lr.factor(mid)
    assert lr.at(-5) == lr.at(0) and lr.at(99999) == lr.at(20000)


@check("a calibration loaded from a literal behaves exactly like the built-in one")
def _():
    """Every stage has to be drivable from a literal with no I/O, which is what makes
    all of these checks arithmetic rather than integration."""
    tiny = Calibration(
        merge=OverheadEntry(10.0, IDENTIFIED, "made up"),
        sites={"a": SiteOverhead(OverheadEntry(1.0, IDENTIFIED), OverheadEntry(2.0, IDENTIFIED))},
        regimes=(MeasuredRegime("a solo", (("a", 1),), 10, (1,), 100.0),),
        val_windows=0,
    )
    m = Member("a-l0", "a", 1, 0.5, 8, sum(site_overhead_s("a", 1.0, tiny)[:1]), IDENTIFIED)
    cost = round_cost([m], inner_steps=10, calibration=tiny, balance=False)
    # phase 10*0.5 = 5, overhead 10 + (1 + 2 + 0/(3*1)) = 13, period 18
    assert close(cost.period_s, 18.0, 1e-9), cost.period_s
    assert cost.quality == IDENTIFIED
    # tokens use the calibration's own seq_len, not a global constant
    assert cost.tokens == 10 * 2048 * 8


def main() -> int:
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for name, exc in FAILED:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
