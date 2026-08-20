"""What one Flower round costs, and how many inner steps it runs.

THE ONE STRUCTURAL FACT: a round is a barrier. The inner phase is a MAX over the
live sites and the non-training overhead is a SUM over them. Everything the planner
concludes follows from those two words.

    period(S) = H * max_i(a_i * step_i)  +  o_merge  +  sum_i o_i(g_i)

The mistake this module exists to prevent is `round_wallclock = H * step_time`. The
number the server logs as ">> Round took Ns" and the wandb series train/round_seconds
are `max(metrics["seconds"])`, i.e. the slowest client's INNER PHASE only
(central/server.py:362,379). They exclude three 1.32 GiB WAN crossings per site per
round, the fp32 merge, the 5.29 GiB checkpoint write and the whole evaluate barrier.
Measured overhead is ~180 s on a two-site round against a 170 s phase: a planner
built on the logged number overestimates round throughput by ~2.1x.

WHY THE OVERHEAD IS PER-SITE AND NOT A CONSTANT PER MEMBER COUNT. Only half of it
scales with device count. `xfer` is a fixed pair of crossings of the whole model;
the evaluate barrier is a fixed cost plus V/tput of validation compute. At reduced
geometry the compute half grows sharply (Snellius' evaluate segment goes 25.5 s at
4 GPUs to 55 s at 1 GPU), and the 1-device shape is precisely what a data cap
recommends -- so a membership-count-only overhead model under-prices its own
recommendation by tens of seconds per round.

The calibration below is fitted to four regimes differenced out of the central logs.
Three are identified and the model reproduces them; the fourth (two sites at one
device each) is NOT reproduced by any additive form and is shipped tagged
`extrapolated` rather than quietly priced. `residuals()` recomputes all four at
import-time cost of nothing, and the report prints the tag beside every period
derived from an extrapolated cell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..darl.space import BlockSpace, blocks_for_phase
from .model import (
    DERIVED,
    EXTRAPOLATED,
    IDENTIFIED,
    Calibration,
    Candidate,
    Member,
    MeasuredRegime,
    OverheadEntry,
    SiteOverhead,
    worst_quality,
)

# Forward-only validation: no backward pass, no optimiser step, so the evaluate
# barrier's compute runs at roughly 3x training throughput. This is not a free
# parameter -- it is what makes the two solo regimes agree. Fitting the per-site
# constants on the 4-GPU Snellius round and then predicting the 1-GPU accum-5 round
# gives +2.9% with a speedup of 1 and -1.2% with 3, and the two-site 1-device regime
# goes from +26% to +6%. Neither regime was used to fit the constants, so both are
# genuine out-of-sample checks of this number.
EVAL_FORWARD_SPEEDUP = 3.0

# Measured once (latejoin round 98): a first-ever cold join stalled the incumbent's
# next inner phase from 38 s to 416 s. The mechanism is not in any log, so this is a
# one-sample constant and the report says so. A warm rejoin from the lane's own
# checkpoint showed no stall at all (churn round 147), which is the whole argument
# for the lane abstraction: a self-resubmitting chain pays this once, not per link.
TAU_STALL_S = 378.0

DEFAULT_CALIBRATION = Calibration(
    # 15-19 s in every log and every membership: the fp32 weighted mean, the Nesterov
    # outer step and the 5.29 GiB npz write. The one genuinely site-independent term.
    merge=OverheadEntry(17.0, IDENTIFIED, "merge-complete deltas, all four central logs, n=399"),
    sites={
        "snellius": SiteOverhead(
            xfer=OverheadEntry(
                33.5, IDENTIFIED,
                "solo fit-transport, latejoin+churn logs (33.5/34), n=199"),
            eval_fix=OverheadEntry(
                23.6, IDENTIFIED,
                "solo evaluate 25.5 s - 512 windows / (3 x 89.8 seq/s)"),
        ),
        "lumi": SiteOverhead(
            # No LUMI-solo round exists in any log -- LUMI never ran without Snellius
            # -- so both numbers are the two-site segment minus Snellius' identified
            # share. That is a subtraction of two measurements, not a measurement.
            xfer=OverheadEntry(
                64.2, DERIVED,
                "two-site fit-transport 97.7 s - snellius 33.5 s; no LUMI-solo round exists"),
            eval_fix=OverheadEntry(
                34.7, DERIVED,
                "two-site evaluate 64.7 s - snellius 25.5 s - 512/(3 x 38.2); same caveat"),
        ),
    },
    regimes=(
        MeasuredRegime(
            label="snellius@4 solo",
            members=(("snellius", 4),),
            inner_steps=100,
            accums=(1,),
            measured_period_s=113.0,
            quality=IDENTIFIED,
            note="111-117 s over 98+100 solo rounds, latejoin/churn/full-run6",
        ),
        MeasuredRegime(
            label="snellius@4 + lumi@8",
            members=(("snellius", 4), ("lumi", 8)),
            inner_steps=100,
            accums=(1, 1),
            measured_period_s=353.0,
            quality=IDENTIFIED,
            note="348-358 s over 197+99+101 two-site rounds",
        ),
        MeasuredRegime(
            label="snellius@1 solo, accum 5",
            members=(("snellius", 1),),
            inner_steps=100,
            accums=(5,),
            measured_period_s=248.0,
            quality=IDENTIFIED,
            note="balanced-LIVE; out-of-sample for the fitted constants",
        ),
        MeasuredRegime(
            label="snellius@1 + lumi@1",
            members=(("snellius", 1), ("lumi", 1)),
            inner_steps=100,
            accums=(1, 1),
            measured_period_s=363.0,
            quality=EXTRAPOLATED,
            note=(
                "n=2 rounds. No additive form reproduces this: LUMI@1's predicted "
                "evaluate compute alone approaches the whole measured evaluate segment, "
                "so either that barrier is partly a MAX or validation.local_batch_size "
                "differs at reduced geometry. ONE two-site 1-device round with "
                "PWW_VAL_WINDOWS logged, differenced merge-complete to merge-complete, "
                "closes it."
            ),
        ),
    ),
    val_windows=512,
    tau_stall_s=TAU_STALL_S,
    seq_len=2048,
    block_size=1024,
)


@dataclass(frozen=True)
class RoundCost:
    period_s: float
    phase_s: float
    overhead_s: float
    accums: tuple[int, ...]
    tokens: int
    blocks: float
    quality: str
    arithmetic: str
    stall_s: float = 0.0


def site_overhead_s(site: str, tput_seq_s: float, calibration: Calibration) -> tuple[float, str]:
    """o_i(g) = xfer_i + eval_fix_i + V / (speedup * tput_i(g)).

    Raises rather than defaulting: a site with no calibration entry must become an
    Exclusion upstream, not a plan priced off another site's constants.
    """
    entry = calibration.sites.get(site)
    if entry is None:
        raise KeyError(
            f"no overhead calibration for site {site!r}; add xfer/eval_fix to the "
            f"calibration table (known: {sorted(calibration.sites)})"
        )
    if not tput_seq_s > 0:
        # A zero here is a ZeroDivisionError traceback out of the CLI, and a negative
        # one is worse: it produces a negative step time and a plausible-looking plan
        # in which the slow site is the fast one. Both come from one typo in
        # configs/site_throughput.env, so both are refused by name.
        raise KeyError(
            f"PWW_TPUT_{site.upper()} is {tput_seq_s!r}: throughput must be a positive "
            f"number of sequences/second. Fix configs/site_throughput.env, or re-run "
            f"scripts/titan/calibrate_throughput.sh on a job log from {site}"
        )
    compute = calibration.val_windows / (EVAL_FORWARD_SPEEDUP * tput_seq_s)
    return entry.xfer.value_s + entry.eval_fix.value_s + compute, worst_quality(
        entry.xfer.quality, entry.eval_fix.quality
    )


def make_member(lane_id: str, candidate: Candidate) -> Member:
    """Resolve a lane into the flat record the round loop reads.

    Reads the overhead the candidate was priced with at admission rather than
    recomputing it: the search evaluates on the order of 10^5 plans and a member is
    built for each, so this is a hot path, and it keeps the round loop free of any
    reference to the calibration table (which is a Mapping, hence unhashable, hence
    not usable in the memo keys the search depends on).
    """
    return Member(
        lane_id=lane_id,
        site=candidate.site,
        gpus=candidate.gpus,
        step_s=candidate.geometry.step_s,
        batch_seq=candidate.geometry.batch_seq,
        overhead_s=candidate.overhead_s,
        quality=candidate.overhead_quality,
    )


def balance_accums(members: Sequence[Member], *, balance: bool, cap: int = 8) -> tuple[int, ...]:
    """Gradient accumulation per member, mirroring scripts/titan/run_train.sh:305-308.

    `int(x + 0.5)` rather than round(), because awk's int(s/m + 0.5) is what actually
    runs on the cluster and Python's banker's rounding disagrees with it on exact
    halves. Floored at 1 and capped, same as the shell.

    Accumulation is the lever that fills the barrier without touching H, drift, the LR
    schedule or peak memory. What it costs is DARL blocks, linearly -- which is why
    `balance` is solved jointly with membership rather than switched on by policy.
    """
    if not balance or len(members) < 2:
        return tuple(1 for _ in members)
    # An infinite throughput cell used to arrive here as step_s == 0.0 and leave as a
    # ZeroDivisionError out of the CLI, nothing on stdout. The reader refuses such a
    # cell now; this names the lane if one ever gets past it, because the ratio below
    # is the one number the operator cannot check by eye.
    bad = [m.lane_id for m in members
           if not (math.isfinite(m.step_s) and m.step_s > 0)]
    if bad:
        raise ValueError(
            f"step time is not a duration for {', '.join(bad)}: balancing divides by "
            f"it, so the accumulation would be meaningless. A step time comes from a "
            f"measured (throughput, batch) pair -- refuse the registry cell instead.")
    slowest = max(m.step_s for m in members)
    return tuple(min(cap, max(1, int(slowest / m.step_s + 0.5))) for m in members)


def plan_accums(
    members: Sequence[Member], *, balance: bool, cap: int = 8
) -> dict[str, int]:
    """{lane_id: PWW_GRAD_ACCUM} for the WHOLE job, from the WHOLE planned membership.

    Accumulation is NOT a per-round quantity and must not be modelled as one.
    run_train.sh resolves PWW_GRAD_ACCUM once, before torchrun, into
    --training.global_batch_size (run_train.sh:323-329); it cannot change when a
    partner leaves the round. So the accumulation a site runs at is the one its
    PLANNED membership implies, and a solo headstart inside a two-site plan burns
    corpus at the balanced rate, not at accum 1.

    Simulating the live-membership accumulation instead under-prices a Snellius
    headstart by 5x in blocks and 2.3x in period, which is exactly the difference
    between "195 solo rounds then 9 federated ones" and "the corpus is gone before
    the partner arrives" -- i.e. between a plan and a trap.
    """
    accums = balance_accums(members, balance=balance, cap=cap)
    out: dict[str, int] = {}
    for accum, member in zip(accums, members):
        out[member.lane_id] = accum
    return out


def round_cost(
    fit: Sequence[Member],
    *,
    inner_steps: int,
    calibration: Calibration,
    balance: bool,
    balance_max: int = 8,
    eval_only: Sequence[Member] = (),
    stall_s: float = 0.0,
    explain: bool = False,
    accum_by_lane: Mapping[str, int] | None = None,
) -> RoundCost:
    """Wall-clock, tokens and DARL blocks of one round.

    `eval_only` is a lane that connected mid-round: Flower's configure_evaluate
    samples it before configure_fit ever does, so it pays the evaluate barrier that
    round and contributes nothing to the merge. Measured at latejoin round 98
    (`configure_fit: sampled 1`, then `configure_evaluate: sampled 2`).

    `stall_s` is the one-off cold-join stall charged to the incumbents' phase.

    `accum_by_lane` pins accumulation to what the JOBS WERE LAUNCHED WITH, which is
    what the simulator must pass: PWW_GRAD_ACCUM is resolved once at job start and
    does not follow the live membership. Omitting it re-derives accumulation from the
    members present THIS round, which is only correct when the round's membership is
    the whole plan.
    """
    if not fit:
        raise ValueError("a round needs at least one fitting member")
    if accum_by_lane is None:
        accums = balance_accums(fit, balance=balance, cap=balance_max)
    else:
        accums = tuple(max(1, int(accum_by_lane.get(m.lane_id, 1))) for m in fit)

    slow_i = max(range(len(fit)), key=lambda i: accums[i] * fit[i].step_s)
    phase_s = inner_steps * accums[slow_i] * fit[slow_i].step_s + stall_s

    overhead_s = calibration.merge.value_s
    for m in fit:
        overhead_s += m.overhead_s
    for m in eval_only:
        # It is in the evaluate barrier but not the fit barrier, so it costs its
        # evaluate half only. Splitting this out is cheap and keeps a joining lane
        # from being billed for a transport it never did.
        entry = calibration.sites[m.site]
        overhead_s += m.overhead_s - entry.xfer.value_s

    seq_per_round = sum(a * m.batch_seq for a, m in zip(accums, fit))
    tokens = inner_steps * calibration.seq_len * seq_per_round
    # Fractional on purpose: the ceil in blocks_for_phase only sets acquisition
    # granularity, and DARLDataSource._carry rides the remainder into the next phase,
    # so long-run consumption is exactly this rate.
    blocks = inner_steps * seq_per_round / calibration.block_size

    quality = worst_quality(
        calibration.merge.quality,
        calibration.regime_quality(tuple((m.site, m.gpus) for m in fit)),
        *(m.quality for m in fit),
        *(m.quality for m in eval_only),
    )

    period_s = phase_s + overhead_s
    # Rendered only on demand. The search calls this a few hundred thousand times and
    # f-string formatting was measurably the largest single line in the profile; the
    # report needs the arithmetic for a handful of headline regimes, not for every
    # round of every plan it discarded.
    arithmetic = ""
    if explain:
        terms = [f"{inner_steps}*{accums[slow_i] * fit[slow_i].step_s:.3f} [{fit[slow_i].lane_id}]"]
        if stall_s:
            terms.append(f"{stall_s:.0f} stall")
        terms.append(f"{calibration.merge.value_s:.0f}")
        terms += [f"{m.overhead_s:.1f} [{m.lane_id}]" for m in fit]
        terms += [f"{m.overhead_s - calibration.sites[m.site].xfer.value_s:.1f} [{m.lane_id} eval]"
                  for m in eval_only]
        arithmetic = f"{' + '.join(terms)} = {period_s:.1f} s"

    return RoundCost(
        period_s=period_s,
        phase_s=phase_s,
        overhead_s=overhead_s,
        accums=accums,
        tokens=tokens,
        blocks=blocks,
        quality=quality,
        arithmetic=arithmetic,
        stall_s=stall_s,
    )


def blocks_at_risk(
    member: Member, *, inner_steps: int, accum: int, block_size: int = 1024
) -> int:
    """Blocks a job death would put back through the TTL path: exactly one phase.

    Commits are checkpoint-gated and both shipped tomls checkpoint every round, so
    this is the whole cost of a short-job policy -- 4 blocks for Snellius at H=100
    accum 1, i.e. 0.15% of the corpus. Delegates to darl.space so the two cannot
    drift; `ranks=1` because batch_seq is already ranks x local_batch.
    """
    space = BlockSpace(num_samples=block_size, block_size=block_size)
    return blocks_for_phase(
        space,
        inner_steps=inner_steps,
        batch_size=member.batch_seq,
        ranks=1,
        grad_accum=accum,
    )


def residuals(calibration: Calibration = DEFAULT_CALIBRATION) -> list[dict]:
    """Model vs measured for every regime in the table. The basis for trusting a plan.

    A regime is reproduced when |predicted - measured| / measured <= tolerance. The
    extrapolated regime is expected to FAIL this and is reported, not hidden.
    """
    out: list[dict] = []
    for regime in calibration.regimes:
        members = []
        for idx, (site, gpus) in enumerate(regime.members):
            tput, batch = _regime_geometry(site, gpus)
            overhead_s, quality = site_overhead_s(site, tput, calibration)
            members.append(
                Member(
                    lane_id=f"{site}@{gpus}",
                    site=site,
                    gpus=gpus,
                    step_s=batch / tput,
                    batch_seq=batch,
                    overhead_s=overhead_s,
                    quality=quality,
                )
            )
        accums = regime.accums
        slow = max(range(len(members)), key=lambda i: accums[i] * members[i].step_s)
        phase = regime.inner_steps * accums[slow] * members[slow].step_s
        predicted = phase + calibration.merge.value_s + sum(m.overhead_s for m in members)
        error = (predicted - regime.measured_period_s) / regime.measured_period_s
        out.append(
            {
                "label": regime.label,
                "quality": regime.quality,
                "predicted_s": predicted,
                "measured_s": regime.measured_period_s,
                "rel_error": error,
                "within_tolerance": abs(error) <= regime.tolerance,
                "note": regime.note,
            }
        )
    return out


# The geometry cells the four measured regimes were run at. Kept here rather than in
# configs/site_throughput.env because residuals() must reproduce the calibration
# without reading a file -- a self-check that depends on an editable config checks
# nothing. The planner's own throughput comes from the registry, not from this.
_REGIME_GEOMETRY = {
    ("snellius", 4): (89.8, 32),
    ("snellius", 1): (24.4, 8),
    ("lumi", 8): (38.2, 64),
    ("lumi", 1): (4.75, 8),
}


def _regime_geometry(site: str, gpus: int) -> tuple[float, int]:
    try:
        return _REGIME_GEOMETRY[(site, gpus)]
    except KeyError:
        raise KeyError(f"no measured geometry for the calibration regime {site}@{gpus}") from None


# --------------------------------------------------------------------------
# H: constant on the full arm, a controller on the DCLT arm, 5x apart
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LRSchedule:
    """torchtitan's linear_warmup_stable_decay, reimplemented because QSR is a
    function of the LR and the planner must run with no torch installed.

    Verbatim from third_party/torchtitan/torchtitan/components/lr_scheduler.py:128-180,
    including the 0-indexed +1 adjustments -- a half-step error here is invisible in
    the LR and squared into H by the QSR rule.
    """

    lr: float = 4.5e-4
    warmup_steps: int = 300
    training_steps: int = 20000
    decay_ratio: float | None = 0.5
    decay_type: str = "cosine"
    min_lr_factor: float = 0.05

    def factor(self, step: int) -> float:
        warmup = min(self.warmup_steps, self.training_steps)
        if self.decay_ratio is not None:
            decay = round(self.training_steps * self.decay_ratio)
            if warmup + decay > self.training_steps:
                decay = self.training_steps - warmup
        else:
            decay = self.training_steps - warmup
        stable = self.training_steps + 1 - warmup - decay
        if step < warmup:
            return float((step + 1) / warmup) if warmup else 1.0
        if step < warmup + stable:
            return 1.0
        progress = float(step + 1 - warmup - stable) / decay
        if self.decay_type == "linear":
            adj = 1.0 - progress
        elif self.decay_type == "sqrt":
            adj = 1.0 - math.sqrt(progress)
        else:
            adj = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_factor + (1.0 - self.min_lr_factor) * adj

    def at(self, step: int) -> float:
        return self.lr * self.factor(max(0, min(step, self.training_steps)))


class JensenController:
    """The DCLT arm's H multiplier, exactly as central/strategy.py:597-633.

    The planner cannot know J -- it is a property of the loss landscape, not of the
    schedule -- so `observe` takes whatever gap the caller can supply (a replayed
    trace, or nothing). What is modelled unconditionally is the FREEZE: the
    controller is skipped whenever fewer than two clusters returned a finite
    local_eval_loss, so a federation that is solo most of the time runs the DCLT arm
    with a stale multiplier and H does not adapt while a site is queued. That is a
    scheduling consequence of a scheduling decision, which is why it belongs here.
    """

    def __init__(self, *, lo: float = 0.015, hi: float = 0.06, warmup_rounds: int = 5) -> None:
        self.lo = lo
        self.hi = hi
        self.warmup_rounds = warmup_rounds
        self.multiplier = 1.0
        self.ema: float | None = None
        self.gauge_rounds = 0

    def observe(self, gap: float | None, n_clusters: int) -> None:
        if gap is None or n_clusters < 2:
            return  # the freeze
        self.ema = gap if self.ema is None else 0.5 * self.ema + 0.5 * gap
        self.gauge_rounds += 1
        if self.gauge_rounds <= self.warmup_rounds:
            return  # measured, deliberately not acted on
        if self.ema >= 0.0:
            self.multiplier *= 0.7
        elif self.ema > -self.lo:
            self.multiplier *= 1.15
        elif self.ema < -self.hi:
            self.multiplier *= 0.85
        else:
            self.multiplier += (1.0 - self.multiplier) * 0.10
        self.multiplier = min(2.0, max(0.5, self.multiplier))


class HSchedule:
    """H per round. Subclasses are stateful across a simulation and must be reset."""

    def reset(self) -> None:
        raise NotImplementedError

    def next_h(self, *, global_step: int, n_fit: int) -> int:
        raise NotImplementedError

    def observe(self, *, n_fit: int, jensen_gap: float | None = None) -> None:
        return None

    @property
    def constant(self) -> bool:
        """True when the interval fast path is exact. Round-stepping is only needed
        where H moves."""
        return False


class FixedH(HSchedule):
    """The full arm. configs/central_aggregator_titan.yaml has no qsr-h0 key, so
    --qsr-h0 defaults to 0 and H is darl.inner_steps = 100 for the whole run.
    Applying the QSR formula to that arm is wrong by up to 5x per round."""

    def __init__(self, inner_steps: int) -> None:
        self.inner_steps = int(inner_steps)

    def reset(self) -> None:
        return None

    def next_h(self, *, global_step: int, n_fit: int) -> int:
        return self.inner_steps

    @property
    def constant(self) -> bool:
        return True


class QsrSchedule(HSchedule):
    """The DCLT arm. strategy._next_inner_steps (strategy.py:400-415), exactly.

    The cap is applied BEFORE the Jensen multiplier, which is not a detail: with the
    multiplier floored at 0.5 the effective ceiling is qsr_max/2, so the configured
    500 is really 250. The measured maximum across 213 merges was 400 and the last
    ~20 merges sat at 250-331.
    """

    def __init__(
        self,
        *,
        h0: int = 100,
        qsr_max: int = 500,
        warmup_steps: int = 300,
        lr: LRSchedule | None = None,
        lr_ref_pin: float | None = None,
        jensen: JensenController | None = None,
        fallback_inner_steps: int = 100,
    ) -> None:
        self.h0 = int(h0)
        self.qsr_max = int(qsr_max)
        self.warmup_steps = int(warmup_steps)
        self.lr = lr or LRSchedule()
        self.lr_ref_pin = lr_ref_pin
        self.jensen = jensen or JensenController()
        self.fallback_inner_steps = int(fallback_inner_steps)
        # eta_max is seeded from the configured peak rather than accumulated from
        # observations. On the server it is "the largest LR any cluster has ever
        # reported", which after warmup IS the peak -- and QSR is gated on
        # global_step >= qsr_warmup_steps, i.e. on warmup being over. Accumulating it
        # instead would make next_h() depend on whether the caller happened to step
        # the schedule from zero, which is a trap for anyone asking it about one round.
        self._lr_ref = self.lr.lr

    def reset(self) -> None:
        self._lr_ref = self.lr.lr
        self.jensen = JensenController(
            lo=self.jensen.lo, hi=self.jensen.hi, warmup_rounds=self.jensen.warmup_rounds
        )

    def next_h(self, *, global_step: int, n_fit: int) -> int:
        if self.h0 <= 0:
            return self.fallback_inner_steps
        lr_last = self.lr.at(global_step)
        if self.lr_ref_pin is not None:
            lr_ref = self.lr_ref_pin
        else:
            self._lr_ref = max(self._lr_ref, lr_last)
            lr_ref = self._lr_ref
        h = float(self.h0)
        if global_step >= self.warmup_steps and lr_last > 0 and lr_ref > 0:
            h *= min(lr_ref / lr_last, 1e3) ** 2
        h = min(h, float(self.qsr_max)) * self.jensen.multiplier
        floor = max(1, self.h0 // 2)
        return int(min(self.qsr_max, max(floor, round(h))))

    def observe(self, *, n_fit: int, jensen_gap: float | None = None) -> None:
        self.jensen.observe(jensen_gap, n_fit)


class ReplayH(HSchedule):
    """The measured H trace, for reproducing a run that happened. Holds the last
    value once the trace runs out rather than silently reverting to h0."""

    def __init__(self, trace: Sequence[int], fallback: int = 100) -> None:
        if not trace:
            raise ValueError("replay needs at least one H value")
        self.trace = tuple(int(h) for h in trace)
        self.fallback = int(fallback)
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def next_h(self, *, global_step: int, n_fit: int) -> int:
        h = self.trace[min(self._i, len(self.trace) - 1)]
        self._i += 1
        return h


# The DCLT arm's first 24 merges, from all_logs/pww-snellius-titan-dclt-25747722.out.
# The trajectory matters more than any single value: the controller spent ~100 rounds
# BELOW h0, so QSR did not reduce the merge count in the one real run (213 merges for
# a 20,000-step budget, against 200 at a fixed H=100).
DCLT_H_TRACE_HEAD = (
    100, 70, 50, 55, 60, 64, 67, 50, 57, 62, 71, 74,
    77, 79, 81, 83, 85, 97, 98, 99, 113, 79, 56, 50,
)


def make_schedule(
    h_model: str,
    *,
    inner_steps: int = 100,
    qsr_h0: int = 100,
    qsr_max: int = 500,
    qsr_warmup_steps: int = 300,
    lr: LRSchedule | None = None,
    trace: Sequence[int] | None = None,
) -> HSchedule:
    if h_model == "fixed":
        return FixedH(inner_steps)
    if h_model == "qsr":
        return QsrSchedule(
            h0=qsr_h0,
            qsr_max=qsr_max,
            warmup_steps=qsr_warmup_steps,
            lr=lr,
            fallback_inner_steps=inner_steps,
        )
    if h_model == "replay":
        return ReplayH(trace or DCLT_H_TRACE_HEAD, fallback=inner_steps)
    raise ValueError(f"unknown h_model {h_model!r} (fixed|qsr|replay)")
