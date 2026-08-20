"""Walk a submission plan forward in time, round by round, and count what happens.

This is the authoritative engine. The closed-form inequalities in `search` are
cross-checks against it, never a second decision path, because only a simulator that
debits DARL blocks round by round finds the result that matters most here: a solo
headstart can EXHAUST THE CORPUS before its partner arrives and yield zero federated
merges, outcome-identical to the staggered-start trap it was meant to avoid.

Three things it models that no closed form does:

  * membership is a step function of time, and both the barrier max and the
    accumulation each site is given change with it;
  * a lane that connects mid-round is sampled by configure_evaluate before
    configure_fit ever sees it, so it pays the evaluate barrier and contributes
    nothing that round (measured, latejoin round 98);
  * DARL blocks and Flower round attempts are budgets that bind BEFORE walltime does,
    and with max_epochs = 1 exhausting the corpus ends the run outright -- `acquire`
    returns epoch_complete forever, there is no wraparound.

No RNG anywhere: identical inputs give a byte-identical timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Iterable, Sequence

from .model import (
    Calibration,
    Candidate,
    DarlState,
    Interval,
    Link,
    Member,
    Option,
    PlanConfig,
    Round,
    SiteLedger,
    Timeline,
    worst_quality,
    IDENTIFIED,
)
from .rounds import (
    HSchedule, RoundCost, make_member, make_schedule, plan_accums, round_cost)

# Float slop for "does this round finish before that job dies". Seconds of walltime
# are not measured to better than this and an exact-equality test here turns into a
# phantom departure.
_EPS = 1e-6


@dataclass(frozen=True)
class _Window:
    """One link's productive interval, with the member record it puts on the wire."""

    lane_id: str
    site: str
    start_s: float
    end_s: float
    cold: bool
    member: Member


def expand_links(
    option: Option,
    *,
    config: PlanConfig,
    wait_quantile: str | None = None,
    warm_checkpoint: bool = False,
) -> tuple[Link, ...]:
    """One Link per job this option would submit.

    A LANE is a durable identity -- replica id, its own PWW_DUMP, its own DCP
    checkpoint, its own DARL cluster record, its own Flower client id and delta-blob
    key. A LINK is one job inside it. Concurrency comes from adding lanes; it must
    never come from oversubscribing a lane, because two live jobs under one cluster
    id are refused outright (DARL 503 cluster_busy, and the aggregator separately
    drops every contribution under a duplicated id).

    Chaining defaults to self-resubmission rather than --dependency: a
    dependency-held job is not eligible for backfill, which forfeits exactly the
    advantage that motivated short jobs in the first place.
    """
    quantile = wait_quantile or config.wait_quantile
    wait_s = option.candidate.wait.eff_at(config.discount_strength, quantile)
    walltime_s = float(option.candidate.walltime_s)
    startup_s = option.candidate.startup_s
    lead_s = config.chain_lead_s

    links: list[Link] = []
    for lane in range(option.lanes):
        lane_id = f"{option.site}-l{lane}"
        arrival = option.begin_s + wait_s
        submit = option.begin_s
        prev_end = arrival + walltime_s
        for index in range(option.links_per_lane):
            if index > 0:
                if option.chain == "self":
                    # Submitted from inside the predecessor, before training starts,
                    # with --begin=now+(T-lead). Under chain_wait_overlap the queue
                    # clock and the begin hold run concurrently, which is the
                    # optimistic reading and the one the design assumes: the job is
                    # pending from submission and eligible from its begin time. If
                    # Slurm instead defers the whole wait until eligibility, each
                    # link costs a further w(T) -- flip chain_wait_overlap and the
                    # plan reprices. The report names this as an assumption.
                    submit = arrival
                    if config.chain_wait_overlap:
                        arrival = arrival + max(walltime_s - lead_s, wait_s)
                    else:
                        arrival = arrival + (walltime_s - lead_s) + wait_s
                elif option.chain == "singleton":
                    # --dependency=singleton/afterany: not backfill-eligible while
                    # held, so the wait starts only once the predecessor is gone.
                    submit = prev_end
                    arrival = prev_end + wait_s
                else:
                    break
            productive = arrival + startup_s
            end = arrival + walltime_s
            links.append(
                Link(
                    lane_id=lane_id,
                    site=option.site,
                    index=index,
                    submit_s=submit,
                    arrival_s=arrival,
                    productive_s=productive,
                    end_s=end,
                    # Only the lane's first link is cold. Every later link resumes the
                    # lane's own DCP checkpoint -- weights, AdamW moments, LR schedule
                    # and dataloader -- so it pays no join transient. That is the whole
                    # return on the lane abstraction.
                    cold=(index == 0) and not warm_checkpoint,
                    walltime_s=walltime_s,
                )
            )
            prev_end = end
    return tuple(links)


# Beside the branches above, because that is the only place a policy becomes real.
# cli.py used to keep its own copy for argparse and search.py a third for pricing;
# a policy taught to expand_links and not to those two is refused at the flag while
# the simulator supports it, which is the "priced one way, emitted another" bug with
# the sign flipped. Anything added to the loop above belongs here in the same commit
# -- test_plan_integration.py reads the loop's own string literals and fails if not.
CHAIN_POLICIES: tuple[str, ...] = ("none", "self", "singleton")


def can_chain(policy: str, *, config: PlanConfig, candidate: Candidate,
              warm: bool = False) -> bool:
    """Does expand_links really produce a second link for this policy?

    Asked of the simulator, not of CHAIN_POLICIES, so a name that is listed but whose
    branch was removed still answers False. "none" is chainable-by-omission: it is
    priced as one job per lane, which is what it expands to.
    """
    if policy == "none":
        return True
    probe = Option(site=candidate.site, candidate=candidate, lanes=1,
                   links_per_lane=2, chain=policy, begin_s=0.0)
    return len(expand_links(probe, config=config, warm_checkpoint=warm)) == 2


@lru_cache(maxsize=8192)
def _option_windows(
    option: Option, config: PlanConfig, wait_quantile: str | None, warm: bool,
) -> tuple[tuple[Link, ...], tuple[_Window, ...], tuple[str, ...]]:
    """One option's jobs and productive windows, memoised.

    The search re-simulates the same option inside thousands of different selections
    and this is pure in its arguments, so it is computed once. Everything in the key
    is a frozen dataclass of builtins, which is the reason model.py is written that way.
    """
    links: list[Link] = []
    windows: list[_Window] = []
    warnings: list[str] = []
    proto = make_member("", option.candidate)
    option_links = expand_links(
        option, config=config, wait_quantile=wait_quantile, warm_checkpoint=warm)
    links.extend(option_links)

    by_lane: dict[str, list[Link]] = {}
    for link in option_links:
        by_lane.setdefault(link.lane_id, []).append(link)
    for lane_id, lane_links in by_lane.items():
        member = replace(proto, lane_id=lane_id)
        for prev, link in zip([None] + lane_links[:-1], lane_links):
            start = min(link.productive_s, config.horizon_s)
            end = min(link.end_s, config.horizon_s)
            if end - start <= _EPS:
                if link.productive_s >= link.end_s:
                    warnings.append(
                        f"{lane_id} link {link.index}: startup "
                        f"{link.productive_s - link.arrival_s:.0f} s exceeds the "
                        f"{link.walltime_s / 3600:g} h walltime, so it contributes nothing")
                continue
            if prev is not None and link.arrival_s < prev.end_s - _EPS:
                # Two links of one lane overlapping is the DARL failure mode, not a
                # scheduling nicety: register() refuses a second live incarnation of a
                # cluster id with 503 cluster_busy, and the client gives up after ~13-40 s
                # of retries and dies at startup.
                warnings.append(
                    f"{lane_id} link {link.index} starts {prev.end_s - link.arrival_s:.0f} s "
                    f"before link {link.index - 1} ends: same cluster id, so the successor "
                    f"is refused (DARL 503 cluster_busy). Give it its own lane, or rely on "
                    f"the SIGTERM release path and keep chain_lead_s at 0.")
            windows.append(_Window(lane_id, link.site, start, end, link.cold, member))
    return tuple(links), tuple(windows), tuple(warnings)


def _windows(
    selection: Sequence[Option],
    *,
    config: PlanConfig,
    wait_quantile: str | None,
    warm: dict[str, bool] | None = None,
) -> tuple[tuple[Link, ...], list[_Window], list[str]]:
    warm = warm or {}
    links: list[Link] = []
    windows: list[_Window] = []
    warnings: list[str] = []
    for option in selection:
        o_links, o_windows, o_warnings = _option_windows(
            option, config, wait_quantile, warm.get(option.site, False))
        links.extend(o_links)
        windows.extend(o_windows)
        warnings.extend(o_warnings)
    windows.sort(key=lambda w: (w.start_s, w.lane_id))
    return tuple(links), windows, warnings


class _Accumulator:
    """Per-site running totals. A plain object rather than a dataclass because the
    round loop writes to it a few hundred thousand times per search."""

    __slots__ = ("compute_s", "merges", "federated", "tokens", "blocks", "accums")

    def __init__(self) -> None:
        self.compute_s = 0.0
        self.merges = 0
        self.federated = 0
        self.tokens = 0
        self.blocks = 0.0
        self.accums: set[int] = set()


def simulate(
    selection: Sequence[Option],
    *,
    config: PlanConfig,
    calibration: Calibration,
    darl: DarlState,
    balance: bool,
    schedule: HSchedule | None = None,
    wait_quantile: str | None = None,
    warm: dict[str, bool] | None = None,
    record_rounds: bool = False,
) -> Timeline:
    """Round-step the plan to the horizon, or to whichever budget binds first.

    `record_rounds` off by default because the search discards almost every plan it
    simulates and building one dataclass per round of each was the largest allocation
    in the profile. The winner is re-simulated with it on, which is what `explain`
    replays.
    """
    links, windows, warnings = _windows(
        selection, config=config, wait_quantile=wait_quantile, warm=warm)
    schedule = schedule or make_schedule(config.h_model, inner_steps=config.inner_steps)
    schedule.reset()

    # Accumulation is fixed for the life of a JOB, not recomputed per round.
    # run_train.sh resolves PWW_GRAD_ACCUM once, before torchrun, into
    # --training.global_batch_size; it does not fall back to 1 when the partner
    # leaves. Deriving it from the live membership instead prices a solo headstart
    # inside a two-site plan at 1/5 of its real corpus burn -- which is the
    # difference between a headstart and trap_corpus_exhausted.
    lane_members: list[Member] = []
    seen_lanes: set[str] = set()
    for w in windows:
        if w.lane_id not in seen_lanes:
            seen_lanes.add(w.lane_id)
            lane_members.append(w.member)
    accum_by_lane = plan_accums(lane_members, balance=balance, cap=config.balance_max)

    blocks_available = max(0.0, darl.unassigned - config.reserve_blocks)
    blocks_left = blocks_available
    attempts_left = max(0, config.num_rounds)
    global_step = 0
    t = 0.0
    horizon = config.horizon_s

    rounds: list[Round] = []
    intervals: list[Interval] = []
    per_site: dict[str, _Accumulator] = {}
    cold_charged: set[str] = set()
    darl_exhausted_s: float | None = None
    attempts_exhausted_s: float | None = None
    # Rounds per calibration quality. The plan's quality is the DOMINANT regime's,
    # not the worst one ever touched: two lumi-solo rounds either side of a
    # membership change must not make 79 measured federated rounds unrankable.
    quality_rounds: dict[str, int] = {}
    starts = sorted({w.start_s for w in windows})

    n_fed = 0
    n_solo = 0
    tokens_total = 0
    first_fed_s: float | None = None
    last_fed_s: float | None = None

    cur_members: tuple[str, ...] | None = None
    cur_quality = IDENTIFIED
    cur_t0 = 0.0
    cur_rounds = 0
    cur_fed = 0
    cur_period_sum = 0.0
    cur_h_sum = 0
    cur_stop = "horizon"

    def flush(t1: float, stop: str) -> None:
        nonlocal cur_members, cur_rounds, cur_fed, cur_period_sum, cur_h_sum, cur_quality
        if cur_members is None:
            return
        intervals.append(
            Interval(
                t0_s=cur_t0,
                t1_s=t1,
                members=cur_members,
                rounds=cur_rounds,
                federated_rounds=cur_fed,
                inner_steps=int(cur_h_sum / cur_rounds) if cur_rounds else 0,
                period_s=cur_period_sum / cur_rounds if cur_rounds else 0.0,
                blocks_left=blocks_left,
                stop_cause=stop,
                # THIS interval's own regime, not the running worst-so-far. A tag
                # that leaks forward makes every later identified regime look
                # untrustworthy and hides which cell is actually unmeasured.
                quality=cur_quality,
            )
        )
        cur_members = None
        cur_quality = IDENTIFIED
        cur_rounds = 0
        cur_fed = 0
        cur_period_sum = 0.0
        cur_h_sum = 0

    def charge(n: int, cost: RoundCost, fit: Sequence[Member], start_s: float, h: int) -> None:
        nonlocal blocks_left, attempts_left, global_step, cur_quality
        nonlocal n_fed, n_solo, tokens_total, first_fed_s, last_fed_s
        # Federated means two SITES, not two clients. Two lanes at one site are two
        # Flower endpoints and the server does merge them, but they share the
        # hardware, the queue, the WAN link and the failure -- averaging them is not
        # the measurement this campaign exists to make, and counting it as one would
        # let the planner "federate" by submitting the same site twice.
        federated = len({m.site for m in fit}) >= 2
        if federated:
            n_fed += n
            first_fed_s = start_s if first_fed_s is None else first_fed_s
            last_fed_s = start_s + (n - 1) * cost.period_s
        else:
            n_solo += n
        tokens_total += n * cost.tokens
        if record_rounds:
            for k in range(n):
                rounds.append(
                    Round(
                        index=len(rounds),
                        start_s=start_s + k * cost.period_s,
                        period_s=cost.period_s,
                        phase_s=cost.phase_s,
                        overhead_s=cost.overhead_s,
                        fit=tuple(m.lane_id for m in fit),
                        eval_only=(),
                        accums=cost.accums,
                        inner_steps=h,
                        tokens=cost.tokens,
                        blocks=cost.blocks,
                        federated=federated,
                        quality=cost.quality,
                        stall_s=cost.stall_s if k == 0 else 0.0,
                    )
                )
        blocks_left -= n * cost.blocks
        attempts_left -= n
        global_step += n * h
        cur_quality = worst_quality(cur_quality, cost.quality)
        quality_rounds[cost.quality] = quality_rounds.get(cost.quality, 0) + n
        for accum, member in zip(cost.accums, fit):
            acc = per_site.setdefault(member.site, _Accumulator())
            acc.compute_s += n * h * accum * member.step_s
            acc.merges += n
            acc.federated += n if federated else 0
            acc.tokens += n * h * accum * member.batch_seq * calibration.seq_len
            acc.blocks += n * h * accum * member.batch_seq / calibration.block_size
            acc.accums.add(accum)

    while t < horizon - _EPS and attempts_left > 0 and blocks_left > 0:
        live = [w for w in windows if w.start_s <= t + _EPS < w.end_s]
        if not live:
            nxt = next((s for s in starts if s > t + _EPS), None)
            if nxt is None:
                break
            flush(t, "idle")
            t = nxt
            continue

        fit_windows = list(live)
        cost = None
        # ONE draw per round. next_h is stateful for the replay schedule (it walks
        # the measured trace) and drawing it again below would consume two entries
        # per round, replaying every other value and holding the last one at half
        # the trace length. No schedule reads n_fit, so the live count is the right
        # argument even before the departure fixpoint settles.
        h = schedule.next_h(global_step=global_step, n_fit=len(fit_windows))
        # Departure fixpoint: a member whose walltime ends before the round would
        # close forfeits it, which shortens the round, which may let another member
        # survive after all. At most |S| passes.
        while fit_windows:
            cost = round_cost(
                [w.member for w in fit_windows],
                inner_steps=h,
                calibration=calibration,
                balance=balance,
                balance_max=config.balance_max,
                accum_by_lane=accum_by_lane,
            )
            survivors = [w for w in fit_windows if w.end_s >= t + cost.period_s - _EPS]
            if len(survivors) == len(fit_windows):
                break
            fit_windows = survivors
        if not fit_windows or cost is None:
            nxt = min([w.end_s for w in live] + [horizon])
            flush(t, "walltime")
            t = max(nxt, t + _EPS)
            continue

        members = tuple(w.lane_id for w in fit_windows)
        if cur_members != members:
            flush(t, cur_stop)
            cur_members = members
            cur_t0 = t

        next_start = next((s for s in starts if s > t + _EPS), horizon)
        stable_until = min([w.end_s for w in fit_windows] + [next_start, horizon])

        if schedule.constant and t + cost.period_s <= stable_until + _EPS:
            # Fast path. Membership, H and therefore the period are all constant over
            # [t, stable_until), so the round count is arithmetic rather than a loop.
            # This is what makes exhaustive search over ~10^5 plans affordable.
            n_time = int((stable_until - t + _EPS) // cost.period_s)
            n_blocks = int(blocks_left // cost.blocks) if cost.blocks > 0 else n_time
            n = max(0, min(n_time, n_blocks, attempts_left))
            if n > 0:
                charge(n, cost, [w.member for w in fit_windows], t, h)
                cur_rounds += n
                cur_fed += n if len({w.site for w in fit_windows}) >= 2 else 0
                cur_period_sum += n * cost.period_s
                cur_h_sum += n * h
                t += n * cost.period_s
                schedule.observe(n_fit=len(fit_windows))
            if n < n_time:
                cur_stop = "darl_exhausted" if n == n_blocks else "attempts_exhausted"
                if n == n_blocks and darl_exhausted_s is None:
                    darl_exhausted_s = t
                if attempts_left <= 0 and attempts_exhausted_s is None:
                    attempts_exhausted_s = t
                break
            cur_stop = "membership" if stable_until < horizon else "horizon"
            continue

        # Slow path: H moves, or a lane joins inside this round.
        joiners = [w for w in windows if t + _EPS < w.start_s < t + cost.period_s]
        stall_s = 0.0
        eval_only: list[_Window] = []
        for _ in range(len(windows) + 1):
            new_eval = [w for w in joiners if w not in fit_windows]
            cold_new = [w for w in new_eval if w.cold and w.lane_id not in cold_charged]
            # The stall is charged to the round the cold lane joins DURING, matching
            # the one measurement (round 98: the incumbent's phase went 38 s -> 416 s
            # while LUMI registered). It needs an incumbent to stall; a cold lane
            # arriving into an empty federation stalls nobody.
            stall_s = calibration.tau_stall_s if (cold_new and fit_windows) else 0.0
            trial = round_cost(
                [w.member for w in fit_windows],
                inner_steps=h,
                calibration=calibration,
                balance=balance,
                balance_max=config.balance_max,
                eval_only=[w.member for w in new_eval],
                stall_s=stall_s,
                accum_by_lane=accum_by_lane,
            )
            grown = [w for w in windows if t + _EPS < w.start_s < t + trial.period_s]
            if len(grown) == len(joiners):
                cost = trial
                eval_only = new_eval
                break
            joiners = grown
        if cost.blocks > blocks_left:
            cur_stop = "darl_exhausted"
            darl_exhausted_s = darl_exhausted_s or t
            break
        for w in eval_only:
            if w.cold:
                cold_charged.add(w.lane_id)
        charge(1, cost, [w.member for w in fit_windows], t, h)
        if record_rounds:
            rounds[-1] = replace(rounds[-1], eval_only=tuple(w.lane_id for w in eval_only))
        cur_rounds += 1
        cur_fed += 1 if len({w.site for w in fit_windows}) >= 2 else 0
        cur_period_sum += cost.period_s
        cur_h_sum += h
        schedule.observe(n_fit=len(fit_windows))
        t += cost.period_s
        cur_stop = "horizon"

    if blocks_left <= 0 and darl_exhausted_s is None:
        darl_exhausted_s = t
    if attempts_left <= 0 and attempts_exhausted_s is None:
        attempts_exhausted_s = t
    flush(min(t, horizon), cur_stop)

    run_end_s = min(max(t, 0.0), horizon)
    ledgers = _ledgers(links, windows, per_site, config, run_end_s)
    billed_s = sum(l.gpu_s for l in ledgers)
    trained_s = sum(l.live_s * l.gpus for l in ledgers)
    if billed_s > trained_s + 3600.0:
        warnings.append(
            f"the run ends at {run_end_s / 3600:.1f} h but the jobs hold their "
            f"allocation to the horizon: {(billed_s - trained_s) / 3600:.0f} GPU-h of "
            f"the {billed_s / 3600:.0f} reported are billed AFTER there is anything "
            f"left to train on. The client does not exit when the corpus is gone -- "
            f"FlowerClient sets self.done (flower_client.py:383,447) and nothing reads "
            f"it -- so it keeps answering rounds with zero tokens until Slurm kills it. "
            f"Shorten the chain, or accept the tokens/GPU-h below as the real figure.")
    return Timeline(
        links=links,
        intervals=tuple(intervals),
        rounds=tuple(rounds),
        ledgers=ledgers,
        federated_merges=n_fed,
        solo_merges=n_solo,
        tokens=tokens_total,
        blocks_used=blocks_available - blocks_left,
        blocks_available=blocks_available,
        attempts_used=n_fed + n_solo,
        gpu_s=billed_s,
        compute_s=sum(l.compute_s for l in ledgers),
        darl_exhausted_s=darl_exhausted_s,
        attempts_exhausted_s=attempts_exhausted_s,
        first_federated_s=first_fed_s,
        last_federated_s=last_fed_s,
        quality=_dominant_quality(quality_rounds),
        warnings=tuple(warnings),
        run_end_s=run_end_s,
    )


def _dominant_quality(rounds_by_quality: dict[str, int]) -> str:
    """The quality of the regime the plan actually SPENDS ITS ROUNDS IN.

    `--overhead-model measured` refuses to rank a plan whose DOMINANT regime is
    extrapolated. Folding every regime the plan ever touches into a monotone
    worst-so-far instead makes two lumi-solo rounds either side of a membership
    change demote 79 measured federated ones, and the demotion silently changes the
    recommendation without appearing anywhere in the report. Ties go to the worse
    quality, so a plan that is half unmeasured is still refused.
    """
    if not rounds_by_quality:
        return IDENTIFIED
    top = max(rounds_by_quality.values())
    return worst_quality(*[q for q, n in rounds_by_quality.items() if n == top])


# --------------------------------------------------------------------------
# hour accounting -- three columns, not one
# --------------------------------------------------------------------------


def _merge(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if end - start <= _EPS:
            continue
        if out and start <= out[-1][1] + _EPS:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _overlap(spans: Sequence[tuple[float, float]], lo: float, hi: float) -> float:
    return sum(max(0.0, min(end, hi) - max(start, lo)) for start, end in spans)


def _coresidency(by_site: dict[str, list[tuple[float, float]]]) -> list[tuple[float, float]]:
    """Instants at which two DIFFERENT sites are both live.

    Two lanes of one site do not federate with each other in any interesting sense --
    they are the same hardware, the same queue and the same failure -- so co-residency
    is defined between sites, not between lanes.
    """
    edges = sorted({t for spans in by_site.values() for span in spans for t in span})
    out: list[tuple[float, float]] = []
    for lo, hi in zip(edges, edges[1:]):
        mid = 0.5 * (lo + hi)
        n = sum(1 for spans in by_site.values()
                if any(s <= mid < e for s, e in spans))
        if n >= 2:
            out.append((lo, hi))
    return _merge(out)


def _ledgers(
    links: Sequence[Link],
    windows: Sequence[_Window],
    per_site: dict[str, _Accumulator],
    config: PlanConfig,
    run_end_s: float,
) -> tuple[SiteLedger, ...]:
    by_site: dict[str, list[tuple[float, float]]] = {}
    billed_by_site: dict[str, list[tuple[float, float]]] = {}
    gpus: dict[str, int] = {}
    for w in windows:
        # TWO span sets, because they answer two different questions. The first is
        # clipped to when the run ended and drives the hour SPLIT: nothing trains
        # after the corpus is gone. The second is the allocation the plan actually
        # asks Slurm for, and it is what gets BILLED -- the jobs do NOT exit when
        # DARL runs dry (FlowerClient sets self.done and nothing reads it), so they
        # sit on the hardware answering empty rounds until walltime. Clipping GPU-h
        # to run_end flattered a 6-link chain by 2.7x.
        by_site.setdefault(w.site, []).append((w.start_s, min(w.end_s, run_end_s)))
        billed_by_site.setdefault(w.site, []).append((w.start_s, w.end_s))
        gpus[w.site] = w.member.gpus
    live = {site: _merge(spans) for site, spans in by_site.items()}
    billed = {site: _merge(spans) for site, spans in billed_by_site.items()}
    fed = _coresidency(live)
    fed_lo = fed[0][0] if fed else None
    fed_hi = fed[-1][1] if fed else None

    out: list[SiteLedger] = []
    for site in sorted(set(list(live) + [l.site for l in links])):
        spans = live.get(site, [])
        site_links = [l for l in links if l.site == site]
        total = sum(end - start for start, end in spans)
        coresident = sum(_overlap(spans, lo, hi) for lo, hi in fed)
        headstart = _overlap(spans, 0.0, fed_lo) if fed_lo is not None else total
        tail = _overlap(spans, fed_hi, run_end_s) if fed_hi is not None else 0.0
        if fed_lo is None:
            headstart, tail = total, 0.0
        # Presence that is neither before the first co-residency nor after the last
        # one: solo hours BETWEEN two co-residency spells. Without this column the
        # three-way split silently drops them and live_s -- hence idle_fraction and
        # the barrier-idle headline -- is computed against the wrong denominator,
        # by up to 2.5x when the partner's presence is fragmented.
        between = max(0.0, total - headstart - coresident - tail)
        acc = per_site.get(site, _Accumulator())
        # gpu_s counts BILLED hardware over the ALLOCATION, not over the part of it
        # that had corpus left. The hours a site spends idling -- at the barrier or
        # after the epoch is covered -- are exactly the hours this planner exists to
        # expose, and hiding them in the denominator would defeat that.
        gpu_s = sum(end - start for start, end in billed.get(site, [])) * gpus.get(site, 0)
        span_lo = min((s for s, _ in spans), default=0.0)
        span_hi = max((e for _, e in spans), default=0.0)
        out.append(
            SiteLedger(
                site=site,
                queued_s=sum(l.arrival_s - l.submit_s for l in site_links),
                startup_s=sum(l.productive_s - l.arrival_s for l in site_links),
                headstart_s=headstart,
                coresident_s=coresident,
                tail_s=tail,
                between_s=between,
                gpus=gpus.get(site, 0),
                gap_s=max(0.0, (span_hi - span_lo) - total),
                compute_s=acc.compute_s,
                merges=acc.merges,
                federated_merges=acc.federated,
                tokens=acc.tokens,
                blocks=acc.blocks,
                gpu_s=gpu_s,
                accums=tuple(sorted(acc.accums)),
            )
        )
    return tuple(out)
