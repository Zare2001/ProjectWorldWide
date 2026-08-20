"""Admission, option generation, and the joint solve over membership x shape x
accumulation.

Membership, shape and balancing are ONE decision and cannot be taken in sequence.
Adding a site changes the barrier, which changes every incumbent's accumulation,
which changes every site's corpus burn, which changes when the run ends -- so an
incumbent's best shape is not the one it had before the partner arrived. The greedy
pass therefore re-optimises every incumbent whenever it adds a site, and for two
sites the enumeration is exhaustive so the answer is proved rather than argued.

The closed-form inequalities at the bottom are CROSS-CHECKS. They are computed
alongside the simulator and reported when they disagree with it, because a
disagreement means one of the two is wrong and the reader should know which. The
simulator is authoritative: it is the only one of the two that debits DARL blocks.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import NamedTuple, Sequence

from .model import (
    Calibration,
    Candidate,
    CrossCheck,
    DarlState,
    Exclusion,
    MarginalEntry,
    Option,
    Plan,
    PlanConfig,
    PlannerInputs,
    Score,
    SearchReport,
    SensitivityRow,
    ShapeKey,
    SiteInput,
    Timeline,
    Trap,
    EXTRAPOLATED,
)
from .rounds import DEFAULT_CALIBRATION, make_member, make_schedule, round_cost, site_overhead_s
from .timeline import can_chain, expand_links, simulate

_EPS = 1e-9


# --------------------------------------------------------------------------
# phase 0 -- admission. Never silently plan.
# --------------------------------------------------------------------------


def probe_shape_json(site: str, key: ShapeKey, name: str | None = None) -> str:
    """The exact object to paste into configs/slurm_probe/<site>.json.

    An exclusion a reader cannot act on is a silent drop with extra words, and this
    is the exclusion that matters most: the planner must never extrapolate w(T) to a
    walltime nobody probed, so the only honest response to a wanted-but-unprobed
    shape is to name the shape.
    """
    hours = key.walltime_s // 3600
    minutes = (key.walltime_s % 3600) // 60
    name = name or f"{key.partition.replace('-', '')}_{key.nodes}node_{hours}h"
    args = []
    if key.account:
        args += ["-A", key.account]
    args += ["-p", key.partition, "-N", str(key.nodes),
             "--gpus-per-node", str(key.gpus_per_node),
             "-t", f"{hours}:{minutes:02d}:00"]
    rendered = ", ".join(f'"{a}"' for a in args)
    return '{"name": "%s", "args": [%s]}' % (name, rendered)


def admit(
    site: SiteInput,
    *,
    config: PlanConfig,
    calibration: Calibration,
) -> tuple[list[Candidate], list[Exclusion]]:
    """Turn one site's raw inputs into priced candidates, or into reasons why not."""
    candidates: list[Candidate] = []
    exclusions: list[Exclusion] = []

    for key in site.wanted_shapes:
        if not any(s.key == key for s in site.shapes):
            exclusions.append(
                Exclusion(
                    code="shape_not_probed",
                    # describe() already leads with the cluster; prefixing site.site
                    # too printed "lumi/lumi/standard-g 8 gpu 4 h".
                    subject=key.describe(),
                    reason=(f"no probe ROW for {key.describe()} in the window read; "
                            f"w(T) is never interpolated in walltime, device count or "
                            f"throughput"),
                    # This set is built FROM configs/slurm_probe/<site>.json, so the
                    # shape is almost always already configured and the gap is a
                    # COLLECTOR problem, not a config one. Saying "add it" sends the
                    # reader to edit a file that already contains the entry -- and
                    # editing the -t of an existing entry is itself the thing that
                    # blends two walltimes into one wait distribution.
                    fix=(f"configs/slurm_probe/{site.site}.json is where this shape was "
                         f"asked for, so it is probably already listed: check that the "
                         f"collector cron is running on the {site.site} login node and "
                         f"that its `server` is the instance this plan read (a failed "
                         f"post is dropped outright -- there is no retry or queue), and "
                         f"that --probe-window-h covers a cycle. Only if it is genuinely "
                         f"absent, add {probe_shape_json(site.site, key)} -- probes run "
                         f"serially with a 60 s timeout each, so a shape is not free: 11 "
                         f"already stretch a cycle toward the 10-minute cron."),
                )
            )

    for shape in site.shapes:
        subject = f"{site.site}/{shape.name}"
        wait = site.waits.get(shape.name)
        if wait is None:
            exclusions.append(Exclusion(
                "no_probe", subject,
                f"no probe row for shape {shape.name!r}",
                f"add {probe_shape_json(site.site, shape.key, shape.name)} to "
                f"configs/slurm_probe/{site.site}.json"))
            continue
        if not wait.ok:
            exclusions.append(Exclusion(
                "probe_failed", subject,
                f"last probe for {shape.name!r} failed: {wait.message or '(no message)'}",
                "fix the sbatch arguments or the account, then wait one collector cycle"))
            continue
        if wait.probe_age_s > config.max_probe_age_s:
            exclusions.append(Exclusion(
                "probe_stale", subject,
                f"probe is {wait.probe_age_s / 3600:.1f} h old, limit is "
                f"{config.max_probe_age_s / 3600:.1f} h",
                "check the collector cron on the login node; a failed post is lost "
                "outright, there is no retry or queue"))
            continue
        if site.submitter and wait.probed_by_user and wait.probed_by_user != site.submitter:
            exclusions.append(Exclusion(
                "probed_by_other_account", subject,
                f"probed by {wait.probed_by_user!r}, you submit as {site.submitter!r}: "
                f"--test-only is conditioned on the probing account's fairshare, QOS and "
                f"priority, so this number describes a different queue",
                f"run the collector as {site.submitter}"))
            continue
        geometry = site.geometries.get(shape.key.gpus)
        if geometry is None:
            exclusions.append(Exclusion(
                "no_throughput", subject,
                f"no measured throughput for {site.site}@{shape.key.gpus} devices",
                f"run scripts/titan/calibrate_throughput.sh WITHOUT --write on a "
                f"{shape.key.gpus}-device job log from {site.site}, then add its two "
                f"printed lines to configs/site_throughput.env RENAMED to "
                f"PWW_TPUT_{site.site.upper()}_{shape.key.gpus}/"
                f"PWW_BATCH_{site.site.upper()}_{shape.key.gpus}. Do NOT use --write: it "
                f"only ever writes the site-level PWW_TPUT_{site.site.upper()}/"
                f"PWW_BATCH_{site.site.upper()} pair and REPLACES it, so running it on a "
                f"reduced-geometry log overwrites the reference cell and every other "
                f"shape at this site is then excluded for the same reason. Step time is "
                f"device-count invariant to within 8%, but that is a finding, not a "
                f"licence to extrapolate."))
            continue
        try:
            overhead_s, quality = site_overhead_s(site.site, geometry.tput_seq_s, calibration)
        except KeyError as exc:
            exclusions.append(Exclusion(
                "no_overhead_calibration", subject, str(exc),
                "difference one round of this site's logs merge-complete to "
                "merge-complete and add xfer/eval_fix to the calibration table"))
            continue
        if shape.key.walltime_s <= site.startup_s:
            exclusions.append(Exclusion(
                "walltime_under_startup", subject,
                f"{shape.key.walltime_s / 3600:g} h walltime does not cover "
                f"{site.startup_s / 60:.0f} min of startup",
                "probe a longer shape, or measure c properly: no job script prints a "
                "timestamp before torchrun, so every c in this plan is a lower bound"))
            continue
        candidates.append(Candidate(
            site=site.site, shape=shape, wait=wait, geometry=geometry,
            startup_s=site.startup_s, overhead_s=overhead_s, overhead_quality=quality))

    if not candidates:
        per_shape = [e for e in exclusions if e.subject.startswith(site.site + "/")]
        if per_shape:
            reason = "no admitted shape at this site: " + "; ".join(
                e.reason for e in per_shape)
            fix = "see the per-shape fixes above"
        elif not site.shapes:
            # The site is KNOWN (it has a throughput entry, so it is in
            # configs/site_throughput.env) but nothing was ever probed for it, so there
            # is no per-shape exclusion to point at and the old message pointed the
            # reader at fixes that do not exist -- after a reason that ended in a colon
            # and nothing at all.
            reason = (f"{site.site} has a throughput entry but no probed shape: nothing "
                      f"was read from the scanner for it, so there is no queue wait to "
                      f"price and no shape to submit")
            fix = (f"add configs/slurm_probe/{site.site}.json with the shapes you want "
                   f"probed and run the collector on the {site.site} login node, or "
                   f"drop PWW_TPUT_{site.site.upper()}/PWW_BATCH_{site.site.upper()} "
                   f"from the throughput registry if this site is not part of the run")
        else:
            reason = (f"no admitted shape at this site: {len(site.shapes)} shape(s) were "
                      f"read but none survived admission")
            fix = "see the exclusions above"
        exclusions.append(Exclusion("site_unusable", site.site, reason, fix))
    return candidates, exclusions


# --------------------------------------------------------------------------
# phase 1 -- option generation
# --------------------------------------------------------------------------


def _links_to_cover(candidate: Candidate, config: PlanConfig, begin_s: float) -> int:
    """How many links a chained lane needs to reach the horizon.

    Link COUNT is derived rather than swept, because link LENGTH is already swept
    through the probed shape list and the duty-cycle argument makes "stay present to
    the horizon" dominant once the length is fixed. Sweeping both would multiply the
    option space by max_links for no decision the length sweep does not already make.
    """
    wait_s = candidate.wait.eff_at(config.discount_strength, config.wait_quantile)
    # inputs.py refuses a non-finite wait cell at the door, so reaching here means a
    # caller built the WaitEstimate itself. Named rather than left to math.ceil, whose
    # "cannot convert float NaN to integer" names neither the site nor the cell and
    # left the CLI at exit 1 with nothing on stdout. An INFINITE wait is not an error
    # -- it says the job never starts inside the horizon, which the span test below
    # already answers with one link.
    if math.isnan(wait_s):
        raise ValueError(
            f"{candidate.site}/{candidate.shape.name}: the queue wait is "
            f"{wait_s!r}, which is not a duration, so the number of links to the "
            f"horizon cannot be derived. A wait is a MEASUREMENT -- refuse the probe "
            f"row rather than planning against it.")
    if not candidate.walltime_s > 0:
        raise ValueError(
            f"{candidate.site}/{candidate.shape.name}: walltime is "
            f"{candidate.walltime_s!r}; a shape with no walltime cannot be chained.")
    # A wait is never negative, so the reachable span never exceeds the horizon. The
    # clamp is what keeps a nonsense -inf out of math.ceil, which raises OverflowError
    # there rather than being caught by the min() below.
    span = min(config.horizon_s - begin_s - wait_s, config.horizon_s)
    if span <= 0:
        return 1
    return max(1, min(config.max_links_per_lane, math.ceil(span / candidate.walltime_s)))


def _require_simulable_chain(
    chain: str, candidates: Sequence[Candidate], config: PlanConfig, warm: bool,
) -> None:
    """Refuse a chain policy this module prices as a chain but the simulator cannot expand.

    Two readings of one word, and they drifted: _links_to_cover chains anything that is
    not "none", while timeline.expand_links falls through to `break` for anything that
    is not "self" or "singleton". A typo'd --chain was therefore PRICED as one job per
    lane and EMITTED as a full N-link chain -- 14 jobs submitted where 2 were costed,
    and the headline merge counts off by 7x. argparse rejecting unknown names is the
    door, not the cause: anything that builds a PlanConfig directly walks past it.

    Asked of the simulator rather than checked against a second copy of the policy
    names, so a policy added to expand_links cannot go missing here. The predicate
    itself lives in timeline.py, beside the branches that answer it.
    """
    if chain == "none" or not candidates:
        return
    if can_chain(chain, config=config, candidate=candidates[0], warm=warm):
        return
    raise ValueError(
        f"chain policy {chain!r} would be priced as a chain here but the simulator "
        f"expands only its first link, so the plan would cost one job per lane and "
        f"submit up to {config.max_links_per_lane}. Teach timeline.expand_links this "
        f"policy, or pass one it already handles.")


def begin_grid(
    site: str,
    candidates_by_site: dict[str, list[Candidate]],
    config: PlanConfig,
) -> list[float]:
    """{now} u {when each OTHER site is predicted to become productive} u user grid.

    The interesting begins are exactly the moments a partner shows up: that is the
    whole staggered-start question, and a uniform time grid would either miss those
    instants or cost a hundred times more to cover them.
    """
    grid = {0.0}
    for other, cands in candidates_by_site.items():
        if other == site:
            continue
        for cand in cands:
            arrival = cand.wait.eff_at(config.discount_strength, config.wait_quantile)
            grid.add(round(arrival + cand.startup_s, 3))
    grid.update(config.begin_grid_s)
    return sorted(g for g in grid if 0.0 <= g < config.horizon_s)


def _presence(option: Option, config: PlanConfig, warm: bool) -> tuple[tuple[float, float], ...]:
    """The option's productive spans, clipped to the horizon. The dominance test needs
    the actual profile, not a summary of it -- see _prune_dominated."""
    links = expand_links(option, config=config, warm_checkpoint=warm)
    spans = [(round(min(l.productive_s, config.horizon_s), 3),
              round(min(l.end_s, config.horizon_s), 3)) for l in links]
    return tuple(sorted((a, b) for a, b in spans if b > a))


def generate_options(
    site: str,
    candidates: Sequence[Candidate],
    candidates_by_site: dict[str, list[Candidate]],
    config: PlanConfig,
    *,
    warm: bool = False,
    prune: bool = True,
) -> tuple[list[Option], list[Exclusion]]:
    """(shape x lanes x link policy x begin), minus the dominated ones."""
    options: list[Option] = []
    exclusions: list[Exclusion] = []
    grid = begin_grid(site, candidates_by_site, config)
    for chain in config.chain_policies:
        _require_simulable_chain(chain, candidates, config, warm)

    for candidate in candidates:
        for begin_s in grid:
            for lanes in range(1, max(1, config.lanes_max) + 1):
                for chain in config.chain_policies:
                    links = 1 if chain == "none" else _links_to_cover(candidate, config, begin_s)
                    if chain != "none" and links == 1:
                        continue  # identical to the single-job option; do not duplicate
                    options.append(Option(
                        site=site, candidate=candidate, lanes=lanes,
                        links_per_lane=links, chain=chain, begin_s=begin_s))

    kept = _prune_dominated(options, config, warm) if prune else options
    if prune and len(kept) < len(options):
        exclusions.append(Exclusion(
            "duplicate_option", site,
            f"{len(options) - len(kept)} of {len(options)} options dropped as exact "
            f"duplicates (identical productive spans, more jobs to achieve them)",
            "pass prune=False to check that the optimum is unchanged"))
    return kept, exclusions


def _prune_dominated(options: Sequence[Option], config: PlanConfig, warm: bool) -> list[Option]:
    """Drop options that are the SAME PLAN written twice, and nothing else.

    The tempting prune -- "y arrives no later, ends no earlier, so x is dominated" --
    is unsound here, and unsound in exactly the direction this planner exists to
    catch. Under a DARL cap, arriving EARLIER is not weakly better: the solo hours it
    buys spend corpus that the federated phase then does not have, and a full-node
    headstart can exhaust the epoch before its partner ever arrives. Pruning on
    presence therefore deletes the delayed-start option that is often the optimum, and
    the search silently returns the trap instead.

    So the only thing removed is a duplicate: two options whose productive spans are
    identical, keeping the one that submits fewest jobs (fewer submissions cost less
    fairshare, and Snellius' priority is ~98% fairshare with a one-day half-life).
    That is provably optimum-preserving, and it is why the exact enumeration has to
    stay affordable rather than be pruned into affordability.
    """
    best: dict[tuple, tuple[Option, int]] = {}
    for option in options:
        spans = _presence(option, config, warm)
        jobs = option.lanes * option.links_per_lane
        key = (option.site, option.candidate.gpus, option.candidate.geometry.batch_seq,
               option.lanes, spans)
        incumbent = best.get(key)
        if incumbent is None or jobs < incumbent[1]:
            best[key] = (option, jobs)
    kept = [o for o, _ in best.values()]
    kept.sort(key=lambda o: (o.site, o.candidate.shape.name, o.begin_s, o.lanes,
                             o.links_per_lane, o.chain))
    return kept


# --------------------------------------------------------------------------
# objective
# --------------------------------------------------------------------------


def score(timeline: Timeline, config: PlanConfig) -> Score:
    """U = N_fed + alpha*N_solo + beta*(Tok/1e9).

    Solo progress is real work -- this campaign's own finding is that a centralized
    run beats the federated one at matched tokens -- but a run whose PURPOSE is a
    federated measurement needs federated rounds. alpha is the exchange rate and it is
    never buried: the plan reports alpha*, the value at which the recommendation
    changes, on every run.
    """
    utility = (timeline.federated_merges
               + config.alpha * timeline.solo_merges
               + config.beta * timeline.tokens / 1e9)
    # Weighted by device count, because it is divided by gpu_s (GPU-seconds) to give the
    # federation's barrier idle fraction. Timeline.compute_s is a sum of per-site
    # WALL-clock compute, so dividing that by GPU-seconds would report a 4-GPU site as
    # idle whenever it is merely one device wide. Per site, SiteLedger.idle_fraction is
    # the same quantity against wall-clock presence.
    # The weight is the DEVICE COUNT, read off the ledger, not gpu_s/live_s: those two
    # measure different intervals (billed allocation vs presence while the run lasted),
    # so their ratio is a device count only by coincidence.
    compute_gpu_s = sum(l.compute_s * l.gpus for l in timeline.ledgers)
    return Score(
        utility=utility,
        federated_merges=timeline.federated_merges,
        solo_merges=timeline.solo_merges,
        tokens=timeline.tokens,
        blocks=timeline.blocks_used,
        gpu_s=timeline.gpu_s,
        compute_s=compute_gpu_s,
        attempts=timeline.attempts_used,
        alpha=config.alpha,
        beta=config.beta,
    )


def detect_traps(
    selection: Sequence[Option], timeline: Timeline, config: PlanConfig
) -> tuple[Trap, ...]:
    """A plan that federates zero times is degenerate, not merely bad.

    Both known ways of getting there produce the same outcome and must be reported
    separately, because the fixes are opposite: a site that dies before its partner
    arrives wants a LONGER walltime or a later --begin, while a headstart that ate the
    corpus wants a SHORTER one.
    """
    if len({o.site for o in selection}) < 2 or timeline.federated_merges > 0:
        return ()
    productive = {}
    for link in timeline.links:
        productive.setdefault(link.site, []).append(link.productive_s)
        productive[link.site].append(link.productive_s)
    ends = {}
    for link in timeline.links:
        ends[link.site] = max(ends.get(link.site, 0.0), link.end_s)
    first = {s: min(v) for s, v in productive.items()}
    early = min(first, key=lambda s: first[s])
    late = max(first, key=lambda s: first[s])
    if (timeline.darl_exhausted_s is not None
            and timeline.darl_exhausted_s <= first[late] + _EPS):
        return (Trap(
            "trap_corpus_exhausted", f"{early}->{late}",
            f"{early} exhausted the corpus at {timeline.darl_exhausted_s / 3600:.1f} h, "
            f"before {late} became productive at {first[late] / 3600:.1f} h. With "
            f"max_epochs = 1 the run then ends regardless of walltime left: zero "
            f"federated merges, outcome-identical to the staggered-start trap. "
            f"Shorten the headstart or reduce its accumulation."),)
    return (Trap(
        "trap_no_overlap", f"{early}->{late}",
        f"{early}'s last job ends at {ends[early] / 3600:.1f} h and {late} is not "
        f"productive until {first[late] / 3600:.1f} h: the sites never overlap, so "
        f"every round is solo. This is what 'start ASAP, ask for the wait you were "
        f"quoted' produces. Ask {early} for a longer walltime, or delay it with --begin."),)


# --------------------------------------------------------------------------
# phase 2 -- search
# --------------------------------------------------------------------------


class _Eval(NamedTuple):
    """One simulated plan and everything the ranking rules need about it.

    Carried together so that every consumer -- the search, the alpha envelope, the
    sensitivity re-ranks -- applies the SAME demotions. Re-ranking cached plans while
    forgetting that some of them federate zero times, or are priced off an
    extrapolated overhead cell, is how a planner recommends a trap.
    """

    score: Score
    timeline: Timeline
    balance: bool
    traps: tuple[Trap, ...]
    rankable: bool
    selection: tuple[Option, ...]


class _Evaluator:
    """Simulates a selection and caches the result.

    `balance` is resolved here rather than by policy: it multiplies tokens per round
    and DARL blocks per round by the same factor, so under a data cap it costs
    federated merges for zero extra tokens, and under a walltime cap it is free extra
    tokens. Which one binds is not known until the plan is simulated, so 'auto'
    simulates both and keeps the better -- and the report says which budget bound.
    """

    def __init__(self, config, calibration, darl, warm, wait_quantile=None, schedule_factory=None):
        self.config = config
        self.calibration = calibration
        self.darl = darl
        self.warm = warm
        self.wait_quantile = wait_quantile or config.wait_quantile
        self.schedule_factory = schedule_factory or (
            lambda: make_schedule(config.h_model, inner_steps=config.inner_steps))
        self.cache: dict[tuple, _Eval] = {}
        self.evaluated: list[_Eval] = []

    def key(self, selection: Sequence[Option]) -> tuple:
        return tuple(sorted(
            (o.site, o.candidate.shape.name, o.lanes, o.links_per_lane, o.chain, o.begin_s)
            for o in selection))

    def _once(self, selection, balance: bool) -> tuple[Score, Timeline]:
        timeline = simulate(
            selection, config=self.config, calibration=self.calibration, darl=self.darl,
            balance=balance, schedule=self.schedule_factory(),
            wait_quantile=self.wait_quantile, warm=self.warm)
        return score(timeline, self.config), timeline

    def __call__(self, selection: Sequence[Option]) -> _Eval:
        selection = tuple(selection)
        key = self.key(selection)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if not selection:
            empty = simulate([], config=self.config, calibration=self.calibration,
                             darl=self.darl, balance=False,
                             schedule=self.schedule_factory(), warm=self.warm)
            score_, timeline, balance = score(empty, self.config), empty, False
        elif self.config.balance == "auto" and _balance_matters(selection):
            off = self._once(selection, False)
            on = self._once(selection, True)
            balance = _auto_balance(off[1], on[1])
            score_, timeline = on if balance else off
        else:
            balance = self.config.balance == "on"
            score_, timeline = self._once(selection, balance)
        out = _Eval(
            score=score_, timeline=timeline, balance=balance,
            traps=detect_traps(selection, timeline, self.config),
            rankable=(self.config.assume_overhead
                      or not self.config.require_measured_overhead
                      or timeline.quality != EXTRAPOLATED),
            selection=selection,
        )
        self.cache[key] = out
        self.evaluated.append(out)
        return out


def _balance_matters(selection: Sequence[Option]) -> bool:
    """Accumulation is a no-op unless two members have different step times."""
    steps = {o.candidate.geometry.step_s for o in selection}
    lanes = sum(o.lanes for o in selection)
    return lanes >= 2 and len(steps) > 1


def _auto_balance(off: Timeline, on: Timeline) -> bool:
    """WHICH BUDGET BINDS decides this, not the objective.

    Balancing multiplies tokens/round and DARL blocks/round by the same factor while
    the barrier moves only by the accumulation rounding, so:

      * DATA-bound (the corpus runs out before the horizon): it buys nothing and
        costs merges -- the same tokens are spent over ~2.3x fewer rounds. OFF.
      * WALLTIME-bound (corpus left at the end): the extra sequences per round are
        free, because the round was going to wait for the slow site anyway. ON.

    Deciding it by U instead makes `auto` a fixed OFF policy and always has: U counts
    merges, balancing can only ever reduce them, and at the default beta = 0 the
    tokens it buys are worth exactly zero. That is a defensible objective and a
    dishonest knob -- the flag says it chooses from which budget binds, so it does.
    Turning it on must not CREATE a data cap, hence the second clause.
    """
    if off.darl_exhausted_s is not None:
        return False
    if on.darl_exhausted_s is not None:
        # Balancing burned the corpus that the unbalanced plan had to spare: it is
        # not free after all, and it has just cost the run its tail.
        return False
    return on.tokens > off.tokens


def _rank_key(ev: "_Eval", alpha: float | None = None, beta: float | None = None) -> tuple:
    # Trapped plans sort below everything else regardless of utility. A plan known to
    # federate zero times must never be returned while an alternative exists, and the
    # unranked-but-priced rule for extrapolated overheads works the same way. alpha and
    # beta are overridable so the sensitivity pass re-ranks the SAME cached plans under
    # the SAME demotions rather than inventing a second, laxer ordering.
    s = ev.score
    if alpha is None and beta is None:
        utility = s.utility
    else:
        a = s.alpha if alpha is None else alpha
        b = s.beta if beta is None else beta
        utility = s.federated_merges + a * s.solo_merges + b * s.tokens / 1e9
    # Ties are broken toward the plan a human would rather run: fewer jobs, submitted
    # sooner. Without this the search returns whichever equal-utility option the
    # enumeration happened to reach first, which reads as noise in a diff.
    jobs = sum(o.lanes * o.links_per_lane for o in ev.selection)
    begins = sum(o.begin_s for o in ev.selection)
    return (0 if ev.traps else 1, 1 if ev.rankable else 0, utility, -jobs, -begins)


def solve(
    options_by_site: dict[str, list[Option]],
    *,
    config: PlanConfig,
    calibration: Calibration,
    darl: DarlState,
    warm: dict[str, bool] | None = None,
    method: str = "auto",
    wait_quantile: str | None = None,
) -> tuple[tuple[Option, ...], Timeline, Score, _Evaluator, SearchReport]:
    ev = _Evaluator(config, calibration, darl, warm or {}, wait_quantile)
    sites = sorted(options_by_site)
    space = 1
    for site in sites:
        space *= len(options_by_site[site]) + 1

    use_exact = method == "exact" or (
        method == "auto" and space <= config.max_exact_plans)
    best_exact = None
    if use_exact:
        best_exact = _exact(options_by_site, sites, ev, config)
    best_greedy = _greedy(options_by_site, sites, ev, config)

    if best_exact is not None:
        best = best_exact
        gap = None
        if best_greedy is not None and best_exact[2].utility > 0:
            gap = (best_exact[2].utility - best_greedy[2].utility) / abs(best_exact[2].utility)
        report = SearchReport(
            method="exact+greedy" if best_greedy else "exact",
            plans_evaluated=len(ev.cache),
            optimality_gap=gap,
            exact_proved=True,
            note=f"enumerated {space} selections over {len(sites)} sites",
        )
    else:
        best = best_greedy
        report = SearchReport(
            method="greedy",
            plans_evaluated=len(ev.cache),
            optimality_gap=None,
            exact_proved=False,
            note=(f"{space} selections exceeds max_exact_plans={config.max_exact_plans}; "
                  f"marginal-value greedy with incumbent re-optimisation and 2-exchange. "
                  f"The optimality gap is unknown, not assumed."),
        )
    if best is None:
        empty = ev(())
        return (), empty.timeline, empty.score, ev, report
    return (*best, ev, report)


def _exact(options_by_site, sites, ev, config):
    best = None
    best_key = None
    pools = [[None] + list(options_by_site[s]) for s in sites]
    for combo in itertools.product(*pools):
        result = ev(tuple(o for o in combo if o is not None))
        key = _rank_key(result)
        if best_key is None or key > best_key:
            best_key, best = key, (result.selection, result.timeline, result.score)
    return best


def _greedy(options_by_site, sites, ev, config):
    """Marginal-value greedy. Its REJECTION RECORD is the answer to 'is this site
    worth submitting at all', which is why it runs even when the exact search does.

    Incumbent re-optimisation happens once per accepted addition rather than once per
    trial: adding a site moves the barrier and every incumbent's accumulation with it,
    so their best shapes genuinely change -- but re-optimising inside the argmax makes
    the pass quadratic in the option count for a decision the following re-optimise
    reaches anyway.
    """
    start = ev(())
    best: tuple[tuple[Option, ...], Timeline, Score] = ((), start.timeline, start.score)
    best_key = _rank_key(start)

    while True:
        used = {o.site for o in best[0]}
        contender = None
        contender_key = best_key
        for site in sites:
            if site in used:
                continue
            for option in options_by_site[site]:
                trial = best[0] + (option,)
                key = _rank_key(ev(trial))
                if key > contender_key:
                    contender_key, contender = key, trial
        if contender is None:
            break
        contender = _reoptimise(contender, options_by_site, ev, config)
        result = ev(contender)
        key = _rank_key(result)
        if key <= best_key:
            break
        best_key, best = key, (contender, result.timeline, result.score)

    # 2-exchange to a fixpoint: swap one chosen option for another at the same site.
    improved = True
    while improved:
        improved = False
        for idx, current in enumerate(best[0]):
            for option in options_by_site[current.site]:
                if option is current:
                    continue
                trial = best[0][:idx] + (option,) + best[0][idx + 1:]
                result = ev(trial)
                key = _rank_key(result)
                if key > best_key:
                    best_key, best, improved = key, (trial, result.timeline, result.score), True
    return best


def _reoptimise(selection, options_by_site, ev, config):
    """Hill-climb every member's shape once, holding the membership fixed."""
    current = tuple(selection)
    best_key = _rank_key(ev(current))
    for idx, member in enumerate(current):
        for option in options_by_site.get(member.site, []):
            trial = current[:idx] + (option,) + current[idx + 1:]
            key = _rank_key(ev(trial))
            if key > best_key:
                best_key, current = key, trial
    return current


# --------------------------------------------------------------------------
# alpha*, the value at which the recommendation changes
# --------------------------------------------------------------------------


def alpha_breakpoints(
    evaluated: Sequence["_Eval"], beta: float
) -> list[tuple[float, tuple[Option, ...]]]:
    """Upper envelope of U(alpha) = N_fed + alpha*N_solo + beta*Tok over the plans the
    search actually evaluated.

    Every plan is a straight line in alpha, so the winner as a function of alpha is
    the UPPER hull of (N_solo, N_fed + beta*Tok) and the breakpoints are exact -- no
    grid, no re-simulation, and it costs nothing because the search already has the
    points.

    Returns [(alpha_i, selection_i)] with alpha_0 = 0 and alpha strictly increasing:
    selection_i wins on [alpha_i, alpha_i+1). A NON-monotone list is the signature of
    a lower hull, i.e. of the sign of the concavity test being wrong, and it is what
    the previous implementation returned whenever three or more non-dominated plans
    existed.
    """
    # (x, y) -> the selection a reader would actually be handed there. Ties are broken
    # the way _rank_key breaks them -- fewer jobs, submitted sooner -- so alpha* names
    # the same plan as the sensitivity table's alpha rows rather than whichever
    # equal-scoring plan the enumeration happened to reach first.
    pts: dict[tuple[float, float], tuple[Option, ...]] = {}
    for result in evaluated:
        # Only plans that could actually be recommended shape the envelope. A trapped
        # or unrankable plan winning at some alpha is not a finding about alpha.
        if result.traps or not result.rankable:
            continue
        x = float(result.score.solo_merges)
        y = result.score.federated_merges + beta * result.score.tokens / 1e9
        incumbent = pts.get((x, y))
        if incumbent is None or _tiebreak(result.selection) > _tiebreak(incumbent):
            pts[(x, y)] = result.selection
    if not pts:
        return []

    # One y per x: at equal solo count only the best federated count can ever win.
    best_y: dict[float, float] = {}
    for x, y in pts:
        if x not in best_y or y > best_y[x]:
            best_y[x] = y
    points = sorted((x, y) for x, y in best_y.items())

    # Upper hull, x ascending. keep[-1] stays only while it is strictly ABOVE the
    # chord joining its neighbours; on or below it, it is never the argmax of
    # y + alpha*x for any alpha and is dropped. The comparison is
    #     slope(p1 -> p2) <= slope(p1 -> p)   ==>   pop p2
    # written cross-multiplied because x is an integer count and floats here would
    # make the breakpoint list depend on rounding.
    hull: list[tuple[float, float]] = []
    for x, y in points:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (x - x1) <= (y - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append((x, y))

    # alpha >= 0 only. The envelope starts at the alpha = 0 winner, which is the
    # largest y (ties to the largest x, since a larger solo count then wins for every
    # alpha > 0); everything left of it is on the hull only for alpha < 0.
    peak = max(range(len(hull)), key=lambda i: (hull[i][1], hull[i][0]))
    hull = hull[peak:]

    out: list[tuple[float, tuple[Option, ...]]] = [(0.0, pts[hull[0]])]
    for (x1, y1), (x2, y2) in zip(hull, hull[1:]):
        if x2 <= x1:
            continue
        out.append((max(0.0, (y1 - y2) / (x2 - x1)), pts[(x2, y2)]))
    return out


def _tiebreak(selection: Sequence[Option]) -> tuple:
    """_rank_key's tail, so equal-scoring plans are ordered identically everywhere."""
    return (-sum(o.lanes * o.links_per_lane for o in selection),
            -sum(o.begin_s for o in selection))


def alpha_star(
    evaluated, beta: float, alpha: float, winner: tuple[Option, ...]
) -> float | None:
    """The nearest alpha at which the recommendation CHANGES. None when it never does
    over alpha >= 0, which is itself worth printing.

    The winner occupies one segment [a_i, a_i+1) of the envelope, so the answer is a
    boundary of THAT segment -- not, as before, the start of some other plan's
    segment, which for a winner sitting in the last segment returned 0.0 (the origin)
    and reported that the recommendation changes at an alpha where it does not.
    """
    breaks = alpha_breakpoints(evaluated, beta)
    if not breaks:
        return None
    idx = 0
    for i, (a, _) in enumerate(breaks):
        if a <= alpha + 1e-9:
            idx = i
    cands: list[float] = []
    if idx + 1 < len(breaks):
        cands.append(breaks[idx + 1][0])          # where the winner stops winning
    if idx > 0:
        cands.append(breaks[idx][0])              # where the winner started winning
    if _label(breaks[idx][1]) != _label(tuple(winner)) and breaks[idx][0] > 1e-9:
        # The envelope's holder of this segment is not the plan the search returned
        # (a tie broken elsewhere, or a demotion). Its own boundary is still the
        # nearest alpha at which the printed recommendation moves.
        cands.append(breaks[idx][0])
    if not cands:
        return None
    return min(cands, key=lambda a: abs(a - alpha))


# --------------------------------------------------------------------------
# cross-checks -- computed alongside the simulator, never instead of it
# --------------------------------------------------------------------------


def duty_cycle(walltime_s: float, startup_s: float, wait_s: float) -> float:
    """(T - c) / (T + w). The fraction of the submit-to-end clock that trains."""
    return (walltime_s - startup_s) / (walltime_s + wait_s)


def chain_breakeven_c(t_s: float, T_s: float, w_t_s: float, w_T_s: float) -> float:
    """c* such that chaining links of length t beats one job of length T iff c < c*.

        c* = [t*w(T) - T*w(t)] / [(T - t) + w(T) - w(t)]

    The degenerate case is the point of the whole formula: if the queue is insensitive
    to walltime, w(t) = w(T) = w, the numerator is w(t - T) < 0 and c* is negative, so
    chaining is NEVER worth it. Chaining only ever buys a queue advantage, and paying
    per-job startup for one that does not exist is pure loss.
    """
    denom = (T_s - t_s) + (w_T_s - w_t_s)
    if abs(denom) < _EPS:
        return float("-inf")
    return (t_s * w_T_s - T_s * w_t_s) / denom


def _resolved_balance(timeline) -> bool:
    """Whether this plan ACTUALLY balances, read off the simulated ledgers.

    NOT `config.balance == "on"`. Under the default --balance auto the decision is made
    per selection by _auto_balance, so re-deriving it from the config prices the
    cross-checks and the round budget for a plan the planner did not choose: an auto
    plan that resolved to ON was cross-checked at accums (1,1) and had NUM_ROUNDS sized
    against the unbalanced timeline. Same read as emit.accums_for, so the emitter, the
    cross-checks and the round budget cannot disagree about a single question.
    """
    return any(a > 1 for ledger in timeline.ledgers for a in ledger.accums)


def headstart_checks(
    early: Candidate, late: Candidate, *, config: PlanConfig, period_s: float,
    solo_seq: float, fed_seq: float, data_bound: bool,
    begin_early_s: float = 0.0, begin_late_s: float = 0.0,
) -> list[CrossCheck]:
    """The two conditions for a solo headstart to be worth taking.

    (i) is time feasibility read at p90, because "A must still be alive when B
    arrives" is a feasibility question and a median is the wrong statistic for it.
    (ii) is the marginal data value: when DARL binds, every solo round is corpus that
    a federated round will not get, so the exchange rate is the share of the round's
    sequences the solo site contributes. When walltime binds instead, the right-hand
    side is zero and any alpha > 0 makes the headstart free.
    """
    q = config.feasibility_quantile
    # The --begin offsets belong INSIDE these clauses. A plan whose whole point is to
    # delay one site until the other arrives is exactly the plan this cross-check is
    # asked about, and reading the raw queue wait without the offset reports the site
    # arriving hours before the plan actually starts it.
    w_a = begin_early_s + early.wait.eff_at(config.discount_strength, q)
    w_b = begin_late_s + late.wait.eff_at(config.discount_strength, q)
    out: list[CrossCheck] = []

    exists = w_a + early.startup_s < w_b
    out.append(CrossCheck(
        "headstart_exists",
        "yes" if exists else "no",
        f"{early.site} productive at {(w_a + early.startup_s) / 3600:.2f} h vs "
        f"{late.site} arriving at {w_b / 3600:.2f} h ({q})"))

    needed = w_b + late.startup_s + config.min_federated_rounds * period_s
    survives = w_a + early.walltime_s >= needed
    out.append(CrossCheck(
        "early_site_survives_to_federate",
        "yes" if survives else "TRAP",
        f"{early.site} ends at {(w_a + early.walltime_s) / 3600:.2f} h; "
        f"{late.site} is productive at {(w_b + late.startup_s) / 3600:.2f} h and "
        f"{config.min_federated_rounds} federated round(s) need "
        f"{needed / 3600:.2f} h ({q})"))

    if data_bound and fed_seq > 0:
        ratio = solo_seq / fed_seq
        out.append(CrossCheck(
            "headstart_worth_the_corpus",
            "yes" if config.alpha > ratio else "no",
            f"alpha {config.alpha:.2f} vs zeta_solo/zeta_fed = {solo_seq:.0f}/{fed_seq:.0f} "
            f"= {ratio:.3f}: under a data cap a solo round spends corpus a federated "
            f"round will not get"))
    else:
        out.append(CrossCheck(
            "headstart_worth_the_corpus", "free",
            "walltime binds, not DARL: there is corpus left at the horizon, so solo "
            "progress costs the federated phase nothing and any alpha > 0 takes it"))
    return out


def crosschecks(
    selection: Sequence[Option],
    timeline: Timeline,
    *,
    config: PlanConfig,
    calibration: Calibration,
    options_by_site: dict[str, list[Option]],
) -> tuple[CrossCheck, ...]:
    out: list[CrossCheck] = []

    # chain vs one long job, per site, against that site's OWN probed w(T). The
    # asymmetry falls out of this by itself: the site with the bad queue chains and
    # the site with the good queue takes one long job, and they routinely get
    # different policies in the same plan.
    for option in selection:
        site = option.site
        # SAME GEOMETRY, or this is not a w(T) curve at all. Keying on the walltime
        # alone let a 1-GPU 1 h probe be differenced against a 4-GPU 8 h one, so c*
        # came out of two shapes that differ in device count as well as in T -- and
        # the duty-cycle inequality is only about T. The reduced-geometry shape has
        # its own, much shorter, queue: mixing them makes chaining look free.
        gpus = option.candidate.gpus
        shapes = {}
        for other in options_by_site.get(site, []):
            cand = other.candidate
            if cand.gpus != gpus:
                continue
            shapes[cand.walltime_s] = cand
        if len(shapes) < 2:
            # Say so rather than dropping it. "No second walltime was probed at this
            # geometry" is a fact about the collector config, and it is the reason the
            # duty-cycle argument -- the one that decides chain vs one long job --
            # cannot be checked for this plan.
            out.append(CrossCheck(
                f"chain_or_one_long_job[{site}@{gpus}]", "not evaluable",
                f"only one walltime is probed at {site}@{gpus} devices "
                f"({option.candidate.walltime_s / 3600:g} h), so there is no w(T) curve "
                f"to difference and c* cannot be computed. Probe a second walltime at "
                f"this geometry -- the SHORT end is the half that decides chaining -- "
                f"and this check starts answering.",
                agrees=None))
            continue
        # T is the walltime the plan ACTUALLY chose, not the longest one probed: the
        # question the cross-check has to answer is whether the decision the simulator
        # made was the right one, and comparing two shapes it did not pick answers a
        # different question.
        T = option.candidate.walltime_s
        t = min(w for w in shapes if w != T)
        shapes[T] = option.candidate
        c = option.candidate.startup_s
        w_T = shapes[T].wait.eff_at(config.discount_strength, config.wait_quantile)
        w_t = shapes[t].wait.eff_at(config.discount_strength, config.wait_quantile)
        c_star = chain_breakeven_c(t, T, w_t, w_T)
        says_chain = c < c_star
        did_chain = option.links_per_lane > 1
        out.append(CrossCheck(
            f"chain_or_one_long_job[{site}@{gpus}]",
            "chain" if says_chain else "one long job",
            f"at {gpus} device(s), the geometry the plan chose: "
            f"c* = [{t / 3600:g}h*{w_T / 3600:.2f}h - {T / 3600:g}h*{w_t / 3600:.2f}h] / "
            f"[({T / 3600:g}-{t / 3600:g})h + ({w_T / 3600:.2f}-{w_t / 3600:.2f})h] = "
            f"{c_star / 60:.1f} min against a measured c of {c / 60:.1f} min; "
            f"duty cycle {duty_cycle(T, c, w_T):.3f} at {T / 3600:g} h vs "
            f"{duty_cycle(t, c, w_t):.3f} at {t / 3600:g} h",
            agrees=(says_chain == did_chain)))

    # headstart, for the earliest/latest pair actually selected
    if len({o.site for o in selection}) >= 2:
        by_first = sorted(
            selection,
            key=lambda o: o.begin_s
            + o.candidate.wait.eff_at(config.discount_strength, config.wait_quantile)
            + o.candidate.startup_s)
        early, late = by_first[0], by_first[-1]
        members = [make_member(f"{o.site}-l0", o.candidate) for o in selection]
        cost = round_cost(members, inner_steps=config.inner_steps, calibration=calibration,
                          balance=_resolved_balance(timeline),
                          balance_max=config.balance_max)
        # BOTH sides carry the accumulation multiplier, or this ratio is not the
        # exchange rate it claims to be. PWW_GRAD_ACCUM is fixed for the whole job
        # (see rounds.plan_accums), so a solo round at the early site burns
        # accum * batch_seq sequences, not batch_seq -- the same defect that had the
        # simulator pricing solo rounds at 1/5 of their real corpus burn. Leaving the
        # multiplier off here understates the solo cost by exactly the early site's
        # accumulation, and it understates it precisely when the early site is the
        # ACCUMULATING one -- i.e. the fast site, which is the site with the short
        # queue and therefore the one the headstart is about.
        solo_seq = next(a * m.batch_seq
                        for a, m in zip(cost.accums, members) if m.site == early.site)
        fed_seq = sum(a * m.batch_seq for a, m in zip(cost.accums, members))
        out += headstart_checks(
            early.candidate, late.candidate, config=config, period_s=cost.period_s,
            solo_seq=solo_seq, fed_seq=fed_seq,
            data_bound=timeline.darl_exhausted_s is not None,
            begin_early_s=early.begin_s, begin_late_s=late.begin_s)
    return tuple(out)


# --------------------------------------------------------------------------
# marginal ledger -- why each site is in or out, as a number
# --------------------------------------------------------------------------


def marginal_ledger(
    winner: Sequence[Option],
    options_by_site: dict[str, list[Option]],
    ev: _Evaluator,
    config: PlanConfig,
    calibration: Calibration,
) -> tuple[MarginalEntry, ...]:
    base = ev(tuple(winner))
    base_score = base.score
    chosen = {o.site: o for o in winner}
    entries: list[MarginalEntry] = []
    for site in sorted(options_by_site):
        if site in chosen:
            without = tuple(o for o in winner if o.site != site)
            alt = ev(without)
            delta = base_score.utility - alt.score.utility
            blocked = ""
            best_option = chosen[site]
            before, after = without, tuple(winner)
        else:
            best, best_sel, best_eval = None, None, None
            for option in options_by_site[site]:
                trial = _reoptimise(tuple(winner) + (option,), options_by_site, ev, config)
                result = ev(trial)
                if best is None or result.score.utility > best.utility:
                    best, best_sel, best_eval = result.score, trial, result
            if best is None:
                continue
            delta = best.utility - base_score.utility
            best_option = next(o for o in best_sel if o.site == site)
            before, after = tuple(winner), best_sel
            # A positive delta-U next to an exclusion reads as a bug unless the reason
            # is stated: the plan scores higher but is demoted by a ranking rule, and
            # which rule it is decides what the reader should do about it.
            blocked = ""
            if best_eval.traps:
                blocked = f" -- but that plan is a {best_eval.traps[0].code}, so it is flagged, not ranked"
            elif not best_eval.rankable:
                blocked = (" -- but that plan's round regime is not one of the measured ones, so it is "
                           "priced and NOT ranked; pass assume_overhead=True to rank it")

        # Priced at the balancing decision each plan was actually simulated with,
        # not at a fixed one: under a data cap 'auto' turns accumulation OFF, and
        # reporting the balanced tokens/round beside an unbalanced plan's delta-U
        # describes two different plans in one line.
        rate_before, tok_before = _round_shape(before, config, calibration, ev(before).balance)
        rate_after, tok_after = _round_shape(after, config, calibration, ev(after).balance)
        flip = _flip_alpha(base_score, ev(after if site not in chosen else before).score, config)
        verb = "kept" if site in chosen else "EXCLUDED"
        detail = (
            f"{site} {verb} at alpha={config.alpha:.2f}: round rate "
            f"{rate_before:.1f}/h -> {rate_after:.1f}/h, tokens/round "
            f"{tok_before / 1e6:.1f}M -> {tok_after / 1e6:.1f}M, delta-U = {delta:+.1f}{blocked}")
        if flip is not None:
            detail += f". The verdict flips at alpha = {flip:.2f}"
        entries.append(MarginalEntry(
            site=site, included=site in chosen, delta_utility=delta,
            round_rate_before_h=rate_before, round_rate_after_h=rate_after,
            tokens_per_round_before=int(tok_before), tokens_per_round_after=int(tok_after),
            flip_alpha=flip, best_option=best_option.describe(), detail=detail))
    return tuple(entries)


def _round_shape(selection, config, calibration, balance: bool) -> tuple[float, float]:
    """Round rate per hour and tokens per round for a membership, at accum-resolved
    balance. Reported alongside delta-U so a reader can see WHERE the utility went."""
    if not selection:
        return 0.0, 0.0
    members = [make_member(f"{o.site}-l{k}", o.candidate)
               for o in selection for k in range(o.lanes)]
    cost = round_cost(members, inner_steps=config.inner_steps, calibration=calibration,
                      balance=balance, balance_max=config.balance_max)
    return 3600.0 / cost.period_s, float(cost.tokens)


def _flip_alpha(a: Score, b: Score, config: PlanConfig) -> float | None:
    dn = a.solo_merges - b.solo_merges
    if dn == 0:
        return None
    flip = -((a.federated_merges - b.federated_merges)
             + config.beta * (a.tokens - b.tokens) / 1e9) / dn
    if flip < 0:
        return None
    return 0.0 if abs(flip) < 1e-9 else flip


# --------------------------------------------------------------------------
# phase 4 -- sensitivity, always all of it
# --------------------------------------------------------------------------


def sensitivity(
    options_by_site: dict[str, list[Option]],
    winner: Sequence[Option],
    ev: _Evaluator,
    *,
    config: PlanConfig,
    calibration: Calibration,
    darl: DarlState,
    warm: dict[str, bool],
) -> tuple[SensitivityRow, ...]:
    """Six passes. A recommendation that survives none of them is not a recommendation.

    alpha and beta re-rank the plans the base search already simulated, so they cost
    nothing. The other four change the timeline and need a re-solve; they run greedy,
    which is stated in the row rather than hidden.
    """
    rows: list[SensitivityRow] = []
    base = _label(winner)

    # alpha and beta only re-weight a plan's counts, so the plans the base search
    # already simulated can be re-ranked in place -- exact for this option set, and free.
    for alpha in (0.0, 0.25, 0.5, 1.0):
        best = max(ev.evaluated, key=lambda e: _rank_key(e, alpha=alpha))
        util = _rank_key(best, alpha=alpha)[2]
        rows.append(SensitivityRow("alpha", f"{alpha:g}", _label(best.selection), util,
                                   best.score.federated_merges, best.score.solo_merges,
                                   _label(best.selection) != base))
    for beta in (0.0, 1.0):
        best = max(ev.evaluated, key=lambda e: _rank_key(e, beta=beta))
        util = _rank_key(best, beta=beta)[2]
        rows.append(SensitivityRow("beta", f"{beta:g}", _label(best.selection), util,
                                   best.score.federated_merges, best.score.solo_merges,
                                   _label(best.selection) != base))

    def resolve(cfg: PlanConfig, opts=None, quantile=None) -> tuple[str, Score]:
        sel, _, sc, _, _ = solve(opts or options_by_site, config=cfg, calibration=calibration,
                                 darl=darl, warm=warm, method="greedy",
                                 wait_quantile=quantile)
        return _label(sel), sc

    for strength in (0.0, 0.5, 1.0):
        label, sc = resolve(replace(config, discount_strength=strength))
        rows.append(SensitivityRow("discount_strength", f"{strength:g}", label, sc.utility,
                                   sc.federated_merges, sc.solo_merges, label != base))
    for factor, name in ((0.5, "c/2"), (1.0, "c"), (2.0, "2c")):
        scaled = {s: [_scale_startup(o, factor) for o in opts]
                  for s, opts in options_by_site.items()}
        label, sc = resolve(config, opts=scaled)
        rows.append(SensitivityRow("startup_cost", name, label, sc.utility,
                                   sc.federated_merges, sc.solo_merges, label != base))
    for h_model in ("fixed", "qsr", "replay"):
        label, sc = resolve(replace(config, h_model=h_model))
        rows.append(SensitivityRow("h_model", h_model, label, sc.utility,
                                   sc.federated_merges, sc.solo_merges, label != base))
    for quantile in ("p50", "p90"):
        label, sc = resolve(replace(config, wait_quantile=quantile), quantile=quantile)
        rows.append(SensitivityRow("wait_quantile", quantile, label, sc.utility,
                                   sc.federated_merges, sc.solo_merges, label != base))
    return tuple(rows)


def _scale_startup(option: Option, factor: float) -> Option:
    cand = replace(option.candidate, startup_s=option.candidate.startup_s * factor)
    return replace(option, candidate=cand)


def _label(selection: Sequence[Option]) -> str:
    if not selection:
        return "(nothing)"
    return " + ".join(sorted(o.describe() for o in selection))


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------


def make_plan(inputs: PlannerInputs, config: PlanConfig | None = None) -> Plan:
    """Admit, enumerate, solve, cross-check, and report why.

    Pure: every byte of the answer comes from `inputs` and `config`, there is no RNG
    and no clock, so the same inputs give a byte-identical plan and `explain` can
    replay one from its JSON.
    """
    config = config or PlanConfig()
    calibration = inputs.calibration or DEFAULT_CALIBRATION
    exclusions = list(inputs.exclusions)
    warnings = list(inputs.warnings)

    candidates_by_site: dict[str, list[Candidate]] = {}
    for site in inputs.sites:
        cands, excl = admit(site, config=config, calibration=calibration)
        exclusions.extend(excl)
        if cands:
            candidates_by_site[site.site] = cands
    warm = {s.site: s.warm_checkpoint for s in inputs.sites}

    options_by_site: dict[str, list[Option]] = {}
    for site in inputs.sites:
        if site.site not in candidates_by_site:
            continue
        opts, excl = generate_options(
            site.site, candidates_by_site[site.site], candidates_by_site, config,
            warm=site.warm_checkpoint)
        exclusions.extend(excl)
        if opts:
            options_by_site[site.site] = opts

    limits_note = [
        f"{s.site}: MaxSubmitJobs/MaxRunningJobs {s.limits.source} "
        f"({s.limits.max_submit_jobs}/{s.limits.max_running_jobs})"
        for s in inputs.sites if s.limits.source == "assumed"
    ]
    if limits_note:
        warnings.append(
            "site submission limits are ASSUMED, not read: there is no sbatch, sacct or "
            "scontrol on the aggregator VM. " + "; ".join(limits_note))

    selection, timeline, score_, ev, report = solve(
        options_by_site, config=config, calibration=calibration, darl=inputs.darl, warm=warm)

    traps = detect_traps(selection, timeline, config)
    if timeline.quality == EXTRAPOLATED and config.require_measured_overhead:
        warnings.append(
            "the dominant round regime is not one of the measured ones, so its period is "
            "priced but NOT ranked. Pass assume_overhead=True to rank it anyway, or "
            "measure it: one round at this membership and geometry, differenced "
            "merge-complete to merge-complete with PWW_VAL_WINDOWS logged, closes it.")
    if any(o.links_per_lane > 1 and o.chain == "self" for o in selection) and config.chain_wait_overlap:
        warnings.append(
            "chained links assume the successor's queue wait runs concurrently with its "
            "--begin hold. If Slurm defers eligibility until the begin time instead, each "
            "link costs a further w(T); set chain_wait_overlap=False to reprice.")
    warnings.extend(timeline.warnings)

    recommended = _recommend_num_rounds(selection, config, calibration, inputs.darl, warm,
                                        balance=_resolved_balance(timeline))
    return Plan(
        config=config,
        selection=selection,
        timeline=timeline,
        score=score_,
        exclusions=tuple(exclusions),
        traps=traps,
        marginal=marginal_ledger(selection, options_by_site, ev, config, calibration),
        crosschecks=crosschecks(selection, timeline, config=config, calibration=calibration,
                                options_by_site=options_by_site),
        sensitivity=sensitivity(options_by_site, selection, ev, config=config,
                                calibration=calibration, darl=inputs.darl, warm=warm),
        search=report,
        alpha_star=alpha_star(ev.evaluated, config.beta, config.alpha, selection),
        warnings=tuple(warnings),
        recommended_num_rounds=recommended,
    )


def _recommend_num_rounds(selection, config, calibration, darl, warm, balance) -> int:
    """What to pass as NUM_ROUNDS to start_central_services.sh.

    Sized against an unbounded attempt budget, because a round attempt is consumed by
    every STARTED round including solo ones, and setting it too high costs nothing
    while setting it too low ends the run early. Zero live clients burns nothing --
    sample() blocks in wait_for on min_available_clients -- so queued time is free.
    """
    if not selection:
        return config.num_rounds
    generous = replace(config, num_rounds=1_000_000)
    timeline = simulate(selection, config=generous, calibration=calibration, darl=darl,
                        balance=balance,
                        schedule=make_schedule(config.h_model, inner_steps=config.inner_steps),
                        warm=warm)
    lanes = len({l.lane_id for l in timeline.links})
    return int(math.ceil(timeline.attempts_used * 1.15)) + lanes + 10
