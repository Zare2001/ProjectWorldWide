"""Plain data the planner moves between its stages. No behaviour beyond arithmetic
that belongs to a single value.

Everything here is a frozen dataclass of builtins, so `dataclasses.asdict` renders a
whole `Plan` as JSON without a custom encoder, and every stage can be driven from a
literal in a test with no I/O. Seconds internally; the report converts to hours (and
only fields named `*_h` are hours).

Two conventions that are load-bearing elsewhere:

  * `Shape.args` is the probe row's argument string VERBATIM. The emitter copies it
    substring-exact into the sbatch line, which is what makes it structurally
    impossible to quote a wait measured at one walltime and then submit another --
    the bug the upstream planner ships (it reads GPUs from plan.json and the walltime
    cap from a cluster-level setting, and never looks at `-t` at all). Do not
    normalise, re-order or re-render it.

  * nothing is ever silently defaulted. A value that could not be read becomes an
    `Exclusion` carrying the reason AND the command or config edit that fixes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# Calibration quality, worst-first when combined. The distinction is not pedantry:
# `--overhead-model measured` refuses to RANK a plan whose dominant regime is
# extrapolated, because the additive overhead form is only checked against the four
# regimes that exist in all_logs/, and it over-predicts the untested one by ~15%.
IDENTIFIED = "identified"
DERIVED = "derived_by_subtraction"
EXTRAPOLATED = "extrapolated"
_QUALITY_RANK = {IDENTIFIED: 0, DERIVED: 1, EXTRAPOLATED: 2}


def worst_quality(*qualities: str) -> str:
    return max(qualities, key=lambda q: _QUALITY_RANK.get(q, len(_QUALITY_RANK)))


@dataclass(frozen=True)
class Exclusion:
    """Why something is not in the plan, and what to do about it.

    `fix` is mandatory in spirit: an exclusion a reader cannot act on is a silent
    drop with extra words. For an unprobed shape it is the JSON object to paste into
    configs/slurm_probe/<site>.json; for an unmeasured geometry it is the
    calibrate_throughput.sh invocation.
    """

    code: str
    subject: str
    reason: str
    fix: str = ""


@dataclass(frozen=True)
class SiteLimits:
    """Submission caps. Not readable from the aggregator VM -- there is no sbatch,
    sacct or scontrol on it -- so these are assumptions until someone runs
    `scontrol show assoc_mgr` on a login node, and they are echoed as such."""

    max_submit_jobs: int | None = None
    max_running_jobs: int | None = None
    max_array_size: int | None = None
    source: str = "assumed"


@dataclass(frozen=True)
class ShapeKey:
    """(partition, device count, walltime) -- the only key a wait may be looked up by.

    Walltime is part of the key, not a free knob attached afterwards. A 40 h job does
    not start when a 4 h job would, and the whole point of probing discrete shapes is
    that the w(T) curve is visible; interpolating in T throws away the one thing the
    scanner measures.
    """

    cluster: str
    partition: str
    nodes: int
    gpus_per_node: int
    walltime_s: int
    account: str | None = None

    @property
    def gpus(self) -> int:
        return self.nodes * self.gpus_per_node

    def describe(self) -> str:
        return (f"{self.cluster}/{self.partition} {self.gpus} gpu "
                f"{self.walltime_s / 3600:g} h")


@dataclass(frozen=True)
class Shape:
    name: str
    key: ShapeKey
    args: str  # verbatim probe row; see the module docstring


@dataclass(frozen=True)
class WaitEstimate:
    """The queue wait as a distribution, because a single newest-probe scalar is
    false precision: `sbatch --test-only` answers for the queue as it is now and
    assumes every running job runs to its full walltime.

    p50 prices the plan; p90 answers "will A still be alive when B arrives", which is
    a feasibility question and must not be decided on a median.
    """

    p50_raw_s: float
    p90_raw_s: float
    p50_eff_s: float
    p90_eff_s: float
    samples: int
    probe_age_s: float
    ok: bool = True
    used_ratio: float | None = None
    discount_strength: float = 0.5
    probed_by_user: str | None = None
    message: str = ""

    def raw(self, quantile: str) -> float:
        return self.p90_raw_s if quantile == "p90" else self.p50_raw_s

    def eff(self, quantile: str) -> float:
        return self.p90_eff_s if quantile == "p90" else self.p50_eff_s

    def eff_at(self, strength: float, quantile: str) -> float:
        """The upstream discount, reused verbatim rather than reinvented:

            w_eff = w * (1 - strength * (1 - used_ratio))

        Recomputed here rather than read off `p50_eff_s` so the sensitivity pass over
        discount_strength needs no second trip to the scanner. It is a linear BLEND,
        not a model: it assumes the whole queue shrinks by one factor, which is wrong
        exactly when the blocking jobs are the long ones -- they are the jobs that do
        NOT finish early.
        """
        raw = self.raw(quantile)
        if not self.discounted:
            return raw
        return raw * (1.0 - strength * (1.0 - float(self.used_ratio)))

    @property
    def discounted(self) -> bool:
        """False when the discount silently no-opped -- used_ratio missing, <= 0 or
        > 1. A partition whose sacct window found no finishing jobs produces no usage
        row at all, so the plan LOOKS discounted when it is not and the only tell is
        this flag."""
        return self.used_ratio is not None and 0.0 < self.used_ratio <= 1.0


@dataclass(frozen=True)
class Geometry:
    """One measured (site, device count) throughput cell.

    Both numbers are needed and neither is derivable from the other: what has to be
    equalised at the barrier is the time of one optimiser STEP, and sites do not run
    the same batch per step. Same reasoning as the header of configs/site_throughput.env.
    """

    site: str
    gpus: int
    tput_seq_s: float  # sequences/second for the whole site at accumulation 1
    batch_seq: int  # sequences per optimiser step = ranks x local_batch_size
    source: str = ""

    @property
    def step_s(self) -> float:
        return self.batch_seq / self.tput_seq_s


@dataclass(frozen=True)
class OverheadEntry:
    value_s: float
    quality: str
    provenance: str = ""


@dataclass(frozen=True)
class SiteOverhead:
    """Per-site, per-round cost that is NOT the inner phase.

    Split because only one half scales with geometry: `xfer` is a fixed pair of WAN
    crossings of the whole model, while the evaluate barrier is a fixed cost plus
    V/tput of validation compute -- which is exactly why a 1-device plan pays more
    per round than a full node, and why a site-blind constant overhead under-prices
    the reduced-geometry plans that a data cap otherwise recommends.
    """

    xfer: OverheadEntry
    eval_fix: OverheadEntry


@dataclass(frozen=True)
class MeasuredRegime:
    """A (membership x geometry) cell that exists in all_logs/, with the period that
    was actually observed. rounds.residuals() checks the model against every one of
    these; a cell that is not here is priced but tagged `extrapolated`."""

    label: str
    members: tuple[tuple[str, int], ...]  # sorted (site, gpus)
    inner_steps: int
    accums: tuple[int, ...]
    measured_period_s: float
    quality: str = IDENTIFIED
    tolerance: float = 0.05
    note: str = ""


@dataclass(frozen=True)
class Calibration:
    merge: OverheadEntry
    sites: Mapping[str, SiteOverhead]
    regimes: tuple[MeasuredRegime, ...]
    val_windows: int = 512
    tau_stall_s: float = 378.0
    seq_len: int = 2048
    block_size: int = 1024

    def regime_quality(self, members: tuple[tuple[str, int], ...]) -> str:
        key = tuple(sorted(members))
        for regime in self.regimes:
            if tuple(sorted(regime.members)) == key:
                return regime.quality
        return EXTRAPOLATED


@dataclass(frozen=True)
class DarlState:
    """What GET /status said, plus which coordinator said it.

    The port matters and is reported: 29510/29520/29530/29540 are four different
    epochs of four different arms, and querying the wrong one returns a completely
    different remaining-corpus figure with no indication that it is the wrong answer.
    """

    num_blocks: int
    committed: int
    leased: int
    unassigned: int
    quarantined: int = 0
    epoch: int = 0
    max_epochs: int = 1
    digest: str = ""
    block_size: int = 1024
    source: str = ""
    read_at: float | None = None
    # True only when a coordinator actually ANSWERED GET /status. False for --blocks
    # and for the whole-corpus assumption, which is what `fresh_epoch` below turns on.
    observed: bool = False

    @property
    def fresh_epoch(self) -> bool:
        """True only when a coordinator ANSWERED and said the epoch is untouched.

        Deliberately False for --blocks and for the whole-corpus fallback. Both build
        `unassigned == num_blocks` because they have no committed count to subtract,
        NOT because the corpus is untouched -- --blocks N is the operator quoting the
        REMAINING corpus, and the fallback is a guess. Reading that equality as "this
        is a fresh epoch" is what makes the emitter hand out PWW_FRESH_RUN=1 /
        PWW_FRESH_DELETE=1 against a coordinator 69% through its epoch: the lease
        table resets to the full 2692, the global model and its Nesterov momentum are
        discarded, and run_train.sh rm -rf's both lanes' DCP checkpoints -- while the
        plan printed next to those commands still quotes the 822 blocks they destroy.
        Not knowing must therefore read as "do not wipe", which is this property.
        """
        return bool(self.observed and self.num_blocks
                    and self.unassigned >= self.num_blocks)

    @property
    def wraps(self) -> bool:
        """With max_epochs = 1 `acquire` returns epoch_complete forever once the
        corpus is covered: exhaustion ENDS the run regardless of walltime left."""
        return self.epoch + 1 < self.max_epochs


@dataclass(frozen=True)
class SiteInput:
    """Everything read from the outside world for one site, already parsed."""

    site: str
    shapes: tuple[Shape, ...]
    waits: Mapping[str, WaitEstimate]  # keyed by Shape.name
    geometries: Mapping[int, Geometry]  # keyed by device count
    startup_s: float
    startup_quality: str = "lower_bound"
    submitter: str | None = None
    limits: SiteLimits = field(default_factory=SiteLimits)
    warm_checkpoint: bool = False  # a lane that already has a local DCP checkpoint
    # Shapes the plan would like to consider but that nobody has probed. They are
    # never priced -- they become an Exclusion naming the JSON to add to the collector
    # config, because the one thing the planner must not do is invent a w(T).
    wanted_shapes: tuple[ShapeKey, ...] = ()


@dataclass(frozen=True)
class PlannerInputs:
    sites: tuple[SiteInput, ...]
    calibration: Calibration
    darl: DarlState
    exclusions: tuple[Exclusion, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanConfig:
    """Every knob, with the defaults the campaign actually wants.

    `alpha` is deliberately at the top and deliberately not 1.0: solo progress is
    real work but a run whose purpose is a federated measurement needs federated
    rounds, not tokens. It is never buried -- search re-solves across a grid and
    reports alpha*, the value at which the recommendation changes.
    """

    alpha: float = 0.25
    beta: float = 0.0
    horizon_s: float = 48 * 3600.0
    num_rounds: int = 400
    inner_steps: int = 100  # darl.inner_steps; the H of the full arm, QSR off
    h_model: str = "fixed"  # fixed | qsr | replay
    balance: str = "auto"  # auto | on | off
    balance_max: int = 8
    wait_quantile: str = "p50"
    feasibility_quantile: str = "p90"
    min_federated_rounds: int = 1
    reserve_blocks: int = 0
    lanes_max: int = 2
    max_links_per_lane: int = 8
    chain_policies: tuple[str, ...] = ("none", "self")
    chain_lead_s: float = 0.0
    chain_wait_overlap: bool = True
    begin_grid_s: tuple[float, ...] = ()
    max_probe_age_s: float = 6 * 3600.0
    discount_strength: float = 0.5
    require_measured_overhead: bool = True
    assume_overhead: bool = False
    max_exact_plans: int = 250_000
    seed_note: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.inner_steps < 1:
            raise ValueError(f"inner_steps must be >= 1, got {self.inner_steps}")
        if self.balance not in ("auto", "on", "off"):
            raise ValueError(f"balance must be auto|on|off, got {self.balance!r}")
        if self.h_model not in ("fixed", "qsr", "replay"):
            raise ValueError(f"h_model must be fixed|qsr|replay, got {self.h_model!r}")


@dataclass(frozen=True)
class Candidate:
    """One admitted (site, shape): priced, with a measured throughput cell and a
    measured wait. Anything that could not be resolved is an Exclusion instead, so a
    Candidate never carries a guess."""

    site: str
    shape: Shape
    wait: WaitEstimate
    geometry: Geometry
    startup_s: float
    overhead_s: float
    overhead_quality: str

    @property
    def gpus(self) -> int:
        return self.geometry.gpus

    @property
    def walltime_s(self) -> int:
        return self.shape.key.walltime_s


@dataclass(frozen=True)
class Member:
    """A live participant in a round: one LANE, not one site.

    Two lanes at one site are two Flower clients, two DARL cluster records and two
    delta-blob keys, so they enter the barrier max and the overhead sum separately.
    Fields are pre-resolved rather than looked up per round because the search
    evaluates ~10^5 plans and this is the hot loop.
    """

    lane_id: str
    site: str
    gpus: int
    step_s: float
    batch_seq: int
    overhead_s: float
    quality: str


@dataclass(frozen=True)
class Option:
    """A submission decision for one site: what shape, how many lanes, how each lane
    chains, and when to submit."""

    site: str
    candidate: Candidate
    lanes: int
    links_per_lane: int
    chain: str  # none | self | singleton
    begin_s: float

    def describe(self) -> str:
        chain = "" if self.links_per_lane == 1 else f" x{self.links_per_lane} {self.chain}"
        begin = "" if self.begin_s <= 0 else f" begin +{self.begin_s / 3600:.1f} h"
        lanes = "" if self.lanes == 1 else f" {self.lanes} lanes"
        return f"{self.site} {self.shape_label}{lanes}{chain}{begin}"

    @property
    def shape_label(self) -> str:
        return f"{self.candidate.shape.name}({self.candidate.gpus}g/{self.candidate.walltime_s / 3600:g}h)"


@dataclass(frozen=True)
class Link:
    """One job. `productive_s` is when it can first contribute, i.e. after startup."""

    lane_id: str
    site: str
    index: int
    submit_s: float
    arrival_s: float
    productive_s: float
    end_s: float
    cold: bool
    walltime_s: float


@dataclass(frozen=True)
class Round:
    index: int
    start_s: float
    period_s: float
    phase_s: float
    overhead_s: float
    fit: tuple[str, ...]
    eval_only: tuple[str, ...]
    accums: tuple[int, ...]
    inner_steps: int
    tokens: int
    blocks: float
    federated: bool
    quality: str
    stall_s: float = 0.0


@dataclass(frozen=True)
class Interval:
    """A stretch of constant membership. The report's timeline table is one row per
    interval; `stop_cause` is why round-stepping left it."""

    t0_s: float
    t1_s: float
    members: tuple[str, ...]
    rounds: int
    federated_rounds: int
    inner_steps: int
    period_s: float
    blocks_left: float
    stop_cause: str
    quality: str


@dataclass(frozen=True)
class SiteLedger:
    site: str
    queued_s: float
    startup_s: float
    headstart_s: float
    coresident_s: float
    tail_s: float
    gap_s: float
    compute_s: float
    merges: int
    federated_merges: int
    tokens: int
    blocks: float
    gpu_s: float
    accums: tuple[int, ...]
    # Solo presence BETWEEN two co-residency spells. A fourth column rather than a
    # rounding error: headstart/co-resident/tail is a three-way split of a
    # CONTIGUOUS story, and a partner that comes and goes -- which elastic
    # membership makes normal -- leaves hours in none of the three. Dropping them
    # understates live_s, and with it the barrier-idle fraction, by up to 2.5x.
    between_s: float = 0.0
    gpus: int = 0

    @property
    def live_s(self) -> float:
        return self.headstart_s + self.between_s + self.coresident_s + self.tail_s

    @property
    def idle_fraction(self) -> float:
        """1 - compute/present. The measured pre-balancing figure on Snellius was
        0.74: 44 s of work inside a 172 s round, on hardware billed for all of it.

        `present` is wall-clock presence while the run was still going, not the
        billed allocation: gpu_s may run past the end of the corpus and that is
        reported separately rather than folded in here.
        """
        return 1.0 - self.compute_s / self.live_s if self.live_s > 0 else 0.0

    @property
    def tokens_per_gpu_hour(self) -> float:
        return self.tokens / (self.gpu_s / 3600.0) if self.gpu_s > 0 else 0.0


@dataclass(frozen=True)
class Score:
    """The objective and, beside it, the vector a reader needs to re-rank by hand
    without re-running the planner. Only `utility` is optimised."""

    utility: float
    federated_merges: int
    solo_merges: int
    tokens: int
    blocks: float
    gpu_s: float
    compute_s: float
    attempts: int
    alpha: float
    beta: float

    @property
    def tokens_per_gpu_hour(self) -> float:
        return self.tokens / (self.gpu_s / 3600.0) if self.gpu_s > 0 else 0.0

    @property
    def idle_fraction(self) -> float:
        return 1.0 - self.compute_s / self.gpu_s if self.gpu_s > 0 else 0.0


@dataclass(frozen=True)
class Timeline:
    links: tuple[Link, ...]
    intervals: tuple[Interval, ...]
    rounds: tuple[Round, ...]
    ledgers: tuple[SiteLedger, ...]
    federated_merges: int
    solo_merges: int
    tokens: int
    blocks_used: float
    blocks_available: float
    attempts_used: int
    gpu_s: float
    compute_s: float
    darl_exhausted_s: float | None
    attempts_exhausted_s: float | None
    first_federated_s: float | None
    last_federated_s: float | None
    quality: str
    warnings: tuple[str, ...] = ()
    # When the run actually stopped. Not the horizon: exhausting DARL ends the run
    # (max_epochs = 1, so acquire returns epoch_complete forever and the dataloader
    # ends iteration), and the jobs exit rather than idling on billed hardware. The
    # ledgers are clipped to this, which is why tokens/GPU-hour is meaningful.
    run_end_s: float = 0.0


@dataclass(frozen=True)
class Trap:
    """A plan that is not merely bad but degenerate: it federates zero times. Flagged
    before ranking, because the ranking would otherwise quietly return the second
    trap when the first one loses."""

    code: str
    subject: str
    reason: str


@dataclass(frozen=True)
class CrossCheck:
    """A closed-form result computed ALONGSIDE the simulator, never instead of it.
    The simulator is authoritative; a disagreement is printed as a finding because it
    means one of the two is wrong."""

    name: str
    verdict: str
    detail: str
    agrees: bool | None = None


@dataclass(frozen=True)
class MarginalEntry:
    """Why a site is in or out, as a number. This is the answer to 'is this site
    worth submitting at all'."""

    site: str
    included: bool
    delta_utility: float
    round_rate_before_h: float
    round_rate_after_h: float
    tokens_per_round_before: int
    tokens_per_round_after: int
    flip_alpha: float | None
    best_option: str
    detail: str


@dataclass(frozen=True)
class SensitivityRow:
    knob: str
    value: str
    winner: str
    utility: float
    federated_merges: int
    solo_merges: int
    changed: bool


@dataclass(frozen=True)
class SearchReport:
    method: str  # exact | greedy | exact+greedy
    plans_evaluated: int
    optimality_gap: float | None  # measured, not assumed; None when not both ran
    exact_proved: bool
    note: str = ""


@dataclass(frozen=True)
class Plan:
    config: PlanConfig
    selection: tuple[Option, ...]
    timeline: Timeline
    score: Score
    exclusions: tuple[Exclusion, ...]
    traps: tuple[Trap, ...]
    marginal: tuple[MarginalEntry, ...]
    crosschecks: tuple[CrossCheck, ...]
    sensitivity: tuple[SensitivityRow, ...]
    search: SearchReport
    alpha_star: float | None
    warnings: tuple[str, ...]
    recommended_num_rounds: int

    @property
    def rankable(self) -> bool:
        return not self.traps

    def describe(self) -> str:
        if not self.selection:
            return "empty plan (no site was worth submitting)"
        return " + ".join(o.describe() for o in self.selection)


def hours(seconds: float | None) -> float | None:
    """Seconds to hours, keeping None as None. The report renders `*_h` fields; every
    number inside the package is seconds."""
    return None if seconds is None else seconds / 3600.0
