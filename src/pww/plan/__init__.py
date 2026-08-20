"""pww-plan: which sites to submit, at what shape, for how long, starting when.

THE ONE STRUCTURAL FACT. A Flower round is a barrier. Every live site runs H inner
AdamW steps and the server merges whoever delivered; the round cannot close until the
SLOWEST live site is done. So the inner phase is a MAX over the live sites and the
transport/merge/evaluate overhead is a SUM over them, and three consequences follow
that no waterfill planner can express:

  * sites must OVERLAP IN TIME to federate at all -- a site whose queue opens after
    the others' walltime has expired contributes nothing federated;
  * adding a slow site LOWERS the round rate for everyone already in, so "is this
    site worth submitting" is a real question with a computable answer;
  * gradient accumulation fills the barrier, and it costs DARL blocks linearly, so
    balancing, membership and shape are one decision rather than three.

    src/pww/plan/model.py      plain data: shapes, waits, geometries, options, ledgers
    src/pww/plan/rounds.py     what one round costs; H schedules; the calibration
    src/pww/plan/timeline.py   the simulator -- authoritative, debits DARL round by round
    src/pww/plan/search.py     admission, options, the joint solve, the cross-checks
    src/pww/plan/inputs.py     the adapter: scanner, throughput registry, DARL /status

Everything except `inputs` is pure: no clock, no network, no filesystem, no RNG. The
same inputs give a byte-identical plan.

    from pww.plan import make_plan, PlanConfig
    plan = make_plan(inputs, PlanConfig(alpha=0.25, horizon_s=48 * 3600))

Read `plan.traps` before `plan.selection`: a plan that federates zero times is
degenerate rather than merely low-scoring, and it is flagged rather than ranked.
"""

from __future__ import annotations

from .model import (
    Calibration,
    Candidate,
    CrossCheck,
    DarlState,
    Exclusion,
    Geometry,
    Interval,
    Link,
    MarginalEntry,
    Member,
    MeasuredRegime,
    Option,
    OverheadEntry,
    Plan,
    PlanConfig,
    PlannerInputs,
    Round,
    Score,
    SearchReport,
    SensitivityRow,
    Shape,
    ShapeKey,
    SiteInput,
    SiteLedger,
    SiteLimits,
    SiteOverhead,
    Timeline,
    Trap,
    WaitEstimate,
)
from .rounds import (
    DEFAULT_CALIBRATION,
    RoundCost,
    balance_accums,
    blocks_at_risk,
    make_member,
    make_schedule,
    residuals,
    round_cost,
    site_overhead_s,
)
from .search import (
    admit,
    alpha_star,
    chain_breakeven_c,
    crosschecks,
    detect_traps,
    duty_cycle,
    generate_options,
    make_plan,
    marginal_ledger,
    probe_shape_json,
    score,
    sensitivity,
    solve,
)
from .timeline import expand_links, simulate

__all__ = [
    "Calibration",
    "Candidate",
    "CrossCheck",
    "DEFAULT_CALIBRATION",
    "DarlState",
    "Exclusion",
    "Geometry",
    "Interval",
    "Link",
    "MarginalEntry",
    "Member",
    "MeasuredRegime",
    "Option",
    "OverheadEntry",
    "Plan",
    "PlanConfig",
    "PlannerInputs",
    "Round",
    "RoundCost",
    "Score",
    "SearchReport",
    "SensitivityRow",
    "Shape",
    "ShapeKey",
    "SiteInput",
    "SiteLedger",
    "SiteLimits",
    "SiteOverhead",
    "Timeline",
    "Trap",
    "WaitEstimate",
    "admit",
    "alpha_star",
    "balance_accums",
    "blocks_at_risk",
    "chain_breakeven_c",
    "crosschecks",
    "detect_traps",
    "duty_cycle",
    "expand_links",
    "generate_options",
    "make_member",
    "make_plan",
    "make_schedule",
    "marginal_ledger",
    "probe_shape_json",
    "residuals",
    "round_cost",
    "score",
    "sensitivity",
    "simulate",
    "site_overhead_s",
    "solve",
]
