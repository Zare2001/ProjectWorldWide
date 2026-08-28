"""Which HPC site gets a client shape running soonest, right now -- CONTEXT for
a plan, not a gate on one.

    python3 -m pww.plan.siteselect
    python3 -m pww.plan.siteselect --dry-run tests/fixtures/plan/two-site.json
    python3 -m pww.plan.siteselect --sites lumi,snellius

Answers ONE question -- "of these sites, which gets my client shape running
soonest" -- and stops there. It does not decide membership, lane count, chain
length or begin time; `pww.plan` (or `pww.plan.schedule`) does that, and its
search is free to use one site, both, or neither.

WHY THIS DOES NOT RESTRICT `pww.plan` TO ITS WINNER
------------------------------------------------------
An earlier version of this module pinned membership to whichever site was
faster to start, specifically to avoid a federated round ever combining a
LUMI lane with a Snellius lane -- TODO.md #4 documents a live bug where one
site's contribution comes back non-finite on merge. Reading the actual merge
code changed that: `central/strategy.py`'s FedAvg step (the MAX_WEIGHT_GROWTH
guard, added the day the wire format moved to bfloat16) already drops a
diverging contribution -- finite or not -- and renormalises over the survivors
rather than merging it in. So a federated LUMI+Snellius round cannot corrupt
the global model on this failure mode; the open question is why a site
diverges in the first place (most likely the DTensor gather/scatter path,
verified only at 2 gloo ranks, never at LUMI's 8-way or Snellius's 4-way
sharding), which costs that round's compute at the affected site, not
correctness. `pww.plan`'s search already prices exactly that tradeoff (a slow
or unreliable site's contribution against the barrier overhead of including
it), which is a better answer than pre-excluding it here. This module still
runs first and prints its numbers -- they are useful context for reading the
plan's membership decision -- but nothing downstream is filtered by them.

WHAT "CLIENT SHAPE" MEANS HERE
-------------------------------
Per site, this compares the FULLEST probed node size, and among those the
LONGEST probed walltime -- the shape an actual chained submission would use
(fewer links, fewer chain-boundary restarts), not whichever shape happens to
have the shortest queue. A site offering a fast 1-GPU slot while its full node
is backed up is not "the empty site" for a run that wants a full node.

Reuses `pww.plan`'s own CLI parser, sources and `admit()` rather than a second
copy of any of it: a site-selection tool whose idea of "admitted" drifts from
the planner's is worse than none, because the two would silently disagree
about the same probe row.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Sequence

from . import adapter, cli as plan_cli, report as report_mod
from .adapter import Collected
from .model import Candidate, PlanConfig, PlannerInputs
from .search import admit


@dataclass(frozen=True)
class SiteVerdict:
    """One site's answer to "how long until my client shape starts running".

    `candidate` is None when nothing was admitted at all -- see `exclusions`.
    """

    site: str
    candidate: Candidate | None
    wait_s: float | None  # eff_at(discount_strength, wait_quantile) of `candidate`
    exclusions: tuple[str, ...]


def client_candidate(candidates: Sequence[Candidate]) -> Candidate | None:
    """The shape this module compares a site by: fullest probed node, then
    longest probed walltime among ties at that size. See the module docstring
    for why -- this is the shape an actual chained submission would request,
    not whichever probe happens to answer fastest.
    """
    if not candidates:
        return None
    max_gpus = max(c.gpus for c in candidates)
    full_node = [c for c in candidates if c.gpus == max_gpus]
    return max(full_node, key=lambda c: c.shape.key.walltime_s)


def rank_sites(inputs: PlannerInputs, config: PlanConfig) -> list[SiteVerdict]:
    """One verdict per `inputs.sites`, ascending by effective wait. A site with
    no admitted candidate sorts last, never first by virtue of `None < float`
    happening to work out that way in whatever Python version runs this.
    """
    verdicts: list[SiteVerdict] = []
    for site in inputs.sites:
        cands, excl = admit(site, config=config, calibration=inputs.calibration)
        winner = client_candidate(cands)
        wait_s = (winner.wait.eff_at(config.discount_strength, config.wait_quantile)
                   if winner is not None else None)
        verdicts.append(SiteVerdict(
            site=site.site, candidate=winner, wait_s=wait_s,
            exclusions=tuple(f"[{e.code}] {e.subject}: {e.reason}" for e in excl)))
    return sorted(verdicts, key=lambda v: (v.wait_s is None,
                                            v.wait_s if v.wait_s is not None else 0.0))


def render(verdicts: Sequence[SiteVerdict], config: PlanConfig) -> str:
    lines = [report_mod.RULE,
             f"SITE SELECTION -- client shape = fullest probed node, longest probed "
             f"walltime, wait {config.wait_quantile} at discount "
             f"{config.discount_strength:g}",
             report_mod.RULE]
    lines.append(f"  {'site':<10} {'admitted':<9} {'shape':<16} {'dev':>4} "
                 f"{'walltime':>9} {'wait(eff)':>10} {'wait(raw)':>10} "
                 f"{'age':>6}  probed_by")
    for v in verdicts:
        if v.candidate is None:
            lines.append(f"  {v.site:<10} {'no':<9} -- nothing admitted")
            for reason in v.exclusions:
                for i, line in enumerate(report_mod._wrap(reason, 70)):
                    lines.append(("      " if i else "      REFUSE ") + line)
            continue
        c = v.candidate
        raw = c.wait.raw(config.wait_quantile)
        lines.append(
            f"  {v.site:<10} {'yes':<9} {c.shape.name:<16} {c.gpus:>4} "
            f"{c.shape.key.walltime_s / 3600:>8.1f}h {v.wait_s / 3600:>9.2f}h "
            f"{raw / 3600:>9.2f}h {c.wait.probe_age_s / 60:>5.0f}m  "
            f"{c.wait.probed_by_user or '-'}")
    lines.append("")

    admitted = [v for v in verdicts if v.candidate is not None]
    if not admitted:
        lines.append("  NO WINNER: nothing was admitted at any site. See the "
                      "REFUSE lines above for what to fix.")
        return "\n".join(lines)

    winner = admitted[0]
    if len(admitted) > 1:
        runner_up = admitted[1]
        gap_h = (runner_up.wait_s - winner.wait_s) / 3600
        lines.append(f"  WINNER: {winner.site}  ({winner.wait_s / 3600:.2f}h vs "
                     f"{runner_up.site}'s {runner_up.wait_s / 3600:.2f}h -- "
                     f"{gap_h:.2f}h sooner to start)")
    else:
        lines.append(f"  WINNER: {winner.site}  (only site admitted)")
    return "\n".join(lines)


def next_command(args, winner: str) -> str:
    """The real next step (pww.plan.schedule, which prices every site itself
    and is not restricted to `winner`), plus the single-site alternative for
    when you deliberately want one -- e.g. `--sites {winner}` to rule out
    Snellius's queue entirely rather than let the search decide it is not
    worth the wait.
    """
    common = (f"--registry {args.registry} --planner-config {args.planner_config} "
             f"--probe-config-dir {args.probe_config_dir} --config {args.config} "
             f"--root {args.root}"
             + (f" --dry-run {args.dry_run}" if args.dry_run else ""))
    return (
        f"python3 -m pww.plan.schedule --tokens <N> --num-windows <N> {common}\n"
        f"    (to force this site alone instead: add --sites {winner})")


def main(argv: list[str] | None = None) -> int:
    args = plan_cli.build_parser().parse_args(argv)
    src = plan_cli.sources_from(args)
    config = plan_cli.config_from(args)

    collected: Collected = adapter.collect(src)
    print(f"  ... read {len(collected.inputs.sites)} site(s)", file=sys.stderr)
    for note in collected.notes:
        for i, line in enumerate(report_mod._wrap(note, 74)):
            print(("  ** " if i == 0 else "     ") + line, file=sys.stderr)

    if not collected.inputs.sites:
        print("NO DECISION: no site could be assembled at all.", file=sys.stderr)
        return 2

    verdicts = rank_sites(collected.inputs, config)
    print(render(verdicts, config))

    admitted = [v for v in verdicts if v.candidate is not None]
    if not admitted:
        return 2

    winner = admitted[0].site
    print()
    print("  next:")
    print(f"    {next_command(args, winner)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
