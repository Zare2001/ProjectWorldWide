"""End to end: size a plan for a token budget across whichever sites are worth
using, show it, and only submit once a human accepts it.

    python3 -m pww.plan.schedule --tokens 20e9 --num-windows 2756597 \\
        --lanes-max 4 --horizon-h 44
    python3 -m pww.plan.schedule --tokens 20e9 --num-windows 2756597 --show-only
    python3 -m pww.plan.schedule --tokens 20e9 --num-windows 2756597 --yes
    python3 -m pww.plan.schedule --tokens 20e9 --num-windows 2756597 --sites lumi

Three stages, each already built and tested separately, wired together here:

    1. siteselect  -- per-site queue wait, printed as CONTEXT only. It used to
                      gate membership down to one site, to avoid a federated
                      merge across LUMI+Snellius reviving what TODO.md #4
                      calls a live, open bug. Reading central/strategy.py's
                      actual merge code (the MAX_WEIGHT_GROWTH guard, added
                      after the wire moved to bfloat16) shows a diverging
                      contribution -- finite or not -- is already dropped and
                      renormalised, never merged into the global model. So the
                      open question is efficiency (a site's compute wasted
                      that round), not correctness, and there is no reason to
                      pre-exclude a site before the search below prices it.
                      Pass --sites to restrict membership yourself if you want
                      that instead.
    2. pww.plan    -- membership, lane count, chain length and begin time,
                      objective pinned to `objective="tokens"` (pure token
                      maximisation across however many sites end up worth
                      using) and `assume_overhead=True` (a multi-LANE site is
                      not one of the measured overhead regimes in
                      configs/plan/federation.json; plain 1-lane LUMI+Snellius
                      is, and ranks without needing this flag at all). NOT
                      `alpha=0, beta=1` under the package's default objective:
                      that formula's federated-round count is unweighted, so
                      it dominates beta*(tokens/1e9) at any realistic scale
                      regardless of alpha/beta, and silently caps every site
                      but the busiest at 1 client instead of actually
                      maximising tokens. See PlanConfig.objective.
    3. submit.py   -- only after a human types `y`, and only the commands this
                      host can actually run; see submit.py for why this cannot
                      run everywhere from one place

THE CONFIRMATION GATE IS THE POINT OF THIS FILE. Everything above it
(siteselect + pww.plan) already runs standalone and prints its own reasoning;
this module's job is to fold that into one screen a human reads once, and to
make "nothing is submitted until you type y" a property of the code path, not
a habit of whoever is running it. `--yes` exists for a cron-driven top-up, not
for making the gate a formality: use it deliberately.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from . import adapter, budget as budget_mod, cli as plan_cli, emit as emit_mod
from . import report as report_mod, submit as submit_mod
from .search import make_plan
from .siteselect import rank_sites, render as render_siteselect


def _fmt_tokens(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    return f"{n:,.0f}"


def build_parser() -> argparse.ArgumentParser:
    # Starts from pww.plan's own parser -- every scanner/DARL/registry/dry-run
    # flag it accepts is accepted here too, verbatim, so those cannot drift
    # into a second copy. `command` (plan/show/explain/...) is inherited but
    # unused: this tool always does the plan-then-confirm flow.
    ap = plan_cli.build_parser()
    ap.prog = "pww-plan-schedule"
    ap.description = __doc__
    ap.formatter_class = argparse.RawDescriptionHelpFormatter

    budget = ap.add_argument_group("token budget")
    budget.add_argument("--tokens", type=float, default=None, metavar="N",
                        help="target token budget, e.g. 20e9. Converted to "
                             "DARL_EPOCHS (see pww.plan.budget) and pinned as "
                             "the corpus size this plan is sized against -- "
                             "overrides whatever the live coordinator (if any) "
                             "reports, same as --blocks")
    budget.add_argument("--manifest", metavar="PATH",
                        help="corpus manifest.json, to read num_windows/seq_len "
                             "for --tokens (see pww.plan.budget)")
    budget.add_argument("--num-windows", type=int, metavar="N",
                        help="corpus size directly, when there is no "
                             "manifest.json reachable from here")

    out = ap.add_argument_group("confirmation and output")
    out.add_argument("--yes", action="store_true",
                     help="skip the confirmation prompt (for a cron-driven "
                          "top-up, not for routine use -- see module docstring)")
    out.add_argument("--show-only", action="store_true",
                     help="print the plan and stop; never prompt, never submit")
    out.add_argument("--out", default=None, metavar="PLAN.JSON",
                     help="where to write the accepted plan (default: "
                          "<root>/runs/scheduler/plan-<ts>.json)")
    out.add_argument("--state", default=None, metavar="EVENTS.JSONL",
                     help="default: <root>/runs/scheduler/events.jsonl")
    return ap


def resolve_blocks(args) -> tuple[int | None, str | None]:
    """(blocks_override, note) from --tokens, or (None, None) if not given."""
    if args.tokens is None:
        return None, None
    if args.manifest:
        num_windows, seq_len = budget_mod.read_manifest(args.manifest)
    elif args.num_windows:
        num_windows, seq_len = args.num_windows, 2048
    else:
        raise SystemExit("--tokens needs --manifest or --num-windows to know the "
                         "corpus size")
    tb = budget_mod.plan_epochs(int(args.tokens), num_windows, seq_len)
    # DARL_EPOCHS is a dimension the planner's own simulator does not model --
    # DarlState carries no epoch count into `blocks_available`, only a flat
    # `unassigned`. So the corpus a multi-epoch coordinator will actually serve
    # is represented here as a SINGLE, larger block count (tokens across every
    # epoch, not one epoch's worth): tokens_at_epochs / (block_size * seq_len).
    # This is the whole reason `resolve_blocks` exists rather than leaving the
    # planner to read live DARL state, which -- before the coordinator for this
    # campaign is even started -- would see either nothing (whole corpus
    # assumed, i.e. ONE epoch) or a different arm's leftover count.
    # block_size=1024 is DARL's own coordinator flag, identical in every job
    # script and collector config in this repo (never derived, always literal),
    # so it is safe to fix here rather than wait on Calibration to load it.
    blocks = tb.tokens_at_epochs // (1024 * seq_len)
    return int(blocks), budget_mod.describe(tb)


def render_summary(plan, verdicts, budget_note: str | None, config) -> str:
    """`verdicts` is context, not a gate: a busy site is not excluded here, only
    priced -- pww.plan's own search decides whether its contribution is worth
    the barrier overhead a federated round with it would cost, which is a
    better answer than pre-excluding it before the search runs."""
    lines = [report_mod.RULE, "PROPOSED PLAN", report_mod.RULE]
    admitted = [v for v in verdicts if v.candidate is not None]
    if len(admitted) >= 2:
        w, r = admitted[0], admitted[1]
        lines.append(f"  queue wait     {w.site} {w.wait_s / 3600:.2f}h vs "
                     f"{r.site} {r.wait_s / 3600:.2f}h (fullest node, longest "
                     f"walltime probed at each; not a membership decision -- "
                     f"see 'membership' below for what the search actually chose)")
    elif admitted:
        lines.append(f"  queue wait     {admitted[0].site} {admitted[0].wait_s / 3600:.2f}h "
                     f"(only site admitted)")
    if budget_note:
        lines.append(f"  token budget   {budget_note}")
    lines.append(f"  horizon        {config.horizon_s / 3600:.1f} h")
    lines.append(f"  objective      pure token maximisation (config.objective='tokens'): "
                 f"the search is free to use one site, both, or neither, "
                 f"whichever maximises tokens in the horizon")
    if not plan.selection:
        lines.append("\n  NO SUBMISSION: nothing admissible. See the exclusions above.")
        return "\n".join(lines)
    lines.append("")
    lines.append("  membership:")
    for opt in plan.selection:
        lines.append(f"    {opt.describe()}")
    sites_used = {o.site for o in plan.selection}
    if len(sites_used) > 1:
        lines.append(f"    -- federated across {', '.join(sorted(sites_used))}: a "
                     f"diverging contribution is dropped and renormalised, not merged "
                     f"in (central/strategy.py); watch monitor.py for a site whose "
                     f"committed blocks stall while it stays 'alive'")
    tl = plan.timeline
    lines.append("")
    # Three genuinely different stop reasons, and conflating any two of them is
    # exactly the kind of thing this codebase's own philosophy warns against: a
    # run that stopped on attempts_exhausted (raise --num-rounds) reads nothing
    # like one that stopped because the corpus ran out (nothing left to fix) or
    # one that simply reached the horizon with slack (fine as-is). Checked in
    # this order because a run can be flagged on more than one axis at once and
    # DARL exhaustion is the one that ends training outright.
    if tl.darl_exhausted_s is not None:
        stop_reason = " (DARL corpus exhausted before the horizon)"
    elif tl.attempts_exhausted_s is not None:
        stop_reason = (f" (round-ATTEMPT budget exhausted at "
                       f"{tl.attempts_used} of {config.num_rounds} -- corpus was "
                       f"NOT the limit; raise --num-rounds to use more of the "
                       f"horizon)")
    else:
        stop_reason = " (reached the horizon; corpus and attempts both had slack left)"
    lines.append(f"  estimated      {_fmt_tokens(tl.tokens)} tokens, {tl.gpu_s / 3600:.1f} "
                 f"GPU-h, run ends at {tl.run_end_s / 3600:.1f}h{stop_reason}")
    lines.append(f"  rounds         recommended NUM_ROUNDS={plan.recommended_num_rounds} "
                 f"of {config.num_rounds} attempts")
    if plan.recommended_num_rounds > config.num_rounds:
        lines.append(f"  ** the recommended NUM_ROUNDS ({plan.recommended_num_rounds}) "
                     f"is ABOVE the {config.num_rounds} attempts this plan was costed "
                     f"against -- the aggregator would hit attempts_exhausted before "
                     f"the estimate above is reached. Re-run with "
                     f"--num-rounds {plan.recommended_num_rounds} (or higher) before "
                     f"submitting.")
    if not plan.rankable:
        lines.append("\n  ** plan.rankable is FALSE: the dominant round regime is "
                     "extrapolated, not one of the measured ones in "
                     "configs/plan/federation.json -- --assume-overhead is what let "
                     "it rank anyway. See PLANNER.md 'when to distrust the answer' "
                     "before trusting the numbers above.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src = plan_cli.sources_from(args)
    # objective="tokens" scores plans on Tok/1e9 alone. Setting alpha=0, beta=1
    # under the DEFAULT objective ("federated") does NOT achieve this: U = N_fed +
    # alpha*N_solo + beta*(Tok/1e9) leaves N_fed unweighted, so it dominates the
    # token term at any realistic scale (tens-to-hundreds of rounds vs.
    # single-digit-billions of tokens) regardless of alpha/beta -- confirmed by
    # comparing selections under both objectives across every scenario in
    # docs/scheduler_eval: the federated-round-count-maximising choice capped LUMI
    # at 1 client in every one, where token-maximisation always preferred more.
    config = dataclasses.replace(plan_cli.config_from(args), objective="tokens",
                                 assume_overhead=True)

    blocks_override, budget_note = resolve_blocks(args)
    if blocks_override is not None:
        src = dataclasses.replace(src, blocks=blocks_override)

    collected = adapter.collect(src)
    if not collected.inputs.sites:
        print("NO PLAN: no site could be assembled.", file=sys.stderr)
        for exc in collected.inputs.exclusions:
            print(f"  [{exc.code}] {exc.subject}: {exc.reason}", file=sys.stderr)
        return 2

    # Informational only: which site would get a lone client running soonest.
    # Does NOT restrict membership below -- pww.plan's own search already
    # prices whether combining sites is worth it (a federated round's barrier
    # overhead against a slow/busy site's contribution), which is a strictly
    # better answer than pre-excluding one before the search ever runs. See
    # `git log` on src/pww/central/strategy.py around MAX_WEIGHT_GROWTH for why
    # a federated merge across sites is safe to plan for: a diverging
    # contribution (finite or not) is dropped and renormalised, never merged
    # into the global model.
    verdicts = rank_sites(collected.inputs, config)
    print(render_siteselect(verdicts, config))
    admitted = [v for v in verdicts if v.candidate is not None]
    if not admitted:
        print("\nNO PLAN: no site was admitted.", file=sys.stderr)
        return 2

    plan = make_plan(collected.inputs, config)
    emit_cfg = plan_cli.emit_config_from(args)
    submissions = emit_mod.emit(plan, collected.inputs.calibration, emit_cfg,
                                darl=collected.inputs.darl)

    print()
    print(render_summary(plan, verdicts, budget_note, config))

    if args.show_only or not plan.selection:
        return 0 if plan.selection else 2

    print()
    if not args.yes:
        answer = input("Submit this plan? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Not submitted.")
            return 3

    problems = emit_mod.preflight(emit_cfg)
    for p in problems:
        print(f"  !! {p}")
    if problems:
        print("\nRefusing to submit: preflight found the problem(s) above.")
        return 4

    out_path = Path(args.out) if args.out else (
        Path(args.root) / "runs" / "scheduler" / f"plan-{int(plan.timeline.run_end_s)}"
                                                  f"-{winner}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = report_mod.as_json(plan, collected, submissions, collected.inputs.calibration)
    out_path.write_text(report_mod.dump_json(tree))
    print(f"\nplan written: {out_path}")

    state_path = args.state or str(Path(args.root) / "runs" / "scheduler" / "events.jsonl")
    print(f"submitting what this host can run (state: {state_path}) ...")
    submit_mod.submit_all(submissions, root=args.root, state_path=state_path)
    print(f"\nfor whatever was skipped above, run on the right host:\n"
         f"  python3 -m pww.plan.submit --plan {out_path} --state {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
