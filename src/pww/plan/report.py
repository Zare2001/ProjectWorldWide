"""Render a Plan for a terminal, and the identical tree for --json.

Eight sections, in the order a reader needs them: what was read, what was refused,
WHEN each site is alive, what each site got for its hours, the verdict, how fragile
the verdict is, why each site is in or out, and the commands.

Two rules the layout exists to serve.

FIRST, a table hides co-residency. The one quantity this planner is about -- how many
hours two sites are simultaneously alive -- is a relation between rows, and no column
shows it. Hence the ASCII presence bar, with a separate row for the federated
intersection, and hence three separate hour columns per site (headstart, co-resident,
tail) rather than one "walltime".

SECOND, every headline number prints its own arithmetic. A period of 353 s is not
checkable; `100*1.675 + 17 + 63.4 + 103.4 = 351.3 s` is. The reader is a scientist
who will disagree with at least one constant, and the report's job is to let them
recompute the consequence rather than re-run the planner.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Sequence

from .adapter import Collected
from .emit import Submission, balance_crosscheck
from .model import Calibration, Member, Plan, hours
from .rounds import make_member, plan_accums, residuals, round_cost

BAR_WIDTH = 66
RULE = "=" * 78
THIN = "-" * 78


# --------------------------------------------------------------------------
# formatting atoms
# --------------------------------------------------------------------------


def _h(seconds: float | None, width: int = 0) -> str:
    if seconds is None:
        return "-".rjust(width)
    return f"{seconds / 3600:.1f}".rjust(width)


def _tok(n: float) -> str:
    for scale, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= scale:
            return f"{n / scale:.2f} {suffix}"
    return f"{n:.0f}"


def _section(n: int, title: str) -> str:
    return f"\n{RULE}\n{n}. {title}\n{RULE}"


# --------------------------------------------------------------------------
# 3. the timeline -- the thing a table cannot show
# --------------------------------------------------------------------------


def presence_bars(plan: Plan, width: int = BAR_WIDTH) -> list[str]:
    """One row per site plus a federated row, over a common time axis.

    Four states, because they are four different costs: `~` is queued (free, nothing
    is billed), `:` is allocated but starting up (billed, contributing nothing -- this
    is c, and it is why chaining is not free), `#` is productive, `.` is absent.
    """
    timeline = plan.timeline
    if not timeline.links:
        return ["(no jobs submitted)"]
    run_end = timeline.run_end_s or plan.config.horizon_s
    axis_end = max(run_end, max(l.end_s for l in timeline.links))
    axis_end = min(axis_end, plan.config.horizon_s)
    if axis_end <= 0:
        return ["(nothing happens before the horizon)"]

    sites = sorted({l.site for l in timeline.links})
    per_site: dict[str, list[str]] = {}
    for site in sites:
        links = [l for l in timeline.links if l.site == site]
        row = []
        for col in range(width):
            t = (col + 0.5) * axis_end / width
            char = "."
            for link in links:
                if link.productive_s <= t < min(link.end_s, run_end):
                    char = "#"
                    break
                if link.arrival_s <= t < link.productive_s:
                    char = ":" if char == "." or char == "~" else char
                elif link.submit_s <= t < link.arrival_s and char == ".":
                    char = "~"
            row.append(char)
        per_site[site] = row

    fed = []
    for col in range(width):
        live = sum(1 for site in sites if per_site[site][col] == "#")
        fed.append("^" if live >= 2 else " ")

    label_w = max(len(s) for s in sites + ["federated"]) + 1
    out = [f"{'':<{label_w}} 0 h{' ' * (width - 8)}{axis_end / 3600:.0f} h"]
    for site in sites:
        out.append(f"{site:<{label_w}}|{''.join(per_site[site])}|")
    out.append(f"{'federated':<{label_w}}|{''.join(fed)}|")
    out.append(f"{'':<{label_w}} legend: ~ queued (free)   : starting up (billed, idle)   "
               f"# productive   ^ two sites live")
    if run_end < axis_end - 1:
        out.append(f"{'':<{label_w}} the run ENDED at {run_end / 3600:.1f} h; the jobs' "
                   f"remaining walltime past that is not drawn")
    return out


def interval_table(plan: Plan) -> list[str]:
    head = (f"  {'from':>6} {'to':>6}  {'members':<26} {'H':>4} {'period':>7} "
            f"{'rounds':>6} {'fed':>4} {'blocks':>7}  stop")
    rows = [head, "  " + THIN[:len(head) - 2]]
    for iv in plan.timeline.intervals:
        members = ",".join(iv.members) or "(idle)"
        if len(members) > 26:
            members = members[:23] + "..."
        flag = "" if iv.quality == "identified" else f"  [{iv.quality}]"
        rows.append(
            f"  {_h(iv.t0_s, 6)} {_h(iv.t1_s, 6)}  {members:<26} {iv.inner_steps:>4} "
            f"{iv.period_s:>6.0f}s {iv.rounds:>6} {iv.federated_rounds:>4} "
            f"{iv.blocks_left:>7.0f}  {iv.stop_cause}{flag}")
    return rows


# --------------------------------------------------------------------------
# 5. the verdict, with the arithmetic beside it
# --------------------------------------------------------------------------


def _members_by_lane(plan: Plan) -> dict[str, Member]:
    return {f"{o.site}-l{k}": make_member(f"{o.site}-l{k}", o.candidate)
            for o in plan.selection for k in range(o.lanes)}


def period_arithmetic(plan: Plan, calibration: Calibration) -> list[str]:
    """`period{...} = H*step + merge + per-site overheads = N s`, once per regime.

    Printed for every membership the plan actually spends time in, because the
    two-site period is the number the whole plan turns on and it is 2.1x the inner
    phase the server logs as ">> Round took Ns".
    """
    lanes = _members_by_lane(plan)
    balance = any(a > 1 for ledger in plan.timeline.ledgers for a in ledger.accums)
    # The accumulation the JOBS were launched with, not the one this membership
    # would imply on its own -- otherwise the printed arithmetic for a solo interval
    # disagrees with the period the simulator charged for it.
    accum_by_lane = plan_accums(list(lanes.values()), balance=balance,
                                cap=plan.config.balance_max)
    seen: set[tuple[str, ...]] = set()
    out: list[str] = []
    for iv in plan.timeline.intervals:
        if not iv.members or iv.members in seen or iv.rounds == 0:
            continue
        seen.add(iv.members)
        members = [lanes[m] for m in iv.members if m in lanes]
        if not members:
            continue
        cost = round_cost(members, inner_steps=iv.inner_steps or plan.config.inner_steps,
                          calibration=calibration, balance=balance,
                          balance_max=plan.config.balance_max, explain=True,
                          accum_by_lane=accum_by_lane)
        tag = "" if cost.quality == "identified" else f"   [overhead: {cost.quality.upper()}]"
        out.append(f"  period{{{', '.join(iv.members)}}} = {cost.arithmetic}{tag}")
        out.append(f"    -> {3600 / cost.period_s:.1f} rounds/h, "
                   f"{_tok(cost.tokens)} tokens/round, {cost.blocks:.2f} blocks/round, "
                   f"accum {list(cost.accums)}")
    return out


# --------------------------------------------------------------------------
# the whole report
# --------------------------------------------------------------------------


def render(
    plan: Plan,
    collected: Collected,
    submissions: Sequence[Submission],
    calibration: Calibration,
    *,
    registry_geometries: dict | None = None,
    sections: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
) -> str:
    out: list[str] = []
    w = out.append
    cfg = plan.config
    tl = plan.timeline

    w(RULE)
    w(f"pww-plan  --  {plan.describe()}")
    w(f"alpha {cfg.alpha:g}  beta {cfg.beta:g}  horizon {cfg.horizon_s / 3600:g} h  "
      f"H-model {cfg.h_model} (H={cfg.inner_steps})  balance {cfg.balance}  "
      f"wait {cfg.wait_quantile}")
    w(RULE)

    if not plan.selection:
        # An empty plan is not "nothing to do": it means no (site, shape) survived
        # admission, and the reason is in section 2 with the command that fixes it.
        # Saying so here stops the reader concluding that the federation is a bad idea
        # when what actually happened is that a probe was missing.
        w("")
        w("  NO SUBMISSION IS RECOMMENDED, because nothing was admissible -- not "
          "because")
        w("  federating is a bad idea. Section 2 lists every refusal and the exact "
          "command,")
        w("  config edit or measurement that would make it admissible.")

    # Traps and blocking warnings come BEFORE anything that looks like a
    # recommendation. A degenerate plan that federates zero times must not be read as
    # an answer just because it sorted first among equally degenerate alternatives.
    if plan.traps:
        w("")
        for trap in plan.traps:
            w(f"  !! TRAP [{trap.code}] {trap.subject}")
            for line in _wrap(trap.reason, 74):
                w(f"     {line}")
        w("  This plan is FLAGGED, not recommended: every alternative the search saw "
          "was also degenerate.")
    # collect() copies its notes into PlannerInputs.warnings, so make_plan hands them
    # back alongside its own. Printing the union once, in read-then-planned order,
    # rather than the concatenation.
    seen: set[str] = set()
    for note in list(collected.notes) + list(plan.warnings):
        if note in seen:
            continue
        seen.add(note)
        w("")
        for i, line in enumerate(_wrap(note, 74)):
            w(("  ** " if i == 0 else "     ") + line)

    if 1 in sections:
        w(_section(1, "INPUTS AS READ"))
        for prov in collected.provenance:
            w("  " + prov.describe())
        w("")
        w(f"  {'site':<9} {'shape':<15} {'dev':>3} {'T':>5} {'w_raw p50':>10} "
          f"{'p90':>7} {'w_eff':>7} {'ratio':>7} {'age':>6} {'n':>3}  probed_by")
        w("  " + THIN[:100])
        for site in collected.inputs.sites:
            for shape in sorted(site.shapes, key=lambda s: (s.key.gpus, s.key.walltime_s)):
                wait = site.waits.get(shape.name)
                if wait is None:
                    continue
                ratio = "null" if wait.used_ratio is None else f"{wait.used_ratio:.4f}"
                w(f"  {site.site:<9} {shape.name:<15} {shape.key.gpus:>3} "
                  f"{shape.key.walltime_s / 3600:>4.0f}h {wait.p50_raw_s / 3600:>9.2f}h "
                  f"{wait.p90_raw_s / 3600:>6.2f}h "
                  f"{wait.eff_at(cfg.discount_strength, cfg.wait_quantile) / 3600:>6.2f}h "
                  f"{ratio:>7} {wait.probe_age_s / 60:>5.0f}m {wait.samples:>3}  "
                  f"{wait.probed_by_user or '?'}")
        w("")
        for option in plan.selection:
            wait = option.candidate.wait
            raw = wait.raw(cfg.wait_quantile)
            if wait.discounted:
                w(f"  discount [{option.site}/{option.candidate.shape.name}]: "
                  f"{raw / 3600:.2f} h * (1 - {cfg.discount_strength:g}*(1 - "
                  f"{wait.used_ratio:.4f})) = "
                  f"{wait.eff_at(cfg.discount_strength, cfg.wait_quantile) / 3600:.2f} h")
            else:
                w(f"  discount [{option.site}/{option.candidate.shape.name}]: NOT APPLIED "
                  f"(used_ratio {wait.used_ratio}); w_eff = w_raw = {raw / 3600:.2f} h")

    if 2 in sections:
        w(_section(2, "EXCLUSIONS -- what was refused, and what fixes it"))
        if not plan.exclusions:
            w("  (nothing was refused)")
        for exc in plan.exclusions:
            w(f"  [{exc.code}] {exc.subject}")
            for line in _wrap(exc.reason, 72):
                w(f"      {line}")
            if exc.fix:
                for i, line in enumerate(_wrap(exc.fix, 68)):
                    w(("      fix: " if i == 0 else "           ") + line)

    if 3 in sections:
        w(_section(3, "TIMELINE -- who is alive when, and when they overlap"))
        for line in presence_bars(plan):
            w("  " + line)
        w("")
        for line in interval_table(plan):
            w(line)

    if 4 in sections:
        w(_section(4, "PER-SITE LEDGER"))
        w(f"  {'site':<9} {'shape':<15} {'queue':>6} {'start':>6} {'HEAD':>6} "
          f"{'BETW':>6} {'CO-RES':>7} {'TAIL':>6} {'GAP':>5} {'idle':>6} {'merges':>7} "
          f"{'tokens':>9} {'blocks':>7} {'GPU-h':>7} {'tok/GPU-h':>10} accum")
        w("  " + THIN[:126])
        by_site = {o.site: o for o in plan.selection}
        for ledger in tl.ledgers:
            option = by_site.get(ledger.site)
            shape = option.candidate.shape.name if option else "-"
            w(f"  {ledger.site:<9} {shape:<15} {_h(ledger.queued_s, 5)}h "
              f"{ledger.startup_s / 60:>5.0f}m {_h(ledger.headstart_s, 5)}h "
              f"{_h(ledger.between_s, 5)}h "
              f"{_h(ledger.coresident_s, 6)}h {_h(ledger.tail_s, 5)}h "
              f"{_h(ledger.gap_s, 4)}h {ledger.idle_fraction * 100:>5.0f}% "
              f"{ledger.merges:>4}({ledger.federated_merges:>3}) "
              f"{_tok(ledger.tokens):>9} {ledger.blocks:>7.1f} "
              f"{ledger.gpu_s / 3600:>7.1f} {_tok(ledger.tokens_per_gpu_hour):>10} "
              f"{list(ledger.accums)}")
        w("")
        w("  `queue` is the SUM over a lane's links. On a self-resubmitting chain the "
          "successor")
        w("  queues WHILE the predecessor runs, so most of it is not dead time -- "
          "which is the")
        w("  whole reason the chain is submitted that way rather than with "
          "--dependency.")
        w("  HEAD/BETW/CO-RES/TAIL are four separate columns on purpose: one "
          "'walltime' number hides")
        w("  exactly what this planner exists to expose. BETW is solo presence BETWEEN "
          "two")
        w("  co-residency spells, which elastic membership makes normal and which the "
          "other three")
        w("  columns cannot hold. `idle` is 1 - inner-phase / presence, so it counts "
          "transport,")
        w("  merge and evaluate as idle too -- those are hours the GPUs are billed and "
          "not")
        w("  training. 74% was measured on Snellius before balancing. GPU-h is the "
          "ALLOCATION,")
        w("  which can outlast the run: the client does not exit when DARL is "
          "exhausted.")

    if 5 in sections:
        w(_section(5, "VERDICT"))
        s = plan.score
        w(f"  federated merges   {s.federated_merges}")
        w(f"  solo merges        {s.solo_merges}")
        w(f"  tokens             {_tok(s.tokens)}")
        w(f"  DARL blocks        {tl.blocks_used:.0f} used of {tl.blocks_available:.0f} "
          f"available"
          + (f"   EXHAUSTED at {tl.darl_exhausted_s / 3600:.1f} h"
             if tl.darl_exhausted_s is not None else "   (corpus left at the end)"))
        w(f"  round attempts     {tl.attempts_used} of {cfg.num_rounds}")
        w(f"  GPU-hours          {s.gpu_s / 3600:.1f}   tokens/GPU-h {_tok(s.tokens_per_gpu_hour)}"
          f"   idle {s.idle_fraction * 100:.0f}% of billed GPU-h")
        w("                     (idle = 1 - inner-phase/allocation: it counts the "
          "barrier wait, but also")
        w("                     transport, merge, evaluate, startup and any hours held "
          "after the corpus ends)")
        w(f"  run ends at        {tl.run_end_s / 3600:.1f} h"
          + (" (DARL exhausted; with max_epochs=1 there is no wraparound, so the run "
             "ENDS regardless of walltime left)" if tl.darl_exhausted_s is not None else ""))
        w(f"  first federated    {_h(tl.first_federated_s)} h    "
          f"last federated {_h(tl.last_federated_s)} h")
        w("")
        w(f"  U = N_fed + {cfg.alpha:g}*N_solo + {cfg.beta:g}*Tok/1e9 = "
          f"{s.federated_merges} + {cfg.alpha:g}*{s.solo_merges} + "
          f"{cfg.beta:g}*{s.tokens / 1e9:.2f} = {s.utility:.2f}")
        w(f"  alpha* (the value at which the recommendation changes): "
          + (f"{plan.alpha_star:.3f}" if plan.alpha_star is not None
             else "none -- the same plan wins for every alpha >= 0"))
        w(f"  search             {plan.search.method}, {plan.search.plans_evaluated} plans"
          + (f", measured optimality gap {plan.search.optimality_gap * 100:.1f}%"
             if plan.search.optimality_gap is not None else "")
          + (" (exact, proved optimal)" if plan.search.exact_proved else ""))
        w(f"  RECOMMENDED NUM_ROUNDS={plan.recommended_num_rounds}  "
          f"(for scripts/central_node/start_central_services.sh)")
        w("")
        w("  round period, with the arithmetic:")
        for line in period_arithmetic(plan, calibration):
            w(line)
        w("")
        w("  the round model against every regime that exists in all_logs/ "
          "(this is what makes")
        w("  the periods above worth anything):")
        for row in residuals(calibration):
            verdict = "ok" if row["within_tolerance"] else "FAILS"
            w(f"    {row['label']:<26} predicted {row['predicted_s']:>6.1f}s vs measured "
              f"{row['measured_s']:>6.1f}s  {row['rel_error'] * 100:>+6.2f}%  {verdict}"
              f"  [{row['quality']}]")
        bad = [r for r in residuals(calibration) if not r["within_tolerance"]]
        for row in bad:
            for i, line in enumerate(_wrap(
                    f"{row['label']} is NOT reproduced by the additive overhead model "
                    f"and is shipped tagged extrapolated: {row['note']}", 70)):
                w(("    !! " if i == 0 else "       ") + line)
        w("")
        for check in plan.crosschecks:
            mark = "" if check.agrees is None else ("  [agrees]" if check.agrees
                                                    else "  [DISAGREES with the simulator]")
            w(f"  cross-check {check.name}: {check.verdict}{mark}")
            for line in _wrap(check.detail, 70):
                w(f"      {line}")
            if check.agrees is False and check.name.startswith("chain_or_one_long_job"):
                site = check.name.split("[")[-1].rstrip("]")
                longest = max((o.candidate.walltime_s for o in plan.selection
                               if o.site == site), default=0.0)
                if longest and longest < cfg.horizon_s:
                    for line in _wrap(
                            f"expected here: {site}'s longest PROBED walltime is "
                            f"{longest / 3600:g} h against a {cfg.horizon_s / 3600:g} h "
                            f"horizon, so the simulator chains to stay PRESENT, not to "
                            f"buy a queue advantage. The closed form compares duty "
                            f"cycles at equal presence and cannot express that. Probe a "
                            f"longer shape and the two will agree.", 70):
                        w(f"      -- {line}")

    if 6 in sections:
        w(_section(6, "SENSITIVITY -- does the recommendation survive?"))
        w(f"  {'knob':<18} {'value':<8} {'U':>8} {'fed':>5} {'solo':>5}  winner")
        w("  " + THIN[:100])
        for row in plan.sensitivity:
            flag = " *CHANGES*" if row.changed else ""
            w(f"  {row.knob:<18} {row.value:<8} {row.utility:>8.2f} "
              f"{row.federated_merges:>5} {row.solo_merges:>5}  {row.winner}{flag}")
        by_q = {r.value: r.winner for r in plan.sensitivity if r.knob == "wait_quantile"}
        if len(set(by_q.values())) > 1:
            w("")
            w("  w-FRAGILE: the winner is not the same at p50 and p90. The queue wait is "
              "a")
            w("  distribution and this decision sits on the wrong side of it -- treat "
              "the plan as")
            w("  a bet on the median, not as a recommendation, and re-run when the "
              "probes move.")
        changed = sorted({r.knob for r in plan.sensitivity if r.changed})
        w("")
        if changed:
            w(f"  the winner CHANGES with: {', '.join(changed)}. A recommendation that "
              f"moves under")
            w(f"  a knob you cannot pin down is provisional -- c in particular is a "
              f"LOWER BOUND at")
            w(f"  every site, since no job script prints a timestamp before torchrun.")
        else:
            w("  the same plan wins under every knob swept. That is the strongest "
              "statement this")
            w("  planner can make about a recommendation.")

    if 7 in sections:
        w(_section(7, "MARGINAL LEDGER -- is each site worth submitting at all?"))
        for entry in plan.marginal:
            w("")
            for i, line in enumerate(_wrap(entry.detail, 72)):
                w(("  " if i == 0 else "    ") + line)
            w(f"    best option considered: {entry.best_option}")

    if 8 in sections:
        w(_section(8, "SBATCH -- paste in this order"))
        w("  TWO WAYS THE PREDICTED WAIT IS WRONG, and they point in opposite directions:")
        w("")
        w("  (a) `sbatch --test-only` answers for the queue AS IT IS NOW and assumes "
          "every running")
        # Read off the shapes THIS plan chose, never quoted from a past capture. The
        # constant here used to say "~0.11 ... 11%" while section 1 of the same report
        # printed 0.1617 for the same partition -- and this is the number the reader is
        # being asked to reason about when judging whether the wait is pessimistic.
        seen: list[str] = []
        for option in plan.selection:
            site_in = next((si for si in collected.inputs.sites
                            if si.site == option.site), None)
            if site_in is None:
                continue
            wait_in = site_in.waits.get(option.candidate.shape.name)
            if wait_in is not None and wait_in.used_ratio is not None:
                seen.append(f"{option.site} {option.candidate.shape.name} "
                            f"{wait_in.used_ratio:.4f}")
        if seen:
            w("      job runs to its full walltime. Measured used_ratio on the shapes "
              "this plan chose:")
            for line in _wrap("; ".join(seen), 66):
                w("        " + line)
            w("      -- i.e. finished jobs consumed that fraction of what they asked "
              "for, so the raw")
            w("      number is usually PESSIMISTIC. It is also conditioned on the "
              "PROBING account's")
            w("      fairshare, QOS and")
        else:
            w("      job runs to its full walltime, so the raw number is usually "
              "PESSIMISTIC -- though")
            w("      no usage row was found for the shapes this plan chose, so the "
              "discount below did")
            w("      NOT fire. It is also conditioned on the PROBING account's "
              "fairshare, QOS and")
        w("      priority -- if probed_by above is not you, it describes a different "
          "queue -- and it")
        w("      is a 10-minute snapshot, which is why probe age is printed beside it.")
        w("")
        w("  (b) The discount is a linear BLEND, not a model: w_eff = w * (1 - s*(1 - "
          "used_ratio)).")
        w("      It assumes the whole queue shrinks by one factor, which is wrong "
          "exactly when the")
        w("      blocking jobs are the long ones -- they are the jobs that do NOT "
          "finish early. And")
        w("      it SILENTLY NO-OPS when used_ratio is null, <= 0 or > 1, so the plan "
          "looks")
        w("      discounted when it is not; the only tell is `ratio null` in section 1.")
        w("")
        for sub in submissions:
            w("  # " + "\n  # ".join(_wrap(sub.comment, 74)))
            w("  " + sub.command.replace("\n", "\n  "))
            w("")
        if registry_geometries is not None:
            for line in balance_crosscheck(plan, calibration, registry_geometries):
                w("  " + line)
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    words, line, out = text.split(), "", []
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out or [""]


# --------------------------------------------------------------------------
# --json: the same tree, for a machine
# --------------------------------------------------------------------------


def as_json(
    plan: Plan, collected: Collected, submissions: Sequence[Submission], calibration: Calibration,
) -> dict[str, Any]:
    """`dataclasses.asdict(plan)` plus the things that are properties rather than fields.

    Properties do not survive asdict, and three of them (tokens/GPU-hour, barrier idle
    fraction, live hours) are exactly the always-reported-never-optimised series the
    design insists on, so they are materialised here rather than left for the consumer
    to re-derive.
    """
    tree = dataclasses.asdict(plan)
    tree["derived"] = {
        "tokens_per_gpu_hour": plan.score.tokens_per_gpu_hour,
        "idle_fraction": plan.score.idle_fraction,
        "rankable": plan.rankable,
        "describe": plan.describe(),
        "run_end_h": hours(plan.timeline.run_end_s),
        "darl_exhausted_h": hours(plan.timeline.darl_exhausted_s),
        "first_federated_h": hours(plan.timeline.first_federated_s),
        "per_site": {
            l.site: {
                "live_h": hours(l.live_s),
                "idle_fraction": l.idle_fraction,
                "tokens_per_gpu_hour": l.tokens_per_gpu_hour,
            } for l in plan.timeline.ledgers},
        "period_arithmetic": period_arithmetic(plan, calibration),
        "presence_bars": presence_bars(plan),
    }
    tree["provenance"] = [dataclasses.asdict(p) for p in collected.provenance]
    tree["notes"] = list(collected.notes)
    tree["sbatch"] = [dataclasses.asdict(s) for s in submissions]
    return tree


def dump_json(tree: dict[str, Any]) -> str:
    return json.dumps(tree, indent=1, sort_keys=True, default=str)
