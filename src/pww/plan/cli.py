#!/usr/bin/env python3
"""Plan a federated run: which sites to submit, at what shape, for how long, when.

    python3 -m pww.plan                                   # live: scanner + registry + DARL
    python3 -m pww.plan --dry-run tests/fixtures/plan/two-site.json
    python3 -m pww.plan show                              # every input, with provenance
    python3 -m pww.plan --alpha 0 --json > plan.json
    python3 -m pww.plan sbatch                            # just the commands

A Flower round is a barrier: the inner phase is a MAX over the live sites and the
transport/merge/evaluate overhead is a SUM over them. So sites must OVERLAP IN TIME to
federate at all, adding a slow site lowers the round rate for everyone already in, and
a solo headstart spends DARL corpus the federated phase then does not have. This tool
simulates that forward, round by round, over every combination of site, shape, lane
count, chaining policy and start time, and prints the decision with its arithmetic.

Sources, and what happens when one is missing:

    scanner     GET /overview, then /probes + /usage for the wait DISTRIBUTION.
                Unreachable -> says so and stops: with no measured wait there is no
                plan. --scanner-data-dir reads the CSVs directly; --dry-run replays a
                capture.
    registry    configs/site_throughput.env, keyed (site, devices). A geometry with no
                entry is EXCLUDED naming calibrate_throughput.sh, never extrapolated.
    DARL        GET /status for blocks left. Optional: --blocks pins it, and without
                either the whole corpus is ASSUMED and labelled as such.

Nothing is silently defaulted. Every input prints (value, source, staleness); every
refusal prints the command that fixes it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from . import adapter, emit as emit_mod, report as report_mod
from .adapter import Collected, Sources
from .model import PlanConfig
from .rounds import residuals
from .search import admit, make_plan
from .timeline import CHAIN_POLICIES, simulate

WHY_THIS_OBJECTIVE = """\
U = N_fed + alpha*N_solo + beta*(Tok/1e9)

  N_fed    merges with >= 2 distinct SITES contributing (two lanes at one site are two
           Flower clients and the server does merge them, but that is not the
           measurement this campaign exists to make)
  N_solo   merges with one site: DiLoCo k=1, outer lr 1.0, nothing discarded
  alpha    the exchange rate between them. Default 0.25. NEVER buried: every run
           re-solves over alpha and prints alpha*, the value at which the
           recommendation changes.

Always reported, never optimised: tokens, DARL blocks, GPU-hours, TOKENS PER GPU-HOUR
and BARRIER IDLE FRACTION. They are printed so a reader who rejects alpha can re-rank
by hand without re-running anything.

REJECTED, and why:

  tokens alone      This campaign's own finding is that a centralized run beats the
                    federated one at matched tokens, so maximising tokens recommends
                    "do not federate". It is also near-degenerate: DARL fixes the
                    total at whatever is left in the epoch. Reachable with --beta.
  round rate 1/T    Ignores duration. Recommends the shortest presence and the
                    smallest membership, i.e. it always excludes the slow site.
  tokens per GPU-h  Reported always, but optimising it recommends the cheapest
                    hardware; the run's purpose is a measurement, not a cost minimum.
  the upstream      slurm-scanner's POST /plan waterfills embarrassingly-parallel
  waterfill         work: per-site capacity integrals, no interaction between sites, a
                    site whose wait exceeds the horizon simply gets zero units. There
                    is no per-site independent work quantity here to split, because
                    the barrier makes the sites interact. It cannot be coaxed into
                    this shape by any parameter choice, which is why this planner
                    reads only its /probes and /usage and ignores its /plan.
"""


def _kv(text: str, scale: float = 1.0) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        if not value:
            raise argparse.ArgumentTypeError(f"expected site=value, got {item!r}")
        out[key.strip()] = float(value) * scale
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pww-plan",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="plan",
                    choices=["plan", "show", "explain", "sweep", "verify", "sbatch"],
                    help="plan (default) | show: every input with provenance, then stop | "
                         "explain: plan, then replay the winner round by round | "
                         "sweep: sensitivity only | verify: re-read live and diff against "
                         "a saved plan (--against) | sbatch: the commands only")

    src = ap.add_argument_group("sources")
    src.add_argument("--scanner-url", default=None,
                     help="slurm-scanner base URL. Default: the `server` field of "
                          "configs/slurm_probe/*.json, i.e. the instance this "
                          "checkout's own collectors POST to -- a planner whose "
                          "default is not where its probes went cannot read them "
                          f"(fallback {adapter.DEFAULT_SCANNER_URL}; "
                          f"{adapter.UPSTREAM_SCANNER_URL} is upstream's and does not "
                          f"answer from the aggregator VM)")
    src.add_argument("--scanner-data-dir", metavar="DIR",
                     help="read <DIR>/<cluster>/probes.csv directly instead of HTTP. This "
                          "is the path that works when the scanner is not up")
    src.add_argument("--dry-run", metavar="FIXTURE",
                     help="replay a recorded capture; no network at all. Replayed at the "
                          "capture's own timestamp so probe ages match")
    src.add_argument("--record", metavar="FILE",
                     help="write what was read as a fixture, so this plan can be "
                          "re-derived after the probe window has rolled over")
    src.add_argument("--darl-url", default=adapter.DEFAULT_DARL_URL,
                     help="DARL coordinator (the PORT names the arm: 29510/29520/29530/"
                          "29540 are four different epochs). 'none' to skip")
    src.add_argument("--darl-token", help="token value (prefer --darl-token-file)")
    src.add_argument("--darl-token-file", help="file holding the token; $DARL_TOKEN and "
                                               "the paths the job scripts try are used too")
    src.add_argument("--blocks", type=int, metavar="N",
                     help="pin the remaining corpus in DARL blocks (GET /status "
                          "'unassigned'), instead of asking the coordinator")
    src.add_argument("--registry", default="configs/site_throughput.env")
    src.add_argument("--planner-config", default="configs/plan/federation.json")
    src.add_argument("--probe-config-dir", default="configs/slurm_probe",
                     help="collector configs; a shape listed there with no probe row "
                          "becomes an exclusion naming the JSON to add")
    src.add_argument("--config", default="configs/titan/qwen3_0.6b_c4_diloco.toml",
                     help="the training toml the emitted jobs will use")
    src.add_argument("--root", default=".", help="PWW checkout the commands will run in")
    src.add_argument("--sites", help="comma-separated subset, e.g. snellius,lumi")
    src.add_argument("--require-own-probes", action="store_true",
                     help="refuse a probe collected by another account instead of "
                          "warning. --test-only is conditioned on the PROBING account's "
                          "fairshare, QOS and priority")
    src.add_argument("--now", type=float, metavar="EPOCH",
                     help="pretend it is this instant (fixtures replay at capture time)")

    obj = ap.add_argument_group("objective and model")
    obj.add_argument("--alpha", type=float, default=0.25,
                     help="credit for a SOLO merge relative to a federated one (default "
                          "0.25). alpha* is printed on every run")
    obj.add_argument("--beta", type=float, default=0.0, help="credit per 1e9 tokens")
    obj.add_argument("--horizon-h", type=float, default=48.0)
    obj.add_argument("--num-rounds", type=int, default=400,
                     help="Flower round ATTEMPT budget; a started round consumes one, "
                          "solo rounds included")
    obj.add_argument("--inner-steps", type=int, default=100, help="H (darl.inner_steps)")
    obj.add_argument("--h-model", default="fixed", choices=["fixed", "qsr", "replay"],
                     help="fixed = the full arm (QSR is OFF there: no qsr-h0 key, so H "
                          "is 100 for the whole run). qsr = the DCLT arm")
    obj.add_argument("--balance", default="auto", choices=["auto", "on", "off"],
                     help="gradient accumulation to fill the barrier. 'auto' decides "
                          "from which budget binds: under a DATA cap balancing costs "
                          "2.33x of the remaining federated merges for zero extra tokens")
    obj.add_argument("--discount-strength", type=float, default=0.5,
                     help="w_eff = w * (1 - s*(1 - used_ratio)); silently a no-op when "
                          "used_ratio is null")
    obj.add_argument("--wait-quantile", default="p50", choices=["p50", "p90"])
    obj.add_argument("--max-probe-age-h", type=float, default=6.0)
    obj.add_argument("--probe-window-h", type=float,
                     default=adapter.DEFAULT_PROBE_WINDOW_H,
                     help="how far back to ask the scanner for probe rows (default "
                          "%(default)g h). The shape_not_probed exclusion tells the "
                          "reader to check that this covers a collector cycle, so it "
                          "has to be reachable from the command line")
    obj.add_argument("--reserve-blocks", type=int, default=0)
    obj.add_argument("--startup-cost", type=_kv, default={}, metavar="SITE=SECONDS",
                     help="override c. It is a LOWER BOUND everywhere until a job script "
                          "prints a timestamp before torchrun")
    obj.add_argument("--startup-cost-h", type=lambda s: _kv(s, 3600.0), default={},
                     metavar="SITE=HOURS")
    obj.add_argument("--assume-overhead", action="store_true",
                     help="rank a plan whose round regime is not one of the measured "
                          "ones. Without this it is priced but NOT ranked")
    obj.add_argument("--warm", default="", metavar="SITES",
                     help="comma-separated sites that already hold a local DCP "
                          "checkpoint, so a first link pays no cold-join transient")

    sub = ap.add_argument_group("submission shape")
    sub.add_argument("--lanes-max", type=int, default=2,
                     help="a lane is a durable identity (replica id + own dump + own "
                          "checkpoint). Concurrency comes from lanes, never from two "
                          "jobs under one identity -- DARL refuses that with 503")
    sub.add_argument("--max-links-per-lane", type=int, default=8)
    sub.add_argument("--chain", default="self,none",
                     help="comma-separated of none|self|singleton. 'self' resubmits from "
                          "inside the running job; a --dependency-held job is not "
                          "backfill-eligible, which forfeits the point of short jobs")
    sub.add_argument("--begin-grid-h", default="", metavar="H,H,...",
                     help="extra --begin offsets to consider, on top of {now} and each "
                          "other site's predicted arrival")
    sub.add_argument("--darl-port", type=int, default=29510)
    sub.add_argument("--flower-port", type=int, default=29511)
    sub.add_argument("--output-dir", default=os.environ.get("PWW_OUTPUT_DIR", ""),
                     metavar="DIR",
                     help="PWW_OUTPUT_DIR for the aggregator line. What actually "
                          "separates two arms: start_central_services.sh keys its pid "
                          "files, token, launch.env and DARL state on "
                          "${PWW_OUTPUT_DIR}/central, NEVER on the port, so a "
                          "non-default --darl-port/--flower-port without this is a "
                          "silent no-op against an already-running default stack "
                          "(defaults to $PWW_OUTPUT_DIR)")
    sub.add_argument("--wandb-project", default="")
    sub.add_argument("--job-tag", default="", help="suffix for -J and the wandb run name")
    sub.add_argument("--no-fresh", action="store_true",
                     help="omit PWW_FRESH_RUN/PWW_FRESH_DELETE (they belong on the FIRST "
                          "link of a lane only, and this plan never puts them elsewhere)")

    out = ap.add_argument_group("search and output")
    out.add_argument("--exact", action="store_true", help="force exhaustive enumeration")
    out.add_argument("--greedy", action="store_true",
                     help="force marginal-value greedy; the optimality gap is then "
                          "unknown, and reported as unknown")
    out.add_argument("--max-exact-plans", type=int, default=60000,
                     help="above this the search falls back to greedy (default 60000, "
                          "roughly a minute)")
    out.add_argument("--json", action="store_true")
    out.add_argument("--sbatch-only", action="store_true")
    out.add_argument("--sections", default="", metavar="N,N",
                     help="report sections to print, 1-8")
    out.add_argument("--against", metavar="PLAN.JSON", help="for `verify`")
    out.add_argument("--why-this-objective", action="store_true")
    out.add_argument("--residuals", action="store_true",
                     help="print the round model against every measured regime and stop")
    return ap


def sources_from(args: argparse.Namespace) -> Sources:
    startup = dict(args.startup_cost)
    startup.update(args.startup_cost_h)
    return Sources(
        scanner_url=args.scanner_url or adapter.default_scanner_url(args.probe_config_dir),
        scanner_data_dir=args.scanner_data_dir,
        fixture=args.dry_run,
        darl_url=args.darl_url,
        darl_token=args.darl_token,
        darl_token_file=args.darl_token_file,
        registry=args.registry,
        planner_config=args.planner_config,
        probe_config_dir=args.probe_config_dir,
        sites=tuple(s.strip() for s in args.sites.split(",")) if args.sites else (),
        submitter=_whoami() if args.require_own_probes else None,
        blocks=args.blocks,
        discount_strength=args.discount_strength,
        probe_window_h=args.probe_window_h,
        startup_overrides=startup,
        warm_sites=tuple(s.strip() for s in args.warm.split(",") if s.strip()),
        now=args.now,
    )


def _whoami() -> str:
    import getpass
    try:
        return getpass.getuser()
    except Exception:
        return ""


def config_from(args: argparse.Namespace) -> PlanConfig:
    max_exact = args.max_exact_plans
    if args.greedy:
        max_exact = 0
    if args.exact:
        max_exact = 10 ** 12
    grid = tuple(float(h) * 3600 for h in args.begin_grid_h.split(",") if h.strip())
    return PlanConfig(
        alpha=args.alpha,
        beta=args.beta,
        horizon_s=args.horizon_h * 3600.0,
        num_rounds=args.num_rounds,
        inner_steps=args.inner_steps,
        h_model=args.h_model,
        balance=args.balance,
        wait_quantile=args.wait_quantile,
        reserve_blocks=args.reserve_blocks,
        lanes_max=args.lanes_max,
        max_links_per_lane=args.max_links_per_lane,
        chain_policies=_chain_policies(args.chain),
        begin_grid_s=grid,
        max_probe_age_s=args.max_probe_age_h * 3600.0,
        discount_strength=args.discount_strength,
        assume_overhead=args.assume_overhead,
        max_exact_plans=max_exact,
    )


# Imported from timeline, not restated: this used to be a hand-maintained third copy
# of the names, and a copy that falls BEHIND the simulator refuses at the flag a policy
# the planner can price -- the same divergence as a typo getting through, reversed.
_CHAIN_POLICIES = CHAIN_POLICIES


def _chain_policies(raw: str) -> tuple[str, ...]:
    policies = tuple(c.strip() for c in raw.split(",") if c.strip())
    unknown = [c for c in policies if c not in _CHAIN_POLICIES]
    if unknown:
        raise SystemExit(
            f"--chain: unknown policy {', '.join(repr(u) for u in unknown)}. "
            f"Choose from {'|'.join(_CHAIN_POLICIES)}. This is refused rather than "
            f"ignored because the simulator prices an unknown policy as a single job "
            f"per lane while the emitter still submits the whole chain, so the plan's "
            f"merge counts and the jobs you paste would not describe the same run.")
    if not policies:
        raise SystemExit(f"--chain: empty. Choose from {'|'.join(_CHAIN_POLICIES)}.")
    return policies


def emit_config_from(args: argparse.Namespace) -> emit_mod.EmitConfig:
    return emit_mod.EmitConfig(
        root=args.root,
        config_toml=args.config,
        darl_port=args.darl_port,
        flower_port=args.flower_port,
        output_dir=getattr(args, "output_dir", "") or "",
        wandb_project=args.wandb_project,
        job_name_tag=args.job_tag,
        fresh=not args.no_fresh,
    )


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------


def cmd_show(collected: Collected, config: PlanConfig) -> int:
    print(report_mod.RULE)
    print("INPUTS AS READ -- value, source, staleness")
    print(report_mod.RULE)
    for prov in collected.provenance:
        print("  " + prov.describe())
    print()
    for note in collected.notes:
        for i, line in enumerate(report_mod._wrap(note, 74)):
            print(("  ** " if i == 0 else "     ") + line)
    for site in collected.inputs.sites:
        cands, excl = admit(site, config=config, calibration=collected.inputs.calibration)
        print(f"\n  {site.site}: c = {site.startup_s:.0f} s [{site.startup_quality}], "
              f"{len(site.shapes)} probed shapes, geometries "
              f"{sorted(site.geometries)}, limits {site.limits.source}")
        for cand in cands:
            wait = cand.wait.eff_at(config.discount_strength, config.wait_quantile)
            print(f"    ADMIT  {cand.shape.name:<16} {cand.gpus:>3} dev "
                  f"{cand.walltime_s / 3600:>4.0f} h  wait {wait / 3600:>6.2f} h  "
                  f"step {cand.geometry.step_s:.3f} s  overhead {cand.overhead_s:.1f} s "
                  f"[{cand.overhead_quality}]")
        for exc in excl:
            print(f"    REFUSE {exc.subject}: {exc.reason}")
    for exc in collected.inputs.exclusions:
        print(f"  REFUSE [{exc.code}] {exc.subject}: {exc.reason}")
    return 0


def cmd_explain(plan, calibration) -> int:
    """Replay the winner round by round.

    The simulation is deterministic -- no RNG, no clock -- so re-simulating the chosen
    selection reproduces the plan exactly; the only thing this adds is
    record_rounds=True, which the search leaves off because building one dataclass per
    round of every plan it discards was the largest allocation in the profile.
    """
    # reserve_blocks is zeroed because _darl_of already hands back the POST-reserve
    # figure the plan was simulated against; applying the reserve twice would replay a
    # shorter run than the one being explained.
    timeline = simulate(
        plan.selection, config=dataclasses.replace(plan.config, reserve_blocks=0),
        calibration=calibration, darl=_darl_of(plan),
        balance=any(a > 1 for l in plan.timeline.ledgers for a in l.accums),
        record_rounds=True)
    print(f"{'#':>4} {'start':>7} {'period':>7} {'phase':>7} {'over':>6} {'H':>4} "
          f"{'fed':>4} {'tokens':>9} {'blk':>6}  members")
    print(report_mod.THIN)
    for rnd in timeline.rounds:
        members = ",".join(rnd.fit) + ("  +eval " + ",".join(rnd.eval_only)
                                       if rnd.eval_only else "")
        stall = f"  (+{rnd.stall_s:.0f}s cold-join stall)" if rnd.stall_s else ""
        print(f"{rnd.index:>4} {rnd.start_s / 3600:>6.2f}h {rnd.period_s:>6.0f}s "
              f"{rnd.phase_s:>6.0f}s {rnd.overhead_s:>5.0f}s {rnd.inner_steps:>4} "
              f"{'YES' if rnd.federated else '-':>4} {rnd.tokens:>9,} "
              f"{rnd.blocks:>6.2f}  {members}{stall}")
    print(f"\n  {len(timeline.rounds)} rounds, "
          f"{timeline.federated_merges} federated, {timeline.solo_merges} solo")
    return 0


def _darl_of(plan):
    from .model import DarlState
    return DarlState(
        num_blocks=int(plan.timeline.blocks_available),
        committed=0, leased=0,
        unassigned=int(plan.timeline.blocks_available),
        source="replayed from the plan")


def cmd_verify(plan, against: str | None) -> int:
    if not against:
        print("verify needs --against <a plan written with --json>", file=sys.stderr)
        return 2
    saved = json.loads(Path(against).expanduser().read_text())
    old = saved.get("derived", {}).get("describe", "(unknown)")
    new = plan.describe()
    print(f"  saved plan : {old}")
    print(f"  re-planned : {new}")
    if old == new:
        print("  UNCHANGED -- the queue and the corpus still support the same decision.")
        return 0
    print("  CHANGED. The inputs moved since that plan was written; the sections above "
          "say which.")
    old_fed = saved.get("score", {}).get("federated_merges")
    print(f"  federated merges: {old_fed} -> {plan.score.federated_merges}")
    return 1


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.why_this_objective:
        print(WHY_THIS_OBJECTIVE)
        return 0
    if args.residuals:
        print(f"  {'regime':<28} {'predicted':>10} {'measured':>9} {'error':>8}  ok  quality")
        for row in residuals():
            print(f"  {row['label']:<28} {row['predicted_s']:>9.1f}s "
                  f"{row['measured_s']:>8.1f}s {row['rel_error'] * 100:>+7.2f}%  "
                  f"{'ok' if row['within_tolerance'] else 'NO':<3} {row['quality']}")
        return 0

    src = sources_from(args)
    config = config_from(args)
    t0 = time.monotonic()
    collected = adapter.collect(src)
    print(f"  ... read {len(collected.inputs.sites)} site(s) in "
          f"{time.monotonic() - t0:.1f} s", file=sys.stderr)

    if args.record:
        path = adapter.record(collected, args.record)
        print(f"  ... recorded the inputs to {path}", file=sys.stderr)

    if args.command == "show":
        return cmd_show(collected, config)

    if not collected.inputs.sites:
        # Not an exception: the sources said what they said, and the reasons are the
        # useful output. Printing them and exiting non-zero is more use than a stack.
        print(report_mod.RULE)
        print("NO PLAN: no site could be assembled.")
        print(report_mod.RULE)
        for note in collected.notes:
            for i, line in enumerate(report_mod._wrap(note, 74)):
                print(("  ** " if i == 0 else "     ") + line)
        for exc in collected.inputs.exclusions:
            print(f"  [{exc.code}] {exc.subject}: {exc.reason}")
            if exc.fix:
                print(f"      fix: {exc.fix}")
        return 2

    t1 = time.monotonic()
    plan = make_plan(collected.inputs, config)
    print(f"  ... {plan.search.plans_evaluated} plans in {time.monotonic() - t1:.1f} s "
          f"({plan.search.method})", file=sys.stderr)

    submissions = emit_mod.emit(plan, collected.inputs.calibration, emit_config_from(args),
                                darl=collected.inputs.darl)
    problems = emit_mod.preflight(emit_config_from(args))

    if args.json:
        tree = report_mod.as_json(plan, collected, submissions, collected.inputs.calibration)
        tree["preflight"] = problems
        print(report_mod.dump_json(tree))
        # Exit status describes the PLAN, not the tree: 0 recommended, 1 flagged as a
        # trap or priced off an unmeasured regime, 2 nothing admissible. Preflight
        # problems travel in the tree so a caller can act on them without parsing
        # text, but they do not change the status -- they are about the checkout, not
        # about whether the plan is sound.
        if not plan.selection:
            return 2
        return 0 if plan.rankable else 1

    if args.command == "explain":
        return cmd_explain(plan, collected.inputs.calibration)

    sections = tuple(int(s) for s in args.sections.split(",") if s.strip())
    if args.sbatch_only or args.command == "sbatch":
        sections = (8,)
    elif args.command == "sweep":
        sections = (5, 6)
    elif not sections:
        sections = (1, 2, 3, 4, 5, 6, 7, 8)

    geometries = {s.site: dict(s.geometries) for s in collected.inputs.sites}
    print(report_mod.render(
        plan, collected, submissions, collected.inputs.calibration,
        registry_geometries=geometries, sections=sections))

    if problems:
        print("\n" + report_mod.RULE)
        print("PREFLIGHT -- fix before pasting the commands above")
        print(report_mod.RULE)
        for problem in problems:
            for i, line in enumerate(report_mod._wrap(problem, 74)):
                print(("  !! " if i == 0 else "     ") + line)

    if args.command == "verify":
        return cmd_verify(plan, args.against)
    if not plan.selection:
        return 2  # nothing admissible: the exclusions are the output, and this is a failure
    return 0 if plan.rankable else 1


if __name__ == "__main__":
    raise SystemExit(main())
