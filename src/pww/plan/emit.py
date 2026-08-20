"""Turn a Plan into sbatch lines a scientist can paste.

THE INVARIANT, and the reason this module exists at all: the shape flags of every
emitted line are a SUBSTRING-EXACT copy of the probe row's own `args` string. Not
re-rendered from the parsed ShapeKey, not normalised, not re-ordered. Two things
follow, and both are bugs that have actually shipped:

  * it is structurally impossible to quote a wait measured at one walltime and then
    submit another. The upstream planner does exactly that -- it reads the GPU count
    from plan.json and the walltime cap from a cluster-level setting and never looks
    at `-t` -- and will answer a probe for an 8 h shape with "submit 12.4 h".
  * --cpus-per-task and --mem travel with --gpus-per-node whether or not anything
    here understands them. Overriding the device count alone leaves the full-node
    core and memory request in the job script's header, so a "1-GPU" job queues and
    bills as a whole node.

Everything else on the line is run-specific and appended after the verbatim block:
-J, --begin, --export.

WHAT IS DELIBERATELY NOT DELEGATED. Accumulation is emitted as an explicit
PWW_GRAD_ACCUM computed from the planner's own (site, devices) table for the
membership this plan actually chose. run_train.sh's PWW_BALANCE block keys the
registry on the bare site name (PWW_TPUT_LUMI), so on a 1-GCD job it would balance
using the 8-GCD numbers; and its default peer set is every PWW_TPUT_* variable in the
file, so adding a third site silently re-derives accumulation for the other two.
Setting PWW_GRAD_ACCUM explicitly is the documented way to disable that derivation
(`if [[ "${PWW_BALANCE:-0}" == "1" && -z "${PWW_GRAD_ACCUM:-}" ]]`). The equivalent
PWW_BALANCE line is printed as a comment so the two can be compared, and a
disagreement between them is a finding, not a formatting choice.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .model import DarlState, Plan
from .rounds import make_member, plan_accums, round_cost


@dataclass(frozen=True)
class EmitConfig:
    """Everything on the sbatch line that is not the shape."""

    root: str = "."
    config_toml: str = "configs/titan/qwen3_0.6b_c4_diloco.toml"
    darl_port: int = 29510
    flower_port: int = 29511
    token_var: str = "DARL_TOKEN"  # shell variable holding the token, not the token
    wandb_project: str = ""
    val_windows: int = 512
    job_name_tag: str = ""
    fresh: bool = True
    # The aggregator's whole state -- pid files, token, launch.env, space.env, the
    # DARL lease table, the global model -- lives under ${PWW_OUTPUT_DIR}/central, and
    # is keyed on THAT, never on the port. So a non-default port pair without a
    # matching output dir is not a second arm: start_central_services.sh finds the
    # default stack's live darl.pid/flower.pid, prints "already running", and exits 0
    # having started nothing on the new ports, while the sites queue for hours against
    # a Flower server that is not there.
    output_dir: str = ""
    dump_base: str = ""  # default: read dump_folder out of the toml
    chain_lead_s: float = 0.0


@dataclass(frozen=True)
class Submission:
    """One sbatch line, plus the pieces the report and the tests want separately."""

    site: str
    lane_id: str
    order: float  # submit time in seconds from now; the report emits in this order
    begin_s: float
    args_verbatim: str
    command: str
    comment: str


def _read_dump_folder(toml_path: str | Path) -> str:
    """dump_folder out of the toml, the same way run_train.sh greps for it.

    Same grep, deliberately: if the two disagree about which directory a lane owns,
    PWW_FRESH_RUN clears one and the job resumes the other.
    """
    try:
        text = Path(toml_path).expanduser().read_text()
    except OSError:
        return "./outputs/pww"
    match = re.search(r'^\s*dump_folder\s*=\s*"([^"]+)"', text, re.M)
    return match.group(1) if match else "./outputs/pww"


def preflight(emit: EmitConfig) -> list[str]:
    """Blocking problems with the tree these commands are about to run in.

    Checked rather than assumed because both of these silently produce the WRONG RUN
    rather than an error: a lane whose PWW_DUMP is ignored shares the other lane's
    checkpoint, and PWW_FRESH_RUN then clears the shared one.
    """
    problems: list[str] = []
    root = Path(emit.root)
    run_train = root / "scripts" / "titan" / "run_train.sh"
    try:
        text = run_train.read_text()
    except OSError:
        problems.append(f"{run_train} not found: --root is probably not the PWW checkout")
        return problems
    if not re.search(r'^DUMP="\$\{PWW_DUMP:-\}"', text, re.M):
        problems.append(
            f"{run_train}:40 still reads DUMP=\"\" . Both job scripts call run_train.sh "
            f"with a fixed argument list terminated by `-- \"$@\"`, so --dump is "
            f"unreachable from any sbatch line and PWW_DUMP is ignored. A multi-lane "
            f"plan then shares one checkpoint directory, and PWW_FRESH_RUN=1 reads "
            f"fresh_dump=\"${{DUMP:-}}\" and falls back to the toml's dump_folder -- so it "
            f"clears the SHARED checkpoint rather than the lane's. Fix: change line 40 to "
            f"DUMP=\"${{PWW_DUMP:-}}\" .")
    return problems


def accums_for(plan: Plan, calibration) -> dict[str, int]:
    """{site: PWW_GRAD_ACCUM} at the planned membership.

    The SAME helper the simulator debits blocks with (rounds.plan_accums), so the
    number exported here is the number the plan was priced at. They used to be
    computed two different ways -- the simulator per live round, the emitter once
    over the whole plan -- and a solo headstart was then simulated at accum 1 while
    the job it emitted ran at accum 5, i.e. at 5x the corpus burn.
    """
    balance = any(a > 1 for ledger in plan.timeline.ledgers for a in ledger.accums)
    members = [make_member(f"{o.site}-l{k}", o.candidate)
               for o in plan.selection for k in range(o.lanes)]
    if not members:
        return {}
    by_lane = plan_accums(members, balance=balance, cap=plan.config.balance_max)
    out: dict[str, int] = {}
    for member in members:
        out[member.site] = max(out.get(member.site, 1), by_lane[member.lane_id])
    return out


# Resource flags that belong to the SITE, not to the shape: the probe args carry
# -p/-N/--gpus-per-node/-t (those are what the wait was measured at) and nothing else,
# so a chained submission -- whose batch script is job_chain_link.sh, not the site's
# own job script -- silently loses the #SBATCH header the throughput calibration was
# measured with. --cpus-per-task in particular defaults to 1, which starves the
# dataloader; --output defaults to slurm-%j.out in the submit directory, so the log
# the planner tells you to read is not written.
_SITE_RESOURCE_FLAGS = {
    "--cpus-per-task": "-c",
    "--ntasks-per-node": None,
    "--mem": None,
    "--mem-per-cpu": None,
    "--mem-per-gpu": None,
    "--output": "-o",
    "--error": "-e",
}


def site_resource_flags(job_script: str | Path, args_verbatim: str) -> list[str]:
    """The site job script's own #SBATCH resource flags that `args_verbatim` omits.

    Read from the script rather than hard-coded: Snellius asks for 64 cores and no
    --mem, LUMI for 56 and 480G, and those numbers are the ones the measured
    throughput belongs to.
    """
    try:
        text = Path(job_script).read_text()
    except OSError:
        return []
    present = set(args_verbatim.split())
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("#SBATCH"):
            continue
        token = line[len("#SBATCH"):].strip().split()
        if not token:
            continue
        flag, _, inline = token[0].partition("=")
        if flag not in _SITE_RESOURCE_FLAGS:
            continue
        short = _SITE_RESOURCE_FLAGS[flag]
        if flag in present or (short and short in present):
            continue  # the probe args already say it; the shape wins
        value = inline or (token[1] if len(token) > 1 else "")
        if not value:
            continue
        out.append(f"{flag}={value}")
    return out


def _minutes(seconds: float) -> int:
    """--begin=now+Nminutes. Rounded UP: arriving a minute late costs one round;
    arriving early risks a DARL 503 against a predecessor that has not released."""
    return int(math.ceil(max(0.0, seconds) / 60.0))


def _export(pairs: Sequence[tuple[str, str]]) -> str:
    """--export="ALL,K=V,..." -- quoted as ONE shell word, deliberately.

    Slurm splits the list on commas, so a value may contain spaces, but the SHELL
    splits on spaces first: an unquoted --export=...,PWW_CHAIN_ARGS=-p gpu_h100 loses
    everything after the first space and sbatch sees `gpu_h100` as a positional
    argument. Quoting the whole thing also lets $DARL_TOKEN expand at paste time, so
    the token itself never appears in a plan, a log or a --json dump.
    """
    return '--export="' + ",".join(["ALL"] + [f"{k}={v}" for k, v in pairs]) + '"'


def emit(plan: Plan, calibration, emit_cfg: EmitConfig | None = None,
         darl: DarlState | None = None) -> list[Submission]:
    """One Submission per lane, in submission order, aggregator first.

    A lane gets ONE line even when it is several links: the successors are submitted
    by the running job itself (scripts/titan/job_chain_link.sh), not by the scientist.
    That is not a convenience -- a --dependency-held job is not eligible for backfill,
    which forfeits exactly the advantage that motivated short jobs, and precomputed
    --begin offsets drift on the first mispredicted wait.
    """
    cfg = emit_cfg or EmitConfig()
    accums = accums_for(plan, calibration)
    dump_base = cfg.dump_base or _read_dump_folder(Path(cfg.root) / cfg.config_toml)
    out: list[Submission] = []

    # PWW_FRESH_RUN wipes state at BOTH ends or at neither, and which one this plan
    # describes is a fact about the coordinator it was priced against, not a taste.
    # A plan built on "822 of 2692 blocks left" is a plan for a run 69% through its
    # corpus: telling the operator to delete both sites' checkpoints while leaving
    # the coordinator's lease table alone serves no intent, and passing
    # PWW_FRESH_RUN=1 to the aggregator too would reset the corpus to 2692 and make
    # every number in the plan wrong by 3.3x.
    # NOT `unassigned < num_blocks`: --blocks N and the whole-corpus fallback both
    # build unassigned == num_blocks because they have no committed count to subtract,
    # so that test reads them as "fresh epoch" and wipes a coordinator that is 69%
    # through. Only a coordinator that actually ANSWERED can license a wipe.
    mid_run = bool(darl is not None and not darl.fresh_epoch)
    fresh = cfg.fresh and not mid_run
    if darl is None or darl.fresh_epoch:
        no_fresh_note = ""
    elif darl.observed:
        no_fresh_note = (
            f" This plan is priced against a coordinator that is PART WAY THROUGH its "
            f"epoch ({darl.unassigned} of {darl.num_blocks} blocks left), so no fresh "
            f"flags are emitted at either end: resetting one side and not the other "
            f"produces a run that is neither the old one nor a new one.")
    else:
        # --blocks / assumed. The corpus figure is real enough to PRICE the plan and
        # not good enough to license deleting anything, and those are different bars.
        no_fresh_note = (
            f" The remaining corpus was NOT read from a coordinator (source: "
            f"{darl.source or 'unknown'}), so this plan cannot tell a fresh epoch from "
            f"one most of the way through, and no fresh flags are emitted at either "
            f"end. That is deliberate: PWW_FRESH_RUN=1 here would reset the lease "
            f"table to the full corpus, discard the global model and its momentum, and "
            f"rm -rf both lanes' checkpoints -- making every number in this plan wrong "
            f"against the run it just destroyed. If you KNOW this is a new campaign, "
            f"add PWW_FRESH_RUN=1 to the aggregator line and "
            f"PWW_FRESH_RUN=1,PWW_FRESH_DELETE=1 to each site line by hand.")

    # The aggregator first, because a site that connects to nothing gets a Flower
    # connect error, and because NUM_ROUNDS is sized off this plan.
    central = [f"NUM_ROUNDS={plan.recommended_num_rounds}"]
    # NOT optional, and not the same variables the site lines use. Omitting
    # AGGREGATOR_CONFIG falls back to configs/central_aggregator.yaml on a stack with
    # no runs/central/launch.env -- that is the ResNet/CIFAR file: min-clients 2 (so
    # every solo round this plan counts is impossible), server-momentum 0.0 (FedAvg,
    # not FedMom) and round-timeout 300 s, which the two-site period exceeds.
    # start_central_services.sh reads DARL_PORT/FLOWER_PORT, NOT the PWW_-prefixed
    # names the job scripts read, so a non-default port pair has to be spelled both
    # ways or the sites connect to a server that is not there.
    central.append("AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml")
    central.append(f"DARL_PORT={cfg.darl_port}")
    central.append(f"FLOWER_PORT={cfg.flower_port}")
    # The ports alone do NOT separate two arms; PWW_OUTPUT_DIR does. See EmitConfig.
    if cfg.output_dir:
        central.append(f"PWW_OUTPUT_DIR={cfg.output_dir}")
    non_default_ports = (cfg.darl_port, cfg.flower_port) != (29510, 29511)
    port_warning = ""
    if non_default_ports and not cfg.output_dir:
        port_warning = (
            f" WARNING: DARL_PORT/FLOWER_PORT are non-default ({cfg.darl_port}/"
            f"{cfg.flower_port}) but no PWW_OUTPUT_DIR is set. The aggregator keys ALL "
            f"of its state -- pid files, token, launch.env, space.env, the lease table "
            f"and the global model -- on ${{PWW_OUTPUT_DIR}}/central and never on the "
            f"port, so on a machine already running the default stack this line is a "
            f"silent no-op: it prints 'already running' and exits 0 having started "
            f"nothing on {cfg.darl_port}/{cfg.flower_port}, and both site jobs then "
            f"spend their queue wait connecting to a Flower server that is not there. "
            f"With the default stack DOWN it is worse: this arm writes its lease table "
            f"and snapshot into the SAME runs/darl as the 29510 arm. Re-run with "
            f"--output-dir <dir> (and pass BLOB_PORT too, as RUNBOOK_DCLT.md does).")
    if fresh:
        # Both ends or neither: DARL_FRESH alone resets the lease table and keeps the
        # global model, which is the half-reset PWW_FRESH_RUN exists to prevent.
        central.append("PWW_FRESH_RUN=1")
    out.append(Submission(
        site="(central)", lane_id="(central)", order=-1.0, begin_s=0.0, args_verbatim="",
        command=" ".join(central) + " \\\n  ./scripts/central_node/start_central_services.sh",
        comment=(f"attempts: a round in which every site was queued costs nothing "
                 f"(sample() blocks on min_available_clients), but every STARTED round "
                 f"including solo ones consumes one. This plan starts "
                 f"{plan.timeline.attempts_used}; {plan.recommended_num_rounds} leaves "
                 f"margin, and too high costs nothing. NUM_ROUNDS is applied only when "
                 f"the Flower server is actually (re)started -- if one is already "
                 f"running the script skips it and this value is silently ignored."
                 + port_warning
                 + ("" if fresh else no_fresh_note))))

    for option in sorted(plan.selection, key=lambda o: (o.begin_s, o.site)):
        job_script = f"scripts/{option.site}/job_titan_diloco.sh"
        accum = accums.get(option.site, 1)
        for lane in range(option.lanes):
            lane_id = f"{option.site}-l{lane}"
            multi_lane = option.lanes > 1
            chained = option.links_per_lane > 1 and option.chain != "none"

            name = f"pww-{option.site}-titan"
            if cfg.job_name_tag:
                name += f"-{cfg.job_name_tag}"
            if multi_lane:
                name += f"-l{lane}"

            exports: list[tuple[str, str]] = [(cfg.token_var, f"${cfg.token_var}")]
            if cfg.config_toml:
                exports.append(("CONFIG", cfg.config_toml))
            exports += [("PWW_DARL_PORT", str(cfg.darl_port)),
                        ("PWW_FLOWER_PORT", str(cfg.flower_port))]
            # Explicit, never delegated -- see the module docstring.
            exports.append(("PWW_GRAD_ACCUM", str(accum)))
            if cfg.val_windows:
                exports.append(("PWW_VAL_WINDOWS", str(cfg.val_windows)))
            if multi_lane:
                # REPLICA is never emitted without a matching PWW_DUMP: --replica
                # separates the DARL cluster id, the Flower client id and the delta-blob
                # key, but NOT dump_folder, so two lanes would still share one DCP
                # checkpoint, one blob-staging directory and one tb directory.
                # The dump carries the LANE ID, not just the index: run_train.sh
                # builds the DARL cluster id as "${SITE}-${REPLICA}", so lane_id and
                # the directory name then say the same thing, and a directory listing
                # on a shared filesystem is unambiguous about which site owns what.
                exports += [("REPLICA", f"l{lane}"),
                            ("PWW_DUMP", f"{dump_base}-{lane_id}")]
            if fresh:
                # First link of the lane only. Copied onto a successor they delete the
                # lane's checkpoint, which is the whole thing a lane exists to keep --
                # job_chain_link.sh unsets them for the sbatch call AND pins them to 0
                # in the successor's --export so that ALL cannot carry them across.
                exports += [("PWW_FRESH_RUN", "1"), ("PWW_FRESH_DELETE", "1")]
            if cfg.wandb_project:
                exports += [("ENABLE_WANDB", "1"), ("WANDB_PROJECT", cfg.wandb_project)]
                # WANDB_RUN_NAME is deliberately absent on a chained lane: run_train.sh
                # appends the Slurm job id only when it derives the name itself, so an
                # exported name gives every link an identical display name and the
                # chain reads as one confusing run.
                if not chained:
                    exports.append(("WANDB_RUN_NAME", f"diloco-{option.site}"
                                    + (f"-l{lane}" if multi_lane else "")
                                    + (f"-{cfg.job_name_tag}" if cfg.job_name_tag else "")))

            target = job_script
            # The site job script's own #SBATCH resource flags, resolved ONCE and used
            # in BOTH places they are needed. job_chain_link.sh is the batch script for
            # EVERY link of a chained lane, not just the first, and it carries no
            # #SBATCH header, so these flags have to travel on the sbatch command line
            # of link 1 AND inside PWW_CHAIN_ARGS, which is the only thing links 2..N
            # are submitted with (job_chain_link.sh:131 additionally `env -u`s
            # SLURM_CPUS_PER_TASK/SLURM_NTASKS_PER_NODE/SLURM_MEM_PER_NODE, so there is
            # deliberately no inheritance path for them either). Putting them on link 1
            # alone leaves links 2..N at --cpus-per-task=1 -- a CPU shape the measured
            # throughput in configs/site_throughput.env was never calibrated at -- and
            # writes their logs to slurm-%j.out instead of the logs/ path the report
            # tells the operator to read.
            res_flags = (site_resource_flags(Path(cfg.root) / job_script,
                                             option.candidate.shape.args)
                         if chained else [])
            if chained:
                exports += [
                    ("PWW_CHAIN_LINKS", str(option.links_per_lane)),
                    ("PWW_CHAIN_SCRIPT", job_script),
                    ("PWW_CHAIN_LANE", lane_id),
                    # '|' rather than a space: --export values with spaces survive
                    # the shell only by luck and not every Slurm build round-trips
                    # them. job_chain_link.sh splits on '|' with IFS.
                    ("PWW_CHAIN_ARGS",
                     "|".join((option.candidate.shape.args + f" -J {name}").split()
                              + res_flags)),
                ]
                if cfg.chain_lead_s:
                    exports.append(("PWW_CHAIN_LEAD_S", str(int(cfg.chain_lead_s))))
                target = "scripts/titan/job_chain_link.sh"

            parts = ["sbatch", option.candidate.shape.args, f"-J {name}"]
            if chained:
                # Appended AFTER the verbatim probe args and only for flags those args
                # do not set, so the shape -- and therefore the walltime the wait was
                # measured at -- is untouched. Same list that went into PWW_CHAIN_ARGS
                # above; if these two ever disagree, links 1 and 2..N run at different
                # CPU shapes and only one of them matches the calibrated throughput.
                parts += res_flags
            if option.begin_s > 0:
                parts.append(f"--begin=now+{_minutes(option.begin_s)}minutes")
            parts.append(_export(exports))
            parts.append(target)
            command = " \\\n  ".join(parts)

            wait_s = option.candidate.wait.eff_at(
                plan.config.discount_strength, plan.config.wait_quantile)
            comment = (
                f"{lane_id}: {option.candidate.shape.name} "
                f"({option.candidate.gpus} dev, {option.candidate.walltime_s / 3600:g} h) "
                f"x{option.links_per_lane} link(s) {option.chain}; "
                f"begin +{option.begin_s / 3600:.1f} h, queue {wait_s / 3600:.1f} h "
                f"({plan.config.wait_quantile}), productive from "
                f"{(option.begin_s + wait_s + option.candidate.startup_s) / 3600:.1f} h; "
                f"accumulation {accum}"
                + (f". Stop the chain early with: touch "
                   f"logs/chain-{lane_id}.stop" if chained else ""))
            out.append(Submission(
                site=option.site, lane_id=lane_id, order=option.begin_s,
                begin_s=option.begin_s, args_verbatim=option.candidate.shape.args,
                command=command, comment=comment))
    return out


def _balance_factor(plan: Plan, calibration) -> float:
    """blocks/round balanced / blocks/round unbalanced, for the planned membership.

    Reported rather than quoted from the design note: the 2.33x in the brief is the
    two-full-node figure, and at one device per site the same arithmetic gives 3.0x.
    """
    members = [make_member(f"{o.site}-l{k}", o.candidate)
               for o in plan.selection for k in range(o.lanes)]
    if len(members) < 2:
        return 0.0
    kwargs = dict(inner_steps=plan.config.inner_steps, calibration=calibration,
                  balance_max=plan.config.balance_max)
    off = round_cost(members, balance=False, **kwargs)
    on = round_cost(members, balance=True, **kwargs)
    return on.blocks / off.blocks if off.blocks > 0 else 0.0


def balance_crosscheck(plan: Plan, calibration, registry_geometries: dict) -> list[str]:
    """What run_train.sh's own PWW_BALANCE would have derived, and whether it agrees.

    It keys PWW_TPUT_<SITE> on the site name alone, so at any geometry other than the
    one the registry was calibrated at it balances with the wrong step time. The
    disagreement is the finding; printing both is how a reader sees it.
    """
    lines: list[str] = []
    planned = accums_for(plan, calibration)
    if not planned:
        return lines
    sites = sorted(planned)
    balanced = any(v > 1 for v in planned.values())
    lines.append("# equivalent shell derivation (for comparison, NOT emitted):")
    lines.append(f"#   PWW_BALANCE=1 PWW_BALANCE_PEERS=\"{' '.join(sites)}\"")
    if not balanced and plan.config.balance == "auto":
        # Two different reasons produce accum 1 and they must not be confused, so the
        # reason is READ OFF THE TIMELINE rather than asserted. The factor is computed
        # too: it is blocks/round balanced over blocks/round unbalanced for THIS
        # membership and geometry, which at one device per site is 3.0x, not 2.3x.
        factor = _balance_factor(plan, calibration)
        rate = f"{factor:.2g}x" if factor else "some multiple"
        if plan.timeline.darl_exhausted_s is not None:
            lines.append(f"#   the planner chose NOT to balance (--balance auto): the "
                         f"CORPUS binds -- it runs")
            lines.append(f"#   out at {plan.timeline.darl_exhausted_s / 3600:.1f} h -- so "
                         f"balancing would spend it {rate} faster for the same")
            lines.append("#   tokens, i.e. it would buy fewer federated merges and "
                         "nothing else.")
        else:
            lines.append("#   the planner chose NOT to balance (--balance auto): "
                         "WALLTIME binds and there is")
            lines.append(f"#   corpus left at the end, but balancing here would itself "
                         f"exhaust it ({rate} the")
            lines.append("#   burn rate), which is not the free-tokens case the flag "
                         "turns on for.")
    elif not balanced and plan.config.balance == "on":
        # --balance on DID apply; there was simply nothing for it to do. Saying
        # "balancing is OFF because --balance on says so" is self-contradicting and
        # sends the reader looking for a bug in the flag.
        if sum(o.lanes for o in plan.selection) < 2:
            lines.append("#   --balance on is in force but IDLE: this plan has a single "
                         "member, and")
            lines.append("#   accumulation equalises the inner phase BETWEEN sites at "
                         "the barrier. With no")
            lines.append("#   peer to catch up to there is nothing to equalise, so "
                         "rounds.balance_accums")
            lines.append("#   floors every member at 1x (`len(members) < 2`).")
        else:
            lines.append("#   --balance on is in force but every member already lands "
                         "at 1x: the step")
            lines.append("#   times are close enough that int(slowest/step + 0.5) "
                         "rounds to 1.")
    elif not balanced:
        lines.append(f"#   balancing is OFF because --balance {plan.config.balance} says "
                     f"so, not because of a budget.")
    # run_train.sh reads the SITE-level entry, i.e. the reference geometry, whatever
    # geometry the job actually asked for.
    ref: dict[str, float] = {}
    for site in sites:
        cells = registry_geometries.get(site) or {}
        if not cells:
            continue
        # step_s is batch/tput, and this registry is the RAW file -- admission's
        # positive-throughput guard (rounds.site_overhead_s) runs on the planning
        # path, not on this one, so a single zeroed cell reaching this division
        # aborted the whole text report AFTER the plan had been computed: exit 1,
        # nothing on stdout, no plan and no diagnosis, while `show` and `--json` on
        # the same inputs printed the proper refusal. A cell that cannot yield a step
        # time is skipped here and named below.
        usable = {g: c for g, c in cells.items() if c.tput_seq_s > 0 and c.batch_seq > 0}
        if not usable:
            continue
        gpus = max(usable)  # the site-level PWW_BATCH_/PWW_TPUT_ pair is the full-site one
        ref[site] = usable[gpus].step_s
    if len(ref) < len(sites):
        # Say so rather than dropping the comparison silently: its absence is not
        # evidence that the planner and the shell agree.
        lines.append("#   (no shell comparison for "
                     + ", ".join(s for s in sites if s not in ref)
                     + ": the registry has no usable PWW_TPUT_/PWW_BATCH_ pair for it)")
    if len(ref) == len(sites) and ref:
        slowest = max(ref.values())
        shell = {s: max(1, min(plan.config.balance_max, int(slowest / ref[s] + 0.5)))
                 for s in sites}
        chosen_gpus = {o.site: o.candidate.gpus for o in plan.selection}
        for site in sites:
            agree = shell.get(site) == planned[site]
            note = ""
            if not agree and not balanced:
                note = ("   <-- differs because the planner turned balancing OFF, not "
                        "because of geometry; PWW_BALANCE=1 would set this to "
                        f"{shell.get(site)}x and burn the corpus faster")
            elif not agree:
                ref_gpus = max((registry_geometries.get(site) or {1: None}))
                note = ("   <-- DISAGREE: run_train.sh keys the registry on the bare "
                        f"site name, so on this {chosen_gpus.get(site)}-device shape it "
                        f"balances with the {ref_gpus}-device reference step time. The "
                        "explicit PWW_GRAD_ACCUM above wins.")
            lines.append(f"#   {site}: planner {planned[site]}x vs shell "
                         f"{shell.get(site)}x{note}")
    return lines
