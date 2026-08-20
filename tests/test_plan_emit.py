"""What actually gets pasted, and what the chain link does with it.

    python3 tests/test_plan_emit.py

The emitter and scripts/titan/job_chain_link.sh had no tests at all, and that is where
the two worst defects in the planner lived: a chain link that deleted its own lane's
checkpoint on every link, and an aggregator command that brought up the CIFAR strategy
on the wrong ports. Neither is visible in a plan's numbers -- both produce a report
that reads correctly and a run that does not exist -- so they can only be caught here.

THREE INVARIANTS THIS FILE EXISTS TO PIN:

  * the shape flags of every emitted line are a SUBSTRING-EXACT copy of a probe row's
    own `args`, so a wait measured at one walltime can never be quoted for another;
  * PWW_FRESH_RUN/PWW_FRESH_DELETE appear at BOTH ends of the stack or at neither, and
    never on a successor link;
  * the accumulation exported is the accumulation the plan was priced at.

The shell checks run the real script with a stub `sbatch` on PATH. They are the only
way to observe --export=ALL, which is a property of the command line and not of any
Python object.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASSED, FAILED = [], []
CHECK_TIMEOUT_S = 120


def check(name: str):
    def decorator(fn):
        def on_timeout(signum, frame):
            raise TimeoutError(f"exceeded {CHECK_TIMEOUT_S}s")

        previous = signal.signal(signal.SIGALRM, on_timeout)
        signal.alarm(CHECK_TIMEOUT_S)
        try:
            fn()
            PASSED.append(name)
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        return fn

    return decorator


from pww.plan import (  # noqa: E402
    DEFAULT_CALIBRATION as CAL,
    DarlState,
    Geometry,
    PlanConfig,
    PlannerInputs,
    Shape,
    SiteInput,
    WaitEstimate,
    make_plan,
)
from pww.plan import emit as emit_mod  # noqa: E402
from pww.plan import inputs as io  # noqa: E402
from pww.plan.rounds import make_member, plan_accums  # noqa: E402

HOUR = 3600.0
# The live coordinator today: 822 of 2692 blocks left, i.e. a run 69% through its
# epoch. FRESH is a coordinator that has not started. Both are `observed=True`
# because both stand for an answered GET /status -- that flag, not the
# unassigned/num_blocks comparison, is what licenses the destructive fresh flags.
MID_RUN = DarlState(num_blocks=2692, committed=1870, leased=0, unassigned=822,
                    source="http://145.38.206.143:29510", observed=True)
FRESH = DarlState(num_blocks=2692, committed=0, leased=0, unassigned=2692,
                  source="http://145.38.206.143:29510", observed=True)
# The two fallbacks, built exactly as adapter.py builds them: no committed count to
# subtract, so unassigned == num_blocks even though the corpus may be nearly gone.
PINNED = DarlState(num_blocks=822, committed=0, leased=0, unassigned=822,
                   source="--blocks")
ASSUMED = DarlState(num_blocks=2692, committed=0, leased=0, unassigned=2692,
                    source="assumed: whole corpus")


def shape(site, partition, gpus, hours, account=None) -> Shape:
    h, m = int(hours), int(round((hours - int(hours)) * 60))
    args = (f"-A {account} " if account else "") + \
        f"-p {partition} -N 1 --gpus-per-node {gpus} -t {h}:{m:02d}:00"
    return Shape(f"{site}_{gpus}g_{hours:g}h", io.parse_shape_args(site, args), args)


def site(name, partition, gpus, curve, tput, batch, startup_s, account=None) -> SiteInput:
    shapes, waits = [], {}
    for hours, wait_h in curve:
        sh = shape(name, partition, gpus, hours, account)
        shapes.append(sh)
        waits[sh.name] = WaitEstimate(
            p50_raw_s=wait_h * HOUR, p90_raw_s=wait_h * HOUR,
            p50_eff_s=wait_h * HOUR, p90_eff_s=wait_h * HOUR,
            samples=3, probe_age_s=60.0)
    return SiteInput(site=name, shapes=tuple(shapes), waits=waits,
                     geometries={gpus: Geometry(name, gpus, tput, batch)},
                     startup_s=startup_s)


def snellius(curve=((8, 0.0),), **kw):
    return site("snellius", "gpu_h100", 4, curve, 89.8, 32, 300.0, **kw)


def lumi(curve=((8, 0.0),), **kw):
    return site("lumi", "standard-g", 8, curve, 38.2, 64, 600.0,
                account="project_462000226", **kw)


def plan_of(sites, darl=MID_RUN, **cfg_kw):
    base = dict(horizon_s=24 * HOUR, num_rounds=1_000_000, lanes_max=1,
                assume_overhead=True)
    base.update(cfg_kw)
    return make_plan(PlannerInputs(sites=tuple(sites), calibration=CAL, darl=darl),
                     PlanConfig(**base))


def emit(plan, darl=MID_RUN, **cfg_kw):
    base = dict(root=str(ROOT))
    base.update(cfg_kw)
    return emit_mod.emit(plan, CAL, emit_mod.EmitConfig(**base), darl=darl)


def central(subs):
    return next(s for s in subs if s.site == "(central)")


def sites_only(subs):
    return [s for s in subs if s.site != "(central)"]


def _chain_args_of(command):
    """The PWW_CHAIN_ARGS list as job_chain_link.sh will actually see it.

    Parsed the way the shell does it -- pull the value out of the --export list, then
    split on '|' with IFS -- so the test asserts against the argv the SUCCESSOR is
    submitted with, not against the text of the first sbatch line.
    """
    m = re.search(r"PWW_CHAIN_ARGS=([^,\"]*)", command)
    assert m, f"no PWW_CHAIN_ARGS in {command}"
    return m.group(1).split("|")


# --------------------------------------------------------------------------
# the verbatim-args invariant
# --------------------------------------------------------------------------


@check("every emitted shape flag string is a substring of some probe's args")
def _():
    """The one structural guarantee that a wait measured at one walltime cannot be
    quoted for another. Re-rendering the flags from the parsed ShapeKey would look
    identical and lose it, which is exactly what the upstream planner does."""
    sites = (snellius(((1, 9.0), (8, 12.0))), lumi(((1, 0.0), (8, 0.5))))
    plan = plan_of(sites)
    probed = {sh.args for s in sites for sh in s.shapes}
    for sub in sites_only(emit(plan)):
        assert sub.args_verbatim in probed, sub.args_verbatim
        assert sub.args_verbatim in sub.command, sub.command
        # and it is the args of the shape the plan actually chose
        option = next(o for o in plan.selection if o.site == sub.site)
        assert sub.args_verbatim == option.candidate.shape.args


@check("a chained lane still asks for the site's own cores, memory and log paths")
def _():
    """REGRESSION. A chained lane's batch script is job_chain_link.sh, which carries no
    #SBATCH header, so everything the site job script's header would have supplied has
    to travel on the command line. Without it --cpus-per-task defaults to 1 -- the
    throughput in configs/site_throughput.env was measured at 64 -- and the job's
    output goes to slurm-%j.out instead of the logs/ path the planner tells you to
    read."""
    plan = plan_of((snellius(((8, 0.0),)),), horizon_s=24 * HOUR)
    option = plan.selection[0]
    assert option.links_per_lane > 1, "this scenario is supposed to chain"
    sub = sites_only(emit(plan))[0]
    assert "scripts/titan/job_chain_link.sh" in sub.command
    header = emit_mod.site_resource_flags(
        ROOT / "scripts/snellius/job_titan_diloco.sh", option.candidate.shape.args)
    assert "--cpus-per-task=64" in header, header
    for flag in header:
        assert flag in sub.command, (flag, sub.command)
    # ... and never at the cost of the shape: the verbatim block is still intact and
    # nothing overrides -t, which is what the wait was measured at.
    assert sub.args_verbatim in sub.command
    assert sub.command.count("-t ") == 1, sub.command
    # THE HALF THIS TEST USED TO MISS. sub.command is link 1 only. Links 2..N are
    # submitted by job_chain_link.sh as `sbatch "${CHAIN_ARGS[@]}"`, built by
    # splitting PWW_CHAIN_ARGS on '|' -- so a flag that is on the first line and not
    # in PWW_CHAIN_ARGS is a flag that 5 of these 6 links do not get. The script also
    # `env -u`s SLURM_CPUS_PER_TASK/SLURM_NTASKS_PER_NODE/SLURM_MEM_PER_NODE for the
    # sbatch call, so there is no inheritance path to fall back on.
    chain_args = _chain_args_of(sub.command)
    for flag in header:
        assert flag in chain_args, (flag, chain_args)
    # PWW_CHAIN_ARGS is passed through to each successor unchanged, so links 3..N
    # inherit it too; and it must still carry the verbatim shape, not just resources.
    for tok in option.candidate.shape.args.split():
        assert tok in chain_args, (tok, chain_args)


# --------------------------------------------------------------------------
# fresh flags: both ends or neither, first link only
# --------------------------------------------------------------------------


@check("no fresh flags are emitted against a coordinator that is mid-epoch")
def _():
    """REGRESSION. The plan is priced against `822 of 2692 blocks left`, and DARL
    exhaustion is what decides the recommendation. Telling the operator to wipe both
    sites' checkpoints while leaving the coordinator's lease table at 822 serves
    neither intent -- and adding PWW_FRESH_RUN=1 to the aggregator instead would reset
    the corpus to 2692 and make every number in the plan wrong by 3.3x."""
    subs = emit(plan_of((snellius(), lumi())), darl=MID_RUN)
    for sub in subs:
        assert "PWW_FRESH_RUN" not in sub.command, sub.command
        assert "PWW_FRESH_DELETE" not in sub.command, sub.command
    assert "PART WAY THROUGH" in central(subs).comment


@check("a genuinely fresh run sets the fresh flags at BOTH ends, not just the sites")
def _():
    """PWW_FRESH_RUN on the sites alone clears the local checkpoints and leaves the
    coordinator's lease table and the global model -- a half-reset that produces a run
    that is neither the old one nor a new one. start_central_services.sh turns
    PWW_FRESH_RUN into DARL_FRESH=1 plus a discarded global model, so it belongs on
    the aggregator line too."""
    subs = emit(plan_of((snellius(), lumi()), darl=FRESH), darl=FRESH)
    assert "PWW_FRESH_RUN=1" in central(subs).command, central(subs).command
    for sub in sites_only(subs):
        assert "PWW_FRESH_RUN=1" in sub.command and "PWW_FRESH_DELETE=1" in sub.command
    # --no-fresh still wins: it is an explicit instruction.
    quiet = emit(plan_of((snellius(),), darl=FRESH), darl=FRESH, fresh=False)
    assert all("PWW_FRESH_RUN" not in s.command for s in quiet)


@check("a corpus that was not READ from a coordinator never licenses the fresh flags")
def _():
    """REGRESSION. `unassigned < num_blocks` is not the mid-run test it looks like.
    --blocks N and the whole-corpus fallback both build unassigned == num_blocks --
    they have no committed count to subtract, not because the corpus is untouched --
    so that comparison called them fresh and emitted PWW_FRESH_RUN=1 /
    PWW_FRESH_DELETE=1 against a coordinator 69% through its epoch. --blocks 822 is
    exactly what scripts/plan_campaign.sh tells the operator to pass when no DARL
    token is found, so this was the DEFAULT tokenless path. Pasting it reset the lease
    table to the full 2692, discarded the global model and its Nesterov momentum, and
    rm -rf'd both lanes' DCP checkpoints -- while the plan alongside still quoted the
    822 blocks its own commands destroyed."""
    for darl in (PINNED, ASSUMED):
        subs = emit(plan_of((snellius(), lumi()), darl=darl), darl=darl)
        for sub in subs:
            assert "PWW_FRESH_RUN=1" not in sub.command, (darl.source, sub.command)
            assert "PWW_FRESH_DELETE=1" not in sub.command, (darl.source, sub.command)
        # and it says so, naming the provenance -- silence here reads as "fresh run".
        note = central(subs).comment
        assert "NOT read from a coordinator" in note, note
        assert darl.source in note, (darl.source, note)
    # The flags are not disabled in general: an ANSWERED /status that says the epoch
    # is untouched still gets them, at both ends.
    fresh_subs = emit(plan_of((snellius(), lumi()), darl=FRESH), darl=FRESH)
    assert "PWW_FRESH_RUN=1" in central(fresh_subs).command
    # The distinction is provenance, not the numbers: ASSUMED and FRESH carry byte
    # for byte the same block counts and differ only in whether anything answered.
    assert (ASSUMED.num_blocks, ASSUMED.unassigned) == (FRESH.num_blocks, FRESH.unassigned)
    assert FRESH.fresh_epoch and not ASSUMED.fresh_epoch and not PINNED.fresh_epoch


@check("REPLICA is never emitted without a matching PWW_DUMP")
def _():
    """--replica separates the DARL cluster id, the Flower client id and the delta-blob
    key, but NOT dump_folder: two lanes would still share one DCP checkpoint, one
    blob-staging directory and one tb directory.

    The multi-lane option is BUILT here rather than asked for with lanes_max=2.
    lanes_max is a ceiling, not a forcing knob, and the search never picks two lanes
    at one site because two lanes at one site do not federate -- so the version of
    this check that planned with lanes_max=2 got lanes=1 everywhere, compared
    False == False, and never entered the branch it exists to guard. That left the
    whole multi-lane emission path -- the exact case the shared-checkpoint bug was
    about -- unexecuted by the suite.
    """
    plan = plan_of((snellius(), lumi()), darl=FRESH)
    twin = dataclasses.replace(plan.selection[0], lanes=2)
    plan = dataclasses.replace(plan, selection=(twin,) + tuple(plan.selection[1:]))
    subs = sites_only(emit(plan, darl=FRESH))

    twinned = [s for s in subs if s.site == twin.site]
    assert len(twinned) == 2, [s.lane_id for s in subs]   # the branch really ran
    dumps, replicas = set(), set()
    for sub in subs:
        assert ("REPLICA=" in sub.command) == ("PWW_DUMP=" in sub.command), sub.command
        if "REPLICA=" in sub.command:
            replica = sub.command.split("REPLICA=")[1].split(",")[0]
            dump = sub.command.split("PWW_DUMP=")[1].split(",")[0]
            assert dump.endswith(f"{sub.site}-{replica}"), (dump, replica)
            dumps.add(dump)
            replicas.add(replica)
    # The point of the pairing: two lanes at one site must not share one checkpoint.
    assert len(dumps) == 2 and len(replicas) == 2, (dumps, replicas)
    # A single-lane site is still emitted WITHOUT either -- REPLICA changes the DARL
    # cluster id, so setting it on a solo lane renames the cluster for no reason.
    solo = [s for s in subs if s.site != twin.site]
    assert solo and all("REPLICA=" not in s.command for s in solo)
    # ... and the job names are distinct too, or squeue shows two identical rows.
    assert len({s.command.split("-J ")[1].split()[0] for s in twinned}) == 2


@check("WANDB_RUN_NAME is never set on a multi-link lane")
def _():
    """run_train.sh appends the Slurm job id only when it derives the name itself, so
    an exported name gives every link of a chain the same display name."""
    plan = plan_of((snellius(((8, 0.0),)),), horizon_s=24 * HOUR)
    assert plan.selection[0].links_per_lane > 1
    sub = sites_only(emit(plan, wandb_project="pww"))[0]
    assert "WANDB_PROJECT=pww" in sub.command
    assert "WANDB_RUN_NAME" not in sub.command, sub.command
    # a single-job lane does get one, because there is nothing to disambiguate
    single = plan_of((snellius(((24, 0.0),)),), horizon_s=24 * HOUR)
    assert single.selection[0].links_per_lane == 1
    assert "WANDB_RUN_NAME=" in sites_only(emit(single, wandb_project="pww"))[0].command


# --------------------------------------------------------------------------
# the aggregator line
# --------------------------------------------------------------------------


@check("the aggregator line names its config and both ports, not just NUM_ROUNDS")
def _():
    """REGRESSION. start_central_services.sh falls back to
    configs/central_aggregator.yaml when AGGREGATOR_CONFIG is unset and no launch.env
    exists -- that is the CIFAR/ResNet file: min-clients 2, so every solo round this
    planner counts is impossible; server-momentum 0.0, which is FedAvg not FedMom; and
    round-timeout 300 s, which the two-site period of ~347 s exceeds. It also reads
    DARL_PORT/FLOWER_PORT, NOT the PWW_-prefixed names the job scripts read, so a
    non-default port pair has to be spelled both ways."""
    plan = plan_of((snellius(), lumi()))
    subs = emit(plan, darl_port=29520, flower_port=29521)
    line = central(subs).command
    assert "AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml" in line, line
    assert "DARL_PORT=29520" in line and "FLOWER_PORT=29521" in line, line
    assert f"NUM_ROUNDS={plan.recommended_num_rounds}" in line, line
    # and the sites are told the same pair under the names THEY read
    for sub in sites_only(subs):
        assert "PWW_DARL_PORT=29520" in sub.command and "PWW_FLOWER_PORT=29521" in sub.command
    # the yaml the line names must actually exist and be the titan one
    text = (ROOT / "configs/central_aggregator_titan.yaml").read_text()
    assert "min-clients: 1" in text and "round-timeout: 1800.0" in text


@check("a non-default port pair also carries the output dir that actually separates arms")
def _():
    """REGRESSION. The ports do NOT separate two arms -- PWW_OUTPUT_DIR does.
    start_central_services.sh keys STATE_DIR, and with it the pid files, the token,
    launch.env, space.env, the DARL lease table and the global model, on
    ${PWW_OUTPUT_DIR}/central and never on the port. So an aggregator line carrying
    only DARL_PORT/FLOWER_PORT is, on a machine already running the default stack, a
    silent no-op: it finds the default stack's live darl.pid/flower.pid, prints
    'already running' and 'successfully launched', and exits 0 having started nothing
    on the new ports -- and both site jobs then spend their whole queue wait (7.7 h
    for Snellius here) connecting to a Flower server that is not there. With the
    default stack down it is worse: the two arms share one state directory. The
    repo's own second-arm runbook (RUNBOOK_DCLT.md) passes PWW_OUTPUT_DIR alongside
    the ports; the planner emitted two of the four.
    """
    plan = plan_of((snellius(), lumi()))
    # Given a dir, it is on the line -- and before the script that reads it.
    line = central(emit(plan, darl_port=29530, flower_port=29531,
                        output_dir="/data/x/runs-dclt")).command
    assert "PWW_OUTPUT_DIR=/data/x/runs-dclt" in line, line
    assert line.index("PWW_OUTPUT_DIR=") < line.index("start_central_services.sh"), line
    assert "WARNING: DARL_PORT" not in line, line

    # Not given one, the no-op is stated rather than shipped silently.
    bare = central(emit(plan, darl_port=29530, flower_port=29531))
    assert "PWW_OUTPUT_DIR=" not in bare.command, bare.command
    assert "WARNING: DARL_PORT/FLOWER_PORT are non-default" in bare.comment, bare.comment
    assert "--output-dir" in bare.comment, bare.comment

    # The default pair is the normal case and must stay quiet.
    plain = central(emit(plan))
    assert "WARNING: DARL_PORT" not in plain.comment, plain.comment
    assert "PWW_OUTPUT_DIR=" not in plain.command, plain.command


# --------------------------------------------------------------------------
# accumulation
# --------------------------------------------------------------------------


@check("the emitted grad accum matches the balance law for the planned membership")
def _():
    """And, more importantly, matches what the SIMULATOR debited: they are computed by
    the same helper for exactly this reason."""
    # alpha 0: federated merges are the only currency, so both sites are selected --
    # which is the membership the balance law is about.
    plan = plan_of((snellius(), lumi()), balance="on", alpha=0.0)
    assert {o.site for o in plan.selection} == {"snellius", "lumi"}, plan.describe()
    accums = emit_mod.accums_for(plan, CAL)
    members = [make_member(f"{o.site}-l{k}", o.candidate)
               for o in plan.selection for k in range(o.lanes)]
    law = plan_accums(members, balance=True, cap=plan.config.balance_max)
    for member in members:
        assert accums[member.site] == law[member.lane_id]
    # snellius step 0.356347 s, lumi 1.675393 s -> int(1.675393/0.356347 + 0.5) = 5
    assert accums == {"snellius": 5, "lumi": 1}, accums
    for ledger in plan.timeline.ledgers:
        assert ledger.accums == (accums[ledger.site],), (ledger.site, ledger.accums)
    for sub in sites_only(emit(plan)):
        assert f"PWW_GRAD_ACCUM={accums[sub.site]}" in sub.command, sub.command


@check("the balance cross-check states the budget that actually bound, not a fixed one")
def _():
    """It used to print 'the corpus, not walltime, is what binds here' unconditionally,
    with a hard-coded 2.3x -- on plans whose corpus was untouched. Both the verdict and
    the factor are now read off the plan."""
    geometries = {"snellius": {4: Geometry("snellius", 4, 89.8, 32)},
                  "lumi": {8: Geometry("lumi", 8, 38.2, 64)}}
    capped = plan_of((snellius(), lumi()), balance="auto", alpha=0.0, darl=MID_RUN)
    lines = "\n".join(emit_mod.balance_crosscheck(capped, CAL, geometries))
    assert capped.timeline.darl_exhausted_s is not None
    assert "CORPUS binds" in lines, lines
    # 21.875 blocks/round balanced over 9.375 unbalanced = 2.3333...
    assert "2.3x faster" in lines, lines

    # --balance ON with a single member is not "OFF because --balance on says so".
    # Accumulation equalises the barrier BETWEEN sites, so a one-member round has no
    # peer to catch up to and balance_accums floors everyone at 1x. The old message
    # contradicted itself and sent the reader hunting for a bug in the flag.
    solo = plan_of((snellius(),), balance="on", darl=MID_RUN)
    assert sum(o.lanes for o in solo.selection) == 1, solo.describe()
    solo_lines = "\n".join(emit_mod.balance_crosscheck(solo, CAL, geometries))
    assert "balancing is OFF because --balance on says so" not in solo_lines, solo_lines
    assert "IDLE" in solo_lines and "single member" in solo_lines, solo_lines
    # --balance off still says the plain thing, because there it IS the reason.
    off = plan_of((snellius(), lumi()), balance="off", darl=MID_RUN)
    off_lines = "\n".join(emit_mod.balance_crosscheck(off, CAL, geometries))
    assert "balancing is OFF because --balance off says so" in off_lines, off_lines


@check("a registry cell with no usable step time is skipped, not divided by")
def _():
    """REGRESSION. This argument is the RAW registry: the positive-throughput guard
    that admission applies (rounds.site_overhead_s) sits on the PLANNING path and not
    on this one, so a single zeroed PWW_TPUT_ reached `batch_seq / tput_seq_s` here and
    took the whole text report down with a ZeroDivisionError -- AFTER the plan had been
    computed. What the operator got was exit 1 and an empty stdout: no plan, no
    exclusion and no diagnosis, while `show` and `--json` on the same inputs printed
    the proper refusal by name."""
    plan = plan_of((snellius(), lumi()), balance="on", alpha=0.0)
    assert {o.site for o in plan.selection} == {"snellius", "lumi"}, plan.describe()

    # One cell of two zeroed: the site still has a step time, so the comparison the
    # reader is here for is made rather than lost to the neighbouring typo.
    mixed = {"snellius": {4: Geometry("snellius", 4, 0.0, 32),
                          1: Geometry("snellius", 1, 24.4, 8)},
             "lumi": {8: Geometry("lumi", 8, 38.2, 64)}}
    lines = "\n".join(emit_mod.balance_crosscheck(plan, CAL, mixed))
    assert "snellius: planner" in lines and "lumi: planner" in lines, lines

    # Every cell unusable, by either half -- batch 0 divides one line further down,
    # at `slowest / ref[s]`. The site is NAMED, because a missing comparison reads as
    # agreement between the planner and the shell, which is what it cannot say.
    for dead in ({4: Geometry("snellius", 4, 0.0, 32)},
                 {4: Geometry("snellius", 4, 89.8, 0)}):
        lines = "\n".join(emit_mod.balance_crosscheck(
            plan, CAL, {"snellius": dead, "lumi": {8: Geometry("lumi", 8, 38.2, 64)}}))
        assert "no shell comparison for snellius" in lines, lines
        assert "snellius: planner" not in lines, lines


# --------------------------------------------------------------------------
# scripts/titan/job_chain_link.sh -- observable only by running it
# --------------------------------------------------------------------------


def _sandbox(tmp: Path) -> Path:
    """A Slurm-shaped sandbox: the script staged as a node spool copy (which is what
    $0 and BASH_SOURCE really are inside a job), a stub `sbatch` on PATH, and a
    checkout elsewhere holding a stub site job script that prints what it inherited."""
    (tmp / "bin").mkdir()
    # The stub reports the SLURM_* inputs too: sbatch reads them as if they were
    # command-line options, so "what did the successor inherit" is only answerable by
    # looking at the environment the real sbatch would have been called with.
    (tmp / "bin" / "sbatch").write_text(
        '#!/bin/bash\necho "SBATCH_ARGS: $*"\n'
        'for v in SLURM_CPUS_PER_TASK SLURM_NTASKS_PER_NODE SLURM_MEM_PER_NODE \\\n'
        '         SLURM_JOB_NAME SLURM_NNODES SLURM_GPUS_PER_NODE SLURM_EXPORT_ENV; do\n'
        '    echo "SBATCH_ENV: ${v}=[${!v:-<unset>}]"\n'
        'done\n')
    (tmp / "bin" / "sbatch").chmod(0o755)
    spool = tmp / "spool" / "job123"
    spool.mkdir(parents=True)
    shutil.copy(ROOT / "scripts/titan/job_chain_link.sh", spool / "slurm_script")
    root = tmp / "checkout"
    (root / "scripts" / "titan").mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "env.sh").write_text("# stub\n")
    shutil.copy(ROOT / "scripts/titan/job_chain_link.sh",
                root / "scripts" / "titan" / "job_chain_link.sh")
    (root / "scripts" / "site.sh").write_text(
        '#!/bin/bash\necho "CHILD PWW_FRESH_RUN=[${PWW_FRESH_RUN:-}]"\n'
        'echo "CHILD PWW_FRESH_DELETE=[${PWW_FRESH_DELETE:-}]"\n'
        'echo "CHILD PWW_ROOT=[${PWW_ROOT:-}]"\n')
    (root / "scripts" / "site.sh").chmod(0o755)
    return root


def run_chain_link(env_extra: dict, *, submit_dir: bool = True) -> tuple[int, str, Path]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = _sandbox(tmp)
        env = dict(os.environ)
        env.pop("PWW_ROOT", None)
        env["PATH"] = f"{tmp / 'bin'}:{env['PATH']}"
        env["SLURM_SUBMIT_DIR"] = str(root) if submit_dir else ""
        env.update({k: str(v) for k, v in env_extra.items()})
        proc = subprocess.run(["bash", str(tmp / "spool" / "job123" / "slurm_script")],
                              cwd=str(tmp / "spool" / "job123"), env=env,
                              capture_output=True, text=True, timeout=60)
        return proc.returncode, proc.stdout + proc.stderr, root


def chain_env() -> dict:
    return {
        "PWW_CHAIN_LINKS": "3",
        "PWW_CHAIN_LANE": "lumi-l0",
        "PWW_CHAIN_SCRIPT": "scripts/site.sh",
        "PWW_CHAIN_ARGS": "-A|proj|-p|standard-g|-N|1|--gpus-per-node|8|-t|1:00:00",
        "PWW_CHAIN_STOP": "/nonexistent",
        "DARL_TOKEN": "secret",
    }


@check("a chain link's own geometry never leaks into its successor's sbatch request")
def _():
    """REGRESSION GUARD, previously untested: deleting the `env -u SLURM_*` scrub left
    the whole suite green. sbatch reads SLURM_* environment variables as if they were
    command-line options, so without the scrub a link running at, say, 8 nodes would
    silently request 8 nodes for its successor no matter what PWW_CHAIN_ARGS says --
    the shape the queue wait was measured at would not be the shape submitted, which is
    the one guarantee this planner is built around. The scrub is also why the site's
    --cpus-per-task/--mem MUST travel inside PWW_CHAIN_ARGS: there is deliberately no
    inheritance path left for them."""
    leaky = {
        "SLURM_CPUS_PER_TASK": "56", "SLURM_NTASKS_PER_NODE": "8",
        "SLURM_MEM_PER_NODE": "491520", "SLURM_NNODES": "8",
        "SLURM_GPUS_PER_NODE": "8", "SLURM_JOB_NAME": "pww-lumi-titan",
        "SLURM_EXPORT_ENV": "ALL",
    }
    code, out, _root = run_chain_link(dict(chain_env(), **leaky))
    assert code == 0, out
    seen = dict(line.split("SBATCH_ENV: ")[1].split("=[", 1)
                for line in out.splitlines() if line.startswith("SBATCH_ENV:"))
    for var in leaky:
        assert seen[var] == "<unset>]", (var, seen[var], out)
    # ... while the shape that SHOULD decide the successor is passed explicitly.
    line = next(l for l in out.splitlines() if l.startswith("SBATCH_ARGS:"))
    assert "-N 1" in line and "--gpus-per-node 8" in line and "-t 1:00:00" in line, line


@check("the emitted PWW_CHAIN_ARGS puts the site's resources on EVERY link's argv")
def _():
    """REGRESSION, and the half of the previous repair that was left undone: the site's
    #SBATCH resource flags were appended to the FIRST sbatch line only, and the check
    written for it read only that line. Links 2..N are submitted from here as
    `sbatch "${CHAIN_ARGS[@]}"`, and their batch script is job_chain_link.sh, which
    carries no #SBATCH header of its own -- so 12 of the 14 jobs the recommended plan
    submits ran at Slurm's default --cpus-per-task=1, which is not the CPU shape the
    89.8 / 38.2 seq/s in configs/site_throughput.env were measured at and is what the
    whole plan is priced against, and wrote their logs to slurm-%j.out instead of the
    logs/ path the report tells the operator to read. `env -u SLURM_CPUS_PER_TASK` in
    resubmit() closes the only accidental path they could have arrived by.

    Driven through the real script, from the planner's own emitted value, because "the
    string is right" and "the argv is right" are two different claims and it was the
    first one that was checked."""
    subs = [s for s in sites_only(emit(plan_of((snellius(), lumi()))))
            if "PWW_CHAIN_ARGS=" in s.command]
    assert subs, "expected the default plan to chain"
    for sub in subs:
        chain_args = "|".join(_chain_args_of(sub.command))
        code, out, _root = run_chain_link(dict(chain_env(), PWW_CHAIN_LANE=sub.lane_id,
                                               PWW_CHAIN_ARGS=chain_args))
        assert code == 0, out
        argv = next(l for l in out.splitlines() if l.startswith("SBATCH_ARGS:"))
        header = emit_mod.site_resource_flags(
            ROOT / f"scripts/{sub.site}/job_titan_diloco.sh", sub.args_verbatim)
        assert header, sub.site
        for flag in header:
            assert f" {flag}" in argv, (flag, argv)
        # ... and handed on unchanged, so link 3 is submitted with them too.
        assert f"PWW_CHAIN_ARGS={chain_args}," in argv, argv


@check("the --export list is one shell word and begins with ALL")
def _():
    """Both invariants are load-bearing and neither was asserted anywhere.

    ALL first: the successor and the job script need the submitting environment
    (PATH, module state, PWW_ROOT), and Slurm applies the list left to right, so a
    later K=V beats what ALL carried in -- which is exactly how job_chain_link.sh pins
    PWW_FRESH_RUN=0.

    Quoted as ONE word: Slurm splits the list on commas so a VALUE may contain spaces,
    but the shell splits on spaces first. Unquoted, `--export=...,PWW_CHAIN_ARGS=-p
    gpu_h100` loses everything after the first space and sbatch reads `gpu_h100` as a
    positional argument -- i.e. as the batch script. The quoting is also what lets
    $DARL_TOKEN expand at paste time, so the token never appears in a plan or a log.
    """
    subs = sites_only(emit(plan_of((snellius(), lumi()))))
    assert subs
    for sub in subs:
        export = re.search(r'--export="([^"]*)"', sub.command)
        assert export, sub.command          # quoted, and as a single word
        assert export.group(1).startswith("ALL,"), export.group(1)
        # the whole list is one shell word: no unquoted space inside it
        token = next(t for t in sub.command.split() if t.startswith('--export='))
        assert token.endswith('"') or '"' not in token, token
        # the token is never inlined -- only the shell variable is
        assert "DARL_TOKEN=$DARL_TOKEN" in sub.command, sub.command
    # A chained lane is the case that actually carries spaces, via PWW_CHAIN_ARGS'
    # '|' encoding; check the encoding really removed them from the value.
    chained = [s for s in subs if "PWW_CHAIN_ARGS=" in s.command]
    assert chained, "expected the default plan to chain"
    for sub in chained:
        value = re.search(r'PWW_CHAIN_ARGS=([^,"]*)', sub.command).group(1)
        assert " " not in value, value
        assert "|" in value, value


@check("a chain link does not hand its successor the fresh flags")
def _():
    """REGRESSION, and the worst defect the planner shipped. The successor's --export
    began with the literal ALL, so PWW_FRESH_RUN=1/PWW_FRESH_DELETE=1 from the FIRST
    link's own environment crossed into every later link, and run_train.sh:156-176 then
    `rm -rf`s the lane's DCP checkpoint at the start of each one. Weights survive
    (configure_fit re-seeds from the server) but the AdamW moments are zeroed and the
    300-step LR warmup restarts once per link -- so an 8-link chain trains nothing,
    while timeline.py charges the cold-join transient only on index 0."""
    code, out, _root = run_chain_link(
        dict(chain_env(), PWW_FRESH_RUN="1", PWW_FRESH_DELETE="1"))
    assert code == 0, out
    assert "SBATCH_ARGS:" in out, out
    line = next(l for l in out.splitlines() if l.startswith("SBATCH_ARGS:"))
    assert "PWW_FRESH_RUN=0" in line and "PWW_FRESH_DELETE=0" in line, line
    # ... and nothing re-enables them: the LAST assignment in the list is what Slurm
    # applies, so a `PWW_FRESH_RUN=1` after the pin would undo it.
    assert line.rindex("PWW_FRESH_RUN=0") > line.find("ALL"), line
    assert "PWW_FRESH_RUN=1" not in line, line
    # the link itself still runs, and still sees its own flags -- it IS the first link
    assert "CHILD PWW_FRESH_RUN=[1]" in out, out


@check("a chain link finds the checkout from SLURM_SUBMIT_DIR, not from BASH_SOURCE")
def _():
    """REGRESSION. Slurm stages the batch script as
    /var/spool/slurmd/job<id>/slurm_script, so `dirname ${BASH_SOURCE[0]}/../..`
    resolves to /var/spool: the link exited 2 before training AND before resubmitting,
    silently ending the whole lane. Every site job script already handles this with
    `: "${PWW_ROOT:=${SLURM_SUBMIT_DIR}}"`; this one did not."""
    code, out, root = run_chain_link(chain_env())
    assert code == 0, out
    assert f"CHILD PWW_ROOT=[{root}]" in out, out
    # ... and with NEITHER available it fails loudly, naming both, rather than
    # silently cd'ing into /var/spool and exiting 2 with "no such job script".
    code, out, _ = run_chain_link(chain_env(), submit_dir=False)
    assert code == 2, (code, out)
    assert "no env.sh under PWW_ROOT" in out and "SLURM_SUBMIT_DIR" in out, out


@check("a chain link submits the CHECKOUT's copy of itself, at begin now+T")
def _():
    """`sbatch "$0"` would submit the node-local spool copy: the file the operator can
    edit -- and the file a `touch logs/chain-<lane>.stop` is meant to govern -- is the
    one in the checkout."""
    code, out, root = run_chain_link(chain_env())
    assert code == 0, out
    line = next(l for l in out.splitlines() if l.startswith("SBATCH_ARGS:"))
    assert str(root / "scripts" / "titan" / "job_chain_link.sh") in line, line
    assert "PWW_CHAIN_LINKS=2" in line, line          # one fewer than this link's 3
    assert "--begin=now+60minutes" in line, line      # -t 1:00:00, lead 0
    # a `.stop` sentinel halts the chain without halting this link
    code, out, _ = run_chain_link(dict(chain_env(), PWW_CHAIN_STOP="/etc/hostname"))
    assert code == 0 and "SBATCH_ARGS:" not in out, out
    assert "stopping the chain here" in out, out


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


@check("preflight passes now that run_train.sh reads PWW_DUMP")
def _():
    """--dump is unreachable from any sbatch line (both job scripts call run_train.sh
    with a fixed argument list terminated by `-- "$@"`), so PWW_DUMP is the only way a
    lane gets its own dump folder -- and PWW_FRESH_RUN reads ${DUMP:-}, so without this
    it clears the SHARED checkpoint rather than the lane's."""
    assert emit_mod.preflight(emit_mod.EmitConfig(root=str(ROOT))) == []
    text = (ROOT / "scripts/titan/run_train.sh").read_text()
    assert 'DUMP="${PWW_DUMP:-}"' in text
    # and a checkout without the fix is still reported
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "scripts" / "titan"
        broken.mkdir(parents=True)
        (broken / "run_train.sh").write_text('DUMP=""\n')
        problems = emit_mod.preflight(emit_mod.EmitConfig(root=tmp))
        assert len(problems) == 1 and "PWW_DUMP" in problems[0]


# --------------------------------------------------------------------------
# the report and the CLI: rendered at all, and the same tree in --json
# --------------------------------------------------------------------------


@check("the report renders every section and the JSON tree carries the same numbers")
def _():
    """report.py and cli.py had no coverage at all, and both are the primary interface:
    a rendering error in section 4 is a traceback instead of a plan."""
    from pww.plan import report as report_mod
    from pww.plan.adapter import Collected
    from pww.plan.model import PlannerInputs

    sites = (snellius(((8, 0.0),)), lumi(((8, 2.0),)))
    plan = plan_of(sites, alpha=0.0)
    collected = Collected(
        inputs=PlannerInputs(sites=sites, calibration=CAL, darl=MID_RUN),
        provenance=(), notes=("a note",), raw={})
    subs = emit(plan)
    geometries = {s.site: dict(s.geometries) for s in sites}
    text = report_mod.render(plan, collected, subs, CAL, registry_geometries=geometries)
    for heading in ("1. INPUTS AS READ", "2. EXCLUSIONS", "3. TIMELINE",
                    "4. PER-SITE LEDGER", "5. VERDICT", "6. SENSITIVITY",
                    "7. MARGINAL LEDGER", "8. SBATCH"):
        assert heading in text, heading
    assert "BETW" in text, "the fourth hour column is printed"
    assert "period{" in text and " = " in text, "every headline period prints its arithmetic"

    tree = report_mod.as_json(plan, collected, subs, CAL)
    assert tree["score"]["federated_merges"] == plan.score.federated_merges
    assert tree["derived"]["describe"] == plan.describe()
    for site in tree["derived"]["per_site"].values():
        assert site["live_h"] is not None and site["idle_fraction"] <= 1.0
    # it must actually serialise -- properties do not survive asdict, which is why
    # `derived` exists
    import json as _json
    _json.loads(report_mod.dump_json(tree))


@check("section 8 quotes the used_ratio it MEASURED, not a constant from a past capture")
def _():
    """It hardcoded "used_ratio on Snellius gpu_h100 is ~0.11 ... finished jobs consumed
    11% of what they asked for" while section 1 of the SAME report printed the live
    0.1617 for that partition. It is not decoration: it is the number the reader is
    asked to reason about when judging whether the quoted wait is pessimistic, and the
    discount w_eff = w*(1 - s*(1 - used_ratio)) is computed from the real one."""
    from pww.plan import report as report_mod
    from pww.plan.adapter import Collected
    from pww.plan.model import PlannerInputs

    def with_ratio(site_in, ratio):
        return dataclasses.replace(site_in, waits={
            k: dataclasses.replace(v, used_ratio=ratio, discount_strength=0.5)
            for k, v in site_in.waits.items()})

    sites = (with_ratio(snellius(((8, 0.0),)), 0.4242),
             with_ratio(lumi(((8, 2.0),)), 0.3131))
    plan = plan_of(sites, alpha=0.0)
    collected = Collected(
        inputs=PlannerInputs(sites=sites, calibration=CAL, darl=MID_RUN),
        provenance=(), notes=(), raw={})
    text = report_mod.render(plan, collected, emit(plan), CAL,
                             registry_geometries={s.site: dict(s.geometries) for s in sites})
    section8 = text[text.index("8. SBATCH"):]
    assert "~0.11" not in section8 and "consumed 11%" not in section8, section8[:900]
    chosen = {o.site for o in plan.selection}
    assert chosen, plan.describe()
    for site_in in sites:
        if site_in.site in chosen:
            ratio = next(iter(site_in.waits.values())).used_ratio
            assert f"{ratio:.4f}" in section8, (site_in.site, ratio, section8[:900])


@check("an unrecognised --chain policy is refused, not priced one way and emitted another")
def _():
    """The simulator knows none|self|singleton and `break`s out of the link loop on
    anything else, so a typo was SIMULATED as one job per lane while generate_options
    still set links_per_lane from _links_to_cover and the emitter still shipped the
    full N-link chain: the plan's headline merge counts and the number of jobs actually
    submitted diverged by ~7x, silently, at exit 0."""
    from pww.plan import cli

    import contextlib
    import io as _io

    fixture = str(ROOT / "tests/fixtures/plan/two-site.json")
    base = ["--dry-run", fixture, "--root", str(ROOT), "--now", "1787142156"]
    err = _io.StringIO()
    with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(err):
        try:
            code = cli.main(base + ["--chain", "bogus"])
        except SystemExit as exc:                      # argparse-style refusal
            code = exc.code if isinstance(exc.code, int) else str(exc)
    assert code != 0, code
    message = str(code) + err.getvalue()
    assert "bogus" in message and "none|self|singleton" in message, message

    # every policy the simulator DOES implement still plans
    for policy in ("none", "self", "singleton", "self,none"):
        out = _io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(_io.StringIO()):
            assert cli.main(base + ["--chain", policy]) == 0, policy
        assert "pww-plan  --" in out.getvalue(), policy


@check("the presence bar draws the four states it exists to distinguish")
def _():
    """report.py's docstring says the ASCII bar is the artifact the module exists for --
    'the thing a table cannot show' -- and nothing asserted a single character of it.
    A bar showing nobody productive, or no `^` on a plan with 71 federated merges,
    passed. The four glyphs are four different COSTS: `~` queued is free, `:` is
    allocated-but-starting-up (billed, contributing nothing -- this is c, and it is why
    chaining is not free), `#` is productive, `.` is absent."""
    from pww.plan import report as report_mod

    plan = plan_of((snellius(), lumi()))
    assert len({o.site for o in plan.selection}) == 2, plan.describe()
    bars = report_mod.presence_bars(plan)
    body = [b for b in bars if "|" in b]
    assert len(body) == 3, bars              # two sites plus the federated row
    rows = {b.split("|")[0].strip(): b.split("|")[1] for b in body}
    assert set(rows) == {"snellius", "lumi", "federated"}, rows
    # every row is the same width, or the axis is a lie
    assert len({len(v) for v in rows.values()}) == 1, {k: len(v) for k, v in rows.items()}

    # All four states are actually drawn, and each means a different cost.
    for site in ("snellius", "lumi"):
        assert "#" in rows[site], (site, rows[site])   # productive
        assert "~" in rows[site], (site, rows[site])   # queued, free
        assert ":" in rows[site], (site, rows[site])   # billed but starting up
        assert "." in rows[site], (site, rows[site])   # absent
    # the federated row marks exactly the columns where BOTH sites are productive
    for col, ch in enumerate(rows["federated"]):
        both = rows["snellius"][col] == "#" and rows["lumi"][col] == "#"
        assert (ch == "^") == both, (col, ch, rows)
    assert "^" in rows["federated"], rows["federated"]
    # only the documented alphabet appears
    assert set("".join(rows.values())) <= set(".~:#^ "), set("".join(rows.values()))
    assert any("legend:" in b for b in bars), bars

    # A solo plan has no federated columns at all -- the bar must say so rather than
    # drawing '^' for a single live site.
    solo = report_mod.presence_bars(plan_of((snellius(((8, 0.0),)),)))
    solo_rows = {b.split("|")[0].strip(): b.split("|")[1] for b in solo if "|" in b}
    assert "^" not in solo_rows["federated"], solo_rows


@check("the CLI plans from a fixture and exits on the plan, not on the tree")
def _():
    from pww.plan import cli

    import contextlib
    import io as _io

    fixture = str(ROOT / "tests/fixtures/plan/two-site.json")
    argv = ["--dry-run", fixture, "--root", str(ROOT), "--sbatch-only"]
    out = _io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(_io.StringIO()):
        assert cli.main(argv) == 0
    assert "sbatch \\" in out.getvalue(), out.getvalue()[:400]

    tree = _io.StringIO()
    with contextlib.redirect_stdout(tree), contextlib.redirect_stderr(_io.StringIO()):
        assert cli.main(argv + ["--json"]) == 0
    import json as _json
    assert _json.loads(tree.getvalue())["derived"]["describe"]

    # a site nobody has data for admits nothing, and that is status 2 with the reason
    # printed -- not a crash and not a confident empty plan
    quiet = _io.StringIO()
    with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(_io.StringIO()):
        assert cli.main(["--dry-run", fixture, "--root", str(ROOT),
                         "--sites", "nowhere"]) == 2
    assert "site_not_in_sources" in quiet.getvalue()


def main() -> int:
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for name, exc in FAILED:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
