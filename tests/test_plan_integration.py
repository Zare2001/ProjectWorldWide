"""The seams between the plan modules, and the couplings no single module can police.

    python3 tests/test_plan_integration.py

Four agents fixed src/pww/plan in parallel over disjoint files. Everything here is a
defect that lived in a seam -- either between two modules that no one owned at once, or
between a module and a file it reads at run time. Three kinds:

  * a fact stated in two places, where the copies drifted. The chain policy names were
    written down three times (timeline's branches, cli's argparse tuple, search's
    pricing); a name in the simulator and not in argparse is refused at the flag, which
    is the "priced one way, submitted another" bug with the sign reversed.

  * a value that must be right in EVERY constructor of one type. DarlState.observed
    decides whether the emitter hands out PWW_FRESH_RUN=1, and a new construction that
    forgets it loses the flags silently. Five call sites, none of them pinned.

  * arithmetic that divides by a measurement, in a module that does not read it. The
    reader refuses a zero or NaN cell now, but the divides were reached from four
    directions and only one door was shut.

The census checks here are deliberately structural -- they read the source. A behaviour
check cannot see a SIXTH DarlState construction that nobody wrote yet, and that is the
regression that keeps happening.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASSED, FAILED = [], []
CHECK_TIMEOUT_S = 180


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
    Shape,
    WaitEstimate,
)
from pww.plan import cli as cli_mod  # noqa: E402
from pww.plan import rounds as rounds_mod  # noqa: E402
from pww.plan import search as search_mod  # noqa: E402
from pww.plan import timeline as timeline_mod  # noqa: E402
from pww.plan.adapter import collect  # noqa: E402
from pww.plan.inputs import parse_shape_args  # noqa: E402
from pww.plan.model import Candidate, Member, Option  # noqa: E402

PLAN_SRC = ROOT / "src" / "pww" / "plan"
FIXTURE = ROOT / "tests" / "fixtures" / "plan" / "two-site.json"
REGISTRY = ROOT / "configs" / "site_throughput.env"


def run_cli(*args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "pww.plan", *args],
                          cwd=str(ROOT), env=env, capture_output=True,
                          text=True, timeout=CHECK_TIMEOUT_S - 20)


def cand(*, wait_s: float, walltime_s: int = 3600, site: str = "lumi") -> Candidate:
    key = parse_shape_args(
        site, f"-p standard-g -N 1 --gpus-per-node 8 -t {walltime_s // 3600}:00:00")
    return Candidate(
        site=site,
        shape=Shape(name="1node_1h", key=key, args="-p standard-g -N 1"),
        wait=WaitEstimate(p50_raw_s=wait_s, p90_raw_s=wait_s, p50_eff_s=wait_s,
                          p90_eff_s=wait_s, samples=8, probe_age_s=60.0),
        geometry=Geometry(site=site, gpus=8, tput_seq_s=38.2, batch_seq=64),
        startup_s=216.0, overhead_s=17.0, overhead_quality="measured")


def member(step_s: float, lane: str) -> Member:
    return Member(lane_id=lane, site=lane.split("-")[0], gpus=1, step_s=step_s,
                  batch_seq=8, overhead_s=10.0, quality="measured")


# --------------------------------------------------------------------------
# one fact, one place
# --------------------------------------------------------------------------


@check("the chain policy names exist once: argparse and pricing both read the simulator")
def _():
    """cli.py kept a third hand-maintained copy of ("none", "self", "singleton").

    A copy that falls BEHIND expand_links refuses at the flag a policy the planner can
    price and emit; a copy that runs AHEAD is the original defect, a name priced as an
    N-link chain and simulated as one job. Neither is visible from inside either file,
    so the tuple has to BE the simulator's, not equal it.
    """
    assert cli_mod._CHAIN_POLICIES is timeline_mod.CHAIN_POLICIES, (
        "cli.py has its own copy of the policy names again")

    # And the simulator's tuple must match the branches that give it meaning. Read
    # from expand_links' own source, so teaching it a policy and forgetting the tuple
    # fails here rather than at a `sbatch` seven hours into a campaign.
    tree = ast.parse((PLAN_SRC / "timeline.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "expand_links")
    branched = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Attribute)
                and node.left.attr == "chain"):
            for comp in node.comparators:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                    branched.add(comp.value)
    assert branched, "expand_links no longer branches on option.chain by name"
    # "none" is the fall-through `break`, so it is named by absence, not by a branch.
    assert branched | {"none"} == set(timeline_mod.CHAIN_POLICIES), (
        sorted(branched | {"none"}), sorted(timeline_mod.CHAIN_POLICIES))


@check("can_chain answers from the simulator, so a listed-but-dead policy is still refused")
def _():
    """The predicate must not degrade into a membership test on the very tuple it
    guards -- that is what made a typo'd --chain priceable in the first place."""
    config = PlanConfig()
    c = cand(wait_s=0.0)
    for policy in timeline_mod.CHAIN_POLICIES:
        assert timeline_mod.can_chain(policy, config=config, candidate=c), policy
    assert not timeline_mod.can_chain("slef", config=config, candidate=c)

    # search prices only what the simulator expands, and says so by name.
    search_mod._require_simulable_chain("self", [c], config, False)
    try:
        search_mod._require_simulable_chain("mirror", [c], config, False)
    except ValueError as exc:
        assert "mirror" in str(exc) and "expands only its first link" in str(exc), exc
    else:
        raise AssertionError("an unexpandable chain policy was priced as a chain")

    # The CLI door and the simulator agree in BOTH directions.
    assert cli_mod._chain_policies("none,self,singleton") == ("none", "self", "singleton")
    for bad in ("slef", "mirror"):
        try:
            cli_mod._chain_policies(bad)
        except SystemExit as exc:
            assert bad in str(exc), exc
        else:
            raise AssertionError(f"--chain {bad} was accepted")


# --------------------------------------------------------------------------
# DarlState.observed -- the flag that decides whether checkpoints get deleted
# --------------------------------------------------------------------------


@check("every DarlState construction in the planner is classified observed or not")
def _():
    """emit.py:239 reads `not darl.fresh_epoch`, which is `observed and ...`.

    A construction added for a path where a coordinator really answered, and missing
    observed=True, silently drops the fresh flags at both ends -- the safe direction,
    but wrong. One that sets it on a FALLBACK resurrects PWW_FRESH_RUN=1 against a
    live epoch, which is the critical this whole guard exists for. A behaviour test
    covers the five that exist; this covers the sixth.
    """
    answered = {  # source= reads as "a coordinator answered GET /status"
        ("inputs.py", "fetch_darl_status"),
        ("adapter.py", "_state_from_payload"),
    }
    fallback = {  # derived, assumed or replayed: must NOT claim to be observed
        ("adapter.py", "_darl_state"),
        ("cli.py", "_darl_of"),
    }
    found: dict[tuple[str, str], list[bool]] = {}
    for path in sorted(PLAN_SRC.glob("*.py")):
        tree = ast.parse(path.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "DarlState"):
                    continue
                observed = any(
                    kw.arg == "observed" and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True for kw in node.keywords)
                found.setdefault((path.name, fn.name), []).append(observed)

    assert set(found) == answered | fallback, (
        "a DarlState is built somewhere this test does not classify; decide whether "
        "that path is a coordinator ANSWERING or a fallback, then add it here: "
        f"{sorted(set(found) ^ (answered | fallback))}")
    for site in sorted(answered):
        assert all(found[site]), f"{site} answered GET /status but does not set observed"
    for site in sorted(fallback):
        assert not any(found[site]), f"{site} is a fallback but claims observed=True"


@check("fresh_epoch is True down exactly one of the four routes a corpus figure arrives by")
def _():
    """--blocks and the whole-corpus assumption both build unassigned == num_blocks
    for reasons that have nothing to do with the epoch being untouched. Reading that
    equality as freshness is what deletes a live campaign's checkpoints."""
    fresh = DarlState(num_blocks=2692, committed=0, leased=0, unassigned=2692,
                      source="http://145.38.206.143:29510/status", observed=True)
    mid = DarlState(num_blocks=2692, committed=1870, leased=0, unassigned=822,
                    source="http://145.38.206.143:29510/status", observed=True)
    blocks = DarlState(num_blocks=822, committed=0, leased=0, unassigned=822,
                       source="--blocks")
    assumed = DarlState(num_blocks=2692, committed=0, leased=0, unassigned=2692,
                        source="assumed: whole corpus")
    assert fresh.fresh_epoch, "an answered, untouched epoch is the one fresh case"
    assert not mid.fresh_epoch
    assert not blocks.fresh_epoch, "--blocks 822 must never read as a fresh epoch"
    assert not assumed.fresh_epoch, "the whole-corpus guess must never read as fresh"
    # blocks and assumed differ from `fresh` only in provenance, which is the point.
    assert (blocks.unassigned >= blocks.num_blocks
            and assumed.unassigned >= assumed.num_blocks)


# --------------------------------------------------------------------------
# dividing by a measurement, in a module that does not read it
# --------------------------------------------------------------------------


@check("a wait that is not a duration is named by the code that divides by it")
def _():
    """`math.ceil(span / walltime)` raised "cannot convert float NaN to integer" --
    exit 1, nothing on stdout, no site named. inputs.py refuses such a cell now, so
    this is the second door: anything building a WaitEstimate directly walks past the
    first."""
    config = PlanConfig()
    for bad in (float("nan"), float("-nan")):
        try:
            search_mod._links_to_cover(cand(wait_s=bad), config, 0.0)
        except ValueError as exc:
            assert "lumi/1node_1h" in str(exc) and "not a duration" in str(exc), exc
        else:
            raise AssertionError(f"a wait of {bad!r} was priced")

    # An INFINITE wait is not an error: it means the job never starts inside the
    # horizon, which is one link by the same arithmetic that handles a late begin.
    assert search_mod._links_to_cover(cand(wait_s=float("inf")), config, 0.0) == 1
    # A real wait still chains to the horizon: 48 h of 1 h links, capped at 8.
    assert search_mod._links_to_cover(cand(wait_s=0.0), config, 0.0) == 8
    assert config.max_links_per_lane == 8, config.max_links_per_lane
    # A zero walltime divides too, and is refused rather than caught downstream.
    try:
        search_mod._links_to_cover(cand(wait_s=0.0, walltime_s=0), config, 0.0)
    except ValueError as exc:
        assert "walltime" in str(exc), exc
    else:
        raise AssertionError("a shape with no walltime was chained")


@check("a step time that is not a duration is named by balance_accums, not divided by")
def _():
    """An infinite throughput cell arrived as step_s == 0.0 and left as
    ZeroDivisionError out of the CLI. The lane has to be NAMED: the balance ratio is
    the one number in the report an operator cannot check by eye."""
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            rounds_mod.balance_accums(
                [member(bad, "lumi-l0"), member(2.0, "snellius-l0")],
                balance=True, cap=8)
        except ValueError as exc:
            assert "lumi-l0" in str(exc), exc
        else:
            raise AssertionError(f"a step time of {bad!r} set the pace")

    # The measured pair still balances to the 5x/1x the cluster actually runs, and
    # an unbalanced round never reaches the divide at all.
    assert rounds_mod.balance_accums(
        [member(0.356, "snellius-l0"), member(1.675, "lumi-l0")],
        balance=True, cap=8) == (5, 1)
    assert rounds_mod.balance_accums(
        [member(0.0, "lumi-l0"), member(2.0, "snellius-l0")], balance=False) == (1, 1)


@check("a non-positive registry cell is an exclusion at the CLI, not a traceback")
def _():
    """inputs.py is the FIRST door for this, and emit.balance_crosscheck the second.

    The two were fixed by different agents and only the second was pinned at its own
    function boundary, so removing the reader's guard would have left the suite green
    while the CLI went back to exiting 1 with zero bytes on stdout -- the shape of the
    original ZeroDivisionError report.
    """
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "poisoned.env"
        bad.write_text(
            REGISTRY.read_text()
            + "\nPWW_TPUT_SNELLIUS_4=0\nPWW_BATCH_LUMI_8=-64\n")
        # The shipped fixture PINS a throughput block, and a pinned block wins over
        # --registry -- so poisoning the registry against the fixture as shipped tests
        # nothing at all. Strip it, which is also what a --record capture produces.
        raw = json.loads(FIXTURE.read_text())
        assert raw.pop("throughput", None), "the fixture no longer pins throughput"
        thin = Path(tmp) / "no-throughput.json"
        thin.write_text(json.dumps(raw))
        proc = run_cli("--dry-run", str(thin), "--registry", str(bad))
        bad_path = str(bad)

    assert proc.returncode == 0, (proc.returncode, proc.stderr[-2000:])
    assert "Traceback" not in proc.stderr, proc.stderr[-2000:]
    assert len(proc.stdout) > 10_000, f"the report collapsed to {len(proc.stdout)} bytes"
    named = re.findall(r"^\s+\[bad_registry_value\] (\S+)$", proc.stdout, re.M)
    assert set(named) == {"PWW_TPUT_SNELLIUS_4", "PWW_TPUT_LUMI_8"}, named
    # The fix has to name the file that was actually read, not the default. The
    # report wraps it, so compare against the unwrapped text.
    assert bad_path in re.sub(r"\s+", " ", proc.stdout), \
        "the exclusion's fix names a registry the plan did not read"
    assert "configs/site_throughput.env by hand" not in proc.stdout


# --------------------------------------------------------------------------
# what the report tells the operator about shapes it could not use
# --------------------------------------------------------------------------


@check("an unprobed shape names its site once, the way PLANNER.md documents it")
def _():
    """ShapeKey.describe() already leads with the cluster, and search prefixed
    site.site as well: "lumi/lumi/standard-g 8 gpu 4 h". PLANNER.md's own example of
    this exclusion shows the single-site form, so the doc was right and the code was
    not -- and no test compared them."""
    proc = run_cli("--dry-run", str(FIXTURE))
    assert proc.returncode == 0, proc.stderr[-2000:]
    subjects = re.findall(r"^\s+\[shape_not_probed\] (.+)$", proc.stdout, re.M)
    assert subjects, "no shape_not_probed exclusion in the fixture plan"
    for subject in subjects:
        site = subject.split("/")[0]
        assert not subject.startswith(f"{site}/{site}/"), subject
        assert subject.count(f"{site}/") == 1, subject

    documented = (ROOT / "PLANNER.md").read_text()
    quoted = re.search(r"^shape_not_probed\s+(\S+/\S.*)$", documented, re.M)
    assert quoted, "PLANNER.md no longer shows a shape_not_probed example"
    assert quoted.group(1).strip() in subjects, (quoted.group(1), subjects)


@check("a configured shape that cannot be keyed AND was never probed is still reported")
def _():
    """adapter._wanted_shapes swallowed an unkeyable entry with `except ValueError:
    continue`, on the grounds that build_shapes reports it -- which is only true when a
    probe ROW carries that name. lumi's 1gpu_{4,8,24,40}h are `--gpus=1` with no rows,
    so four configured shapes were absent from the report entirely while their
    1node_* siblings produced the shape_not_probed a reader can act on.

    Exactly once, though: the two doors overlap on 1gpu_1h and 2gpu_1h, which DO have
    rows, and printing those twice with two different fixes is its own defect.
    """
    lumi_config = json.loads((ROOT / "configs" / "slurm_probe" / "lumi.json").read_text())
    unkeyable = []
    for shape in lumi_config["shapes"]:
        try:
            parse_shape_args("lumi", " ".join(shape["args"]))
        except ValueError:
            unkeyable.append(shape["name"])
    assert len(unkeyable) >= 4, unkeyable

    probed = {r.get("name") for r in
              json.loads(FIXTURE.read_text())["probes"]["rows"]
              if r.get("cluster") == "lumi"}
    silent = [n for n in unkeyable if n not in probed]
    assert silent, "the fixture no longer exercises the unkeyable-and-unprobed case"

    proc = run_cli("--dry-run", str(FIXTURE))
    assert proc.returncode == 0, proc.stderr[-2000:]
    subjects = re.findall(r"^\s+\[unparseable_shape\] (.+)$", proc.stdout, re.M)
    for name in unkeyable:
        assert subjects.count(f"lumi/{name}") == 1, (name, subjects)
    # and the entry that IS probed keeps build_shapes' own fix, not the config one.
    both = next(n for n in unkeyable if n in probed)
    block = proc.stdout.split(f"[unparseable_shape] lumi/{both}")[1]
    block = block.split("[unparseable_shape]")[0].split("[shape_not_probed]")[0]
    assert "No probe row carries this name" not in block, (
        f"{both} has a probe row; it must not be told the collector never submits it")


# --------------------------------------------------------------------------
# the planner reads files that live training jobs also read
# --------------------------------------------------------------------------


@check("the site job scripts still ask for the geometry the throughputs were measured at")
def _():
    """emit.site_resource_flags() reads these #SBATCH headers at emit time and copies
    them onto links 2..N, so links inherit whatever the header says today while the
    plan is priced from throughputs calibrated against what it said at calibration
    time. Nothing else warns when those drift apart.

    The device counts are the load-bearing half -- they key the registry cell -- and
    the cores/memory are what run_train.sh's dataloader and MIOpen cache were measured
    with. Changing either is legitimate; changing it without recalibrating is not, and
    this check is where that argument has to happen.
    """
    registry = REGISTRY.read_text()
    expected = {
        "scripts/snellius/job_titan_diloco.sh": {
            "--gpus-per-node": "4", "--cpus-per-task": "64",
            "--ntasks-per-node": "1", "--partition": "gpu_h100"},
        "scripts/lumi/job_titan_diloco.sh": {
            "--gpus-per-node": "8", "--cpus-per-task": "56", "--mem": "480G",
            "--ntasks-per-node": "1"},
    }
    for script, want in expected.items():
        text = (ROOT / script).read_text()
        header = {}
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("#SBATCH"):
                continue
            flag, _, inline = line[len("#SBATCH"):].strip().partition("=")
            header[flag.strip()] = inline.strip()
        for flag, value in want.items():
            assert header.get(flag) == value, (
                f"{script} {flag}={header.get(flag)!r}, calibrated at {value!r}: "
                f"recalibrate with scripts/titan/calibrate_throughput.sh and update "
                f"configs/site_throughput.env, or put the header back")

    # The device count is what selects the registry cell the plan is priced from.
    for site, script in (("SNELLIUS", "scripts/snellius/job_titan_diloco.sh"),
                         ("LUMI", "scripts/lumi/job_titan_diloco.sh")):
        gpus = expected[script]["--gpus-per-node"]
        assert f"PWW_TPUT_{site}_{gpus}=" in registry, (
            f"{script} asks for {gpus} devices but {REGISTRY.name} has no "
            f"PWW_TPUT_{site}_{gpus} cell, so the plan cannot price that geometry")


@check("the plan is the same bytes under any hash seed; only the stderr timer moves")
def _():
    """The search walks dicts and sets of frozen dataclasses. Iteration order there is
    insertion order, but `set` is not, and a tie broken by set order would reorder the
    selection under PYTHONHASHSEED without changing any number -- which reads as a
    different plan to anyone diffing two runs."""
    seeds = ("0", "1", "42", "12345", "99999")
    outs = {}
    for seed in seeds:
        proc = run_cli("--dry-run", str(FIXTURE), env_extra={"PYTHONHASHSEED": seed})
        assert proc.returncode == 0, (seed, proc.stderr[-2000:])
        outs[seed] = proc.stdout
    first = outs[seeds[0]]
    for seed in seeds[1:]:
        assert outs[seed] == first, f"the plan differs under PYTHONHASHSEED={seed}"
    assert "pww-plan" in first and "SBATCH" in first, "that was not a plan"


def main() -> int:
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
