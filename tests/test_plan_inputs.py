"""The reading layer: every malformed cell has to come back as a refusal.

    python3 tests/test_plan_inputs.py

`inputs` is the only module that touches the outside world, and the failure it exists
to prevent is not a crash -- it is a plausible number. A wait of 0 s because the cell
was empty, a step time of 0 s because a throughput cell said `inf`, a 2-node job
priced against the 1-node cell because `-N2` did not parse: each of these produces a
plan that looks like every other plan and is wrong. So every check below asserts two
things about one bad input -- that no Geometry, Shape or WaitEstimate was built from
it, and that an Exclusion carrying a fix came back instead.

The doors are the reason this file exists rather than a section of another: the same
value arrives through the registry, through a --dry-run fixture and through the
scanner CSVs, and a guard on one door is not a guard.

    Snellius@4  89.8 seq/s at 32 seq/step -> 0.356 s/step
    LUMI@8      38.2 seq/s at 64 seq/step -> 1.675 s/step
"""

from __future__ import annotations

import json
import math
import signal
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURE = ROOT / "tests" / "fixtures" / "plan" / "two-site.json"

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


from pww.plan import DEFAULT_CALIBRATION as CAL, adapter, inputs as io  # noqa: E402

ARGS = "-p gpu_h100 -N 1 --gpus-per-node 4 -t 8:00:00"


def probe(wait, *, name="s1", collected_at=1000, ok=True, args=ARGS):
    return {"cluster": "snellius", "name": name, "ok": ok, "args": args,
            "collected_at": collected_at, "estimated_wait_sec": wait}


def registry(*lines):
    """A throughput registry in a temp dir, returned already read."""
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "site_throughput.env"
    path.write_text("\n".join(lines) + "\n")
    out = io.load_throughput(path)
    tmp.cleanup()
    return out


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------


@check("a short option with an attached value parses, rather than halving -N in silence")
def _():
    """`-N2` is valid Slurm and was read as nodes=1: the device count that picks the
    throughput cell was halved, so a 2-node wait was priced with the 1-node step time
    while the emitter -- which copies the argument string verbatim -- still submitted
    two nodes. Nothing failed and nothing was printed."""
    attached = io.parse_shape_args("snellius", "-N2 -pgpu_h100 --gpus-per-node 4 -t8:00:00")
    assert attached.nodes == 2 and attached.gpus == 8, attached
    assert attached.partition == "gpu_h100" and attached.walltime_s == 8 * 3600, attached
    assert attached == io.parse_shape_args(
        "snellius", "-N 2 -p gpu_h100 --gpus-per-node 4 -t 8:00:00")
    assert io.parse_shape_args(
        "lumi", "-Aproj -p standard-g --gpus-per-node 8 -t 1:00:00").account == "proj"
    # The shipped lumi shapes use --gpus=N, which is NOT a device count this can key
    # a wait to. That one must stay loud rather than become a 1-GPU guess.
    try:
        io.parse_shape_args("lumi", "-A proj -p small-g --gpus=1 -t 1:00:00")
    except ValueError as exc:
        assert "--gpus-per-node" in str(exc), exc
    else:
        raise AssertionError("--gpus=1 was keyed to a shape instead of being refused")


# --------------------------------------------------------------------------
# waits
# --------------------------------------------------------------------------


@check("a present-but-unreadable wait is a named refusal on the JSON path too")
def _():
    """The CSV path coerced a bad cell to None and named it `_bad_cells`; the fixture
    and HTTP paths handed the raw JSON value to float() and raised a bare ValueError
    out of the CLI -- exit 1, zero bytes on stdout, no plan and no indication of which
    cluster or shape was malformed. One input, two doors, one refusal."""
    rows = [probe(bad) for bad in ("soon", "n/a", "")]
    shapes, waits, excl = io.build_shapes("snellius", rows, [], now=2000.0)
    assert not shapes and not waits, (shapes, waits)
    refusal = [e for e in excl if e.code == "no_wait_reading"]
    assert len(refusal) == 1 and refusal[0].fix, excl
    assert "'soon'" in refusal[0].reason, refusal[0].reason


@check("a wait that float() accepts but is not a reading is refused, not priced")
def _():
    """REGRESSION. The guard above tested float(), and 'nan', 'inf' and a leading
    minus all pass float(). Each failed differently: 'nan' reached
    search._links_to_cover as `cannot convert float NaN to integer` (exit 1, nothing
    on stdout, no exclusion); 'inf' made the shape silently unschedulable; and
    -36000 priced h100_full_8h as having started ten hours ago, which at exit 0 bought
    it a seventh chain link and CHANGED the recommendation. The collector writes
    max(0, start - now), so none of the three can come off a probe."""
    for bad in ("nan", "inf", "-inf", -36000, float("nan")):
        shapes, waits, excl = io.build_shapes("snellius", [probe(bad)], [], now=2000.0)
        assert not shapes and not waits, (bad, shapes, waits)
        refusal = [e for e in excl if e.code == "no_wait_reading"]
        assert len(refusal) == 1 and refusal[0].fix, (bad, excl)
        assert "unreadable numeric cells" in refusal[0].reason, refusal[0].reason
    # A readable row among unreadable ones still prices the shape: the refusal is per
    # shape, not per file, or one bad cell would cost the whole site.
    shapes, waits, excl = io.build_shapes(
        "snellius", [probe("nan"), probe(3600, collected_at=1500)], [], now=2000.0)
    assert waits["s1"].p50_raw_s == 3600.0 and waits["s1"].samples == 1, waits
    assert not [e for e in excl if e.code == "no_wait_reading"], excl


@check("the CSV door and the JSON door agree cell for cell")
def _():
    """The scanner's own coercion turns a bad numeric cell into None plus a
    `_bad_cells` note; a fixture hands the same cell over untouched. Both must end at
    the same refusal, or a plan replayed from a capture differs from the plan made
    live off the same rows."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "snellius"
        root.mkdir()
        (root / "probes.csv").write_text(
            "collected_at,name,args,ok,estimated_wait_sec,placed_partition,probed_by_user\n"
            f"1000,s1,{ARGS},True,soon,gpu_h100,zare\n")
        rows = io.read_csv_table(Path(tmp), "snellius", "probes")
    assert rows[0]["estimated_wait_sec"] is None and rows[0]["_bad_cells"] == ["estimated_wait_sec"]
    csv_excl = io.build_shapes("snellius", rows, [], now=2000.0)[2]
    json_excl = io.build_shapes("snellius", [probe("soon")], [], now=2000.0)[2]
    assert [e.code for e in csv_excl] == [e.code for e in json_excl] == ["no_wait_reading"]


# --------------------------------------------------------------------------
# the throughput registry
# --------------------------------------------------------------------------


@check("a throughput or batch that is not POSITIVE is refused by name, not divided by")
def _():
    """Geometry.step_s is batch/tput, so PWW_BATCH_<SITE>=0 was a ZeroDivisionError
    out of the CLI with zero bytes on stdout, and -32 was worse: a full plan at exit 0
    with a negative step time, -1.69 G tokens and -806 DARL blocks, so blocks_left
    never decreased, the corpus never exhausted and the run was scored to the horizon.
    Both halves of the pair, because both come from one typo in the same file."""
    for bad, culprit in ((["PWW_TPUT_SNELLIUS_4=0", "PWW_BATCH_SNELLIUS_4=32"],
                          "PWW_TPUT_SNELLIUS_4"),
                         (["PWW_TPUT_SNELLIUS_4=89.8", "PWW_BATCH_SNELLIUS_4=0"],
                          "PWW_BATCH_SNELLIUS_4"),
                         (["PWW_TPUT_SNELLIUS_4=89.8", "PWW_BATCH_SNELLIUS_4=-32"],
                          "PWW_BATCH_SNELLIUS_4"),
                         (["PWW_TPUT_SNELLIUS_4=-89.8", "PWW_BATCH_SNELLIUS_4=32"],
                          "PWW_TPUT_SNELLIUS_4")):
        # A good cell alongside, so the refusal is per cell and the plan can still
        # fall back to snellius@1.
        cells, excl = registry("PWW_TPUT_SNELLIUS_1=24.4", "PWW_BATCH_SNELLIUS_1=8", *bad)
        assert 4 not in cells.get("snellius", {}), (bad, cells)
        assert cells["snellius"][1].tput_seq_s == 24.4, cells
        hit = [e for e in excl if e.code == "bad_registry_value"]
        assert len(hit) == 1 and hit[0].fix, (bad, excl)
        assert culprit in hit[0].reason and "not positive" in hit[0].reason, hit[0].reason
        # --registry is not always the default (plan_campaign.sh passes its own), and
        # a fix naming a file this plan never opened is not a fix.
        assert "configs/site_throughput.env" not in hit[0].fix, hit[0].fix
        assert hit[0].fix.split()[1].endswith("site_throughput.env"), hit[0].fix


@check("a throughput cell that is positive but not FINITE is refused too")
def _():
    """REGRESSION. `> 0` admits inf, and float() reads 'inf' and overflows 1e400 into
    it. step_s is then batch/inf = 0.0 -- not a crash where it was read, but a
    ZeroDivisionError one module later in rounds.balance_accums, out of the CLI at
    exit 1 with nothing on stdout: the very failure the positivity guard was added to
    prevent, through a value that satisfies it."""
    for spelling in ("inf", "Infinity", "1e400", "-inf", "nan"):
        cells, excl = registry("PWW_TPUT_SNELLIUS_1=24.4", "PWW_BATCH_SNELLIUS_1=8",
                               f"PWW_TPUT_SNELLIUS_4={spelling}", "PWW_BATCH_SNELLIUS_4=32")
        assert 4 not in cells.get("snellius", {}), (spelling, cells)
        assert cells["snellius"][1].step_s == 8 / 24.4, cells
        hit = [e for e in excl if e.code == "bad_registry_value"]
        assert len(hit) == 1 and hit[0].fix, (spelling, excl)
    for spelling in ("inf", "1e400"):
        cells, excl = registry(f"PWW_TPUT_SNELLIUS=89.8", f"PWW_BATCH_SNELLIUS={spelling}")
        assert cells == {}, (spelling, cells)
        assert [e.code for e in excl] == ["bad_registry_value"], excl


@check("a cell keyed on a device count nothing can run on is refused")
def _():
    """PWW_TPUT_<SITE>_<N> pins the device count, and `_0` sailed through into a
    Geometry with gpus=0 -- a cell that no shape can ever match, printed in the
    provenance block as a measurement the site has."""
    cells, excl = registry("PWW_TPUT_SNELLIUS=89.8", "PWW_BATCH_SNELLIUS=32",
                           "PWW_TPUT_SNELLIUS_0=89.8", "PWW_BATCH_SNELLIUS_0=32")
    assert sorted(cells["snellius"]) == [4], cells      # 32/8 = the reference geometry
    assert [e.code for e in excl] == ["bad_registry_value"], excl
    assert excl[0].fix and "0 devices" in excl[0].reason, excl[0].reason


@check("the shipped registry still reads, and reads to the measured constants")
def _():
    """The guards above must refuse bad cells without refusing the real file."""
    cells, excl = io.load_throughput(ROOT / "configs" / "site_throughput.env")
    assert excl == [], excl
    assert cells["snellius"][4].tput_seq_s == 89.8 and cells["snellius"][4].batch_seq == 32
    assert cells["lumi"][8].tput_seq_s == 38.2 and cells["lumi"][8].batch_seq == 64
    assert math.isclose(cells["snellius"][4].step_s, 0.356347, rel_tol=1e-5)
    assert math.isclose(cells["lumi"][8].step_s, 1.675393, rel_tol=1e-5)


# --------------------------------------------------------------------------
# the same cell, arriving through a fixture
# --------------------------------------------------------------------------


def fixture_geometries(cells):
    src = adapter.Sources(fixture="capture.json", registry="configs/site_throughput.env")
    return adapter._throughput(src, {"throughput": cells}, [])


@check("a fixture-pinned throughput cell is read with the registry's guard, not float()")
def _():
    """REGRESSION. A --dry-run fixture may pin throughput so a replay is hermetic, and
    that block is hand-written: it is the registry with more ways to be edited. It was
    read with a bare float()/int() and no positivity check at all, so the defect the
    registry door refuses walked in through this one -- `"batch_seq": -32` gave a full
    plan at exit 0 with -1.69 G tokens and -806 DARL blocks, `0` a ZeroDivisionError,
    and `"89.8 seq/s"` a ValueError traceback with zero bytes on stdout."""
    good = {"1": {"tput_seq_s": 24.4, "batch_seq": 8}}
    for bad in ({"tput_seq_s": 89.8, "batch_seq": -32},
                {"tput_seq_s": 0, "batch_seq": 32},
                {"tput_seq_s": -89.8, "batch_seq": 32},
                {"tput_seq_s": float("inf"), "batch_seq": 32},
                {"tput_seq_s": "89.8 seq/s", "batch_seq": 32},
                {"tput_seq_s": 89.8},
                89.8):
        cells, excl = fixture_geometries({"snellius": dict(good, **{"4": bad})})
        assert 4 not in cells.get("snellius", {}), (bad, cells)
        assert cells["snellius"][1].tput_seq_s == 24.4, (bad, cells)   # per cell, still
        assert len(excl) == 1 and excl[0].code == "bad_registry_value", (bad, excl)
        assert "capture.json" in excl[0].subject and excl[0].fix, excl[0]
        # The fix has to name the fixture: telling an operator to re-run the
        # calibrator would edit a file this plan never read.
        assert "capture.json" in excl[0].fix, excl[0].fix
    # A cell that lost its nesting names the value it found, because "are not
    # numbers" about two absent keys does not tell the operator what to delete.
    cells, excl = fixture_geometries({"snellius": dict(good, **{"4": 89.8})})
    assert "89.8" in excl[0].reason and "batch" in excl[0].reason, excl[0].reason
    # A key that is not a device count is refused the same way, rather than raising
    # ValueError out of int() with the report unwritten.
    cells, excl = fixture_geometries({"snellius": dict(good, **{"four": {
        "tput_seq_s": 89.8, "batch_seq": 32}})})
    assert sorted(cells["snellius"]) == [1], cells
    assert len(excl) == 1 and excl[0].fix, excl
    # and the good block still reads exactly as recorded
    cells, excl = fixture_geometries({"snellius": good, "lumi": {"8": {
        "tput_seq_s": 38.2, "batch_seq": 64, "source": "all_logs/pww-lumi-titan.out"}}})
    assert excl == [] and cells["lumi"][8].step_s == 64 / 38.2, (cells, excl)
    assert cells["lumi"][8].source == "all_logs/pww-lumi-titan.out", cells


@check("a poisoned fixture cell reaches collect() as an exclusion, not as a geometry")
def _():
    """End to end on the path that works TODAY (the scanner instance is not up, so
    --dry-run is how plans get made): the whole read must come back with the site
    intact, the bad cell absent and a fix naming the file that carries it."""
    payload = json.loads(FIXTURE.read_text())
    payload["throughput"]["snellius"]["4"]["batch_seq"] = -32
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "poisoned.json"
        path.write_text(json.dumps(payload))
        collected = adapter.collect(adapter.Sources(
            fixture=str(path),
            registry=str(ROOT / "configs" / "site_throughput.env"),
            planner_config=str(ROOT / "configs" / "plan" / "federation.json"),
            probe_config_dir=str(ROOT / "configs" / "slurm_probe"),
            darl_url=None))
    snellius = [s for s in collected.inputs.sites if s.site == "snellius"]
    assert snellius, [s.site for s in collected.inputs.sites]
    assert sorted(snellius[0].geometries) == [1], snellius[0].geometries
    assert all(g.batch_seq > 0 for g in snellius[0].geometries.values()), snellius[0].geometries
    bad = [e for e in collected.inputs.exclusions if e.code == "bad_registry_value"]
    assert len(bad) == 1 and "poisoned.json" in bad[0].subject, collected.inputs.exclusions
    assert bad[0].fix, bad[0]


# --------------------------------------------------------------------------
# the numbers that are inputs rather than readings
# --------------------------------------------------------------------------


@check("a startup cost that is not a duration is refused, not printed as measured")
def _():
    """c is the wall-clock from allocation to the first training step and it pays for
    every chain link, so a negative one makes links free and the search buys them:
    `startup_cost_s.snellius.value_s: -3600` changed the recommendation at exit 0 and
    printed `-3600 s (-60.0 min)` in the provenance block as though it had been
    measured. A string went out of float() as a bare ValueError with no report."""
    config = {"startup_cost_s": {"_": "a note, not a site",
                                 "lumi": {"value_s": 216, "quality": "lower_bound"},
                                 "snellius": {"value_s": -3600}}}
    for bad in (-3600, "soon", None, float("inf")):
        config["startup_cost_s"]["snellius"] = {"value_s": bad}
        prov = []
        costs, excl = adapter._startup_costs(
            adapter.Sources(planner_config="configs/plan/federation.json"), config, prov)
        assert sorted(costs) == ["lumi"], (bad, costs)
        assert [e.code for e in excl] == ["bad_startup_cost"] and excl[0].fix, (bad, excl)
        assert not [p for p in prov if "snellius" in p.field], prov
    # --startup-cost is the same number by another route and gets the same guard.
    costs, excl = adapter._startup_costs(
        adapter.Sources(startup_overrides={"lumi": -1.0}), {}, [])
    assert costs == {} and [e.code for e in excl] == ["bad_startup_cost"], (costs, excl)
    # and a real config still reads
    costs, excl = adapter._startup_costs(
        adapter.Sources(), json.loads((ROOT / "configs" / "plan" / "federation.json").read_text()), [])
    assert excl == [] and costs["snellius"][0] > 0 and costs["lumi"][0] > 0, (costs, excl)


@check("a calibration overhead that is not a duration is refused, and says so")
def _():
    """xfer and eval_fix are seconds at the barrier, so a negative one is a round that
    costs less than its own inner phase: snellius.xfer at -33.5 came out at exit 0
    recommending a different geometry with nothing in the report to say why, and
    merge: 'seventeen' raised out of float() with the report unwritten. The built-in
    table is the documented default, so it stands -- but the refusal has to say that,
    or a silently ignored override is indistinguishable from an applied one."""
    payload = json.loads((ROOT / "configs" / "plan" / "federation.json").read_text())
    payload["sites"]["snellius"]["xfer"]["value_s"] = -33.5
    payload["merge"]["value_s"] = "seventeen"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "federation.json"
        path.write_text(json.dumps(payload))
        calibration, problems = io.load_calibration(path)
    codes = sorted(e.code for e in problems)
    assert codes == ["bad_calibration_value", "bad_calibration_value"], problems
    assert all(e.fix for e in problems), problems
    # neither half of the refused site survives, and the built-in table stands
    assert calibration.sites["snellius"] == CAL.sites["snellius"], calibration.sites["snellius"]
    assert calibration.merge == CAL.merge, calibration.merge
    # the file's GOOD entries are still overrides, not casualties of the bad one
    assert calibration.sites["lumi"].xfer.value_s == 64.2, calibration.sites["lumi"]


def main() -> int:
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for name, exc in FAILED:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
