"""submit.py: which commands this host can run, and what happens when it runs them.

    python3 tests/test_plan_submit.py

No pytest, no network, and NO real sbatch/start_central_services.sh is ever
touched: every scenario builds its own throwaway root/PATH so this file is safe
to run on a laptop or in CI with no HPC access at all.
"""

from __future__ import annotations

import os
import signal
import stat
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASSED, FAILED = [], []
CHECK_TIMEOUT_S = 30


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


from pww.plan import state as st  # noqa: E402
from pww.plan import submit as sub_mod  # noqa: E402
from pww.plan.emit import Submission  # noqa: E402


def fake_bin(dir_path: Path, name: str, script: str) -> None:
    """A throwaway executable on a PATH this test controls, so `can_run_here`
    finding "sbatch" never means the real one."""
    path = dir_path / name
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


class isolated_path:
    """Prepend a tempdir to PATH for the duration of a `with` block: whatever
    fake binary it holds (or does not hold) is what `shutil.which` finds first,
    while `bash`/`echo`/etc. stay resolvable off the rest of the real PATH --
    `run_submission` execs through `bash -c`, so replacing PATH outright breaks
    the very thing being tested rather than isolating it."""

    def __init__(self, dir_path: Path):
        self.dir_path = str(dir_path)
        self._old = None

    def __enter__(self):
        self._old = os.environ.get("PATH", "")
        os.environ["PATH"] = self.dir_path + os.pathsep + self._old
        return self

    def __exit__(self, *exc):
        os.environ["PATH"] = self._old


# --------------------------------------------------------------------------
# can_run_here
# --------------------------------------------------------------------------


@check("central line: runnable iff start_central_services.sh exists under --root")
def _():
    central = Submission(site="(central)", lane_id="(central)", order=-1.0, begin_s=0.0,
                         args_verbatim="", command="true", comment="")
    with tempfile.TemporaryDirectory() as d:
        ok, reason = sub_mod.can_run_here(central, root=d, site="central")
        assert not ok, reason
        assert "start_central_services.sh" in reason
        script = Path(d) / "scripts" / "central_node" / "start_central_services.sh"
        script.parent.mkdir(parents=True)
        script.write_text("#!/bin/sh\ntrue\n")
        ok, reason = sub_mod.can_run_here(central, root=d, site="central")
        assert ok, reason


@check("site line: no sbatch on PATH -> skipped, names the site to run it from")
def _():
    site_sub = Submission(site="lumi", lane_id="lumi-l0", order=0.0, begin_s=0.0,
                          args_verbatim="-p standard-g", command="true", comment="")
    with tempfile.TemporaryDirectory() as empty_path_dir:
        with isolated_path(Path(empty_path_dir)):
            ok, reason = sub_mod.can_run_here(site_sub, root=".", site="lumi")
    assert not ok, reason
    assert "no sbatch" in reason


@check("site line: sbatch present but wrong site -> skipped, names the right site")
def _():
    site_sub = Submission(site="lumi", lane_id="lumi-l0", order=0.0, begin_s=0.0,
                          args_verbatim="-p standard-g", command="true", comment="")
    with tempfile.TemporaryDirectory() as bindir:
        fake_bin(Path(bindir), "sbatch", "echo fake")
        with isolated_path(Path(bindir)):
            ok, reason = sub_mod.can_run_here(site_sub, root=".", site="snellius")
    assert not ok, reason
    assert "this host is snellius" in reason, reason
    assert "lumi's login node" in reason, reason


@check("site line: sbatch present and site matches -> runnable")
def _():
    site_sub = Submission(site="lumi", lane_id="lumi-l0", order=0.0, begin_s=0.0,
                          args_verbatim="-p standard-g", command="true", comment="")
    with tempfile.TemporaryDirectory() as bindir:
        fake_bin(Path(bindir), "sbatch", "echo fake")
        with isolated_path(Path(bindir)):
            ok, reason = sub_mod.can_run_here(site_sub, root=".", site="lumi")
    assert ok, reason


# --------------------------------------------------------------------------
# run_submission: parses (or fails to parse) a fake sbatch's own output
# --------------------------------------------------------------------------


@check("run_submission parses a job id out of sbatch's real success message")
def _():
    sub = Submission(site="lumi", lane_id="lumi-l0", order=0.0, begin_s=0.0,
                     args_verbatim="", command="echo 'Submitted batch job 42424'",
                     comment="")
    ev = sub_mod.run_submission(sub, root=".", host="lumi")
    assert ev["event"] == "submitted", ev
    assert ev["job_id"] == "42424", ev


@check("run_submission: nonzero exit -> submit_failed, with sbatch's own stderr")
def _():
    sub = Submission(site="lumi", lane_id="lumi-l0", order=0.0, begin_s=0.0,
                     args_verbatim="", command="echo 'boom' >&2; exit 1", comment="")
    ev = sub_mod.run_submission(sub, root=".", host="lumi")
    assert ev["event"] == "submit_failed", ev
    assert "boom" in ev["reason"]


@check("run_submission: exit 0 but no job id -> submit_failed, not silently 'ok'")
def _():
    sub = Submission(site="lumi", lane_id="lumi-l0", order=0.0, begin_s=0.0,
                     args_verbatim="", command="echo 'something unexpected'",
                     comment="")
    ev = sub_mod.run_submission(sub, root=".", host="lumi")
    assert ev["event"] == "submit_failed", ev
    assert "no job id" in ev["reason"]


@check("run_submission: the (central) role needs no job id to count as submitted")
def _():
    sub = Submission(site="(central)", lane_id="(central)", order=-1.0, begin_s=0.0,
                     args_verbatim="", command="echo 'central services started'",
                     comment="")
    ev = sub_mod.run_submission(sub, root=".", host="central")
    assert ev["event"] == "submitted", ev
    assert "job_id" not in ev


# --------------------------------------------------------------------------
# submit_all: the whole flow, still with no real HPC tools anywhere in reach
# --------------------------------------------------------------------------


@check("submit_all: nothing on PATH -> every site line skipped, all logged")
def _():
    subs = [
        Submission(site="lumi", lane_id="lumi-l0", order=0.0, begin_s=0.0,
                  args_verbatim="", command="true", comment=""),
        Submission(site="snellius", lane_id="snellius-l0", order=0.0, begin_s=0.0,
                  args_verbatim="", command="true", comment=""),
    ]
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as empty:
        state_path = Path(d) / "events.jsonl"
        with isolated_path(Path(empty)):
            results = sub_mod.submit_all(subs, root=d, state_path=str(state_path))
        assert all(r["event"] == "submit_skipped" for r in results), results
        rows = st.read_events(state_path)
        assert len(rows) == 2
        assert all(r["event"] == "submit_skipped" for r in rows)


@check("submit_all: --dry-run writes nothing to the state file")
def _():
    subs = [Submission(site="lumi", lane_id="lumi-l0", order=0.0, begin_s=0.0,
                       args_verbatim="", command="true", comment="")]
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as empty:
        state_path = Path(d) / "events.jsonl"
        with isolated_path(Path(empty)):
            results = sub_mod.submit_all(subs, root=d, state_path=str(state_path),
                                         dry_run=True)
        assert results[0]["event"] == "would_skip", results
        assert not state_path.exists(), "dry-run must not touch the state file"


@check("submit_all: a runnable site line actually submits and records the job id")
def _():
    subs = [
        Submission(site="lumi", lane_id="lumi-l0", order=0.0, begin_s=0.0,
                  args_verbatim="", command="echo 'Submitted batch job 7'", comment=""),
        Submission(site="snellius", lane_id="snellius-l0", order=0.0, begin_s=0.0,
                  args_verbatim="", command="echo 'Submitted batch job 8'", comment=""),
    ]
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bindir:
        fake_bin(Path(bindir), "sbatch", "echo fake")
        os.environ["PWW_SITE"] = "lumi"
        try:
            state_path = Path(d) / "events.jsonl"
            with isolated_path(Path(bindir)):
                results = sub_mod.submit_all(subs, root=d, state_path=str(state_path))
            rows = st.read_events(state_path)  # while `d` still exists
        finally:
            del os.environ["PWW_SITE"]
    lumi_result = next(r for r in results if r["site"] == "lumi")
    snellius_result = next(r for r in results if r["site"] == "snellius")
    assert lumi_result["event"] == "submitted", lumi_result
    assert lumi_result["job_id"] == "7", lumi_result
    assert snellius_result["event"] == "submit_skipped", snellius_result
    assert len(rows) == 2, rows


def main() -> int:
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for name, exc in FAILED:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
