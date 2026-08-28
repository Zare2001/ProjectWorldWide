"""monitor.py: DARL liveness/progress, squeue parsing, and the merge into one log.

    python3 tests/test_plan_monitor.py

No real network and no real squeue: DARL's HTTP call is monkeypatched at
`pww.plan.inputs.fetch_json` (looked up dynamically by `monitor.py`'s `io.` calls,
so patching the module attribute is enough -- no need to patch monitor's own
namespace) and squeue is a throwaway PATH entry, same trick as test_plan_submit.py.
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


import pww.plan.inputs as inputs_mod  # noqa: E402
from pww.plan import monitor as mon, state as st  # noqa: E402

BLOCK_SIZE, SEQ_LEN = 1024, 2048


class fake_fetch_json:
    """Monkeypatch `pww.plan.inputs.fetch_json` for the duration of a `with`
    block, so darl_snapshot never makes a real HTTP call."""

    def __init__(self, payload: dict):
        self.payload = payload
        self._old = None

    def __enter__(self):
        self._old = inputs_mod.fetch_json
        inputs_mod.fetch_json = lambda *a, **kw: self.payload
        return self

    def __exit__(self, *exc):
        inputs_mod.fetch_json = self._old


def fake_squeue_bin(dir_path: Path, lines: list[str]) -> None:
    body = "\n".join(f'echo "{line}"' for line in lines)
    path = dir_path / "squeue"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


class isolated_path:
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
# darl_snapshot
# --------------------------------------------------------------------------


@check("darl_snapshot: a fresh last_seen is alive, a stale one is not")
def _():
    import time as _time
    now = _time.time()  # darl_snapshot has no `now` override -- it always reads
                        # real wall-clock time via io.darl_liveness, so the fixture
                        # has to anchor on the same clock rather than a fictional one
    payload = {
        "epoch": 0, "max_epochs": 4, "committed": 1000, "num_blocks": 10768,
        "unassigned": 9768,
        "clusters": {
            "lumi-l0": {"blocks_committed": 500, "ranks": 8, "last_seen": now - 10},
            "lumi-l1": {"blocks_committed": 300, "ranks": 8, "last_seen": now - 10_000},
        },
    }
    with fake_fetch_json(payload):
        snap = mon.darl_snapshot("http://fake:29510", None, block_size=BLOCK_SIZE,
                                 seq_len=SEQ_LEN, fresh_within_s=300.0)
    assert snap["lumi-l0"]["alive"] is True, snap
    assert snap["lumi-l1"]["alive"] is False, snap
    assert snap["lumi-l0"]["tokens"] == 500 * BLOCK_SIZE * SEQ_LEN
    assert snap["_epoch"]["max_epochs"] == 4


# --------------------------------------------------------------------------
# squeue_snapshot
# --------------------------------------------------------------------------


@check("squeue_snapshot: parses pipe-delimited rows and filters by name prefix")
def _():
    with tempfile.TemporaryDirectory() as bindir:
        fake_squeue_bin(Path(bindir), [
            "12345|pww-lumi-titan-l0|RUNNING",
            "12346|pww-lumi-titan-l1|PENDING",
            "99999|someone-elses-job|RUNNING",
        ])
        with isolated_path(Path(bindir)):
            rows = mon.squeue_snapshot(name_prefix="pww-")
    assert len(rows) == 2, rows
    assert {r["job_id"] for r in rows} == {"12345", "12346"}
    assert all(r["name"].startswith("pww-") for r in rows)


@check("squeue_snapshot: no squeue on PATH raises, rather than reporting empty")
def _():
    with tempfile.TemporaryDirectory() as empty:
        with isolated_path(Path(empty)):
            try:
                mon.squeue_snapshot()
                raise AssertionError("expected RuntimeError")
            except RuntimeError as exc:
                assert "login node" in str(exc)


# --------------------------------------------------------------------------
# render: pure formatting
# --------------------------------------------------------------------------


@check("render: an unreachable/absent source is just omitted, not a crash")
def _():
    out = mon.render(None, None, None)
    assert "nothing to report" in out


@check("render: a stale DARL cluster is flagged NO, not silently dropped")
def _():
    darl = {"lumi-l0": {"alive": True, "blocks_committed": 5, "tokens": 5 * BLOCK_SIZE * SEQ_LEN,
                        "ranks": 8, "last_seen_age_s": 3.0},
           "lumi-l1": {"alive": False, "blocks_committed": 2, "tokens": 2 * BLOCK_SIZE * SEQ_LEN,
                        "ranks": 8, "last_seen_age_s": 9999.0},
           "_epoch": {"epoch": 0, "max_epochs": 4, "committed": 7, "num_blocks": 100,
                      "unassigned": 93}}
    out = mon.render(darl, None, None)
    assert "lumi-l0" in out and "yes" in out
    assert "lumi-l1" in out and "NO" in out


# --------------------------------------------------------------------------
# poll_once: the merge into the state log
# --------------------------------------------------------------------------


@check("poll_once: DARL + squeue events land in the same state file, folded together")
def _():
    payload = {
        "epoch": 0, "max_epochs": 1, "committed": 5, "num_blocks": 100, "unassigned": 95,
        "clusters": {"lumi-l0": {"blocks_committed": 5, "ranks": 8, "last_seen": __import__("time").time()}},
    }
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as bindir:
        fake_squeue_bin(Path(bindir), ["777|pww-lumi-titan-l0|RUNNING"])
        state_path = str(Path(d) / "events.jsonl")
        with fake_fetch_json(payload), isolated_path(Path(bindir)):
            out = mon.poll_once(darl_url="http://fake:29510", darl_token=None,
                                block_size=BLOCK_SIZE, seq_len=SEQ_LEN, use_squeue=True,
                                state_path=state_path, name_prefix="pww-")
        events = st.read_events(state_path)
    assert any(e.get("source") == "darl" for e in events), events
    assert any(e.get("source") == "squeue" for e in events), events
    assert "TRACKED JOBS" in out


def main() -> int:
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for name, exc in FAILED:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
