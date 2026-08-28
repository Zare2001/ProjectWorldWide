"""The scheduler's event log: append, read back, fold into per-job status.

    python3 tests/test_plan_state.py

No pytest, no network -- everything lives in a tempdir this file creates and
cleans up itself.
"""

from __future__ import annotations

import signal
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


@check("append_event then read_events round-trips, ts auto-filled")
def _():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "events.jsonl"
        ev = st.append_event(path, {"event": "submitted", "site": "lumi",
                                    "lane_id": "lumi-l0", "job_id": "1"})
        assert "ts" in ev
        rows = st.read_events(path)
        assert len(rows) == 1
        assert rows[0]["job_id"] == "1"


@check("read_events on a missing file is empty, not an error")
def _():
    assert st.read_events("/nonexistent/path/does-not-exist.jsonl") == []


@check("a malformed line is skipped, not fatal to the rest of the log")
def _():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "events.jsonl"
        st.append_event(path, {"event": "submitted", "site": "lumi", "lane_id": "l0"})
        with open(path, "a") as f:
            f.write("{not json\n")
        st.append_event(path, {"event": "submitted", "site": "snellius", "lane_id": "l0"})
        rows = st.read_events(path)
        assert len(rows) == 2, rows
        assert {r["site"] for r in rows} == {"lumi", "snellius"}


@check("fold: submitted -> status by job_id lands on the same row")
def _():
    events = [
        {"ts": 1, "event": "submitted", "site": "lumi", "role": "site",
         "lane_id": "lumi-l0", "link": 1, "job_id": "42"},
        {"ts": 2, "event": "status", "job_id": "42", "state": "RUNNING"},
        {"ts": 3, "event": "status", "job_id": "42", "state": "COMPLETED"},
    ]
    views = st.fold(events)
    assert len(views) == 1, views
    v = views[0]
    assert v.job_id == "42"
    assert v.site == "lumi" and v.lane_id == "lumi-l0"
    assert v.state == "completed"
    assert v.history == ("submitted", "running", "completed"), v.history


@check("fold: a skipped submission has no job_id and a reason, not a crash")
def _():
    events = [{"ts": 1, "event": "submit_skipped", "site": "snellius", "role": "site",
              "lane_id": "snellius-l0", "link": 1,
              "reason": "no sbatch on this host"}]
    views = st.fold(events)
    assert len(views) == 1
    assert views[0].job_id is None
    assert views[0].state == "skipped"
    assert "no sbatch" in views[0].reason


@check("fold: two lanes at the same site stay separate rows")
def _():
    events = [
        {"ts": 1, "event": "submitted", "site": "lumi", "role": "site",
         "lane_id": "lumi-l0", "link": 1, "job_id": "1"},
        {"ts": 2, "event": "submitted", "site": "lumi", "role": "site",
         "lane_id": "lumi-l1", "link": 1, "job_id": "2"},
    ]
    views = st.fold(events)
    assert len(views) == 2
    assert {v.lane_id for v in views} == {"lumi-l0", "lumi-l1"}
    assert {v.job_id for v in views} == {"1", "2"}


@check("fold: order is first-seen, not sorted, so a log reads chronologically")
def _():
    events = [
        {"ts": 1, "event": "submitted", "site": "snellius", "role": "site",
         "lane_id": "s-l0", "link": 1, "job_id": "9"},
        {"ts": 2, "event": "submitted", "site": "lumi", "role": "site",
         "lane_id": "l-l0", "link": 1, "job_id": "1"},
    ]
    views = st.fold(events)
    assert [v.site for v in views] == ["snellius", "lumi"]


@check("fold: a status event with an unknown job_id starts its own row rather than "
      "crashing")
def _():
    events = [{"ts": 1, "event": "status", "job_id": "999", "state": "PENDING",
              "site": "", "lane_id": ""}]
    views = st.fold(events)
    assert len(views) == 1
    assert views[0].state == "pending"


def main() -> int:
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for name, exc in FAILED:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
