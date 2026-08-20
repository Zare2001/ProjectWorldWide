"""Collector tests. No cluster required: sbatch and sacct output is fixed text."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "collector"))
import slurm_probe as probe


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_parse_duration_forms():
    assert probe.parse_duration("02:31:00") == 9060
    assert probe.parse_duration("1-00:00:00") == 86400
    assert probe.parse_duration("88-22:24:58") == 88 * 86400 + 22 * 3600 + 24 * 60 + 58
    assert probe.parse_duration("05:00") == 300


def test_unlimited_timelimit_is_none_not_infinity():
    # An UNLIMITED job has no ratio at all. Treating it as infinity would drag a
    # partition's usage to zero; treating it as a number would invent a limit.
    assert probe.parse_duration("UNLIMITED") is None
    assert probe.parse_duration("Partition_Limit") is None
    assert probe.parse_duration("") is None


def test_parse_time_roundtrip():
    import time
    stamp = probe.parse_time("2026-08-04T13:55:08")
    assert time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stamp)) == "2026-08-04T13:55:08"
    assert probe.parse_time("Unknown") is None


# --------------------------------------------------------------------------
# sbatch --test-only
# --------------------------------------------------------------------------

VERDICT = ("sbatch: Job 25196946 to start at 2026-08-04T13:55:08 using 16 "
           "processors on nodes gcn82 in partition gpu_h100")


def test_parse_test_only_success():
    now = probe.parse_time("2026-08-04T13:34:08")
    record = probe.parse_test_only(VERDICT, now)
    assert record["ok"] is True
    assert record["estimated_wait_sec"] == 21 * 60
    assert record["placed_partition"] == "gpu_h100"
    assert record["placed_nodes"] == "gcn82"


def test_parse_test_only_keeps_refusals_as_data():
    # A partition that cannot run the request is not an outage, and the
    # dashboard has to be able to tell the two apart.
    text = "sbatch: error: Batch job submission failed: Requested node configuration is not available"
    record = probe.parse_test_only(text, 1785843244)
    assert record["ok"] is False
    assert record["estimated_start"] is None
    assert "not available" in record["message"]


def test_parse_test_only_immediate_start():
    now = probe.parse_time("2026-08-04T13:55:08")
    record = probe.parse_test_only(VERDICT, now + 60)   # clock skew past the estimate
    assert record["estimated_wait_sec"] == 0            # never negative


def test_probe_runs_every_shape(monkeypatch):
    calls = []
    monkeypatch.setattr(probe, "sh", lambda cmd, timeout=0: calls.append(cmd) or VERDICT)
    payload = probe.probe({
        "cluster": "testsite",
        "shapes": [
            {"name": "one", "args": ["-p", "gpu_h100", "-t", "1:00:00"]},
            {"name": "two", "args": ["-p", "gpu_a100", "-N", "2"]},
        ],
    })
    assert [p["name"] for p in payload["probes"]] == ["one", "two"]
    assert payload["cluster"] == "testsite"
    assert calls[0] == ["sbatch", "--test-only", "-p", "gpu_h100", "-t", "1:00:00",
                        "--wrap", "true"]
    assert payload["probes"][1]["args"] == "-p gpu_a100 -N 2"


# --------------------------------------------------------------------------
# sacct usage
# --------------------------------------------------------------------------

def sacct(*rows):
    header = "Partition|State|End|Elapsed|Timelimit"
    return "\n".join([header] + ["|".join(r) for r in rows]) + "\n"


IN_WINDOW = "2026-08-04T12:00:00"
BEFORE_WINDOW = "2026-08-03T12:00:00"
WINDOW_START = probe.parse_time("2026-08-04T00:00:00")
WINDOW_END = probe.parse_time("2026-08-05T00:00:00")


def aggregate(text):
    return probe.aggregate(text, WINDOW_START, WINDOW_END)


def test_ratio_is_time_weighted():
    # One 24h job using 1h, plus ten 1h jobs using all of it. Time-weighted this
    # is 11/34; an unweighted mean of the per-job ratios would be ~0.92 and
    # would say the opposite about the machine.
    rows = [("gpu_h100", "COMPLETED", IN_WINDOW, "1:00:00", "1-00:00:00")]
    rows += [("gpu_h100", "COMPLETED", IN_WINDOW, "1:00:00", "1:00:00")] * 10
    result = aggregate(sacct(*rows))
    h100 = next(p for p in result if p["partition"] == "gpu_h100")
    assert h100["n_jobs"] == 11
    assert h100["used_ratio"] == round(11 / 34, 4)


def test_long_job_ending_in_window_counts():
    # Started before the window, finished inside it. This is the observation
    # that matters most, and selecting on start time would drop it.
    result = aggregate(sacct(("gpu_h100", "COMPLETED", IN_WINDOW, "2:00:00", "1-00:00:00")))
    assert result[0]["n_jobs"] == 1
    assert result[0]["used_ratio"] == round(7200 / 86400, 4)


def test_job_ending_outside_window_is_dropped():
    assert aggregate(sacct(("gpu_h100", "COMPLETED", BEFORE_WINDOW, "1:00:00", "1:00:00"))) == []


def test_running_jobs_are_dropped():
    # A RUNNING job's elapsed is a snapshot, not a measurement.
    text = sacct(("gpu_h100", "RUNNING", "Unknown", "3:00:00", "1-00:00:00"))
    assert aggregate(text) == []


def test_timeout_counted_separately():
    # A job killed at its limit is a ratio of 1.0 for a different reason than
    # one that genuinely needed its full request.
    result = aggregate(sacct(("gpu_h100", "TIMEOUT", IN_WINDOW, "1:00:00", "1:00:00")))
    assert result[0]["n_timeout"] == 1
    assert result[0]["used_ratio"] == 1.0


def test_ratio_clamped_to_one():
    # Elapsed can exceed the limit slightly during teardown.
    result = aggregate(sacct(("gpu_h100", "COMPLETED", IN_WINDOW, "1:00:30", "1:00:00")))
    assert result[0]["used_ratio"] == 1.0


def test_unlimited_job_excluded_not_zeroed():
    text = sacct(("gpu_h100", "COMPLETED", IN_WINDOW, "1:00:00", "UNLIMITED"),
                 ("gpu_h100", "COMPLETED", IN_WINDOW, "0:30:00", "1:00:00"))
    result = aggregate(text)
    assert result[0]["n_jobs"] == 1
    assert result[0]["used_ratio"] == 0.5


def test_partitions_reported_separately():
    text = sacct(("gpu_h100", "COMPLETED", IN_WINDOW, "0:30:00", "1:00:00"),
                 ("gpu_a100", "COMPLETED", IN_WINDOW, "0:15:00", "1:00:00"))
    result = aggregate(text)
    assert [p["partition"] for p in result] == ["gpu_a100", "gpu_h100"]
    assert result[0]["used_ratio"] == 0.25
    assert result[1]["used_ratio"] == 0.5


def test_empty_sacct_output_is_not_an_error():
    assert aggregate("") == []
    assert aggregate("Partition|State|End|Elapsed|Timelimit\n") == []


def test_usage_window_is_recomputed_each_run(monkeypatch):
    seen = {}
    monkeypatch.setattr(probe, "sh", lambda cmd, timeout=0: seen.setdefault("cmd", cmd) and "" or "")
    payload = probe.usage({"cluster": "c", "partitions": ["gpu_h100"], "usage_hours": 48})
    # The bounds ride along on the payload precisely because consecutive runs
    # overlap: without them a reader could not tell these apart from increments.
    assert payload["window_end"] - payload["window_start"] == 48 * 3600
    assert "-r" in seen["cmd"] and "gpu_h100" in seen["cmd"]


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def write_config(tmp_path, monkeypatch, data, mode=0o600):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    os.chmod(path, mode)
    monkeypatch.setattr(probe, "CONFIG_PATH", str(path))
    return path


VALID = {"cluster": "snellius", "partitions": ["gpu_h100"],
         "shapes": [{"name": "a", "args": ["-p", "gpu_h100"]}]}


def test_config_defaults_usage_hours(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, VALID)
    assert probe.load_config()["usage_hours"] == 48


@pytest.mark.parametrize("missing", ["cluster", "partitions", "shapes"])
def test_config_requires_key(tmp_path, monkeypatch, missing):
    data = {k: v for k, v in VALID.items() if k != missing}
    write_config(tmp_path, monkeypatch, data)
    with pytest.raises(SystemExit) as excinfo:
        probe.load_config()
    assert missing in str(excinfo.value)


def test_config_with_readable_token_is_refused(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, dict(VALID, token="secret"), mode=0o644)
    with pytest.raises(SystemExit) as excinfo:
        probe.load_config()
    assert "chmod" in str(excinfo.value)


def test_config_without_token_ignores_permissions(tmp_path, monkeypatch):
    write_config(tmp_path, monkeypatch, VALID, mode=0o644)
    assert probe.load_config()["cluster"] == "snellius"
