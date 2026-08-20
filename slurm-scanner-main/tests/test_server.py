"""Server tests: CSV storage and the ingest/query endpoints."""

import importlib
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TOKEN = "test-token"


def plan_config(directory):
    """plan.json is required at startup. Nothing here plans, so it declares no
    clusters -- which is also the supported way to run ingest-only."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plan.json").write_text('{"clusters": []}')
    return directory


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_SCANNER_DATA_DIR", str(plan_config(tmp_path)))
    monkeypatch.setenv("SLURM_SCANNER_TOKENS", TOKEN)
    monkeypatch.delenv("SLURM_SCANNER_PLAN_CONFIG", raising=False)
    from server import app as module
    importlib.reload(module)          # config is read at import time
    return TestClient(module.app)


def auth(**kwargs):
    return dict(headers={"Authorization": "Bearer " + TOKEN}, **kwargs)


def probe_payload(cluster="snellius", at=1785843244, wait=1260, ok=True, name="h100_1gpu_1h"):
    return {
        "cluster": cluster, "collected_at": at, "collector_version": "2.0.0",
        "probed_by_user": "douwew",
        "probes": [{
            "name": name, "args": "-p gpu_h100 -t 1:00:00", "ok": ok,
            "estimated_start": at + wait if ok else None,
            "estimated_wait_sec": wait if ok else None,
            "placed_partition": "gpu_h100" if ok else None,
            "placed_nodes": "gcn82" if ok else None,
            "message": "sbatch: Job 1 to start at ..." if ok else "sbatch: error: nope",
        }],
    }


def usage_payload(cluster="snellius", at=1785843244, ratio=0.112, partition="gpu_h100"):
    return {
        "cluster": cluster, "collected_at": at, "collector_version": "2.0.0",
        "window_start": at - 172800, "window_end": at, "window_hours": 48,
        "partitions": [{
            "partition": partition, "n_jobs": 240, "n_timeout": 3,
            "sum_elapsed_sec": 43581, "sum_timelimit_sec": 388400,
            "used_ratio": ratio,
        }],
    }


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------


def test_ingest_requires_a_known_token(client):
    assert client.post("/ingest/probe", json=probe_payload()).status_code == 401
    bad = {"Authorization": "Bearer wrong"}
    assert client.post("/ingest/probe", json=probe_payload(), headers=bad).status_code == 401


def test_ingest_fails_closed_with_no_tokens_configured(tmp_path, monkeypatch):
    # An unconfigured server must not silently collect numbers a scheduler will
    # act on, so this is 503 rather than an open door.
    monkeypatch.setenv("SLURM_SCANNER_DATA_DIR", str(plan_config(tmp_path)))
    monkeypatch.setenv("SLURM_SCANNER_TOKENS", "")
    monkeypatch.delenv("SLURM_SCANNER_PLAN_CONFIG", raising=False)
    from server import app as module
    importlib.reload(module)
    response = TestClient(module.app).post("/ingest/probe", json=probe_payload())
    assert response.status_code == 503


def test_a_tilde_in_the_data_dir_is_expanded_not_taken_literally(tmp_path, monkeypatch):
    # Otherwise the CSVs land in a directory named "~" under the working
    # directory, which looks like success.
    monkeypatch.setenv("HOME", str(plan_config(tmp_path / "home")))
    monkeypatch.setenv("SLURM_SCANNER_DATA_DIR", "~")
    monkeypatch.setenv("SLURM_SCANNER_TOKENS", TOKEN)
    monkeypatch.delenv("SLURM_SCANNER_PLAN_CONFIG", raising=False)
    from server import app as module
    importlib.reload(module)
    assert module.DATA_DIR == tmp_path / "home"

    TestClient(module.app).post("/ingest/probe", **auth(json=probe_payload()))
    assert (tmp_path / "home" / "snellius" / "probes.csv").exists()
    assert not (tmp_path / "~").exists()


def test_query_needs_no_token(client):
    assert client.get("/overview").status_code == 200
    assert client.get("/healthz").json()["ok"] is True


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


def test_probe_roundtrip(client, tmp_path):
    assert client.post("/ingest/probe", **auth(json=probe_payload())).json()["rows"] == 1
    assert (tmp_path / "snellius" / "probes.csv").exists()

    rows = client.get("/probes?hours=100000").json()["rows"]
    assert len(rows) == 1
    assert rows[0]["cluster"] == "snellius"
    assert rows[0]["estimated_wait_sec"] == 1260
    assert rows[0]["ok"] is True
    # Payload-level fields are flattened onto each row so a row stands alone.
    assert rows[0]["probed_by_user"] == "douwew"


def test_usage_roundtrip(client):
    client.post("/ingest/usage", **auth(json=usage_payload()))
    rows = client.get("/usage?hours=100000").json()["rows"]
    assert rows[0]["used_ratio"] == 0.112
    assert rows[0]["window_hours"] == 48
    assert rows[0]["n_jobs"] == 240


def test_appends_accumulate_and_header_written_once(client, tmp_path):
    for n in range(3):
        client.post("/ingest/probe", **auth(json=probe_payload(at=1785843244 + n * 600)))
    text = (tmp_path / "snellius" / "probes.csv").read_text()
    assert text.count("collected_at") == 1
    assert len(client.get("/probes?hours=100000").json()["rows"]) == 3


def test_failed_probe_is_stored_not_dropped(client):
    client.post("/ingest/probe", **auth(json=probe_payload(ok=False)))
    row = client.get("/probes?hours=100000").json()["rows"][0]
    assert row["ok"] is False
    assert row["estimated_wait_sec"] is None
    assert "nope" in row["message"]


def test_clusters_are_separate_files(client, tmp_path):
    client.post("/ingest/probe", **auth(json=probe_payload(cluster="snellius")))
    client.post("/ingest/probe", **auth(json=probe_payload(cluster="vega")))
    assert sorted(d.name for d in tmp_path.iterdir() if d.is_dir()) == ["snellius", "vega"]
    assert len(client.get("/probes?cluster=vega&hours=100000").json()["rows"]) == 1
    assert len(client.get("/probes?hours=100000").json()["rows"]) == 2


def test_plan_json_beside_the_data_is_not_taken_for_a_cluster(client):
    # plan.json shares the data directory, and the scan only looks at
    # subdirectories -- otherwise it would show up as a cluster named plan.json.
    client.post("/ingest/probe", **auth(json=probe_payload(cluster="snellius")))
    assert client.get("/healthz").json()["clusters"] == ["snellius"]


def test_rejects_payload_without_cluster(client):
    payload = probe_payload()
    del payload["cluster"]
    assert client.post("/ingest/probe", **auth(json=payload)).status_code == 400


def test_rejects_payload_without_items(client):
    assert client.post("/ingest/probe", **auth(json={"cluster": "x"})).status_code == 400


def test_rejects_cluster_name_that_escapes_the_data_dir(client, tmp_path):
    # The cluster name becomes a directory name, so it cannot be trusted.
    payload = probe_payload(cluster="../../etc")
    assert client.post("/ingest/probe", **auth(json=payload)).status_code == 400
    assert not (tmp_path.parent / "etc").exists()


def test_hours_filter_cuts_old_rows(client):
    now = int(time.time())
    client.post("/ingest/probe", **auth(json=probe_payload(at=now - 86400)))
    client.post("/ingest/probe", **auth(json=probe_payload(at=now - 60)))
    assert len(client.get("/probes?hours=1").json()["rows"]) == 1
    assert len(client.get("/probes?hours=48").json()["rows"]) == 2


# --------------------------------------------------------------------------
# overview
# --------------------------------------------------------------------------


def test_overview_shows_latest_probe_per_shape(client):
    now = int(time.time())
    client.post("/ingest/probe", **auth(json=probe_payload(at=now - 600, wait=9000)))
    client.post("/ingest/probe", **auth(json=probe_payload(at=now, wait=1260)))
    rows = client.get("/overview").json()["rows"]
    assert len(rows) == 1
    assert rows[0]["estimated_wait_sec"] == 1260
    assert rows[0]["age_sec"] < 60


def test_overview_pairs_the_probe_with_its_partition_usage(client):
    now = int(time.time())
    client.post("/ingest/probe", **auth(json=probe_payload(at=now)))
    client.post("/ingest/usage", **auth(json=usage_payload(at=now, ratio=0.112)))
    row = client.get("/overview").json()["rows"][0]
    # Both numbers are present and neither is folded into the other.
    assert row["estimated_wait_sec"] == 1260
    assert row["used_ratio"] == 0.112
    assert row["used_ratio_n_jobs"] == 240


def test_overview_without_usage_still_reports_the_estimate(client):
    client.post("/ingest/probe", **auth(json=probe_payload(at=int(time.time()))))
    row = client.get("/overview").json()["rows"][0]
    assert row["estimated_wait_sec"] == 1260
    assert row["used_ratio"] is None


def test_overview_does_not_attach_another_partitions_usage(client):
    now = int(time.time())
    client.post("/ingest/probe", **auth(json=probe_payload(at=now)))
    client.post("/ingest/usage", **auth(json=usage_payload(at=now, partition="gpu_a100")))
    assert client.get("/overview").json()["rows"][0]["used_ratio"] is None


def test_overview_spans_clusters(client):
    now = int(time.time())
    client.post("/ingest/probe", **auth(json=probe_payload(cluster="snellius", at=now)))
    client.post("/ingest/probe", **auth(json=probe_payload(cluster="vega", at=now, wait=60)))
    rows = client.get("/overview").json()["rows"]
    assert {r["cluster"]: r["estimated_wait_sec"] for r in rows} == {
        "snellius": 1260, "vega": 60}


def test_clusters_endpoint_lists_shapes_and_freshness(client):
    now = int(time.time())
    client.post("/ingest/probe", **auth(json=probe_payload(at=now, name="a")))
    client.post("/ingest/probe", **auth(json=probe_payload(at=now, name="b")))
    client.post("/ingest/usage", **auth(json=usage_payload(at=now)))
    entry = client.get("/clusters").json()["clusters"][0]
    assert entry["shapes"] == ["a", "b"]
    assert entry["last_probe"] == now
    assert entry["last_usage"] == now


def test_dashboard_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "slurm_scanner" in response.text


def test_dashboard_carries_the_planner(client):
    # The page asks /plan for the allocation it draws, so it has to be the one
    # served here -- a dashboard planning against its own arithmetic would be a
    # second implementation to keep in step.
    page = client.get("/").text
    assert 'id="plan-form"' in page
    assert '"/plan"' in page
