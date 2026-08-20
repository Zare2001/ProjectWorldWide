"""Planner tests: the waterfill on its own, then the endpoint end to end."""

import importlib
import json
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import plan as planner

TOKEN = "test-token"

# Snellius: starts soon, slow. Frontier: starts late, four times faster.
CLUSTERS = [
    {"cluster": "snellius", "shape": "snellius_shape",
     "units_per_gpu_per_hour": 10, "shapes": {"snellius_shape": 1},
     "max_walltime_hours": 120},
    {"cluster": "frontier", "shape": "frontier_shape",
     "units_per_gpu_per_hour": 10, "shapes": {"frontier_shape": 4},
     "max_walltime_hours": 120},
]


def site(cluster="snellius", rate=10.0, wait_hours=2.0, min_units=0,
         max_walltime=120.0, reserve_hours=0.0, gpus=1, units_per_gpu_per_hour=None):
    usable = max_walltime - reserve_hours
    if units_per_gpu_per_hour is None:
        units_per_gpu_per_hour = rate / gpus
    return {
        "cluster": cluster, "shape": cluster + "_shape", "partition": "gpu",
        "gpus": gpus, "units_per_gpu_per_hour": units_per_gpu_per_hour,
        "units_per_hour": rate, "max_walltime_hours": max_walltime,
        "min_units": min_units, "wait_raw_sec": wait_hours * 3600,
        "wait_eff_sec": wait_hours * 3600, "wait_hours": wait_hours,
        "used_ratio": None, "probe_age_sec": 60, "reserve_hours": reserve_hours,
        "usable_hours": usable, "cap_units": int(usable * rate),
    }


# --------------------------------------------------------------------------
# the waterfill
# --------------------------------------------------------------------------


def test_waterfill_gives_the_early_cluster_more_than_its_share_of_rate():
    # 100 units over (wait 2 h, 10 u/h) and (wait 10 h, 40 u/h) meet at a 10.4 h
    # horizon: 84 units on the slow-but-early site, 16 on the fast-but-late one.
    sites = [site("snellius", 10, 2), site("frontier", 40, 10)]
    assert planner.split(sites, 100) == [84, 16]


def test_a_cluster_past_the_horizon_gets_nothing():
    # Snellius alone finishes 20 units at 4 h, before Frontier has even started.
    sites = [site("snellius", 10, 2), site("frontier", 40, 10)]
    assert planner.split(sites, 20) == [20, 0]


def test_horizon_is_the_moment_the_last_unit_lands():
    sites = [site("snellius", 10, 2), site("frontier", 40, 10)]
    assert planner.horizon_for(sites, 100) == pytest.approx(10.4, abs=1e-6)


def test_walltime_limit_caps_a_cluster_and_the_rest_absorb_it():
    # 8 h of walltime at 10 u/h is 80 units however long the horizon grows.
    sites = [site("snellius", 10, 0, max_walltime=8),
             site("frontier", 40, 10, max_walltime=120)]
    counts = planner.split(sites, 200)
    assert counts[0] == 80
    assert sum(counts) == 200


def test_shares_are_integers_that_sum_to_the_request():
    sites = [site("a", 7, 1), site("b", 13, 3), site("c", 3, 0.5)]
    for units in range(1, 200):
        counts = planner.split(sites, units)
        assert sum(counts) == units
        assert all(isinstance(c, int) for c in counts)


def test_a_request_larger_than_the_clusters_can_hold_is_short_not_wrong():
    sites = [site("snellius", 10, 0, max_walltime=10)]      # 100 units, ever
    assert planner.split(sites, 500) == [100]


def test_min_units_moves_a_small_share_to_the_clusters_that_stay():
    # Frontier's honest share is 16 units, under its 20-unit minimum, so all 100
    # go to Snellius as a longer job.
    sites = [site("snellius", 10, 2, min_units=20),
             site("frontier", 40, 10, min_units=20)]
    counts, active, dropped = planner.assign(sites, 100)
    assert counts == [100]
    assert [s["cluster"] for s in active] == ["snellius"]
    assert dropped[0]["cluster"] == "frontier"
    assert "min_units 20" in dropped[0]["reason"]


def test_min_units_is_not_honoured_when_it_would_orphan_data():
    # Snellius cannot hold 200 units on its own, so Frontier keeps its small
    # share rather than the data going unprocessed.
    sites = [site("snellius", 10, 0, min_units=20, max_walltime=15),
             site("frontier", 40, 10, min_units=500)]
    counts, active, dropped = planner.assign(sites, 200)
    assert dropped == []
    assert sum(counts) == 200


def test_dropping_cascades_until_every_survivor_clears_its_minimum():
    sites = [site("a", 10, 0, min_units=50), site("b", 10, 8, min_units=50),
             site("c", 10, 9, min_units=50)]
    counts, active, dropped = planner.assign(sites, 100)
    assert [s["cluster"] for s in active] == ["a"]
    assert counts == [100]
    assert len(dropped) == 2


# --------------------------------------------------------------------------
# the discount
# --------------------------------------------------------------------------


def test_discount_strength_interpolates_between_the_two_numbers():
    assert planner.discount(1000, 0.1, 0.0) == 1000       # trust the estimate
    assert planner.discount(1000, 0.1, 1.0) == 100        # trust the ratio
    assert planner.discount(1000, 0.1, 0.5) == 550        # split the difference


def test_a_missing_or_absurd_ratio_leaves_the_estimate_alone():
    assert planner.discount(1000, None, 1.0) == 1000
    assert planner.discount(1000, 0, 1.0) == 1000
    assert planner.discount(1000, 1.4, 1.0) == 1000


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def write_config(tmp_path, clusters=None, defaults=None):
    body = {"clusters": clusters if clusters is not None else CLUSTERS}
    if defaults:
        body["defaults"] = defaults
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(body))
    return path


def test_config_is_found_next_to_the_data_without_anything_being_set(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_SCANNER_PLAN_CONFIG", raising=False)
    assert planner.config_path(tmp_path) == tmp_path / "plan.json"


def test_the_env_var_overrides_the_location(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_SCANNER_PLAN_CONFIG", "/etc/elsewhere.json")
    assert planner.config_path(tmp_path) == Path("/etc/elsewhere.json")


def test_a_literal_tilde_resolves_to_the_home_directory(monkeypatch):
    # Unquoted bash expands ~ before the server ever sees it, but a quoted value
    # or a systemd Environment= line does not. Left alone it would become a
    # relative directory *named* "~" and the server would look like it worked.
    home = Path.home()
    monkeypatch.delenv("SLURM_SCANNER_PLAN_CONFIG", raising=False)
    assert planner.config_path("~/.slurm_scanner") == home / ".slurm_scanner/plan.json"
    monkeypatch.setenv("SLURM_SCANNER_PLAN_CONFIG", "~/.slurm_scanner/plan.json")
    assert planner.config_path("/ignored") == home / ".slurm_scanner/plan.json"


def test_a_missing_config_is_a_startup_error(tmp_path):
    # Not a runtime 503: a scheduler that silently has no clusters is worse than
    # one that refuses to boot.
    with pytest.raises(ValueError) as error:
        planner.load_config(tmp_path / "plan.json")
    assert "plan.example.json" in str(error.value)


def test_no_clusters_is_a_deliberate_way_to_run_without_planning(tmp_path):
    config = planner.load_config(write_config(tmp_path, clusters=[]))
    assert config["clusters"] == []


def test_config_defaults_fill_in_the_unset_knobs(tmp_path):
    config = planner.load_config(write_config(tmp_path, defaults={"min_units": 7}))
    assert config["defaults"]["min_units"] == 7
    assert config["defaults"]["discount_strength"] == planner.DEFAULTS["discount_strength"]


@pytest.mark.parametrize("clusters, defaults", [
    ([{"units_per_gpu_per_hour": 1, "shapes": {"s": 1}, "max_walltime_hours": 1}], None),
    ([{"cluster": "x", "shapes": {"s": 1}, "units_per_gpu_per_hour": 0, "max_walltime_hours": 1}], None),
    ([{"cluster": "x", "shapes": {"s": 1}, "units_per_gpu_per_hour": 1, "max_walltime_hours": -1}], None),
    ([{"cluster": "x", "shapes": {"s": 0}, "units_per_gpu_per_hour": 1, "max_walltime_hours": 1}], None),
    ([{"cluster": "x", "shapes": {}, "units_per_gpu_per_hour": 1, "max_walltime_hours": 1}], None),
    ([{"cluster": "x", "shape": "other", "shapes": {"s": 1}, "units_per_gpu_per_hour": 1, "max_walltime_hours": 1}], None),
    ([{"cluster": "x", "units_per_gpu_per_hour": 1, "max_walltime_hours": 1}], None),
    ([dict(CLUSTERS[0]), dict(CLUSTERS[0])], None),
    (None, {"discount_strength": 1.5}),
    (None, {"reserve_units": -1}),
    (None, {"max_gpus": -1}),
])
def test_a_broken_config_refuses_to_load(tmp_path, clusters, defaults):
    # Loading happens at startup, so a bad file stops the server rather than
    # producing plans against numbers nobody meant.
    with pytest.raises(ValueError):
        planner.load_config(write_config(tmp_path, clusters, defaults))


def test_an_unknown_request_parameter_is_rejected(tmp_path):
    config = planner.load_config(write_config(tmp_path))
    with pytest.raises(HTTPException):
        planner.merge_params(config, {"discount_strenght": 0.5})


# --------------------------------------------------------------------------
# the endpoint
# --------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_SCANNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SLURM_SCANNER_TOKENS", TOKEN)
    monkeypatch.setenv("SLURM_SCANNER_PLAN_CONFIG", str(write_config(tmp_path)))
    from server import app as module
    importlib.reload(module)          # config is read at import time
    return TestClient(module.app)


def feed(client, cluster, shape, wait_sec, ratio=None, age_sec=60):
    at = int(time.time()) - age_sec
    headers = {"Authorization": "Bearer " + TOKEN}
    client.post("/ingest/probe", headers=headers, json={
        "cluster": cluster, "collected_at": at, "collector_version": "2.0.0",
        "probes": [{"name": shape, "args": "-p gpu", "ok": True,
                    "estimated_start": at + wait_sec,
                    "estimated_wait_sec": wait_sec, "placed_partition": "gpu",
                    "placed_nodes": "n1", "message": "ok"}],
    })
    if ratio is not None:
        client.post("/ingest/usage", headers=headers, json={
            "cluster": cluster, "collected_at": at, "collector_version": "2.0.0",
            "window_start": at - 172800, "window_end": at, "window_hours": 48,
            "partitions": [{"partition": "gpu", "n_jobs": 100, "n_timeout": 0,
                            "sum_elapsed_sec": 100, "sum_timelimit_sec": 1000,
                            "used_ratio": ratio}],
        })


def test_plan_splits_across_the_clusters_it_has_numbers_for(client):
    feed(client, "snellius", "snellius_shape", 7200)         # 2 h
    feed(client, "frontier", "frontier_shape", 36000)        # 10 h
    body = client.post("/plan", json={"units": 100, "discount_strength": 0}).json()
    assert body["feasible"] is True
    assert {r["cluster"]: r["units"] for r in body["clusters"]} == {
        "snellius": 84, "frontier": 16}
    assert body["horizon_hours"] == pytest.approx(10.4, abs=0.01)
    assert body["unassigned_units"] == 0


def test_plan_needs_only_units(client):
    feed(client, "snellius", "snellius_shape", 7200)
    body = client.post("/plan", json={"units": 50}).json()
    assert body["clusters"][0]["units"] == 50
    assert body["params"] == planner.DEFAULTS


def test_walltime_covers_the_share_plus_the_reserve(client):
    feed(client, "snellius", "snellius_shape", 7200)
    body = client.post("/plan", json={"units": 100, "reserve_units": 20}).json()
    row = body["clusters"][0]
    assert row["walltime_hours"] == pytest.approx(12.0)      # (100 + 20) / 10
    assert row["reserve_hours"] == pytest.approx(2.0)
    # The horizon is when the work is predicted to finish; the reserve sits past
    # it, unused unless the job starts late.
    assert body["horizon_hours"] == pytest.approx(12.0)      # 2 h wait + 10 h
    assert body["latest_finish_hours"] == pytest.approx(14.0)


def test_every_cluster_carries_the_reserve_at_its_own_rate(client):
    # The same spare capacity everywhere -- 40 units -- costing each cluster the
    # hours 40 units actually take it. Nothing is rounded up: the fast cluster's
    # reserve is exactly a quarter of the slow one's.
    feed(client, "snellius", "snellius_shape", 0)
    feed(client, "frontier", "frontier_shape", 0)
    body = client.post("/plan", json={"units": 100, "reserve_units": 40}).json()
    reserve = {r["cluster"]: r["reserve_hours"] for r in body["clusters"]}
    assert reserve == {"snellius": 4.0, "frontier": 1.0}
    spare = {r["cluster"]: r["reserve_hours"] * r["units_per_hour"]
             for r in body["clusters"]}
    assert spare == {"snellius": 40.0, "frontier": 40.0}


def test_a_small_reserve_on_a_fast_cluster_stays_small(client):
    # 10 units at 40 units/h is 15 minutes, and it is reported as 0.25 h rather
    # than rounded to anything coarser.
    feed(client, "frontier", "frontier_shape", 0)
    body = client.post("/plan", json={"units": 40, "reserve_units": 10}).json()
    row = body["clusters"][0]
    assert row["reserve_hours"] == pytest.approx(0.25)
    assert row["walltime_hours"] == pytest.approx(1.25)      # 40 / 40 h + 0.25 h


def test_the_reserve_never_pushes_a_job_over_the_walltime_limit(client):
    # usable_hours is cut before the waterfill runs, so no plan can ask a site for
    # a job it would reject; the cost lands on how much work fits instead.
    feed(client, "snellius", "snellius_shape", 0)
    body = client.post("/plan", json={"units": 2000, "reserve_units": 400}).json()
    row = body["clusters"][0]
    assert row["reserve_hours"] == pytest.approx(40.0)       # 400 / 10 u/h
    assert row["walltime_hours"] == pytest.approx(120.0)     # exactly the limit
    assert row["units"] == 800                               # (120 - 40) h x 10 u/h
    assert body["feasible"] is False


def test_a_cluster_whose_walltime_cannot_hold_the_reserve_is_excluded(client):
    # 1300 units is 130 h on Snellius, past its 120 h limit, but only 32.5 h on
    # Frontier -- so this excludes one cluster and not the other.
    feed(client, "snellius", "snellius_shape", 0)
    feed(client, "frontier", "frontier_shape", 0)
    body = client.post("/plan", json={"units": 100, "reserve_units": 1300}).json()
    assert [r["cluster"] for r in body["clusters"]] == ["frontier"]
    reasons = {r["cluster"]: r["reason"] for r in body["excluded"]}
    assert "over the 120 h walltime limit" in reasons["snellius"]


def test_the_discount_pulls_a_busy_cluster_forward(client):
    feed(client, "snellius", "snellius_shape", 7200)
    feed(client, "frontier", "frontier_shape", 36000, ratio=0.1)
    full = client.post("/plan", json={"units": 100, "discount_strength": 1}).json()
    none = client.post("/plan", json={"units": 100, "discount_strength": 0}).json()
    share = lambda body: {r["cluster"]: r["units"] for r in body["clusters"]}
    assert share(full)["frontier"] > share(none)["frontier"]
    row = [r for r in full["clusters"] if r["cluster"] == "frontier"][0]
    # Both numbers stay in the response, so the blend can be checked by hand.
    assert row["wait_raw_sec"] == 36000
    assert row["used_ratio"] == 0.1
    assert row["wait_eff_sec"] == 3600


def test_a_stale_probe_is_excluded_with_a_reason(client):
    feed(client, "snellius", "snellius_shape", 7200)
    feed(client, "frontier", "frontier_shape", 7200, age_sec=30 * 3600)
    body = client.post("/plan", json={"units": 100}).json()
    assert [r["cluster"] for r in body["clusters"]] == ["snellius"]
    excluded = {e["cluster"]: e["reason"] for e in body["excluded"]}
    assert "old" in excluded["frontier"]


def test_a_cluster_with_no_probe_yet_is_excluded_not_assumed_idle(client):
    feed(client, "snellius", "snellius_shape", 7200)
    body = client.post("/plan", json={"units": 100}).json()
    assert "no probe" in {e["cluster"]: e["reason"] for e in body["excluded"]}["frontier"]


def test_a_failed_probe_carries_slurms_message_into_the_exclusion(client):
    at = int(time.time())
    client.post("/ingest/probe", headers={"Authorization": "Bearer " + TOKEN}, json={
        "cluster": "snellius", "collected_at": at, "collector_version": "2.0.0",
        "probes": [{"name": "snellius_shape", "ok": False, "estimated_wait_sec": None,
                    "message": "sbatch: error: drained"}],
    })
    body = client.post("/plan", json={"units": 10}).json()
    assert body["clusters"] == []
    assert "drained" in body["excluded"][0]["reason"]
    assert body["feasible"] is False
    assert body["unassigned_units"] == 10
    assert body["horizon_hours"] is None


def test_nothing_to_plan_on_is_reported_rather_than_failed(client):
    body = client.post("/plan", json={"units": 10})
    assert body.status_code == 200
    assert body.json()["feasible"] is False


def test_plan_rejects_a_bad_unit_count(client):
    for units in (0, -5, 1.5, "many", None):
        assert client.post("/plan", json={"units": units}).status_code == 400


def test_the_server_will_not_start_without_a_plan_config(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_SCANNER_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("SLURM_SCANNER_TOKENS", TOKEN)
    monkeypatch.delenv("SLURM_SCANNER_PLAN_CONFIG", raising=False)
    from server import app as module
    with pytest.raises(ValueError):
        importlib.reload(module)


def test_a_config_with_no_clusters_serves_a_plan_that_places_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_SCANNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SLURM_SCANNER_TOKENS", TOKEN)
    monkeypatch.setenv("SLURM_SCANNER_PLAN_CONFIG", str(write_config(tmp_path, clusters=[])))
    from server import app as module
    importlib.reload(module)
    body = TestClient(module.app).post("/plan", json={"units": 10})
    assert body.status_code == 200
    assert body.json()["unassigned_units"] == 10
    assert body.json()["clusters"] == []


def test_plan_needs_no_token(client):
    feed(client, "snellius", "snellius_shape", 7200)
    assert client.post("/plan", json={"units": 10}).status_code == 200


def test_plan_writes_nothing(client, tmp_path):
    feed(client, "snellius", "snellius_shape", 7200)
    before = sorted(p.stat().st_mtime_ns for p in (tmp_path / "data").rglob("*.csv"))
    first = client.post("/plan", json={"units": 100}).json()
    assert first == client.post("/plan", json={"units": 100}).json()
    assert before == sorted(p.stat().st_mtime_ns
                            for p in (tmp_path / "data").rglob("*.csv"))


def test_throughput_derived_from_units_per_gpu_per_hour_and_shape_gpus(tmp_path, monkeypatch):
    clusters = [
        {"cluster": "frontier", "shape": "mi250_8gpu_24h",
         "units_per_gpu_per_hour": 22.5, "shapes": {"mi250_8gpu_24h": 8, "mi250_4gpu_24h": 4},
         "max_walltime_hours": 72},
    ]
    monkeypatch.setenv("SLURM_SCANNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SLURM_SCANNER_TOKENS", TOKEN)
    monkeypatch.setenv("SLURM_SCANNER_PLAN_CONFIG", str(write_config(tmp_path, clusters=clusters)))
    from server import app as module
    importlib.reload(module)
    client = TestClient(module.app)
    feed(client, "frontier", "mi250_8gpu_24h", 0)
    body = client.post("/plan", json={"units": 180}).json()
    assert body["feasible"] is True
    job = body["clusters"][0]
    assert job["units_per_gpu_per_hour"] == 22.5
    assert job["gpus"] == 8
    assert job["units_per_hour"] == 180.0
    assert job["units"] == 180
    assert job["walltime_hours"] == 1.0


def test_config_without_top_level_shape_is_valid(tmp_path):
    clusters = [
        {"cluster": "snellius", "units_per_gpu_per_hour": 10,
         "shapes": {"h100_1gpu_8h": 1, "h100_4gpu_8h": 4}, "max_walltime_hours": 120},
    ]
    config = planner.load_config(write_config(tmp_path, clusters=clusters))
    assert len(config["clusters"]) == 1
    assert "shape" not in config["clusters"][0]


def test_plan_auto_selects_best_shape_from_multiple_options(tmp_path, monkeypatch):
    # Snellius has 1-GPU shape (fast start, 10 u/h) and 4-GPU shape (longer wait, 40 u/h)
    clusters = [
        {"cluster": "snellius", "units_per_gpu_per_hour": 10,
         "shapes": {"h100_1gpu_8h": 1, "h100_4gpu_8h": 4}, "max_walltime_hours": 120},
    ]
    monkeypatch.setenv("SLURM_SCANNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SLURM_SCANNER_TOKENS", TOKEN)
    monkeypatch.setenv("SLURM_SCANNER_PLAN_CONFIG", str(write_config(tmp_path, clusters=clusters)))
    from server import app as module
    importlib.reload(module)
    client = TestClient(module.app)
    feed(client, "snellius", "h100_1gpu_8h", 1800)     # 0.5 h wait
    feed(client, "snellius", "h100_4gpu_8h", 7200)     # 2.0 h wait

    # For 10 units: 1-GPU shape starts at 0.5h, finishes at 1.5h (earlier than 4-GPU shape start of 2.0h)
    small = client.post("/plan", json={"units": 10}).json()
    assert small["clusters"][0]["shape"] == "h100_1gpu_8h"
    assert small["clusters"][0]["gpus"] == 1

    # For 200 units: 4-GPU shape (wait 2h + 5h run = 7h) beats 1-GPU shape (wait 0.5h + 20h run = 20.5h)
    large = client.post("/plan", json={"units": 200}).json()
    assert large["clusters"][0]["shape"] == "h100_4gpu_8h"
    assert large["clusters"][0]["gpus"] == 4


def test_plan_respects_max_gpus_filter(tmp_path, monkeypatch):
    clusters = [
        {"cluster": "snellius", "units_per_gpu_per_hour": 10,
         "shapes": {"h100_1gpu_8h": 1, "h100_4gpu_8h": 4}, "max_walltime_hours": 120},
    ]
    monkeypatch.setenv("SLURM_SCANNER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SLURM_SCANNER_TOKENS", TOKEN)
    monkeypatch.setenv("SLURM_SCANNER_PLAN_CONFIG", str(write_config(tmp_path, clusters=clusters)))
    from server import app as module
    importlib.reload(module)
    client = TestClient(module.app)
    feed(client, "snellius", "h100_1gpu_8h", 1800)
    feed(client, "snellius", "h100_4gpu_8h", 7200)

    # With max_gpus=1, 4-GPU shape cannot be chosen even for 200 units
    body = client.post("/plan", json={"units": 200, "max_gpus": 1}).json()
    assert body["clusters"][0]["shape"] == "h100_1gpu_8h"
    assert body["clusters"][0]["gpus"] == 1


