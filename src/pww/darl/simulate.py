"""m heterogeneous clusters against a real coordinator, and a coverage audit.

    python3 -m pww.darl.simulate                       # 4 clusters, one dies
    python3 -m pww.darl.simulate --clusters 8 --kill 2 --late 1

Runs on a login node in seconds and needs no allocation, no GPUs and no torch.
Each simulated cluster is a **separate process** running the real `LeaseClient`
and `LeaseSession` over real HTTP against a real `pww.darl.server`, with a queue
delay before it starts and its own throughput. One of them is killed mid-lease
without releasing anything, which is what a walltime kill or a node crash looks
like from the coordinator's side.

Then the audit, which is the point of the whole exercise: every process reports
the sample ranges it committed, and the driver checks

    completeness      the union of committed sample indices is exactly [0, N)
    disjointness      no index appears in two clusters' reports
    zero duplication  the total count equals N

This is a test of the protocol under genuine concurrency -- separate address
spaces, real sockets, a real kill -- rather than of the state machine in
isolation, which is what `tests/test_darl.py` covers. Both matter: the unit tests
find off-by-ones, this finds races.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger, setup_logging
from .client import DarlError, LeaseClient, LeaseSession
from .server import DEFAULT_PORT, format_status
from .space import BlockSpace


def _worker(config: dict[str, Any], results: "mp.Queue[dict[str, Any]]") -> None:
    """One simulated cluster. Runs in its own process."""
    setup_logging(rank=0 if config["verbose"] else 1)
    log = get_logger()
    space = BlockSpace(**config["space"])
    time.sleep(config["arrival_s"])          # batch queue delay

    committed: list[tuple[int, int]] = []
    consumed_blocks = 0
    try:
        client = LeaseClient(config["url"], config["cluster"], token=config["token"],
                             timeout=10.0, retries=4)
        session = LeaseSession(client, space, blocks_per_phase=config["blocks_per_phase"],
                               ranks=config["ranks"], min_ttl=config["min_ttl"])
    except DarlError as exc:
        results.put({"cluster": config["cluster"], "error": str(exc)})
        return

    deadline = time.monotonic() + config["walltime_s"]
    phases = 0
    try:
        while time.monotonic() < deadline:
            acquisition = session.acquire(wait=True, timeout=config["min_ttl"] * 4)
            if acquisition.epoch_complete:
                log.info("%s: epoch complete", config["cluster"])
                break
            if not acquisition.granted:
                continue

            for span in acquisition.spans:
                # "Training": time proportional to the work, so a slow cluster is
                # slow in the way that matters -- it holds its lease longer.
                blocks = span.end - span.start
                phase_s = blocks / config["blocks_per_s"]
                if config["kill_after_phases"] and phases >= config["kill_after_phases"]:
                    # Die holding an uncommitted lease and without releasing it:
                    # no SIGTERM handler, no final commit, nothing. The coordinator
                    # must notice via the missing heartbeat and reclaim the span.
                    log.warning("%s: simulating a crash while holding %s (%d blocks)",
                                config["cluster"], span.lease_id, blocks)
                    results.put({"cluster": config["cluster"], "committed": committed,
                                 "consumed_blocks": consumed_blocks, "phases": phases,
                                 "crashed": True, "stats": session.stats()})
                    # Flush the queue's feeder thread before the hard exit, or the
                    # report is lost. Everything else -- the heartbeat thread, the
                    # lease, any release -- dies with the process, which is the
                    # point.
                    results.close()
                    results.join_thread()
                    os._exit(0)
                time.sleep(phase_s)
                session.note_consumed(span.lease_id, span.end)
                consumed_blocks += blocks
                phases += 1
                session.note_phase_time(phase_s)

                # Checkpoint-gated commit: pretend the checkpoint landed, then
                # commit exactly what it covers.
                newly = session.commit(span.lease_id, span.end)
                if newly:
                    committed.append((span.start, span.start + newly))
            if session.epoch_complete:
                break
    finally:
        try:
            session.close(release=True)
        except Exception:                                        # noqa: BLE001
            pass
    results.put({"cluster": config["cluster"], "committed": committed,
                 "consumed_blocks": consumed_blocks, "phases": phases,
                 "crashed": False, "stats": session.stats()})


def _wait_for_health(url: str, token: str, timeout: float = 20.0) -> bool:
    headers = {"X-DARL-Token": token} if token else {}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(f"{url}/health", headers=headers)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=2.0) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(0.2)
    return False


def audit(space: BlockSpace, reports: list[dict[str, Any]], epoch: int = 0) -> dict[str, Any]:
    """Check the three invariants against what the clusters say they processed.

    Deliberately done from the *clients'* reports rather than from the
    coordinator's table: the coordinator's own accounting agreeing with itself
    proves much less than m independent processes' accounts of what they consumed
    adding up to the dataset exactly once.
    """
    owner: dict[int, str] = {}
    duplicates: list[tuple[int, str, str]] = []
    for report in reports:
        cluster = report.get("cluster", "?")
        for start, end in report.get("committed", []):
            for position in range(start, end):
                previous = owner.get(position)
                if previous is not None:
                    duplicates.append((position, previous, cluster))
                else:
                    owner[position] = cluster

    covered = set(owner)
    missing = sorted(set(range(space.num_blocks)) - covered)
    samples = sum(len(space.block_samples(p, epoch)) for p in covered)
    return {
        "blocks_total": space.num_blocks,
        "blocks_covered": len(covered),
        "duplicates": duplicates[:32],
        "duplicate_count": len(duplicates),
        "missing": missing[:32],
        "missing_count": len(missing),
        "samples_covered": samples,
        "samples_total": space.num_samples,
        "complete": not missing,
        "disjoint": not duplicates,
        "exact": samples == space.num_samples and not duplicates and not missing,
        "per_cluster": {r.get("cluster", "?"): sum(e - s for s, e in r.get("committed", []))
                        for r in reports},
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simulate multi-HPC leasing end to end")
    p.add_argument("--url", default=None,
                   help="Existing coordinator. Omitted: one is started and stopped here")
    p.add_argument("--port", type=int, default=DEFAULT_PORT + 1)
    p.add_argument("--token", default="")
    p.add_argument("--clusters", type=int, default=4)
    p.add_argument("--num-samples", type=int, default=100_000)
    p.add_argument("--block-size", type=int, default=100)
    p.add_argument("--blocks-per-phase", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-ttl", type=float, default=3.0,
                   help="Lease TTL floor. Short so that a crashed cluster's span is "
                        "reclaimed within the run rather than minutes later")
    p.add_argument("--walltime", type=float, default=120.0,
                   help="Per-cluster time budget, i.e. its Slurm walltime")
    p.add_argument("--kill", type=int, default=1,
                   help="Clusters that crash mid-lease without releasing (0 disables)")
    p.add_argument("--kill-after-phases", type=int, default=2)
    p.add_argument("--late", type=int, default=1,
                   help="Clusters that join late, as if still queued")
    p.add_argument("--late-delay", type=float, default=6.0)
    p.add_argument("--spread", type=float, default=8.0,
                   help="Throughput ratio between the fastest and slowest cluster")
    p.add_argument("--base-rate", type=float, default=4.0,
                   help="Blocks per second for the slowest cluster")
    p.add_argument("--state-dir", default=None)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(rank=0)
    log = get_logger()

    space = BlockSpace(num_samples=args.num_samples, block_size=args.block_size,
                       seed=args.seed)
    log.info("simulating %d clusters over %s", args.clusters, space.describe())

    url = args.url
    server: subprocess.Popen | None = None
    if url is None:
        url = f"http://127.0.0.1:{args.port}"
        command = [
            sys.executable, "-m", "pww.darl.server",
            "--num-samples", str(args.num_samples),
            "--block-size", str(args.block_size),
            "--seed", str(args.seed),
            "--port", str(args.port),
            "--host", "127.0.0.1",
            "--min-ttl", str(args.min_ttl),
            "--fresh",
        ]
        if args.token:
            command += ["--token", args.token]
        if args.state_dir:
            command += ["--state-dir", args.state_dir]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get(
            "PYTHONPATH", "")
        server = subprocess.Popen(command, env=env)
        if not _wait_for_health(url, args.token):
            server.terminate()
            log.error("coordinator did not come up on %s", url)
            return 1

    rng = random.Random(args.seed)
    # Heterogeneous throughput, spread over --spread x, plus queue delays. This is
    # the situation static partitioning cannot handle: the ratio is not known
    # before the jobs start, and it changes when a cluster is requeued.
    rates = [args.base_rate * (args.spread ** (i / max(1, args.clusters - 1)))
             for i in range(args.clusters)]
    rng.shuffle(rates)

    results: "mp.Queue[dict[str, Any]]" = mp.Queue()
    processes = []
    for index in range(args.clusters):
        config = {
            "url": url,
            "token": args.token,
            "cluster": f"sim-{index}",
            "space": {"num_samples": args.num_samples, "block_size": args.block_size,
                      "seed": args.seed},
            "blocks_per_phase": args.blocks_per_phase,
            "blocks_per_s": rates[index],
            "ranks": 4,
            "arrival_s": args.late_delay * rng.random() if index < args.late else 0.0,
            "walltime_s": args.walltime,
            "kill_after_phases": args.kill_after_phases if index < args.kill else 0,
            "min_ttl": args.min_ttl,
            "verbose": args.verbose,
        }
        process = mp.Process(target=_worker, args=(config, results), name=config["cluster"])
        process.start()
        processes.append(process)

    reports: list[dict[str, Any]] = []
    t0 = time.monotonic()
    # Crashed workers still report before exiting, so one report per process is
    # expected either way. The timeout is the backstop for a genuinely stuck one.
    while len(reports) < len(processes) and time.monotonic() - t0 < args.walltime + 60:
        try:
            reports.append(results.get(timeout=5.0))
        except Exception:                                        # noqa: BLE001
            alive = [p.name for p in processes if p.is_alive()]
            log.info("waiting on %d cluster(s): %s", len(alive), ", ".join(alive) or "-")
    for process in processes:
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()

    client = LeaseClient(url, "audit", token=args.token, retries=2)
    try:
        status = client.status()
    except DarlError as exc:
        log.error("could not read final status: %s", exc)
        status = {}

    report = audit(space, reports)
    print()
    if status:
        print(format_status(status))
    print(f"{'cluster':<12}{'blocks':>9}{'phases':>8}{'crashed':>9}{'lost':>7}{'rpcs':>7}")
    for r in sorted(reports, key=lambda r: r.get("cluster", "")):
        stats = r.get("stats", {})
        print(f"{r.get('cluster', '?'):<12}"
              f"{sum(e - s for s, e in r.get('committed', [])):>9,}"
              f"{r.get('phases', 0):>8}{str(r.get('crashed', False)):>9}"
              f"{stats.get('blocks_lost', 0):>7}{stats.get('rpcs', 0):>7}")

    print()
    print(f"blocks covered   {report['blocks_covered']:,} of {report['blocks_total']:,}")
    print(f"samples covered  {report['samples_covered']:,} of {report['samples_total']:,}")
    print(f"duplicates       {report['duplicate_count']}")
    print(f"missing blocks   {report['missing_count']}"
          + (f"  e.g. {report['missing'][:8]}" if report["missing"] else ""))
    verdict = "PASS" if report["exact"] else "FAIL"
    print(f"\n{verdict}: coverage is "
          f"{'exactly once' if report['exact'] else 'NOT exactly once'}"
          f" -- completeness {report['complete']}, disjointness {report['disjoint']}")

    if server is not None:
        server.terminate()
        server.wait(timeout=10)
    return 0 if report["exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
