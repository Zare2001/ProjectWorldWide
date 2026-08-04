"""The lease coordinator: HTTP transport, durability, and a CLI.

    python3 -m pww.darl.server --num-samples 1000000 --block-size 1000 \
        --port 8760 --state-dir runs/darl --token "$DARL_TOKEN"

Run it on a host every participating cluster can reach -- a login or edge node --
not inside a batch job, because it has to outlive every job that talks to it.
`scripts/darl_coordinator.sh` wraps this with the site's environment and a
nohup/pid file so it survives the SSH session.

Why plain HTTP and stdlib
-------------------------
The design calls for a Raft-backed store (etcd, or Redis with a consensus
front end). On an HPC facility you cannot assume you may run one: no container
orchestration on login nodes, no root, no service registry, and often no
outbound ports at all. What you *can* assume is CPython and a TCP port on a
login node. So this is one process, one lock, and a write-ahead log:

  * one lock            makes every RPC atomic in the sense the design needs --
                        the read and the write of the lease table cannot be
                        interleaved by another cluster, so two clusters can never
                        be told about the same block
  * write-ahead log     every mutation is appended to journal.jsonl before the
                        reply goes out, so a coordinator killed mid-epoch comes
                        back knowing exactly what it had handed out
  * periodic snapshot   bounds replay time, and is the artifact you pair with a
                        model checkpoint

What this does *not* give you is availability: while the coordinator is down,
clusters cannot acquire (they keep training on the span they hold, and their
leases are held open by the restore grace period). Replicating the state machine
is the one thing etcd would add, and the journal here is exactly what a Raft log
would replicate -- if a site ever offers a usable etcd, `Coordinator` is the only
class that has to change.

Load, in perspective: a lease covers one DiLoCo local phase, which for an LLM is
minutes to hours. Ten clusters produce a handful of requests per minute plus a
heartbeat every TTL/3. A single-threaded Python HTTP server is four orders of
magnitude away from being the bottleneck, and the whole point of coarse-grained
leasing is that this stays true.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import signal
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger, setup_logging
from .space import BlockSpace
from .table import HEARTBEAT_DIVISOR, MIN_TTL, LeaseTable

DEFAULT_PORT = 8760
# Mutations that are cheap to lose are batched; a commit is fsynced because
# losing one costs the epoch a full re-run of that span.
_FSYNC_OPS = {"commit"}
# Idempotency keys retained per coordinator. A few per cluster is plenty: a
# client only ever retries the acquire it is currently blocked on.
_GRANT_CACHE = 256


class Coordinator:
    """A lease table plus a lock, a write-ahead log, and periodic snapshots."""

    def __init__(
        self,
        table: LeaseTable,
        state_dir: str | Path | None = None,
        *,
        snapshot_interval: float = 60.0,
        verify_on_snapshot: bool = True,
    ):
        self.table = table
        self.lock = threading.RLock()
        self.state_dir = Path(state_dir) if state_dir else None
        self.snapshot_interval = float(snapshot_interval)
        self.verify_on_snapshot = verify_on_snapshot
        self._journal = None
        self._last_snapshot = 0.0
        self._grant_cache: "OrderedDict[tuple[str, str], dict[str, Any]]" = OrderedDict()
        self.requests = 0
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self._journal = open(self.state_dir / "journal.jsonl", "a", buffering=1)

    # --- persistence ------------------------------------------------------

    @property
    def snapshot_path(self) -> Path | None:
        return None if self.state_dir is None else self.state_dir / "snapshot.json"

    @property
    def journal_path(self) -> Path | None:
        return None if self.state_dir is None else self.state_dir / "journal.jsonl"

    def _log_op(self, op: str, payload: dict[str, Any], now: float) -> None:
        if self._journal is None:
            return
        self._journal.write(json.dumps({"op": op, "now": now, "payload": payload}) + "\n")
        if op in _FSYNC_OPS:
            self._journal.flush()
            os.fsync(self._journal.fileno())

    def save_snapshot(self, force: bool = False) -> Path | None:
        """Write the full state and truncate the journal, atomically enough.

        Order matters: snapshot first, then truncate. A crash between the two
        leaves ops that are already in the snapshot to be replayed on top of it,
        and every op is idempotent under replay (commits are watermarks, acquires
        are recorded with their granted spans), so a double replay is harmless.
        Truncating first and crashing would lose them outright.
        """
        if self.snapshot_path is None:
            return None
        with self.lock:
            now = time.time()
            if not force and now - self._last_snapshot < self.snapshot_interval:
                return None
            if self.verify_on_snapshot:
                self.table.verify()
            payload = self.table.snapshot()
            payload["saved_at"] = now
            tmp = self.snapshot_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.snapshot_path)
            if self._journal is not None:
                self._journal.close()
                self._journal = open(self.journal_path, "w", buffering=1)
            self._last_snapshot = now
            return self.snapshot_path

    @classmethod
    def load(
        cls,
        state_dir: str | Path,
        *,
        expect_blocks: int | None = None,
        expect_digest: str = "",
        grace_s: float = 0.0,
        **kwargs: Any,
    ) -> "Coordinator | None":
        """Rebuild from snapshot + journal replay, or None if there is no state."""
        state_dir = Path(state_dir)
        snapshot_file = state_dir / "snapshot.json"
        if not snapshot_file.exists():
            return None
        snapshot = json.loads(snapshot_file.read_text())
        if expect_blocks is not None and snapshot["num_blocks"] != expect_blocks:
            raise ValueError(
                f"{snapshot_file} holds a {snapshot['num_blocks']}-block epoch but this "
                f"coordinator was started for {expect_blocks}. Point --state-dir "
                f"somewhere else, or fix --num-samples/--block-size to match the run "
                f"you are resuming."
            )
        if expect_digest and snapshot.get("digest") and snapshot["digest"] != expect_digest:
            raise ValueError(
                f"{snapshot_file} was written for block-space digest "
                f"{snapshot['digest'][:12]}... but this one is {expect_digest[:12]}...; "
                f"the permutation changed, so committed positions no longer mean the "
                f"same samples"
            )
        table = LeaseTable.restore(snapshot, grace_s=grace_s)
        self = cls(table, state_dir, **kwargs)

        replayed = 0
        journal = state_dir / "journal.jsonl"
        if journal.exists():
            for line in journal.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # A torn last line is expected after a hard kill; everything
                    # before it is intact, so stop rather than fail.
                    get_logger().warning("journal: ignoring truncated final entry")
                    break
                self._replay(entry, grace_s=grace_s)
                replayed += 1
        get_logger().info(
            "restored coordinator from %s (+%d journal entries): epoch %d, %d/%d blocks "
            "committed, %d leases held",
            snapshot_file, replayed, table.epoch, table.committed, table.num_blocks,
            len(table.leases),
        )
        return self

    def _replay(self, entry: dict[str, Any], *, grace_s: float) -> None:
        """Re-apply one journalled mutation.

        Acquires are replayed from their *recorded outcome* rather than by
        re-running the allocator: the allocator's choice depends on wall-clock
        ordering that replay cannot reproduce, and what has to survive a restart
        is which blocks a cluster was told it owns, not how they were chosen.
        """
        op, payload, now = entry["op"], entry["payload"], entry["now"]
        table = self.table
        try:
            if op == "register":
                table.register(payload["cluster"], digest=payload.get("digest", ""),
                               ranks=payload.get("ranks", 0), now=now)
            elif op == "acquire":
                table.register(payload["cluster"], now=now)
                for lease in payload["leases"]:
                    table.replay_lease(lease, deadline_floor=time.time() + grace_s)
            elif op == "commit":
                table.commit(payload["cluster"], payload["lease"], payload["through"], now=now)
            elif op == "release":
                table.release(payload["cluster"], payload.get("lease"), now=now)
            elif op == "heartbeat":
                table.heartbeat(payload["cluster"], payload.get("progress"),
                                ttl=payload.get("ttl", 0.0), now=now)
            elif op == "reap":
                table.reap(now)
            elif op == "advance_epoch":
                table.advance_epoch()
        except (KeyError, PermissionError, ValueError) as exc:
            # Replay of an op whose target no longer exists is not an error: the
            # snapshot may already contain its effect.
            get_logger().debug("journal replay skipped %s: %s", op, exc)

    # --- RPCs -------------------------------------------------------------

    def register(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            record = self.table.register(
                payload["cluster"],
                digest=payload.get("digest", ""),
                ranks=int(payload.get("ranks", 0)),
                now=now,
            )
            self._log_op("register", {"cluster": record.cluster_id,
                                      "digest": payload.get("digest", ""),
                                      "ranks": record.ranks}, now)
            return {
                "cluster": record.cluster_id,
                "epoch": self.table.epoch,
                "num_blocks": self.table.num_blocks,
                "digest": self.table.digest,
                "heartbeat_divisor": HEARTBEAT_DIVISOR,
                "committed": self.table.committed,
                "known_clusters": sorted(self.table.clusters),
            }

    def acquire(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Grant a work vector, deduplicating retransmitted requests.

        An acquire is the one RPC that is not naturally idempotent, and a client
        that retries after a lost reply must not be handed a second span -- the
        first one would sit LEASED and unworked until its TTL expired, which on a
        long TTL is a real hole in the epoch's throughput. The `request_id` the
        client generates makes the retry a lookup instead. The cache is
        deliberately in-memory: losing it to a coordinator restart costs at most
        one leaked lease, which expires on its own.
        """
        request_id = payload.get("request_id")
        with self.lock:
            if request_id:
                cached = self._grant_cache.get((payload["cluster"], request_id))
                if cached is not None:
                    get_logger().info("darl: replaying cached grant for %s/%s",
                                      payload["cluster"], request_id[:8])
                    return cached
            now = time.time()
            grant = self.table.acquire(
                payload["cluster"],
                int(payload.get("blocks", 1)),
                ttl=float(payload.get("ttl", 0.0)),
                max_spans=int(payload.get("max_spans", 4)),
                macro_step_s=float(payload.get("macro_step_s", 0.0)),
                rtt_s=float(payload.get("rtt_s", 0.0)),
                now=now,
            )
            if grant.leases:
                self._log_op("acquire", {"cluster": payload["cluster"],
                                         "leases": [l.to_dict() for l in grant.leases]}, now)
            reply = grant.to_dict()
            if request_id and grant.leases:
                self._grant_cache[(payload["cluster"], request_id)] = reply
                while len(self._grant_cache) > _GRANT_CACHE:
                    self._grant_cache.popitem(last=False)
            return reply

    def heartbeat(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            reply = self.table.heartbeat(
                payload["cluster"],
                {str(k): int(v) for k, v in (payload.get("progress") or {}).items()},
                macro_step_s=float(payload.get("macro_step_s", 0.0)),
                rtt_s=float(payload.get("rtt_s", 0.0)),
                ttl=float(payload.get("ttl", 0.0)),
                now=now,
            )
            # Heartbeats are not journalled: they only move deadlines and progress
            # watermarks, both of which are re-established by the next heartbeat
            # after a restart. Logging them would dominate the journal.
            return reply

    def commit(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            result = self.table.commit(payload["cluster"], payload["lease"],
                                       int(payload["through"]), now=now)
            self._log_op("commit", {"cluster": payload["cluster"], "lease": payload["lease"],
                                    "through": int(payload["through"])}, now)
            return result

    def release(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            blocks = self.table.release(payload["cluster"], payload.get("lease"), now=now)
            self._log_op("release", {"cluster": payload["cluster"],
                                     "lease": payload.get("lease")}, now)
            return {"released": blocks, "epoch": self.table.epoch}

    def status(self) -> dict[str, Any]:
        with self.lock:
            self.table.reap()
            status = self.table.status()
            status["requests"] = self.requests
            return status

    def tick(self) -> None:
        """Background maintenance: reap expired leases, snapshot periodically."""
        with self.lock:
            reclaimed = self.table.reap()
            if reclaimed:
                self._log_op("reap", {"blocks": reclaimed}, time.time())
        self.save_snapshot()


# --- HTTP transport --------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    """One method per route. Errors come back as JSON with a meaningful code."""

    server_version = "darl/1.0"
    protocol_version = "HTTP/1.1"

    # Set by serve()
    coordinator: Coordinator = None            # type: ignore[assignment]
    token: str = ""

    def log_message(self, fmt: str, *args: Any) -> None:
        # BaseHTTPRequestHandler logs every request to stderr, which at one
        # heartbeat per cluster per TTL/3 is noise in a long-lived process.
        get_logger().debug("darl: " + fmt, *args)

    # --- plumbing ---------------------------------------------------------

    def _authorised(self) -> bool:
        if not self.token:
            return True
        return hmac.compare_digest(self.headers.get("X-DARL-Token", ""), self.token)

    def _send(self, code: int, payload: dict[str, Any] | str) -> None:
        if isinstance(payload, str):
            body = payload.encode()
            ctype = "text/plain; charset=utf-8"
        else:
            body = json.dumps(payload).encode()
            ctype = "application/json"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length))

    # --- routes -----------------------------------------------------------

    def do_GET(self) -> None:               # noqa: N802 (stdlib naming)
        if not self._authorised():
            self._send(401, {"error": "bad or missing X-DARL-Token"})
            return
        route = self.path.split("?")[0].rstrip("/") or "/"
        try:
            if route == "/":
                self._send(200, format_status(self.coordinator.status()))
            elif route == "/status":
                self._send(200, self.coordinator.status())
            elif route == "/snapshot":
                with self.coordinator.lock:
                    self._send(200, self.coordinator.table.snapshot())
            elif route == "/health":
                self._send(200, {"ok": True, "epoch": self.coordinator.table.epoch})
            else:
                self._send(404, {"error": f"no route {route}"})
        except Exception as exc:                                  # noqa: BLE001
            get_logger().exception("darl GET %s failed", route)
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self) -> None:              # noqa: N802
        if not self._authorised():
            self._send(401, {"error": "bad or missing X-DARL-Token"})
            return
        route = self.path.split("?")[0].rstrip("/")
        coordinator = self.coordinator
        coordinator.requests += 1
        handlers = {
            "/register": coordinator.register,
            "/acquire": coordinator.acquire,
            "/heartbeat": coordinator.heartbeat,
            "/commit": coordinator.commit,
            "/release": coordinator.release,
        }
        if route == "/snapshot":
            path = coordinator.save_snapshot(force=True)
            self._send(200, {"snapshot": str(path) if path else None})
            return
        if route == "/advance":
            with coordinator.lock:
                advanced = coordinator.table.advance_epoch()
                coordinator._log_op("advance_epoch", {}, time.time())
            self._send(200, {"advanced": advanced, "epoch": coordinator.table.epoch})
            return
        handler = handlers.get(route)
        if handler is None:
            self._send(404, {"error": f"no route {route}"})
            return
        try:
            self._send(200, handler(self._body()))
        except KeyError as exc:
            # A lease that no longer exists. 409 rather than 404 because the
            # client's own state is what is stale, and it has to act on that.
            self._send(409, {"error": str(exc), "code": "lease_gone"})
        except PermissionError as exc:
            self._send(403, {"error": str(exc), "code": "not_owner"})
        except ValueError as exc:
            self._send(400, {"error": str(exc), "code": "bad_request"})
        except Exception as exc:                                  # noqa: BLE001
            get_logger().exception("darl POST %s failed", route)
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def format_status(status: dict[str, Any]) -> str:
    """Human-readable status, for `curl http://host:port/`."""
    lines = [
        f"epoch {status['epoch']} of {status['max_epochs']}  "
        f"({status['epochs_completed']} completed)",
        f"blocks     {status['committed']:,} committed | {status['leased']:,} leased | "
        f"{status['unassigned']:,} free | {status['quarantined']:,} quarantined "
        f"of {status['num_blocks']:,}",
        f"progress   {100 * status['progress']:.2f}%  at {status['blocks_per_s']:.3f} blocks/s"
        + (f"  eta {status['eta_s'] / 60:.1f} min" if status["eta_s"] else ""),
        f"pool       {status['fragments']} free fragments | {status['active_leases']} "
        f"active leases | {status['requests']} requests served",
        "",
        f"{'cluster':<24}{'ranks':>6}{'committed':>11}{'leased':>8}{'lost':>7}"
        f"{'blk/s':>9}{'step s':>9}{'rtt ms':>8}{'seen s':>8}",
    ]
    now = time.time()
    held: dict[str, int] = {}
    for lease in status["leases"]:
        held[lease["cluster"]] = held.get(lease["cluster"], 0) + lease["end"] - lease["committed_end"]
    for cid, c in sorted(status["clusters"].items()):
        lines.append(
            f"{cid:<24}{c['ranks']:>6}{c['blocks_committed']:>11,}{held.get(cid, 0):>8,}"
            f"{c['blocks_lost']:>7,}{c['rate']:>9.3f}{c['macro_step_s']:>9.1f}"
            f"{1000 * c['rtt_s']:>8.1f}{now - c['last_seen']:>8.0f}"
        )
    if status["quarantined"]:
        lines += ["", f"WARNING: {status['quarantined']} blocks quarantined -- this epoch "
                      f"will NOT cover the dataset exactly once"]
    return "\n".join(lines) + "\n"


def make_server(
    coordinator: Coordinator,
    *,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    token: str = "",
) -> ThreadingHTTPServer:
    """Bind a coordinator to a socket without blocking or touching signals.

    Split out from `serve` so tests (and anything else that wants to embed a
    coordinator) can run one on an ephemeral port from a non-main thread. Note the
    handler holds the coordinator as a class attribute, so there is one coordinator
    per process -- which is all a login node ever needs.
    """
    _Handler.coordinator = coordinator
    _Handler.token = token
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True
    return httpd


def serve(
    coordinator: Coordinator,
    *,
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    token: str = "",
    tick_s: float = 0.0,
) -> None:
    """Run the coordinator until SIGINT/SIGTERM. Blocks."""
    log = get_logger()
    httpd = make_server(coordinator, host=host, port=port, token=token)

    tick_s = tick_s or max(5.0, MIN_TTL / HEARTBEAT_DIVISOR)
    stop = threading.Event()

    def maintenance() -> None:
        while not stop.wait(tick_s):
            try:
                coordinator.tick()
            except Exception:                                     # noqa: BLE001
                log.exception("darl maintenance tick failed")

    thread = threading.Thread(target=maintenance, name="darl-reaper", daemon=True)
    thread.start()

    def shutdown(signum, _frame) -> None:
        log.info("darl: signal %d, snapshotting and shutting down", signum)
        stop.set()
        coordinator.save_snapshot(force=True)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, shutdown)

    log.info("darl coordinator on http://%s:%d | %d blocks | epoch %d/%d | token %s",
             host, port, coordinator.table.num_blocks, coordinator.table.epoch,
             coordinator.table.max_epochs, "yes" if token else "NO -- anyone can lease")
    if coordinator.state_dir is not None:
        log.info("darl state dir: %s", coordinator.state_dir)
    try:
        httpd.serve_forever()
    finally:
        stop.set()
        coordinator.save_snapshot(force=True)
        httpd.server_close()
        log.info("darl coordinator stopped: %s",
                 format_status(coordinator.status()).splitlines()[1])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DARL lease coordinator")
    g = p.add_argument_group("dataset")
    g.add_argument("--num-samples", type=int, required=True,
                   help="N: tokenised sequences in the corpus")
    g.add_argument("--block-size", type=int, default=10_000,
                   help="K: samples per block. The design's medium granularity -- a "
                        "few GB of text -- is the right starting point; see the "
                        "granularity trade-off in table.py")
    g.add_argument("--seed", type=int, default=42,
                   help="Derives the global block permutation. Every client must use "
                        "the same value; mismatches are rejected at registration")
    g.add_argument("--no-shuffle", action="store_true",
                   help="Identity permutation: positions are block ids (debugging)")
    g.add_argument("--epochs", type=int, default=1,
                   help="Epochs to serve. The default of 1 is the design's "
                        "single-epoch guarantee: once every block is committed, "
                        "acquire returns epoch_complete forever")

    g = p.add_argument_group("leasing")
    g.add_argument("--max-blocks", type=int, default=0,
                   help="Ceiling on one lease (0 = no ceiling beyond fair share)")
    g.add_argument("--min-blocks", type=int, default=1)
    g.add_argument("--max-attempts", type=int, default=3,
                   help="Reclaims of one block before it is QUARANTINED. 0 retries "
                        "forever, which never duplicates but can hang an epoch on a "
                        "corrupt shard")
    g.add_argument("--first-grant-fraction", type=float, default=0.5,
                   help="Fraction of the request a cluster gets before it has any "
                        "measured throughput")
    g.add_argument("--no-stealing", action="store_true",
                   help="Disable work stealing. Straggler tails then have to expire "
                        "rather than being taken from")
    g.add_argument("--min-ttl", type=float, default=None,
                   help="Floor on a granted lease TTL, in seconds (default 30). Sized "
                        "for a WAN; lower it only for a single-site pool with short "
                        "phases, since a TTL below one macro-step reclaims spans from "
                        "clusters that are working perfectly well")

    g = p.add_argument_group("serving")
    g.add_argument("--host", type=str, default="0.0.0.0")
    g.add_argument("--port", type=int, default=DEFAULT_PORT)
    g.add_argument("--token", type=str, default=os.environ.get("DARL_TOKEN", ""),
                   help="Shared secret, sent as X-DARL-Token. Not confidentiality -- "
                        "it stops an unrelated job from corrupting a month-long run. "
                        "Tunnel over SSH if the link is untrusted")
    g.add_argument("--state-dir", type=str, default=None,
                   help="Snapshot + journal directory. Without it the coordinator is "
                        "in-memory only and a restart loses the epoch")
    g.add_argument("--snapshot-interval", type=float, default=60.0)
    g.add_argument("--restore-grace", type=float, default=300.0,
                   help="Seconds added to restored lease deadlines, so a coordinator "
                        "restart does not reclaim spans from clusters that are alive "
                        "and mid-phase")
    g.add_argument("--fresh", action="store_true",
                   help="Ignore any existing state in --state-dir and start a new epoch")
    g.add_argument("--no-verify", action="store_true",
                   help="Skip the invariant check on each snapshot")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(rank=0)
    log = get_logger()

    space = BlockSpace(
        num_samples=args.num_samples,
        block_size=args.block_size,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )
    log.info("block space: %s | digest %s", space.describe(), space.digest()[:16])

    coordinator = None
    if args.state_dir and not args.fresh:
        coordinator = Coordinator.load(
            args.state_dir,
            expect_blocks=space.num_blocks,
            expect_digest=space.digest(),
            grace_s=args.restore_grace,
            snapshot_interval=args.snapshot_interval,
            verify_on_snapshot=not args.no_verify,
        )
    if coordinator is None:
        table = LeaseTable(
            space.num_blocks,
            digest=space.digest(),
            max_epochs=args.epochs,
            max_attempts=args.max_attempts,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            first_grant_fraction=args.first_grant_fraction,
            allow_stealing=not args.no_stealing,
            **({"min_ttl": args.min_ttl} if args.min_ttl else {}),
        )
        coordinator = Coordinator(
            table,
            args.state_dir,
            snapshot_interval=args.snapshot_interval,
            verify_on_snapshot=not args.no_verify,
        )
    if not args.token:
        log.warning("no --token: any process that can reach this port can lease, "
                    "commit and release blocks in this run")

    serve(coordinator, host=args.host, port=args.port, token=args.token)


if __name__ == "__main__":
    main()
