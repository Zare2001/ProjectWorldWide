"""CPU-only DARL tests -- runnable on a login node, no allocation, no GPUs.

    source env.sh && pww_run python3 tests/test_darl.py
    python3 tests/test_darl.py          # torch is only needed for 4 of the checks

Three layers, in increasing cost:

  state machine   `LeaseTable` with an injected clock, so expiry, stealing and
                  quarantine are exercised deterministically in microseconds
                  rather than by waiting out real timeouts
  transport       a real coordinator on an ephemeral port with concurrent client
                  threads, which is what actually tests that two clusters cannot
                  be granted the same block
  trainer         the sharding maths in `torch_data`, with the broadcast stubbed

`python3 -m pww.darl.simulate` goes one further -- separate processes, real
crashes -- and is the end-to-end audit; this file is what you run after every
edit.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pww.darl.client import LeaseClient, LeaseSession  # noqa: E402
from pww.darl.server import Coordinator, make_server  # noqa: E402
from pww.darl.space import BlockSpace, blocks_for_phase  # noqa: E402
from pww.darl.table import (  # noqa: E402
    TTL_ALPHA,
    TTL_BETA,
    DoubleFreeError,
    IntervalSet,
    LeaseTable,
    compute_ttl,
)

PASSED, FAILED = [], []


def check(name: str):
    def decorator(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        return fn

    return decorator


class Clock:
    """Injectable time, so a 15-minute TTL expires in a microsecond."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


def table(blocks: int = 100, **kwargs) -> tuple[LeaseTable, Clock]:
    clock = Clock()
    kwargs.setdefault("min_ttl", 10.0)
    # Grants are honoured in full here so that each check exercises one mechanism;
    # the first-grant halving has its own check below.
    kwargs.setdefault("first_grant_fraction", 1.0)
    return LeaseTable(blocks, clock=clock, **kwargs), clock


def drain(lease_table: LeaseTable, cluster: str, blocks: int = 8,
          limit: int | None = None) -> int:
    """Acquire and commit until this epoch is exhausted, or `limit` blocks are done.

    Stops at an epoch boundary and hands back what it was granted there, so a
    multi-epoch table can be driven one epoch at a time.
    """
    epoch = lease_table.epoch
    total = 0
    while limit is None or total < limit:
        grant = lease_table.acquire(cluster, blocks, ttl=100.0)
        if grant.status != "granted":
            break
        if grant.epoch != epoch:
            for lease in list(grant.leases):
                lease_table.release(cluster, lease.lease_id)
            break
        for lease in list(grant.leases):
            lease_table.commit(cluster, lease.lease_id, lease.end)
            total += lease.blocks
    return total


# --- the index space -------------------------------------------------------


@check("block space geometry, including a short final block")
def _():
    space = BlockSpace(num_samples=1000, block_size=100)
    assert space.num_blocks == 10, space.num_blocks
    ragged = BlockSpace(num_samples=1005, block_size=100)
    assert ragged.num_blocks == 11, ragged.num_blocks
    counts = [len(ragged.block_samples(p)) for p in range(11)]
    assert sum(counts) == 1005, counts
    assert sorted(counts)[0] == 5, "exactly one block should be short"


@check("global permutation is a bijection and covers every sample once")
def _():
    space = BlockSpace(num_samples=997, block_size=10, seed=3)
    positions = [space.physical_block(p) for p in range(space.num_blocks)]
    assert sorted(positions) == list(range(space.num_blocks)), "not a permutation"
    seen = [i for p in range(space.num_blocks) for i in space.block_samples(p)]
    assert sorted(seen) == list(range(997)), "sample coverage is not exactly once"


@check("digest pins the permutation and changes with the seed")
def _():
    a = BlockSpace(num_samples=10_000, block_size=100, seed=1)
    b = BlockSpace(num_samples=10_000, block_size=100, seed=1)
    c = BlockSpace(num_samples=10_000, block_size=100, seed=2)
    assert a.digest() == b.digest(), "same config must give the same digest"
    assert a.digest() != c.digest(), "a different seed must be visible in the digest"
    assert a.digest(epoch=1) != a.digest(epoch=0), "epochs must reshuffle"


@check("lease granularity derives from H, batch and ranks")
def _():
    space = BlockSpace(num_samples=10_000_000, block_size=1000)
    # 100 steps x 8 batch x 4 ranks = 3200 samples = 3.2 blocks -> 4
    assert blocks_for_phase(space, inner_steps=100, batch_size=8, ranks=4) == 4
    # Exactly divisible stays exact.
    assert blocks_for_phase(space, inner_steps=100, batch_size=10, ranks=1) == 1
    assert blocks_for_phase(space, inner_steps=500, batch_size=16, ranks=8,
                            grad_accum=2) == 128


@check("TTL follows alpha*step + beta*rtt and clamps")
def _():
    assert compute_ttl(100.0, 0.05, floor=0) == TTL_ALPHA * 100 + TTL_BETA * 0.05
    assert compute_ttl(0.0, 0.0, floor=30.0) == 30.0, "floor applies"
    assert compute_ttl(1e9, 0.0) <= 6 * 3600, "ceiling applies"


# --- the free pool ---------------------------------------------------------


@check("interval set merges, splits and accounts exactly")
def _():
    pool = IntervalSet([(0, 10)])
    assert pool.total == 10
    taken = pool.take_near(0, 4)
    assert taken == [(0, 4)], taken
    assert pool.total == 6
    pool.add(0, 4)                        # give it back; must merge into one run
    assert pool.total == 10 and len(pool) == 1, pool.intervals()
    pool.take_near(0, 10)
    assert pool.total == 0 and not pool


@check("interval set refuses a double free")
def _():
    pool = IntervalSet([(0, 10)])
    pool.take_near(0, 5)
    pool.add(0, 5)
    try:
        pool.add(2, 4)
    except DoubleFreeError:
        return
    raise AssertionError("overlapping add must raise: it means two leases shared a block")


@check("allocation prefers the cursor, wraps, and bounds the span count")
def _():
    pool = IntervalSet([(0, 10), (20, 30), (40, 50)])
    assert pool.take_near(20, 5) == [(20, 25)], "should start at the cursor"
    # 12 blocks cannot come from one 5-block fragment, so it spills into the next.
    pool = IntervalSet([(0, 5), (10, 15), (20, 25)])
    spans = pool.take_near(0, 12, max_spans=2)
    assert len(spans) == 2 and sum(e - s for s, e in spans) == 10, spans
    assert pool.total == 5


# --- disjointness and completeness ----------------------------------------


@check("two clusters never receive the same block")
def _():
    # Stealing off for this one: it deliberately *transfers* unstarted blocks
    # between clusters, so ownership moves and the "handed out once" model below
    # would not hold. Stealing has its own checks further down.
    lease_table, _clock = table(60, allow_stealing=False)
    seen: set[int] = set()
    handed = 0
    for _ in range(10):
        for cluster in ("lumi", "snellius"):
            grant = lease_table.acquire(cluster, 5, ttl=100.0)
            for lease in grant.leases:
                blocks = set(range(lease.start, lease.end))
                assert not (blocks & seen), f"{cluster} got blocks already handed out"
                seen |= blocks
                handed += len(blocks)
    assert handed == len(seen) == 60, (handed, len(seen))
    lease_table.verify()


@check("an epoch drains to exactly M committed blocks")
def _():
    lease_table, _clock = table(97)
    total = drain(lease_table, "a", 7) + drain(lease_table, "b", 13)
    assert total == 97, total
    assert lease_table.committed == 97
    assert lease_table.epoch_complete
    counts = lease_table.verify()
    assert counts["committed"] == 97 and counts["unassigned"] == 0, counts


@check("a completed epoch keeps saying so, and does not roll into the next")
def _():
    lease_table, _clock = table(20)
    drain(lease_table, "a", 20)
    for _ in range(3):
        grant = lease_table.acquire("a", 10)
        assert grant.status == "epoch_complete", grant.status
        assert not grant.leases


@check("multi-epoch mode advances and reshuffles")
def _():
    lease_table, _clock = table(20, max_epochs=2)
    drain(lease_table, "a", 20)
    grant = lease_table.acquire("a", 5)
    assert grant.status == "granted" and grant.epoch == 1, (grant.status, grant.epoch)
    assert lease_table.committed == 0, "the new epoch starts from nothing committed"
    lease_table.verify()


@check("commits are monotonic and idempotent")
def _():
    lease_table, _clock = table(30)
    lease = lease_table.acquire("a", 10, ttl=100.0).leases[0]
    lease_table.commit("a", lease.lease_id, 5)
    assert lease_table.committed == 5
    lease_table.commit("a", lease.lease_id, 5)                 # replayed
    assert lease_table.committed == 5, "a retried commit must not double-count"
    lease_table.commit("a", lease.lease_id, 3)                 # older watermark
    assert lease_table.committed == 5, "commits must not go backwards"
    result = lease_table.commit("a", lease.lease_id, 10)
    assert result["lease_closed"] and lease_table.committed == 10
    lease_table.verify()


@check("a foreign cluster cannot commit someone else's lease")
def _():
    lease_table, _clock = table(30)
    lease = lease_table.acquire("a", 10, ttl=100.0).leases[0]
    try:
        lease_table.commit("b", lease.lease_id, 5)
    except PermissionError:
        lease_table.verify()
        return
    raise AssertionError("expected PermissionError")


# --- expiry ----------------------------------------------------------------


@check("a silent cluster's uncommitted blocks return to the pool")
def _():
    lease_table, clock = table(40)
    lease = lease_table.acquire("dead", 20, ttl=60.0).leases[0]
    assert lease_table._free.total == 20
    clock.advance(61)
    assert lease_table.reap() == 20
    assert lease_table._free.total == 40, "the whole lease should be reclaimed"
    assert lease_table.clusters["dead"].blocks_lost == 20
    lease_table.verify()
    # And the blocks are grantable again, to someone else.
    grant = lease_table.acquire("alive", 40, ttl=60.0)
    assert sum(l.blocks for l in grant.leases) == 40


@check("expiry keeps the committed prefix and reclaims only the tail")
def _():
    lease_table, clock = table(40)
    lease = lease_table.acquire("half", 20, ttl=60.0).leases[0]
    lease_table.commit("half", lease.lease_id, lease.start + 12)
    clock.advance(61)
    assert lease_table.reap() == 8, "only the uncommitted 8 should come back"
    assert lease_table.committed == 12
    counts = lease_table.verify()
    assert counts["committed"] == 12 and counts["leased"] == 0, counts


@check("heartbeats renew, and report a reclaimed lease as invalid")
def _():
    lease_table, clock = table(40)
    lease = lease_table.acquire("a", 10, ttl=60.0).leases[0]
    clock.advance(40)
    reply = lease_table.heartbeat("a", {lease.lease_id: lease.start + 3}, ttl=60.0)
    assert reply["leases"][lease.lease_id]["valid"]
    clock.advance(40)
    assert not lease_table.leases or lease_table.leases[lease.lease_id].deadline > clock.t, \
        "the heartbeat should have pushed the deadline out"
    clock.advance(61)
    reply = lease_table.heartbeat("a", {lease.lease_id: lease.start + 3})
    assert reply["leases"][lease.lease_id]["valid"] is False, reply
    lease_table.verify()


@check("release returns the tail immediately, without waiting for the TTL")
def _():
    lease_table, _clock = table(40)
    grant = lease_table.acquire("leaving", 20, ttl=3600.0)
    lease_table.commit("leaving", grant.leases[0].lease_id, grant.leases[0].start + 5)
    returned = lease_table.release("leaving")
    assert returned == 15, returned
    assert lease_table._free.total == 35
    lease_table.verify()


# --- work stealing ---------------------------------------------------------


@check("an idle cluster steals the unstarted tail of a straggler")
def _():
    lease_table, clock = table(40)
    slow = lease_table.acquire("slow", 40, ttl=600.0).leases[0]
    assert slow.blocks == 40, "the slow cluster should hold the whole space"
    lease_table.heartbeat("slow", {slow.lease_id: slow.start + 4}, ttl=600.0)

    grant = lease_table.acquire("fast", 20, ttl=600.0)
    assert grant.status == "granted", grant.reason
    stolen = sum(l.blocks for l in grant.leases)
    assert stolen > 0, "nothing was stolen"
    assert lease_table.leases[slow.lease_id].end == grant.leases[0].start, \
        "the victim's end must be exactly where the thief's span begins"
    assert lease_table.leases[slow.lease_id].end > 4, "must not cut below progress"
    lease_table.verify()


@check("a steal never crosses the victim's progress watermark")
def _():
    lease_table, _clock = table(20)
    victim = lease_table.acquire("slow", 20, ttl=600.0).leases[0]
    assert victim.blocks == 20
    # The victim is 12 blocks in; only [12, 20) may be taken.
    lease_table.heartbeat("slow", {victim.lease_id: 12}, ttl=600.0)
    grant = lease_table.acquire("fast", 20, ttl=600.0)
    assert grant.leases, "there were 8 unstarted blocks to steal"
    for lease in grant.leases:
        assert lease.start >= 12, f"stole in-flight work at {lease.start}"
    lease_table.verify()


@check("stealing off, and the pool simply drains")
def _():
    lease_table, _clock = table(20, allow_stealing=False)
    lease_table.acquire("slow", 20, ttl=600.0)
    grant = lease_table.acquire("fast", 10, ttl=600.0)
    assert grant.status == "drain", grant.status
    assert grant.retry_after > 0, "a drained client needs a retry hint"


# --- quarantine ------------------------------------------------------------


@check("a block that keeps expiring is quarantined, and the epoch still ends")
def _():
    lease_table, clock = table(10, max_attempts=2)
    for _ in range(2):
        lease_table.acquire("cursed", 10, ttl=10.0)
        clock.advance(11)
        lease_table.reap()
    assert lease_table.quarantined == 10, lease_table.quarantined
    assert lease_table.epoch_complete, "quarantine has to let the epoch terminate"
    grant = lease_table.acquire("cursed", 10)
    assert grant.status == "epoch_complete"
    assert "QUARANTINED" in grant.reason, grant.reason
    lease_table.verify()


@check("max_attempts=0 never quarantines")
def _():
    lease_table, clock = table(10, max_attempts=0)
    for _ in range(5):
        lease_table.acquire("cursed", 10, ttl=10.0)
        clock.advance(11)
        lease_table.reap()
    assert lease_table.quarantined == 0
    assert not lease_table.epoch_complete
    lease_table.verify()


# --- grant sizing ----------------------------------------------------------


@check("a cluster's first grant is smaller than what it asked for")
def _():
    lease_table, _clock = table(1000, first_grant_fraction=0.5)
    first = lease_table.acquire("new", 100, ttl=600.0)
    assert sum(l.blocks for l in first.leases) == 50, first.leases
    for lease in first.leases:
        lease_table.commit("new", lease.lease_id, lease.end)
    second = lease_table.acquire("new", 100, ttl=600.0)
    assert sum(l.blocks for l in second.leases) == 100, "measured clusters get the full ask"


@check("a scarce pool is split in proportion to measured throughput")
def _():
    lease_table, clock = table(1000)
    # Give both clusters a throughput history: 'fast' commits 10x as quickly.
    for cluster, seconds in (("fast", 1.0), ("slow", 10.0)):
        lease = lease_table.acquire(cluster, 100, ttl=600.0).leases[0]
        lease_table.commit(cluster, lease.lease_id, lease.end)
        clock.advance(seconds)
        lease = lease_table.acquire(cluster, 100, ttl=600.0).leases[0]
        clock.advance(seconds)
        lease_table.commit(cluster, lease.lease_id, lease.end)
    assert lease_table.clusters["fast"].rate > 5 * lease_table.clusters["slow"].rate

    free = lease_table._free.total
    fast = sum(l.blocks for l in lease_table.acquire("fast", free, ttl=600.0).leases)
    slow = sum(l.blocks for l in lease_table.acquire("slow", free, ttl=600.0).leases)
    assert fast > slow, f"fast got {fast}, slow got {slow}"
    lease_table.verify()


# --- durability ------------------------------------------------------------


@check("snapshot round-trips state, free pool and leases")
def _():
    lease_table, clock = table(100)
    drain(lease_table, "a", 10, limit=30)
    held = lease_table.acquire("b", 20, ttl=600.0).leases[0]
    lease_table.commit("b", held.lease_id, held.start + 5)

    snapshot = json.loads(json.dumps(lease_table.snapshot()))   # must be JSON-clean
    restored = LeaseTable.restore(snapshot, clock=clock, grace_s=300.0)
    assert restored.committed == lease_table.committed
    assert restored._free.total == lease_table._free.total
    assert set(restored.leases) == set(lease_table.leases)
    assert restored.leases[held.lease_id].deadline >= clock.t + 300, "grace not applied"
    counts = restored.verify()
    assert counts == lease_table.verify(), (counts, lease_table.verify())


@check("coordinator recovers committed blocks from snapshot plus journal")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        first = Coordinator(LeaseTable(60, min_ttl=10.0, first_grant_fraction=1.0),
                            tmp, snapshot_interval=0.0)
        first.register({"cluster": "a", "ranks": 4})
        grant = first.acquire({"cluster": "a", "blocks": 20, "ttl": 600.0,
                               "request_id": "r1"})
        lease = grant["leases"][0]
        first.save_snapshot(force=True)          # snapshot, then more work
        first.commit({"cluster": "a", "lease": lease["lease_id"], "through": lease["end"]})
        second_grant = first.acquire({"cluster": "a", "blocks": 10, "ttl": 600.0,
                                      "request_id": "r2"})
        assert second_grant["status"] == "granted"

        recovered = Coordinator.load(tmp, expect_blocks=60, grace_s=600.0)
        assert recovered is not None
        assert recovered.table.committed == 20, recovered.table.committed
        held = sum(l.outstanding for l in recovered.table.leases.values())
        assert held == 10, f"the post-snapshot lease should come back, got {held}"
        recovered.table.verify()


@check("replaying the journal twice does not double-count")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        coordinator = Coordinator(LeaseTable(40, min_ttl=10.0, first_grant_fraction=1.0),
                                  tmp, snapshot_interval=0.0)
        coordinator.register({"cluster": "a"})
        grant = coordinator.acquire({"cluster": "a", "blocks": 20, "ttl": 600.0,
                                     "request_id": "x"})
        lease = grant["leases"][0]
        coordinator.commit({"cluster": "a", "lease": lease["lease_id"],
                            "through": lease["end"]})
        coordinator.save_snapshot(force=True)
        # Replay the pre-snapshot journal on top of the snapshot by re-appending it.
        journal = Path(tmp) / "journal.jsonl"
        journal.write_text(json.dumps({"op": "acquire", "now": time.time(),
                                       "payload": {"cluster": "a", "leases": [lease]}}) + "\n")
        recovered = Coordinator.load(tmp, expect_blocks=40)
        assert recovered.table.committed == 20, recovered.table.committed
        recovered.table.verify()


# --- transport: a real coordinator, concurrent clients ---------------------


class _Server:
    """A real coordinator on an ephemeral port, for the duration of a `with`."""

    def __init__(self, blocks: int, token: str = "", **kwargs):
        kwargs.setdefault("min_ttl", 5.0)
        self.coordinator = Coordinator(LeaseTable(blocks, **kwargs), None)
        self.httpd = make_server(self.coordinator, host="127.0.0.1", port=0, token=token)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_Server":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


@check("HTTP: concurrent clusters cover the space exactly once")
def _():
    space = BlockSpace(num_samples=4000, block_size=10)      # 400 blocks
    with _Server(space.num_blocks, digest=space.digest()) as server:
        owner: dict[int, str] = {}
        duplicates: list[int] = []
        lock = threading.Lock()

        def cluster(name: str) -> None:
            client = LeaseClient(server.url, name, retries=2, timeout=5.0)
            session = LeaseSession(client, space, blocks_per_phase=7, min_ttl=5.0,
                                   heartbeat=False)
            while True:
                result = session.acquire(wait=False)
                if not result.granted:
                    if result.epoch_complete:
                        return
                    continue
                for span in result.spans:
                    session.note_consumed(span.lease_id, span.end)
                    if session.commit(span.lease_id, span.end):
                        with lock:
                            for position in range(span.start, span.end):
                                if position in owner:
                                    duplicates.append(position)
                                owner[position] = name

        threads = [threading.Thread(target=cluster, args=(f"c{i}",)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not any(t.is_alive() for t in threads), "a client thread hung"
        assert not duplicates, f"{len(duplicates)} duplicated blocks"
        missing = set(range(space.num_blocks)) - set(owner)
        assert not missing, f"{len(missing)} blocks never committed"
        server.coordinator.table.verify()
        assert len({v for v in owner.values()}) > 1, "the work should have been shared"


@check("HTTP: a retried acquire returns the same grant, not a second one")
def _():
    with _Server(100) as server:
        client = LeaseClient(server.url, "retrier", retries=1)
        first = client.acquire(10, ttl=600.0, request_id="same-key")
        second = client.acquire(10, ttl=600.0, request_id="same-key")
        assert [s.lease_id for s in first.spans] == [s.lease_id for s in second.spans]
        held = sum(l.outstanding for l in server.coordinator.table.leases.values())
        granted = sum(s.blocks for s in first.spans)
        assert held == granted, f"the retry leaked a second lease: {held} vs {granted}"


@check("HTTP: a drain is not waited out when this session's own prefetch holds the blocks")
def _():
    """The deadlock this pins down, observed intermittently in the titan dataloader.

    A phase's foreground acquire and the previous phase's prefetch are in flight at
    the same time; the prefetch wins the last blocks in the pool; the foreground is
    told `drain` and settles in to wait. Nothing can ever release it -- the only
    thread that would consume and commit those blocks is the one asleep in
    `acquire`, and the heartbeat keeps renewing the lease, so the TTL never expires
    either. The pool looks drained from the outside while the session is sitting on
    exactly the span it is waiting for.

    Two blocks, one per phase, so there is precisely one winner to hand out.
    """
    space = BlockSpace(num_samples=8, block_size=4)
    with _Server(space.num_blocks, digest=space.digest(), min_ttl=600.0,
                 first_grant_fraction=1.0) as server:
        client = LeaseClient(server.url, "solo", retries=1, timeout=5.0)
        session = LeaseSession(client, space, blocks_per_phase=1, heartbeat=False)

        first = session.acquire(wait=False)
        assert first.granted, first
        # Stage the next span the way next_phase() does, and let it land.
        session.start_prefetch()
        assert session._prefetch_done.wait(10.0), "the prefetch never finished"
        assert session._prefetch is not None and session._prefetch.granted

        # Retire the first span, so the only uncommitted blocks anywhere in the
        # coordinator are the ones this session's own prefetch is holding.
        for span in first.spans:
            session.note_consumed(span.lease_id, span.end)
        session.commit_all()
        assert server.coordinator.table.committed == 1

        # timeout=0.0 makes this an assertion about *ordering* rather than about the
        # wall clock: the pending grant has to be adopted before the deadline is
        # consulted. Before the fix this returned the drain reply, and with the
        # production timeout=None it never returned at all.
        result = session.acquire(wait=True, timeout=0.0)
        assert result.granted, (
            f"waited on a drain instead of consuming the span this session already "
            f"holds: {result.status} ({result.reason})"
        )
        session.close(release=True)


@check("HTTP: a caller collects the prefetch in flight instead of racing it")
def _():
    """The other half of the fix: don't create the race in the first place.

    Two acquires from one cluster queued at the coordinator at once is what put the
    session in the state above. `take_prefetched(wait=...)` closes it -- a caller
    that wants the next span joins the request already in flight rather than issuing
    a competing one. Here the prefetch is held mid-RPC, so waiting is the only thing
    that can produce a span, and the acquire count proves nothing raced it.
    """
    space = BlockSpace(num_samples=40, block_size=4)
    with _Server(space.num_blocks, digest=space.digest(), min_ttl=600.0,
                 first_grant_fraction=1.0) as server:
        client = LeaseClient(server.url, "gated", retries=1, timeout=5.0)
        session = LeaseSession(client, space, blocks_per_phase=1, heartbeat=False)

        inflight, release = threading.Event(), threading.Event()
        acquires = []
        real_acquire = client.acquire

        def gated(*args, **kwargs):
            acquires.append(1)
            inflight.set()
            release.wait(10.0)
            return real_acquire(*args, **kwargs)

        client.acquire = gated
        session.start_prefetch()
        assert inflight.wait(10.0), "the prefetch never issued its RPC"

        # A non-blocking peek has nothing yet, which is exactly the moment a caller
        # used to go off and acquire in parallel.
        assert session.take_prefetched(wait=0.0) is None
        release.set()

        result = session.take_prefetched(wait=10.0)
        assert result is not None and result.granted, result
        assert len(acquires) == 1, f"{len(acquires)} acquires reached the coordinator"
        assert len(server.coordinator.table.leases) == 1, "a second lease was taken"
        session.close(release=True)


@check("HTTP: releasing drops an uncollected prefetch as well as the held spans")
def _():
    """Otherwise a released span can still be handed to a phase.

    `release` returns every lease this cluster holds, including whatever the
    prefetcher won. If the session kept serving that grant afterwards, the cluster
    would train blocks another cluster is now free to lease -- a duplicate, and one
    nothing downstream would catch, because from the trainer's side the span looks
    perfectly valid.
    """
    space = BlockSpace(num_samples=40, block_size=4)
    with _Server(space.num_blocks, digest=space.digest(), min_ttl=600.0,
                 first_grant_fraction=1.0) as server:
        client = LeaseClient(server.url, "quitter", retries=1, timeout=5.0)
        session = LeaseSession(client, space, blocks_per_phase=1, heartbeat=False)
        session.start_prefetch()
        assert session._prefetch_done.wait(10.0)
        assert session._prefetch is not None and session._prefetch.granted

        session.release_all()
        assert session.take_prefetched(wait=0.0) is None, (
            "a released span was still on offer"
        )
        assert not server.coordinator.table.leases


@check("HTTP: two concurrent jobs sharing a cluster id corrupt each other's leases")
def _():
    """Documents a hazard rather than a fixed bug, so it asserts what DOES happen.

    Both HPCs allow partial-node allocations, so submitting two jobs to one facility
    can land them on the same node -- and the DARL cluster id defaults to the site name
    alone, on purpose, so a requeued job keeps the throughput history that sizes its
    grants. The consequence is that two *concurrent* jobs at one site are indis-
    tinguishable to the coordinator, and /release with no lease id is scoped to the
    cluster id: the first job to exit hands back the second job's live leases, which
    then get handed to someone else while the second job is still training them.

    Nothing downstream catches that, which is why scripts/titan/run_train.sh has
    --replica and why it is documented as a correctness flag. If the protocol ever
    grows a per-incarnation token (see TODO.md) this check should start failing, and
    the fix is to assert isolation instead.
    """
    space = BlockSpace(num_samples=400, block_size=10)
    with _Server(space.num_blocks, digest=space.digest(), min_ttl=600.0,
                 first_grant_fraction=1.0) as server:
        table = server.coordinator.table

        # Two jobs, same site, no --replica: the same cluster id.
        job_a = LeaseSession(LeaseClient(server.url, "lumi", retries=1, timeout=5.0),
                             space, blocks_per_phase=2, heartbeat=False)
        job_b = LeaseSession(LeaseClient(server.url, "lumi", retries=1, timeout=5.0),
                             space, blocks_per_phase=2, heartbeat=False)

        granted_a = job_a.acquire(wait=False)
        granted_b = job_b.acquire(wait=False)
        assert granted_a.granted and granted_b.granted
        # Disjoint spans, so the coordinator is at least not double-granting.
        blocks_a = {p for s in granted_a.spans for p in range(s.start, s.end)}
        blocks_b = {p for s in granted_b.spans for p in range(s.start, s.end)}
        assert not (blocks_a & blocks_b), "the coordinator granted the same block twice"

        # Job A hits walltime and releases on the way out, as it should.
        job_a.release_all()

        # Job B is still running and still believes it holds its span -- but the
        # coordinator has already given those blocks back to the free pool.
        assert job_b.spans, "job B still thinks it holds a lease"
        for lease_id in job_b.spans:
            assert lease_id not in table.leases, (
                "expected job A's release to have taken job B's lease with it; if this "
                "now holds, the protocol gained per-job isolation and this check "
                "should be inverted"
            )
        # Which is the actual damage: every one of job B's blocks is back in the free
        # pool, claimable by anyone, while job B goes on training them. Asserted against
        # the block states rather than by acquiring again -- the allocator prefers a
        # cluster's cursor, so where a third cluster happens to be handed blocks from
        # says nothing about which ones are claimable.
        from pww.darl.table import BlockState

        assert all(table._state[p] == BlockState.UNASSIGNED for p in blocks_b), (
            "job B's blocks should be back in the pool after job A's release"
        )


@check("HTTP: distinct replica ids keep two jobs at one site isolated")
def _():
    """The fix available today: --replica in scripts/titan/run_train.sh.

    With distinct cluster ids the release is scoped to the job that made it, which is
    what makes two concurrent jobs at one facility safe.
    """
    space = BlockSpace(num_samples=400, block_size=10)
    with _Server(space.num_blocks, digest=space.digest(), min_ttl=600.0,
                 first_grant_fraction=1.0) as server:
        table = server.coordinator.table
        job_a = LeaseSession(LeaseClient(server.url, "lumi-a", retries=1, timeout=5.0),
                             space, blocks_per_phase=2, heartbeat=False)
        job_b = LeaseSession(LeaseClient(server.url, "lumi-b", retries=1, timeout=5.0),
                             space, blocks_per_phase=2, heartbeat=False)
        assert job_a.acquire(wait=False).granted
        granted_b = job_b.acquire(wait=False)
        assert granted_b.granted

        job_a.release_all()

        for lease_id in job_b.spans:
            assert lease_id in table.leases, (
                f"job A's release took lease {lease_id} from job B despite distinct "
                f"cluster ids"
            )
        assert all(l.cluster == "lumi-b" for l in table.leases.values())


@check("HTTP: the token is enforced")
def _():
    from pww.darl.client import DarlError

    with _Server(20, token="secret") as server:
        good = LeaseClient(server.url, "ok", token="secret", retries=1)
        assert good.status()["num_blocks"] == 20
        bad = LeaseClient(server.url, "no", token="wrong", retries=1)
        try:
            bad.status()
        except DarlError:
            return
        raise AssertionError("an unauthenticated client must be refused")


@check("HTTP: a digest mismatch is refused at registration")
def _():
    from pww.darl.client import DarlError

    space = BlockSpace(num_samples=1000, block_size=10, seed=1)
    other = BlockSpace(num_samples=1000, block_size=10, seed=999)
    with _Server(space.num_blocks, digest=space.digest()) as server:
        client = LeaseClient(server.url, "skewed", retries=1)
        try:
            client.register(other)
        except DarlError as exc:
            assert "digest" in str(exc), exc
            return
    raise AssertionError("a client with a different permutation must be refused")


# --- the trainer side ------------------------------------------------------


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


if not _torch_available():
    print("  SKIP  torch_data checks (torch not importable)")
else:
    from pww.darl.torch_data import DARLDataSource, LeasedSampler  # noqa: E402

    class _FakeSession:
        """Enough of a LeaseSession to drive DARLDataSource in one process."""

        def __init__(self, lease_table: LeaseTable, cluster: str = "sim"):
            self.table = lease_table
            self.cluster = cluster
            self.consumed: dict[str, int] = {}
            self.client = type("C", (), {"cluster_id": cluster})()
            self.epoch_complete = False

        def take_prefetched(self):
            return None

        def acquire(self, blocks=None, *, wait=True, timeout=None):
            from pww.darl.client import Acquisition, Span

            grant = self.table.acquire(self.cluster, blocks or 4, ttl=600.0)
            self.epoch_complete = grant.status == "epoch_complete"
            return Acquisition(
                status=grant.status, epoch=grant.epoch,
                spans=[Span(lease_id=l.lease_id, epoch=l.epoch, start=l.start, end=l.end,
                            ttl=l.ttl, deadline=l.deadline) for l in grant.leases],
            )

        def note_consumed(self, lease_id, through):
            self.consumed[lease_id] = through

        def start_prefetch(self, blocks=None):
            pass

        def note_phase_time(self, seconds):
            pass

        def commit_all(self):
            return sum(self.table.commit(self.cluster, lease_id, through)["committed_end"]
                       for lease_id, through in self.consumed.items()
                       if lease_id in self.table.leases)

    def _sources(lease_table: LeaseTable, space: BlockSpace, world: int):
        """One DARLDataSource per rank, sharing a stubbed broadcast."""
        box: dict[str, object] = {}

        def broadcast(payload, is_leader):
            if is_leader:
                box["payload"] = payload
            return box.get("payload")

        session = _FakeSession(lease_table)
        return [
            DARLDataSource(space, session if rank == 0 else None, rank=rank,
                           world_size=world, blocks_per_phase=4, broadcast=broadcast,
                           seed=11)
            for rank in range(world)
        ]

    @check("a phase is split across ranks with no overlap and equal counts")
    def _():
        space = BlockSpace(num_samples=800, block_size=10)
        lease_table, _clock = table(space.num_blocks)
        sources = _sources(lease_table, space, world=4)
        phases = [source.next_phase() for source in sources]
        assert all(p is not None for p in phases)
        counts = {len(p.indices) for p in phases}
        assert len(counts) == 1, f"ranks got different sample counts: {counts}"
        flat = [i for p in phases for i in p.indices]
        assert len(flat) == len(set(flat)), "two ranks were handed the same sample"
        expected = space.span_indices(*phases[0].spans[0][1:], epoch=0)
        assert set(flat) == set(expected), "the phase must cover exactly its span"

    @check("the per-phase remainder is carried, not dropped")
    def _():
        # 3 blocks of 10 samples over 4 ranks: 30 samples, 2 left over.
        space = BlockSpace(num_samples=120, block_size=10)
        lease_table, _clock = table(space.num_blocks)
        sources = _sources(lease_table, space, world=4)
        for source in sources:
            source.blocks_per_phase = 3
        seen: list[int] = []
        for _ in range(4):
            phases = [source.next_phase() for source in sources]
            if any(p is None for p in phases):
                break
            seen += [i for p in phases for i in p.indices]
        assert len(seen) == len(set(seen)), "carried samples were handed out twice"
        assert len(seen) >= 108, f"only {len(seen)} of 120 samples reached a rank"

    @check("a leader without a session, or a follower with one, is refused")
    def _():
        space = BlockSpace(num_samples=100, block_size=10)
        lease_table, _clock = table(space.num_blocks)
        try:
            DARLDataSource(space, None, rank=0, world_size=2)
        except ValueError:
            pass
        else:
            raise AssertionError("a leader must hold the session")
        try:
            DARLDataSource(space, _FakeSession(lease_table), rank=1, world_size=2)
        except ValueError:
            return
        raise AssertionError("a follower must not hold a session")

    @check("the sampler yields exactly the indices it was given")
    def _():
        sampler = LeasedSampler([5, 3, 1])
        assert list(sampler) == [5, 3, 1] and len(sampler) == 3
        sampler.set_indices(range(4))
        assert list(sampler) == [0, 1, 2, 3], "set_indices must replace, not append"


print()
print(f"{len(PASSED)} passed, {len(FAILED)} failed")
for name, exc in FAILED:
    print(f"  {name}: {type(exc).__name__}: {exc}")
sys.exit(1 if FAILED else 0)
