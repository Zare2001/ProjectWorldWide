"""The lease table: DARL's state machine.

Pure Python, no torch, no I/O, no threads, and time enters through an injectable
clock. Everything that has to be *correct* lives here, so that all of it is
testable in a single process on a login node (`tests/test_darl.py`), and the
server around it (`server.py`) is only transport and durability.

The three invariants, restated as code (`LeaseTable.verify` asserts all three):

    completeness      every block ends the epoch COMMITTED (or QUARANTINED, and
                      then loudly)
    disjointness      a block is in exactly one state, and at most one lease
                      covers it
    zero duplication  sum of committed blocks over clusters == M

What "processed" means is worth being precise about, because it is where a naive
implementation leaks duplicates. A block counts as processed when it is
COMMITTED, and a client is supposed to commit only what a durable checkpoint
covers. So a cluster that trains on a span and then dies before checkpointing
has *not* processed it: those gradients died with the process, the model rolled
back to the last checkpoint, and the block correctly returns to the pool. If you
instead commit on consumption (cheaper, one fewer round trip), then a job that
dies between the commit and the checkpoint gives you a real gap, and an expiry
after consumption gives you a real duplicate -- bounded by one lease either way.
`client.CommitPolicy` makes that choice explicit rather than accidental.

Block states
------------
    UNASSIGNED   free; any cluster may lease it
    LEASED       held by exactly one cluster under a heartbeat TTL
    COMMITTED    durably processed, never re-issued this epoch
    QUARANTINED  reclaimed `max_attempts` times without ever being committed

The design writeup lists EXPIRED as a fourth state; here expiry is a transition,
not a resting state, because a reaped block must be immediately indistinguishable
from UNASSIGNED or the allocator would need a second sweep before it could hand
it out. The information EXPIRED carried is kept as a per-block attempt counter,
which is strictly more useful: a block that has been reclaimed three times is not
a straggler, it is poison -- a corrupt shard, or a sample that OOMs whoever
touches it -- and cycling it forever means the epoch never completes. Hence
QUARANTINED, which trades exact completeness for termination and says so in the
status report. `max_attempts=0` disables it, for runs where a missing block is
worse than a hung epoch.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Callable, Iterable

# Lease TTL policy from the design note: Delta_t_TTL = alpha * macro_step + beta * RTT.
# alpha covers the local phase the client is presently inside plus a margin for
# the one it will start before its next heartbeat; beta * RTT is what keeps a slow
# or proxied WAN link from looking like a dead cluster.
TTL_ALPHA = 2.5
TTL_BETA = 10.0
DEFAULT_TTL = 900.0
MIN_TTL = 30.0
MAX_TTL = 6 * 3600.0

# Heartbeats go out at TTL/HEARTBEAT_DIVISOR, so a lease survives losing two
# consecutive heartbeats. Anything less makes a single dropped packet on a
# congested WAN look like a crashed cluster.
HEARTBEAT_DIVISOR = 3


def compute_ttl(
    macro_step_s: float,
    rtt_s: float = 0.0,
    *,
    alpha: float = TTL_ALPHA,
    beta: float = TTL_BETA,
    floor: float = MIN_TTL,
    ceiling: float = MAX_TTL,
) -> float:
    """Δt_TTL = α·t̄_step + β·RTT_WAN, clamped.

    Both terms matter. Dropping the RTT term expires leases held by a cluster
    that is merely far away; dropping the step term expires a cluster whose
    macro-step legitimately takes longer than the timeout, which is the normal
    case for an LLM inner loop of several hundred steps.
    """
    ttl = alpha * max(0.0, macro_step_s) + beta * max(0.0, rtt_s)
    return float(min(ceiling, max(floor, ttl)))


def _lease_seq(lease_id: str) -> int:
    """The counter out of `e<epoch>-b<start>-<seq>`, 0 if unparseable.

    Restoring it keeps lease ids unique across a coordinator restart, which
    matters because a client may still be holding an id issued before the crash.
    """
    try:
        return int(lease_id.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0


class BlockState(IntEnum):
    UNASSIGNED = 0
    LEASED = 1
    COMMITTED = 2
    QUARANTINED = 3


@dataclass
class Lease:
    """One contiguous span of positions granted to one cluster.

    committed_end and progress_end are both watermarks into [start, end):

        start .......... committed_end ...... progress_end ...... end
              durable ^            consumed ^          unstarted ^

    Only [committed_end, end) is at risk on expiry. progress_end exists purely to
    bound work stealing: a thief may only take positions the holder has not begun,
    so a steal can never duplicate in-flight work.
    """

    lease_id: str
    cluster: str
    epoch: int
    start: int
    end: int
    committed_end: int
    progress_end: int
    granted_at: float
    deadline: float
    ttl: float
    stolen: int = 0

    @property
    def blocks(self) -> int:
        return self.end - self.start

    @property
    def outstanding(self) -> int:
        return self.end - self.committed_end

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClusterRecord:
    """What the coordinator knows about one participating HPC cluster.

    All of it is derived from the RPCs the cluster already has to make, so a
    cluster never has to describe its own hardware: the grant sizer and the
    straggler ranking both run off measured throughput, which is the only thing
    that actually predicts how long a span will take there.
    """

    cluster_id: str
    joined_at: float
    last_seen: float
    ranks: int = 0
    grants: int = 0
    blocks_granted: int = 0
    blocks_committed: int = 0
    blocks_lost: int = 0          # reclaimed by expiry
    blocks_stolen_from: int = 0
    blocks_stolen_by: int = 0
    expiries: int = 0
    macro_step_s: float = 0.0
    rtt_s: float = 0.0
    rate: float = 0.0             # committed blocks/s, EWMA
    cursor: int = 0               # next position to prefer, for cache locality
    incarnation: str = ""         # which *process* is currently this cluster
    incarnations: int = 0         # how many have held it, i.e. requeue count
    _last_commit_at: float = 0.0

    @property
    def is_new(self) -> bool:
        """No completed work yet, so no throughput estimate to size a grant by."""
        return self.blocks_committed == 0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}


@dataclass
class Grant:
    """Reply to an acquire. `status` decides what the client does next."""

    status: str                   # granted | drain | epoch_complete
    epoch: int
    leases: list[Lease] = field(default_factory=list)
    retry_after: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "epoch": self.epoch,
            "leases": [lease.to_dict() for lease in self.leases],
            "retry_after": self.retry_after,
            "reason": self.reason,
        }


class DoubleFreeError(RuntimeError):
    """A block was returned to the free pool while already in it.

    Fatal on purpose: it means two leases covered the same position, which is the
    one bug this whole module exists to prevent. Better to stop the run than to
    train on silently duplicated data.
    """


class ClusterBusy(RuntimeError):
    """A second live process tried to register under an id one already holds.

    Distinct from a digest mismatch, which is permanent and means the operator has
    two different corpora: this one clears by itself when the incumbent's leases
    expire, so the client is told to wait rather than to give up. See
    `LeaseTable.register`.
    """


class IntervalSet:
    """Sorted, merged, half-open integer intervals -- the free pool.

    Intervals rather than a per-block free list because grants must be contiguous
    (contiguity is what keeps reads sequential), and because the pool spends most
    of an epoch as a handful of large runs. Fragmentation is bounded by the number
    of leases that have ever expired, which is small.
    """

    __slots__ = ("_starts", "_ends", "_total")

    def __init__(self, intervals: Iterable[tuple[int, int]] = ()) -> None:
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._total = 0
        for start, end in intervals:
            self.add(start, end)

    # --- mutation ---------------------------------------------------------

    def add(self, start: int, end: int) -> None:
        """Insert [start, end), merging into neighbours. Rejects overlap."""
        if end <= start:
            return
        import bisect

        added = end - start
        removed = 0
        i = bisect.bisect_left(self._starts, start)

        # The interval before i can overlap or abut this one.
        if i > 0 and self._ends[i - 1] >= start:
            i -= 1
            removed += self._ends[i] - self._starts[i]
            start = min(start, self._starts[i])
            end = max(end, self._ends[i])
            del self._starts[i], self._ends[i]

        # Absorb every following interval that touches the merged range.
        while i < len(self._starts) and self._starts[i] <= end:
            removed += self._ends[i] - self._starts[i]
            end = max(end, self._ends[i])
            del self._starts[i], self._ends[i]

        merged = end - start
        if merged < removed + added:
            # A merge that loses length means the added range overlapped one that
            # was already free.
            raise DoubleFreeError(
                f"[{start}, {end}) overlaps the free pool: adding {added} blocks to "
                f"{removed} produced only {merged}"
            )
        self._starts.insert(i, start)
        self._ends.insert(i, end)
        self._total += merged - removed

    def take_near(self, cursor: int, blocks: int, max_spans: int = 4) -> list[tuple[int, int]]:
        """Remove up to `blocks` blocks as at most `max_spans` contiguous spans.

        Allocation starts at the first interval at or after `cursor` and wraps,
        which gives each cluster an affinity region: a cluster that has been
        leasing around position p keeps getting positions near p, so whatever it
        has already staged or cached stays useful. When its region is exhausted it
        wraps and takes from wherever work remains, which is what makes the pool
        common rather than statically partitioned.

        Returning several spans in one call is deliberate: after a few expiries
        the tail of an epoch is fragmented, and one RPC yielding a work vector of
        four spans beats four round trips over a WAN.
        """
        import bisect

        if blocks <= 0 or not self._starts:
            return []

        spans: list[tuple[int, int]] = []
        remaining = blocks
        start_index = bisect.bisect_left(self._starts, cursor)
        # Not bisect_right: an interval containing the cursor is still the nearest.
        if start_index > 0 and self._ends[start_index - 1] > cursor:
            start_index -= 1

        visited = 0
        index = start_index
        while remaining > 0 and len(spans) < max_spans and visited < len(self._starts):
            if index >= len(self._starts):
                index = 0                       # wrap
                if not self._starts:
                    break
            lo, hi = self._starts[index], self._ends[index]
            take_from = max(lo, cursor) if (lo <= cursor < hi and not spans) else lo
            take_to = min(hi, take_from + remaining)
            spans.append((take_from, take_to))
            remaining -= take_to - take_from
            self._remove_at(index, take_from, take_to)
            # _remove_at may leave 0, 1 or 2 intervals where there was one.
            if take_from > lo and take_to < hi:
                index += 2                      # split: skip the new right part
            elif take_from == lo and take_to == hi:
                pass                            # consumed: next interval slid in
            else:
                index += 1
            visited += 1

        return spans

    def remove(self, start: int, end: int) -> None:
        """Remove an exact range that must be entirely free.

        Only used by journal replay, which knows precisely which blocks a lease
        held and must reproduce that rather than re-run the allocator.
        """
        import bisect

        if end <= start:
            return
        index = bisect.bisect_right(self._starts, start) - 1
        if index < 0 or self._ends[index] < end:
            raise ValueError(f"[{start}, {end}) is not a free range")
        self._remove_at(index, start, end)

    def _remove_at(self, index: int, start: int, end: int) -> None:
        lo, hi = self._starts[index], self._ends[index]
        if not (lo <= start < end <= hi):
            raise ValueError(f"[{start}, {end}) is not inside [{lo}, {hi})")
        self._total -= end - start
        left = (lo, start) if start > lo else None
        right = (end, hi) if end < hi else None
        del self._starts[index], self._ends[index]
        for offset, piece in enumerate([p for p in (left, right) if p]):
            self._starts.insert(index + offset, piece[0])
            self._ends.insert(index + offset, piece[1])

    # --- reading ----------------------------------------------------------

    @property
    def total(self) -> int:
        return self._total

    def intervals(self) -> list[tuple[int, int]]:
        return list(zip(self._starts, self._ends))

    def __bool__(self) -> bool:
        return bool(self._starts)

    def __len__(self) -> int:
        """Number of intervals -- i.e. fragmentation, not block count."""
        return len(self._starts)


class LeaseTable:
    """Authoritative state of one epoch's block assignment.

    Single-threaded by design: every mutation is a few microseconds, and the whole
    point of coarse leases is that the call rate is per-macro-step per-cluster --
    single digits per second even with dozens of clusters. `server.py` therefore
    just wraps every method in one lock rather than trying to make this concurrent,
    which is also what keeps the state machine auditable.
    """

    def __init__(
        self,
        num_blocks: int,
        *,
        digest: str = "",
        epoch: int = 0,
        max_epochs: int = 1,
        max_attempts: int = 3,
        min_blocks: int = 1,
        max_blocks: int = 0,
        first_grant_fraction: float = 0.5,
        allow_stealing: bool = True,
        steal_min_blocks: int = 2,
        min_ttl: float = MIN_TTL,
        clock: Callable[[], float] | None = None,
    ):
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")
        self.num_blocks = int(num_blocks)
        self.digest = digest
        self.epoch = int(epoch)
        self.max_epochs = int(max_epochs)
        self.max_attempts = int(max_attempts)
        self.min_blocks = max(1, int(min_blocks))
        self.max_blocks = int(max_blocks) or self.num_blocks
        self.first_grant_fraction = float(first_grant_fraction)
        self.allow_stealing = bool(allow_stealing)
        self.steal_min_blocks = max(1, int(steal_min_blocks))
        # Floor on any granted TTL. The default is sized for a real WAN; a
        # single-site pool with millisecond RTTs and short phases can lower it,
        # and the simulation does exactly that to make expiry observable in
        # seconds rather than minutes.
        self.min_ttl = float(min_ttl)
        self._clock = clock or time.time

        self.clusters: dict[str, ClusterRecord] = {}
        self.leases: dict[str, Lease] = {}
        self.epochs_completed = 0
        self.events: list[dict[str, Any]] = []      # ring buffer for /status
        self._seq = 0
        self._reset_epoch(self.epoch)

    # --- epoch lifecycle --------------------------------------------------

    def _reset_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.epoch_started_at = self._clock()
        self._state = bytearray(self.num_blocks)     # all UNASSIGNED
        self._attempts: dict[int, int] = {}
        self._free = IntervalSet([(0, self.num_blocks)])
        self.committed = 0
        self.quarantined = 0
        self.leases.clear()
        for record in self.clusters.values():
            record.cursor = 0

    def advance_epoch(self) -> bool:
        """Start the next epoch. False when the run's epoch budget is spent.

        Deliberately not automatic on the last epoch: "the epoch is complete" is
        the signal a trainer needs to stop, and silently rolling into epoch 2
        would turn a one-epoch pre-training run into an endless one.
        """
        if self.epoch + 1 >= self.max_epochs:
            return False
        self.epochs_completed += 1
        self._reset_epoch(self.epoch + 1)
        self._record("epoch_advanced", epoch=self.epoch)
        return True

    @property
    def epoch_complete(self) -> bool:
        return self.committed + self.quarantined >= self.num_blocks

    # --- cluster registration ---------------------------------------------

    def is_live(self, record: ClusterRecord, now: float) -> bool:
        """Whether a cluster is plausibly still running: heartbeating, or holding work.

        The same predicate `_active_clusters` uses to decide who counts when the pool
        is split proportionally, factored out because `register` needs it too -- it is
        the only thing that distinguishes a requeued job from a second concurrent one.
        """
        holding = any(lease.cluster == record.cluster_id for lease in self.leases.values())
        window = max(self.min_ttl, record.rtt_s * TTL_BETA)
        return holding or (now - record.last_seen) < window

    def register(
        self,
        cluster_id: str,
        *,
        digest: str = "",
        ranks: int = 0,
        incarnation: str = "",
        check_conflict: bool = True,
        now: float | None = None,
    ) -> ClusterRecord:
        """Announce a cluster. Idempotent -- a requeued job re-registers.

        Refuses a digest mismatch. That check is the difference between "these two
        clusters disagree about the permutation and will duplicate half the corpus"
        being a startup error and being an invisible data-quality bug.

        Also refuses a *second concurrent process* claiming an id that a live one
        already holds. Both facilities allow partial-node allocations, so two jobs
        submitted to one site can run at once, and the cluster id defaults to the site
        name alone -- deliberately, because excluding the Slurm job id is what lets a
        requeued job keep the measured throughput that sizes its grants. The cost was
        that two concurrent jobs were indistinguishable here, and the damage was
        silent: `release` is scoped by cluster id, so the first to exit handed back the
        other's live leases and it went on training blocks that were back in the pool.

        `incarnation` is a per-process random id, which makes the two cases
        distinguishable by liveness rather than by identity:

            predecessor stale  ->  a requeue. Take over, keep the record, so rate,
                                   cursor and commit history survive as before.
            predecessor live   ->  a second concurrent job. Refuse, and say so.

        Clients that send no incarnation get the old behaviour, so this cannot break a
        mixed-version deployment mid-run.

        `check_conflict=False` records the incarnation without the liveness test, which
        is what journal replay needs: replay re-applies a register that was already
        authorised when it was served, against wall-clock timestamps that make `is_live`
        meaningless. It still has to record the value, or a coordinator restart would
        leave the incumbent's incarnation blank and silently disable this check for the
        rest of the run -- a live client registers once, at session start, and never
        again.
        """
        now = self._now(now)
        if digest and self.digest and digest != self.digest:
            raise ValueError(
                f"block-space digest mismatch: coordinator has {self.digest[:12]}..., "
                f"cluster {cluster_id!r} computed {digest[:12]}.... The two sides "
                f"disagree about which samples a position refers to, so leases "
                f"would not be disjoint. Check num_samples, block_size and seed."
            )
        record = self.clusters.get(cluster_id)
        if record is None:
            record = ClusterRecord(cluster_id=cluster_id, joined_at=now, last_seen=now)
            self.clusters[cluster_id] = record
            self._record("registered", cluster=cluster_id)
        elif incarnation and record.incarnation and incarnation != record.incarnation:
            if check_conflict and self.is_live(record, now):
                held = sum(l.outstanding for l in self.leases.values()
                           if l.cluster == cluster_id)
                raise ClusterBusy(
                    f"cluster id {cluster_id!r} is already held by a running process "
                    f"(last seen {now - record.last_seen:.0f}s ago, {held} blocks "
                    f"outstanding). Two concurrent jobs sharing one cluster id corrupt "
                    f"each other's leases and overwrite each other's weight deltas, so "
                    f"this is refused rather than allowed to happen quietly. Give each "
                    f"job its own id: pass --replica a / --replica b to "
                    f"scripts/titan/run_train.sh, or set --darl.cluster_id directly. "
                    f"If the previous job really is dead, this clears by itself once "
                    f"its leases expire."
                )
            # Stale predecessor: a requeue, which is the normal walltime path. Its
            # leases are reaped on their own deadlines; nothing to do but take over.
            self._record("superseded", cluster=cluster_id,
                         silent_for=round(now - record.last_seen, 1))

        if incarnation and incarnation != record.incarnation:
            record.incarnation = incarnation
            record.incarnations += 1
        record.last_seen = now
        if ranks:
            record.ranks = int(ranks)
        return record

    # --- the three hot RPCs -----------------------------------------------

    def acquire(
        self,
        cluster_id: str,
        blocks: int,
        *,
        ttl: float = 0.0,
        max_spans: int = 4,
        macro_step_s: float = 0.0,
        rtt_s: float = 0.0,
        now: float | None = None,
    ) -> Grant:
        """Atomically move blocks from UNASSIGNED to LEASED for this cluster.

        Atomic here means what it means in the design note: the state transition
        and the reply are one indivisible step, so two clusters asking at the same
        instant cannot both be told about the same block. That falls out of the
        single lock in `server.py` -- there is no compare-and-swap loop and no
        window between the read and the write.
        """
        now = self._now(now)
        self.reap(now)
        record = self.register(cluster_id, ranks=0, now=now)
        if macro_step_s:
            record.macro_step_s = float(macro_step_s)
        if rtt_s:
            record.rtt_s = float(rtt_s)

        if self.epoch_complete and not self.leases:
            if not self.advance_epoch():
                return Grant(status="epoch_complete", epoch=self.epoch,
                             reason=self._completion_reason())

        want = self._grant_size(record, blocks)
        spans = self._free.take_near(record.cursor, want, max_spans=max_spans)
        stolen = 0
        if not spans and self.allow_stealing:
            spans, stolen = self._steal(record, want, now)

        if not spans:
            # Nothing free and nothing stealable: every remaining block is held by
            # a cluster that is making progress on it. The right move is to wait
            # for the next lease boundary, not to spin.
            outstanding = sum(lease.outstanding for lease in self.leases.values())
            return Grant(
                status="drain" if outstanding else "epoch_complete",
                epoch=self.epoch,
                retry_after=self._retry_after(ttl),
                reason=(f"{outstanding} blocks outstanding across "
                        f"{len({l.cluster for l in self.leases.values()})} clusters"
                        if outstanding else self._completion_reason()),
            )

        ttl = compute_ttl(record.macro_step_s, record.rtt_s) if not ttl else float(ttl)
        ttl = min(MAX_TTL, max(self.min_ttl, ttl))
        leases = []
        for start, end in spans:
            self._seq += 1
            lease = Lease(
                lease_id=f"e{self.epoch}-b{start}-{self._seq}",
                cluster=cluster_id,
                epoch=self.epoch,
                start=start,
                end=end,
                committed_end=start,
                progress_end=start,
                granted_at=now,
                deadline=now + ttl,
                ttl=ttl,
            )
            self._fill(start, end, BlockState.LEASED)
            self.leases[lease.lease_id] = lease
            leases.append(lease)
            record.cursor = end % self.num_blocks

        granted = sum(lease.blocks for lease in leases)
        record.grants += 1
        record.blocks_granted += granted
        record.blocks_stolen_by += stolen
        record.last_seen = now
        self._record("acquired", cluster=cluster_id, blocks=granted,
                     spans=len(leases), stolen=stolen, ttl=round(ttl, 1))
        return Grant(status="granted", epoch=self.epoch, leases=leases)

    def heartbeat(
        self,
        cluster_id: str,
        progress: dict[str, int] | None = None,
        *,
        macro_step_s: float = 0.0,
        rtt_s: float = 0.0,
        ttl: float = 0.0,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Refresh this cluster's leases and report the authoritative view back.

        The reply is not an acknowledgement, it is an instruction. Each lease comes
        back with the `end` the coordinator believes in, which may be lower than
        the client's if the tail was stolen, and `valid: false` if the lease was
        reaped while the cluster was silent. A client that ignores the reply and
        trains to its own remembered `end` breaks disjointness -- which is why
        `client.LeaseSession` applies it before handing out any more samples.
        """
        now = self._now(now)
        self.reap(now)
        record = self.register(cluster_id, now=now)
        if macro_step_s:
            record.macro_step_s = float(macro_step_s)
        if rtt_s:
            record.rtt_s = float(rtt_s)
        record.last_seen = now

        renewed_ttl = float(ttl) if ttl else compute_ttl(record.macro_step_s, record.rtt_s)
        renewed_ttl = min(MAX_TTL, max(self.min_ttl, renewed_ttl))

        reply: dict[str, Any] = {}
        for lease_id, watermark in (progress or {}).items():
            lease = self.leases.get(lease_id)
            if lease is None or lease.cluster != cluster_id or lease.epoch != self.epoch:
                reply[lease_id] = {"valid": False, "end": 0, "reason": "expired or unknown"}
                continue
            # Clamp rather than reject: a client that prefetched past a steal is
            # not misbehaving, it just has stale information, and the clamp is the
            # correction.
            lease.progress_end = max(lease.progress_end, min(int(watermark), lease.end))
            lease.deadline = now + renewed_ttl
            lease.ttl = renewed_ttl
            reply[lease_id] = {"valid": True, "end": lease.end,
                               "committed_end": lease.committed_end}
        return {"epoch": self.epoch, "ttl": renewed_ttl, "leases": reply,
                "epoch_complete": self.epoch_complete}

    def commit(
        self,
        cluster_id: str,
        lease_id: str,
        through: int,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Mark [committed_end, through) durably processed.

        Monotonic and idempotent: a retried commit after a dropped reply is a
        no-op, which matters because the client retries commits (losing one would
        cost the epoch a re-run of the span).
        """
        now = self._now(now)
        lease = self.leases.get(lease_id)
        if lease is None:
            raise KeyError(
                f"lease {lease_id!r} is not active: it expired and its blocks were "
                f"returned to the pool, or it belongs to a previous epoch. Discard "
                f"this span's work and acquire again."
            )
        if lease.cluster != cluster_id:
            raise PermissionError(f"lease {lease_id!r} belongs to {lease.cluster!r}")
        through = min(max(int(through), lease.committed_end), lease.end)

        newly = through - lease.committed_end
        if newly:
            self._fill(lease.committed_end, through, BlockState.COMMITTED)
            self.committed += newly
            lease.committed_end = through
            lease.progress_end = max(lease.progress_end, through)

            record = self.clusters[cluster_id]
            record.blocks_committed += newly
            record.last_seen = now
            if record._last_commit_at:
                elapsed = max(1e-6, now - record._last_commit_at)
                rate = newly / elapsed
                # EWMA: one slow phase (a checkpoint, a node reboot) should move
                # the grant sizer, not redefine it.
                record.rate = rate if record.rate <= 0 else 0.7 * record.rate + 0.3 * rate
            record._last_commit_at = now

        closed = lease.committed_end >= lease.end
        if closed:
            del self.leases[lease_id]
        if newly:
            self._record("committed", cluster=cluster_id, blocks=newly, lease=lease_id)
        return {
            "ok": True,
            "epoch": self.epoch,
            "committed_end": lease.committed_end,
            "lease_closed": closed,
            "committed": self.committed,
            "epoch_complete": self.epoch_complete,
        }

    def release(
        self,
        cluster_id: str,
        lease_id: str | None = None,
        *,
        incarnation: str = "",
        count_attempt: bool = True,
        now: float | None = None,
    ) -> int:
        """Give back the uncommitted tail of one lease, or of all of them.

        The graceful counterpart to expiry, and worth wiring into a SIGTERM
        handler: Slurm sends one before walltime kill, so a job that releases on
        the way out returns its blocks in milliseconds instead of after a whole
        TTL. On a 15-minute TTL that is the difference between the surviving
        clusters idling for 15 minutes and not idling at all.

        A release-everything is scoped to the *incarnation* that asks, not just to the
        cluster id. `register` normally stops two live processes sharing an id in the
        first place, but a client from before that check -- or one that registered while
        the incumbent looked stale and then raced it -- must not be able to hand back
        leases it never held. Belt and braces on the one operation whose blast radius
        is every lease a name owns.
        """
        now = self._now(now)
        if lease_id:
            targets = [lease_id]
        else:
            targets = [lid for lid, l in self.leases.items() if l.cluster == cluster_id]
            record = self.clusters.get(cluster_id)
            if (incarnation and record is not None and record.incarnation
                    and incarnation != record.incarnation):
                self._record("release_refused", cluster=cluster_id,
                             leases=len(targets))
                return 0
        returned = 0
        for lid in targets:
            lease = self.leases.get(lid)
            if lease is None or lease.cluster != cluster_id:
                continue
            returned += self._reclaim(lease, now, reason="released",
                                      count_attempt=count_attempt)
        if returned:
            self._record("released", cluster=cluster_id, blocks=returned)
        return returned

    def replay_lease(self, data: dict[str, Any], *, deadline_floor: float = 0.0) -> bool:
        """Re-install a lease from the write-ahead log. Idempotent.

        Only positions that are *still* UNASSIGNED are re-taken, and the lease is
        clamped to them. That is what makes replaying an acquire on top of a
        snapshot that already contains its commits a no-op instead of a
        double-count -- which happens routinely, because the snapshot is written
        without waiting for the journal to be replayed away.
        """
        lease = Lease(**data)
        if lease.epoch != self.epoch or lease.lease_id in self.leases:
            return False

        position = lease.start
        while position < lease.end and self._state[position] != BlockState.UNASSIGNED:
            position += 1
        lo = position
        while position < lease.end and self._state[position] == BlockState.UNASSIGNED:
            position += 1
        hi = position
        if hi <= lo:
            return False

        self._free.remove(lo, hi)
        self._fill(lo, hi, BlockState.LEASED)
        lease.start = lo
        lease.end = hi
        lease.committed_end = lo
        lease.progress_end = min(max(lease.progress_end, lo), hi)
        lease.deadline = max(lease.deadline, deadline_floor)
        self.leases[lease.lease_id] = lease
        self._seq = max(self._seq, _lease_seq(lease.lease_id))
        return True

    # --- expiry and stealing ----------------------------------------------

    def reap(self, now: float | None = None) -> int:
        """Return every overdue lease's uncommitted blocks to the pool.

        Called at the top of every mutating RPC and by the server's background
        tick, so a dead cluster's blocks become claimable without anyone having to
        notice the death.
        """
        now = self._now(now)
        expired = [l for l in self.leases.values() if l.deadline < now]
        reclaimed = 0
        for lease in expired:
            blocks = self._reclaim(lease, now, reason="expired")
            reclaimed += blocks
            record = self.clusters.get(lease.cluster)
            if record is not None:
                record.blocks_lost += blocks
                record.expiries += 1
            if blocks:
                self._record("expired", cluster=lease.cluster, blocks=blocks,
                             lease=lease.lease_id,
                             silent_for=round(now - lease.deadline + lease.ttl, 1))
        return reclaimed

    def _reclaim(self, lease: Lease, now: float, *, reason: str,
                 count_attempt: bool = True) -> int:
        """Uncommitted tail -> UNASSIGNED (or QUARANTINED), lease gone.

        `count_attempt=False` returns the tail without holding the *blocks*
        responsible. Attempt counting exists to find a corrupt shard -- a position
        that fails under several different clusters -- so it must not fire when the
        cluster has already said the fault is its own. A site whose phase produced a
        non-finite loss hands its spans back for that reason, and quarantining them
        after three such rounds would retire perfectly good data because one
        participant was broken: worse than the waste it replaced, because the
        positions then leave the epoch permanently.
        """
        lo, hi = lease.committed_end, lease.end
        del self.leases[lease.lease_id]
        if hi <= lo:
            return 0

        # Attempt counting is per block, because a lease that expires repeatedly
        # is usually a cluster problem while a *block* that expires repeatedly
        # under different clusters is a data problem.
        free_runs: list[tuple[int, int]] = []
        run_start = lo
        for position in range(lo, hi):
            if not count_attempt:
                continue
            attempts = self._attempts.get(position, 0) + 1
            self._attempts[position] = attempts
            if self.max_attempts and attempts >= self.max_attempts:
                if run_start < position:
                    free_runs.append((run_start, position))
                run_start = position + 1
                self._state[position] = BlockState.QUARANTINED
                self.quarantined += 1
                self._record("quarantined", block=position, attempts=attempts,
                             cluster=lease.cluster)
        if run_start < hi:
            free_runs.append((run_start, hi))

        for start, end in free_runs:
            self._fill(start, end, BlockState.UNASSIGNED)
            self._free.add(start, end)
        return hi - lo

    def _steal(self, thief: ClusterRecord, blocks: int, now: float) -> tuple[list[tuple[int, int]], int]:
        """Take unstarted tails from the cluster that will finish last.

        Work stealing, with the deque tail played by the far end of a lease. The
        victim is chosen by estimated time to finish its own outstanding blocks
        (`outstanding / rate`), not by size, because the point is to unload the
        cluster that is actually going to make everyone wait -- a big lease on a
        fast cluster is not a straggler.

        Only positions at or beyond the victim's progress watermark are eligible,
        so a steal cannot duplicate work that is already in flight, and the victim
        finds out at its next heartbeat. Nothing is interrupted, and nothing has
        to be re-run.
        """
        candidates = []
        for lease in self.leases.values():
            if lease.cluster == thief.cluster_id:
                continue
            available = lease.end - max(lease.progress_end, lease.committed_end)
            if available < self.steal_min_blocks * 2:
                continue
            victim = self.clusters.get(lease.cluster)
            rate = victim.rate if victim and victim.rate > 0 else 1e-9
            candidates.append((lease.outstanding / rate, available, lease))
        if not candidates:
            return [], 0

        # Slowest first; ties broken by how much is takeable.
        candidates.sort(key=lambda c: (-c[0], -c[1]))
        spans: list[tuple[int, int]] = []
        taken = 0
        for _eta, available, lease in candidates:
            if taken >= blocks:
                break
            # Halve rather than empty: leaving the victim its front half keeps it
            # working, and repeated small steals converge on the right split
            # without needing to model either side's speed precisely.
            want = min(blocks - taken, max(self.steal_min_blocks, available // 2))
            cut = lease.end - want
            floor = max(lease.progress_end, lease.committed_end)
            if cut < floor:
                cut = floor
            if lease.end - cut < self.steal_min_blocks:
                continue
            spans.append((cut, lease.end))
            taken += lease.end - cut
            lease.stolen += lease.end - cut
            lease.end = cut
            victim = self.clusters.get(lease.cluster)
            if victim is not None:
                victim.blocks_stolen_from += spans[-1][1] - spans[-1][0]
            self._record("stolen", thief=thief.cluster_id, victim=lease.cluster,
                         blocks=spans[-1][1] - spans[-1][0], lease=lease.lease_id)
        return spans, taken

    # --- grant sizing -----------------------------------------------------

    def _grant_size(self, record: ClusterRecord, requested: int) -> int:
        """How much of the request to honour.

        Three caps, in order of how often they bite:

        1. A cluster's first grant is halved. It has no measured throughput yet,
           and a newly-queued cluster is the most likely thing in the system to
           die immediately (a bad module load, a missing dataset mirror, an OOM
           on the first step). Halving bounds what that costs.
        2. Rate-proportional fair share, once the pool is scarcer than the
           outstanding demand. A fast cluster gets a bigger slice of what is left,
           which is the whole reason a heterogeneous pool is worth running; equal
           shares here would make the epoch end at the slowest cluster's pace.
        3. An absolute ceiling, so no single cluster can lock up a large fraction
           of the epoch and turn DARL back into static partitioning.
        """
        requested = max(self.min_blocks, int(requested))
        if record.is_new:
            requested = max(self.min_blocks, int(requested * self.first_grant_fraction))
        requested = min(requested, self.max_blocks)

        free = self._free.total
        if free and requested > free // max(1, len(self._active_clusters())):
            active = self._active_clusters()
            rates = {c.cluster_id: max(c.rate, 1e-6) for c in active}
            total_rate = sum(rates.values()) or 1.0
            share = int(free * rates.get(record.cluster_id, 1e-6) / total_rate)
            requested = max(self.min_blocks, min(requested, max(share, self.min_blocks)))
        return requested

    def _active_clusters(self) -> list[ClusterRecord]:
        """Clusters plausibly still alive: heartbeating, or holding a lease."""
        now = self._clock()
        return [c for c in self.clusters.values()
                if self.is_live(c, now)] or list(self.clusters.values())

    # --- observability ----------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Everything a `darl status` call or a stalled-run postmortem needs."""
        now = self._clock()
        leased = sum(lease.outstanding for lease in self.leases.values())
        elapsed = max(1e-9, now - self.epoch_started_at)
        rate = self.committed / elapsed
        remaining = self.num_blocks - self.committed - self.quarantined
        return {
            "epoch": self.epoch,
            "max_epochs": self.max_epochs,
            "epochs_completed": self.epochs_completed,
            "num_blocks": self.num_blocks,
            "committed": self.committed,
            "leased": leased,
            "unassigned": self._free.total,
            "quarantined": self.quarantined,
            "fragments": len(self._free),
            "epoch_complete": self.epoch_complete,
            "progress": self.committed / self.num_blocks,
            "blocks_per_s": rate,
            "eta_s": (remaining / rate) if rate > 0 else None,
            "digest": self.digest,
            "active_leases": len(self.leases),
            "clusters": {cid: c.to_dict() for cid, c in self.clusters.items()},
            "leases": [lease.to_dict() for lease in self.leases.values()],
            "recent_events": self.events[-32:],
        }

    def verify(self) -> dict[str, int]:
        """Assert the three invariants over the whole block space.

        O(M + leases) and pure bookkeeping, so it is cheap enough to run on every
        snapshot rather than only in tests. It has caught more than it should have
        needed to: an off-by-one in a steal or an interval split shows up here as
        a coverage count, long before it shows up as a duplicated sample in a
        30-day pre-training run.
        """
        counts = {state.name.lower(): 0 for state in BlockState}
        for value in self._state:
            counts[BlockState(value).name.lower()] += 1

        # Disjointness: at most one lease covers any position.
        covered = bytearray(self.num_blocks)
        for lease in self.leases.values():
            if lease.epoch != self.epoch:
                raise AssertionError(f"lease {lease.lease_id} is from epoch {lease.epoch}")
            if not 0 <= lease.start <= lease.committed_end <= lease.end <= self.num_blocks:
                raise AssertionError(f"lease {lease.lease_id} has inverted watermarks")
            for position in range(lease.committed_end, lease.end):
                if covered[position]:
                    raise AssertionError(f"position {position} is covered by two leases")
                covered[position] = 1
                if self._state[position] != BlockState.LEASED:
                    raise AssertionError(
                        f"position {position} is in lease {lease.lease_id} but marked "
                        f"{BlockState(self._state[position]).name}"
                    )

        leased_positions = sum(covered)
        if leased_positions != counts["leased"]:
            raise AssertionError(
                f"{counts['leased']} blocks marked LEASED but leases cover {leased_positions}"
            )
        if self._free.total != counts["unassigned"]:
            raise AssertionError(
                f"free pool holds {self._free.total} blocks but {counts['unassigned']} "
                f"are marked UNASSIGNED"
            )
        if counts["committed"] != self.committed:
            raise AssertionError(f"committed counter {self.committed} != {counts['committed']}")
        if counts["quarantined"] != self.quarantined:
            raise AssertionError(
                f"quarantined counter {self.quarantined} != {counts['quarantined']}"
            )
        by_cluster = sum(c.blocks_committed for c in self.clusters.values())
        # Zero-duplication bound, across epochs: the per-cluster commit counters
        # are cumulative, so they may exceed this epoch's total but never fall
        # short of it.
        if by_cluster < self.committed:
            raise AssertionError(
                f"clusters report {by_cluster} committed blocks but the table has "
                f"{self.committed}"
            )
        if sum(counts.values()) != self.num_blocks:
            raise AssertionError("state array does not cover the block space")
        return counts

    # --- durability -------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Serialisable full state, with the block array run-length encoded.

        RLE because the array is highly ordered -- an epoch in progress is a few
        long runs -- so a million-block space snapshots to a few kilobytes instead
        of a few megabytes. That is what makes it cheap enough to pair with every
        model checkpoint, which is the whole point: Psi = {theta, M_committed,
        M_unassigned} has to be one atomic unit or a rollback of the model leaves
        the lease table describing progress that no longer exists.
        """
        runs: list[list[int]] = []
        for value in self._state:
            if runs and runs[-1][0] == value:
                runs[-1][1] += 1
            else:
                runs.append([int(value), 1])
        return {
            "version": 1,
            "num_blocks": self.num_blocks,
            "digest": self.digest,
            "epoch": self.epoch,
            "max_epochs": self.max_epochs,
            "epochs_completed": self.epochs_completed,
            "epoch_started_at": self.epoch_started_at,
            "committed": self.committed,
            "quarantined": self.quarantined,
            "seq": self._seq,
            "state_rle": runs,
            "attempts": {str(k): v for k, v in self._attempts.items()},
            "leases": [lease.to_dict() for lease in self.leases.values()],
            "clusters": {cid: c.to_dict() for cid, c in self.clusters.items()},
            "config": {
                "max_attempts": self.max_attempts,
                "min_blocks": self.min_blocks,
                "max_blocks": self.max_blocks,
                "first_grant_fraction": self.first_grant_fraction,
                "allow_stealing": self.allow_stealing,
                "steal_min_blocks": self.steal_min_blocks,
                "min_ttl": self.min_ttl,
            },
        }

    @classmethod
    def restore(
        cls,
        snapshot: dict[str, Any],
        *,
        clock: Callable[[], float] | None = None,
        grace_s: float = 0.0,
    ) -> "LeaseTable":
        """Rebuild from a snapshot.

        `grace_s` extends every restored lease's deadline. A coordinator on a
        login node will be restarted -- reboots, an accidental Ctrl-C, a full
        filesystem -- and without a grace period the first reap after restart
        yanks every span out from under clusters that are perfectly healthy and
        mid-phase. With it, they heartbeat, get renewed, and never notice.
        """
        config = snapshot.get("config", {})
        table = cls(
            snapshot["num_blocks"],
            digest=snapshot.get("digest", ""),
            epoch=snapshot.get("epoch", 0),
            max_epochs=snapshot.get("max_epochs", 1),
            clock=clock,
            **{k: config[k] for k in
               ("max_attempts", "min_blocks", "max_blocks", "first_grant_fraction",
                "allow_stealing", "steal_min_blocks", "min_ttl") if k in config},
        )
        now = table._clock()

        state = bytearray()
        for value, count in snapshot["state_rle"]:
            state.extend(bytes([value]) * count)
        if len(state) != table.num_blocks:
            raise ValueError(f"snapshot covers {len(state)} blocks, expected {table.num_blocks}")
        table._state = state
        table.committed = snapshot.get("committed", 0)
        table.quarantined = snapshot.get("quarantined", 0)
        table.epochs_completed = snapshot.get("epochs_completed", 0)
        table.epoch_started_at = snapshot.get("epoch_started_at", now)
        table._seq = snapshot.get("seq", 0)
        table._attempts = {int(k): v for k, v in snapshot.get("attempts", {}).items()}

        table._free = IntervalSet()
        run_start = None
        for position, value in enumerate(state):
            if value == BlockState.UNASSIGNED:
                run_start = position if run_start is None else run_start
            elif run_start is not None:
                table._free.add(run_start, position)
                run_start = None
        if run_start is not None:
            table._free.add(run_start, table.num_blocks)

        for cid, data in snapshot.get("clusters", {}).items():
            table.clusters[cid] = ClusterRecord(**data)
        for data in snapshot.get("leases", []):
            lease = Lease(**data)
            lease.deadline = max(lease.deadline, now + grace_s)
            table.leases[lease.lease_id] = lease
        return table

    # --- internals --------------------------------------------------------

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else float(now)

    def _fill(self, start: int, end: int, state: BlockState) -> None:
        self._state[start:end] = bytes([int(state)]) * (end - start)

    def _record(self, event: str, **fields: Any) -> None:
        self.events.append({"t": round(self._clock(), 3), "event": event, **fields})
        if len(self.events) > 512:
            del self.events[:256]

    def _retry_after(self, ttl: float) -> float:
        """How long a drained client should wait before asking again.

        A fraction of the shortest outstanding lease, so the retry lands shortly
        after the next plausible lease boundary rather than immediately.
        """
        if not self.leases:
            return self.min_ttl / HEARTBEAT_DIVISOR
        soonest = min(lease.deadline for lease in self.leases.values()) - self._clock()
        return float(max(1.0, min(ttl or DEFAULT_TTL, soonest) / HEARTBEAT_DIVISOR))

    def _completion_reason(self) -> str:
        if self.quarantined:
            return (
                f"epoch {self.epoch} complete with {self.quarantined} of "
                f"{self.num_blocks} blocks QUARANTINED after {self.max_attempts} failed "
                f"attempts each -- coverage is NOT 100%, check those blocks for "
                f"corruption"
            )
        return f"epoch {self.epoch} complete: all {self.num_blocks} blocks committed"
