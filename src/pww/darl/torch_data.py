"""Wiring DARL into a PyTorch training loop.

The shape of the integration follows from one constraint: **the coordinator talks
to one process per data-parallel stream, not one per rank.** So each stream has a
leader (rank 0 of its process group) that holds the `LeaseSession`, and the work
vector it acquires is distributed to the stream's other ranks over the process
group they already have. A 512-rank job therefore makes the same number of RPCs
as a 4-rank one.

    leader (rank 0)                     other ranks
    ---------------                     -----------
    session.acquire()  ------ broadcast_object_list ------>  same spans
    space.span_indices(...)                                  same call, no RPC
    shuffle(seed, epoch, phase)                              identical permutation
    indices[rank::world_size]                                disjoint stride
    session.start_prefetch()                                 --

Every rank derives its own sample list from the broadcast spans with a shared
seed, so no index lists cross the network -- only three integers per span.

What counts as a stream
-----------------------
Whatever must see disjoint data. Under DiLoCo that is one *replica*, not one job:
k replicas in one allocation are k independent models, and giving them the same
samples would make the outer average a plain gradient average with extra steps.
`for_diloco` therefore builds one stream per replica, each registering with the
coordinator under its own cluster id, which also lets the coordinator balance
between two replicas in the same job when one node is slower than the other.

Steps per phase
---------------
Every rank in a stream gets exactly the same number of samples (the remainder is
carried into the next phase rather than dropped), so with a fixed batch size and
`drop_last=True` every rank runs the same number of steps. That is not cosmetic:
DiLoCo's outer step is a collective triggered by a per-rank counter, so ranks
that disagree about the step count hang in the all-reduce rather than failing.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import torch.distributed as dist
from torch.utils.data import Sampler

from ..logging_utils import get_logger
from .client import Acquisition, LeaseSession
from .space import BlockSpace, blocks_for_phase


@dataclass
class Phase:
    """One local phase of work: which spans it came from, and this rank's samples."""

    index: int
    epoch: int
    spans: list[tuple[str, int, int]] = field(default_factory=list)   # lease_id, start, end
    indices: list[int] = field(default_factory=list)                  # this rank's samples
    blocks: int = 0
    samples_global: int = 0

    @property
    def lease_ids(self) -> list[str]:
        return [lease_id for lease_id, _, _ in self.spans]

    def describe(self) -> str:
        spans = ", ".join(f"[{s},{e})" for _, s, e in self.spans)
        return (f"phase {self.index} epoch {self.epoch}: {self.blocks} blocks {spans} -> "
                f"{len(self.indices)} samples/rank")


class LeasedSampler(Sampler[int]):
    """A sampler whose index list is replaced between phases.

    Not a `DistributedSampler`: the sharding across ranks already happened when
    the phase was built, so this sampler is deliberately dumb -- it yields exactly
    the indices it was given, in the order it was given them.

    Use `persistent_workers=False`. Persistent workers keep a forked copy of the
    sampler from the first iteration and would replay the first phase forever;
    re-forking costs milliseconds once per phase, against a phase that is hundreds
    of steps long.
    """

    def __init__(self, indices: Sequence[int] = ()):
        self.indices = list(indices)

    def set_indices(self, indices: Sequence[int]) -> None:
        self.indices = list(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class DARLDataSource:
    """Turns leases into per-rank sample lists, one phase at a time.

    Owns the collective, so it must be entered by every rank of the stream the
    same number of times -- like any other collective in the loop. `next_phase()`
    returning None is the epoch boundary, and every rank learns it in the same
    call.
    """

    def __init__(
        self,
        space: BlockSpace,
        session: LeaseSession | None,
        *,
        rank: int = 0,
        world_size: int = 1,
        group: Any = None,
        leader_global_rank: int = 0,
        seed: int = 0,
        blocks_per_phase: int | None = None,
        shuffle: bool = True,
        broadcast: Any = None,
    ):
        if (session is None) == (rank == 0 and _global_rank() == leader_global_rank):
            # A leader without a session cannot acquire; a follower with one would
            # make duplicate RPCs and, worse, hold a second set of leases.
            raise ValueError(
                "exactly the stream leader must hold the LeaseSession: rank "
                f"{rank} (global {_global_rank()}, leader {leader_global_rank}) "
                f"{'has no session' if session is None else 'has one'}"
            )
        self.space = space
        self.session = session
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.group = group
        self.leader_global_rank = int(leader_global_rank)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.blocks_per_phase = blocks_per_phase
        # A seam, not an abstraction for its own sake: it lets the sharding maths be
        # tested in one process, and it is where a different transport would go if
        # the stream's ranks ever stop sharing a process group.
        self._broadcast_fn = broadcast

        self.epoch = 0
        self.phase_index = 0
        self.samples_seen = 0
        self.blocks_seen = 0
        self.epoch_complete = False
        self._carry: list[int] = []
        self._phase_t0 = 0.0

    @property
    def is_leader(self) -> bool:
        return self.session is not None

    # --- the phase loop ---------------------------------------------------

    def next_phase(self) -> Phase | None:
        """Acquire (leader) or receive (followers) the next work vector.

        Collective. Returns None when the epoch is complete, at which point every
        rank has been told the same thing and the loop can exit without a barrier.
        """
        payload: dict[str, Any] | None = None
        if self.is_leader:
            payload = self._acquire_payload()
        payload = self._broadcast(payload)

        if payload is None:
            self.epoch_complete = True
            return None

        self.epoch = int(payload["epoch"])
        spans = [(lease_id, int(start), int(end)) for lease_id, start, end in payload["spans"]]
        phase = self._build_phase(spans)

        if self.is_leader:
            # The whole span is handed to the ranks at once, so it is consumed from
            # the coordinator's point of view: a thief may not take any of it, and
            # the next heartbeat says so.
            for lease_id, _start, end in spans:
                self.session.note_consumed(lease_id, end)
            # Overlap the next acquire with this phase's compute.
            self.session.start_prefetch()

        self.phase_index += 1
        self.blocks_seen += phase.blocks
        self.samples_seen += phase.samples_global
        self._phase_t0 = time.monotonic()
        if self.is_leader:
            get_logger().info("darl: %s", phase.describe())
        return phase

    def end_phase(self) -> None:
        """Close out a phase: record its duration so the TTL tracks reality.

        Called by the trainer right after the outer step, i.e. at the same point
        the lease boundary was sized for.
        """
        if self.is_leader and self._phase_t0:
            self.session.note_phase_time(time.monotonic() - self._phase_t0)

    def commit(self) -> int:
        """Mark every held span durably processed. Call *after* the checkpoint.

        Leader-only and non-collective, so it is safe inside a rank-0 checkpoint
        branch. Returns blocks committed.
        """
        return self.session.commit_all() if self.is_leader else 0

    def release_unused(self, *, count_attempt: bool = True) -> int:
        """Release any pre-fetched but un-trained leases back to the pool.

        Call this when the caller is done with its inner steps but the session
        may still hold blocks from aggressive prefetching. Without this, a fast
        cluster can hold blocks that a slower cluster needs to complete its
        epoch, causing a deadlock where the slower cluster spins in
        'pool drained' waiting for blocks that will never be trained on.

        Leader-only, non-collective.
        """
        if not self.is_leader:
            return 0
        released = self.session.release_all(count_attempt=count_attempt)
        if released > 0:
            get_logger().info("darl: released %d unused leases back to pool", released)
        return released

    # --- internals --------------------------------------------------------

    def _acquire_payload(self) -> dict[str, Any] | None:
        session = self.session
        blocks = self.blocks_per_phase
        # Collects a prefetch that is still in flight rather than issuing a second,
        # competing acquire: two requests from one cluster race for the same blocks,
        # and at the end of an epoch only one of them can win. Bounded inside the
        # session by one RPC timeout, after which asking directly is no worse. See
        # LeaseSession.take_prefetched.
        result: Acquisition | None = session.take_prefetched()
        if result is not None and not result.granted:
            result = None                          # prefetch came back empty; ask properly
        if result is None:
            result = session.acquire(blocks)
        if not result.granted:
            return None
        return {
            "epoch": result.epoch,
            "spans": [[span.lease_id, span.start, span.end] for span in result.spans],
        }

    def _broadcast(self, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if self._broadcast_fn is not None:
            return self._broadcast_fn(payload, self.is_leader)
        if not dist.is_initialized() or self.world_size <= 1:
            return payload
        box = [payload]
        dist.broadcast_object_list(box, src=self.leader_global_rank, group=self.group)
        return box[0]

    def _build_phase(self, spans: list[tuple[str, int, int]]) -> Phase:
        """Expand spans to sample indices, shuffle locally, stride across ranks.

        The shuffle is the local half of the two-level scheme: it mixes samples
        drawn from a few gigabytes of already-staged text, which costs one pass
        over an index list, rather than mixing across the whole corpus, which would
        cost a seek per sample. Seeded by (seed, epoch, phase) so every rank of the
        stream produces the same permutation and the stride below is disjoint.
        """
        indices = list(self._carry)
        blocks = 0
        for _lease_id, start, end in spans:
            indices.extend(self.space.span_indices(start, end, self.epoch))
            blocks += end - start
        if self.shuffle:
            # String seed for the same reason as in space.py: reproducible across
            # processes, and `random.seed` rejects tuples.
            random.Random(f"pww-darl-phase:{self.seed}:{self.epoch}:"
                          f"{self.phase_index}").shuffle(indices)

        usable = len(indices) - len(indices) % self.world_size
        # The remainder rides along to the next phase instead of being dropped:
        # dropping it would leave a handful of samples per phase committed but
        # never trained, which over thousands of phases stops being a rounding
        # error. The final phase of an epoch drops fewer than world_size samples.
        self._carry = indices[usable:]
        mine = indices[self.rank:usable:self.world_size]
        return Phase(
            index=self.phase_index,
            epoch=self.epoch,
            spans=spans,
            indices=mine,
            blocks=blocks,
            samples_global=usable,
        )

    # --- checkpoint pairing -----------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """The client half of Psi = {theta, M_committed, M_unassigned}.

        The map M itself lives in the coordinator's snapshot -- it is global state
        and no single cluster owns it. What a cluster has to remember is only where
        it was in its own phase sequence, plus the carried remainder. That
        asymmetry is exactly why commits have to be checkpoint-gated: the model
        checkpoint and the coordinator snapshot are two files written by two
        processes, and `committed` is the only thing that makes them agree.
        """
        return {
            "epoch": self.epoch,
            "phase_index": self.phase_index,
            "samples_seen": self.samples_seen,
            "blocks_seen": self.blocks_seen,
            "carry": list(self._carry),
            "cluster": self.session.client.cluster_id if self.is_leader else "",
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epoch = int(state.get("epoch", 0))
        self.phase_index = int(state.get("phase_index", 0))
        self.samples_seen = int(state.get("samples_seen", 0))
        self.blocks_seen = int(state.get("blocks_seen", 0))
        self._carry = list(state.get("carry", []))

    # --- constructors -----------------------------------------------------

    @classmethod
    def for_diloco(
        cls,
        space: BlockSpace,
        session: LeaseSession | None,
        replicas: Any,
        *,
        seed: int = 0,
        blocks_per_phase: int | None = None,
        shuffle: bool = True,
    ) -> "DARLDataSource":
        """One stream per DiLoCo replica, sharded over that replica's ranks.

        `session` must be non-None exactly on each replica's rank 0 (see
        `session_for_replica`). The broadcast then runs inside the replica's inner
        group -- the same group DDP uses -- so it stays on the fast local links.
        """
        return cls(
            space,
            session,
            rank=replicas.rank_in_replica,
            world_size=replicas.ranks_per_replica,
            group=replicas.inner_group,
            leader_global_rank=replicas.replica_id * replicas.ranks_per_replica,
            seed=seed,
            blocks_per_phase=blocks_per_phase,
            shuffle=shuffle,
        )


def cluster_identity(site: str, replicas: Any = None, job_id: str | None = None) -> str:
    """Stable id for one data-consuming stream.

    Stable across a requeue on purpose: the same stream coming back after a
    walltime kill should be recognised as the same cluster so its measured
    throughput -- which is what sizes its grants -- is not thrown away. The Slurm
    job id is therefore *not* part of it by default.
    """
    parts = [site]
    if replicas is not None and getattr(replicas, "num_replicas", 1) > 1:
        parts.append(f"r{replicas.replica_id}")
    if job_id:
        parts.append(str(job_id))
    return "-".join(parts)


def session_for_replica(
    space: BlockSpace,
    replicas: Any,
    *,
    url: str,
    site: str,
    token: str = "",
    batch_size: int,
    inner_steps: int,
    grad_accum: int = 1,
    commit_policy: str = "checkpoint",
    use_proxy: bool = False,
    blocks_per_phase: int | None = None,
) -> LeaseSession | None:
    """Build the session on replica-local rank 0, None everywhere else.

    Also the place where lease granularity is decided: one phase of data per
    lease, from H, the batch and the replica's rank count. That is the design's
    recommendation and it has a property worth stating -- during the inner loop
    there is no coordinator traffic at all, and the only moment a lease boundary
    can be observed is the outer step, where the ranks are synchronised anyway.
    """
    from .client import LeaseClient

    if replicas is not None and replicas.rank_in_replica != 0:
        return None
    ranks = replicas.ranks_per_replica if replicas is not None else 1
    if blocks_per_phase is None:
        blocks_per_phase = blocks_for_phase(
            space, inner_steps=inner_steps, batch_size=batch_size, ranks=ranks,
            grad_accum=grad_accum,
        )
    client = LeaseClient(url, cluster_identity(site, replicas), token=token,
                         use_proxy=use_proxy)
    return LeaseSession(
        client, space, blocks_per_phase=blocks_per_phase, ranks=ranks,
        commit_policy=commit_policy,
    )


def _global_rank() -> int:
    import os

    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))
