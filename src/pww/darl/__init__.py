"""DARL -- Dynamic Atomic Range Leasing for multi-HPC training.

The problem: m independent HPC clusters train one model, and the dataset must be
covered exactly once between them -- no sample twice, no sample missed -- while
each cluster starts when its batch queue lets it, runs at its own speed, and can
vanish at walltime without warning.

Static partitioning cannot do this. Split the corpus m ways in advance and the
fast cluster finishes its shard and idles, the slow one never drains its own, and
a cluster that dies takes its whole shard out of the epoch. Assigning work
dynamically without a protocol is worse: two clusters race and train on the same
samples, which is invisible in the loss curve.

So the dataset index space is leased, the way a distributed database leases key
ranges:

    space.py       the immutable index space: N samples, blocks of K, one global
                   block permutation derived from the seed
    table.py       the state machine -- UNASSIGNED / LEASED / COMMITTED, TTLs,
                   expiry, work stealing, and the invariant check
    server.py      one coordinator process: HTTP, a lock, a write-ahead log
    client.py      the cluster side: RPCs, heartbeats, prefetch, revocations
    torch_data.py  the trainer side: spans -> per-rank sample lists
    simulate.py    m simulated clusters against a real coordinator, verifying
                   coverage end to end

Quick start (two terminals, no allocation needed):

    python3 -m pww.darl.server --num-samples 1000000 --block-size 1000 --port 8760
    python3 -m pww.darl.simulate --url http://127.0.0.1:8760 --clusters 4

Everything except `torch_data` is stdlib-only, so the coordinator runs on a login
node with no environment to set up.

The guarantees, and their exact scope: the union of committed blocks is the whole
space, no two clusters ever hold the same block, and the sum of per-cluster
committed blocks equals M. "Processed" means committed, and a client should commit
only what a durable checkpoint covers -- see `table.CommitPolicy` for what the
cheaper alternative costs you.
"""

from __future__ import annotations

from .client import (
    Acquisition,
    CommitPolicy,
    DarlError,
    LeaseClient,
    LeaseGone,
    LeaseSession,
    Span,
)
from .space import BlockSpace, blocks_for_phase
from .table import (
    TTL_ALPHA,
    TTL_BETA,
    BlockState,
    ClusterRecord,
    Grant,
    Lease,
    LeaseTable,
    compute_ttl,
)

__all__ = [
    "Acquisition",
    "BlockSpace",
    "BlockState",
    "ClusterRecord",
    "CommitPolicy",
    "DarlError",
    "Grant",
    "Lease",
    "LeaseClient",
    "LeaseGone",
    "LeaseSession",
    "LeaseTable",
    "Span",
    "TTL_ALPHA",
    "TTL_BETA",
    "blocks_for_phase",
    "compute_ttl",
]


def __getattr__(name: str):
    """Expose the torch-dependent pieces lazily.

    `server.py` and the CLI must import on a login node with nothing but CPython,
    so torch cannot be imported at package level.
    """
    if name in ("DARLDataSource", "LeasedSampler", "Phase", "cluster_identity",
                "session_for_replica"):
        from . import torch_data

        return getattr(torch_data, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
