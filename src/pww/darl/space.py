"""The immutable global index space, and the two-level shuffle.

A dataset of N pre-tokenised samples is a plain ordered index space
D = {0, ..., N-1}. DARL never leases samples; it leases *blocks* of K
consecutive samples, because per-sample coordination would make the lease
coordinator the bottleneck (see the granularity note in `table.py`).

Two levels of shuffling, for two different reasons:

  global  the M blocks are permuted once, deterministically from the seed. A
          lease covers a contiguous run of *positions* in that permutation, so
          consecutive leases are statistically independent samples of the corpus
          rather than consecutive text.

  local   within a leased span the client shuffles samples in memory
          (`torch_data.LeasedSampler`). The span is a few GB, so this is a
          random read inside a region already staged on NVMe or in page cache,
          not a random read over the whole corpus.

The alternative -- one global random permutation of samples -- gives marginally
better mixing and destroys sequential I/O: every sample becomes an independent
seek into a multi-terabyte file, which on Lustre costs more than the forward
pass. Block-level global shuffling plus sample-level local shuffling keeps reads
large and contiguous while leaving no correlation between what a cluster reads
in consecutive phases.

The permutation is derived from the seed alone so that every site computes it
without communicating. That is only safe if every site computes the *same*
permutation, so `BlockSpace.digest()` exists and the coordinator refuses to
register a client whose digest disagrees -- a Python or library version skew
between two sites would otherwise silently hand two clusters the same samples
under different position numbers, which is exactly the failure DARL is meant to
make impossible.
"""

from __future__ import annotations

import hashlib
import random
from array import array
from dataclasses import dataclass
from functools import lru_cache

# 4 bytes per block, so a 10M-block space costs 40 MB. Two epochs cached at a
# time: the current one and (during an epoch boundary) the next.
_PERMUTATION_CACHE = 2


@lru_cache(maxsize=_PERMUTATION_CACHE)
def _permutation(num_blocks: int, seed: int, epoch: int) -> array:
    """Fisher-Yates over block ids, seeded by (seed, epoch).

    `random.Random` rather than numpy on purpose: it needs no dependency and its
    shuffle is stable across CPython versions, which matters because two
    different machines must derive bit-identical permutations. Both sites also
    verify agreement via `BlockSpace.digest`, so a future change here fails loudly
    at registration instead of quietly duplicating data.
    """
    order = array("i", range(num_blocks))
    # Seeded from a string, not a tuple: `random.seed` rejects tuples on 3.11+,
    # and str seeding goes through sha512 rather than hash(), so it is stable
    # across processes and machines -- which it has to be, because two sites derive
    # this independently.
    random.Random(f"pww-darl:{seed}:{epoch}").shuffle(order)
    return order


@dataclass(frozen=True)
class BlockSpace:
    """Immutable description of the global dataset index space.

    num_samples   N -- total tokenised sequences in the corpus
    block_size    K -- samples per block; the unit of leasing
    seed          derives the global block permutation
    shuffle       False gives the identity permutation, i.e. positions are block
                  ids. Useful when debugging: a lease then maps to a readable
                  sample range, at the cost of every cluster reading contiguous
                  text.
    """

    num_samples: int
    block_size: int
    seed: int = 0
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")

    @property
    def num_blocks(self) -> int:
        """M. The last block is short when K does not divide N."""
        return -(-self.num_samples // self.block_size)

    # --- positions and physical blocks ------------------------------------

    def physical_block(self, position: int, epoch: int = 0) -> int:
        """Which block of the corpus is consumed at `position` in this epoch."""
        if not 0 <= position < self.num_blocks:
            raise IndexError(f"position {position} outside [0, {self.num_blocks})")
        if not self.shuffle:
            return position
        return int(_permutation(self.num_blocks, self.seed, epoch)[position])

    def block_samples(self, position: int, epoch: int = 0) -> range:
        """The half-open sample range read at `position`."""
        block = self.physical_block(position, epoch)
        start = block * self.block_size
        return range(start, min(start + self.block_size, self.num_samples))

    def span_sample_ranges(self, start: int, end: int, epoch: int = 0) -> list[range]:
        """One range per position in the span [start, end).

        Contiguous positions map to scattered physical blocks, so this is a list
        of ranges rather than one range -- each of which is a single large
        sequential read.
        """
        if end < start:
            raise ValueError(f"span [{start}, {end}) is inverted")
        return [self.block_samples(p, epoch) for p in range(start, end)]

    def span_sample_count(self, start: int, end: int, epoch: int = 0) -> int:
        """Samples in a span, accounting for a possibly short final block."""
        return sum(len(r) for r in self.span_sample_ranges(start, end, epoch))

    def span_indices(self, start: int, end: int, epoch: int = 0) -> list[int]:
        """Flat list of every sample index in the span, in physical order.

        Materialised because the caller shuffles it anyway. At the recommended
        block size this is millions of ints per phase, which is a few tens of MB
        of host memory -- acceptable, and the reason `block_size` should not be
        set to 1.
        """
        out: list[int] = []
        for r in self.span_sample_ranges(start, end, epoch):
            out.extend(r)
        return out

    # --- agreement between sites ------------------------------------------

    def digest(self, epoch: int = 0) -> str:
        """Fingerprint of the space *and* its permutation for this epoch.

        Clients send this at registration and the coordinator compares it against
        its own. A mismatch means the two sides disagree about what position p
        refers to, so every disjointness guarantee below is void.
        """
        h = hashlib.blake2b(digest_size=16)
        h.update(f"{self.num_samples}:{self.block_size}:{self.seed}:{self.shuffle}:{epoch}".encode())
        if self.shuffle:
            h.update(_permutation(self.num_blocks, self.seed, epoch).tobytes())
        return h.hexdigest()

    def describe(self) -> str:
        return (
            f"{self.num_samples:,} samples / {self.block_size:,} per block = "
            f"{self.num_blocks:,} blocks"
            f"{'' if self.shuffle else ' (unshuffled)'}"
        )


def blocks_for_phase(
    space: BlockSpace,
    *,
    inner_steps: int,
    batch_size: int,
    ranks: int,
    grad_accum: int = 1,
) -> int:
    """Blocks that cover exactly one DiLoCo local phase.

    This is the lease granularity the design argues for: a span sized to one
    inner loop means zero coordinator traffic while the GPUs are busy, and a
    lease boundary that lands exactly where the ranks are already synchronised
    for the outer step. Rounded up, so a phase never runs dry mid-loop; the
    remainder is carried by the sampler into the next phase rather than dropped.
    """
    if min(inner_steps, batch_size, ranks, grad_accum) < 1:
        raise ValueError("inner_steps, batch_size, ranks and grad_accum must all be >= 1")
    samples = inner_steps * batch_size * ranks * grad_accum
    return max(1, -(-samples // space.block_size))
