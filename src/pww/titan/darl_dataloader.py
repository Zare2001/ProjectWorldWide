"""A torchtitan dataloader whose data partition comes from DARL leases.

Slots into torchtitan through `TrainSpec.build_dataloader_fn`, so from the
Trainer's point of view this is an ordinary `BaseDataLoader` -- iterable,
`Stateful`, and checkpointed by the same DCP call as everything else. What differs
is where the indices come from: instead of `split_dataset_by_node` handing each
rank a fixed stripe of a stream, each *cluster* leases spans of a permuted window
index space from the DARL coordinator, and only then shards them across its own
data-parallel ranks.

That is the property the fixed-stripe approach cannot offer: clusters that join
late, run at different speeds, or die at walltime still cover the corpus exactly
once, because the partition is decided at run time by a coordinator that knows who
is still alive.

Rank topology
-------------
One process holds the `LeaseSession` -- global rank 0 -- and broadcasts each
acquired span over the default process group. Every rank then derives its own
indices locally from the shared `BlockSpace`, so nothing but three integers per
span crosses the network and a 512-rank job makes the same number of RPCs as a
4-rank one.

The broadcast is over WORLD rather than the dp mesh on purpose. Under tensor or
pipeline parallelism, ranks that share a ``dp_rank`` must see *identical* data;
broadcasting globally and then striding by ``dp_rank`` gives exactly that, and
degenerates to the obvious thing when dp is the only parallelism.

Batches never straddle a step boundary unevenly
-----------------------------------------------
`__iter__` is one continuous generator across phases, not one iteration per phase.
So a batch may draw from two leases and there are no partial batches at phase
boundaries -- which matters because a partial batch would leave data-parallel ranks
disagreeing about how many steps a phase contains, and a DiLoCo outer step is a
collective: ranks that disagree hang in the all-reduce instead of failing.

Epoch exhaustion is a stop, not a no-op
---------------------------------------
When the coordinator says the epoch is complete and no further epoch is
configured, this raises `StopIteration`, which torchtitan turns into
`DataloaderExhaustedError` and a clean end of training. It deliberately does *not*
keep returning empty work: `train_llm_flower.py` did that and a run silently
continued for 23 outer rounds reporting loss 0.0 while the global model sat
frozen.

Anything still held at that point is released rather than left to expire, because a
slower site may be waiting on precisely those blocks to finish its own epoch. The
harder version of that problem -- a lease the *prefetcher* won that no phase will
ever ask for -- is handled in `darl.client.LeaseSession`, not here: a dataloader-side
release cannot fix it, because by the time the coordinator reports the epoch
complete there is by definition nothing uncommitted left to give back.
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchtitan.components.dataloader import BaseDataLoader

from ..darl.client import CommitPolicy, DarlError, LeaseClient, LeaseSession
from ..darl.space import BlockSpace, blocks_for_phase
from ..darl.torch_data import DARLDataSource, cluster_identity
from ..logging_utils import get_logger
from .shards import ShardedTokenCorpus, read_manifest, verify_compatible

logger = get_logger("pww.titan.darl_dataloader")


class DARLWindowDataset(IterableDataset, Stateful):
    """Yields ``({"input": x[:-1]}, x[1:])`` for DARL-leased windows.

    The tuple shape is torchtitan's own (see
    `torchtitan.hf_datasets.text_datasets.HuggingFaceTextDataset`), so the stock
    `forward_backward_step` consumes it unchanged.
    """

    def __init__(
        self,
        corpus: ShardedTokenCorpus,
        source: DARLDataSource,
        *,
        max_epochs: int = 1,
        commit_policy: str = CommitPolicy.CHECKPOINT,
    ) -> None:
        if commit_policy not in CommitPolicy.ALL:
            raise ValueError(f"commit_policy must be one of {CommitPolicy.ALL}")
        self.corpus = corpus
        self.source = source
        self.commit_policy = commit_policy
        self.max_epochs = max(1, int(max_epochs))
        # Counters for the Flower client's honest reporting: a round that trained
        # nothing has to be distinguishable from a round that trained badly.
        self.windows_yielded = 0
        self.phases_started = 0
        self.exhausted = False

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        while True:
            phase = self.source.next_phase()

            if phase is None:
                if not self._advance_epoch():
                    self.exhausted = True
                    # Anything still held is of no use to this cluster now, and a
                    # slower site may be waiting on precisely those blocks to finish
                    # its own epoch. Returning them takes milliseconds; letting them
                    # expire takes a full TTL.
                    self.source.release_unused()
                    logger.info(
                        "darl: corpus exhausted after %s windows in %d phase(s); "
                        "ending iteration",
                        f"{self.windows_yielded:,}", self.phases_started,
                    )
                    return
                continue

            self.phases_started += 1
            for index in phase.indices:
                tokens = torch.from_numpy(self.corpus.window_tokens(index))
                self.windows_yielded += 1
                yield {"input": tokens[:-1]}, tokens[1:]

            # Records the phase duration so the lease TTL tracks how slow this
            # cluster actually is, instead of it being declared dead mid-phase.
            self.source.end_phase()

            if self.commit_policy == CommitPolicy.CONSUMPTION:
                # Recycles the span now rather than waiting for a checkpoint. Under
                # the 'checkpoint' policy this is the trainer's job instead -- see
                # `commit` and titan/trainer.py -- and the span stays held until a
                # checkpoint is actually on disk.
                self.source.commit()

    def _advance_epoch(self) -> bool:
        """True if another epoch was opened and iteration should continue.

        Only the leader talks to the coordinator; followers learn the outcome from
        the next `next_phase` broadcast, so this must not itself be collective.
        """
        next_epoch = self.source.epoch + 1
        if next_epoch >= self.max_epochs:
            return False

        if self.source.is_leader:
            logger.info("darl: epoch %d complete, advancing", self.source.epoch)
            try:
                self.source.session.client.advance_epoch()
            except DarlError as exc:
                logger.error("darl: advance_epoch failed (%s); stopping", exc)
                return False

        self.source.epoch_complete = False
        self.source.epoch = next_epoch
        self.source.phase_index = 0
        self.source._carry = []
        # The coordinator recycles blocks asynchronously; a brief pause avoids an
        # immediate acquire racing the epoch flip and coming back drained.
        time.sleep(1.0)
        return True

    def commit(self) -> int:
        """Tell the coordinator the held spans are durably in a checkpoint.

        Checkpoint-gated on purpose (`darl.client.CommitPolicy.CHECKPOINT`): the
        model state and the coordinator's committed map then fail together, so a
        crash loses the same work from both sides and the epoch stays exactly-once.
        """
        return self.source.commit()

    def state_dict(self) -> dict[str, Any]:
        """Where this cluster is in its phase sequence.

        Note what is *not* here: the offset within the current phase. On a restart
        the leases this cluster held have expired and their uncommitted blocks are
        back in the pool for anyone to take, so replaying a phase prefix from a
        saved offset would double-train blocks that another cluster may already
        have. The correct resume is to acquire afresh, which is what omitting the
        offset produces.
        """
        return {
            "darl": self.source.state_dict(),
            "windows_yielded": self.windows_yielded,
            "phases_started": self.phases_started,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        self.source.load_state_dict(state_dict.get("darl", {}))
        self.windows_yielded = int(state_dict.get("windows_yielded", 0))
        self.phases_started = int(state_dict.get("phases_started", 0))


class DARLDataLoader(torch.utils.data.DataLoader, BaseDataLoader):
    """`BaseDataLoader`-shaped wrapper: batches windows, forwards DARL state.

    Inherits `BaseDataLoader` so the type `CheckpointManager` annotates its
    `dataloader` argument with is actually satisfied, the same way torchtitan's own
    `ParallelAwareDataloader` does it. Nothing enforces it at runtime -- DCP only
    needs `state_dict`/`load_state_dict` -- but an untruthful annotation is how the
    next person ends up debugging a silently unsaved dataloader.

    Not `ParallelAwareDataloader`: that one keys its state by ``dp_rank`` because
    each rank owns an independent stream position. Here the stream position is
    cluster-level state owned by the leader's `LeaseSession`, and every rank
    reconstructs its own share from the broadcast span, so keying per rank would
    save the same thing many times and restore it inconsistently after a
    world-size change.
    """

    def __init__(self, dataset: DARLWindowDataset, batch_size: int, dp_rank: int) -> None:
        # num_workers stays 0: `DARLWindowDataset.__iter__` runs a collective
        # (the span broadcast), which in a forked worker would either deadlock or
        # hand different ranks different data.
        super().__init__(dataset, batch_size=batch_size, num_workers=0, drop_last=True)
        self.darl_dataset = dataset
        self.dp_rank = dp_rank

    def state_dict(self) -> dict[str, Any]:
        return self.darl_dataset.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.darl_dataset.load_state_dict(state_dict)

    def commit(self) -> int:
        return self.darl_dataset.commit()

    @property
    def exhausted(self) -> bool:
        return self.darl_dataset.exhausted


def build_darl_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Any,
    job_config: Any,
    infinite: bool = True,
) -> DARLDataLoader:
    """`TrainSpec.build_dataloader_fn` for DARL-leased pre-tokenised shards.

    Reads its federation settings from the ``[darl]`` config section that
    `pww.titan.config` adds to torchtitan's JobConfig.
    """
    darl_cfg = job_config.darl
    training = job_config.training

    corpus_dir = training.dataset_path
    if not corpus_dir:
        raise ValueError(
            "training.dataset_path must point at a pww token shard directory when "
            "training.dataset is 'pww_tokens' -- build one with "
            "`python3 -m pww.titan.tokenize_corpus`"
        )

    manifest = read_manifest(corpus_dir)
    verify_compatible(
        manifest, seq_len=training.seq_len, assets_path=job_config.model.hf_assets_path
    )
    corpus = ShardedTokenCorpus(corpus_dir, manifest)

    space = BlockSpace(
        num_samples=len(corpus),
        block_size=darl_cfg.block_size,
        seed=darl_cfg.space_seed,
        shuffle=darl_cfg.shuffle,
    )

    global_rank = dist.get_rank() if dist.is_initialized() else 0
    is_leader = global_rank == 0

    # Mirrors how torchtitan's own Trainer derives it: global_batch_size < 0 means
    # "one accumulation step", otherwise it is how many local batches make up a
    # global one.
    local_global = max(1, training.local_batch_size * dp_world_size)
    grad_accum = (
        max(1, training.global_batch_size // local_global)
        if training.global_batch_size > 0
        else 1
    )

    # One lease per inner phase, sized from H and the batch geometry rather than a
    # constant. `train_llm_flower.py` hardcoded 5 blocks per phase and committed
    # whole phases after consuming a fraction of them, which is how ~90% of the
    # WikiText run's blocks got marked trained without ever being read.
    blocks = darl_cfg.blocks_per_phase or blocks_for_phase(
        space,
        inner_steps=darl_cfg.inner_steps,
        batch_size=training.local_batch_size,
        ranks=dp_world_size,
        grad_accum=grad_accum,
    )

    session: LeaseSession | None = None
    if is_leader:
        cluster = darl_cfg.cluster_id or cluster_identity(darl_cfg.site or "pww")
        client = LeaseClient(
            darl_cfg.url,
            cluster,
            token=darl_cfg.token,
            use_proxy=darl_cfg.use_proxy,
        )
        logger.info(
            "darl: corpus %s | %s | space digest %s",
            corpus_dir, manifest.describe(), space.digest(0),
        )
        session = LeaseSession(
            client,
            space,
            blocks_per_phase=blocks,
            ranks=dp_world_size,
            commit_policy=darl_cfg.commit_policy,
        )

    source = DARLDataSource(
        space,
        session,
        rank=dp_rank,
        world_size=dp_world_size,
        leader_global_rank=0,
        seed=darl_cfg.space_seed,
        blocks_per_phase=blocks,
        shuffle=darl_cfg.shuffle,
    )

    dataset = DARLWindowDataset(
        corpus,
        source,
        max_epochs=darl_cfg.epochs if infinite else 1,
        commit_policy=darl_cfg.commit_policy,
    )
    return DARLDataLoader(dataset, training.local_batch_size, dp_rank)
