"""torchtitan's Trainer, driven one DiLoCo inner phase at a time.

`FederatedTrainer` wraps `torchtitan.train.Trainer` without reimplementing it:
`train_step`, `forward_backward_step`, FSDP2 wrapping, the LR schedule, gradient
accumulation, activation checkpointing and DCP checkpointing are all torchtitan's,
unmodified. It adds exactly three things:

1. `run_inner_phase()` -- H optimiser steps, then stop, so the caller can do an
   outer step. torchtitan's own `train()` runs straight to `training.steps`.
2. loss and token accounting across a phase, because `train_step` logs metrics and
   returns nothing, and `Validator.validate` reports through the metrics processor
   rather than returning a value.
3. entry of the profiling contexts once for the whole run rather than once per
   phase, which would restart the profiler schedule every round.

Deliberately free of any Flower dependency, so a single-site run
(`flower.enable = false`) needs neither flwr nor a central node. The federated
half lives in `flower_client.py`.

Checkpoint before commit
------------------------
`run_inner_phase` writes the checkpoint and only then commits the DARL leases it
covers. That order is the exactly-once guarantee: {weights, committed blocks} then
fail together, so a crash between them loses the same work from both sides.
Committing first would mark blocks durable that the next restart has no weights
for, silently dropping them from the epoch.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any

import torch

from ..logging_utils import get_logger

logger = get_logger("pww.titan.trainer")

class FederatedTrainer:
    """Wraps `torchtitan.train.Trainer` with phase-at-a-time execution.

    Composition rather than inheritance for the outer machinery, but the trainer
    instance itself is a real torchtitan Trainer built by
    `build_federated_trainer`, so every component behaves exactly as it does in a
    stock run.
    """

    def __init__(self, trainer: Any, inner_steps: int) -> None:
        self.trainer = trainer
        self.inner_steps = max(1, int(inner_steps))
        self.job_config = trainer.job_config
        self._stack = contextlib.ExitStack()
        self._data_iterator = None
        self._started = False
        self.rounds_done = 0

        # Patched in so loss survives train_step, which logs and returns None.
        self._loss_sum = 0.0
        self._loss_count = 0
        original_fbs = trainer.forward_backward_step

        def counting_forward_backward(input_dict, labels):
            loss = original_fbs(input_dict, labels)
            # float() forces a device sync once per microbatch. torchtitan's own
            # metrics path already syncs at log_freq; this is the price of having a
            # per-round loss to report to the aggregator, and it is amortised over
            # a full forward+backward.
            self._loss_sum += float(loss.detach())
            self._loss_count += 1
            return loss

        trainer.forward_backward_step = counting_forward_backward

        # `Validator.validate` returns None and reports through the metrics
        # processor, so the only way to get the held-out loss back out is to
        # intercept where it is logged.
        self._validation_loss = float("nan")
        self._validation_tokens = 0
        self._validation_baseline = 0
        processor = trainer.metrics_processor
        original_log_validation = processor.log_validation

        def capturing_log_validation(loss, step, *args, **kwargs):
            self._validation_loss = float(loss)
            # Read here because `log_validation` zeroes the counter on its way out, and
            # relative to a baseline taken in validate() because training also
            # accumulates into it between its own log intervals.
            self._validation_tokens = max(
                0, processor.ntokens_since_last_log - self._validation_baseline
            )
            return original_log_validation(loss, step, *args, **kwargs)

        processor.log_validation = capturing_log_validation

    # --- cluster-level numbers --------------------------------------------

    def _cluster_total(self, value: float) -> float:
        """Sum a per-rank quantity across this cluster's data-parallel ranks.

        torchtitan keeps `ntokens_seen` and the training loss **per rank** --
        `self.ntokens_seen += labels.numel()` on the local batch -- and globalises them
        only inside its own logging path. That is right there and wrong here, because
        these numbers leave the process:

        * `num_examples` is the FedMom merge weight. A per-rank count silently collapses
          token weighting back into uniform 1/k averaging whenever two sites share a
          per-rank geometry: LUMI's 8 GCDs and Snellius's 4 H100s report the *same*
          rank-0 token count at the same local_batch_size and seq_len, while LUMI
          actually trained twice the tokens. The deliberate departure from DiLoCo's
          uniform average would then have been undone by an accounting bug.
        * a reported loss that is one rank's rather than the cluster's is then
          token-weighted across clusters as though it were a cluster-level mean.
        * throughput understates by the data-parallel degree, so tokens/s looks 8x worse
          on LUMI than it is.

        Collective, so every rank must reach it the same number of times. Both callers
        do -- rank 0 through the Flower client, the rest through
        `flower_client.run_worker_loop`.

        float64 rather than float32 on purpose: a round at 8 ranks x 8 batch x 2048
        seq_len x 100 steps is ~13M tokens, already near float32's 2**24 exact-integer
        ceiling.
        """
        parallel_dims = getattr(self.trainer, "parallel_dims", None)
        if parallel_dims is None or not getattr(parallel_dims, "dp_cp_enabled", False):
            return float(value)

        from torchtitan.distributed import utils as dist_utils

        tensor = torch.tensor(float(value), dtype=torch.float64,
                              device=self.trainer.device)
        return float(dist_utils.dist_sum(
            tensor,
            parallel_dims.world_mesh["dp_cp"],
            getattr(getattr(self.trainer, "ft_manager", None), "loss_sync_pg", None),
        ))

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        """Load any existing checkpoint and open the long-lived contexts."""
        if self._started:
            return
        from torchtitan.tools.profiling import (
            maybe_enable_memory_snapshot,
            maybe_enable_profiling,
        )

        trainer = self.trainer
        trainer.checkpointer.load(step=self.job_config.checkpoint.load_step)

        # Entered once for the whole federated run, not once per round: these
        # write trace files keyed by step, and re-entering them every round would
        # restart the profiler schedule on each one.
        self._stack.enter_context(
            maybe_enable_profiling(
                self.job_config.profiling,
                global_step=trainer.step,
                base_folder=self.job_config.job.dump_folder,
            )
        )
        self._stack.enter_context(
            maybe_enable_memory_snapshot(
                self.job_config.profiling,
                global_step=trainer.step,
                base_folder=self.job_config.job.dump_folder,
            )
        )
        self._data_iterator = trainer.batch_generator(trainer.dataloader)
        self._started = True
        logger.info("federated trainer ready at step %d", trainer.step)

    def close(self) -> None:
        self._stack.close()
        self.trainer.close()

    # --- one inner phase ---------------------------------------------------

    def run_inner_phase(self) -> dict[str, Any]:
        """H optimiser steps. Returns what the round actually accomplished."""
        from torchtitan.components.dataloader import DataloaderExhaustedError

        self.start()
        trainer = self.trainer
        self._loss_sum = 0.0
        self._loss_count = 0

        tokens_before = trainer.ntokens_seen
        steps_done = 0
        exhausted = False
        started = time.monotonic()

        for _ in range(self.inner_steps):
            if trainer.step >= self.job_config.training.steps:
                logger.info(
                    "reached training.steps=%d; no further inner steps",
                    self.job_config.training.steps,
                )
                exhausted = True
                break
            trainer.step += 1
            trainer.gc_handler.run(trainer.step)
            try:
                trainer.train_step(self._data_iterator)
            except DataloaderExhaustedError:
                # DARL has no more blocks for this cluster and no further epoch is
                # configured. The step was not executed, so undo its number.
                trainer.step -= 1
                exhausted = True
                logger.info("dataloader exhausted at step %d", trainer.step)
                break
            steps_done += 1

        # Checkpoint first, then commit the leases it covers: {weights, committed
        # blocks} then fail together, so a crash loses the same work from both
        # sides and the epoch stays exactly-once. Committing first would silently
        # drop those blocks from the epoch on a crash in between.
        committed = 0
        if steps_done:
            last_step = trainer.step >= self.job_config.training.steps
            # Asked before the call, because `save` returns None and no-ops when the
            # interval has not been reached -- committing regardless would mark
            # blocks durable that no checkpoint covers, which is the 'consumption'
            # policy wearing the 'checkpoint' policy's name.
            wrote = self._will_checkpoint(trainer.step, last_step)
            trainer.checkpointer.save(trainer.step, last_step=last_step)
            if wrote or self._commit_every_phase:
                committed = self._commit_leases()

        elapsed = max(1e-6, time.monotonic() - started)
        # Cluster-level, not rank-level -- see _cluster_total for why that distinction
        # is load-bearing rather than cosmetic. Three tiny all-reduces once per H steps.
        tokens = int(self._cluster_total(trainer.ntokens_seen - tokens_before))
        loss_sum = self._cluster_total(self._loss_sum)
        loss_count = self._cluster_total(self._loss_count)
        # Sum of per-rank loss sums over sum of per-rank microbatch counts. Equal to the
        # token-weighted mean because every microbatch carries the same token count here:
        # fixed seq_len, fixed local_batch_size, and drop_last=True in the dataloader, so
        # there are no partial batches to skew it.
        avg_loss = loss_sum / loss_count if loss_count else 0.0

        self.rounds_done += 1
        return {
            "steps": steps_done,
            "tokens": tokens,
            "loss": avg_loss,
            "exhausted": exhausted,
            "blocks_committed": committed,
            "seconds": elapsed,
            "tokens_per_s": tokens / elapsed,
            **self._hardware_metrics(tokens, elapsed),
        }

    def _hardware_metrics(self, tokens: int, elapsed: float) -> dict[str, float]:
        """MFU, peak memory and learning rate, read off torchtitan's own state.

        These are the parts of a Megatron-style metrics set that mean something here.
        Deliberately not reported, because this configuration does not have them:

        * pipeline-bubble time -- no pipeline parallelism, so there is no bubble
        * P2P / NVLink bandwidth and all-reduce duration -- the intra-site collective is
          FSDP2's, and the only link this project adds is the WAN hop, whose cost shows up
          as the server's merge duration rather than as a fabric metric
        * scaling efficiency -- meaningful against a single-GPU baseline on identical
          hardware; across two facilities with different accelerators it would compare
          MI250X GCDs to H100s and report a number about the hardware, not the code

        MFU is per *rank* on purpose. It answers "is this GPU being used well", which is a
        local question; averaging it across two facilities with different peak FLOPs would
        produce a figure describing neither.
        """
        out: dict[str, float] = {}
        proc = getattr(self.trainer, "metrics_processor", None)
        if proc is None:
            return out

        flops_per_token = getattr(proc, "num_flops_per_token", -1)
        peak_flops = getattr(proc, "gpu_peak_flops", 0)
        ranks = self._dp_degree()
        if flops_per_token > 0 and peak_flops > 0 and ranks > 0 and elapsed > 0:
            # tokens is cluster-level, so divide back down: MFU is a per-device ratio.
            tps_per_rank = (tokens / ranks) / elapsed
            out["mfu_pct"] = 100.0 * flops_per_token * tps_per_rank / peak_flops
            out["tflops_per_rank"] = flops_per_token * tps_per_rank / 1e12

        monitor = getattr(proc, "device_memory_monitor", None)
        if monitor is not None:
            try:
                stats = monitor.get_peak_stats()
                # torchtitan returns a namedtuple; reserved is the number that predicts
                # an OOM, active is what the model actually holds.
                out["peak_memory_gib"] = float(getattr(stats, "max_reserved_gib", 0.0))
                out["peak_memory_pct"] = float(getattr(stats, "max_reserved_pct", 0.0))
            except Exception:                                         # noqa: BLE001
                pass

        schedulers = getattr(self.trainer, "lr_schedulers", None)
        try:
            groups = schedulers.schedulers[0].optimizer.param_groups   # type: ignore[union-attr]
            out["lr"] = float(groups[0]["lr"])
        except Exception:                                             # noqa: BLE001
            pass
        return out

    def _dp_degree(self) -> int:
        """Data-parallel ranks in this cluster, or 1 if not sharded."""
        parallel_dims = getattr(self.trainer, "parallel_dims", None)
        if parallel_dims is None or not getattr(parallel_dims, "dp_cp_enabled", False):
            return 1
        try:
            return int(parallel_dims.world_mesh["dp_cp"].size())
        except Exception:                                             # noqa: BLE001
            return 1

    @property
    def _commit_every_phase(self) -> bool:
        """True under the 'consumption' commit policy.

        The dataloader already commits at phase end in that mode; doing it here too
        is harmless -- a commit through a watermark already committed is a no-op --
        and covers the case where a phase boundary and a round boundary do not line
        up exactly.
        """
        policy = getattr(getattr(self.job_config, "darl", None), "commit_policy", "")
        return policy == "consumption"

    def _will_checkpoint(self, step: int, last_step: bool) -> bool:
        """Whether `checkpointer.save(step, last_step)` is about to write.

        `CheckpointManager.save` returns None, so there is no return value to test,
        and its interval logic lives in `_should_save`. Reaching for that private
        method is deliberate: the alternative is duplicating the interval arithmetic
        here, which would drift from upstream silently. If it ever disappears this
        falls back to committing every round, which is the safe direction -- spans
        recycle sooner than they strictly should rather than never.
        """
        should_save = getattr(self.trainer.checkpointer, "_should_save", None)
        if should_save is None:
            logger.warning(
                "CheckpointManager has no _should_save; committing DARL leases every "
                "round, which is the 'consumption' policy in practice"
            )
            return True
        try:
            return bool(should_save(step, last_step))
        except TypeError:
            return bool(should_save(step))

    def _commit_leases(self) -> int:
        commit = getattr(self.trainer.dataloader, "commit", None)
        if commit is None:
            return 0
        try:
            return commit()
        except Exception as exc:            # a refused commit must not kill the run
            logger.warning("darl commit failed: %s", exc)
            return 0

    def validate(self) -> tuple[float, int]:
        """(held-out loss, tokens it was measured over). Loss is nan when disabled.

        The loss needs no reduction here -- torchtitan's Validator already all-reduces it
        over the dp mesh, so it is a cluster-level mean. The token count does, and it
        matters because it is the weight the central node aggregates by: reporting a
        constant (this returned `1` to Flower) makes the cross-site held-out loss an
        unweighted mean over clusters, so a site that evaluated a hundred thousand tokens
        counts exactly as much as one that evaluated a hundred.
        """
        if not self.job_config.validation.enable:
            return float("nan"), 0
        trainer = self.trainer
        self._validation_loss = float("nan")
        self._validation_tokens = 0
        self._validation_baseline = trainer.metrics_processor.ntokens_since_last_log
        with trainer.loss_fn.no_rescale():
            trainer.validator.validate(trainer.model_parts, trainer.step)
        return self._validation_loss, int(self._cluster_total(self._validation_tokens))
