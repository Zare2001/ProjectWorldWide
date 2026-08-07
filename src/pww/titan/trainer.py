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
        processor = trainer.metrics_processor
        original_log_validation = processor.log_validation

        def capturing_log_validation(loss, step, *args, **kwargs):
            self._validation_loss = float(loss)
            return original_log_validation(loss, step, *args, **kwargs)

        processor.log_validation = capturing_log_validation

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
        tokens = trainer.ntokens_seen - tokens_before
        avg_loss = self._loss_sum / self._loss_count if self._loss_count else 0.0

        self.rounds_done += 1
        return {
            "steps": steps_done,
            "tokens": tokens,
            "loss": avg_loss,
            "exhausted": exhausted,
            "blocks_committed": committed,
            "seconds": elapsed,
            "tokens_per_s": tokens / elapsed,
        }

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

    def validate(self) -> float:
        """Held-out loss, or nan when validation is disabled."""
        if not self.job_config.validation.enable:
            return float("nan")
        trainer = self.trainer
        self._validation_loss = float("nan")
        with trainer.loss_fn.no_rescale():
            trainer.validator.validate(trainer.model_parts, trainer.step)
        return self._validation_loss
