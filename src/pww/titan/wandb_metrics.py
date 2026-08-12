"""One definition of the metrics a torchtitan run reports, used by both drivers.

The axis conventions and the monotonic-step guard live in `pww.wandb_utils`, which the
central aggregator imports too -- it has wandb but no torchtitan, so anything importable
from this package (whose `__init__` registers train specs) is out of reach there.

A central baseline and a DiLoCo run are only comparable if the numbers being compared
were computed the same way. That sounds obvious and was not the case: the two drivers
in `train.py` each grew their own metrics hook, and they disagreed on three things that
all matter.

* **cumulative tokens.** The DiLoCo path all-reduced `ntokens_seen` over the dp mesh;
  the central path multiplied rank 0's count by `dist.get_world_size()`. Those agree
  only under pure data parallelism -- with `tensor_parallel_degree > 1` the ranks of a
  TP group see the *same* tokens, so multiplying overcounts by the TP degree, and the
  baseline's x-axis silently stretches. Both now read torchtitan's own
  `n_tokens_seen`, which `Trainer.train_step` already dist_sums over `dp_cp` and puts
  in `extra_metrics` -- exact, and one fewer collective than the DiLoCo path was doing.
* **held-out loss.** The DiLoCo path published `eval/loss` and `eval/perplexity`; the
  central path published only torchtitan's `validation_metrics/loss`. So the one chart
  that answers "did federating cost anything" had no baseline line on it at all.
* **the step axis.** Every run logs at its *global* optimiser step, so a central run's
  step 1500 and a DiLoCo run's step 1500 are both 1500 optimiser steps in, with the
  DiLoCo one having crossed 15 outer rounds to get there. `darl.inner_steps` is
  deliberately not an axis anywhere. What that needs is a guard against wandb dropping
  a rewound step, which is `pww.wandb_utils.MonotonicStep`.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from ..logging_utils import get_logger
from ..wandb_utils import STEP_METRIC, MonotonicStep

logger = get_logger("pww.titan.wandb_metrics")


def cluster_sum(trainer: Any, value: float) -> float:
    """Sum a per-rank quantity across this cluster's data-parallel ranks.

    torchtitan keeps `ntokens_seen` and the training loss **per rank** and globalises
    them only inside its own logging path. That is right there and wrong wherever the
    number leaves the process, which is why this exists in a shared module: the same
    reduction backs the FedMom merge weight, the reported loss and the token axis, and
    three copies of it would be three chances to differ.

    Collective, so every rank must reach it the same number of times.

    float64 rather than float32 on purpose: a round at 8 ranks x 8 batch x 2048 seq_len
    x 100 steps is ~13M tokens, already near float32's 2**24 exact-integer ceiling.
    """
    parallel_dims = getattr(trainer, "parallel_dims", None)
    if parallel_dims is None or not getattr(parallel_dims, "dp_cp_enabled", False):
        return float(value)

    import torch
    from torchtitan.distributed import utils as dist_utils

    tensor = torch.tensor(float(value), dtype=torch.float64, device=trainer.device)
    return float(dist_utils.dist_sum(
        tensor,
        parallel_dims.world_mesh["dp_cp"],
        getattr(getattr(trainer, "ft_manager", None), "loss_sync_pg", None),
    ))


def install_metric_hooks(
    trainer: Any,
    *,
    cluster_total: Callable[[float], float] | None = None,
    read_power_watts: Callable[[], float | None] | None = None,
) -> None:
    """Publish the shared `train/*` and `eval/*` keys on `trainer.metrics_processor`.

    Idempotent, because the DiLoCo driver calls it from `FederatedTrainer.start()`,
    which is itself called from more than one place.

    `cluster_total` and `read_power_watts` are injected rather than imported so this
    module needs nothing from `trainer.py` and can be used by the plain torchtitan
    path, which builds no `FederatedTrainer` at all.
    """
    proc = getattr(trainer, "metrics_processor", None)
    if proc is None or getattr(proc, "_pww_hooked", False):
        return

    monotonic = MonotonicStep()

    # Wrapped rather than replaced: the container fans out to wandb *and* tensorboard,
    # and tensorboard has no monotonicity requirement, so rewriting the step for both
    # is the lesser evil against dropping the row for both.
    backend = getattr(proc, "logger", None)
    if backend is not None:
        original_backend_log = backend.log

        def _monotonic_log(metrics: dict[str, Any], step: int) -> None:
            return original_backend_log(metrics, monotonic(step))

        backend.log = _monotonic_log

    original_log = proc.log

    def _log(step, global_avg_loss, global_max_loss, grad_norm, extra_metrics=None):
        extra = dict(extra_metrics or {})

        # Read by `FederatedTrainer._hardware_metrics` for the per-round report to the
        # aggregator, which has no other source for it.
        proc.last_grad_norm = grad_norm

        extra[STEP_METRIC] = int(step)

        # torchtitan's own figure: dist_summed over the dp_cp mesh in `train_step`, so
        # already cluster-level and exact. The fallback covers a torchtitan that stops
        # providing it; multiplying a per-rank count by the world size does not, which
        # is what the central path used to do.
        cum_tokens = extra.get("n_tokens_seen")
        if cum_tokens is None:
            reduce_fn = cluster_total or (lambda v: cluster_sum(trainer, v))
            cum_tokens = reduce_fn(getattr(trainer, "ntokens_seen", 0))
        if cum_tokens is not None and int(cum_tokens) > 0:
            extra["train/cum_tokens"] = int(cum_tokens)

        if math.isfinite(global_avg_loss):
            extra["train/loss"] = float(global_avg_loss)
            extra["train/perplexity"] = float(math.exp(min(20.0, global_avg_loss)))
        # Namespaced copies of two numbers torchtitan already logs, so that a chart
        # comparing a site against the aggregator is comparing `train/lr` to
        # `train/lr` rather than to a differently-named series.
        if "lr" in extra:
            extra["train/lr"] = float(extra["lr"])
        if math.isfinite(grad_norm):
            extra["train/grad_norm"] = float(grad_norm)

        if read_power_watts is not None:
            watts = read_power_watts()
            if watts is not None:
                extra["power_watts"] = watts

        return original_log(step, global_avg_loss, global_max_loss, grad_norm,
                            extra_metrics=extra)

    proc.log = _log

    # `Validator.validate` returns None and reports through the processor, so the only
    # place to see the held-out loss is where it is logged.
    original_log_validation = proc.log_validation

    def _log_validation(loss, step, *args, **kwargs):
        value = float(loss)
        if math.isfinite(value):
            backend_logger = getattr(proc, "logger", None)
            if backend_logger is not None:
                ppl = math.exp(min(20.0, value))
                try:
                    backend_logger.log({
                        "eval/loss": value,
                        "eval/perplexity": ppl,
                        "validation_metrics/perplexity": ppl,
                        STEP_METRIC: int(step),
                    }, step)
                except Exception:                                      # noqa: BLE001
                    pass
        return original_log_validation(loss, step, *args, **kwargs)

    proc.log_validation = _log_validation
    proc._pww_hooked = True
