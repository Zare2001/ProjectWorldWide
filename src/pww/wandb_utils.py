"""WandB conventions shared by every PWW run, and nothing that needs torchtitan.

Three separate processes log to the same WandB project and are meant to be read on one
chart:

    central-<site>        the single-node baseline (flower.enable = false)
    diloco-<site>         one participating cluster of the federated run
    central-aggregator    the FedMom server, i.e. the federated run as a whole

They only overlay correctly if they agree on what the axes mean, so the agreement lives
here rather than in each of them. Deliberately importable on the central VM, which has
`wandb` and `flwr` but no torchtitan -- `pww.titan.wandb_metrics` holds the half that
needs a Trainer.

The two axes
------------
``train/step``        global optimiser steps. The same quantity in all three runs: a
                      site's `darl.inner_steps` local steps are a *slice* of the global
                      count, never a separate clock. `FederatedTrainer.align_to_global_
                      step` and the server-authoritative `pww_global_step` are what make
                      that true, and the aggregator accumulates the same counter so its
                      points land on the same axis rather than on a reconstructed one.
``train/cum_tokens``  cumulative tokens trained on. The **default**, because equal steps
                      is not equal work: a two-site federation at 8 + 4 ranks trains
                      three times the tokens per step that a 4-rank baseline does, so a
                      per-step chart alone makes the federated run look better than it
                      is. Read the token axis for the honest comparison and the step
                      axis for the optimisation behaviour.

Note that `train/cum_tokens` is per-*run*: a site's own tokens on a `diloco-<site>` run,
the federation's total on `central-aggregator`. That is what each process actually
trained on, and it is why the aggregator -- not either site -- is the run to compare a
baseline against.
"""

from __future__ import annotations

from typing import Any

from .logging_utils import get_logger

logger = get_logger("pww.wandb")

STEP_METRIC = "train/step"
"""Global optimiser step, logged as a real metric and not only as wandb's index."""

DEFAULT_STEP_METRIC = "train/cum_tokens"
"""The x-axis every chart gets unless it is changed in the UI."""


def bind_axes(wandb_module: Any) -> None:
    """Declare both axes and make `train/cum_tokens` the default.

    Best effort: a wandb without `define_metric`, or an offline run, must not take a
    training job down over a chart default.
    """
    try:
        wandb_module.define_metric(STEP_METRIC)
        wandb_module.define_metric(DEFAULT_STEP_METRIC)
        wandb_module.define_metric("*", step_metric=DEFAULT_STEP_METRIC)
    except Exception as exc:                                          # noqa: BLE001
        logger.debug("could not bind the default wandb x-axis: %s", exc)


class MonotonicStep:
    """Turns a rewound step into one `wandb.log(step=...)` will accept.

    Returns the step unchanged while it does not go *backwards*, and `last + 1` when it
    does. The true value travels as `train/step`, so this only affects wandb's internal
    index -- which is what it is for.

    Needed because a cluster's realignment to the run's global step is authoritative in
    *both* directions. When a contribution is dropped -- non-finite weights, a delta
    rejected by the generation check, a round with no quorum -- the global step does not
    advance while that site's counter already did, so the next round pulls it back and it
    re-logs steps it has already logged. wandb **silently discards** a log call whose
    step is lower than the current one, so a repeated round would otherwise show up as a
    hole in every chart for that site rather than as the repeat it was.

    Repeats are passed through rather than bumped: wandb merges two log calls at the same
    step, and two calls at one step is the ordinary case, not a fault. The aggregator's
    training and held-out metrics for a round both belong at that round's step, and on a
    site a validation at a step that is also a `log_freq` multiple does too.
    """

    def __init__(self) -> None:
        self._last = -1
        self._warned = False

    def __call__(self, step: int) -> int:
        step = int(step)
        if step >= self._last:
            self._last = step
            return step
        if not self._warned:
            self._warned = True
            logger.warning(
                "wandb step went from %d to %d -- this run was realigned to a global "
                "step it had already passed, which happens when a round was not merged. "
                "wandb drops a log call whose step is not increasing, so the index is "
                "being advanced artificially instead; plot against %s or %s for the true "
                "position.",
                self._last, step, STEP_METRIC, DEFAULT_STEP_METRIC,
            )
        self._last += 1
        return self._last
