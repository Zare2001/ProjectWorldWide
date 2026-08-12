"""Entrypoint for torchtitan + DARL + Flower runs.

Launched under torchrun, exactly like stock torchtitan:

    torchrun --nproc_per_node=$RANKS -m pww.titan.train \
        --job.config-file configs/titan/qwen3_diloco.toml \
        --darl.url http://<central>:29510 --darl.site lumi \
        --flower.enable --flower.server_address <central>:29511

Everything torchtitan understands is understood here -- this only parses the
config through torchtitan's own `ConfigManager` and then chooses between two
drivers:

    flower.enable = false   torchtitan's own `train()`, unchanged. DARL still
                            leases the data, so this validates a site end to end
                            without involving the WAN or the central node.
    flower.enable = true    `FederatedTrainer` driven one inner phase per Flower
                            round, with FedMom doing the outer step centrally.

Collective ordering
-------------------
The three setup steps below run on **every** rank, in this order, before the ranks
diverge into client and workers:

    build trainer  ->  federated.start()  ->  gather initial full state

`start()` loads the checkpoint and `gather_full_state` all-gathers, and both are
collectives -- doing either on rank 0 only (which is where it would naturally
belong, being the only rank that talks gRPC) hangs the job. This is also why
`DiLoCoFlowerClient` takes the gathered state as an argument instead of fetching it
in its own constructor.
"""

from __future__ import annotations

import os
import sys

import torch.distributed as dist

from ..logging_utils import get_logger, setup_logging

logger = get_logger("pww.titan.train")


def _parse_config(argv: list[str]):
    """torchtitan's own config pipeline, so every upstream flag keeps working."""
    from torchtitan.config.manager import ConfigManager

    return ConfigManager().parse_args(argv)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"configuration error: {message}")


def main(argv: list[str] | None = None) -> int:
    # torchtitan logs through the ROOT logger, and the only thing that gives that
    # logger a level and a handler is init_logger() -- which upstream calls from its
    # own `if __name__ == "__main__"` block in torchtitan/train.py. We are a different
    # entrypoint, so that block never runs, root stays at its default WARNING with no
    # handler, and every logger.info() is discarded. That includes the per-step
    # `step: N loss: ... tps: ... tflops: ... mfu: ...` line from MetricsProcessor,
    # so a healthy run looks like a silent one: the job logs nothing but warnings.
    #
    # Called before anything else in main() so a config error is logged too.
    from torchtitan.tools.logging import init_logger

    init_logger()
    # Same treatment for the `pww` logger, which propagate=False keeps separate from
    # torchtitan's root handler. RANK rather than dist.get_rank(): the process group
    # does not exist yet here, and torchrun always sets it.
    setup_logging(int(os.environ.get("RANK", "0")))

    argv = list(sys.argv[1:] if argv is None else argv)
    job_config = _parse_config(argv)

    darl_cfg = getattr(job_config, "darl", None)
    flower_cfg = getattr(job_config, "flower", None)
    _require(
        darl_cfg is not None and flower_cfg is not None,
        "no [darl]/[flower] sections -- set job.custom_config_module = "
        '"pww.titan.config" in the TOML',
    )

    uses_darl = job_config.training.dataset == "pww_tokens"
    if uses_darl:
        _require(
            bool(darl_cfg.url),
            "training.dataset is 'pww_tokens' so --darl.url must point at the "
            "coordinator, e.g. http://145.38.206.143:29510",
        )
        if not darl_cfg.token:
            darl_cfg.token = os.environ.get("DARL_TOKEN", "")

    # Imported after the config is parsed so a config error surfaces before the
    # (slow) torch/torchtitan import chain and the distributed init.
    from torchtitan.train import Trainer

    from .trainer import FederatedTrainer

    trainer = Trainer(job_config)

    if not flower_cfg.enable:
        logger.info("flower.enable is false -- running torchtitan's own train loop")

        # Hook the metrics processor so central runs log under the same
        # WandB keys (train/loss, train/perplexity, train/cum_tokens) as
        # DiLoCo runs, enabling direct chart comparison.
        import math

        proc = getattr(trainer, "metrics_processor", None)
        if proc is not None:
            orig_log = proc.log
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            def _central_log_wrapper(
                step, global_avg_loss, global_max_loss, grad_norm,
                extra_metrics=None,
            ):
                if extra_metrics is None:
                    extra_metrics = {}
                cum_tok = int(getattr(trainer, "ntokens_seen", 0) * world_size)
                if cum_tok > 0:
                    extra_metrics["train/cum_tokens"] = cum_tok
                if math.isfinite(global_avg_loss):
                    extra_metrics["train/loss"] = float(global_avg_loss)
                    extra_metrics["train/perplexity"] = float(
                        math.exp(min(20.0, global_avg_loss))
                    )
                return orig_log(
                    step, global_avg_loss, global_max_loss, grad_norm,
                    extra_metrics=extra_metrics,
                )

            proc.log = _central_log_wrapper

        try:
            trainer.train()
        finally:
            trainer.close()
            if dist.is_initialized():
                dist.destroy_process_group()
        return 0

    _require(
        bool(flower_cfg.server_address),
        "--flower.enable needs --flower.server_address host:port",
    )

    federated = FederatedTrainer(trainer, inner_steps=darl_cfg.inner_steps)
    rank = dist.get_rank() if dist.is_initialized() else 0

    try:
        # A collective; every rank, before anyone diverges.
        federated.start()

        from .flower_client import DiLoCoFlowerClient, run_worker_loop
        from .params import gather_full_state

        # Only the inline transport needs this, and it is a collective, so every rank
        # takes the same branch. Under blob transport it is skipped entirely: an
        # all-gather of the full model onto every rank is 14 GiB per rank at 7B and
        # 140 GiB at 70B, which is the cost blob transport exists to avoid paying.
        if flower_cfg.transport == "inline":
            initial_state = gather_full_state(trainer.model_parts)
        else:
            initial_state = None
            logger.info(
                "transport=%s: skipping the initial full-model gather",
                flower_cfg.transport,
            )

        if rank == 0:
            import flwr as fl

            # The try covers construction as well as the connection. Client
            # construction can legitimately fail -- `_check_transport_ceiling`
            # refuses a model too large for a gRPC message -- and every other rank
            # is already blocked in `run_worker_loop`'s broadcast by this point, so
            # an exception that escapes without a STOP leaves them waiting until
            # walltime instead of reporting the error.
            client = None
            try:
                client = DiLoCoFlowerClient(federated, initial_state)
                logger.info(
                    "connecting to Flower server at %s (H=%d inner steps per round)",
                    flower_cfg.server_address, darl_cfg.inner_steps,
                )
                fl.client.start_client(
                    server_address=flower_cfg.server_address,
                    client=client.to_client(),
                    grpc_max_message_length=flower_cfg.max_message_length,
                )
            finally:
                if client is not None:
                    client.stop_workers()
                else:
                    from .flower_client import broadcast_stop

                    broadcast_stop()
        else:
            run_worker_loop(federated)
    finally:
        federated.close()
        if dist.is_initialized():
            dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
