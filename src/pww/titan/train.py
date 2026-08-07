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

from ..logging_utils import get_logger

logger = get_logger("pww.titan.train")


def _parse_config(argv: list[str]):
    """torchtitan's own config pipeline, so every upstream flag keeps working."""
    from torchtitan.config.manager import ConfigManager

    return ConfigManager().parse_args(argv)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"configuration error: {message}")


def main(argv: list[str] | None = None) -> int:
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
