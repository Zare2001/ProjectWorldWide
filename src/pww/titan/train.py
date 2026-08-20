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
from ..wandb_utils import bind_axes
from .wandb_metrics import install_metric_hooks

logger = get_logger("pww.titan.train")


def _parse_config(argv: list[str]):
    """torchtitan's own config pipeline, so every upstream flag keeps working."""
    from torchtitan.config.manager import ConfigManager

    return ConfigManager().parse_args(argv)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"configuration error: {message}")


def _commit_darl_on_checkpoint(trainer) -> None:
    """Commit DARL leases after a checkpoint, on the plain torchtitan path.

    Under the default `checkpoint` commit policy the dataloader deliberately does not
    commit anything itself -- the commit is gated on the work being durably in a
    checkpoint, so that {weights, committed blocks} fail together. `FederatedTrainer`
    does that gating in `run_inner_phase`, and torchtitan's own `train()` has no
    equivalent, so the central baseline committed **nothing for the whole run**.

    What that cost: the coordinator's committed map stayed empty, so its coverage
    counter and `epoch_complete` signal never advanced, every lease the run ever took
    stayed outstanding and heartbeated, and a baseline asked to cover a full epoch would
    have looped on `drain` instead of finishing. Silent throughout -- the run trains
    correctly, it simply never tells the coordinator what it consumed, so the one figure
    that says "the baseline saw the same data as the DiLoCo run" was unavailable.

    Wrapping `save` rather than polling: the commit has to land *after* the write and
    only when there actually was one. `CheckpointManager.save` returns None and no-ops
    between intervals, so the decision is read from `_should_save` first -- exactly as
    `FederatedTrainer._will_checkpoint` does, and losing that private method degrades
    the same safe way, to committing every call.
    """
    dataloader = getattr(trainer, "dataloader", None)
    checkpointer = getattr(trainer, "checkpointer", None)
    commit = getattr(dataloader, "commit", None)
    if commit is None or checkpointer is None:
        return

    original_save = checkpointer.save

    def _save_then_commit(curr_step: int, last_step: bool = False, **kwargs):
        should_save = getattr(checkpointer, "_should_save", None)
        if should_save is None:
            logger.warning(
                "CheckpointManager has no _should_save; committing DARL leases on every "
                "save call, which is the 'consumption' policy in practice"
            )
            wrote = True
        else:
            try:
                wrote = bool(should_save(curr_step, last_step))
            except TypeError:
                wrote = bool(should_save(curr_step))
        result = original_save(curr_step, last_step=last_step, **kwargs)
        if wrote:
            try:
                blocks = commit()
                if blocks:
                    logger.info("darl: committed %d block(s) through step %d",
                                blocks, curr_step)
            except Exception as exc:                                  # noqa: BLE001
                # A refused commit must not kill a run whose checkpoint is already on
                # disk; the spans expire on their TTL instead.
                logger.warning("darl commit failed: %s", exc)
        return result

    checkpointer.save = _save_then_commit


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

    # WandB bookkeeping that applies to both drivers. `Trainer.__init__` builds the
    # metrics processor, which is what calls `wandb.init`, so `wandb.run` exists by here
    # on whichever rank logs.
    #
    # Gated on `wandb.run` rather than on rank 0, because the logging rank is not always
    # rank 0: under a pipeline-parallel schedule torchtitan puts it on the first rank of
    # the last stage (`metrics._get_metrics_rank`). Checking the run object asks the
    # question that actually matters and needs no process group.
    try:
        import wandb

        if wandb.run is not None:
            # Binds the default x-axis to train/cum_tokens, the same as the central
            # aggregator does, so a chart built across runs is not comparing one run's
            # tokens against another's steps.
            bind_axes(wandb)
            slurm_job_id = os.environ.get("SLURM_JOBID", "")
            if slurm_job_id:
                wandb.config.update({"slurm_job_id": slurm_job_id},
                                    allow_val_change=True)
    except Exception:                                                 # noqa: BLE001
        pass

    if not flower_cfg.enable:
        logger.info("flower.enable is false -- running torchtitan's own train loop")

        # The same hooks the DiLoCo driver installs, from the same module. A baseline
        # is only a baseline if `train/loss`, `train/cum_tokens` and `eval/loss` mean
        # the same thing here as they do there -- and until this shared call existed
        # they did not: this path multiplied a per-rank token count by the world size
        # (wrong under tensor parallelism) and published no held-out series at all, so
        # the perplexity chart had a DiLoCo line and no baseline to compare it to.
        install_metric_hooks(trainer)

        if uses_darl:
            _commit_darl_on_checkpoint(trainer)

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

    # The control-plane process group: the tiny command tuples rank 0 broadcasts to the
    # workers, on GLOO rather than NCCL. The workers enter that broadcast the moment
    # their phase ends and legitimately wait out the whole round barrier -- the slow
    # site's remaining phase plus the merge -- and a NCCL collective cannot wait like
    # that: the default group's CUDA watchdog runs at comm.init_timeout_seconds (300s),
    # and a fast site idling longer at the barrier is SIGABRTed by its own watchdog.
    # Observed killing Snellius at round 26 of the first 20k run, after 24 rounds that
    # each survived by seconds. (Commit 8d9a8e1 raised the timeout for this exact
    # symptom, but set_pg_timeouts touches the mesh groups and the command broadcast
    # rode the default group.) gloo has no watchdog and carries CPU tuples without
    # touching a GPU; its own timeout still bounds a genuinely dead leader, at 4x the
    # round timeout so no legitimate barrier wait can reach it.
    #
    # Created on every rank, unconditionally before the rank-0/worker split:
    # new_group is itself a collective.
    control_pg = None
    if dist.is_initialized() and dist.get_world_size() > 1:
        from datetime import timedelta

        wait_budget = 4 * int(getattr(job_config.comm, "train_timeout_seconds", 1800) or 1800)
        control_pg = dist.new_group(
            backend="gloo", timeout=timedelta(seconds=wait_budget)
        )
        logger.info(
            "control-plane broadcasts on gloo (timeout %ds); NCCL never waits at the "
            "round barrier", wait_budget,
        )

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
                client = DiLoCoFlowerClient(federated, initial_state,
                                            control_group=control_pg)
                if getattr(flower_cfg, "protocol", "grpc") == "http":
                    from ..http_round_client import run_http_client

                    logger.info(
                        "connecting to the HTTP round endpoint at %s "
                        "(H=%d inner steps per round)",
                        flower_cfg.server_address, darl_cfg.inner_steps,
                    )
                    run_http_client(
                        client,
                        url=f"http://{flower_cfg.server_address}",
                        cluster_id=client.cluster_id,
                        ranks=dist.get_world_size() if dist.is_initialized() else 1,
                        token=darl_cfg.token,
                        # A site that needs the proxy for DARL needs it here too: both
                        # are plain HTTP to the same host.
                        use_proxy=darl_cfg.use_proxy,
                    )
                else:
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

                    broadcast_stop(control_pg)
        else:
            run_worker_loop(federated, control_group=control_pg)
    finally:
        federated.close()
        if dist.is_initialized():
            dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
