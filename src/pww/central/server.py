"""Central Flower aggregator: FedMom with elastic membership over queued HPC sites.

    python3 -m pww.central.server --config configs/central_aggregator_titan.yaml

Two transports, chosen with `--transport`:

    inline  weights ride inside the Flower gRPC message. One moving part, and hard
            capped by gRPC's 2 GiB per-message limit -- about 1B parameters in
            float16, and no setting raises it.
    blob    the message carries a blob name; the bytes move over HTTP to
            `pww.central.blobstore`, and the merge is streamed one tensor at a time.
            Required above ~1B, and needs `--state-dir`, `--blob-root` and `--blob-url`.

Elastic membership is the default rather than an option: `--min-clients 1` means the
server starts, holds the run, and merges whatever is actually present. With durable
state (`--state-dir`) it also survives its own restart, so "every site is queued" and
"the aggregator was rebooted" are both ordinary states.

Two round numbers, deliberately
-------------------------------
Flower's `server_round` counts *attempts*; it advances even for a round in which every
cluster was killed at walltime and nothing changed. The **merge round** counts
successful merges, is what deltas are validated against, and is what survives in the
state directory. `--num-rounds` bounds the former, so set it generously -- the run
ends when DARL runs out of tokens, not when Flower runs out of round numbers.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from ..config import apply_config_file
from ..logging_utils import get_logger, setup_logging
from .. import fedproto as proto

logger = get_logger("pww.central.server")

# Held-out loss spread across clusters that triggers a warning, in nats.
#
# Deliberately loose. Clusters evaluate the same global model, so on a decent held-out
# split they should agree to well under a tenth of a nat -- but a small
# validation.steps over a re-looped fixture makes the per-site samples genuinely
# different, and then a nat of spread is data, not a bug. 1.0 catches the case that
# preceded a real failure (2.05) without firing on that noise.
EVAL_SPREAD_WARN_NATS = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PWW Central Node Flower Aggregator Server (FedMom)"
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on")
    parser.add_argument("--flower-port", type=int, default=None, help="Alias for --port")
    parser.add_argument("--darl-port", type=int, default=29510, help="DARL coordinator port")
    parser.add_argument(
        "--num-samples", type=int, default=1000000, help="DARL total sample count"
    )
    parser.add_argument("--block-size", type=int, default=10000, help="DARL block size")
    parser.add_argument(
        "--darl-state-dir", type=str, default="./runs/darl", help="DARL state directory"
    )
    parser.add_argument(
        "--num-rounds", type=int, default=50,
        help="Upper bound on Flower rounds (attempts, not merges). Set generously.",
    )
    parser.add_argument(
        "--min-clients", type=int, default=1,
        help="Clusters required before a round starts. 1 = elastic: train with "
             "whoever is out of the Slurm queue. 2 forces both sites and will block "
             "while one is queued.",
    )
    parser.add_argument(
        "--server-learning-rate", type=float, default=1.0,
        help="Outer learning rate (eta), i.e. DiLoCo's OuterOpt lr. The DiLoCo paper "
             "uses 0.7 with momentum 0.9, which is what configs/central_aggregator_"
             "titan.yaml and src/pww/diloco.py both set; the default here stays 1.0 so "
             "that leaving momentum at 0.0 gives exact FedAvg.",
    )
    parser.add_argument(
        "--server-momentum", type=float, default=0.9,
        help="Outer momentum (beta). Together with eta this is Nesterov momentum on "
             "DiLoCo's outer gradient -- algebraically identical to "
             "torch.optim.SGD(momentum=beta, nesterov=True). 0.0 reduces the outer step "
             "to plain FedAvg averaging: correct for BatchNorm models, wrong for an LLM.",
    )
    parser.add_argument(
        "--round-timeout", type=float, default=1800.0,
        help="Seconds to wait for a round's results. One round is H inner steps plus "
             "a weight exchange, so minutes, not seconds.",
    )

    g = parser.add_argument_group("wandb logging")
    g.add_argument(
        "--enable-wandb", "--wandb", action="store_true", default=False,
        help="Enable Weights & Biases logging for the central aggregator",
    )
    g.add_argument(
        "--wandb-project", type=str, default=None,
        help="WandB project name (defaults to WANDB_PROJECT env or 'pww-diloco')",
    )
    g.add_argument(
        "--wandb-entity", type=str, default=None,
        help="WandB team or entity name",
    )
    g.add_argument(
        "--wandb-run-name", type=str, default=None,
        help="WandB run name (defaults to WANDB_RUN_NAME env or 'central-aggregator')",
    )

    g = parser.add_argument_group("transport")
    g.add_argument(
        "--transport", type=str, default=proto.TRANSPORT_INLINE, choices=proto.TRANSPORTS,
        help="inline: weights in the gRPC message (<=~1B params). blob: out of band.",
    )
    g.add_argument(
        "--state-dir", type=str, default=None,
        help="Durable global model, momentum buffer and membership. Required for "
             "blob transport; also what lets the server restart without losing the run.",
    )
    g.add_argument(
        "--blob-root", type=str, default=None,
        help="Blob store directory, as seen from this machine. Put it on the same "
             "filesystem as --state-dir so publishing a global model is a hard link.",
    )
    g.add_argument(
        "--blob-url", type=str, default="",
        help="Blob store base URL as the CLUSTERS see it, e.g. http://<vm>:29512",
    )
    g.add_argument(
        "--storage-dtype", type=str, default="float32", choices=["float32", "bfloat16"],
        help="Dtype for the durable global model and momentum buffer. float32 unless "
             "disk is the binding constraint -- momentum accumulates over hundreds of "
             "rounds.",
    )
    g.add_argument(
        "--run-id", type=str, default="pww",
        help="Names this run's blobs so two runs can share a blob store",
    )
    g.add_argument(
        "--keep-rounds", type=int, default=1,
        help="Recent global blobs to retain, so a cluster mid-download when the round "
             "advances can still finish it",
    )

    g = parser.add_argument_group("outer step")
    g.add_argument(
        "--no-solo-full-step", dest="solo_full_step", action="store_false",
        help="Keep applying server-learning-rate on rounds with a single contributor. "
             "The default applies eta=1 there, because with nothing to average eta<1 only "
             "discards local progress; disable it if you would rather bound how far one "
             "cluster can move the global model",
    )
    parser.set_defaults(solo_full_step=True)

    g = parser.add_argument_group("global-model checkpoints")
    g.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Where to snapshot the merged global model and momentum buffer. Defaults "
             "to <state-dir>/checkpoints. Without a state dir there is nowhere to put "
             "them and checkpointing is off",
    )
    g.add_argument(
        "--keep-ephemeral", type=int, default=2,
        help="Per-merge snapshots retained (crash recovery). 0 disables them",
    )
    g.add_argument(
        "--persist-every", type=int, default=5,
        help="Write a retained snapshot every N merges (rollback history). 0 disables",
    )
    g.add_argument(
        "--fresh-model", action="store_true",
        help="Ignore existing checkpoints and re-seed from the first cluster to connect. "
             "The counterpart of DARL_FRESH: needed when the architecture changes, since "
             "a checkpoint's tensor shapes must match the model the sites build",
    )
    g.add_argument(
        "--keep-persistent", type=int, default=4,
        help="Retained snapshots kept, newest first. 0 means unbounded -- at 0.6B that "
             "is ~4.8 GiB each, so 200 rounds at --persist-every 5 is ~192 GiB",
    )
    return parser


def build_metric_aggregators(
    enable_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    config_dict: dict | None = None,
):
    """The two metric aggregation callbacks, as a pair."""
    state = {
        "total_tokens": 0,
        "site_tokens": {},
        "merge_round": 0,
    }

    wandb_run = None
    if enable_wandb:
        try:
            import os
            import wandb
            project = wandb_project or os.getenv("WANDB_PROJECT", "pww-diloco")
            entity = wandb_entity or os.getenv("WANDB_ENTITY", None)
            name = wandb_run_name or os.getenv("WANDB_RUN_NAME", "central-aggregator")
            wandb_run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                config=config_dict or {},
            )
            logger.info("WandB logging enabled for central aggregator (%s/%s)", project, name)
        except Exception as exc:
            logger.warning("Failed to initialize WandB for central aggregator: %s", exc)
            wandb_run = None

    def aggregate_fit_metrics(metrics: list[tuple[int, dict]]) -> dict:
        """Token-weighted training loss, plus how far replicas drifted."""
        total = sum(n for n, _ in metrics)
        if total == 0:
            logger.info("  >> no tokens trained this round")
            return {}
        state["total_tokens"] += total
        state["merge_round"] += 1
        cum_tokens = state["total_tokens"]

        for n, m in metrics:
            cid = str(m.get(proto.CLUSTER, "?"))
            state["site_tokens"][cid] = state["site_tokens"].get(cid, 0) + n

        avg_loss = sum(n * m.get(proto.LOSS, 0.0) for n, m in metrics) / total
        out = {"loss": avg_loss, "cum_tokens": cum_tokens}
        drifts = [float(m["drift_ratio"]) for _, m in metrics if "drift_ratio" in m]
        detail = ""
        if drifts:
            out["drift_ratio"] = sum(drifts) / len(drifts)
            out["drift_ratio_max"] = max(drifts)
            detail = f", drift {out['drift_ratio']:.4f} (max {max(drifts):.4f})"
        names = ", ".join(str(m.get(proto.CLUSTER, "?")) for _, m in metrics)
        out["perplexity"] = math.exp(min(20.0, avg_loss)) if math.isfinite(avg_loss) else float("nan")
        logger.info(
            "  >> Training loss %.4f (ppl %.2f)  (%d cluster(s) [%s], %s tokens this round | %s total tokens%s)",
            avg_loss, out["perplexity"], len(metrics), names, f"{total:,}", f"{cum_tokens:,}", detail,
        )

        rate = sum(float(m["tokens_per_s"]) for _, m in metrics if "tokens_per_s" in m)
        if rate > 0:
            out["tokens_per_s"] = rate
            slowest = max((float(m.get("seconds", 0.0)) for _, m in metrics), default=0.0)
            per_site = "; ".join(
                f"{m.get(proto.CLUSTER, '?')} {float(m.get('tokens_per_s', 0)):,.0f} tok/s"
                + (f" in {float(m['seconds']):.0f}s" if "seconds" in m else "")
                + f" ({n:,} tok round | {state['site_tokens'].get(str(m.get(proto.CLUSTER, '?')), 0):,} tok total)"
                + (f", {float(m['mfu_pct']):.1f}% MFU" if "mfu_pct" in m else "")
                + (f", {float(m['tflops_per_rank']):.1f} TFLOP/s/rank"
                   if "tflops_per_rank" in m else "")
                + (f", {float(m['peak_memory_gib']):.1f} GiB"
                   f" ({float(m.get('peak_memory_pct', 0)):.0f}%)"
                   if "peak_memory_gib" in m else "")
                + (f", {float(m['power_watts']):.0f}W" if "power_watts" in m else "")
                + (f", grad_norm {float(m['grad_norm']):.4f}" if "grad_norm" in m else "")
                for n, m in metrics
            )
            logger.info("  >> Throughput %s tok/s combined | %s", f"{rate:,.0f}", per_site)
            if slowest > 0:
                out["round_seconds"] = slowest
                logger.info("  >> Round took %.0fs (slowest site's inner phase)", slowest)

        lrs = [float(m["lr"]) for _, m in metrics if "lr" in m]
        if lrs:
            out["lr"] = lrs[0]
            if max(lrs) - min(lrs) > 1e-12:
                logger.warning("  >> learning rate differs across clusters: %s -- the "
                               "schedule is over global steps and should agree",
                               ", ".join(f"{v:.3e}" for v in lrs))

        if wandb_run is not None:
            wb_metrics = {
                "round": state["merge_round"],
                "train/loss": avg_loss,
                "train/perplexity": out["perplexity"],
                "train/cum_tokens": cum_tokens,
                "train/tokens_this_round": total,
                "train/global_batch_tokens": total,
                "train/global_batch_samples": total // 2048,
            }
            if "tokens_per_s" in out:
                wb_metrics["throughput/tokens_per_s_combined"] = out["tokens_per_s"]
            if "round_seconds" in out:
                wb_metrics["train/round_seconds"] = out["round_seconds"]
            if "drift_ratio" in out:
                wb_metrics["train/drift_ratio_avg"] = out["drift_ratio"]
                wb_metrics["train/drift_ratio_max"] = out["drift_ratio_max"]
            if "lr" in out:
                wb_metrics["train/lr"] = out["lr"]
            for n, m in metrics:
                cid = str(m.get(proto.CLUSTER, "?"))
                wb_metrics[f"cluster/{cid}/batch_tokens"] = n
                wb_metrics[f"cluster/{cid}/batch_samples"] = n // 2048
                for k in ("tokens_per_s", "mfu_pct", "tflops_per_rank", "peak_memory_gib", "peak_memory_pct", "power_watts", "grad_norm", "drift_ratio"):
                    if k in m:
                        wb_metrics[f"cluster/{cid}/{k}"] = float(m[k])
            try:
                wandb_run.log(wb_metrics)
            except Exception as exc:
                logger.warning("WandB log fit failed: %s", exc)

        return out

    def aggregate_eval_metrics(metrics: list[tuple[int, dict]]) -> dict:
        """Pooled held-out perplexity, and accuracy for the CIFAR path."""
        total = sum(n for n, _ in metrics)
        if total == 0:
            return {}

        reported = [(n, float(m["eval_loss"]), str(m.get(proto.CLUSTER, "?")))
                    for n, m in metrics if "eval_loss" in m]

        losses = [(n, v, c) for n, v, c in reported if math.isfinite(v)]
        dropped = [(c, v) for n, v, c in reported if not math.isfinite(v)]
        if dropped:
            logger.error(
                "  >> excluded %d non-finite held-out loss(es) from the pooled figure: "
                "%s. A site reporting this is not producing a usable model; its training "
                "loss is where to look.",
                len(dropped), ", ".join(f"{c}={v}" for c, v in dropped),
            )

        res = {}
        if losses:
            weight = sum(n for n, _, _ in losses)
            pooled_loss = sum(n * v for n, v, _ in losses) / weight
            pooled_ppl = math.exp(min(20.0, pooled_loss))
            per_cluster = ", ".join(
                f"{math.exp(min(20.0, v)):.2f}" for _, v, _ in losses
            )
            logger.info(
                "  >> Perplexity %.2f  (held-out loss %.4f; per-cluster ppl [%s], "
                "%s eval tokens)",
                pooled_ppl, pooled_loss, per_cluster, f"{weight:,}",
            )

            if len(losses) > 1:
                worst = max(losses, key=lambda item: item[1])
                best = min(losses, key=lambda item: item[1])
                spread = worst[1] - best[1]
                if spread > EVAL_SPREAD_WARN_NATS:
                    logger.warning(
                        "  >> held-out loss spread %.2f nats across clusters (%s=%.2f vs "
                        "%s=%.2f) on the same global model. Either %s is not applying the "
                        "weights it was sent, or the clusters are not scoring the same "
                        "data -- check validation.dataset/steps and the eval token counts "
                        "above before reading anything into the model.",
                        spread, worst[2], worst[1], best[2], best[1], worst[2],
                    )
            res = {"eval_loss": pooled_loss, "perplexity": pooled_ppl}
        elif reported:
            logger.error("  >> every cluster reported a non-finite held-out loss; "
                         "no perplexity for this round")
            res = {}
        else:
            values = [(n, float(m["accuracy"])) for n, m in metrics if "accuracy" in m]
            if values:
                weight = sum(n for n, _ in values)
                pooled = sum(n * v for n, v in values) / weight
                per_cluster = ", ".join(f"{v:.2f}%" for _, v in values)
                logger.info(
                    "  >> Test accuracy %.2f%%  (per-cluster: [%s], %s samples)",
                    pooled, per_cluster, f"{weight:,}",
                )
                res = {"accuracy": pooled}

        if wandb_run is not None and res:
            wb_eval = {}
            if "perplexity" in res:
                wb_eval["eval/loss"] = res["eval_loss"]
                wb_eval["eval/perplexity"] = res["perplexity"]
                for n, v, c in losses:
                    wb_eval[f"eval_cluster/{c}/perplexity"] = math.exp(min(20.0, v))
            elif "accuracy" in res:
                wb_eval["eval/accuracy"] = res["accuracy"]
            try:
                wandb_run.log(wb_eval)
            except Exception as exc:
                logger.warning("WandB log eval failed: %s", exc)

        return res

    return aggregate_fit_metrics, aggregate_eval_metrics


def main() -> None:
    setup_logging(rank=0)
    args = apply_config_file(build_parser())

    try:
        import flwr as fl
    except ImportError:
        logger.error(
            "Flower (`flwr`) is not installed in the environment. "
            "Please install it using: pip install flwr"
        )
        sys.exit(1)

    from .strategy import FedMom

    port = args.port or args.flower_port or 29511
    server_address = f"{args.host}:{port}"

    state = None
    blob_root = None
    if args.state_dir:
        import torch

        from .globalstate import GlobalState

        dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16}
        state = GlobalState(args.state_dir, storage_dtype=dtypes[args.storage_dtype])

    if args.transport == proto.TRANSPORT_BLOB:
        missing = [
            name for name, value in (
                ("--state-dir", args.state_dir),
                ("--blob-root", args.blob_root),
                ("--blob-url", args.blob_url),
            ) if not value
        ]
        if missing:
            logger.error(
                "blob transport requires %s. The server reads deltas straight off the "
                "local blob directory and tells clients where to reach the store.",
                ", ".join(missing),
            )
            sys.exit(2)
        blob_root = Path(args.blob_root)
        blob_root.mkdir(parents=True, exist_ok=True)
        if state is not None and state.initialised:
            state.log_disk_budget(sites=max(1, args.min_clients))

    # Checkpoints of the merged global model. Only inline transport needs these: the
    # blob path already writes the global model through GlobalState, while inline held
    # it purely in memory -- --state-dir was set, the startup line said "durable state
    # in ...", and the directory stayed empty through every successful merge.
    checkpoint_dir = None
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    elif args.state_dir and args.transport != proto.TRANSPORT_BLOB:
        checkpoint_dir = Path(args.state_dir) / "checkpoints"
    if checkpoint_dir is not None:
        logger.info(
            "global checkpoints in %s | %d ephemeral (every merge) + %s persistent "
            "(every %d merges)",
            checkpoint_dir, args.keep_ephemeral,
            args.keep_persistent or "unbounded", args.persist_every,
        )
        if not args.keep_persistent:
            logger.warning(
                "--keep-persistent 0 is unbounded: at 0.6B a checkpoint is ~4.8 GiB, so "
                "%d rounds at --persist-every %d would need ~%.0f GiB",
                args.num_rounds, max(1, args.persist_every),
                args.num_rounds / max(1, args.persist_every) * 4.8,
            )

    logger.info("Starting Flower Aggregator Server (FedMom) on %s...", server_address)
    logger.info(
        "transport=%s | server_learning_rate=%s, server_momentum=%s | "
        "min_clients=%d%s | num_rounds=%d (attempts), round_timeout=%.0fs",
        args.transport, args.server_learning_rate, args.server_momentum,
        args.min_clients,
        " (elastic: trains with whoever is available)" if args.min_clients <= 1 else
        " (blocks until this many sites are out of the queue)",
        args.num_rounds, args.round_timeout,
    )
    if state is not None:
        logger.info(
            "durable state in %s | merge round %d | clusters known: %s",
            args.state_dir, state.round, sorted(state.clusters) or "none yet",
        )
    else:
        logger.warning(
            "no --state-dir: the global model lives only in memory, so a restart of "
            "this process loses the run and the server cannot start before a cluster "
            "connects"
        )
    if args.server_momentum == 0.0:
        logger.warning(
            "server_momentum is 0.0, which makes the outer step algebraically identical "
            "to plain FedAvg averaging (v = w - 1.0*(w - w_avg) = w_avg). Correct for "
            "a BatchNorm model; for an LLM this switches FedMom off."
        )

    aggregate_fit_metrics, aggregate_eval_metrics = build_metric_aggregators(
        enable_wandb=args.enable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        config_dict=vars(args),
    )

    strategy = FedMom(
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        server_learning_rate=args.server_learning_rate,
        server_momentum=args.server_momentum,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
        transport=args.transport,
        state=state,
        blob_root=blob_root,
        blob_url=args.blob_url,
        run_id=args.run_id,
        keep_rounds=args.keep_rounds,
        checkpoint_dir=checkpoint_dir,
        keep_ephemeral=args.keep_ephemeral,
        persist_every=args.persist_every,
        keep_persistent=args.keep_persistent,
        solo_full_step=args.solo_full_step,
    )

    # Resume from disk before Flower's INIT, so `initialize_parameters` answers from the
    # checkpoint and no client is asked to re-seed a run that already has a model.
    if checkpoint_dir is not None and args.fresh_model:
        # Same reasoning as DARL's --fresh: not loading them is not enough, because they
        # stay on disk and the *next* restart without this flag would adopt them. Moved
        # aside rather than deleted so a mistaken --fresh-model is recoverable.
        moved = 0
        for stale in sorted(checkpoint_dir.glob("round-*.npz")):
            try:
                stale.replace(stale.with_suffix(".npz.superseded"))
                moved += 1
            except OSError as exc:
                logger.warning("could not move %s aside: %s", stale.name, exc)
        if moved:
            logger.info("--fresh-model: moved %d checkpoint(s) aside; this run starts from "
                        "whichever cluster seeds it", moved)

    if checkpoint_dir is not None and not args.fresh_model:
        resumed = strategy.resume_from_checkpoint()
        if resumed is not None:
            logger.info("resumed the global model at merge round %d from %s",
                        resumed, checkpoint_dir)
        else:
            logger.info("no usable checkpoint in %s; the first cluster to connect will "
                        "seed the global model", checkpoint_dir)

    if args.min_clients <= 1:
        logger.info(
            "waiting for the first cluster; every site being queued is a normal state "
            "and nothing is lost while it lasts"
        )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(
            num_rounds=args.num_rounds, round_timeout=args.round_timeout
        ),
        strategy=strategy,
    )

    if state is not None:
        logger.info(
            "server stopped at merge round %d; state in %s is complete and a restart "
            "resumes from it", state.round, args.state_dir,
        )


if __name__ == "__main__":
    main()
