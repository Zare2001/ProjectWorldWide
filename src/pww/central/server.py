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


def build_metric_aggregators():
    """The two metric aggregation callbacks, as a pair.

    Module level and returned rather than closed over inside `main` so the
    arithmetic is reachable from tests. The perplexity pooling in particular is the
    kind of thing that is wrong for years if nothing asserts it -- it produces a
    plausible number either way.
    """
    def aggregate_fit_metrics(metrics: list[tuple[int, dict]]) -> dict:
        """Token-weighted training loss, plus how far replicas drifted."""
        total = sum(n for n, _ in metrics)
        if total == 0:
            logger.info("  >> no tokens trained this round")
            return {}
        avg_loss = sum(n * m.get(proto.LOSS, 0.0) for n, m in metrics) / total
        out = {"loss": avg_loss}
        drifts = [float(m["drift_ratio"]) for _, m in metrics if "drift_ratio" in m]
        detail = ""
        if drifts:
            # Unweighted across clusters, and deliberately: drift is ||local - global|| /
            # ||global||, a property of a replica's trajectory rather than of its tokens,
            # so weighting it by token count would say a fast site drifted more simply by
            # doing more work.
            out["drift_ratio"] = sum(drifts) / len(drifts)
            # The max is the actionable one. This is the quantity H should be tuned
            # against -- once a replica's local update approaches the norm of the weights
            # themselves, averaging replicas destroys rather than combines their progress
            # -- and it is the *worst* replica that decides that, not the average. Two
            # sites at 0.01 and 0.30 average to a reassuring 0.155.
            out["drift_ratio_max"] = max(drifts)
            detail = f", drift {out['drift_ratio']:.4f} (max {max(drifts):.4f})"
        names = ", ".join(str(m.get(proto.CLUSTER, "?")) for _, m in metrics)
        # Training perplexity alongside the loss. exp() of a token-weighted mean loss is
        # the perplexity of the union of those tokens, the same identity the held-out
        # figure uses -- so it is exact here rather than an approximation, and it is the
        # number that is comparable with published pre-training curves.
        out["perplexity"] = math.exp(min(20.0, avg_loss)) if math.isfinite(avg_loss) else float("nan")
        logger.info(
            "  >> Training loss %.4f (ppl %.2f)  (%d cluster(s) [%s], %s tokens%s)",
            avg_loss, out["perplexity"], len(metrics), names, f"{total:,}", detail,
        )

        # Throughput and hardware, per cluster.
        #
        # Aggregated tokens/s is the sum, because the sites train concurrently -- the run's
        # rate is what both produce together. MFU and memory are NOT aggregated: MFU is a
        # ratio against a device's peak FLOPs, and an MI250X GCD and an H100 have different
        # peaks, so a mean of the two describes neither. Reported per cluster instead, which
        # is also the form that tells you *which* site is underused.
        rate = sum(float(m["tokens_per_s"]) for _, m in metrics if "tokens_per_s" in m)
        if rate > 0:
            out["tokens_per_s"] = rate
            slowest = max((float(m.get("seconds", 0.0)) for _, m in metrics), default=0.0)
            per_site = "; ".join(
                f"{m.get(proto.CLUSTER, '?')} {float(m.get('tokens_per_s', 0)):,.0f} tok/s"
                + (f", {float(m['mfu_pct']):.1f}% MFU" if "mfu_pct" in m else "")
                + (f", {float(m['tflops_per_rank']):.1f} TFLOP/s/rank"
                   if "tflops_per_rank" in m else "")
                + (f", {float(m['peak_memory_gib']):.1f} GiB"
                   f" ({float(m.get('peak_memory_pct', 0)):.0f}%)"
                   if "peak_memory_gib" in m else "")
                for _, m in metrics
            )
            logger.info("  >> Throughput %s tok/s combined | %s", f"{rate:,.0f}", per_site)
            if slowest > 0:
                # The straggler sets the round's wall time: every site waits for the last
                # one before the merge. Worth seeing next to the rates, because a site can
                # be fast per-token and still be the one everyone waits on if it was given
                # more blocks.
                out["round_seconds"] = slowest
                logger.info("  >> Round took %.0fs (slowest site's inner phase)", slowest)

        lrs = [float(m["lr"]) for _, m in metrics if "lr" in m]
        if lrs:
            out["lr"] = lrs[0]
            # A spread here means the sites are at different points in the schedule, which
            # they should not be: it is over *global* steps and survives outer rounds.
            if max(lrs) - min(lrs) > 1e-12:
                logger.warning("  >> learning rate differs across clusters: %s -- the "
                               "schedule is over global steps and should agree",
                               ", ".join(f"{v:.3e}" for v in lrs))
        return out

    def aggregate_eval_metrics(metrics: list[tuple[int, dict]]) -> dict:
        """Pooled held-out perplexity, and accuracy for the CIFAR path.

        Report each under its own name: an earlier version logged perplexity as
        "Test Accuracy: 21.69%", which made a model that never improved look like one at
        21% accuracy and climbing.
        """
        total = sum(n for n, _ in metrics)
        if total == 0:
            return {}

        # Perplexity is aggregated through the LOSS, never by averaging perplexities.
        #
        # ppl = exp(mean NLL per token), so the perplexity of the union of the clusters'
        # validation tokens is exp() of their token-weighted mean loss -- exactly, when
        # each cluster reports its own mean NLL per token, which it does. Averaging
        # per-cluster perplexities instead computes mean(exp(L)), and exp is convex, so
        # by Jensen that is always >= exp(mean(L)): a pessimistic number, wrong by more
        # the further apart the clusters are. Two sites at loss 2.0 and 4.0 report 31.0
        # against a true 20.1, and the error moves with cluster skew rather than with the
        # model, which is the worst property a training metric can have.
        reported = [(n, float(m["eval_loss"]), str(m.get(proto.CLUSTER, "?")))
                    for n, m in metrics if "eval_loss" in m]

        # A non-finite loss is excluded, not pooled.
        #
        # One cluster reporting nan otherwise makes the pooled loss nan, and the clamp
        # below hides it: min(20.0, nan) returns 20.0, because `nan < 20.0` is False, so
        # exp(20) = 485165195.41 gets printed as though it were a real perplexity. That
        # is what ten consecutive rounds of a dead run looked like -- a specific,
        # plausible-looking number rather than an obvious nan.
        losses = [(n, v, c) for n, v, c in reported if math.isfinite(v)]
        dropped = [(c, v) for n, v, c in reported if not math.isfinite(v)]
        if dropped:
            logger.error(
                "  >> excluded %d non-finite held-out loss(es) from the pooled figure: "
                "%s. A site reporting this is not producing a usable model; its training "
                "loss is where to look.",
                len(dropped), ", ".join(f"{c}={v}" for c, v in dropped),
            )

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

            # Every cluster evaluates the same global model, so a wide spread in held-out
            # loss is worth looking at. It is a prompt, not a verdict, and the two
            # explanations need different fixes:
            #
            #   the site is not running the weights it was sent -- a parameter-application
            #   bug, which is the serious one, or
            #
            #   the sites are not scoring the same data. With validation.dataset pointed at
            #   a small fixture and validation.steps small, sites with different rank
            #   counts consume different amounts of a re-looped set, and the gap can be
            #   large without anything being wrong with the model.
            #
            # A real run observed 10.85 vs 8.80 nats here one round before a site's
            # contribution arrived as nan, so the signal is worth surfacing -- but that
            # run also had validation on a 2,000-document fixture at 8 vs 4 ranks, which
            # is enough on its own to explain a gap that size. Hence the wording.
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
            return {"eval_loss": pooled_loss, "perplexity": pooled_ppl}
        if reported:
            logger.error("  >> every cluster reported a non-finite held-out loss; "
                         "no perplexity for this round")
            return {}

        values = [(n, float(m["accuracy"])) for n, m in metrics if "accuracy" in m]
        if values:
            # Accuracy is a mean of per-sample 0/1 outcomes, so a sample-weighted mean of
            # per-cluster accuracies *is* the pooled accuracy. Linear, unlike perplexity.
            weight = sum(n for n, _ in values)
            pooled = sum(n * v for n, v in values) / weight
            per_cluster = ", ".join(f"{v:.2f}%" for _, v in values)
            logger.info(
                "  >> Test accuracy %.2f%%  (per-cluster: [%s], %s samples)",
                pooled, per_cluster, f"{weight:,}",
            )
            return {"accuracy": pooled}
        return {}

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

    aggregate_fit_metrics, aggregate_eval_metrics = build_metric_aggregators()

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
