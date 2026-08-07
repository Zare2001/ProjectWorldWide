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
import sys
from pathlib import Path

from ..config import apply_config_file
from ..logging_utils import get_logger, setup_logging
from .. import fedproto as proto

logger = get_logger("pww.central.server")


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
    return parser


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

    def aggregate_fit_metrics(metrics: list[tuple[int, dict]]) -> dict:
        """Token-weighted training loss, plus how far replicas drifted."""
        total = sum(n for n, _ in metrics)
        if total == 0:
            logger.info("  >> no tokens trained this round")
            return {}
        avg_loss = sum(n * m.get(proto.LOSS, 0.0) for n, m in metrics) / total
        out = {"loss": avg_loss}
        drifts = [m["drift_ratio"] for _, m in metrics if "drift_ratio" in m]
        detail = ""
        if drifts:
            out["drift_ratio"] = sum(drifts) / len(drifts)
            # The quantity H should be tuned against: once a replica's local update
            # approaches the norm of the weights themselves, averaging replicas starts
            # destroying rather than combining their progress.
            detail = f", drift {out['drift_ratio']:.4f}"
        names = ", ".join(str(m.get(proto.CLUSTER, "?")) for _, m in metrics)
        logger.info(
            "  >> Training loss %.4f  (%d cluster(s) [%s], %s tokens%s)",
            avg_loss, len(metrics), names, f"{total:,}", detail,
        )
        return out

    def aggregate_eval_metrics(metrics: list[tuple[int, dict]]) -> dict:
        total = sum(n for n, _ in metrics)
        if total == 0:
            return {}
        # LLM clients report perplexity; the CIFAR client reports accuracy. Whichever
        # arrives, report it under its own name -- an earlier version logged perplexity
        # as "Test Accuracy: 21.69%", which made a model that never improved look like
        # one at 21% accuracy and climbing.
        for key, label, fmt in (
            ("perplexity", "Perplexity", "{:.2f}"),
            ("accuracy", "Test accuracy", "{:.2f}%"),
        ):
            values = [(n, m[key]) for n, m in metrics if key in m]
            if not values:
                continue
            weighted = sum(n * v for n, v in values) / sum(n for n, _ in values)
            per_cluster = ", ".join(fmt.format(v) for _, v in values)
            logger.info(
                "  >> %s %s  (per-cluster: [%s], %s samples)",
                label, fmt.format(weighted), per_cluster, f"{total:,}",
            )
            return {key: weighted}
        return {}

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
    )

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
