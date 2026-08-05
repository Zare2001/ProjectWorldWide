"""Central Flower Aggregator Server Entrypoint using FedMom Strategy.

Starts the Flower server listening on port 29511 for incoming Snellius and LUMI
cluster connections, orchestrating the FedMom strategy.

Usage:
    python3 -m pww.central.server --port 29511 --num-rounds 50 --server-learning-rate 1.0 --server-momentum 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import apply_config_file
from ..logging_utils import get_logger, setup_logging
from .strategy import FedMom

logger = get_logger("pww.central.server")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PWW Central Node Flower Aggregator Server (FedMom)"
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument(
        "--port", type=int, default=None, help="Port to listen on"
    )
    parser.add_argument(
        "--flower-port", type=int, default=None, help="Alias for --port"
    )
    parser.add_argument(
        "--darl-port", type=int, default=29510, help="DARL coordinator port"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000000, help="DARL total sample count"
    )
    parser.add_argument(
        "--block-size", type=int, default=10000, help="DARL block size"
    )
    parser.add_argument(
        "--darl-state-dir", type=str, default="./runs/darl", help="DARL state directory"
    )
    parser.add_argument(
        "--num-rounds", type=int, default=50, help="Number of outer training rounds"
    )
    parser.add_argument(
        "--min-clients", type=int, default=2, help="Minimum clients required per round (Snellius + LUMI)"
    )
    parser.add_argument(
        "--server-learning-rate", type=float, default=1.0, help="FedMom server learning rate (eta)"
    )
    parser.add_argument(
        "--server-momentum", type=float, default=0.9, help="FedMom server momentum factor (beta)"
    )
    return parser


def main() -> None:
    setup_logging(rank=0)
    parser = build_parser()
    args = apply_config_file(parser)

    try:
        import flwr as fl
    except ImportError:
        logger.error(
            "Flower (`flwr`) is not installed in the environment. "
            "Please install it using: pip install flwr"
        )
        sys.exit(1)

    port = args.port or args.flower_port or 29511
    server_address = f"{args.host}:{port}"
    logger.info(f"Starting Flower Aggregator Server (FedMom) on {server_address}...")
    logger.info(
        f"Configuration: server_learning_rate={args.server_learning_rate}, "
        f"server_momentum={args.server_momentum}, num_rounds={args.num_rounds}, "
        f"min_clients={args.min_clients}"
    )

    strategy = FedMom(
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        server_learning_rate=args.server_learning_rate,
        server_momentum=args.server_momentum,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
