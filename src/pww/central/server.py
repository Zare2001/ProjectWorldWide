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
        "--port", type=int, default=29511, help="Port to listen on (open SG port: 29511)"
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
    setup_logging()
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

    server_address = f"{args.host}:{args.port}"
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
