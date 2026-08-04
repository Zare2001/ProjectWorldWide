"""Config files layered under argparse.

Precedence: command line > YAML file > argparse defaults. Keeps short
experiments as one-liners while letting a real run be a reviewable file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from . import diloco


def apply_config_file(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args, folding in `--config file.yaml` as overridden defaults."""
    args, _ = parser.parse_known_args(argv)
    config_path = getattr(args, "config", None)
    if config_path:
        raw = yaml.safe_load(Path(config_path).read_text()) or {}
        # YAML files read naturally with the flag spelling (`batch-size`), but
        # argparse stores that as `batch_size`. Accept either.
        data = {key.replace("-", "_"): value for key, value in raw.items()}
        known = {a.dest for a in parser._actions}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown keys in {config_path}: {sorted(unknown)}")
        parser.set_defaults(**data)
    return parser.parse_args(argv)


def save_config(args: argparse.Namespace, output_dir: str | Path) -> None:
    """Snapshot the resolved config next to the run's outputs.

    Cheap, and the only reliable way to answer "what exactly did that run use?"
    three weeks later.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments shared by every training entrypoint (CIFAR and LLM alike)."""
    g = parser.add_argument_group("run")
    g.add_argument("--config", type=str, default=None, help="YAML config file")
    g.add_argument("--run-name", type=str, default=None, help="Subdirectory under the output dir")
    g.add_argument("--output-dir", type=str, default=None,
                   help="Where checkpoints/logs go (default: $PWW_OUTPUT_DIR)")
    g.add_argument("--seed", type=int, default=42)

    g = parser.add_argument_group("parallelism")
    g.add_argument("--parallel", type=str, default="ddp", choices=("single", "ddp", "fsdp"))
    g.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "bf16", "fp16"))

    # DiLoCo lives here rather than in one trainer because the outer loop is
    # independent of what is being trained -- the LLM entrypoint gets it free.
    g = parser.add_argument_group("diloco")
    g.add_argument("--diloco-replicas", type=int, default=0,
                   help="k: number of model replicas. 0 disables DiLoCo. Must divide "
                        "the world size. k=1 is a valid degenerate case (no "
                        "inter-replica traffic) useful for isolating the outer loop")
    g.add_argument("--diloco-inner-steps", type=int, default=diloco.DEFAULT_INNER_STEPS,
                   help="H: inner steps between outer steps. Larger means less "
                        "communication and more replica drift")
    g.add_argument("--diloco-outer-lr", type=float, default=diloco.DEFAULT_OUTER_LR)
    g.add_argument("--diloco-outer-momentum", type=float, default=diloco.DEFAULT_OUTER_MOMENTUM)
    g.add_argument("--diloco-outer-optimizer", type=str, default="nesterov",
                   choices=diloco.OUTER_OPTIMIZERS,
                   help="nesterov is the paper's choice; 'sgd' with --diloco-outer-lr 1 "
                        "--diloco-outer-momentum 0 is FederatedAveraging")
    g.add_argument("--diloco-outer-device", type=str, default="auto", choices=("auto", "cpu"),
                   help="'cpu' keeps theta and the outer momentum in host memory, "
                        "trading two host<->device copies every H steps for two "
                        "fewer model-sized allocations on the accelerator")
    g.add_argument("--diloco-no-sync-buffers", action="store_true",
                   help="Do not average float buffers (BatchNorm statistics) across "
                        "replicas at the outer step")

    # DARL lives here for the same reason DiLoCo does: nothing in it knows what is
    # being trained. A trainer opts in by building a session and a data source; see
    # the DARL section of the README.
    g = parser.add_argument_group("darl")
    g.add_argument("--darl-url", type=str, default=None,
                   help="Lease coordinator, e.g. http://int-node-1:8760. Unset (and "
                        "no $PWW_DARL_URL) disables DARL and the trainer shards data "
                        "over its own ranks as usual")
    g.add_argument("--darl-token", type=str, default=None,
                   help="Shared secret for the coordinator (default: $DARL_TOKEN)")
    g.add_argument("--darl-num-samples", type=int, default=None,
                   help="N: samples in the global corpus. Must match the coordinator; "
                        "a mismatch is refused at registration rather than silently "
                        "duplicating data")
    g.add_argument("--darl-block-size", type=int, default=10_000,
                   help="K: samples per leased block. Coarse enough that leasing "
                        "costs nothing, fine enough that a dead cluster only strands "
                        "one lease")
    g.add_argument("--darl-blocks-per-phase", type=int, default=0,
                   help="Blocks per lease. 0 derives it from H, the batch and the "
                        "rank count, so one lease is exactly one local phase")
    g.add_argument("--darl-commit-policy", type=str, default="checkpoint",
                   choices=("checkpoint", "consumption"),
                   help="'checkpoint' commits only what a durable checkpoint covers, "
                        "which is what makes the epoch exactly-once under a crash; "
                        "'consumption' releases spans sooner and accepts a one-lease "
                        "window of gaps or duplicates")
    g.add_argument("--darl-site", type=str, default=None,
                   help="Name this cluster reports (default: $PWW_SITE). Stable across "
                        "requeues on purpose, so measured throughput is not lost")
    g.add_argument("--darl-proxy", action="store_true",
                   help="Reach the coordinator through $http_proxy. Needed only when "
                        "the coordinator is at another facility")

    g = parser.add_argument_group("checkpointing")
    g.add_argument("--resume", type=str, default=None,
                   help="Checkpoint path, or 'auto' to pick the newest in the output dir")
    g.add_argument("--sharded-checkpoint", action="store_true",
                   help="Write parallel per-rank shards instead of one consolidated file")
    return parser


def resolve_output_dir(args: argparse.Namespace, default_name: str) -> Path:
    import os

    base = args.output_dir or os.environ.get("PWW_OUTPUT_DIR", "./runs")
    name = args.run_name or default_name
    return Path(base) / name


def set_seed(seed: int, rank: int = 0) -> None:
    """Seed the RNGs.

    Offset by rank so that data augmentation and dropout are not identical
    across ranks (which would waste most of the parallelism).
    """
    import random

    import numpy as np
    import torch

    effective = seed + rank
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)


def maybe_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}
