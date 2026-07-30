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
