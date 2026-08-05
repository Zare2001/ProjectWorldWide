"""Rank-aware logging plus a small JSONL metrics writer.

Two rules that make multi-node logs readable:
  * only rank 0 writes human-readable progress to stdout,
  * every log line carries the rank, so a crash on rank 113 is attributable.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

_LOGGER_NAME = "pww"


def setup_logging(rank: int, output_dir: str | Path | None = None, level: int = logging.INFO):
    """Configure the `pww` logger. Non-zero ranks only emit WARNING and above."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level if rank == 0 else logging.WARNING)

    fmt = logging.Formatter(
        fmt=f"[%(asctime)s][r{rank}][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    # Rank 0 also tees to a file so a run survives losing the SLURM output.
    if output_dir is not None and rank == 0:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "train.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = _LOGGER_NAME):
    return logging.getLogger(name)


class MetricsWriter:
    """Append-only JSONL metrics log, written by rank 0 only.

    JSONL rather than a plotting library on purpose: it costs nothing on a
    compute node, survives a crashed job, and is trivial to load afterwards
    with pandas or feed to TensorBoard/W&B later.
    """

    def __init__(self, output_dir: str | Path, rank: int, filename: str = "metrics.jsonl"):
        self.enabled = rank == 0
        self._fh = None
        self._t0 = time.time()
        if self.enabled:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(output_dir / filename, "a", buffering=1)

    def log(self, **kwargs) -> None:
        if not self.enabled or self._fh is None:
            return
        record = {"wall_s": round(time.time() - self._t0, 3), **kwargs}
        self._fh.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def log_environment(logger, info) -> None:
    """Record the facts you always wish you had when a run looks wrong later."""
    import torch

    logger.info("=" * 72)
    logger.info("ProjectWorldWide run")
    logger.info("=" * 72)
    logger.info("topology       : %s", info.describe())
    logger.info("torch          : %s", torch.__version__)
    logger.info("rocm/hip       : %s", getattr(torch.version, "hip", None))
    if torch.cuda.is_available():
        logger.info("gpu            : %s", torch.cuda.get_device_name(info.device))
        props = torch.cuda.get_device_properties(info.device)
        logger.info("gpu memory     : %.1f GiB", props.total_memory / 1024**3)
    for key in ("SLURM_JOB_ID", "SLURM_JOB_NUM_NODES", "SLURM_JOB_NODELIST"):
        if key in os.environ:
            logger.info("%-15s: %s", key.lower(), os.environ[key])
