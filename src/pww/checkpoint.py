"""Checkpoint save/load for DDP and FSDP models.

Uses `torch.distributed.checkpoint.state_dict` helpers, which give the same
call signature regardless of whether the model is bare, DDP-wrapped or
FSDP-wrapped. Two formats:

  consolidated  a single file holding the full unsharded state on rank 0.
                Portable and easy to inspect -- fine up to a few billion
                parameters, but rank 0 must hold the whole model in host RAM.

  sharded       torch.distributed.checkpoint: every rank writes its own shard
                in parallel. This is what you want once models get large; it
                also resumes onto a *different* world size.

CIFAR runs default to `consolidated`; LLM runs should use `sharded`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from .logging_utils import get_logger


def _full_options() -> StateDictOptions:
    return StateDictOptions(full_state_dict=True, cpu_offload=True)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    step: int = 0,
    epoch: int = 0,
    rank: int = 0,
    sharded: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a checkpoint. Collective -- every rank must call it."""
    path = Path(path)
    log = get_logger()

    if sharded:
        # Every rank writes into a directory in parallel.
        state = {"model": get_model_state_dict(model)}
        if optimizer is not None:
            state["optim"] = get_optimizer_state_dict(model, optimizer)
        path.mkdir(parents=True, exist_ok=True)
        dcp.save(state, checkpoint_id=str(path))
        if rank == 0:
            (path / "meta.json").write_text(json.dumps({"step": step, "epoch": epoch, **(extra or {})}))
            log.info("saved sharded checkpoint -> %s (step=%d epoch=%d)", path, step, epoch)
        return

    # Consolidated: gather full state, rank 0 writes one file.
    opts = _full_options()
    model_state = get_model_state_dict(model, options=opts)
    optim_state = (
        get_optimizer_state_dict(model, optimizer, options=opts) if optimizer is not None else None
    )

    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model_state,
            "optim": optim_state,
            "step": step,
            "epoch": epoch,
            "extra": extra or {},
        }
        # Write then rename: a job killed mid-write leaves the previous
        # checkpoint intact instead of a truncated file.
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)
        log.info("saved checkpoint -> %s (step=%d epoch=%d)", path, step, epoch)

    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    sharded: bool = False,
    strict: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint in place. Returns its metadata (step/epoch/extra)."""
    path = Path(path)
    log = get_logger()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    if sharded:
        state = {"model": get_model_state_dict(model)}
        if optimizer is not None:
            state["optim"] = get_optimizer_state_dict(model, optimizer)
        dcp.load(state, checkpoint_id=str(path))
        set_model_state_dict(model, state["model"], options=StateDictOptions(strict=strict))
        if optimizer is not None and "optim" in state:
            set_optimizer_state_dict(model, optimizer, state["optim"])
        meta_file = path / "meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        log.info("loaded sharded checkpoint <- %s", path)
        return meta

    # Consolidated. Load on every rank; set_*_state_dict reshards as needed.
    # weights_only=False because the payload carries plain ints/dicts too.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True, strict=strict)
    set_model_state_dict(model, payload["model"], options=opts)
    if optimizer is not None and payload.get("optim") is not None:
        set_optimizer_state_dict(model, optimizer, payload["optim"], options=opts)
    log.info("loaded checkpoint <- %s (step=%s epoch=%s)", path, payload.get("step"), payload.get("epoch"))
    return {"step": payload.get("step", 0), "epoch": payload.get("epoch", 0), **payload.get("extra", {})}


def latest_checkpoint(output_dir: str | Path, sharded: bool = False) -> Path | None:
    """Find the newest checkpoint in a run directory, for auto-resume."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    pattern = "step_*" if sharded else "*.pt"
    candidates = [p for p in output_dir.glob(pattern) if (p.is_dir() if sharded else p.is_file())]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
