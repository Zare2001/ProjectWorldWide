"""Parallelism strategies: DDP and FSDP.

Choosing between them is a memory-vs-communication trade:

  ddp    Every rank holds a full copy of the model and optimizer state, and
         gradients are all-reduced. Fastest option whenever the model fits
         comfortably. Correct choice for ResNet-18 (11M params).

  fsdp   Parameters, gradients and optimizer state are sharded across ranks and
         gathered layer-by-layer during forward/backward. Costs extra
         communication, buys you models far larger than one GCD's 64 GiB. This
         is the path for LLM training.

Both are exposed through one `wrap_model` call so a training script can switch
strategy with a flag rather than a rewrite.
"""

from __future__ import annotations

import functools
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

from .logging_utils import get_logger

STRATEGIES = ("single", "ddp", "fsdp")

# Precision policy. bf16 is the right default on MI250X: it has the dynamic
# range of fp32 so no loss scaler is needed, unlike fp16.
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"unknown dtype {name!r}, expected one of {sorted(_DTYPES)}")
    return _DTYPES[name]


def build_mesh(device_type: str, world_size: int):
    """1-D mesh over all ranks.

    Kept as a seam: tensor/pipeline parallelism for large LLMs means giving this
    a second dimension, e.g. init_device_mesh(t, (dp, tp), ("dp", "tp")).
    """
    return init_device_mesh(device_type, (world_size,), mesh_dim_names=("dp",))


def wrap_model(
    model: nn.Module,
    *,
    strategy: str,
    device: torch.device,
    dtype: str = "fp32",
    transformer_layer_cls: Iterable[type[nn.Module]] | None = None,
    activation_checkpointing: bool = False,
) -> nn.Module:
    """Move `model` to `device` and apply the requested parallelism strategy.

    transformer_layer_cls tells FSDP which module type to treat as a shard unit.
    Omitting it for a large transformer means the whole model becomes one flat
    shard, which defeats the point -- so pass e.g. {LlamaDecoderLayer}.
    """
    log = get_logger()
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")

    if strategy == "single":
        return model.to(device)

    if not dist.is_initialized():
        log.warning("strategy=%s requested but process group is not initialised; "
                    "running single-process", strategy)
        return model.to(device)

    if strategy == "ddp":
        model = model.to(device)
        device_ids = [device.index] if device.type == "cuda" else None
        return DDP(model, device_ids=device_ids, gradient_as_bucket_view=True)

    # --- FSDP ---------------------------------------------------------------
    mesh = build_mesh(device.type, dist.get_world_size())

    auto_wrap_policy = None
    if transformer_layer_cls:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(transformer_layer_cls),
        )

    mixed_precision = None
    if dtype != "fp32":
        compute_dtype = resolve_dtype(dtype)
        mixed_precision = MixedPrecision(
            param_dtype=compute_dtype,
            reduce_dtype=torch.float32,   # fp32 gradient reduction: cheap, avoids drift
            buffer_dtype=compute_dtype,
        )

    fsdp_model = FSDP(
        model,
        device_mesh=mesh,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device if device.type == "cuda" else None,
        limit_all_gathers=True,
        use_orig_params=True,   # keeps param groups / torch.compile working
    )

    if activation_checkpointing:
        _apply_activation_checkpointing(fsdp_model, transformer_layer_cls)

    return fsdp_model


def _apply_activation_checkpointing(model: nn.Module, layer_cls) -> None:
    """Recompute activations in backward instead of storing them.

    Trades roughly 30% extra compute for a large activation-memory saving --
    usually the difference between fitting a long sequence length and not.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    if not layer_cls:
        get_logger().warning(
            "activation_checkpointing requested without transformer_layer_cls; skipping"
        )
        return

    layer_cls = set(layer_cls)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=lambda submodule: isinstance(submodule, tuple(layer_cls)),
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(total, trainable) parameter counts.

    Under FSDP each rank only holds a shard, so this is summed across ranks to
    report the true model size.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if isinstance(model, FSDP) and dist.is_initialized():
        counts = torch.tensor([total, trainable], dtype=torch.float64,
                              device=next(model.parameters()).device)
        dist.all_reduce(counts)
        total, trainable = int(counts[0].item()), int(counts[1].item())
    return total, trainable
