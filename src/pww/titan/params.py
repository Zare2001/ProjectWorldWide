"""Moving a sharded model's weights on and off the wire.

Under FSDP2 every parameter is a `DTensor` holding one shard, so "the model's
parameters" is not something any single rank has. `torch.distributed.checkpoint`'s
`get_model_state_dict` / `set_model_state_dict` are the supported way to gather and
scatter a full state dict across whatever parallelism is configured, and going
through them (rather than reaching for `.full_tensor()` per parameter) is what
keeps this working when tensor or pipeline parallelism is switched on.

Ordering is the subtle part. Flower carries an ordered *list* of arrays with no
names attached, and the server averages position-wise -- so if two clusters
enumerated their parameters in different orders the aggregate would mix unrelated
tensors together and the loss would look like noise rather than like a bug. Keys
are therefore sorted, and `ParameterCodec` records the exact order, shape and dtype
it flattened so the reverse direction cannot drift.

Cost
----
One gather and one scatter per outer round. That is the whole point of DiLoCo: at
H=100 inner steps it happens 1% as often as a gradient all-reduce would, which is
what makes a WAN hop affordable. But the gather does materialise a full copy of the
model in host memory on every rank -- `get_model_state_dict` with
`full_state_dict=True` all-gathers rather than gathering to rank 0 -- so a 0.6B
model in float32 costs ~2.4 GB of host RAM per rank. The scatter avoids the
symmetric cost by using `broadcast_from_rank0`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from ..logging_utils import get_logger

logger = get_logger("pww.titan.params")

# numpy has no bfloat16, and Flower serialises numpy arrays, so bf16 cannot be a
# wire dtype here even though it is the compute dtype. float16 halves WAN traffic and
# is exact for weights of normal magnitude -- it has more mantissa than bfloat16 --
# but it has far less range: bfloat16 reaches ~1e-38 while float16's smallest
# subnormal is ~6e-08, so weights below that flush toward zero. Harmless (such a
# weight contributes nothing to a forward pass) but real, and pinned by a test rather
# than assumed away. float32 keeps everything, at twice the bytes.
WIRE_DTYPES = {"float16": np.float16, "float32": np.float32}


@dataclass
class ParameterCodec:
    """Flattens a model state dict to an ordered array list, and back.

    Built once per run from the model itself, then reused every round, so the
    ordering is fixed for the lifetime of the client.
    """

    keys: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    wire_dtype: str = "float16"
    numel: int = field(default=0)

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor], wire_dtype: str = "float16") -> "ParameterCodec":
        if wire_dtype not in WIRE_DTYPES:
            raise ValueError(
                f"wire_dtype must be one of {sorted(WIRE_DTYPES)}, got {wire_dtype!r} "
                f"(numpy has no bfloat16, so bf16 cannot cross a Flower connection)"
            )
        keys = tuple(sorted(state))
        return cls(
            keys=keys,
            shapes=tuple(tuple(state[k].shape) for k in keys),
            dtypes=tuple(state[k].dtype for k in keys),
            wire_dtype=wire_dtype,
            numel=sum(state[k].numel() for k in keys),
        )

    @property
    def wire_bytes(self) -> int:
        return self.numel * np.dtype(WIRE_DTYPES[self.wire_dtype]).itemsize

    def encode(self, state: dict[str, torch.Tensor]) -> list[np.ndarray]:
        target = WIRE_DTYPES[self.wire_dtype]
        out = []
        for key in self.keys:
            tensor = state[key]
            # float32 first: casting bf16 straight to a numpy dtype is not
            # supported, and the intermediate is what makes float16 output exact
            # rather than reinterpreted.
            out.append(tensor.detach().to(torch.float32).cpu().numpy().astype(target, copy=False))
        return out

    def decode(self, arrays: list[np.ndarray]) -> dict[str, torch.Tensor]:
        if len(arrays) != len(self.keys):
            raise ValueError(
                f"received {len(arrays)} arrays but this model has {len(self.keys)} "
                f"parameters -- the server is aggregating a different model, or two "
                f"clusters are running different flavors/vocab sizes"
            )
        state = {}
        for key, shape, dtype, array in zip(self.keys, self.shapes, self.dtypes, arrays):
            if tuple(array.shape) != shape:
                raise ValueError(
                    f"parameter {key} arrived with shape {tuple(array.shape)}, "
                    f"expected {shape}"
                )
            state[key] = torch.from_numpy(np.ascontiguousarray(array)).to(dtype)
        return state


def gather_full_state(model_parts: list[nn.Module]) -> dict[str, torch.Tensor]:
    """Full, unsharded, CPU-resident model state on every rank.

    `model_parts` is torchtitan's list because pipeline parallelism splits a model
    across ranks into several modules; without PP it has exactly one entry.
    """
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state: dict[str, torch.Tensor] = {}
    for part in model_parts:
        state.update(get_model_state_dict(part, options=options))
    return state


def scatter_full_state(model_parts: list[nn.Module], state: dict[str, torch.Tensor]) -> None:
    """Load a full state dict back into the sharded model.

    `broadcast_from_rank0` means only rank 0 needs to hold the incoming copy;
    every other rank receives its shard over the process group. `strict=False`
    because with PP each part legitimately holds only its own subset of the keys.
    """
    options = StateDictOptions(
        full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True, strict=False
    )
    for part in model_parts:
        set_model_state_dict(part, model_state_dict=state, options=options)


def state_delta(
    current: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """``current - reference``, in float32.

    Not sent over the wire by default -- the FedMom strategy on the server derives
    its own pseudo-gradient from the parameters it receives -- but it is what
    `outer_agreement` measures, and what a delta-compressing transport would take.
    """
    return {
        key: current[key].to(torch.float32) - reference[key].to(torch.float32)
        for key in current
    }


def outer_agreement(
    delta: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> dict[str, float]:
    """How far this replica drifted during its inner phase.

    The quantity DiLoCo's H is actually tuned against: the ratio of the local
    update's norm to the weights' own norm. When it grows to the same order as the
    weights, replicas have diverged far enough that averaging them is destructive
    rather than helpful and H should come down. Logged every round so the choice of
    H is an observation rather than a guess.
    """
    delta_sq = 0.0
    ref_sq = 0.0
    for key, value in delta.items():
        delta_sq += float(value.pow(2).sum())
        ref_sq += float(reference[key].to(torch.float32).pow(2).sum())
    delta_norm = delta_sq**0.5
    ref_norm = ref_sq**0.5
    return {
        "delta_norm": delta_norm,
        "param_norm": ref_norm,
        "drift_ratio": delta_norm / ref_norm if ref_norm > 0 else 0.0,
    }
