"""Moving a sharded model's weights on and off the wire.

Under FSDP2 every parameter is a `DTensor` holding one shard, so "the model's
parameters" is not something any single rank has. `torch.distributed.checkpoint`'s
`get_model_state_dict` / `set_model_state_dict` are the supported way to gather and
scatter a full state dict across whatever parallelism is configured, and driving the
gather through them (rather than enumerating parameters and calling `.full_tensor()`
on each) is what keeps this working when tensor or pipeline parallelism is switched
on. They stay the mechanism; `as_plain_tensor` only unwraps what they return, because
on torch 2.9 `full_state_dict=True` can still hand back a `DTensor`.

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
)
from torch.distributed.tensor import DTensor

from ..logging_utils import get_logger

logger = get_logger("pww.titan.params")

# Three ways to cross the wire, all 2 or 4 bytes per parameter.
#
# "bfloat16" maps to np.uint16 because numpy has no bfloat16 and Flower serialises numpy
# arrays: the raw 16 bits travel as an integer and both ends **reinterpret** them
# (`.view`) rather than convert them (`.astype`). This is the compute dtype, so it is
# lossless here, and it keeps bfloat16's ~1e-38 range. Mixing up view and astype is the
# one dangerous thing about it -- 0.5 reads back as 16128.0 -- so both directions live in
# `encode`/`decode` and nowhere else, with the server's mirror in
# central/strategy.py::_from_wire.
#
# "float16" also halves WAN traffic and is exact for weights of normal magnitude (11
# mantissa bits against bfloat16's 8), but has far less range: float16's smallest
# subnormal is ~6e-08, so weights below that flush toward zero. Harmless -- such a weight
# contributes nothing to a forward pass -- but real, and pinned by a test rather than
# assumed away.
#
# "float32" keeps everything, at twice the bytes. Watch the 2 GiB gRPC cap.
WIRE_DTYPES = {"float16": np.float16, "float32": np.float32, "bfloat16": np.uint16}


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
                f"wire_dtype must be one of {sorted(WIRE_DTYPES)}, got {wire_dtype!r}"
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
            tensor = as_plain_tensor(state[key]).detach()
            if self.wire_dtype == "bfloat16":
                out.append(tensor.to(torch.bfloat16).cpu().view(torch.uint16).numpy())
            else:
                out.append(tensor.to(torch.float32).cpu().numpy().astype(target, copy=False))
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
            arr = np.ascontiguousarray(array)
            if self.wire_dtype == "bfloat16" or arr.dtype == np.uint16:
                state[key] = torch.from_numpy(arr).view(torch.bfloat16).to(dtype)
            else:
                state[key] = torch.from_numpy(arr).to(dtype)
        return state


def as_plain_tensor(value: torch.Tensor) -> torch.Tensor:
    """A plain tensor, whatever `get_model_state_dict` chose to hand back.

    `full_state_dict=True` is documented to return ordinary tensors, and on torch
    2.7 it did. On 2.9 some entries come back still wrapped as `DTensor`, and
    everything downstream assumes plain tensors: the codec calls `.numpy()`, the
    delta subtracts against a reference decoded from the wire, and
    `outer_agreement` reduces to Python floats. A single mixed pair fails the whole
    outer round with

        aten.sub.Tensor: got mixed torch.Tensor and DTensor

    which is unreachable from any CPU test, so it surfaces only after a queue wait.

    `full_tensor()` covers both shapes this can take. If the entry is already
    Replicate-placed -- `full_state_dict=True` did gather, it just did not unwrap --
    the redistribute is a no-op and no collective is issued. If it is still sharded,
    the collective runs. Either way it is called on every rank in sorted key order
    by the caller, so ranks cannot disagree about the order of any collective.
    """
    if isinstance(value, DTensor):
        return value.full_tensor()
    return value


def gather_full_state(model_parts: list[nn.Module]) -> dict[str, torch.Tensor]:
    """Full, unsharded, CPU-resident model state on every rank.

    `model_parts` is torchtitan's list because pipeline parallelism splits a model
    across ranks into several modules; without PP it has exactly one entry.
    """
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state: dict[str, torch.Tensor] = {}
    wrapped = 0
    for part in model_parts:
        part_state = get_model_state_dict(part, options=options)
        # Sorted, not insertion order: `as_plain_tensor` can issue a collective, and
        # every rank has to reach them in the same sequence. Same ordering contract
        # the codec relies on.
        for key in sorted(part_state):
            value = part_state[key]
            wrapped += isinstance(value, DTensor)
            state[key] = as_plain_tensor(value)
    if wrapped:
        logger.debug(
            "unwrapped %d/%d state entries that get_model_state_dict returned as "
            "DTensor despite full_state_dict=True",
            wrapped, len(state),
        )
    return state


def _report_dtensor_boundary(part: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Say which side of the DTensor boundary each parameter sits on, before the copy.

    Diagnostic only -- this converts nothing. `set_model_state_dict` raises

        aten.copy_.default: got mixed torch.Tensor and DTensor

    without naming which operand is which, and the message is symmetric, so the
    obvious reading (model sharded, incoming plain) may be backwards. On the Qwen3
    0.6B flavor it fails on exactly `norm.weight` and `output.weight` while every
    block parameter and `tok_embeddings.weight` load fine.

    The lead is that torchtitan rebinds the tied head *after* parallelising --
    parallelize.py, "Enable weight tying after applying parallelisms", live here
    because the 0.6B flavor sets enable_weight_tying -- so `output.weight` is not a
    parameter `apply_fsdp` grouped. That does not explain `norm.weight`, which is not
    tied and fails identically, so tying is at most half of it.

    Which direction the mismatch runs decides the fix, and guessing it risks loading
    weights into the wrong shards silently instead of raising. So record it.
    """
    owned: dict[str, torch.Tensor] = {
        **dict(part.named_parameters()),
        **dict(part.named_buffers()),
    }
    for key, incoming in state.items():
        mine = owned.get(key)
        if mine is None:
            continue
        if isinstance(mine, DTensor) == isinstance(incoming, DTensor):
            continue
        logger.warning(
            "scatter boundary mismatch %s: model=%s%s incoming=%s%s",
            key,
            type(mine).__name__,
            f" placements={mine.placements} mesh={tuple(mine.device_mesh.shape)}"
            if isinstance(mine, DTensor) else f" shape={tuple(mine.shape)}",
            type(incoming).__name__,
            f" placements={incoming.placements}"
            if isinstance(incoming, DTensor) else f" shape={tuple(incoming.shape)}",
        )


def _without_tied_aliases(
    part: nn.Module, state: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """`state` minus any key that is not a parameter this module actually owns.

    On the Qwen3 0.6B flavor `set_model_state_dict` raises
    ``aten.copy_.default: got mixed torch.Tensor and DTensor`` on exactly
    ``norm.weight`` and ``output.weight``, while every block parameter and
    ``tok_embeddings.weight`` load -- even though the DTensor/plain split is
    identical for all of them, so the type boundary is not what distinguishes them.

    What distinguishes them is the root parameter group. torchtitan ties the head
    *after* parallelising ("Enable weight tying after applying parallelisms", live
    because the 0.6B flavor sets ``enable_weight_tying``), rebinding
    ``output.weight`` to alias ``tok_embeddings.weight`` once ``apply_fsdp`` has
    already built its groups. The root group is
    ``{tok_embeddings.weight, norm.weight, output.weight}``: the real parameter
    loads, and the two remaining members of the mutated group do not.

    ``output.weight`` is absent from ``named_parameters()`` entirely, because that
    deduplicates shared parameters -- which is the alias showing through. Loading
    ``tok_embeddings.weight`` already writes those weights, so feeding the alias in
    as well is redundant as well as fatal.

    Anything dropped is logged rather than skipped quietly: silently not loading a
    parameter would leave a replica training from stale weights and show up only as
    a slow divergence, which is far worse than the exception this replaces.
    """
    owned = set(dict(part.named_parameters())) | set(dict(part.named_buffers()))
    dropped = [key for key in state if key not in owned]
    if not dropped:
        return state
    logger.info(
        "scatter: dropping %d key(s) this module does not own as distinct "
        "parameters: %s. Expected for a tied head -- named_parameters() "
        "deduplicates shared tensors, and the tying source carries the weights.",
        len(dropped), ", ".join(sorted(dropped)),
    )
    return {key: value for key, value in state.items() if key in owned}


def keys_to_load(part: nn.Module) -> list[str]:
    """The keys `scatter_full_state` should load, in a rank-independent order.

    Derived from the *module*, never from the incoming dict, because neither of the
    two obvious sources is correct:

    * `state` cannot decide it. Worker ranks are called with an empty dict and still
      have to reach every `broadcast` in the same order as rank 0, or the collective
      deadlocks or pairs up mismatched tensors.
    * `named_parameters() | named_buffers()` cannot decide it either, and this is what
      the code used to iterate. `named_buffers()` reports **non-persistent** buffers;
      `state_dict()` -- and therefore `get_model_state_dict`, the codec, and the wire --
      deliberately omits them. On the Qwen3 0.6B flavor that difference is exactly one
      tensor, `rope_cache`, registered `persistent=False` in Qwen3Model.__init__.

    That one tensor was the whole bug. 311 tensors cross the wire (28 layers x 11,
    plus tok_embeddings/norm/output), `rope_cache` is not among them, so iterating
    `owned` reached a key with nothing to load and took the "worker rank" branch:

        value = torch.empty(full_shape, dtype=param.dtype, device="cuda")
        dist.broadcast(value, src=0)      # no-op at world_size 1
        param.detach().copy_(value)       # <- uninitialised memory into rope_cache

    So every `set_parameters` overwrote the RoPE cos/sin table with whatever the
    caching allocator happened to hand back, on **every** rank including rank 0. When
    that block was freshly mapped it was zeros, which silently disables RoPE -- q and k
    are annihilated, attention goes uniform, and the model plateaus around the unigram
    entropy (~7.5 nats) while looking like it is training. When the block was dirty it
    was arbitrary float32 bit patterns, which is a nan on the first microbatch. Same
    root cause, two symptoms, and the zeros case is the more dangerous of the two
    because nothing reports it.

    `rope_cache` is a pure function of (head_dim, max_seq_len, rope_theta) and is built
    in the constructor and rebuilt by `init_weights`, so it must not be sent and must
    not be touched here.

    Tied aliases are dropped for the reason `_without_tied_aliases` documents: the 0.6B
    flavor ties `output.weight` to `tok_embeddings.weight` after parallelising, so it is
    absent from `named_parameters()` and loading the tying source already writes it.
    """
    owned = set(dict(part.named_parameters())) | set(dict(part.named_buffers()))
    return [key for key in sorted(part.state_dict()) if key in owned]


def scatter_full_state(model_parts: list[nn.Module], state: dict[str, torch.Tensor]) -> None:
    """Load a full state dict back into the sharded model.

    Rank 0 holds the full state dict (from Flower); worker ranks call this with
    an empty dict.  Each tensor is broadcast from rank 0, then every rank calls
    ``distribute_tensor`` to shard it and copies its local slice into the
    parameter.

    This replaces ``set_model_state_dict`` with ``broadcast_from_rank0``, which
    on torch 2.9 raises ``aten.copy_.default: got mixed torch.Tensor and
    DTensor`` when the incoming state is plain tensors and the model is
    FSDP2-sharded.

    Which keys get loaded is `keys_to_load`'s decision -- read its docstring before
    changing the iteration, because the obvious choices are both wrong.
    """
    import torch.distributed as dist
    from torch.distributed.tensor import distribute_tensor

    # Whether *this* rank holds the authoritative copy, asked of the process group
    # rather than inferred from `state` being non-empty. Inferring it is what let a
    # missing key silently become a receive buffer on the one rank that had the data.
    is_source = not dist.is_initialized() or dist.get_rank() == 0

    for part in model_parts:
        owned = dict(part.named_parameters())
        owned.update(dict(part.named_buffers()))
        expected = keys_to_load(part)
        if is_source:
            extra = sorted(set(state) - set(expected))
            if extra:
                logger.info(
                    "scatter: %d incoming key(s) not loaded directly: %s. Expected for "
                    "a tied head -- the tying source carries the weights.",
                    len(extra), ", ".join(extra),
                )
        for key in expected:
            param = owned[key]
            # DTensor.shape is the global (unsharded) shape.
            full_shape = tuple(param.shape)
            target_device = param.to_local().device if hasattr(param, "to_local") else param.device
            if is_source:
                incoming = state.get(key)
                if incoming is None:
                    # Loud, because the alternative is training on garbage. A key this
                    # module needs and the wire did not carry means the sender's codec
                    # and this model disagree, and there is no safe default value.
                    raise KeyError(
                        f"parameter {key!r} is required by this model but absent from "
                        f"the incoming state ({len(state)} keys). Refusing to fill it "
                        f"with uninitialised memory -- the sender is serialising a "
                        f"different model."
                    )
                value = incoming.to(dtype=param.dtype, device=target_device).contiguous()
            else:
                # Workers: allocate a receive buffer matching the full shape. Fully
                # overwritten by the broadcast below, so its contents do not matter.
                value = torch.empty(full_shape, dtype=param.dtype, device=target_device)
            # Broadcast the full tensor from rank 0 to all ranks (NCCL, on GPU).
            if dist.is_initialized():
                dist.broadcast(value, src=0)
            # Now every rank has the same full tensor — shard it.
            try:
                with torch.no_grad():
                    if hasattr(param, "to_local"):
                        if hasattr(param, "device_mesh"):
                            sharded = distribute_tensor(
                                value, param.device_mesh, param.placements
                            )
                            param.to_local().data.copy_(sharded.to_local().data)
                        else:
                            param.to_local().data.copy_(value.data)
                    else:
                        param.data.copy_(value.data)
            except Exception as e:
                logger.error(
                    "scatter failed for key=%s: param_type=%s, value_type=%s, value_device=%s, param_device=%s: %s",
                    key, type(param).__name__, type(value).__name__, value.device, getattr(param, "device", None), e
                )
                raise


def state_delta(
    current: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """``current - reference``, in float32.

    Not sent over the wire by default -- the FedMom strategy on the server derives
    its own pseudo-gradient from the parameters it receives -- but it is what
    `outer_agreement` measures, and what a delta-compressing transport would take.
    """
    return {
        key: as_plain_tensor(current[key]).to(torch.float32)
        - as_plain_tensor(reference[key]).to(torch.float32)
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
        delta_sq += float(as_plain_tensor(value).pow(2).sum())
        ref_sq += float(as_plain_tensor(reference[key]).to(torch.float32).pow(2).sum())
    delta_norm = delta_sq**0.5
    ref_norm = ref_sq**0.5
    return {
        "delta_norm": delta_norm,
        "param_norm": ref_norm,
        "drift_ratio": delta_norm / ref_norm if ref_norm > 0 else 0.0,
    }
