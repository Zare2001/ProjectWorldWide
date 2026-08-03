"""DiLoCo -- Distributed Low-Communication training.

Douillard et al., "DiLoCo: Distributed Low-Communication Training of Language
Models" (2023).

Plain data parallelism all-reduces gradients on every step, so it needs a fat,
low-latency link between every pair of ranks. DiLoCo replaces that with two
nested loops:

  inner   k model replicas train independently, each on its own shard of the
          data, for H steps (H >> 1 -- a few hundred) using InnerOpt. No
          communication between replicas at all during this phase.

  outer   every H steps each replica reports how far it travelled,
          Delta_i = theta(t-1) - theta_i(t). Those are averaged across replicas
          and handed to OuterOpt as if they were a gradient, producing the next
          shared theta(t), which is re-dispatched to every replica.

Inter-replica traffic therefore drops by a factor of H, which is what makes it
viable between replicas that are far apart -- different nodes, partitions, or
eventually different sites.

Rank layout
-----------
One allocation is carved into k contiguous equal blocks:

    world_size = 16, k = 4  ->  replica 0 = ranks 0-3
                                replica 1 = ranks 4-7
                                replica 2 = ranks 8-11
                                replica 3 = ranks 12-15

Contiguous on purpose: SLURM numbers ranks node by node, so a block stays inside
as few nodes as possible and the chatty inner all-reduce keeps the fast local
links.

Two families of process group fall out of that:

    inner   {0,1,2,3}, {4,5,6,7}, ...      gradient all-reduce (DDP/FSDP)
    outer   {0,4,8,12}, {1,5,9,13}, ...    outer-gradient all-reduce

The outer step is run redundantly by every rank instead of by one leader per
replica. Within a replica DDP already keeps parameters bit-identical, so every
rank computes the same Delta_i; pairing rank j of each replica with rank j of
the others turns one k-way exchange into ranks_per_replica independent k-way
exchanges that run in parallel, and leaves every rank already holding theta(t)
with no follow-up broadcast. The same layout is what makes FSDP work here: rank
j holds shard j of its replica, and it exchanges with the ranks holding shard j
of the other replicas.

Status
------
The DDP path is exercised on GPU by pww.smoke and against real process groups by
tests/test_diloco_gloo.py. The FSDP path is written to be sharding-aware -- see
_detect_shard_layout -- but is **not tested**: it assumes the outer step only ever
runs outside forward/backward, where FSDP presents parameters as their sharded
views. Validate that before trusting it for an LLM run.

Cost
----
Three extra full-precision copies of the parameters: theta (the shared model),
the outer momentum buffer, and one flat communication scratch buffer. Under FSDP
those are all sharded with the model, so it is 3x model bytes / ranks_per_replica
per rank. `outer_device="cpu"` moves two of the three to host memory.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

from .logging_utils import get_logger

OUTER_OPTIMIZERS = ("nesterov", "sgd", "adamw")

# Paper defaults (section 4): OuterOpt is Nesterov momentum with these values,
# which beat SGD, Adam and plain momentum in their sweep (Figure 6).
DEFAULT_OUTER_LR = 0.7
DEFAULT_OUTER_MOMENTUM = 0.9
DEFAULT_INNER_STEPS = 100


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the tensor this rank actually owns.

    FSDP hands out parameters as DTensors; the outer step operates on each
    rank's local shard, which is valid precisely because the outer groups pair
    equal shard indices (see the module docstring).
    """
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _detect_shard_layout(model: torch.nn.Module) -> str:
    """Whether each rank holds the whole model or a slice of it.

    Decides how theta(0) is equalised, and getting it wrong corrupts weights
    rather than merely slowing things down -- broadcasting one rank's FSDP shard
    to the whole replica would overwrite every other shard with a copy of the
    first. DTensor parameters are the general signal (FSDP1, fully_shard and
    tensor parallelism all produce them); the explicit FSDP check catches
    FSDP1 with use_orig_params=False, where parameters are flattened instead.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if any(hasattr(p, "to_local") for p in model.parameters()):
        return "sharded"
    if any(isinstance(m, FSDP) for m in model.modules()):
        return "sharded"
    return "replicated"


def _views_like(flat: torch.Tensor, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Carve `flat` into views with the same shapes as `tensors`, in order."""
    views, offset = [], 0
    for t in tensors:
        n = t.numel()
        views.append(flat[offset:offset + n].view_as(t))
        offset += n
    return views


@dataclass(frozen=True)
class Replicas:
    """How the flat rank space is carved into DiLoCo replicas."""

    rank: int
    world_size: int
    num_replicas: int          # k
    ranks_per_replica: int
    replica_id: int
    rank_in_replica: int
    # Process groups, or None when there is nothing to communicate (k == 1, or
    # no process group at all).
    inner_group: Any = field(default=None, compare=False)
    outer_group: Any = field(default=None, compare=False)

    @property
    def is_replica_master(self) -> bool:
        return self.rank_in_replica == 0

    def describe(self) -> str:
        return (
            f"diloco k={self.num_replicas} x {self.ranks_per_replica} ranks | "
            f"this rank: replica {self.replica_id}, slot {self.rank_in_replica}"
        )


def build_replicas(
    num_replicas: int,
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> Replicas:
    """Split the world into `num_replicas` blocks and create the process groups.

    Safe to call without a process group (returns a layout with no groups), which
    is what makes the whole module testable on a login node.
    """
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", 0))
    if world_size is None:
        world_size = (
            dist.get_world_size() if dist.is_initialized()
            else int(os.environ.get("WORLD_SIZE", 1))
        )

    if num_replicas < 1:
        raise ValueError(f"num_replicas must be >= 1, got {num_replicas}")
    if num_replicas > world_size:
        raise ValueError(
            f"num_replicas={num_replicas} exceeds world_size={world_size}; "
            f"each replica needs at least one rank"
        )
    if world_size % num_replicas:
        raise ValueError(
            f"world_size={world_size} is not divisible by num_replicas={num_replicas}. "
            f"Replicas must be equal-sized, so pick k from "
            f"{sorted(d for d in range(1, world_size + 1) if world_size % d == 0)}"
        )

    ranks_per_replica = world_size // num_replicas
    replica_id, rank_in_replica = divmod(rank, ranks_per_replica)

    inner_group = outer_group = None
    if dist.is_initialized() and num_replicas > 1:
        # new_group is collective: every rank must enter every call, in the same
        # order, including for groups it will not be a member of.
        for r in range(num_replicas):
            group = dist.new_group(ranks=list(range(r * ranks_per_replica,
                                                    (r + 1) * ranks_per_replica)))
            if r == replica_id:
                inner_group = group
        for slot in range(ranks_per_replica):
            group = dist.new_group(ranks=list(range(slot, world_size, ranks_per_replica)))
            if slot == rank_in_replica:
                outer_group = group

    return Replicas(
        rank=rank,
        world_size=world_size,
        num_replicas=num_replicas,
        ranks_per_replica=ranks_per_replica,
        replica_id=replica_id,
        rank_in_replica=rank_in_replica,
        inner_group=inner_group,
        outer_group=outer_group,
    )


def _build_outer_optimizer(name: str, param: torch.Tensor, lr: float, momentum: float):
    if name == "nesterov":
        # Nesterov needs momentum > 0; PyTorch raises otherwise, but the message
        # does not mention which of the two knobs to change.
        if momentum <= 0:
            raise ValueError("outer_optimizer='nesterov' requires outer_momentum > 0")
        return torch.optim.SGD([param], lr=lr, momentum=momentum, nesterov=True)
    if name == "sgd":
        # lr=1, momentum=0 makes the outer step theta(t) = mean_i theta_i(t),
        # i.e. FederatedAveraging. Useful as a baseline and as a test oracle.
        return torch.optim.SGD([param], lr=lr, momentum=momentum)
    if name == "adamw":
        # weight_decay=0: this optimizer acts on outer gradients, and decaying
        # theta here would be a second, uncontrolled regulariser on top of the
        # inner optimizer's own weight decay.
        return torch.optim.AdamW([param], lr=lr, weight_decay=0.0)
    raise ValueError(f"unknown outer optimizer {name!r}, expected one of {OUTER_OPTIMIZERS}")


class DiLoCo:
    """The outer loop: owns theta, the outer optimizer, and the H-step schedule.

    Usage inside a training loop:

        diloco = DiLoCo(model, replicas, inner_steps=100)
        for batch in loader:
            ...                       # inner optimizer step as usual
            diloco.inner_step()       # may trigger an outer step
        diloco.finish()               # flush a partial inner phase

    Every rank must call `inner_step()` the same number of times, because the
    outer step it triggers is collective.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        replicas: Replicas,
        *,
        inner_steps: int = DEFAULT_INNER_STEPS,
        outer_lr: float = DEFAULT_OUTER_LR,
        outer_momentum: float = DEFAULT_OUTER_MOMENTUM,
        outer_optimizer: str = "nesterov",
        outer_device: str | torch.device | None = None,
        sync_buffers: bool = True,
        shard_layout: str = "auto",
    ):
        if inner_steps < 1:
            raise ValueError(f"inner_steps (H) must be >= 1, got {inner_steps}")
        if shard_layout not in ("auto", "replicated", "sharded"):
            raise ValueError(f"unknown shard_layout {shard_layout!r}")

        self.replicas = replicas
        self.shard_layout = (
            _detect_shard_layout(model) if shard_layout == "auto" else shard_layout
        )
        self.inner_steps = int(inner_steps)
        self.outer_optimizer_name = outer_optimizer
        self.inner_counter = 0      # inner steps since the last outer step
        self.total_inner_steps = 0
        self.outer_steps = 0        # t

        self._params = [_local(p) for p in model.parameters() if p.requires_grad]
        if not self._params:
            raise ValueError("model has no trainable parameters")

        # Non-float buffers (BatchNorm's num_batches_tracked) are counters, not
        # state worth averaging, so they are left alone.
        self._buffers = (
            [_local(b) for b in model.buffers() if b.is_floating_point()]
            if sync_buffers else []
        )

        comm_device = self._params[0].device
        outer_device = torch.device(outer_device) if outer_device is not None else comm_device
        self.comm_device = comm_device
        self.outer_device = outer_device

        # One flat fp32 buffer per role. Flat because a 1-parameter optimizer and
        # a single all-reduce are both dramatically cheaper than one per tensor:
        # a transformer has thousands of parameters, and per-tensor collectives
        # would make the outer step latency-bound.
        numel = sum(p.numel() for p in self._params)
        self._flat = torch.zeros(numel, dtype=torch.float32, device=comm_device)
        self._flat_views = _views_like(self._flat, self._params)
        self._global = torch.zeros(numel, dtype=torch.float32, device=outer_device)
        self._global_views = _views_like(self._global, self._params)
        self._global.grad = torch.zeros_like(self._global)

        self._buf_flat = self._buf_views = None
        self._global_buffers = self._global_buffer_views = None
        buffer_numel = sum(b.numel() for b in self._buffers)
        if buffer_numel:
            self._buf_flat = torch.zeros(buffer_numel, dtype=torch.float32, device=comm_device)
            self._buf_views = _views_like(self._buf_flat, self._buffers)
            self._global_buffers = torch.zeros(buffer_numel, dtype=torch.float32,
                                               device=outer_device)
            self._global_buffer_views = _views_like(self._global_buffers, self._buffers)

        self._opt = _build_outer_optimizer(outer_optimizer, self._global,
                                           outer_lr, outer_momentum)

        self.sync_from_model()

    # --- setup ------------------------------------------------------------

    @torch.no_grad()
    def sync_from_model(self) -> None:
        """Adopt the model's current parameters as theta, equal across replicas.

        Called at construction, and again after loading a checkpoint so that
        theta and the model do not disagree.

        The broadcast is not optional at construction. DDP equalises parameters
        within its own process group, so with replica-scoped groups each replica
        would otherwise start from a different random initialisation -- and
        averaging deltas between models that were never the same model is
        meaningless. config.set_seed offsets the seed by rank, which guarantees
        that divergence rather than merely allowing it.
        """
        group, src = self._broadcast_route()
        self._copy_local_into(self._flat_views, self._params)
        if src is not None:
            dist.broadcast(self._flat, src=src, group=group)
            self._copy_into_local(self._params, self._flat_views)
        self._global.copy_(self._flat)

        if self._buf_flat is not None:
            self._copy_local_into(self._buf_views, self._buffers)
            if src is not None:
                dist.broadcast(self._buf_flat, src=src, group=group)
                self._copy_into_local(self._buffers, self._buf_views)
            self._global_buffers.copy_(self._buf_flat)

    def _broadcast_route(self):
        """(group, src) for equalising theta, or (None, None) if there is nothing to do.

        Replicated parameters go out from global rank 0 over the whole world:
        every rank holds a full copy, so one source is enough and it does not
        matter whether DDP has already equalised within each replica.

        Sharded parameters instead go out over the outer group from replica 0's
        rank in the *same slot*, because rank j owns shard j and must only ever
        receive shard j. FSDP has already made the shards consistent within the
        replica, so only the cross-replica direction is left.
        """
        if not dist.is_initialized():
            return None, None
        if self.shard_layout == "sharded":
            if self.replicas.outer_group is None:
                return None, None
            return self.replicas.outer_group, self.replicas.rank_in_replica
        if self.replicas.world_size <= 1:
            return None, None
        return None, 0      # None group == the default group == the whole world

    # --- the schedule -----------------------------------------------------

    def inner_step(self) -> dict[str, float] | None:
        """Count one completed inner step; run the outer step on every H-th.

        Returns the outer step's diagnostics, or None if this was not a
        boundary.
        """
        self.inner_counter += 1
        self.total_inner_steps += 1
        if self.inner_counter < self.inner_steps:
            return None
        return self.outer_step()

    def finish(self) -> dict[str, float] | None:
        """Flush a partial inner phase at the end of training.

        Without this, a run whose step count is not a multiple of H ends with
        each replica sitting on its own drifted parameters and theta stale by up
        to H steps, so the final checkpoint would not be the model you trained.
        """
        if self.inner_counter == 0:
            return None
        return self.outer_step()

    @torch.no_grad()
    def outer_step(self) -> dict[str, float]:
        """One outer optimization step (Algorithm 1, lines 12 and 14)."""
        steps_taken = self.inner_counter
        self.inner_counter = 0

        # Delta_i = theta(t-1) - theta_i(t)
        self._flat.copy_(self._global)
        for view, param in zip(self._flat_views, self._params):
            view.sub_(param)
        local_norm = self._flat.norm().item()

        # Average over replicas. Every rank does this against its own slot in the
        # other replicas, so all ranks_per_replica exchanges proceed in parallel.
        if self.replicas.outer_group is not None:
            dist.all_reduce(self._flat, op=dist.ReduceOp.SUM,
                            group=self.replicas.outer_group)
            self._flat.div_(self.replicas.num_replicas)
        averaged_norm = self._flat.norm().item()

        # OuterOpt(theta(t-1), Delta). Delta enters as a gradient, so descending
        # it moves theta in the direction the replicas actually travelled: with
        # lr=1 and no momentum, theta(t) = theta(t-1) - Delta = mean_i theta_i(t).
        self._global.grad.copy_(self._flat)
        self._opt.step()
        self.outer_steps += 1

        # Re-dispatch theta(t) to every replica (line 3 of the next iteration).
        self._flat.copy_(self._global)
        self._copy_into_local(self._params, self._flat_views)

        if self._buf_flat is not None:
            self._copy_local_into(self._buf_views, self._buffers)
            if self.replicas.outer_group is not None:
                dist.all_reduce(self._buf_flat, op=dist.ReduceOp.SUM,
                                group=self.replicas.outer_group)
                self._buf_flat.div_(self.replicas.num_replicas)
            self._copy_into_local(self._buffers, self._buf_views)
            self._global_buffers.copy_(self._buf_flat)

        return {
            "outer_step": float(self.outer_steps),
            "inner_steps": float(steps_taken),
            "delta_norm": local_norm,
            "avg_delta_norm": averaged_norm,
            # 1.0 means the replicas moved in identical directions; ~1/sqrt(k)
            # means they were mutually orthogonal, i.e. the averaging is
            # cancelling most of the progress and H is probably too large.
            "agreement": averaged_norm / local_norm if local_norm > 0 else 0.0,
        }

    # --- reading theta ----------------------------------------------------

    @contextlib.contextmanager
    def global_model(self):
        """Temporarily swap theta(t) into the model.

        Mid-inner-phase the model holds replica-local parameters, which differ
        per replica -- so evaluating or checkpointing there gives you one
        arbitrary replica, and a metric reduced over the world would be an
        average of k different models. theta(t) is identical on every rank, so
        anything measured inside this block is a single well-defined number. It
        lags by up to H steps, which is the correct thing to report for DiLoCo.

        Uses the communication scratch buffer to stash the local parameters, so
        it costs no extra memory. Do not call it inside an outer step.
        """
        stash_buffers = None
        with torch.no_grad():
            self._copy_local_into(self._flat_views, self._params)
            self._copy_into_local(self._params, self._global_views)
            if self._global_buffer_views is not None:
                stash_buffers = [b.detach().clone() for b in self._buffers]
                self._copy_into_local(self._buffers, self._global_buffer_views)
        try:
            yield
        finally:
            with torch.no_grad():
                self._copy_into_local(self._params, self._flat_views)
                if stash_buffers is not None:
                    self._copy_into_local(self._buffers, stash_buffers)

    # --- checkpointing ----------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Outer-loop state: the momentum buffer and the step counters.

        theta itself is deliberately absent: callers checkpoint the model inside
        `global_model()`, so the saved weights already *are* theta and storing
        them twice would only create a way for the two to disagree.
        """
        return {
            "outer_optim": self._opt.state_dict(),
            "outer_optimizer_name": self.outer_optimizer_name,
            "outer_steps": self.outer_steps,
            "total_inner_steps": self.total_inner_steps,
            "inner_counter": self.inner_counter,
            "numel": self._global.numel(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the outer momentum. Refuses a state that cannot belong here.

        Dropping the momentum instead would silently restart the outer optimizer
        cold on resume, which costs several outer steps of progress at
        momentum 0.9 -- a slow, hard-to-attribute regression.
        """
        if state.get("numel") != self._global.numel():
            raise ValueError(
                f"outer state has {state.get('numel')} elements but this rank owns "
                f"{self._global.numel()}; the parallelism or world size changed"
            )
        if state.get("outer_optimizer_name") != self.outer_optimizer_name:
            raise ValueError(
                f"outer state was written by outer_optimizer="
                f"{state.get('outer_optimizer_name')!r}, but this run uses "
                f"{self.outer_optimizer_name!r}"
            )
        self._opt.load_state_dict(state["outer_optim"])
        self.outer_steps = int(state.get("outer_steps", 0))
        self.total_inner_steps = int(state.get("total_inner_steps", 0))
        self.inner_counter = int(state.get("inner_counter", 0))

    def save(self, path) -> None:
        """Write the outer state, atomically, next to the model checkpoint."""
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), tmp)
        tmp.replace(path)

    def load(self, path) -> None:
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"outer state not found: {path}")
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        get_logger().info("restored DiLoCo outer state <- %s (t=%d)", path, self.outer_steps)

    # --- internals --------------------------------------------------------

    @staticmethod
    def _copy_local_into(views: list[torch.Tensor], tensors: list[torch.Tensor]) -> None:
        for view, tensor in zip(views, tensors):
            view.copy_(tensor)

    @staticmethod
    def _copy_into_local(tensors: list[torch.Tensor], views: list[torch.Tensor]) -> None:
        for tensor, view in zip(tensors, views):
            tensor.copy_(view)


def describe_plan(replicas: Replicas, inner_steps: int, total_inner_steps: int | None) -> str:
    """One-line summary for the top of a run log."""
    outer = "unknown" if not total_inner_steps else str(total_inner_steps // inner_steps)
    return (
        f"{replicas.describe()} | H={inner_steps} inner steps per outer step | "
        f"T~{outer} outer steps"
    )
