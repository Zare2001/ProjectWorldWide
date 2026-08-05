"""Distributed process-group setup and rank-aware helpers.

Everything here works unchanged for 1 GPU, 8 GPUs on one node, and N nodes,
because the launcher (scripts/task_wrapper.sh) always presents the same
contract: RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT are set,
and each process sees exactly one GCD.
"""

from __future__ import annotations

import contextlib
import datetime as _datetime
import os
import socket
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    """Resolved topology for the current process."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str
    hostname: str

    @property
    def is_master(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def describe(self) -> str:
        return (
            f"rank={self.rank}/{self.world_size} local_rank={self.local_rank} "
            f"host={self.hostname} device={self.device} backend={self.backend}"
        )


def _resolve_device(local_rank: int) -> torch.device:
    """Pick this process's device.

    task_wrapper.sh pins each rank to one device via whichever variable the site
    uses -- ROCR_VISIBLE_DEVICES on ROCm, CUDA_VISIBLE_DEVICES on NVIDIA -- so a
    pinned rank sees a single GCD or GPU that it must address as cuda:0. When
    running unpinned (e.g. a bare `python3` on a login/interactive node) fall
    back to indexing by local rank.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    index = local_rank if torch.cuda.device_count() > 1 else 0
    torch.cuda.set_device(index)
    return torch.device(f"cuda:{index}")


def setup(timeout_minutes: int = 30) -> DistInfo:
    """Initialise the process group and return the resolved topology."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = _resolve_device(local_rank)
    backend = "nccl" if device.type == "cuda" else "gloo"

    if world_size > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            # Bind the rank to its device explicitly. Without this, RCCL infers
            # the mapping lazily and warns that a wrong guess can hang the job;
            # it also enables eager connection setup.
            device_id=device if device.type == "cuda" else None,
            # The default 10 min can be too tight while thousands of ranks page
            # a dataset in off Lustre for the first time.
            timeout=_datetime.timedelta(minutes=timeout_minutes),
        )

    return DistInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
        hostname=socket.gethostname(),
    )


def cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_leader() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", 0)) == 0


def world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    """Average a scalar across ranks. Returns `value` unchanged if not distributed."""
    if not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()


def all_reduce_sum(value: float, device: torch.device) -> float:
    """Sum a scalar across ranks. Needed to aggregate per-rank counts correctly."""
    if not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def all_reduce_avg_(tensors: torch.Tensor | list[torch.Tensor]) -> None:
    """Average tensor(s) in-place across all distributed ranks in the process group."""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    world_size = float(dist.get_world_size())
    for tensor in tensors:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor.div_(world_size)


@contextlib.contextmanager
def master_first():
    """Let rank 0 run the block before anyone else.

    Use for one-time side effects that must not race: creating directories,
    downloading or building a dataset cache, writing a tokenizer.
    """
    is_master = int(os.environ.get("RANK", 0)) == 0
    if not is_master:
        barrier()
    try:
        yield is_master
    finally:
        if is_master:
            barrier()
