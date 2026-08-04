"""Infrastructure smoke test -- run this first after any environment change.

Checks, in order of what actually breaks:
  1. every rank sees exactly one distinct GCD/GPU (pinning works),
  2. the all-reduce produces the mathematically correct answer,
  3. collective bandwidth is in the right ballpark (catches the classic failure
     where RCCL/NCCL silently falls back to TCP instead of Slingshot or
     InfiniBand),
  4. a real fwd/bwd step runs on the GPU.

With --diloco-replicas k it additionally verifies the DiLoCo rank layout, both
families of process group, and one outer step against a known answer, then
measures what an outer step costs relative to an inner one.

    srun ... python3 -m pww.smoke
    srun ... python3 -m pww.smoke --diloco-replicas 2
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

from . import distributed as D
from .logging_utils import get_logger, log_environment, setup_logging


def check_gpu_visibility(info) -> bool:
    log = get_logger()
    if info.device.type != "cuda":
        log.warning("no GPU visible -- running on CPU")
        return False

    # Each rank prints its own line: the fastest way to spot two ranks pinned to
    # the same device, which halves throughput without erroring. The variable
    # that does the pinning differs per site (ROCR_ on ROCm, CUDA_ on NVIDIA),
    # so read its name from the environment rather than assuming either.
    pin_var = os.environ.get("PWW_GPU_VISIBLE_VAR", "CUDA_VISIBLE_DEVICES")
    print(f"[r{info.rank}] host={info.hostname} "
          f"{pin_var}={os.environ.get(pin_var, 'unset')} "
          f"device_count={torch.cuda.device_count()} "
          f"name={torch.cuda.get_device_name(info.device)}", flush=True)
    return True


def check_allreduce_correctness(info) -> None:
    """Sum of rank ids must equal world_size*(world_size-1)/2."""
    log = get_logger()
    if not info.is_distributed:
        log.info("single rank -- skipping collective check")
        return

    t = torch.full((16,), float(info.rank), device=info.device)
    dist.all_reduce(t)
    expected = info.world_size * (info.world_size - 1) / 2
    got = t[0].item()
    if abs(got - expected) > 1e-3:
        raise RuntimeError(f"all_reduce wrong: got {got}, expected {expected}")
    log.info("all_reduce correct (sum of ranks = %.0f)", got)


def benchmark_allreduce(info, size_mb: int = 256, iters: int = 20) -> None:
    """Measure all-reduce bandwidth.

    Expect tens to hundreds of GB/s within a node -- measured 123 GB/s on LUMI
    (Infinity Fabric) and 300.8 GB/s on a Snellius H100 node. Single-digit GB/s
    across nodes means the collective is going over TCP rather than the fast
    fabric: on LUMI check that /opt/aws-ofi-rccl is on LD_LIBRARY_PATH and the
    libfabric binds are present; on Snellius set NCCL_SOCKET_IFNAME to the
    InfiniBand interface (ibp*/mlx5, not ib0).
    """
    log = get_logger()
    if not info.is_distributed:
        return

    numel = size_mb * 1024 * 1024 // 4
    t = torch.ones(numel, dtype=torch.float32, device=info.device)

    for _ in range(5):                      # warm up: first call pays setup cost
        dist.all_reduce(t)
    if info.device.type == "cuda":
        torch.cuda.synchronize()

    dist.barrier()
    start = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(t)
    if info.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Ring all-reduce moves ~2*(N-1)/N * size bytes per rank.
    factor = 2 * (info.world_size - 1) / info.world_size
    gb = size_mb / 1024 * factor * iters
    log.info("all_reduce %d MiB: %.1f ms/iter, %.1f GB/s effective bus bandwidth",
             size_mb, elapsed / iters * 1e3, gb / elapsed)


def check_training_step(info) -> None:
    """One real fwd/bwd on the actual model, wrapped as it will be in training."""
    from .models.resnet import resnet18
    from .parallel import wrap_model

    log = get_logger()
    strategy = "ddp" if info.is_distributed else "single"
    model = wrap_model(resnet18(), strategy=strategy, device=info.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    x = torch.randn(32, 3, 32, 32, device=info.device)
    y = torch.randint(0, 10, (32,), device=info.device)

    t0 = time.perf_counter()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    if info.device.type == "cuda":
        torch.cuda.synchronize()

    log.info("3 training steps ok (%s), final loss %.4f, %.0f ms/step",
             strategy, loss.item(), (time.perf_counter() - t0) / 3 * 1e3)
    if info.device.type == "cuda":
        log.info("peak GPU memory: %.2f GiB", torch.cuda.max_memory_allocated(info.device) / 1024**3)


class _OneParam(torch.nn.Module):
    """Smallest possible model whose outer step has an answer you can do by hand."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(8, device=device))


def check_diloco(info, num_replicas: int) -> None:
    """Verify the DiLoCo layout, its two group families, and one outer step.

    The outer step is checked against an exact answer rather than a tolerance on
    a training curve: with OuterOpt = SGD(lr=1, momentum=0) the update collapses
    to theta(t) = mean_i theta_i(t), so seeding replica i with the constant i
    must produce exactly (k-1)/2 on every rank. A wrong group layout, a missed
    division by k, or a sign error all fail this loudly.
    """
    from .diloco import DiLoCo, build_replicas

    log = get_logger()
    replicas = build_replicas(num_replicas)
    rpr = replicas.ranks_per_replica
    print(f"[r{info.rank}] {replicas.describe()}", flush=True)

    if replicas.inner_group is not None:
        # Sum of the global rank ids inside this replica.
        t = torch.full((4,), float(info.rank), device=info.device)
        dist.all_reduce(t, group=replicas.inner_group)
        expected = sum(range(replicas.replica_id * rpr, (replicas.replica_id + 1) * rpr))
        if abs(t[0].item() - expected) > 1e-3:
            raise RuntimeError(f"inner group wrong on rank {info.rank}: "
                               f"got {t[0].item()}, expected {expected}")

        t = torch.full((4,), float(info.rank), device=info.device)
        dist.all_reduce(t, group=replicas.outer_group)
        expected = sum(range(replicas.rank_in_replica, info.world_size, rpr))
        if abs(t[0].item() - expected) > 1e-3:
            raise RuntimeError(f"outer group wrong on rank {info.rank}: "
                               f"got {t[0].item()}, expected {expected}")
        log.info("inner and outer process groups have the expected membership")

    model = _OneParam(info.device)
    dl = DiLoCo(model, replicas, inner_steps=1, outer_optimizer="sgd",
                outer_lr=1.0, outer_momentum=0.0, sync_buffers=False)
    with torch.no_grad():
        model.w.fill_(float(replicas.replica_id))     # stand in for H inner steps
    dl.outer_step()
    expected = (num_replicas - 1) / 2.0
    got = model.w[0].item()
    if abs(got - expected) > 1e-6:
        raise RuntimeError(f"outer step wrong on rank {info.rank}: got {got}, "
                           f"expected mean of 0..{num_replicas - 1} = {expected}")
    log.info("outer step matches FederatedAveraging exactly (%.4f)", got)


def benchmark_diloco_outer(info, num_replicas: int, iters: int = 10) -> None:
    """Time an outer step against an inner step on the real model.

    This is the number that decides whether DiLoCo is worth it: the outer step
    costs roughly one all-reduce of the whole model, paid once every H steps, so
    the overhead is (outer_ms / inner_ms) / H.
    """
    from .diloco import DiLoCo, build_replicas
    from .models.resnet import resnet18
    from .parallel import wrap_model

    log = get_logger()
    replicas = build_replicas(num_replicas)
    strategy = "ddp" if replicas.ranks_per_replica > 1 else "single"
    model = wrap_model(resnet18(), strategy=strategy, device=info.device,
                       process_group=replicas.inner_group)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # inner_steps large enough that inner_step() never triggers on its own.
    dl = DiLoCo(model, replicas, inner_steps=10**9)

    x = torch.randn(32, 3, 32, 32, device=info.device)
    y = torch.randint(0, 10, (32,), device=info.device)

    def inner():
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()

    for _ in range(3):
        inner()
        dl.outer_step()
    if info.device.type == "cuda":
        torch.cuda.synchronize()

    D.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        inner()
    if info.device.type == "cuda":
        torch.cuda.synchronize()
    inner_ms = (time.perf_counter() - t0) / iters * 1e3

    D.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        dl.outer_step()
    if info.device.type == "cuda":
        torch.cuda.synchronize()
    outer_ms = (time.perf_counter() - t0) / iters * 1e3

    log.info("inner step %.1f ms | outer step %.1f ms (ratio %.1fx)",
             inner_ms, outer_ms, outer_ms / max(inner_ms, 1e-9))
    for h in (10, 100, 500):
        log.info("  H=%-4d -> %.2f%% outer overhead", h, 100.0 * outer_ms / inner_ms / h)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ProjectWorldWide infrastructure smoke test")
    p.add_argument("--diloco-replicas", type=int, default=0,
                   help="Also check the DiLoCo layout with k replicas (must divide "
                        "the world size). 0 skips those checks")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    info = D.setup()
    log = setup_logging(info.rank)
    log_environment(log, info)

    check_gpu_visibility(info)
    D.barrier()
    check_allreduce_correctness(info)
    benchmark_allreduce(info)
    check_training_step(info)

    if args.diloco_replicas > 0:
        log.info("-" * 72)
        check_diloco(info, args.diloco_replicas)
        benchmark_diloco_outer(info, args.diloco_replicas)

    log.info("SMOKE TEST PASSED")
    D.cleanup()


if __name__ == "__main__":
    main()
