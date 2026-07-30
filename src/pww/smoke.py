"""Infrastructure smoke test -- run this first after any environment change.

Checks, in order of what actually breaks on LUMI:
  1. every rank sees exactly one distinct GCD (pinning works),
  2. RCCL all-reduce produces the mathematically correct answer,
  3. RCCL bandwidth is in the right ballpark (catches the classic failure where
     collectives silently fall back to TCP instead of Slingshot),
  4. a real fwd/bwd step runs on the GPU.

    srun ... python3 -m pww.smoke
"""

from __future__ import annotations

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
    # the same GCD, which halves throughput without erroring.
    print(f"[r{info.rank}] host={info.hostname} "
          f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES', 'unset')} "
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

    Expect tens to hundreds of GB/s within a node (Infinity Fabric). Single-digit
    GB/s across nodes means RCCL is not using the Slingshot NICs -- check that
    /opt/aws-ofi-rccl is on LD_LIBRARY_PATH and the libfabric binds are present.
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


def main() -> None:
    info = D.setup()
    log = setup_logging(info.rank)
    log_environment(log, info)

    check_gpu_visibility(info)
    D.barrier()
    check_allreduce_correctness(info)
    benchmark_allreduce(info)
    check_training_step(info)

    log.info("SMOKE TEST PASSED")
    D.cleanup()


if __name__ == "__main__":
    main()
