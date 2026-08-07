"""DiLoCo correctness with real process groups -- CPU only, no allocation needed.

tests/test_local.py pins down the outer-step arithmetic in one process. This file
covers the part that only exists once there are several: that the inner and outer
groups have the membership the layout claims, that the outer gradient really is
averaged across replicas, and that replicas start from one theta(0) and return to
one theta(t) at every outer step.

It runs gloo over localhost, so it belongs on a login node next to test_local.py
rather than in a queued job. pww.smoke covers the same ground on real GPUs with
RCCL/NCCL, where the failure modes are different.

    source env.sh && pww_run python3 tests/test_diloco_gloo.py
    source env.sh && pww_run python3 tests/test_diloco_gloo.py --world-size 6 --replicas 3

pww_run enters the container on LUMI and is a no-op inside the venv on Snellius,
so the same line works on both. Last measured 14/14 on Snellius.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pww.diloco import DiLoCo, build_replicas  # noqa: E402
from pww.parallel import wrap_model  # noqa: E402

OUTER_LR = 0.7
OUTER_MOMENTUM = 0.9


def _tiny_model(seed: int) -> torch.nn.Module:
    """Deliberately seeded per rank, so a missing theta(0) broadcast is visible."""
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(6, 5), torch.nn.BatchNorm1d(5), torch.nn.ReLU(),
        torch.nn.Linear(5, 3),
    )


def _flat_params(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _spread(tensor: torch.Tensor, group=None) -> float:
    """Largest elementwise disagreement about `tensor` across a group."""
    high = tensor.clone()
    low = tensor.clone()
    dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
    return (high - low).abs().max().item()


def _fill(model: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(value)


# --- checks -----------------------------------------------------------------
# Each takes (replicas, log) and raises on failure. Every rank runs every check,
# so a failure is symmetric and cannot deadlock the others.


def check_group_membership(replicas, log) -> None:
    rank, rpr = replicas.rank, replicas.ranks_per_replica

    t = torch.full((3,), float(rank))
    dist.all_reduce(t, group=replicas.inner_group)
    want = sum(range(replicas.replica_id * rpr, (replicas.replica_id + 1) * rpr))
    assert abs(t[0].item() - want) < 1e-6, f"inner group: got {t[0].item()}, want {want}"

    t = torch.full((3,), float(rank))
    dist.all_reduce(t, group=replicas.outer_group)
    want = sum(range(replicas.rank_in_replica, replicas.world_size, rpr))
    assert abs(t[0].item() - want) < 1e-6, f"outer group: got {t[0].item()}, want {want}"
    log("inner and outer groups have the expected membership")


def check_theta0_is_shared(replicas, log) -> None:
    """Replicas must be forced onto one theta(0) despite per-rank seeds."""
    model = _tiny_model(seed=100 + replicas.rank)
    before = _spread(_flat_params(model))
    assert before > 1e-3, "test is vacuous: the per-rank seeds produced equal weights"

    DiLoCo(model, replicas, inner_steps=10)
    after = _spread(_flat_params(model))
    assert after == 0.0, f"theta(0) still differs across ranks by {after}"
    log(f"theta(0) aligned across all ranks (spread {before:.4f} -> 0)")


def check_federated_averaging(replicas, log) -> None:
    """OuterOpt = SGD(lr=1, momentum=0) must give exactly the mean of the replicas."""
    model = _tiny_model(seed=0)
    # Zero the weights *before* constructing DiLoCo, so theta(0) is exactly 0 and
    # the expected answer is a plain constant.
    _fill(model, 0.0)
    dl = DiLoCo(model, replicas, inner_steps=1, outer_optimizer="sgd",
                outer_lr=1.0, outer_momentum=0.0, sync_buffers=False)

    _fill(model, float(replicas.replica_id + 1))
    dl.outer_step()

    want = (replicas.num_replicas + 1) / 2.0        # mean of 1..k
    got = _flat_params(model)
    assert torch.allclose(got, torch.full_like(got, want), atol=1e-6), \
        f"got {got[0].item()}, want mean of 1..{replicas.num_replicas} = {want}"
    assert _spread(got) == 0.0, "ranks disagree about theta after the outer step"
    log(f"outer step averages exactly over {replicas.num_replicas} replicas ({want})")


def check_nesterov_against_reference(replicas, log) -> None:
    """Two outer steps against a reference worked out independently of the code.

    Momentum is what makes DiLoCo's outer loop more than FederatedAveraging, so
    it needs a check that would fail if the buffer were dropped or reused wrong.
    """
    k = replicas.num_replicas
    model = _tiny_model(seed=0)
    _fill(model, 0.0)        # theta(0) = 0 exactly, and no momentum yet
    dl = DiLoCo(model, replicas, inner_steps=1, outer_optimizer="nesterov",
                outer_lr=OUTER_LR, outer_momentum=OUTER_MOMENTUM, sync_buffers=False)
    theta = 0.0
    buf = 0.0

    for phase, offset in enumerate((1.0, 10.0)):
        local = replicas.replica_id + offset
        _fill(model, float(local))
        dl.outer_step()

        # Reference: mean over replicas of (theta - local), then PyTorch's
        # nesterov update  buf <- m*buf + d ;  theta <- theta - lr*(d + m*buf).
        mean_local = offset + (k - 1) / 2.0
        delta = theta - mean_local
        buf = OUTER_MOMENTUM * buf + delta
        theta = theta - OUTER_LR * (delta + OUTER_MOMENTUM * buf)

        got = _flat_params(model)
        assert torch.allclose(got, torch.full_like(got, theta), atol=1e-5), \
            f"outer step {phase + 1}: got {got[0].item():.6f}, want {theta:.6f}"
    log(f"two nesterov outer steps match the hand-computed reference ({theta:.4f})")


def check_buffers_are_averaged(replicas, log) -> None:
    model = _tiny_model(seed=0)
    dl = DiLoCo(model, replicas, inner_steps=1, outer_optimizer="sgd",
                outer_lr=1.0, outer_momentum=0.0)
    with torch.no_grad():
        for b in model.buffers():
            if b.is_floating_point():
                b.fill_(float(replicas.replica_id + 1))
    dl.outer_step()

    want = (replicas.num_replicas + 1) / 2.0
    for b in model.buffers():
        if b.is_floating_point():
            assert torch.allclose(b, torch.full_like(b, want), atol=1e-6), \
                f"buffer not averaged: {b.flatten()[0].item()} != {want}"
    log(f"float buffers averaged across replicas ({want})")


def check_ddp_training_converges_to_one_theta(replicas, log) -> None:
    """The whole loop end to end: DDP inside a replica, DiLoCo between replicas.

    Asserts the two properties that define a correct DiLoCo run:
      * during the inner phase, ranks agree within a replica and disagree across
        replicas -- if they agreed across replicas, DDP would be scoped to the
        world and there would be no low-communication behaviour to test;
      * immediately after an outer step, every rank in the world holds the same
        theta again.
    """
    torch.manual_seed(7)
    model = wrap_model(_tiny_model(seed=200 + replicas.rank), strategy="ddp",
                       device=torch.device("cpu"), process_group=replicas.inner_group)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    inner_steps = 4
    dl = DiLoCo(model, replicas, inner_steps=inner_steps, outer_lr=OUTER_LR,
                outer_momentum=OUTER_MOMENTUM)

    # Each rank gets its own shard, as DistributedSampler would give it.
    generator = torch.Generator().manual_seed(1000 + replicas.rank)
    x = torch.randn(16, 6, generator=generator)
    y = torch.randn(16, 3, generator=generator)

    saw_outer = 0
    for step in range(2 * inner_steps):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
        stats = dl.inner_step()

        flat = _flat_params(model)
        if stats is None:
            assert _spread(flat, group=replicas.inner_group) < 1e-9, \
                f"step {step}: DDP left ranks of one replica disagreeing"
            if replicas.num_replicas > 1:
                assert _spread(flat) > 1e-9, \
                    (f"step {step}: replicas are identical mid-phase, so gradients "
                     f"are being all-reduced across replicas")
        else:
            saw_outer += 1
            assert _spread(flat) < 1e-9, \
                f"step {step}: ranks disagree about theta right after an outer step"
            assert 0.0 < stats["agreement"] <= 1.0 + 1e-6, stats

    assert saw_outer == 2, f"expected 2 outer steps, got {saw_outer}"
    log(f"DDP-in-replica training reconverges to one theta at every outer step "
        f"(agreement {stats['agreement']:.3f})")


def check_global_model_is_identical_everywhere(replicas, log) -> None:
    """What evaluation and checkpointing see must be one model, not k."""
    model = _tiny_model(seed=0)
    dl = DiLoCo(model, replicas, inner_steps=100)
    _fill(model, float(replicas.replica_id + 1))     # drift, no outer step

    drifted = _flat_params(model).clone()
    if replicas.num_replicas > 1:
        assert _spread(drifted) > 1e-9, "test is vacuous: replicas did not drift"

    with dl.global_model():
        assert _spread(_flat_params(model)) == 0.0, "global_model() is not rank-invariant"
    assert torch.equal(_flat_params(model), drifted), "local weights were not restored"
    log("global_model() exposes one theta on every rank and restores local weights")


CHECKS = (
    check_group_membership,
    check_theta0_is_shared,
    check_federated_averaging,
    check_nesterov_against_reference,
    check_buffers_are_averaged,
    check_ddp_training_converges_to_one_theta,
    check_global_model_is_identical_everywhere,
)


def _worker(rank: int, world_size: int, num_replicas: int, rendezvous: str,
            results) -> None:
    if rendezvous.startswith("file://"):
        init_method = rendezvous
    else:
        # TCP rendezvous, only when a port was asked for explicitly. See main().
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=rendezvous)
        init_method = None
    dist.init_process_group(backend="gloo", init_method=init_method, rank=rank,
                            world_size=world_size,
                            timeout=datetime.timedelta(seconds=120))
    try:
        replicas = build_replicas(num_replicas)
        for fn in CHECKS:
            def log(message, _fn=fn):
                if rank == 0:
                    print(f"  PASS  {message}", flush=True)
            try:
                fn(replicas, log)
            except Exception:            # noqa: BLE001
                if rank == 0:
                    print(f"  FAIL  {fn.__name__}", flush=True)
                    traceback.print_exc()
                results.append(f"{fn.__name__} (rank {rank})")
                # Keep going: every rank fails the same check, so the remaining
                # collectives stay balanced.
    finally:
        dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world-size", type=int, default=None)
    p.add_argument("--replicas", type=int, default=None)
    p.add_argument(
        "--port", type=str, default=None,
        help="Force a TCP rendezvous on this port. Left unset, ranks meet through a "
             "temporary file instead, which is what makes back-to-back runs of this "
             "suite safe -- see main().",
    )
    args = p.parse_args(argv)

    if args.world_size and args.replicas:
        layouts = [(args.world_size, args.replicas)]
    else:
        # (4, 2) is the interesting case: two replicas of two ranks, so both group
        # families do real work. (4, 4) covers one rank per replica, where the
        # inner group is a singleton and the outer group is the whole world.
        layouts = [(4, 2), (4, 4)]

    # A file rendezvous rather than a TCP one, unless --port forces the old path.
    #
    # The ranks used to meet on a port derived arithmetically from --port, which made
    # it *fixed* across invocations: every run of this suite used the same two ports.
    # A just-torn-down rendezvous lingers in TIME_WAIT for ~60s, so running the suite
    # twice in quick succession -- which is exactly what checking for flakiness
    # involves -- could fail to bind. Observed once in 35 runs, and not reproducible
    # in isolation, which is the signature of a collision with the previous run rather
    # than of anything in the code under test.
    #
    # The old expression also folded `len(failures)` into the port, so a failure in
    # the first layout silently moved the second one somewhere else. A FileStore has
    # no port, no TIME_WAIT and no arithmetic.
    failures = []
    rendezvous_dir = tempfile.mkdtemp(
        prefix="pww-gloo-", dir=os.environ.get("PWW_TEST_TMPDIR") or None
    )
    try:
        for index, (world_size, num_replicas) in enumerate(layouts):
            print(f"\nworld_size={world_size} k={num_replicas} "
                  f"({world_size // num_replicas} ranks per replica)")
            # Per layout either way. The file must not exist yet -- FileStore creates
            # it, and reusing one across layouts would have the second rendezvous read
            # the first's stale keys. On the TCP path the port still has to step, for
            # the TIME_WAIT reason above, but by layout index only: folding the failure
            # count in is what made the second layout's port depend on whether the
            # first one passed.
            rendezvous = (
                str(int(args.port) + index) if args.port
                else f"file://{rendezvous_dir}/layout-{index}"
            )
            with mp.Manager() as manager:
                results = manager.list()
                mp.spawn(_worker,
                         args=(world_size, num_replicas, rendezvous, results),
                         nprocs=world_size, join=True)
                failures.extend(results)
    finally:
        shutil.rmtree(rendezvous_dir, ignore_errors=True)

    print(f"\n{len(layouts) * len(CHECKS)} checks across {len(layouts)} layouts: "
          f"{len(set(failures))} failed")
    if failures:
        for name in sorted(set(failures)):
            print(f"  {name}")
        return 1
    print("all diloco gloo tests passed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
