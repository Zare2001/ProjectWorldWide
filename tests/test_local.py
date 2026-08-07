"""CPU-only tests -- runnable on a login node, no allocation needed.

Covers the logic that is painful to debug inside a queued job: model shapes,
checkpoint round-trips, config layering, LR schedule, and DiLoCo's outer-step
arithmetic. Anything that needs real collectives belongs in
tests/test_diloco_gloo.py (still CPU-only) or pww.smoke (needs GPUs).

    source env.sh && pww_run python3 tests/test_local.py

pww_run enters the container on LUMI and is a no-op inside the venv on Snellius,
so the same line works on both. Last measured 28/28 on Snellius.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pww.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402
from pww.config import set_seed  # noqa: E402
from pww.models.resnet import RESNET_FACTORY, build_resnet  # noqa: E402
from pww.parallel import count_parameters, resolve_dtype, wrap_model  # noqa: E402

PASSED, FAILED = [], []


def check(name: str):
    def decorator(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        return fn

    return decorator


@check("resnet forward shapes")
def _():
    for name in sorted(RESNET_FACTORY):
        model = build_resnet(name, num_classes=10)
        out = model(torch.randn(4, 3, 32, 32))
        assert out.shape == (4, 10), f"{name} produced {out.shape}"


@check("resnet18 parameter count ~11.2M")
def _():
    total, trainable = count_parameters(build_resnet("resnet18"))
    assert 11_000_000 < total < 11_500_000, total
    assert total == trainable


@check("cifar stem preserves 32x32 resolution")
def _():
    # The whole point of the CIFAR variant: layer1 must still see 32x32.
    model = build_resnet("resnet18")
    x = torch.randn(2, 3, 32, 32)
    feat = torch.nn.functional.relu(model.bn1(model.conv1(x)))
    assert feat.shape[-2:] == (32, 32), feat.shape
    assert model.layer1(feat).shape[-2:] == (32, 32)


@check("unknown model name rejected")
def _():
    try:
        build_resnet("resnet999")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


@check("single-strategy wrap is a no-op on cpu")
def _():
    model = wrap_model(build_resnet("resnet18"), strategy="single", device=torch.device("cpu"))
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


@check("ddp falls back gracefully without a process group")
def _():
    # A bare `python3 -m pww.train_cifar` on a login node must not crash.
    model = wrap_model(build_resnet("resnet18"), strategy="ddp", device=torch.device("cpu"))
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


@check("dtype resolution")
def _():
    assert resolve_dtype("bf16") is torch.bfloat16
    assert resolve_dtype("fp32") is torch.float32
    try:
        resolve_dtype("fp8")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


@check("checkpoint round-trip restores weights and optimizer")
def _():
    set_seed(0)
    model = build_resnet("resnet18")
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Take a step so weights and optimizer momentum are both non-trivial.
    loss = torch.nn.functional.cross_entropy(model(torch.randn(4, 3, 32, 32)),
                                             torch.randint(0, 10, (4,)))
    loss.backward()
    opt.step()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "epoch_1.pt"
        save_checkpoint(path, model, opt, epoch=7, rank=0, extra={"eval_acc": 42.0})
        assert path.exists()

        fresh = build_resnet("resnet18")
        fresh_opt = torch.optim.SGD(fresh.parameters(), lr=0.1, momentum=0.9)
        meta = load_checkpoint(path, fresh, fresh_opt)

        assert meta["epoch"] == 7, meta
        assert meta["eval_acc"] == 42.0, meta
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), fresh.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2), f"weight mismatch at {n1}"


@check("atomic write leaves no .tmp behind")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        model = build_resnet("resnet18")
        save_checkpoint(Path(tmp) / "a.pt", model, None, rank=0)
        assert not list(Path(tmp).glob("*.tmp"))


@check("latest_checkpoint picks the newest")
def _():
    import os
    import time

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        assert latest_checkpoint(tmp) is None
        for i in (1, 2, 3):
            (tmp / f"epoch_{i}.pt").write_bytes(b"x")
            os.utime(tmp / f"epoch_{i}.pt", (time.time() + i, time.time() + i))
        assert latest_checkpoint(tmp).name == "epoch_3.pt"


@check("missing checkpoint raises FileNotFoundError")
def _():
    try:
        load_checkpoint("/nonexistent/ckpt.pt", build_resnet("resnet18"))
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


@check("config file overrides defaults, cli overrides config")
def _():
    from pww.train_cifar import parse_args

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "c.yaml"
        cfg.write_text("epochs: 99\nbatch-size: 7\nmodel: resnet50\n")

        args = parse_args(["--config", str(cfg)])
        assert args.epochs == 99 and args.batch_size == 7 and args.model == "resnet50"

        args = parse_args(["--config", str(cfg), "--epochs", "3"])
        assert args.epochs == 3, "cli must win over config file"
        assert args.batch_size == 7, "unset keys must still come from config"


@check("unknown config key is rejected loudly")
def _():
    from pww.train_cifar import parse_args

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "c.yaml"
        cfg.write_text("epochs: 5\ntypo_here: 1\n")
        try:
            parse_args(["--config", str(cfg)])
        except ValueError as exc:
            assert "typo_here" in str(exc)
            return
    raise AssertionError("expected ValueError")


@check("shipped configs/ files all parse")
def _():
    """Guards against a config that only fails once it is submitted to a queue.

    Routed by consumer, because `configs/` holds three unrelated kinds of file and
    every parser rejects unknown keys -- so checking them all against the CIFAR
    trainer's parser reports valid configs as broken:

      cifar10_*.yaml            the CIFAR trainer
      llm_*.yaml                pww.train_llm_flower (--seq-len, --attn-implementation)
      central_aggregator*.yaml  pww.central.server on the central VM; no epochs or
                                batch size at all

    TOML configs under `configs/titan/` belong to torchtitan's own ConfigManager and
    are covered by tests/test_titan.py instead.
    """
    from pww.central.server import build_parser as build_aggregator_parser
    from pww.config import apply_config_file
    from pww.train_cifar import parse_args
    from pww.train_llm_flower import build_parser as build_llm_parser

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    files = sorted(config_dir.glob("*.yaml"))
    assert files, f"no configs found in {config_dir}"

    aggregator = [p for p in files if p.name.startswith("central_aggregator")]
    llm = [p for p in files if p.name.startswith("llm_")]
    trainer = [p for p in files if p not in aggregator and p not in llm]
    for group, label in ((aggregator, "aggregator"), (llm, "llm"), (trainer, "trainer")):
        assert group, f"no {label} configs found in {config_dir}"

    for path in trainer:
        args = parse_args(["--config", str(path)])
        assert args.epochs > 0 and args.batch_size > 0, f"{path.name}: {args}"

    for path in llm:
        args = apply_config_file(build_llm_parser(), ["--config", str(path)])
        assert args.batch_size > 0 and args.seq_len > 0, f"{path.name}: {args}"

    for path in aggregator:
        args = apply_config_file(build_aggregator_parser(), ["--config", str(path)])
        assert args.num_rounds > 0, f"{path.name}: num_rounds={args.num_rounds}"
        assert args.min_clients > 0, f"{path.name}: min_clients={args.min_clients}"


@check("lr schedule warms up then decays to ~0")
def _():
    import argparse

    from pww.train_cifar import build_scheduler

    model = build_resnet("resnet18")
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    args = argparse.Namespace(warmup_epochs=2, epochs=10)
    steps = 100
    sched = build_scheduler(opt, args, steps)

    lrs = []
    for _ in range(args.epochs * steps):
        lrs.append(sched.get_last_lr()[0])
        sched.step()

    assert lrs[0] < 0.02, f"warmup should start near zero, got {lrs[0]}"
    peak = max(lrs)
    assert abs(peak - 1.0) < 1e-6, f"should reach base lr, peaked at {peak}"
    assert lrs.index(peak) <= 2 * steps, "peak should land at end of warmup"
    assert lrs[-1] < 0.01, f"cosine should decay to ~0, ended at {lrs[-1]}"


@check("metrics writer emits valid jsonl for rank 0 only")
def _():
    import json

    from pww.logging_utils import MetricsWriter

    with tempfile.TemporaryDirectory() as tmp:
        with MetricsWriter(tmp, rank=0) as m:
            m.log(split="train", epoch=1, loss=0.5)
            m.log(split="eval", epoch=1, loss=0.4)
        lines = (Path(tmp) / "metrics.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert rec["loss"] == 0.5 and "wall_s" in rec

    with tempfile.TemporaryDirectory() as tmp:
        with MetricsWriter(tmp, rank=3) as m:
            m.log(loss=1.0)
        assert not list(Path(tmp).glob("*.jsonl")), "non-zero ranks must not write"


@check("one full train+eval step end to end on cpu")
def _():
    """Exercise the real training loop against synthetic data."""
    import argparse

    from pww.distributed import DistInfo
    from pww.logging_utils import MetricsWriter, setup_logging
    from pww.train_cifar import build_scheduler, evaluate, train_one_epoch

    setup_logging(rank=0)
    info = DistInfo(rank=0, world_size=1, local_rank=0, device=torch.device("cpu"),
                    backend="gloo", hostname="local")

    dataset = torch.utils.data.TensorDataset(
        torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,))
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8)

    model = build_resnet("resnet18")
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    args = argparse.Namespace(max_steps_per_epoch=2, log_every=1, warmup_epochs=0, epochs=1)
    sched = build_scheduler(opt, args, 2)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    with tempfile.TemporaryDirectory() as tmp:
        with MetricsWriter(tmp, rank=0) as metrics:
            tr = train_one_epoch(model, loader, opt, sched, crit, info, args, 0, None, metrics)
            ev = evaluate(model, loader, crit, info, args, 0, None, metrics)

    for stats, keys in ((tr, ("loss", "acc", "images_per_s")), (ev, ("loss", "acc"))):
        for k in keys:
            assert k in stats, f"missing {k}"
        assert stats["loss"] > 0 and 0 <= stats["acc"] <= 100, stats


# --- DiLoCo -----------------------------------------------------------------
# Single-process, so the outer step reduces over one replica. That is enough to
# pin down every part of the algorithm except the collectives themselves, which
# tests/test_diloco_gloo.py covers with real process groups.


def _tiny_model(value: float = 0.0) -> torch.nn.Module:
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.BatchNorm1d(3))
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(value)
    return model


@check("replica layout splits the world into contiguous equal blocks")
def _():
    from pww.diloco import build_replicas

    r = build_replicas(4, rank=6, world_size=16)
    assert (r.num_replicas, r.ranks_per_replica) == (4, 4), r
    assert (r.replica_id, r.rank_in_replica) == (1, 2), r
    assert not r.is_replica_master

    for rank, want in ((0, (0, 0)), (3, (0, 3)), (4, (1, 0)), (15, (3, 3))):
        r = build_replicas(4, rank=rank, world_size=16)
        assert (r.replica_id, r.rank_in_replica) == want, (rank, r)

    # k == world_size: one rank per replica, no inner group work to do.
    r = build_replicas(8, rank=5, world_size=8)
    assert (r.ranks_per_replica, r.replica_id, r.rank_in_replica) == (1, 5, 0), r


@check("replica layout rejects sizes that cannot be split evenly")
def _():
    from pww.diloco import build_replicas

    for k, world in ((3, 8), (5, 16), (9, 8), (0, 8), (-1, 8)):
        try:
            build_replicas(k, rank=0, world_size=world)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for k={k}, world={world}")

    # The message has to name the usable values, or the next person guesses.
    try:
        build_replicas(3, rank=0, world_size=8)
    except ValueError as exc:
        assert "1, 2, 4, 8" in str(exc), str(exc)


@check("outer step with sgd(lr=1, momentum=0) and k=1 is exactly the identity")
def _():
    """FederatedAveraging over one replica must return the replica's own weights.

    This is the load-bearing sign convention: Delta is theta(t-1) - theta_i(t),
    so descending it has to *reproduce* the inner progress, not undo it. A flipped
    sign still trains, just backwards, and only shows up as a bad loss curve.
    """
    from pww.diloco import DiLoCo, build_replicas

    model = _tiny_model()
    dl = DiLoCo(model, build_replicas(1, rank=0, world_size=1), inner_steps=2,
                outer_optimizer="sgd", outer_lr=1.0, outer_momentum=0.0)

    with torch.no_grad():                       # pretend two inner steps happened
        for p in model.parameters():
            p.add_(torch.arange(p.numel(), dtype=torch.float32).view_as(p))
    trained = [p.detach().clone() for p in model.parameters()]

    assert dl.inner_step() is None, "must not fire before H steps"
    stats = dl.inner_step()
    assert stats is not None and stats["outer_step"] == 1.0, stats
    for after, before in zip(model.parameters(), trained):
        assert torch.allclose(after, before, atol=1e-6), "outer step changed the weights"


@check("nesterov outer step overshoots the inner progress by (1 + momentum)")
def _():
    from pww.diloco import DiLoCo, build_replicas

    model = _tiny_model()
    theta0 = [p.detach().clone() for p in model.parameters()]
    dl = DiLoCo(model, build_replicas(1, rank=0, world_size=1), inner_steps=1,
                outer_optimizer="nesterov", outer_lr=0.7, outer_momentum=0.9)

    step = 0.25
    with torch.no_grad():
        for p in model.parameters():
            p.add_(step)                        # inner progress: +0.25 everywhere

    dl.inner_step()
    # PyTorch nesterov, first step from a zero buffer: d_used = (1 + m) * Delta.
    # Delta = theta0 - theta_local = -0.25, so theta(1) = theta0 + 0.7*1.9*0.25.
    want = 0.7 * 1.9 * step
    for after, before in zip(model.parameters(), theta0):
        assert torch.allclose(after - before, torch.full_like(after, want), atol=1e-6), \
            f"expected +{want}, got {(after - before).flatten()[0].item()}"


@check("finish() flushes a partial inner phase")
def _():
    from pww.diloco import DiLoCo, build_replicas

    model = _tiny_model()
    dl = DiLoCo(model, build_replicas(1, rank=0, world_size=1), inner_steps=100)
    for _ in range(7):
        assert dl.inner_step() is None
    stats = dl.finish()
    assert stats is not None and stats["inner_steps"] == 7.0, stats
    assert dl.finish() is None, "a second finish() must be a no-op"


@check("finish() warns when a short flush meets outer momentum")
def _():
    """Measured to cost ~1.8 points of eval accuracy, so it must not be silent."""
    import logging

    from pww.diloco import DiLoCo, build_replicas

    replicas = build_replicas(1, rank=0, world_size=1)
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("pww")
    handler = _Capture()
    logger.addHandler(handler)
    try:
        # With momentum: 3 of H=100 is well under half, so warn.
        dl = DiLoCo(model := _tiny_model(), replicas, inner_steps=100,
                    outer_momentum=0.9)
        for _ in range(3):
            dl.inner_step()
        dl.finish()
        assert any("mis-scaled" in m for m in records), records

        # Without momentum the flush is harmless, so stay quiet.
        records.clear()
        dl = DiLoCo(model, replicas, inner_steps=100, outer_optimizer="sgd",
                    outer_lr=1.0, outer_momentum=0.0)
        for _ in range(3):
            dl.inner_step()
        dl.finish()
        assert not any("mis-scaled" in m for m in records), records
    finally:
        logger.removeHandler(handler)


@check("global_model() swaps theta in and restores local weights on exit")
def _():
    from pww.diloco import DiLoCo, build_replicas

    model = _tiny_model()
    dl = DiLoCo(model, build_replicas(1, rank=0, world_size=1), inner_steps=1,
                outer_optimizer="sgd", outer_lr=1.0, outer_momentum=0.0)
    theta = [p.detach().clone() for p in model.parameters()]

    with torch.no_grad():                       # drift away from theta
        for p in model.parameters():
            p.add_(3.0)
        for b in model.buffers():
            if b.is_floating_point():
                b.add_(5.0)
    drifted = [p.detach().clone() for p in model.parameters()]
    drifted_buffers = [b.detach().clone() for b in model.buffers() if b.is_floating_point()]

    with dl.global_model():
        for p, want in zip(model.parameters(), theta):
            assert torch.allclose(p, want, atol=1e-6), "theta was not swapped in"
        for b, drifted_b in zip((b for b in model.buffers() if b.is_floating_point()),
                                drifted_buffers):
            assert not torch.allclose(b, drifted_b) or torch.allclose(
                drifted_b, torch.zeros_like(drifted_b)), "buffers were not swapped"

    for p, want in zip(model.parameters(), drifted):
        assert torch.allclose(p, want, atol=1e-6), "local weights were not restored"
    for b, want in zip((b for b in model.buffers() if b.is_floating_point()),
                       drifted_buffers):
        assert torch.allclose(b, want, atol=1e-6), "local buffers were not restored"


@check("outer state round-trips and refuses a mismatched configuration")
def _():
    from pww.diloco import DiLoCo, build_replicas

    replicas = build_replicas(1, rank=0, world_size=1)
    model = _tiny_model()
    dl = DiLoCo(model, replicas, inner_steps=1, outer_lr=0.7, outer_momentum=0.9)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)
    dl.inner_step()
    dl.inner_step()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "diloco_outer_r0.pt"
        dl.save(path)
        assert path.exists() and not list(Path(tmp).glob("*.tmp"))

        fresh = DiLoCo(_tiny_model(), replicas, inner_steps=1,
                       outer_lr=0.7, outer_momentum=0.9)
        assert fresh.outer_steps == 0
        fresh.load(path)
        assert fresh.outer_steps == dl.outer_steps == 2, fresh.outer_steps
        assert fresh.total_inner_steps == dl.total_inner_steps

        # Momentum must actually come back, not just the counters.
        got = fresh._opt.state_dict()["state"][0]["momentum_buffer"]
        want = dl._opt.state_dict()["state"][0]["momentum_buffer"]
        assert torch.allclose(got, want), "momentum buffer was not restored"

        # Swapping the outer optimizer invalidates the momentum, so refuse it
        # rather than silently reinterpreting Adam moments as SGD momentum.
        other = DiLoCo(_tiny_model(), replicas, inner_steps=1, outer_optimizer="adamw")
        try:
            other.load(path)
        except ValueError as exc:
            assert "outer_optimizer" in str(exc), str(exc)
        else:
            raise AssertionError("expected ValueError on optimizer mismatch")


@check("outer step rejects contradictory settings")
def _():
    from pww.diloco import DiLoCo, build_replicas

    replicas = build_replicas(1, rank=0, world_size=1)
    for kwargs, needle in (
        ({"inner_steps": 0}, "inner_steps"),
        ({"inner_steps": 1, "outer_optimizer": "rmsprop"}, "rmsprop"),
        ({"inner_steps": 1, "outer_momentum": 0.0}, "momentum"),   # nesterov needs it
    ):
        try:
            DiLoCo(_tiny_model(), replicas, **kwargs)
        except ValueError as exc:
            assert needle in str(exc), f"{kwargs}: {exc}"
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


@check("outer state does not shadow the newest model checkpoint")
def _():
    """--resume auto globs *.pt, so the outer state must not live among them."""
    from pww.train_cifar import _outer_state_path

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "epoch_1.pt").write_bytes(b"x")
        outer = _outer_state_path(tmp, rank=0, sharded=False)
        outer.parent.mkdir(parents=True, exist_ok=True)
        outer.write_bytes(b"y")                       # written after, so newer
        assert latest_checkpoint(tmp).name == "epoch_1.pt", latest_checkpoint(tmp)


@check("diloco flags reach the trainer and default to off")
def _():
    from pww.train_cifar import parse_args

    args = parse_args([])
    assert args.diloco_replicas == 0, "DiLoCo must be opt-in"

    args = parse_args(["--diloco-replicas", "4", "--diloco-inner-steps", "50",
                       "--diloco-outer-optimizer", "sgd", "--diloco-outer-lr", "1.0",
                       "--diloco-outer-momentum", "0.0", "--inner-optimizer", "adamw"])
    assert args.diloco_replicas == 4 and args.diloco_inner_steps == 50
    assert args.diloco_outer_optimizer == "sgd" and args.inner_optimizer == "adamw"

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "c.yaml"
        cfg.write_text("diloco-replicas: 2\ndiloco-inner-steps: 200\n")
        args = parse_args(["--config", str(cfg)])
        assert args.diloco_replicas == 2 and args.diloco_inner_steps == 200


if __name__ == "__main__":
    print(f"\n{len(PASSED) + len(FAILED)} tests: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("\nfailures:")
        for name, exc in FAILED:
            print(f"  {name}: {exc}")
        sys.exit(1)
    print("all local tests passed\n")
