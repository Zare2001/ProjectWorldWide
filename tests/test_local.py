"""CPU-only tests -- runnable on a login node, no allocation needed.

Covers the logic that is painful to debug inside a queued job: model shapes,
checkpoint round-trips, config layering, LR schedule. Anything that needs real
collectives belongs in pww.smoke instead.

    singularity exec $PWW_CONTAINER python3 tests/test_local.py
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
    """Guards against a config that only fails once it is submitted to a queue."""
    from pww.train_cifar import parse_args

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    files = sorted(config_dir.glob("*.yaml"))
    assert files, f"no configs found in {config_dir}"
    for path in files:
        args = parse_args(["--config", str(path)])
        assert args.epochs > 0 and args.batch_size > 0, f"{path.name}: {args}"


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


if __name__ == "__main__":
    print(f"\n{len(PASSED) + len(FAILED)} tests: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("\nfailures:")
        for name, exc in FAILED:
            print(f"  {name}: {exc}")
        sys.exit(1)
    print("all local tests passed\n")
