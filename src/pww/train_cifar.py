"""CIFAR-10 training entrypoint.

    python3 -m pww.train_cifar --epochs 30 --batch-size 128

Small enough to iterate on in minutes, but exercises every piece the LLM runs
will need: process-group setup, GPU pinning, DDP/FSDP wrapping, mixed
precision, distributed metric reduction, checkpoint/resume and JSONL metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from . import distributed as D
from .checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .config import add_common_args, apply_config_file, resolve_output_dir, save_config, set_seed
from .data.cifar import NUM_CLASSES, build_cifar10_loaders
from .diloco import DiLoCo, build_replicas, describe_plan
from .logging_utils import MetricsWriter, get_logger, log_environment, setup_logging
from .models.resnet import RESNET_FACTORY, build_resnet
from .parallel import count_parameters, resolve_dtype, wrap_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIFAR-10 distributed training")
    add_common_args(p)

    g = p.add_argument_group("model")
    g.add_argument("--model", type=str, default="resnet18", choices=sorted(RESNET_FACTORY))

    g = p.add_argument_group("data")
    g.add_argument("--data-root", type=str, default=None, help="default: $PWW_DATA_DIR/cifar10")
    g.add_argument("--batch-size", type=int, default=128, help="PER-RANK batch size")
    g.add_argument("--num-workers", type=int, default=6)

    g = p.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--lr", type=float, default=0.1, help="Base LR at a global batch of 128")
    g.add_argument("--inner-optimizer", type=str, default="sgd", choices=("sgd", "adamw"),
                   help="DiLoCo's InnerOpt. sgd+nesterov is right for ResNet/CIFAR; "
                        "the paper uses adamw, which is what an LLM run wants")
    g.add_argument("--momentum", type=float, default=0.9, help="SGD only")
    g.add_argument("--weight-decay", type=float, default=5e-4)
    g.add_argument("--warmup-epochs", type=int, default=2)
    g.add_argument("--no-lr-scaling", action="store_true",
                   help="Do not scale LR by global batch size")

    g = p.add_argument_group("execution")
    g.add_argument("--save-every", type=int, default=10, help="Checkpoint every N epochs")
    g.add_argument("--log-every", type=int, default=50, help="Log every N steps")
    g.add_argument("--max-steps-per-epoch", type=int, default=None,
                   help="Cut epochs short -- for smoke tests")
    return apply_config_file(p, argv)


def build_scheduler(optimizer, args, steps_per_epoch: int):
    """Linear warmup then cosine decay.

    Warmup matters here: with 8 or 16 ranks the effective batch is large enough
    that a cold start at the scaled LR diverges.
    """
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)
    total_steps = max(1, args.epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _outer_state_path(output_dir: Path, rank: int, sharded: bool) -> Path:
    """Where DiLoCo's outer momentum lives, next to the model checkpoints.

    Per-rank under sharded checkpointing, because each rank then owns a different
    slice of theta and so a different slice of the momentum buffer. Under
    consolidated checkpointing every rank ends an outer step with a bit-identical
    full copy -- the outer gradient is all-reduced before the step -- so one file
    is enough and every rank reads rank 0's.

    In a subdirectory, not beside the checkpoints: latest_checkpoint globs *.pt,
    so a sidecar at the top level becomes the newest "checkpoint" and
    --resume auto tries to load the momentum buffer as a model.
    """
    return Path(output_dir) / "diloco" / f"outer_r{rank if sharded else 0}.pt"


def _save_outer_state(diloco, output_dir: Path, rank: int, sharded: bool) -> None:
    if sharded or rank == 0:
        diloco.save(_outer_state_path(output_dir, rank, sharded))


def train_one_epoch(model, loader, optimizer, scheduler, criterion, info, args,
                    epoch: int, autocast_dtype, metrics, *, diloco=None) -> dict:
    model.train()
    log = get_logger()
    loss_sum, correct, seen = 0.0, 0.0, 0
    outer_steps = 0
    t_epoch = time.time()

    for step, (inputs, targets) in enumerate(loader):
        if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
            break

        inputs = inputs.to(info.device, non_blocking=True)
        targets = targets.to(info.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=info.device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Algorithm 1 lines 11-14, on every H-th inner step. Collective, so it
        # must be reached by every rank the same number of times -- which is why
        # drop_last=True on the training sampler is load-bearing here.
        if diloco is not None:
            outer = diloco.inner_step()
            if outer is not None:
                outer_steps += 1
                metrics.log(split="diloco", epoch=epoch + 1, **outer)
                if info.is_master:
                    log.info("outer step %d | delta norm %.4f -> %.4f | agreement %.3f",
                             int(outer["outer_step"]), outer["delta_norm"],
                             outer["avg_delta_norm"], outer["agreement"])

        batch = targets.size(0)
        loss_sum += loss.detach().item() * batch
        correct += (outputs.detach().argmax(1) == targets).sum().item()
        seen += batch

        if info.is_master and step % args.log_every == 0:
            log.info("epoch %d step %d/%d loss %.4f lr %.5f",
                     epoch + 1, step, len(loader), loss.item(), scheduler.get_last_lr()[0])

    # Reduce over ranks *before* dividing, so the reported number is the true
    # global average rather than rank 0's slice.
    loss_sum = D.all_reduce_sum(loss_sum, info.device)
    correct = D.all_reduce_sum(correct, info.device)
    seen = D.all_reduce_sum(seen, info.device)
    elapsed = time.time() - t_epoch

    stats = {
        "loss": loss_sum / max(1, seen),
        "acc": 100.0 * correct / max(1, seen),
        "images_per_s": seen / elapsed,
        "epoch_s": elapsed,
    }
    if diloco is not None:
        # Note these train numbers are the mean over replicas of each replica's
        # own local model, not a metric of theta. Eval is the one to compare.
        stats["outer_steps"] = outer_steps
    metrics.log(split="train", epoch=epoch + 1, **stats)
    return stats


@torch.no_grad()
def evaluate(model, loader, criterion, info, args, epoch: int, autocast_dtype, metrics) -> dict:
    model.eval()
    loss_sum, correct, seen = 0.0, 0.0, 0

    for step, (inputs, targets) in enumerate(loader):
        if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
            break
        inputs = inputs.to(info.device, non_blocking=True)
        targets = targets.to(info.device, non_blocking=True)
        with torch.autocast(device_type=info.device.type, dtype=autocast_dtype,
                            enabled=autocast_dtype is not None):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        batch = targets.size(0)
        loss_sum += loss.item() * batch
        correct += (outputs.argmax(1) == targets).sum().item()
        seen += batch

    loss_sum = D.all_reduce_sum(loss_sum, info.device)
    correct = D.all_reduce_sum(correct, info.device)
    seen = D.all_reduce_sum(seen, info.device)

    stats = {"loss": loss_sum / max(1, seen), "acc": 100.0 * correct / max(1, seen)}
    metrics.log(split="eval", epoch=epoch + 1, **stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    info = D.setup()

    output_dir = resolve_output_dir(args, default_name=f"cifar10-{args.model}")
    log = setup_logging(info.rank, output_dir)
    if info.is_master:
        save_config(args, output_dir)
    set_seed(args.seed, info.rank)
    log_environment(log, info)

    # --- data ---------------------------------------------------------------
    import os

    data_root = args.data_root or str(Path(os.environ.get("PWW_DATA_DIR", "./data")) / "cifar10")
    train_loader, eval_loader = build_cifar10_loaders(
        data_root,
        batch_size=args.batch_size,
        rank=info.rank,
        world_size=info.world_size,
        num_workers=args.num_workers,
        download=False,   # pre-fetched by scripts/download_data.sh
    )
    steps_per_epoch = args.max_steps_per_epoch or len(train_loader)

    # --- replica layout -----------------------------------------------------
    # With DiLoCo the world is carved into k replicas and DDP/FSDP is scoped to
    # one replica; without it, replicas is None and everything below behaves
    # exactly as before.
    replicas = None
    if args.diloco_replicas > 0:
        replicas = build_replicas(args.diloco_replicas)
        log.info("%s", describe_plan(replicas, args.diloco_inner_steps,
                                     args.epochs * steps_per_epoch))

    # --- model --------------------------------------------------------------
    model = build_resnet(args.model, num_classes=NUM_CLASSES)
    # BatchNorm running stats are per-rank under DDP. At a per-rank batch of 128
    # that is harmless and cheaper than SyncBatchNorm; below ~32 it starts to
    # hurt and SyncBatchNorm becomes worth the communication.
    inner_group = replicas.inner_group if replicas is not None else None
    model = wrap_model(model, strategy=args.parallel, device=info.device, dtype=args.dtype,
                       process_group=inner_group)
    total, trainable = count_parameters(model, group=inner_group)
    log.info("model %s: %.2fM params (%.2fM trainable)", args.model, total / 1e6, trainable / 1e6)

    # Autocast handles mixed precision for DDP; FSDP does it via MixedPrecision
    # in wrap_model, so avoid applying it twice.
    autocast_dtype = None
    if args.dtype != "fp32" and args.parallel != "fsdp":
        autocast_dtype = resolve_dtype(args.dtype)

    # --- optimiser (InnerOpt) -----------------------------------------------
    # Under DiLoCo a step only synchronises within one replica, so the batch that
    # the LR should be scaled to is the replica's, not the world's.
    sync_ranks = replicas.ranks_per_replica if replicas is not None else info.world_size
    global_batch = args.batch_size * sync_ranks
    lr = args.lr if args.no_lr_scaling else args.lr * global_batch / 128.0
    if args.inner_optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = build_scheduler(optimizer, args, steps_per_epoch)

    log.info("%s batch %d (per-rank %d x %d ranks) | lr %.4f | epochs %d | inner opt %s",
             "replica" if replicas is not None else "global", global_batch,
             args.batch_size, sync_ranks, lr, args.epochs, args.inner_optimizer)

    # --- DiLoCo (OuterOpt) --------------------------------------------------
    # Built after wrap_model so it sees the same parameter tensors the inner
    # optimizer updates, and before resume so a restored checkpoint lands in both.
    diloco = None
    if replicas is not None:
        diloco = DiLoCo(
            model,
            replicas,
            inner_steps=args.diloco_inner_steps,
            outer_lr=args.diloco_outer_lr,
            outer_momentum=args.diloco_outer_momentum,
            outer_optimizer=args.diloco_outer_optimizer,
            outer_device=None if args.diloco_outer_device == "auto" else args.diloco_outer_device,
            sync_buffers=not args.diloco_no_sync_buffers,
        )
        log.info("outer opt %s | lr %.3f momentum %.3f | buffers %s",
                 args.diloco_outer_optimizer, args.diloco_outer_lr,
                 args.diloco_outer_momentum,
                 "local" if args.diloco_no_sync_buffers else "averaged")

    # --- resume -------------------------------------------------------------
    start_epoch = 0
    resume_path = args.resume
    if resume_path == "auto":
        found = latest_checkpoint(output_dir, sharded=args.sharded_checkpoint)
        resume_path = str(found) if found else None
        if resume_path is None:
            log.info("--resume auto: no checkpoint found, starting fresh")
    if resume_path:
        meta = load_checkpoint(resume_path, model, optimizer, sharded=args.sharded_checkpoint)
        start_epoch = int(meta.get("epoch", 0))
        for _ in range(start_epoch * steps_per_epoch):
            scheduler.step()
        log.info("resumed at epoch %d", start_epoch)
        if diloco is not None:
            # The checkpoint holds theta (it is written inside global_model), so
            # theta comes back with the weights; only the outer momentum needs
            # restoring separately.
            diloco.sync_from_model()
            outer_state = _outer_state_path(output_dir, info.rank, args.sharded_checkpoint)
            if outer_state.exists():
                diloco.load(outer_state)
                # Only the newest outer state is kept, so resuming from an older
                # checkpoint pairs theta with momentum from a later step.
                expected = start_epoch * steps_per_epoch
                if diloco.total_inner_steps != expected:
                    log.warning("outer state is at inner step %d but this checkpoint is "
                                "at %d -- momentum does not belong to these weights; "
                                "delete %s to start the outer optimizer cold instead",
                                diloco.total_inner_steps, expected, outer_state)
            else:
                log.warning("no outer state at %s -- outer momentum restarts cold, "
                            "which costs a few outer steps of progress", outer_state)

    # --- train --------------------------------------------------------------
    best_acc = 0.0
    with MetricsWriter(output_dir, info.rank) as metrics:
        for epoch in range(start_epoch, args.epochs):
            # Without set_epoch the sampler reshuffles identically every epoch,
            # so each rank sees the same slice forever.
            train_loader.sampler.set_epoch(epoch)

            tr = train_one_epoch(model, train_loader, optimizer, scheduler, criterion,
                                 info, args, epoch, autocast_dtype, metrics, diloco=diloco)

            is_last = epoch + 1 == args.epochs
            if is_last and diloco is not None:
                # Fold the trailing partial inner phase in before the final eval
                # and checkpoint, so both describe the model that was trained.
                outer = diloco.finish()
                if outer is not None:
                    metrics.log(split="diloco", epoch=epoch + 1, **outer)
                    log.info("outer step %d (flush of %d inner steps) | agreement %.3f",
                             int(outer["outer_step"]), int(outer["inner_steps"]),
                             outer["agreement"])

            # Evaluate and checkpoint theta rather than whichever replica this
            # rank happens to hold. See DiLoCo.global_model.
            with diloco.global_model() if diloco is not None else contextlib.nullcontext():
                ev = evaluate(model, eval_loader, criterion, info, args, epoch,
                              autocast_dtype, metrics)

                if (epoch + 1) % args.save_every == 0 or is_last:
                    name = (f"step_{epoch + 1}" if args.sharded_checkpoint
                            else f"epoch_{epoch + 1}.pt")
                    save_checkpoint(output_dir / name, model, optimizer, epoch=epoch + 1,
                                    rank=info.rank, sharded=args.sharded_checkpoint,
                                    extra={"eval_acc": ev["acc"]})
                    if diloco is not None:
                        _save_outer_state(diloco, output_dir, info.rank,
                                          args.sharded_checkpoint)

            log.info(
                "epoch %d/%d | train loss %.4f acc %.2f%% | eval loss %.4f acc %.2f%% "
                "| %.0f img/s | %.1fs",
                epoch + 1, args.epochs, tr["loss"], tr["acc"], ev["loss"], ev["acc"],
                tr["images_per_s"], tr["epoch_s"],
            )
            if ev["acc"] > best_acc:
                best_acc = ev["acc"]

    log.info("done. best eval acc %.2f%% | outputs in %s", best_acc, output_dir)
    if diloco is not None:
        log.info("diloco: %d outer steps over %d inner steps",
                 diloco.outer_steps, diloco.total_inner_steps)
    D.cleanup()


if __name__ == "__main__":
    main()
