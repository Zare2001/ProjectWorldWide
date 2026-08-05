"""Flower + DARL DiLoCo Client Entrypoint for Snellius & LUMI.

Runs on each HPC cluster (Snellius / LUMI). Sets up local GPU parallelism (DDP),
integrates DARL for dynamic data block leasing (port 29510), and connects to the
central Flower server running FedMom (port 29511).

Usage:
    sbatch scripts/snellius/job_flower_diloco.sh
    sbatch scripts/lumi/job_flower_diloco.sh
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from . import distributed as D
from .config import add_common_args, apply_config_file, resolve_output_dir, set_seed
from .darl.client import LeaseClient, LeaseSession
from .darl.space import BlockSpace
from .darl.torch_data import DARLDataSource, LeasedSampler
from .data.cifar import NUM_CLASSES, _transforms, build_cifar10_loaders
from .logging_utils import get_logger, log_environment, setup_logging
from .models.resnet import RESNET_FACTORY, build_resnet
from .parallel import wrap_model

logger = get_logger("pww.train_flower")

try:
    import flwr as fl
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False


class DiLoCoFlowerClient(fl.client.NumPyClient if HAS_FLWR else object):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        eval_loader: DataLoader | None,
        sampler: LeasedSampler,
        darl_source: DARLDataSource,
        inner_steps: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.eval_loader = eval_loader
        self.sampler = sampler
        self.darl_source = darl_source
        self.inner_steps = inner_steps
        self.device = device

    def get_parameters(self, config: dict) -> list:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: list) -> None:
        state_dict = self.model.state_dict()
        for (k, _), val in zip(state_dict.items(), parameters):
            state_dict[k] = torch.from_numpy(val).to(self.device)
        self.model.load_state_dict(state_dict)

    def _execute_local_phase(self, parameters: list | None) -> tuple[list, int, dict]:
        # 1. Synchronize model weights with global params received from FedMom
        if parameters:
            self.set_parameters(parameters)

        # 2. Inner Loop Training (H steps across leased DARL phases)
        self.model.train()
        step_count = 0
        loss_sum = 0.0

        while step_count < self.inner_steps:
            phase = self.darl_source.next_phase()
            if phase is None:
                if D.is_leader():
                    logger.info("DARL dataset leasing complete for this epoch.")
                break

            self.sampler.set_indices(phase.indices)

            for x, y in self.loader:
                if step_count >= self.inner_steps:
                    break
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = nn.functional.cross_entropy(out, y)
                loss.backward()

                # Reduce local gradients across cluster GPUs (DDP)
                if D.world_size() > 1:
                    D.all_reduce_avg_([p.grad for p in self.model.parameters() if p.grad is not None])

                self.optimizer.step()
                loss_sum += loss.item()
                step_count += 1

            self.darl_source.end_phase()
            self.darl_source.commit()

        avg_loss = loss_sum / max(1, step_count)
        if D.is_leader():
            logger.info(f"Completed local inner phase ({step_count} steps), avg loss: {avg_loss:.4f}")

        num_samples_trained = max(1, step_count * self.loader.batch_size)
        return self.get_parameters(config={}), num_samples_trained, {"loss": float(avg_loss)}

    def evaluate(self, parameters: list, config: dict) -> tuple[float, int, dict]:
        if parameters:
            self.set_parameters(parameters)

        if self.eval_loader is None:
            return 0.0, 0, {}

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for x, y in self.eval_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * x.size(0)
                correct += (out.argmax(dim=1) == y).sum().item()
                total += x.size(0)

        avg_loss = total_loss / max(1, total)
        accuracy = (correct / max(1, total)) * 100.0

        if D.is_leader():
            logger.info(
                f"DiLoCo Outer Round Evaluation -> Test Loss: {avg_loss:.4f}, Test Acc: {accuracy:.2f}%"
            )

        return float(avg_loss), total, {"accuracy": float(accuracy)}

    def fit(self, parameters: list, config: dict) -> tuple:
        if D.world_size() > 1 and D.is_leader():
            import torch.distributed as dist
            dist.broadcast_object_list([("FIT", parameters)], src=0)

        return self._execute_local_phase(parameters)

    def run_worker_loop(self) -> None:
        import torch.distributed as dist
        while True:
            box = [None]
            dist.broadcast_object_list(box, src=0)
            msg, params = box[0]
            if msg == "STOP":
                break
            elif msg == "FIT":
                self._execute_local_phase(params)
            elif msg == "EVAL":
                self.evaluate(params, {})


def main() -> None:
    if not HAS_FLWR:
        logger.error("flwr is required on the cluster. Run: pip install flwr")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Flower + DARL DiLoCo Client")
    add_common_args(parser)

    g = parser.add_argument_group("federation")
    g.add_argument("--central-ip", type=str, default="145.38.206.143", help="Central node IP")
    g.add_argument("--darl-port", type=int, default=29510, help="DARL HTTP port")
    g.add_argument("--flower-port", type=int, default=29511, help="Flower gRPC port")
    g.add_argument("--cluster-id", type=str, default=None, help="snellius or lumi")
    g.add_argument("--inner-steps", type=int, default=100, help="H inner steps")

    g = parser.add_argument_group("model")
    g.add_argument("--model", type=str, default="resnet18", choices=sorted(RESNET_FACTORY))

    g = parser.add_argument_group("data")
    g.add_argument("--data-root", type=str, default=None, help="default: $PWW_DATA_DIR/cifar10")
    g.add_argument("--batch-size", type=int, default=128, help="PER-RANK batch size")
    g.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (use 0 for gRPC compatibility)")

    g = parser.add_argument_group("optimisation")
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--lr", type=float, default=0.1)
    g.add_argument("--inner-optimizer", type=str, default="sgd", choices=("sgd", "adamw"))
    g.add_argument("--momentum", type=float, default=0.9)
    g.add_argument("--weight-decay", type=float, default=5e-4)
    g.add_argument("--warmup-epochs", type=int, default=2)

    g = parser.add_argument_group("execution")
    g.add_argument("--save-every", type=int, default=10)
    g.add_argument("--log-every", type=int, default=50)

    args = apply_config_file(parser)

    # 1. Setup PyTorch distributed environment inside cluster
    info = D.setup()
    output_dir = resolve_output_dir(args, default_name=f"cifar10-flower-{args.model}")
    log = setup_logging(info.rank, output_dir)
    device = info.device
    set_seed(args.seed, info.rank)

    cluster_id = args.cluster_id or ("snellius" if info.backend == "nccl" else "lumi")
    darl_url = f"http://{args.central_ip}:{args.darl_port}"
    flower_address = f"{args.central_ip}:{args.flower_port}"

    if info.is_master:
        log.info(
            f"Initializing Flower client on cluster '{cluster_id}' -> "
            f"DARL: {darl_url}, Flower: {flower_address}"
        )

    # 2. Build dataset and DARL Data Source
    data_root = args.data_root or str(Path(os.environ.get("PWW_DATA_DIR", "./data")) / "cifar10")
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=_transforms(True))
    eval_dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=_transforms(False))

    space = BlockSpace(num_samples=len(train_dataset), block_size=1000, seed=args.seed)

    sampler = LeasedSampler()
    loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if info.is_master:
        darl_token = args.darl_token or os.environ.get("DARL_TOKEN", "")
        client = LeaseClient(darl_url, cluster_id, token=darl_token)
        session = LeaseSession(client, space, blocks_per_phase=5)
    else:
        session = None

    darl_source = DARLDataSource(
        space=space,
        session=session,
        rank=info.rank,
        world_size=info.world_size,
        blocks_per_phase=5,
    )

    # 3. Model & Optimizer
    model = build_resnet(args.model).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    client = DiLoCoFlowerClient(
        model=model,
        optimizer=optimizer,
        loader=loader,
        eval_loader=eval_loader,
        sampler=sampler,
        darl_source=darl_source,
        inner_steps=getattr(args, "inner_steps", None) or getattr(args, "diloco_inner_steps", 100),
        device=device,
    )

    # 4. Start Flower client on cluster leader
    if info.is_master:
        fl.client.start_numpy_client(
            server_address=flower_address,
            client=client,
        )
        if info.world_size > 1:
            import torch.distributed as dist
            dist.broadcast_object_list([("STOP", None)], src=0)
    else:
        client.run_worker_loop()

    D.cleanup()


if __name__ == "__main__":
    main()
