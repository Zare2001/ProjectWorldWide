"""Standalone Model Evaluation Script for CIFAR-10 Checkpoints.

Evaluates a saved ResNet checkpoint on the CIFAR-10 test set and prints Top-1 Accuracy.

Usage:
    source env.sh
    python3 -m pww.evaluate --checkpoint ./runs/cifar10-resnet18/checkpoint_epoch_30.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from .config import apply_config_file
from .data.cifar import NUM_CLASSES, _transforms
from .logging_utils import get_logger, setup_logging
from .models.resnet import RESNET_FACTORY, build_resnet

logger = get_logger("pww.evaluate")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on the test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / max(1, total)
    acc = (correct / max(1, total)) * 100.0
    return avg_loss, acc


def main() -> None:
    setup_logging(rank=0)
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 Checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved model checkpoint (.pt file or folder)",
    )
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=sorted(RESNET_FACTORY)
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Evaluation batch size"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="CIFAR-10 data root (default: $PWW_DATA_DIR/cifar10)",
    )

    args = apply_config_file(parser)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build Model
    model = build_resnet(args.model).to(device)

    # Load Checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Strip module. prefix if saved with DDP
    clean_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith("module.") else k
        clean_state_dict[key] = v

    model.load_state_dict(clean_state_dict, strict=False)

    # Build Test Dataset
    data_root = args.data_root or str(
        Path(os.environ.get("PWW_DATA_DIR", "./data")) / "cifar10"
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=_transforms(False)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Evaluate
    logger.info("Evaluating on CIFAR-10 Test Set (10,000 images)...")
    test_loss, test_acc = evaluate(model, test_loader, device)

    print("\n" + "=" * 55)
    print(" CIFAR-10 Evaluation Results")
    print("=" * 55)
    print(f" Checkpoint : {checkpoint_path}")
    print(f" Model      : {args.model}")
    print(f" Test Loss  : {test_loss:.4f}")
    print(f" Test Acc   : {test_acc:.2f}%")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
