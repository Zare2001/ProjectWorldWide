"""CIFAR-10 loaders.

Compute nodes reach the internet only through a slow proxy, and having 8 ranks
race to download the same archive is both slow and a good way to corrupt the
cache. So: download once from a login node (scripts/download_data.sh), and set
download=False everywhere in the job.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# Per-channel statistics of the CIFAR-10 training set.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10


def _transforms(train: bool) -> transforms.Compose:
    ops = []
    if train:
        # The two standard CIFAR augmentations; without them ResNet-18 overfits
        # hard and tops out around 86% instead of ~95%.
        ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return transforms.Compose(ops)


def download_cifar10(data_root: str | Path) -> None:
    """Fetch CIFAR-10 into `data_root`. Run from a login node, once."""
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    for train in (True, False):
        datasets.CIFAR10(root=str(data_root), train=train, download=True)


def build_cifar10_loaders(
    data_root: str | Path,
    *,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 6,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build (train, eval) loaders with a distributed sampler on each.

    batch_size is PER RANK; the global batch is batch_size * world_size.
    """
    data_root = str(data_root)
    pin = torch.cuda.is_available()

    loaders = []
    for train in (True, False):
        dataset = datasets.CIFAR10(
            root=data_root, train=train, download=download, transform=_transforms(train)
        )
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            # Ragged final batches make ranks disagree on step count and hang
            # the collective, so drop the tail when training. Evaluation keeps
            # every sample (DistributedSampler pads instead).
            drop_last=train,
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin,
                drop_last=train,
                # Workers are expensive to respawn each epoch on Lustre.
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
            )
        )
    return loaders[0], loaders[1]
