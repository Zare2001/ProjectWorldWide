from __future__ import annotations

try:
    from .cifar import build_cifar10_loaders, download_cifar10
except ImportError:
    pass

try:
    from .text import TokenizedDataset, build_hf_tokenized_dataset
except ImportError:
    pass
