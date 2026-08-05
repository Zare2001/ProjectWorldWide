"""Tokenized Text Dataset & DARL Integration for LLM Pre-training.

Supports:
1. HuggingFace datasets (e.g. `wikitext`, `allenai/c4`, or custom datasets with `input_ids`).
2. Flat binary / numpy memory-mapped arrays (`.bin`, `.npy` of uint16/uint32 tokens).
3. PyTorch `.pt` token tensors or disk HuggingFace datasets (`load_from_disk`).

Slices token streams into sequences of length `seq_len`, and hooks into DARL
`BlockSpace` and `LeasedSampler` for dynamic token partitioning across HPC sites.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from ..logging_utils import get_logger

logger = get_logger("pww.data.text")


class TokenizedDataset(Dataset):
    """Dataset serving fixed-length token sequences for Causal LM training."""

    def __init__(
        self,
        tokens: Sequence[int] | torch.Tensor | Any,
        seq_len: int = 2048,
    ) -> None:
        self.seq_len = seq_len

        if isinstance(tokens, torch.Tensor):
            self.tokens = tokens.flatten()
        elif hasattr(tokens, "__len__") and hasattr(tokens, "__getitem__"):
            self.tokens = tokens
        else:
            self.tokens = torch.tensor(tokens, dtype=torch.long)

        self.num_sequences = len(self.tokens) // self.seq_len
        logger.info(
            f"TokenizedDataset initialized: {len(self.tokens):,} total tokens -> "
            f"{self.num_sequences:,} sequences (seq_len={self.seq_len})"
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        chunk = self.tokens[start:end]

        if not isinstance(chunk, torch.Tensor):
            chunk = torch.tensor(chunk, dtype=torch.long)

        # For Causal LM: input_ids and labels are identical (shifted inside loss computation)
        return {
            "input_ids": chunk,
            "labels": chunk.clone(),
        }


def build_hf_tokenized_dataset(
    dataset_name_or_path: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    tokenizer_name: str = "gpt2",
    seq_len: int = 1024,
    split: str = "train",
) -> TokenizedDataset:
    """Build a TokenizedDataset from a HuggingFace dataset and tokenizer."""
    try:
        from datasets import load_dataset, load_from_disk
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "HuggingFace `datasets` and `transformers` are required. "
            "Install them using: pip install datasets transformers"
        )

    logger.info(f"Loading tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    path = Path(dataset_name_or_path)
    if path.exists() and path.is_dir():
        logger.info(f"Loading HuggingFace dataset from disk: {path}...")
        hf_dataset = load_from_disk(str(path))
        if isinstance(hf_dataset, dict):
            hf_dataset = hf_dataset[split]
    else:
        logger.info(f"Loading HuggingFace dataset: {dataset_name_or_path} ({dataset_config}, split={split})...")
        hf_dataset = load_dataset(dataset_name_or_path, dataset_config, split=split)

    # Tokenize text column if not already tokenized
    if "input_ids" not in hf_dataset.column_names:
        text_column = "text" if "text" in hf_dataset.column_names else hf_dataset.column_names[0]
        logger.info(f"Tokenizing column '{text_column}'...")

        def tokenize_function(examples: dict) -> dict:
            return tokenizer(examples[text_column], truncation=False, padding=False)

        tokenized = hf_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=hf_dataset.column_names,
            desc="Tokenizing dataset",
        )
    else:
        tokenized = hf_dataset

    # Concatenate all token IDs into a flat 1D sequence
    all_tokens = []
    for item in tokenized["input_ids"]:
        all_tokens.extend(item)

    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)
    return TokenizedDataset(tokens_tensor, seq_len=seq_len)
