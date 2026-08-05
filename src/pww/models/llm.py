"""LLM Architecture Factory supporting FlashAttention-2, PyTorch SDPA, FSDP, and HuggingFace CausalLM.

Supports loading small testing models (e.g. `gpt2`, `Qwen/Qwen2.5-0.5B`, `meta-llama/Llama-3.2-1B`)
and scaling to larger models (`meta-llama/Meta-Llama-3-8B`, `Qwen/Qwen2.5-7B`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..logging_utils import get_logger

logger = get_logger("pww.models.llm")


def build_llm(
    model_name_or_path: str = "gpt2",
    attn_implementation: str = "sdpa",
    dtype: str = "bf16",
    device_map: str | None = None,
) -> nn.Module:
    """Build a Causal Language Model using HuggingFace `AutoModelForCausalLM`."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            "HuggingFace `transformers` is required. "
            "Install it using: pip install transformers"
        )

    torch_dtype = torch.bfloat16 if dtype == "bf16" else (torch.float16 if dtype == "fp16" else torch.float32)

    logger.info(
        f"Building Causal LM '{model_name_or_path}' "
        f"(attn_impl='{attn_implementation}', dtype={torch_dtype})..."
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.warning(
            f"Failed to load '{model_name_or_path}' with attn_impl='{attn_implementation}' ({exc}). "
            "Falling back to default attention implementation..."
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model built successfully: {num_params / 1e6:.2f}M total parameters "
        f"({num_trainable / 1e6:.2f}M trainable)"
    )

    return model
