"""torchtitan train specs and dataset registrations for ProjectWorldWide.

Imported by torchtitan via ``--experimental.custom_import pww.titan``, which is
what registers everything below into torchtitan's own registries. Nothing here
modifies torchtitan; it only calls into the extension points
`torchtitan/docs/extension.md` documents.

Two train specs, differing only in where their data comes from:

    pww_qwen3        stock Qwen3 + the DARL-leased dataloader (cross-site runs)
    pww_qwen3_local  stock Qwen3 + torchtitan's own text dataloader (single-site
                     baseline, no coordinator needed)

Both use torchtitan's built-in Qwen3 flavors -- this repo does not hand-roll a
model shape -- wrapped in `PWWQwen3ModelArgs` so that vocabulary size and EOS id
follow the tokenizer instead of staying pinned to Qwen's own 151936/151645.
That matters here because the tokenizer in use is OpenEuroLLM's 128k
(`scripts/download_tokenizer.sh`), and a model built with a 151936-row embedding
against a 128k tokenizer wastes ~18% of its embedding parameters on rows no token
can ever index, while an id above the vocab would index out of bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.config import JobConfig
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.qwen3 import (
    Qwen3Model,
    Qwen3ModelArgs,
    parallelize_qwen3,
    qwen3_args,
)
from torchtitan.models.qwen3.model.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from ..logging_utils import get_logger
from .darl_dataloader import build_darl_dataloader

logger = get_logger("pww.titan")


@dataclass
class PWWQwen3ModelArgs(Qwen3ModelArgs):
    """Qwen3, with the vocabulary taken from the tokenizer actually in use.

    `update_from_config` is the only hook that sees the resolved `JobConfig`
    before the model is constructed on the meta device, and `model.hf_assets_path`
    is where torchtitan's own `build_hf_tokenizer` reads its tokenizer from -- so
    reading it here is guaranteed to describe the same tokenizer the dataloader
    and loss will use.
    """

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        super().update_from_config(job_config, **kwargs)

        assets_path = job_config.model.hf_assets_path
        if not assets_path:
            logger.warning(
                "model.hf_assets_path is unset, keeping vocab_size=%d from the "
                "Qwen3 flavor -- this is only correct with a Qwen3 tokenizer",
                self.vocab_size,
            )
            return

        from torchtitan.components.tokenizer import HuggingFaceTokenizer

        tokenizer = HuggingFaceTokenizer(assets_path)
        vocab_size = tokenizer.get_vocab_size()

        # Padded up so the embedding and output projection stay aligned and
        # divisible by any tensor-parallel degree. The 128k OpenEuroLLM tokenizer
        # reports 131073 ids -- odd, so TP > 1 would fail and every GEMM on that
        # dimension would be misaligned. See config.Titan.pad_vocab_to_multiple_of.
        multiple = getattr(getattr(job_config, "titan", None), "pad_vocab_to_multiple_of", 0)
        if multiple and multiple > 1:
            padded = -(-vocab_size // multiple) * multiple
            if padded != vocab_size:
                logger.info(
                    "padding vocab_size %d -> %d (multiple of %d); %d rows will "
                    "never be indexed by a token",
                    vocab_size, padded, multiple, padded - vocab_size,
                )
                vocab_size = padded

        if vocab_size != self.vocab_size:
            logger.info(
                "vocab_size %d -> %d, from the tokenizer at %s",
                self.vocab_size, vocab_size, assets_path,
            )
            self.vocab_size = vocab_size

        if tokenizer.eos_id is not None and tokenizer.eos_id != self.eos_id:
            logger.info("eos_id %d -> %d, from the tokenizer", self.eos_id, tokenizer.eos_id)
            self.eos_id = tokenizer.eos_id

        if self.vocab_size <= 0:
            raise ValueError(f"tokenizer at {assets_path} reports vocab_size {self.vocab_size}")


def _rebased_flavors() -> dict[str, PWWQwen3ModelArgs]:
    """torchtitan's Qwen3 flavor table, re-typed onto `PWWQwen3ModelArgs`.

    Field-by-field rather than `dataclasses.asdict`, which would recurse into
    `moe_args` and turn that nested dataclass into a plain dict.
    """
    rebased = {}
    for flavor, args in qwen3_args.items():
        shallow = {f.name: getattr(args, f.name) for f in fields(args)}
        rebased[flavor] = PWWQwen3ModelArgs(**shallow)
    return rebased


PWW_QWEN3_FLAVORS = _rebased_flavors()


def _spec(name: str, dataloader_fn) -> TrainSpec:
    return TrainSpec(
        model_cls=Qwen3Model,
        model_args=PWW_QWEN3_FLAVORS,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=dataloader_fn,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Qwen3StateDictAdapter,
    )


register_train_spec("pww_qwen3", _spec("pww_qwen3", build_darl_dataloader))
register_train_spec("pww_qwen3_local", _spec("pww_qwen3_local", build_text_dataloader))

# Import side effect: registers the pww_tokens dataset (and any staged C4 copies)
# into torchtitan's own DATASETS registry. See datasets.py.
from . import datasets as _datasets  # noqa: E402,F401

__all__ = ["PWWQwen3ModelArgs", "PWW_QWEN3_FLAVORS"]
