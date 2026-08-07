"""Dataset registrations added to torchtitan's own `DATASETS` registry.

Follows the extension pattern in `torchtitan/docs/datasets.md`: this module only
writes into that public dict, it does not modify torchtitan. Imported for its side
effect from `pww.titan`.

torchtitan already ships three C4 entries, and they are used as-is:

    c4             streams allenai/c4 from the hub -- login nodes only, since
                   LUMI and Snellius compute nodes have no internet
    c4_test        the bundled 2000-document fixture under
                   third_party/torchtitan/tests/assets/c4_test (4.7 MB, offline)
    c4_validation  allenai/c4's validation split, same internet caveat

Added here:

    c4_local       raw C4 shards staged onto this site's scratch, so the corpus
                   can be (re-)tokenised inside the facility with no egress
    pww_tokens     the pre-tokenised shard format DARL leases over; registered
                   only so that a config naming it gets a useful error instead of
                   a KeyError from _validate_dataset
"""

from __future__ import annotations

import os
from glob import glob
from typing import Any

from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS

# Per-site scratch path for staged raw corpora. Set from each site file (see
# sites/lumi.sh, sites/snellius.sh) rather than hardcoded, because the two
# facilities have entirely different scratch layouts.
C4_LOCAL_DIR = os.environ.get("PWW_C4_DIR", "./data/c4")


def _process_c4_text(sample: dict[str, Any]) -> str:
    return sample["text"]


def _load_local_c4(path: str):
    """Load raw C4 shards from a local directory.

    Accepts the layout `scripts/stage_c4.sh` produces (whatever allenai/c4's
    own `*.json.gz` files are named) as well as plain `.jsonl`, so a corpus
    staged by hand works without renaming anything.
    """
    from datasets import load_dataset

    patterns = ("*.json.gz", "*.jsonl.gz", "*.jsonl", "*.json")
    shards: list[str] = []
    for pattern in patterns:
        shards.extend(sorted(glob(os.path.join(path, pattern))))
    if not shards:
        raise FileNotFoundError(
            f"no C4 shards under {path} (looked for {', '.join(patterns)}) -- "
            f"stage them first with scripts/stage_c4.sh, or set PWW_C4_DIR"
        )
    return load_dataset("json", data_files=shards, split="train")


def _reject_pww_tokens(path: str):
    raise ValueError(
        f"'pww_tokens' at {path} is a pre-tokenised shard directory, not a "
        f"streamable text dataset. It is read directly by "
        f"pww.titan.darl_dataloader, so it belongs in [training] where the DARL "
        f"dataloader is active -- not in [validation], which uses torchtitan's "
        f"text dataloader. Set validation.dataset to c4_test or c4_validation."
    )


DATASETS["c4_local"] = DatasetConfig(
    path=C4_LOCAL_DIR,
    loader=_load_local_c4,
    sample_processor=_process_c4_text,
)

DATASETS["pww_tokens"] = DatasetConfig(
    path="",
    loader=_reject_pww_tokens,
    sample_processor=_process_c4_text,
)
