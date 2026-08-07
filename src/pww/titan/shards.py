"""Pre-tokenised corpus shards: the on-disk format DARL leases over.

Why this exists instead of tokenising at job start
--------------------------------------------------
`data/text.py` builds a Python ``list`` of every token in the corpus and then
calls ``torch.tensor`` on it. At ~28 bytes per Python int that is ~3.5 GB of
objects for WikiText-103's 118M tokens, and ~4.7 TB for C4-en's 156B -- so the
HuggingFace path cannot reach a real pre-training corpus at all, and it pays the
whole tokenisation cost again on every cluster at every job start.

Here the corpus is tokenised **once**, offline, into flat ``uint32`` files that
are ``mmap``ed at train time. Memory is O(1) in corpus size, startup is a stat
call, and the same bytes are read on LUMI and Snellius.

Windows, not documents
----------------------
The unit of both storage and leasing is a fixed **window** of ``seq_len + 1``
tokens (the extra token is the shifted label; see
`darl_dataloader.DARLWindowDataset`). This is load-bearing for two separate
reasons:

*Exactly-once needs a stable index space.* DARL leases positions in a permuted
index space and every site must agree on what position *p* means
(`darl/space.py`'s digest). Documents give a stable count, but token counts per
document depend on the tokeniser build, so a span of documents is not a
reproducible amount of work. A window index is exact and tokeniser-independent
once the shards exist.

*Data-parallel ranks must run the same number of steps.* A DiLoCo outer step is a
collective. If ranks derive different step counts from the same lease they hang
in the all-reduce rather than failing -- so a phase must divide into an identical
number of batches per rank, which variable-length documents cannot guarantee and
fixed windows do by construction.

No window straddles a shard boundary, so window *i* is one contiguous read from
exactly one file and the index maths is a bisect over cumulative counts.

Cross-site agreement
--------------------
`Manifest.digest` fingerprints the window geometry *and* the tokeniser that
produced it. Both sites check it before training (see `verify_compatible`), which
turns "two clusters silently trained on differently-tokenised text under the same
position numbers" from a corruption you find in the loss curve into a startup
error.
"""

from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..logging_utils import get_logger

logger = get_logger("pww.titan.shards")

MANIFEST_NAME = "manifest.json"
FORMAT = "pww-tokens-v1"

# A 128k-vocab tokeniser does not fit in uint16 (max 65535), so ids are stored as
# uint32 -- 4 bytes per token, i.e. ~8 GB per 2B tokens. uint16 is kept readable
# for corpora tokenised with a small-vocab tokeniser (GPT-2's 50257 fits).
DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


@dataclass(frozen=True)
class ShardInfo:
    """One file's contribution to the window index space."""

    path: str
    windows: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "windows": self.windows}


@dataclass(frozen=True)
class Manifest:
    """Geometry of a tokenised corpus, plus the tokeniser identity behind it.

    ``window`` is stored explicitly rather than derived as ``seq_len + 1`` so that
    a manifest read by a job configured with a different ``seq_len`` fails the
    compatibility check with a clear message instead of silently mis-slicing.
    """

    seq_len: int
    window: int
    dtype: str
    vocab_size: int
    tokenizer_repo: str
    tokenizer_sha256: str
    shards: tuple[ShardInfo, ...] = field(default_factory=tuple)
    format: str = FORMAT

    @property
    def num_windows(self) -> int:
        return sum(s.windows for s in self.shards)

    @property
    def total_tokens(self) -> int:
        return self.num_windows * self.window

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "seq_len": self.seq_len,
            "window": self.window,
            "dtype": self.dtype,
            "vocab_size": self.vocab_size,
            "tokenizer": {
                "repo_id": self.tokenizer_repo,
                "sha256": self.tokenizer_sha256,
            },
            "num_windows": self.num_windows,
            "total_tokens": self.total_tokens,
            "shards": [s.to_dict() for s in self.shards],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Manifest":
        fmt = raw.get("format", "")
        if fmt != FORMAT:
            raise ValueError(
                f"unsupported token shard format {fmt!r}, expected {FORMAT!r} -- "
                f"re-run pww.titan.tokenize_corpus to regenerate"
            )
        tok = raw.get("tokenizer", {})
        return cls(
            seq_len=int(raw["seq_len"]),
            window=int(raw["window"]),
            dtype=str(raw["dtype"]),
            vocab_size=int(raw["vocab_size"]),
            tokenizer_repo=str(tok.get("repo_id", "")),
            tokenizer_sha256=str(tok.get("sha256", "")),
            shards=tuple(
                ShardInfo(path=str(s["path"]), windows=int(s["windows"]))
                for s in raw["shards"]
            ),
        )

    def digest(self) -> str:
        """Fingerprint both sites must agree on before they share an index space.

        Deliberately covers the tokeniser hash and the window geometry but *not*
        the shard file list: two sites may legitimately stage the corpus into a
        different number of files as long as the resulting window sequence is the
        same. It does cover ``num_windows``, which is what the index space is
        built from.
        """
        h = hashlib.blake2b(digest_size=16)
        h.update(
            f"{self.format}:{self.seq_len}:{self.window}:{self.dtype}:"
            f"{self.vocab_size}:{self.tokenizer_sha256}:{self.num_windows}".encode()
        )
        return h.hexdigest()

    def describe(self) -> str:
        return (
            f"{self.num_windows:,} windows x {self.window} tokens = "
            f"{self.total_tokens:,} tokens across {len(self.shards)} shard(s), "
            f"vocab {self.vocab_size:,}, tokenizer {self.tokenizer_repo or '?'}"
        )


def write_manifest(directory: str | os.PathLike[str], manifest: Manifest) -> Path:
    path = Path(directory) / MANIFEST_NAME
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
    return path


def read_manifest(directory: str | os.PathLike[str]) -> Manifest:
    path = Path(directory) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"no {MANIFEST_NAME} in {directory} -- tokenise the corpus first:\n"
            f"  python3 -m pww.titan.tokenize_corpus --dataset c4_test "
            f"--tokenizer <dir> --seq-len <n> --out {directory}"
        )
    return Manifest.from_dict(json.loads(path.read_text()))


def file_sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_fingerprint(assets_path: str | os.PathLike[str]) -> str:
    """Hash of the tokeniser that produced (or will consume) a shard set.

    ``tokenizer.json`` alone: it is the whole fast-tokeniser definition (vocab and
    merges included), so hashing it is enough to detect a tokeniser swap, and it
    avoids depending on which other files a given HF repo happens to ship.
    """
    tok_json = Path(assets_path) / "tokenizer.json"
    if not tok_json.exists():
        raise FileNotFoundError(
            f"{tok_json} not found -- torchtitan's build_hf_tokenizer needs a fast "
            f"tokenizer.json. Fetch one with scripts/download_tokenizer.sh"
        )
    return file_sha256(tok_json)


class ShardedTokenCorpus:
    """Random access to window *i* of a tokenised corpus, via ``mmap``.

    ``np.memmap`` per shard, opened lazily and kept open. The OS page cache does
    the caching, which is what makes a leased span -- contiguous physical blocks,
    a few GB -- a sequential read rather than a scatter of seeks.
    """

    def __init__(self, directory: str | os.PathLike[str], manifest: Manifest | None = None):
        self.directory = Path(directory)
        self.manifest = manifest if manifest is not None else read_manifest(directory)
        if self.manifest.dtype not in DTYPES:
            raise ValueError(
                f"unknown shard dtype {self.manifest.dtype!r}, expected one of "
                f"{sorted(DTYPES)}"
            )
        self._dtype = DTYPES[self.manifest.dtype]
        self._maps: list[np.memmap | None] = [None] * len(self.manifest.shards)
        # Cumulative window counts, so locating window i is a bisect. Index k holds
        # the number of windows in shards [0, k), i.e. the first global window id
        # of shard k.
        self._offsets: list[int] = []
        total = 0
        for shard in self.manifest.shards:
            self._offsets.append(total)
            total += shard.windows
        self._total = total

    def __len__(self) -> int:
        return self._total

    @property
    def window(self) -> int:
        return self.manifest.window

    def _map(self, shard_index: int) -> np.memmap:
        existing = self._maps[shard_index]
        if existing is not None:
            return existing
        shard = self.manifest.shards[shard_index]
        path = self.directory / shard.path
        if not path.exists():
            raise FileNotFoundError(
                f"shard {path} listed in {MANIFEST_NAME} is missing -- the corpus "
                f"was not staged completely on this site"
            )
        expected = shard.windows * self.manifest.window
        actual = path.stat().st_size // np.dtype(self._dtype).itemsize
        if actual < expected:
            raise ValueError(
                f"shard {path} holds {actual:,} tokens but {MANIFEST_NAME} claims "
                f"{expected:,} -- truncated transfer"
            )
        created = np.memmap(path, dtype=self._dtype, mode="r", shape=(expected,))
        self._maps[shard_index] = created
        return created

    def window_tokens(self, index: int) -> np.ndarray:
        """Window ``index`` as a ``window``-length array. Copied out of the map.

        The copy is deliberate: the caller turns this into a torch tensor that
        outlives the call, and handing out a view into an mmap that may be closed
        (or, on some filesystems, remapped) is a use-after-free waiting to happen.
        The copy is 8 KB at seq_len 2048.
        """
        if not 0 <= index < self._total:
            raise IndexError(f"window {index} outside [0, {self._total})")
        shard_index = bisect_right(self._offsets, index) - 1
        local = index - self._offsets[shard_index]
        span = self.manifest.window
        start = local * span
        return np.asarray(self._map(shard_index)[start : start + span], dtype=np.int64)

    def close(self) -> None:
        for i, mapped in enumerate(self._maps):
            if mapped is not None:
                del mapped
                self._maps[i] = None


def verify_compatible(manifest: Manifest, *, seq_len: int, assets_path: str | None) -> None:
    """Fail at startup rather than in the loss curve.

    Checks the two things that silently corrupt a federated run: a ``seq_len``
    that does not match how the shards were cut, and a tokeniser that is not the
    one they were cut with (which would make every token id mean something else
    while the index space still looked valid).
    """
    if manifest.seq_len != seq_len:
        raise ValueError(
            f"token shards were cut for seq_len {manifest.seq_len} but this job is "
            f"configured for {seq_len} -- re-tokenise, or set training.seq_len "
            f"to {manifest.seq_len}"
        )
    if manifest.window != manifest.seq_len + 1:
        raise ValueError(
            f"manifest window {manifest.window} is not seq_len + 1 "
            f"({manifest.seq_len + 1})"
        )
    if not assets_path:
        return
    try:
        actual = tokenizer_fingerprint(assets_path)
    except FileNotFoundError:
        logger.warning(
            "cannot fingerprint the tokenizer at %s; skipping the shard/tokenizer "
            "consistency check", assets_path,
        )
        return
    if manifest.tokenizer_sha256 and actual != manifest.tokenizer_sha256:
        raise ValueError(
            f"tokenizer mismatch: shards in this corpus were produced with "
            f"tokenizer.json sha256 {manifest.tokenizer_sha256[:16]}... but "
            f"{assets_path} has {actual[:16]}.... Every token id would mean a "
            f"different piece of text. Re-tokenise the corpus, or point "
            f"model.hf_assets_path at the tokenizer the shards were built with "
            f"({manifest.tokenizer_repo or 'unrecorded'})."
        )
