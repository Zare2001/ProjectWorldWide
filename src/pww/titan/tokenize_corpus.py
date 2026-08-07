"""Tokenise a corpus once into `shards.ShardedTokenCorpus` files.

Run this on a login or CPU node, once per (corpus, tokeniser, seq_len). Compute
nodes on LUMI and Snellius have no internet, so a corpus that is not already
staged on scratch cannot be fetched from inside a job -- and re-tokenising at job
start is what makes the HuggingFace path unable to hold C4 in memory at all (see
`shards` module docstring).

    # bundled 2000-document fixture, offline, seconds
    python3 -m pww.titan.tokenize_corpus --dataset c4_test \
        --tokenizer $TOKENIZER_DIR --seq-len 2048 --out $DATA_DIR/c4_test-2048

    # real C4 (streams from the hub, so: login node), 512 files ~= 180B tokens
    python3 -m pww.titan.tokenize_corpus --dataset c4 --split train \
        --tokenizer $TOKENIZER_DIR --seq-len 2048 --out $DATA_DIR/c4-2048 \
        --max-files 32

Document boundaries
-------------------
Each document is encoded with BOS and EOS and appended to a rolling buffer that is
cut into fixed windows. That EOS is not cosmetic: `data/text.py` concatenates
documents with no separator at all, so a model trained through it never learns
where a document ends and spends capacity modelling the seam between unrelated
web pages. torchtitan's own text dataloader adds both, and this matches it so a
DARL run and a stock torchtitan run see identically-framed text.

Determinism
-----------
The window sequence depends only on (dataset order, tokeniser, seq_len), so two
sites that run this with the same arguments get byte-identical shards. They are
not required to -- each site checks `Manifest.digest` against the coordinator, and
staging one site's output to the other with rsync is both cheaper and safer than
trusting two independent runs to agree.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from ..logging_utils import get_logger, setup_logging
from .shards import (
    DTYPES,
    Manifest,
    ShardInfo,
    tokenizer_fingerprint,
    write_manifest,
)

logger = get_logger("pww.titan.tokenize_corpus")

# Tokens per output file. At uint32 and 2**28 tokens a shard is 1 GiB, which is a
# comfortable unit for rsync between sites and keeps any single mmap well under
# the per-process mapping limits on Lustre.
DEFAULT_SHARD_TOKENS = 1 << 28


class _ShardWriter:
    """Appends windows to numbered files, rolling over at a token budget.

    Only whole windows are written, so no window straddles a shard boundary -- the
    property `ShardedTokenCorpus` relies on to make window lookup a single read.
    """

    def __init__(self, out_dir: Path, window: int, dtype: str, shard_tokens: int):
        self.out_dir = out_dir
        self.window = window
        self.np_dtype = DTYPES[dtype]
        # Round the budget down to a whole number of windows.
        self.windows_per_shard = max(1, shard_tokens // window)
        self.shards: list[ShardInfo] = []
        self._handle = None
        self._windows_in_shard = 0

    def _open_next(self) -> None:
        name = f"tokens-{len(self.shards):05d}.bin"
        self._path = self.out_dir / name
        self._handle = open(self._path, "wb")
        self._name = name
        self._windows_in_shard = 0

    def add(self, windows: np.ndarray) -> None:
        """``windows`` is (n, self.window) of token ids."""
        if windows.size == 0:
            return
        offset = 0
        while offset < windows.shape[0]:
            if self._handle is None:
                self._open_next()
            room = self.windows_per_shard - self._windows_in_shard
            take = min(room, windows.shape[0] - offset)
            chunk = np.ascontiguousarray(windows[offset : offset + take], dtype=self.np_dtype)
            self._handle.write(chunk.tobytes())
            self._windows_in_shard += take
            offset += take
            if self._windows_in_shard >= self.windows_per_shard:
                self._close_current()

    def _close_current(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None
        self.shards.append(ShardInfo(path=self._name, windows=self._windows_in_shard))
        logger.info(
            "wrote %s (%s windows, %.2f GiB)",
            self._name,
            f"{self._windows_in_shard:,}",
            self._windows_in_shard * self.window * np.dtype(self.np_dtype).itemsize / 2**30,
        )

    def close(self) -> list[ShardInfo]:
        self._close_current()
        return self.shards


def _load_documents(dataset: str, dataset_path: str | None, split: str, max_files: int):
    """Yield raw text, reusing torchtitan's own dataset registry.

    Going through `torchtitan.hf_datasets` rather than calling `load_dataset`
    directly means ``--dataset c4_test`` resolves to the same bundled fixture a
    stock torchtitan run uses, and any dataset registered by
    `pww.titan.datasets` works here with no extra wiring.
    """
    from torchtitan.hf_datasets.text_datasets import DATASETS, _validate_dataset

    name = dataset.lower()
    if name not in DATASETS:
        raise SystemExit(
            f"unknown dataset {dataset!r}. Registered: {sorted(DATASETS)}. "
            f"Add one in src/pww/titan/datasets.py."
        )
    path, loader, processor = _validate_dataset(name, dataset_path)

    # allenai/c4 is thousands of files; `--max-files` bounds a staging run to a
    # slice of it. The bundled fixture and any local shard set ignore this.
    if name in ("c4", "c4_validation") and max_files > 0:
        from datasets import load_dataset

        prefix = "en/c4-train" if split == "train" else "en/c4-validation"
        files = [f"{prefix}.{i:05d}-of-01024.json.gz" for i in range(max_files)]
        logger.info("streaming %d file(s) of %s from %s", len(files), split, path)
        ds = load_dataset(path, data_files={split: files}, split=split, streaming=True)
    else:
        ds = loader(path)

    for sample in ds:
        yield processor(sample)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Tokenise a corpus into pww token shards for DARL leasing"
    )
    p.add_argument("--dataset", default="c4_test",
                   help="name in torchtitan's DATASETS registry (c4, c4_test, ...)")
    p.add_argument("--dataset-path", default=None,
                   help="override the registry path (local staged copy)")
    p.add_argument("--split", default="train")
    p.add_argument("--tokenizer", required=True,
                   help="directory holding tokenizer.json (scripts/download_tokenizer.sh)")
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--out", required=True, help="output directory for shards + manifest")
    p.add_argument("--dtype", default="uint32", choices=sorted(DTYPES),
                   help="uint32 unless the vocab fits in uint16 (<=65535)")
    p.add_argument("--shard-tokens", type=int, default=DEFAULT_SHARD_TOKENS,
                   help="tokens per output file")
    p.add_argument("--max-files", type=int, default=0,
                   help="for allenai/c4: how many source files to take (0 = all)")
    p.add_argument("--max-windows", type=int, default=0,
                   help="stop after this many windows (0 = no limit)")
    p.add_argument("--log-every", type=int, default=100_000,
                   help="progress line every N documents")
    args = p.parse_args(argv)

    setup_logging(rank=0)

    from torchtitan.components.tokenizer import HuggingFaceTokenizer

    tokenizer = HuggingFaceTokenizer(args.tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    fingerprint = tokenizer_fingerprint(args.tokenizer)
    logger.info(
        "tokenizer %s: vocab %s, tokenizer.json sha256 %s",
        args.tokenizer, f"{vocab_size:,}", fingerprint[:16] + "...",
    )

    if args.dtype == "uint16" and vocab_size > 65535:
        raise SystemExit(
            f"--dtype uint16 cannot hold a {vocab_size:,}-token vocab; use uint32"
        )

    window = args.seq_len + 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = _ShardWriter(out_dir, window, args.dtype, args.shard_tokens)
    buffer: list[int] = []
    docs = 0
    windows_written = 0
    started = time.monotonic()

    for text in _load_documents(args.dataset, args.dataset_path, args.split, args.max_files):
        # add_bos/add_eos frame each document, so the model sees where one ends.
        buffer.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        docs += 1

        if len(buffer) >= window:
            full = len(buffer) // window
            if args.max_windows:
                full = min(full, args.max_windows - windows_written)
            if full > 0:
                usable = full * window
                writer.add(np.asarray(buffer[:usable], dtype=np.int64).reshape(full, window))
                windows_written += full
                # The tail carries into the next document rather than being
                # dropped: dropping it would silently discard up to seq_len tokens
                # per document, which over C4 is a large fraction of the corpus.
                del buffer[:usable]

        if args.max_windows and windows_written >= args.max_windows:
            logger.info("reached --max-windows %d, stopping", args.max_windows)
            break

        if args.log_every and docs % args.log_every == 0:
            rate = windows_written * window / max(1e-6, time.monotonic() - started)
            logger.info(
                "%s docs -> %s windows (%.1fM tok/s)",
                f"{docs:,}", f"{windows_written:,}", rate / 1e6,
            )

    shards = writer.close()
    if not shards:
        raise SystemExit(
            f"no complete windows produced from {args.dataset} -- the corpus has "
            f"fewer than {window} tokens, or --max-windows was 0"
        )

    manifest = Manifest(
        seq_len=args.seq_len,
        window=window,
        dtype=args.dtype,
        vocab_size=vocab_size,
        tokenizer_repo=str(args.tokenizer),
        tokenizer_sha256=fingerprint,
        shards=tuple(shards),
    )
    path = write_manifest(out_dir, manifest)

    logger.info("%s", manifest.describe())
    logger.info("digest %s (both sites must match)", manifest.digest())
    logger.info(
        "%s docs consumed, %s tokens dropped as a final partial window",
        f"{docs:,}", f"{len(buffer):,}",
    )
    logger.info("manifest written to %s", path)
    logger.info(
        "point the run at it with: --training.dataset pww_tokens "
        "--training.dataset_path %s", out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
