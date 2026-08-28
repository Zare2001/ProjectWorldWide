"""How many DARL epochs a token budget needs, given the corpus manifest.

    python3 -m pww.plan.budget --manifest .../c4-tokenizer-128k-2048/manifest.json \\
        --tokens 20e9
    python3 -m pww.plan.budget --num-windows 2756597 --tokens 20e9

Exists because `DARL_EPOCHS` (`scripts/central_node/start_central_services.sh`) is a
block-space REPEAT count, not a token count, and nothing today turns "I want N
tokens" into that integer. Getting it wrong in either direction is silent: too few
epochs ends the run early on `epoch_complete` with no warning that the config asked
for more; the epochs knob does not fail loudly like a missing file would.

TOKENS HERE MEANS `seq_len`, NOT `Manifest.window`. `window` is `seq_len + 1` (each
sample carries one extra token so the label can be the input shifted by one; see
`Manifest`'s own docstring in titan/shards.py) -- what a training step actually
consumes as INPUT is `seq_len` tokens. Every other token count in this codebase (the
5.65B in TODO.md, PLANNER.md's "one block = 1024 windows = 2,097,152 tokens") is
already `num_windows * seq_len`, so this module matches that convention rather than
introducing a second, ~0.05% larger "tokens" that would disagree with every existing
number for the same corpus.

DARL_EPOCHS is an integer -- the coordinator recycles the WHOLE block space at an
epoch boundary (`table.py: advance_epoch`), there is no fractional epoch -- so a
token budget almost never divides it evenly. `epochs_needed` rounds up and
`tokens_at_epochs` says what that integer actually buys, which is always >= the
budget. Rounding down would silently hand back fewer tokens than asked for with no
error at all: the coordinator would just report `epoch_complete` at a smaller number.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TokenBudget:
    token_budget: int
    tokens_per_epoch: int
    epochs_needed: int  # DARL_EPOCHS -- an integer repeat count, rounded UP
    tokens_at_epochs: int  # what that integer actually buys; >= token_budget
    overshoot: float  # tokens_at_epochs / token_budget - 1, always >= 0


def plan_epochs(token_budget: int, num_windows: int, seq_len: int) -> TokenBudget:
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    if num_windows <= 0:
        raise ValueError(f"num_windows must be positive, got {num_windows}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    tokens_per_epoch = num_windows * seq_len
    epochs_needed = math.ceil(token_budget / tokens_per_epoch)
    tokens_at_epochs = epochs_needed * tokens_per_epoch
    return TokenBudget(
        token_budget=token_budget,
        tokens_per_epoch=tokens_per_epoch,
        epochs_needed=epochs_needed,
        tokens_at_epochs=tokens_at_epochs,
        overshoot=tokens_at_epochs / token_budget - 1.0,
    )


def read_manifest(path: str | Path) -> tuple[int, int]:
    """(num_windows, seq_len) out of a DARL corpus manifest.json.

    Reads the same two fields `job_titan_central.sh`'s inline snippet reads
    `num_windows` from, so a manifest this cannot parse is one the job scripts
    cannot either -- this is not a second, looser schema.
    """
    raw = json.loads(Path(path).expanduser().read_text())
    missing = [k for k in ("num_windows", "seq_len") if k not in raw]
    if missing:
        raise ValueError(
            f"{path}: manifest missing {missing}; expected the DARL corpus "
            f"manifest schema (format, seq_len, window, num_windows, ...) -- see "
            f"pww.titan.shards.Manifest.to_dict")
    return int(raw["num_windows"]), int(raw["seq_len"])


def describe(budget: TokenBudget) -> str:
    return (
        f"{budget.token_budget / 1e9:.2f}B tokens requested, "
        f"{budget.tokens_per_epoch / 1e9:.2f}B tokens/epoch -> "
        f"DARL_EPOCHS={budget.epochs_needed} "
        f"({budget.tokens_at_epochs / 1e9:.2f}B tokens, "
        f"+{budget.overshoot * 100:.1f}% over the ask)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pww-plan-budget",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=float, required=True, metavar="N",
                     help="token budget, e.g. 20e9 for 20B")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", metavar="PATH",
                      help="DARL corpus manifest.json (the staged shard directory's "
                           "own manifest, e.g. $PWW_DATA_DIR/c4-.../manifest.json)")
    src.add_argument("--num-windows", type=int, metavar="N",
                      help="corpus size directly, when there is no manifest.json "
                           "on this machine to read (e.g. planning from a login "
                           "node that does not mount $PWW_DATA_DIR)")
    ap.add_argument("--seq-len", type=int, default=2048,
                     help="only with --num-windows; --manifest reads its own "
                          "seq_len instead of trusting this default")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.manifest:
        num_windows, seq_len = read_manifest(args.manifest)
    else:
        num_windows, seq_len = args.num_windows, args.seq_len
    budget = plan_epochs(int(args.tokens), num_windows, seq_len)
    print(describe(budget))
    print(f"  DARL_EPOCHS={budget.epochs_needed} "
          f"scripts/central_node/start_central_services.sh ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
