"""Token budget -> DARL_EPOCHS, and back-check against the corpus this campaign staged.

    python3 tests/test_plan_budget.py

No pytest, no network, no filesystem except a throwaway manifest.json this file
writes itself.
"""

from __future__ import annotations

import json
import signal
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASSED, FAILED = [], []
CHECK_TIMEOUT_S = 30


def check(name: str):
    def decorator(fn):
        def on_timeout(signum, frame):
            raise TimeoutError(f"exceeded {CHECK_TIMEOUT_S}s")

        previous = signal.signal(signal.SIGALRM, on_timeout)
        signal.alarm(CHECK_TIMEOUT_S)
        try:
            fn()
            PASSED.append(name)
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        return fn

    return decorator


def expect_raises(exc_type, fn, *, contains: str = ""):
    try:
        fn()
    except exc_type as exc:
        if contains and contains not in str(exc):
            raise AssertionError(
                f"raised {exc_type.__name__} but message lacked {contains!r}: {exc}"
            ) from None
        return
    raise AssertionError(f"expected {exc_type.__name__}, nothing raised")


from pww.plan.budget import plan_epochs, read_manifest  # noqa: E402

# This campaign's real, staged corpus (TODO.md §3: "Real C4, 5.65B tokens
# (2,756,597 windows), digest-matched at both sites"). Used as the reference
# point below rather than a round number, so a regression that quietly changes
# the arithmetic has a real figure to disagree with.
STAGED_WINDOWS = 2_756_597
SEQ_LEN = 2048
STAGED_TOKENS = STAGED_WINDOWS * SEQ_LEN  # 5,645,510,656 -- matches TODO.md's 5.65B


@check("one epoch of the staged corpus is ~5.65B tokens, matching TODO.md")
def _():
    b = plan_epochs(STAGED_TOKENS, STAGED_WINDOWS, SEQ_LEN)
    assert b.epochs_needed == 1, b.epochs_needed
    assert b.tokens_per_epoch == STAGED_TOKENS
    assert abs(b.tokens_per_epoch / 1e9 - 5.65) < 0.01, b.tokens_per_epoch


@check("20B tokens on the staged corpus needs 4 epochs, not 3")
def _():
    # 20e9 / 5,645,510,656 = 3.5425... -> ceil = 4. Pinned as a literal because
    # this is exactly the number a naive `round()` or `int()` gets wrong: both
    # would give 3 or 4 depending on convention, and 3 epochs is only 16.9B
    # tokens -- 3.1B short of the ask, with no error to say so.
    b = plan_epochs(20_000_000_000, STAGED_WINDOWS, SEQ_LEN)
    assert b.epochs_needed == 4, b.epochs_needed
    assert b.tokens_at_epochs > 20_000_000_000
    assert b.tokens_at_epochs == 4 * STAGED_TOKENS


@check("epochs_needed always rounds UP, never silently short-changes the budget")
def _():
    # Exactly on a boundary: must not need a 5th epoch it doesn't.
    exact = plan_epochs(3 * STAGED_TOKENS, STAGED_WINDOWS, SEQ_LEN)
    assert exact.epochs_needed == 3, exact.epochs_needed
    assert exact.overshoot == 0.0
    # One token over the boundary: must round up to 4, not truncate to 3.
    over = plan_epochs(3 * STAGED_TOKENS + 1, STAGED_WINDOWS, SEQ_LEN)
    assert over.epochs_needed == 4, over.epochs_needed
    assert over.tokens_at_epochs >= 3 * STAGED_TOKENS + 1


@check("tokens use seq_len, not manifest.window (which is seq_len + 1)")
def _():
    # window = seq_len + 1 in the real manifest schema (titan/shards.py's own
    # docstring). If this module ever reads `window` instead of `seq_len` by
    # mistake, tokens_per_epoch is ~0.05% too high and every other "tokens"
    # figure in this codebase (PLANNER.md, TODO.md) would disagree with it.
    b = plan_epochs(STAGED_TOKENS, STAGED_WINDOWS, seq_len=2048)
    assert b.tokens_per_epoch == STAGED_WINDOWS * 2048
    assert b.tokens_per_epoch != STAGED_WINDOWS * 2049


@check("non-positive inputs are refused, not silently coerced")
def _():
    expect_raises(ValueError, lambda: plan_epochs(0, STAGED_WINDOWS, SEQ_LEN),
                 contains="token_budget")
    expect_raises(ValueError, lambda: plan_epochs(1_000, 0, SEQ_LEN),
                 contains="num_windows")
    expect_raises(ValueError, lambda: plan_epochs(1_000, STAGED_WINDOWS, 0),
                 contains="seq_len")


@check("read_manifest reads the real Manifest.to_dict() schema")
def _():
    # The dict Manifest.to_dict() produces (pww/titan/shards.py), spelled out here
    # rather than imported: that module pulls in torchtitan, and this file's whole
    # point -- like the rest of tests/test_plan_*.py -- is running with no torch.
    manifest_dict = {
        "format": "pww-tokens-v1", "seq_len": SEQ_LEN, "window": SEQ_LEN + 1,
        "dtype": "uint32", "vocab_size": 131328,
        "tokenizer": {"repo_id": "pww/tokenizer-128k", "sha256": "deadbeef"},
        "num_windows": STAGED_WINDOWS, "total_tokens": STAGED_WINDOWS * (SEQ_LEN + 1),
        "shards": [{"path": "shard-000.bin", "windows": STAGED_WINDOWS}],
    }
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "manifest.json"
        path.write_text(json.dumps(manifest_dict))
        num_windows, seq_len = read_manifest(path)
    assert num_windows == STAGED_WINDOWS, num_windows
    assert seq_len == SEQ_LEN, seq_len


@check("read_manifest names the missing field rather than a bare KeyError")
def _():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "manifest.json"
        path.write_text(json.dumps({"format": "pww-tokens-v1"}))
        expect_raises(ValueError, lambda: read_manifest(path), contains="num_windows")


def main() -> int:
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for name, exc in FAILED:
        print(f"  {name}: {type(exc).__name__}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
