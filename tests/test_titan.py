"""CPU-only tests for the torchtitan path -- login node, no allocation, no GPUs.

    source env.sh && pww_run python3 tests/test_titan.py

Same shape as tests/test_darl.py: increasing cost, no GPU anywhere.

  shards      the on-disk token format -- window indexing across shard
              boundaries, and the two mismatches (seq_len, tokenizer) that
              would otherwise corrupt a federated run silently
  leasing     the DARL dataloader against a real coordinator on an ephemeral
              port, with the rank broadcast stubbed so several ranks run in one
              process. This is the exactly-once check at the granularity that
              actually matters: windows fed to the model.
  wire        the parameter codec (float16, float32, and the bfloat16 bit-pattern
              hop), the float16 aggregation regression that overflowed to inf before
              central/strategy.py did its arithmetic in float32, and the scatter key
              set -- which must exclude non-persistent buffers, because including
              them filled the RoPE cache with uninitialised memory

What this file cannot cover: FSDP2 wrapping, the real Qwen3 forward pass, and the
DTensor gather/scatter in titan/params.py all need GPUs and a process group. Those
are what configs/titan/qwen3_0.6b_smoke.toml is for. Note that the scatter *key
selection* is covered here even though the scatter itself is not -- that was where
the bug lived, and it is pure module introspection.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "torchtitan"))

PASSED, FAILED = [], []

# Several checks drive a real coordinator over HTTP, where the failure mode for a
# bug in the lease/phase handshake is a wait rather than an exception -- a drained
# pool retries forever by design. A wall-clock alarm turns that into a reported
# failure instead of a suite that never finishes.
CHECK_TIMEOUT_S = 60


def check(name: str):
    def decorator(fn):
        import signal

        def on_timeout(signum, frame):
            raise TimeoutError(
                f"exceeded {CHECK_TIMEOUT_S}s -- probably waiting on a lease that "
                f"is never granted"
            )

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


# --- shards ---------------------------------------------------------------

from pww.titan.shards import (  # noqa: E402
    Manifest,
    ShardedTokenCorpus,
    ShardInfo,
    read_manifest,
    verify_compatible,
    write_manifest,
)


def make_corpus(directory: Path, *, seq_len: int, windows_per_shard: list[int],
                sha: str = "deadbeef") -> Manifest:
    """Write shards whose token values encode their own global window index.

    Window i is filled with the value i, so a read that lands in the wrong shard
    or at the wrong offset is caught by value rather than by shape.
    """
    window = seq_len + 1
    shards = []
    global_index = 0
    for shard_index, count in enumerate(windows_per_shard):
        name = f"tokens-{shard_index:05d}.bin"
        block = np.empty((count, window), dtype=np.uint32)
        for local in range(count):
            block[local, :] = global_index
            global_index += 1
        (directory / name).write_bytes(block.tobytes())
        shards.append(ShardInfo(path=name, windows=count))
    manifest = Manifest(
        seq_len=seq_len, window=window, dtype="uint32", vocab_size=1000,
        tokenizer_repo="test", tokenizer_sha256=sha, shards=tuple(shards),
    )
    write_manifest(directory, manifest)
    return manifest


@check("shard windows read back correctly across shard boundaries")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_corpus(directory, seq_len=7, windows_per_shard=[5, 3, 4])
        corpus = ShardedTokenCorpus(directory)
        assert len(corpus) == 12, len(corpus)
        for index in range(12):
            tokens = corpus.window_tokens(index)
            assert tokens.shape == (8,), tokens.shape
            # The value is the global window index, so this catches an off-by-one
            # in the bisect as well as a wrong shard.
            assert set(tokens.tolist()) == {index}, (index, tokens[:3])
        expect_raises(IndexError, lambda: corpus.window_tokens(12))
        expect_raises(IndexError, lambda: corpus.window_tokens(-1))


@check("manifest round-trips and its digest ignores shard layout but not geometry")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c = Path(tmp) / "a", Path(tmp) / "b", Path(tmp) / "c"
        for d in (a, b, c):
            d.mkdir()
        # Same windows, different file split -- two sites may legitimately stage a
        # corpus into different numbers of files.
        one = make_corpus(a, seq_len=7, windows_per_shard=[12])
        split = make_corpus(b, seq_len=7, windows_per_shard=[5, 3, 4])
        assert one.digest() == split.digest(), "shard layout must not change the digest"

        assert read_manifest(a).to_dict() == one.to_dict()

        # A different tokenizer must change it: that is the whole point.
        other = make_corpus(c, seq_len=7, windows_per_shard=[12], sha="cafe")
        assert one.digest() != other.digest(), "tokenizer must be in the digest"


@check("verify_compatible rejects a seq_len that does not match the shards")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        manifest = make_corpus(directory, seq_len=7, windows_per_shard=[4])
        verify_compatible(manifest, seq_len=7, assets_path=None)
        expect_raises(
            ValueError,
            lambda: verify_compatible(manifest, seq_len=2048, assets_path=None),
            contains="re-tokenise",
        )


@check("verify_compatible rejects a tokenizer the shards were not built with")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        assets = directory / "tok"
        assets.mkdir()
        (assets / "tokenizer.json").write_text('{"fake": "tokenizer"}')

        from pww.titan.shards import tokenizer_fingerprint

        actual = tokenizer_fingerprint(assets)

        (directory / "good").mkdir()
        good = make_corpus(directory / "good", seq_len=3, windows_per_shard=[2],
                           sha=actual)
        verify_compatible(good, seq_len=3, assets_path=str(assets))

        (directory / "bad").mkdir()
        wrong = make_corpus(directory / "bad", seq_len=3, windows_per_shard=[2],
                            sha="0" * 64)
        expect_raises(
            ValueError,
            lambda: verify_compatible(wrong, seq_len=3, assets_path=str(assets)),
            contains="tokenizer mismatch",
        )


@check("truncated shard is detected rather than read as zeros")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_corpus(directory, seq_len=7, windows_per_shard=[6])
        target = directory / "tokens-00000.bin"
        data = target.read_bytes()
        target.write_bytes(data[: len(data) // 2])
        corpus = ShardedTokenCorpus(directory)
        expect_raises(ValueError, lambda: corpus.window_tokens(0), contains="truncated")


# --- leasing: the DARL dataloader over real shards, real coordinator -------

from pww.darl.client import LeaseClient, LeaseSession  # noqa: E402
from pww.darl.server import Coordinator, make_server  # noqa: E402
from pww.darl.space import BlockSpace  # noqa: E402
from pww.darl.table import LeaseTable  # noqa: E402
from pww.darl.torch_data import DARLDataSource  # noqa: E402
from pww.titan.darl_dataloader import DARLWindowDataset  # noqa: E402


class _Server:
    """A real coordinator on an ephemeral port, for the duration of a `with`.

    Same helper as tests/test_darl.py's -- `digest` is a `LeaseTable` argument, so
    registration verifies the block space the way it does in a real run.
    """

    def __init__(self, blocks: int, token: str = "", **kwargs):
        import threading

        # A long TTL on purpose. These checks pass heartbeat=False to keep them
        # deterministic, which means nothing refreshes a lease -- so with a short TTL a
        # slow check lets the coordinator reclaim blocks mid-phase and grant them
        # again, and the exactly-once assertions fail on a timing accident rather than a
        # bug. Expiry, stealing and quarantine are covered properly in test_darl.py
        # with an injected clock; here the subject is coverage. Real runs leave
        # heartbeats on (LeaseSession defaults to them, and build_darl_dataloader does
        # not turn them off), so a live cluster refreshes its own leases.
        kwargs.setdefault("min_ttl", 600.0)
        # first_grant_fraction defaults to handing a new cluster only part of what
        # it asks for, which is right in production (it has not proven its
        # throughput yet) but makes a single-cluster test need extra phases.
        kwargs.setdefault("first_grant_fraction", 1.0)
        self.coordinator = Coordinator(LeaseTable(blocks, **kwargs), None)
        self.httpd = make_server(self.coordinator, host="127.0.0.1", port=0, token=token)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def _run_ranks(corpus_dir: Path, space: BlockSpace, url: str, world_size: int,
               max_epochs: int = 1) -> list[list[int]]:
    """Drive `world_size` ranks in one process and return each rank's windows.

    The ranks share a stubbed broadcast rather than a process group. Getting this
    stub right matters more than it looks: `dist.broadcast_object_list` delivers
    **every** message to **every** rank, in order. An earlier version kept only the
    most recent payload, so if the leader reached phase N+1 before a follower had
    consumed phase N, the follower silently skipped a phase -- which both lost
    coverage and let the leader drain the lease pool while a follower lagged, so
    `acquire(wait=True)` span forever and the check timed out. Intermittently, which
    is the worst way to find out.

    Modelled here as an ordered log: the leader appends, and each rank reads the
    entry at its own position. `DARLDataSource` exposes this seam precisely so the
    sharding maths can be tested in one process.
    """
    corpus = ShardedTokenCorpus(corpus_dir)
    log: list[object] = []
    positions = [0] * world_size

    def broadcast_for(rank: int):
        def broadcast(payload, is_leader):
            if is_leader:
                log.append(payload)
            index = positions[rank]
            if index >= len(log):
                raise AssertionError(
                    f"rank {rank} wants broadcast #{index} but the leader has only "
                    f"published {len(log)} -- the ranks are out of lockstep"
                )
            positions[rank] += 1
            return log[index]

        return broadcast

    client = LeaseClient(url, "test-cluster", retries=2, timeout=5.0)
    session = LeaseSession(client, space, blocks_per_phase=2, ranks=world_size,
                           heartbeat=False)

    sources = []
    for rank in range(world_size):
        sources.append(
            DARLDataSource(
                space,
                session if rank == 0 else None,
                rank=rank,
                world_size=world_size,
                broadcast=broadcast_for(rank),
                seed=space.seed,
                blocks_per_phase=2,
            )
        )

    datasets = [
        # 'consumption': there is no trainer here, so nothing would ever write a
        # checkpoint, and under the default 'checkpoint' policy every span would
        # stay held until the pool drained and acquire spun forever.
        DARLWindowDataset(corpus, source, max_epochs=max_epochs,
                          commit_policy="consumption")
        for source in sources
    ]
    iterators = [iter(d) for d in datasets]
    per_rank: list[list[int]] = [[] for _ in range(world_size)]

    # Round-robin so the ranks stay in lockstep at each phase boundary, which is
    # what the shared broadcast stub models. Draining rank 0 fully first would let
    # it acquire every span before rank 1 saw any of them.
    live = list(range(world_size))
    while live:
        for rank in list(live):
            try:
                tokens, _labels = next(iterators[rank])
            except StopIteration:
                live.remove(rank)
                continue
            # Window i was written filled with the value i (see make_corpus).
            per_rank[rank].append(int(tokens["input"][0]))
    session.close(release=False)
    return per_rank


@check("DARL dataloader covers every window exactly once across ranks")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        # 24 windows, 4 per block -> 6 blocks. Two ranks, 2 blocks per phase.
        make_corpus(directory, seq_len=3, windows_per_shard=[10, 14])
        space = BlockSpace(num_samples=24, block_size=4, seed=7)
        assert space.num_blocks == 6, space.num_blocks

        with _Server(space.num_blocks, digest=space.digest()) as server:
            per_rank = _run_ranks(directory, space, server.url, world_size=2)

        seen = [w for rank in per_rank for w in rank]
        assert len(seen) == len(set(seen)), "a window was fed to the model twice"
        assert set(seen) == set(range(24)), (
            f"coverage gap: missing {sorted(set(range(24)) - set(seen))}"
        )
        # Equal counts per rank is not cosmetic: the DiLoCo outer step is a
        # collective, so ranks that disagree about how many steps a phase holds
        # hang in the all-reduce instead of failing.
        assert len(per_rank[0]) == len(per_rank[1]), [len(r) for r in per_rank]


@check("exhausted corpus ends iteration instead of yielding nothing forever")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_corpus(directory, seq_len=3, windows_per_shard=[8])
        space = BlockSpace(num_samples=8, block_size=4, seed=1)

        with _Server(space.num_blocks, digest=space.digest()) as server:
            corpus = ShardedTokenCorpus(directory)
            client = LeaseClient(server.url, "solo", retries=2, timeout=5.0)
            session = LeaseSession(client, space, blocks_per_phase=1, ranks=1,
                                   heartbeat=False)
            source = DARLDataSource(space, session, rank=0, world_size=1,
                                    broadcast=lambda payload, leader: payload,
                                    blocks_per_phase=1)
            dataset = DARLWindowDataset(corpus, source, max_epochs=1,
                                        commit_policy="consumption")

            windows = [int(t["input"][0]) for t, _ in dataset]
            assert len(windows) == 8, windows
            assert sorted(windows) == list(range(8)), windows
            assert dataset.exhausted, "exhaustion must be visible to the caller"
            # And it must stay stopped -- the old HF path kept returning empty work
            # for 23 rounds while the log showed no failures.
            again = list(dataset)
            assert again == [], f"restarted after exhaustion: {len(again)} items"
            session.close(release=False)


@check("dataloader state survives a save/load round trip")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        make_corpus(directory, seq_len=3, windows_per_shard=[8])
        space = BlockSpace(num_samples=8, block_size=4, seed=1)
        with _Server(space.num_blocks, digest=space.digest()) as server:
            corpus = ShardedTokenCorpus(directory)
            client = LeaseClient(server.url, "solo", retries=2, timeout=5.0)
            session = LeaseSession(client, space, blocks_per_phase=1, ranks=1,
                                   heartbeat=False)
            source = DARLDataSource(space, session, rank=0, world_size=1,
                                    broadcast=lambda p, leader: p, blocks_per_phase=1)
            dataset = DARLWindowDataset(corpus, source, max_epochs=1,
                                        commit_policy="consumption")
            iterator = iter(dataset)
            next(iterator)
            state = dataset.state_dict()
            assert "darl" in state and "windows_yielded" in state

            restored = DARLWindowDataset(corpus, source, max_epochs=1,
                                         commit_policy="consumption")
            restored.load_state_dict(state)
            assert restored.windows_yielded == dataset.windows_yielded
            # Deliberately absent: the offset inside the current phase. On restart
            # the held leases have expired and their blocks are back in the pool,
            # so replaying a phase prefix would double-train them.
            assert "phase_offset" not in state
            session.close(release=False)


# --- wire: parameter codec and the float16 aggregation regression ----------

import torch  # noqa: E402

from pww.titan.params import ParameterCodec, outer_agreement  # noqa: E402


@check("parameter codec round-trips in float32 and orders keys deterministically")
def _():
    state = {
        "layers.1.w": torch.randn(3, 4),
        "layers.0.w": torch.randn(2, 2),
        "norm.scale": torch.randn(5),
    }
    codec = ParameterCodec.from_state_dict(state, wire_dtype="float32")
    # Sorted, so two clusters enumerating their parameters independently agree --
    # Flower carries a bare list and averages position-wise, so a different order
    # would mix unrelated tensors together.
    assert codec.keys == ("layers.0.w", "layers.1.w", "norm.scale"), codec.keys
    assert codec.numel == 12 + 4 + 5

    arrays = codec.encode(state)
    back = codec.decode(arrays)
    for key in state:
        assert torch.equal(state[key], back[key]), key


@check("codec rejects a parameter list from a differently-shaped model")
def _():
    state = {"a": torch.randn(2, 2), "b": torch.randn(3)}
    codec = ParameterCodec.from_state_dict(state)
    expect_raises(
        ValueError, lambda: codec.decode([np.zeros((2, 2), dtype=np.float16)]),
        contains="this model has 2 parameters",
    )
    expect_raises(
        ValueError,
        lambda: codec.decode(
            [np.zeros((9, 9), dtype=np.float16), np.zeros((3,), dtype=np.float16)]
        ),
        contains="arrived with shape",
    )


@check("bfloat16 weights survive the float16 wire hop, except below fp16's floor")
def _():
    """float16 has more mantissa than bfloat16 but far less range.

    Exact for weights of any normal magnitude -- bfloat16's 8 mantissa bits fit inside
    float16's 11. But bfloat16 reaches down to ~1.2e-38 while float16's smallest
    subnormal is 5.96e-08, so magnitudes under that get rounded to it or to zero.
    Measured, not assumed: 1 element in 819,200 of `randn(64, 64)` moves, which is why
    asserting exactness here failed once every few hundred runs rather than never.

    Harmless for training -- a weight of 1e-08 contributes nothing to a forward pass --
    but a property of the inline transport worth pinning rather than an invariant to
    rely on.
    """
    state = {"w": torch.randn(64, 64, dtype=torch.bfloat16)}
    codec = ParameterCodec.from_state_dict(state, wire_dtype="float16")
    back = codec.decode(codec.encode(state))
    assert back["w"].dtype == torch.bfloat16

    # Any error is bounded by float16's subnormal floor, not by the scale of the
    # weights -- so nothing of consequence moves.
    error = (state["w"].to(torch.float32) - back["w"].to(torch.float32)).abs()
    assert error.max() <= 6e-8, error.max()

    def hop(values: list[float], wire: str) -> list[float]:
        one = {"w": torch.tensor([values], dtype=torch.bfloat16)}
        codec = ParameterCodec.from_state_dict(one, wire_dtype=wire)
        return codec.decode(codec.encode(one))["w"][0].tolist()

    # Normal magnitudes, including well below float16's min *normal* of 6.1e-05:
    # subnormals are still exact until the mantissa runs out.
    normal = [1.5, -0.25, 3.75e-3, -7.0, 1e-4, 6e-5]
    reference = torch.tensor([normal], dtype=torch.bfloat16)[0].tolist()
    assert hop(normal, "float16") == reference, hop(normal, "float16")

    # Below the floor, the measured behaviour: 3e-08 rounds up to float16's smallest
    # subnormal, and 2e-08 and under go to zero. Neither is a value any real weight
    # takes, but the boundary is documented here rather than discovered in a flake.
    assert hop([3e-8], "float16")[0] == 5.960464477539063e-08
    assert hop([2e-8], "float16")[0] == 0.0
    assert hop([1e-10], "float16")[0] == 0.0

    # float32 on the wire keeps all of it, at twice the bytes.
    assert hop([3e-8, 2e-8, 1e-10], "float32") == torch.tensor(
        [[3e-8, 2e-8, 1e-10]], dtype=torch.bfloat16
    )[0].tolist()


@check("wire_dtype rejects anything it cannot carry")
def _():
    state = {"w": torch.randn(4)}
    expect_raises(
        ValueError,
        lambda: ParameterCodec.from_state_dict(state, wire_dtype="float64"),
        contains="wire_dtype must be one of",
    )


@check("bfloat16 crosses the wire as uint16 bit patterns, exactly")
def _():
    """The bf16 wire path, which the config now actually uses.

    numpy has no bfloat16, so the codec ships the raw 16 bits as uint16 and both ends
    *reinterpret* them (`.view`) rather than convert (`.astype`). Getting that backwards
    is not a rounding error, it is reading exponent bits as an integer: a weight of 0.5
    becomes 16128.0. It cost a run -- see central/strategy.py::_from_wire -- so pin both
    the exactness and the dtype on the wire.
    """
    state = {"w": torch.randn(64, 64, dtype=torch.bfloat16)}
    codec = ParameterCodec.from_state_dict(state, wire_dtype="bfloat16")

    arrays = codec.encode(state)
    assert arrays[0].dtype == np.uint16, arrays[0].dtype
    # Same element count as the tensor: uint16 is 2 bytes and so is bfloat16, so this is
    # a reinterpretation and not a repacking.
    assert arrays[0].shape == (64, 64), arrays[0].shape
    assert codec.wire_bytes == 64 * 64 * 2, codec.wire_bytes

    back = codec.decode(arrays)
    # Exact, unlike the float16 hop above: no dtype change happens at all.
    assert torch.equal(state["w"], back["w"]), (state["w"] - back["w"]).abs().max()

    # The failure that actually happened: treating the bit patterns as values.
    wrong = arrays[0].astype(np.float32)
    assert wrong.max() > 1e3, (
        "astype on the bit patterns should produce absurd magnitudes -- if it does not, "
        "this test no longer demonstrates the bug it exists to pin"
    )


@check("scatter skips non-persistent buffers instead of filling them with garbage")
def _():
    """The rope_cache corruption, at the only granularity a CPU test can reach.

    `scatter_full_state` needs CUDA and a process group, but the bug was entirely in
    *which keys it iterated*: it used `named_parameters() | named_buffers()`, and
    `named_buffers()` reports non-persistent buffers that `state_dict()` -- and so the
    codec, and so the wire -- never carries. The key with nothing to load then took the
    worker-rank branch and copied `torch.empty` into the buffer.

    On Qwen3 that buffer is `rope_cache`. Zeros disable RoPE silently and the run
    plateaus at unigram entropy; dirty memory is a nan on the first microbatch. So the
    invariant is: every key scatter loads must be a key the codec sends.
    """
    from pww.titan.params import keys_to_load

    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embeddings = torch.nn.Embedding(8, 4)
            self.register_buffer("rope_cache", torch.arange(8.0), persistent=False)
            self.register_buffer("kept", torch.ones(3), persistent=True)
            self.output = torch.nn.Linear(4, 8, bias=False)
            # What torchtitan does for the 0.6B flavor: tie after construction.
            self.output.weight = self.tok_embeddings.weight

    model = Tied()

    # The premise: the two enumerations disagree, and they disagree about rope_cache.
    owned = set(dict(model.named_parameters())) | set(dict(model.named_buffers()))
    shipped = set(model.state_dict())
    assert "rope_cache" in owned, "premise broken: named_buffers hides non-persistent"
    assert "rope_cache" not in shipped, "premise broken: state_dict now ships it"
    # And the tied alias runs the other way: shipped but not separately owned.
    assert "output.weight" in shipped and "output.weight" not in owned

    loaded = keys_to_load(model)
    assert "rope_cache" not in loaded, (
        f"scatter would fill rope_cache from nothing: {loaded}"
    )
    assert "output.weight" not in loaded, "tied alias must not be loaded twice"
    assert "kept" in loaded, "persistent buffers still have to be loaded"
    assert "tok_embeddings.weight" in loaded

    # Every key scatter loads is a key the wire carries. This is the actual invariant.
    codec = ParameterCodec.from_state_dict(model.state_dict(), wire_dtype="float32")
    assert set(loaded) <= set(codec.keys), set(loaded) - set(codec.keys)

    # Rank-independent and stable: worker ranks reach the same broadcasts in the same
    # order without ever seeing the incoming dict.
    assert loaded == sorted(loaded), loaded
    assert keys_to_load(model) == loaded


@check("outer_agreement reports drift relative to the weights")
def _():
    reference = {"w": torch.ones(100)}
    delta = {"w": torch.full((100,), 0.01)}
    stats = outer_agreement(delta, reference)
    # |delta| = 0.01*10 = 0.1, |ref| = 10, ratio 0.01
    assert abs(stats["drift_ratio"] - 0.01) < 1e-5, stats
    zero = outer_agreement({"w": torch.zeros(4)}, {"w": torch.zeros(4)})
    assert zero["drift_ratio"] == 0.0, "must not divide by zero"


@check("FedMom aggregation of float16 clients does not overflow to inf")
def _():
    """The regression behind central/strategy.py's float32 arithmetic.

    flwr's own `aggregate` forms `layer * num_examples` before dividing. Once
    clients report token counts (order 1e6) and send float16 to fit a 0.6B model
    under the 2 GiB gRPC cap, that intermediate saturates float16's 65504 and every
    parameter becomes inf on the very first round.
    """
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.common import FitRes, Status, Code

    from pww.central.strategy import FedMom

    weights = np.full((8,), 0.5, dtype=np.float16)
    initial = ndarrays_to_parameters([weights.copy()])

    strategy = FedMom(
        min_fit_clients=2, min_evaluate_clients=2, min_available_clients=2,
        initial_parameters=initial, server_learning_rate=1.0, server_momentum=0.9,
    )
    # Seed the float32 global model the way configure_fit does on round 1.
    strategy._global_fp32 = [weights.astype(np.float32)]

    def result(value: float, examples: int):
        return None, FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters(
                [np.full((8,), value, dtype=np.float16)]
            ),
            num_examples=examples,
            metrics={},
        )

    # Token counts, not sample counts -- this is what the client now reports.
    aggregated, _ = strategy.aggregate_fit(
        1, [result(0.6, 1_600_000), result(0.4, 2_400_000)], []
    )
    assert aggregated is not None
    out = parameters_to_ndarrays(aggregated)[0].astype(np.float32)
    assert np.all(np.isfinite(out)), f"overflowed: {out}"

    # Weighted mean is 0.6*0.4 + 0.4*0.6 = 0.48, so the pseudo-gradient is
    # 0.5 - 0.48 = 0.02 and v_next = 0.48. First round has no previous v, so
    # v_prev = w_t and w_next = 0.48 + 0.9*(0.48 - 0.5) = 0.462.
    assert np.allclose(out, 0.462, atol=2e-3), out
    # Momentum state must stay float32 or it quantises away over hundreds of rounds.
    assert strategy.v_vector[0].dtype == np.float32, strategy.v_vector[0].dtype
    assert strategy._global_fp32[0].dtype == np.float32


@check("a round where every cluster trained nothing does not move the model")
def _():
    """The 23-wasted-rounds failure, as a test.

    When DARL runs dry every client reports 0 examples. The old path floored that
    to 1 with loss 0.0, so FedMom kept averaging untouched weights and the run
    continued as a no-op with zero failures logged.
    """
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.common import FitRes, Status, Code

    from pww.central.strategy import FedMom

    weights = np.full((4,), 0.25, dtype=np.float32)
    initial = ndarrays_to_parameters([weights.copy()])
    strategy = FedMom(min_fit_clients=2, min_evaluate_clients=2,
                      min_available_clients=2, initial_parameters=initial)
    strategy._global_fp32 = [weights.copy()]

    def empty():
        return None, FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters([np.zeros((4,), dtype=np.float32)]),
            num_examples=0,
            metrics={"exhausted": True},
        )

    aggregated, _ = strategy.aggregate_fit(1, [empty(), empty()], [])
    out = parameters_to_ndarrays(aggregated)[0]
    assert np.allclose(out, 0.25), f"empty round moved the model to {out}"


# --- config -----------------------------------------------------------------


@check("every shipped titan config parses through torchtitan's ConfigManager")
def _():
    from torchtitan.config.manager import ConfigManager

    configs = sorted((ROOT / "configs" / "titan").glob("*.toml"))
    assert configs, "no configs under configs/titan/"
    for path in configs:
        config = ConfigManager().parse_args(["--job.config-file", str(path)])
        # The custom_config_module merge has to have happened, or every --darl.*
        # override in the launcher would be rejected as an unknown flag.
        assert hasattr(config, "darl"), f"{path.name}: no [darl] section"
        assert hasattr(config, "flower"), f"{path.name}: no [flower] section"
        assert hasattr(config, "titan"), f"{path.name}: no [titan] section"
        assert config.model.name.startswith("pww_"), config.model.name
        # torchft is deliberately never used; a config that turned it on would
        # need a dependency this repo does not have.
        assert not config.fault_tolerance.enable, (
            f"{path.name} enables torchft, but Flower does the outer step here"
        )


@check("pww_qwen3 train spec registers and takes vocab_size from the tokenizer")
def _():
    import pww.titan  # noqa: F401  -- registration side effect
    from torchtitan.protocols.train_spec import get_train_spec

    spec = get_train_spec("pww_qwen3")
    assert "0.6B" in spec.model_args, sorted(spec.model_args)
    args = spec.model_args["0.6B"]
    # Stock Qwen3 default before any override.
    assert args.vocab_size == 151936, args.vocab_size

    from dataclasses import dataclass, field

    with tempfile.TemporaryDirectory() as tmp:
        assets = Path(tmp)
        # A real 7-token fast tokenizer, so get_vocab_size is genuine.
        from tokenizers import Tokenizer, models

        vocab = {tok: i for i, tok in enumerate(["a", "b", "c", "d", "e", "f", "g"])}
        Tokenizer(models.WordLevel(vocab=vocab, unk_token="a")).save(
            str(assets / "tokenizer.json")
        )

        @dataclass
        class _Model:
            hf_assets_path: str = str(assets)

        @dataclass
        class _Training:
            seq_len: int = 128

        @dataclass
        class _Debug:
            moe_force_load_balance: bool = False

        @dataclass
        class _Titan:
            pad_vocab_to_multiple_of: int = 4

        @dataclass
        class _Job:
            model: _Model = field(default_factory=_Model)
            training: _Training = field(default_factory=_Training)
            debug: _Debug = field(default_factory=_Debug)
            titan: _Titan = field(default_factory=_Titan)

        import copy

        fresh = copy.deepcopy(args)
        fresh.update_from_config(_Job())
        # 7 ids padded to the next multiple of 4.
        assert fresh.vocab_size == 8, fresh.vocab_size
        assert fresh.max_seq_len == 128, fresh.max_seq_len


# --- the shipped configs -----------------------------------------------------

GRPC_CAP = 2**31 - 1
WIRE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}


def _shipped_configs() -> list[tuple[str, dict]]:
    import tomllib

    directory = ROOT / "configs" / "titan"
    return [(path.name, tomllib.loads(path.read_text()))
            for path in sorted(directory.glob("*.toml"))]


@check("every shipped config points at the tokenizer the download script creates")
def _():
    """Caught three configs naming a directory nothing produces.

    scripts/titan/download_tokenizer.sh writes to
    $PWW_DATA_DIR/tokenizers/$(basename $REPO_ID), and the repo id is
    openeurollm/tokenizer-128k -- so the only correct leaf name is 'tokenizer-128k'.
    Three configs said 'oellm-128k', which no script creates: the run would have
    failed at model build, after the queue wait.
    """
    script = (ROOT / "scripts" / "titan" / "download_tokenizer.sh").read_text()
    assert 'REPO_ID="openeurollm/tokenizer-128k"' in script, (
        "the tokenizer repo id moved; this check needs updating with it"
    )
    expected = "tokenizer-128k"

    for name, config in _shipped_configs():
        assets = config["model"]["hf_assets_path"]
        assert assets.rstrip("/").split("/")[-1] == expected, (
            f"{name}: model.hf_assets_path is {assets!r}, but download_tokenizer.sh "
            f"creates a directory ending in {expected!r}"
        )


@check("no shipped config asks for a transport its model cannot fit through")
def _():
    """The inline transport puts a full parameter set in one gRPC message, and the
    2 GiB cap is a protocol limit rather than a setting.

    Caught configs/titan/qwen3_1.7b_scaling.toml declaring transport = "inline" for a
    flavor that is 1,947,329,536 parameters with this vocabulary -- 3.6 GiB in
    float16. The client refuses at startup, so it was never going to silently corrupt
    a round, but it would have failed after the queue wait for a reason a config check
    can catch in milliseconds.

    Built on the meta device: shapes only, no weights, no GPU.
    """
    import torch
    from torchtitan.models.qwen3 import Qwen3Model, qwen3_args

    for name, config in _shipped_configs():
        flower = config.get("flower", {})
        if flower.get("transport", "inline") != "inline":
            continue
        flavor = config["model"]["flavor"]
        assert flavor in qwen3_args, f"{name}: unknown flavor {flavor!r}"

        # The vocabulary the model is actually built with: the tokenizer's id count
        # padded up, which is what makes these flavors bigger than their names.
        pad = config.get("titan", {}).get("pad_vocab_to_multiple_of", 256)
        vocab = -(-131073 // pad) * pad if pad > 1 else 131073

        args = qwen3_args[flavor]
        args.vocab_size = vocab
        with torch.device("meta"):
            model = Qwen3Model(args)
        numel = sum(p.numel() for p in model.parameters())

        dtype = flower.get("wire_dtype", "float16")
        assert dtype in WIRE_BYTES, f"{name}: unknown wire_dtype {dtype!r}"
        wire = numel * WIRE_BYTES[dtype]
        assert wire <= GRPC_CAP, (
            f"{name} declares transport='inline' with wire_dtype={dtype!r}, but "
            f"flavor {flavor} is {numel:,} parameters = {wire / 2**30:.1f} GiB, over "
            f"the {GRPC_CAP:,}-byte gRPC cap. Use transport='blob'."
        )


def main() -> int:
    print(__doc__.strip().splitlines()[0])
    print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print()
        for name, exc in FAILED:
            print(f"FAILED  {name}")
            print(f"        {type(exc).__name__}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
