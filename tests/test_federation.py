"""CPU-only tests for elastic membership and out-of-band weight transport.

    python3 tests/test_federation.py

No GPUs, no torchtitan, no allocation. Three layers:

  blobstore   a real HTTP blob store on an ephemeral port -- streamed upload and
              download, and the two ways a public-facing store gets broken into
              (a traversal in a client-supplied name, a missing token)
  membership  every count of live replicas, because HPC queues make all of them
              normal: zero (all sites queued), one (training alone), several, a
              site joining hundreds of rounds late, a site killed at walltime and
              requeued with a stale delta, and the aggregator itself restarting
  strategy    the Flower round protocol over both transports, with a fake client
              manager, so `configure_fit`/`aggregate_fit` are exercised without a
              server or a model

What needs GPUs and so is not here: the DTensor gather/scatter in `pww/delta.py`
against a really sharded model (covered on CPU gloo by the probe in
tests/test_titan.py's sibling checks), and any actual training.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASSED, FAILED = [], []
CHECK_TIMEOUT_S = 60


def check(name: str):
    def decorator(fn):
        import signal

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


from pww import fedproto as proto  # noqa: E402
from pww.central.blobstore import BlobStore, BlobStoreError, make_server, safe_name  # noqa: E402
from pww.central.globalstate import (  # noqa: E402
    Contribution,
    GlobalState,
    StaleContribution,
)
from pww.delta import BlobClient  # noqa: E402
from pww.tensorio import TensorFile, write_state_dict  # noqa: E402


def scratch() -> tempfile.TemporaryDirectory:
    """A temp directory on a volume with room for a multi-megabyte upload.

    `/tmp` on the central VM is on a nearly-full root filesystem, and the blob store
    deliberately refuses uploads that would fill its volume. Honour PWW_TEST_TMPDIR
    (or PWW_SCRATCH) so these checks exercise the streaming paths rather than the
    disk-full guard.
    """
    import os

    base = os.environ.get("PWW_TEST_TMPDIR") or os.environ.get("PWW_SCRATCH")
    if base and Path(base).is_dir():
        return tempfile.TemporaryDirectory(dir=base)
    return tempfile.TemporaryDirectory()


class _BlobServer:
    """A real blob store on an ephemeral port, for the duration of a `with`."""

    def __init__(self, root: Path, token: str = "", min_free: int = 1 << 20):
        self.store = BlobStore(root)
        # 1 MiB reserve instead of the production 1 GiB: the point here is the
        # transfer path, not the safety margin, which has its own check.
        self.httpd = make_server(
            self.store, host="127.0.0.1", port=0, token=token, min_free=min_free
        )
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_BlobServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


# --- blob store -------------------------------------------------------------


@check("blob store round-trips a file over real HTTP")
def _():
    with scratch() as tmp:
        root = Path(tmp)
        payload = root / "out.bin"
        # Bigger than the 8 MiB streaming chunk, so the chunked paths on both sides
        # are actually exercised rather than fitting in one read.
        payload.write_bytes(bytes(range(256)) * (48 << 10))
        size = payload.stat().st_size
        assert size > (8 << 20), size   # exercises the chunked path on both sides

        with _BlobServer(root / "store") as server:
            client = BlobClient(server.url, retries=2, timeout=60)
            client.put("thing.pww", payload)
            assert client.exists("thing.pww")
            back = root / "back.bin"
            received = client.get("thing.pww", back)
            assert received == size, (received, size)
            assert back.read_bytes() == payload.read_bytes()
            assert not client.exists("absent.pww")


@check("blob store refuses a traversal in a client-supplied name")
def _():
    # The store runs on a public VM and the name comes off the network, so this is
    # the difference between a blob write and an arbitrary file write.
    for bad in ("../etc/passwd", "a/b", ".hidden", "", "..", "x" * 300):
        expect_raises(BlobStoreError, lambda name=bad: safe_name(name))
    for good in ("run-global-r3.pww", "a", "A_b-c.1"):
        assert safe_name(good) == good


@check("blob store enforces its token")
def _():
    with scratch() as tmp:
        root = Path(tmp)
        payload = root / "p.bin"
        payload.write_bytes(b"x" * 1024)
        with _BlobServer(root / "store", token="sekrit") as server:
            good = BlobClient(server.url, token="sekrit", retries=1, timeout=20)
            good.put("ok.pww", payload)

            bad = BlobClient(server.url, token="wrong", retries=1, timeout=20)
            # A 401 is not retried -- it will never succeed, and retrying it would
            # just delay the error by the backoff schedule.
            from pww.delta import BlobTransportError

            expect_raises(
                BlobTransportError, lambda: bad.put("nope.pww", payload), contains="401"
            )

        # /health is deliberately unauthenticated so a monitor can poll it.
        with _BlobServer(root / "store2", token="sekrit") as server:
            with urllib.request.urlopen(f"{server.url}/health", timeout=20) as response:
                assert json.loads(response.read())["ok"] is True


@check("blob store refuses an upload that would fill its volume")
def _():
    from pww.delta import BlobTransportError

    with scratch() as tmp:
        root = Path(tmp)
        payload = root / "p.bin"
        payload.write_bytes(b"x" * 4096)
        # A reserve larger than the whole volume, so every upload is refused. The
        # client pre-flights /usage, so it fails with the reason rather than with the
        # broken pipe a mid-body rejection would produce.
        with _BlobServer(root / "store", min_free=1 << 62) as server:
            client = BlobClient(server.url, retries=1, timeout=20)
            expect_raises(BlobTransportError, lambda: client.put("big.pww", payload))


@check("blob store prunes everything except what it is told to keep")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        store = BlobStore(Path(tmp) / "store")
        for name in ("a.pww", "b.pww", "c.pww"):
            (store.root / name).write_bytes(b"0" * 16)
        removed = store.prune(keep={"b.pww"})
        assert removed == 2, removed
        assert store.exists("b.pww") and not store.exists("a.pww")
        # Without this a 7B run leaves ~14 GiB per site per round behind.
        assert store.usage()["blobs"] == 1


# --- durable global state and membership ------------------------------------


def _model(scale: float = 1.0) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "emb.weight": torch.randn(6, 4) * scale,
        "block.0.w": torch.randn(4, 4) * scale,
        "norm.scale": torch.ones(4) * scale,
    }


def _delta(path: Path, value: float, base_round: int) -> Path:
    write_state_dict(
        path,
        {key: torch.full_like(tensor, value) for key, tensor in _model().items()},
        meta={"base_round": base_round},
    )
    return path


@check("a single live replica merges on its own")
def _():
    """DiLoCo with k=1 is correct, not degraded -- one site out of the queue trains."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_state_dict(root / "init.pww", _model())
        state = GlobalState(root / "state")
        state.initialise_from_file(root / "init.pww")
        before = state.open_global().get("norm.scale", torch.float32).clone()

        state.merge(
            [Contribution("lumi", _delta(root / "d.pww", 0.1, 0), weight=1.0,
                          tokens=1000, base_round=0)],
            server_learning_rate=1.0, server_momentum=0.0,
        )
        after = state.open_global().get("norm.scale", torch.float32)
        # momentum 0, lr 1, single contributor: w_next == w_t + delta exactly.
        assert torch.allclose(after, before + 0.1, atol=1e-6), (before[0], after[0])
        assert state.round == 1
        assert state.clusters["lumi"].rounds_contributed == 1


@check("state survives a restart of the aggregator process")
def _():
    """What makes zero live replicas a wait rather than a lost run."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_state_dict(root / "init.pww", _model())
        first = GlobalState(root / "state")
        first.initialise_from_file(root / "init.pww")
        first.merge(
            [Contribution("lumi", _delta(root / "d.pww", 0.25, 0), weight=1.0,
                          tokens=4242, base_round=0)],
            server_momentum=0.9,
        )
        expected = first.open_global().get("emb.weight", torch.float32).clone()

        # A brand-new object over the same directory is exactly what a restarted
        # server does.
        second = GlobalState(root / "state")
        assert second.round == 1, second.round
        assert second.keys == first.keys
        assert second.total_tokens == 4242
        assert second.clusters["lumi"].rounds_contributed == 1
        assert torch.equal(second.open_global().get("emb.weight", torch.float32), expected)
        # And it can keep training from there, momentum buffer included.
        second.merge(
            [Contribution("lumi", _delta(root / "d2.pww", 0.25, 1), weight=1.0,
                          tokens=1, base_round=1)],
            server_momentum=0.9,
        )
        assert second.round == 2


@check("a stale delta from a requeued site is rejected, not averaged in")
def _():
    """The failure this design exists for: killed at walltime, requeued hours later."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_state_dict(root / "init.pww", _model())
        state = GlobalState(root / "state")
        state.initialise_from_file(root / "init.pww")
        state.merge(
            [Contribution("lumi", _delta(root / "fresh.pww", 0.1, 0), weight=1.0,
                          tokens=100, base_round=0)],
            server_momentum=0.0,
        )
        assert state.round == 1
        settled = state.open_global().get("norm.scale", torch.float32).clone()

        # base_round 0 against a global now at round 1.
        expect_raises(
            StaleContribution,
            lambda: state.merge(
                [Contribution("snellius", _delta(root / "stale.pww", 99.0, 0),
                              weight=1.0, tokens=100, base_round=0)],
                server_momentum=0.0,
            ),
            contains="stale",
        )
        assert state.round == 1, "a rejected round must not advance the counter"
        assert torch.equal(state.open_global().get("norm.scale", torch.float32), settled)
        assert state.clusters["snellius"].stale_rejected == 1

        # A mixed round still merges the fresh contribution and drops the stale one.
        state.merge(
            [
                Contribution("lumi", _delta(root / "f2.pww", 0.1, 1), weight=1.0,
                             tokens=100, base_round=1),
                Contribution("snellius", _delta(root / "s2.pww", 99.0, 0), weight=1.0,
                             tokens=100, base_round=0),
            ],
            server_momentum=0.0,
        )
        assert state.round == 2
        # Only lumi's 0.1 landed; snellius' 99.0 would be unmistakable.
        assert torch.allclose(
            state.open_global().get("norm.scale", torch.float32), settled + 0.1, atol=1e-6
        )


@check("a late joiner is recorded from the round it actually arrived")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_state_dict(root / "init.pww", _model())
        state = GlobalState(root / "state")
        state.initialise_from_file(root / "init.pww")
        for round_index in range(3):
            state.merge(
                [Contribution("lumi", _delta(root / f"l{round_index}.pww", 0.01, round_index),
                              weight=1.0, tokens=10, base_round=round_index)],
                server_momentum=0.0,
            )
        assert state.round == 3

        state.merge(
            [
                Contribution("lumi", _delta(root / "l3.pww", 0.01, 3), weight=0.5,
                             tokens=10, base_round=3),
                Contribution("snellius", _delta(root / "s3.pww", 0.03, 3), weight=0.5,
                             tokens=30, base_round=3),
            ],
            server_momentum=0.0,
        )
        # Recorded against the round each cluster actually trained against, so lumi
        # shows 0 (it saw the freshly seeded model) rather than 1.
        assert state.clusters["lumi"].first_seen_round == 0, state.clusters
        # Joined at 3, and contributed once -- not credited for the rounds it missed.
        assert state.clusters["snellius"].first_seen_round == 3, state.clusters
        assert state.clusters["snellius"].rounds_contributed == 1
        assert state.clusters["lumi"].rounds_contributed == 4


@check("a delta from a mismatched model is refused rather than averaged")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_state_dict(root / "init.pww", _model())
        state = GlobalState(root / "state")
        state.initialise_from_file(root / "init.pww")

        # A different vocab size -- what a tokenizer mismatch between sites looks like.
        wrong = {key: tensor.clone() for key, tensor in _model().items()}
        wrong["emb.weight"] = torch.zeros(999, 4)
        write_state_dict(root / "wrong.pww", wrong)
        expect_raises(
            ValueError,
            lambda: state.merge(
                [Contribution("snellius", root / "wrong.pww", weight=1.0, tokens=1,
                              base_round=0)],
                server_momentum=0.0,
            ),
            contains="mismatched model.flavor or vocab size",
        )
        assert state.round == 0


@check("disk budget arithmetic is reported and scales with the model")
def _():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_state_dict(root / "init.pww", _model())
        state = GlobalState(root / "state")
        state.initialise_from_file(root / "init.pww")
        budget = state.disk_budget(sites=2)
        numel = sum(t.numel() for t in _model().values())
        # global + momentum at float32
        assert budget["resident"] == numel * 4 * 2, budget
        # one bfloat16 delta per site
        assert budget["transient"] == numel * 2 * 2, budget
        assert budget["free"] > 0


@check("pooled perplexity comes from the pooled loss, not from averaging perplexities")
def _():
    """The metric bug that moves with cluster skew rather than with the model.

    ppl = exp(mean NLL per token), so the perplexity of the union of the clusters'
    validation tokens is exp() of their token-weighted mean loss -- exactly, since each
    cluster reports its own mean NLL per token. Averaging per-cluster perplexities
    computes mean(exp(L)) instead, and exp is convex, so by Jensen that is always at
    least exp(mean(L)): pessimistic, and wrong by more the further apart the clusters
    are. A metric whose error tracks how unevenly the sites are running, rather than the
    model, is worse than no metric.
    """
    from pww.central.server import build_metric_aggregators

    _fit, aggregate_eval = build_metric_aggregators()

    # Equal token counts, losses 2.0 and 4.0.
    out = aggregate_eval([
        (1000, {"eval_loss": 2.0, "perplexity": math.exp(2.0)}),
        (1000, {"eval_loss": 4.0, "perplexity": math.exp(4.0)}),
    ])
    assert abs(out["eval_loss"] - 3.0) < 1e-9, out
    assert abs(out["perplexity"] - math.exp(3.0)) < 1e-6, out
    # The wrong answer, for the record: mean(exp(2), exp(4)) = 31.0 against exp(3) = 20.1.
    naive = (math.exp(2.0) + math.exp(4.0)) / 2
    assert naive > out["perplexity"] * 1.5, (naive, out["perplexity"])

    # Token weighting: the site that evaluated ten times as much should dominate.
    out = aggregate_eval([
        (100, {"eval_loss": 5.0}),
        (1000, {"eval_loss": 2.0}),
    ])
    expected = (100 * 5.0 + 1000 * 2.0) / 1100
    assert abs(out["eval_loss"] - expected) < 1e-9, out
    assert abs(out["perplexity"] - math.exp(expected)) < 1e-6, out

    # A single cluster is the degenerate case and must be exact.
    out = aggregate_eval([(500, {"eval_loss": 1.25})])
    assert abs(out["perplexity"] - math.exp(1.25)) < 1e-9, out

    # Nothing evaluated: no metric rather than a divide by zero.
    assert aggregate_eval([(0, {"eval_loss": 3.0})]) == {}


@check("accuracy stays a sample-weighted mean, which for a linear metric is correct")
def _():
    """The CIFAR path shares this aggregator, and accuracy must NOT get the same
    treatment: it is a mean of per-sample 0/1 outcomes, so a sample-weighted mean of
    per-cluster accuracies *is* the pooled accuracy. Linear, unlike perplexity.
    """
    from pww.central.server import build_metric_aggregators

    _fit, aggregate_eval = build_metric_aggregators()
    out = aggregate_eval([(100, {"accuracy": 90.0}), (300, {"accuracy": 70.0})])
    assert abs(out["accuracy"] - 75.0) < 1e-9, out
    assert "perplexity" not in out, "accuracy must not be exponentiated"


@check("training loss is token-weighted across clusters")
def _():
    """A cluster that trained ten times the tokens should move the reported loss ten
    times as much. Unweighted would let a site that managed two steps before walltime
    drag the number as far as one that ran a full phase.
    """
    from pww.central.server import build_metric_aggregators

    aggregate_fit, _eval = build_metric_aggregators()
    out = aggregate_fit([
        (1_000_000, {proto.LOSS: 2.0, proto.CLUSTER: "lumi"}),
        (100_000, {proto.LOSS: 7.0, proto.CLUSTER: "snellius"}),
    ])
    expected = (1_000_000 * 2.0 + 100_000 * 7.0) / 1_100_000
    assert abs(out["loss"] - expected) < 1e-9, out
    # And a round where nothing was trained reports nothing, rather than a loss of 0.0 --
    # which is what let 23 consecutive no-op rounds look like progress.
    assert aggregate_fit([(0, {proto.LOSS: 0.0})]) == {}

    # Drift is reported as both mean and max, because it is the worst replica that
    # decides whether H is too large, and two sites at 0.01 and 0.30 average to a
    # reassuring 0.155.
    out = aggregate_fit([
        (1000, {proto.LOSS: 2.0, "drift_ratio": 0.01}),
        (1000, {proto.LOSS: 2.0, "drift_ratio": 0.30}),
    ])
    assert abs(out["drift_ratio"] - 0.155) < 1e-9, out
    assert abs(out["drift_ratio_max"] - 0.30) < 1e-9, out


@check("the outer step is Nesterov momentum on DiLoCo's outer gradient")
def _():
    """Pins FedMom == torch.optim.SGD(momentum=beta, nesterov=True), which is the
    OuterOpt DiLoCo Algorithm 1 prescribes (arXiv 2311.08105 s4).

    Not a restatement of the implementation: both sides are driven by the *same*
    prescribed sequence of outer gradients, so any divergence is the update rule
    itself. Substituting m_t = v_t - v_(t-1) into

        v_next = w - eta*(w - w_avg)          # w - eta*Delta
        w_next = v_next + beta*(v_next - v_prev)

    gives m_next = beta*m - eta*Delta with Delta evaluated at w = v + beta*m, i.e.
    Nesterov's accelerated gradient in two-sequence form.

    The heavy-ball comparison is the part that makes this a real test: without it,
    an implementation where momentum did nothing at all would also pass.
    """
    eta, beta, rounds = 0.7, 0.9, 4
    shares = [0.5, 0.3, 0.2]          # token-weighted, deliberately not 1/k
    theta0 = _model()
    torch.manual_seed(11)
    grads = [
        {key: 0.1 * torch.randn(value.shape) for key, value in theta0.items()}
        for _ in range(rounds)
    ]

    reference = {k: v.clone().requires_grad_(True) for k, v in theta0.items()}
    control = {k: v.clone().requires_grad_(True) for k, v in theta0.items()}
    nesterov = torch.optim.SGD(list(reference.values()), lr=eta, momentum=beta,
                               nesterov=True)
    heavy_ball = torch.optim.SGD(list(control.values()), lr=eta, momentum=beta,
                                 nesterov=False)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_state_dict(root / "init.pww", theta0)
        state = GlobalState(root / "state", storage_dtype=torch.float32)
        state.initialise_from_file(root / "init.pww")

        worst_nesterov = worst_heavy = 0.0
        for rnd, grad in enumerate(grads, start=1):
            # Per-cluster deltas whose token-weighted mean is exactly -grad. The last
            # is solved for, so the weighting is doing real work rather than every
            # cluster carrying an identical tensor.
            deltas = [
                {key: torch.randn(value.shape) for key, value in theta0.items()}
                for _ in shares[:-1]
            ]
            deltas.append({
                key: (-grad[key] - sum(p * d[key] for p, d in zip(shares, deltas)))
                     / shares[-1]
                for key in theta0
            })
            contributions = []
            for cluster, delta in enumerate(deltas):
                path = root / f"d{rnd}-{cluster}.pww"
                write_state_dict(path, delta)
                contributions.append(Contribution(
                    cluster=f"c{cluster}", path=path, weight=shares[cluster],
                    tokens=1000, base_round=rnd - 1,
                ))
            state.merge(contributions, server_learning_rate=eta, server_momentum=beta)

            for tensors, optimiser in ((reference, nesterov), (control, heavy_ball)):
                optimiser.zero_grad()
                for key in tensors:
                    tensors[key].grad = grad[key].clone()
                optimiser.step()

            with state.open_global() as handle:
                ours = {key: handle.get(key, torch.float32) for key in handle.keys}
            worst_nesterov = max(worst_nesterov, max(
                (ours[k] - reference[k].detach()).abs().max().item() for k in ours))
            worst_heavy = max(worst_heavy, max(
                (ours[k] - control[k].detach()).abs().max().item() for k in ours))

        assert worst_nesterov < 1e-5, (
            f"the outer step has drifted away from SGD(nesterov=True): max abs "
            f"difference {worst_nesterov:.3e} over {rounds} rounds"
        )
        assert worst_heavy > 1e-2, (
            f"heavy ball is indistinguishable here ({worst_heavy:.3e}), so this check "
            f"proves nothing -- momentum is not affecting the iterates"
        )


@check("momentum 0.0 with outer lr 1.0 is exactly FedAvg")
def _():
    """The baseline the WikiText run was unknowingly training with.

    v = w - 1.0*(w - w_avg) = w_avg, then w_next = v + 0.0*(...) = w_avg. Worth a
    test because it is the control arm of every DiLoCo-vs-FedAvg comparison, and
    because an off-by-one in the momentum branch would silently break it.
    """
    theta0 = _model()
    shares = [0.25, 0.75]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_state_dict(root / "init.pww", theta0)
        state = GlobalState(root / "state", storage_dtype=torch.float32)
        state.initialise_from_file(root / "init.pww")

        torch.manual_seed(3)
        locals_ = [{k: v + torch.randn(v.shape) for k, v in theta0.items()}
                   for _ in shares]
        contributions = []
        for cluster, local in enumerate(locals_):
            path = root / f"d{cluster}.pww"
            write_state_dict(path, {k: local[k] - theta0[k] for k in theta0})
            contributions.append(Contribution(
                cluster=f"c{cluster}", path=path, weight=shares[cluster],
                tokens=1000, base_round=0,
            ))
        state.merge(contributions, server_learning_rate=1.0, server_momentum=0.0)

        with state.open_global() as handle:
            for key in theta0:
                expected = sum(p * local[key] for p, local in zip(shares, locals_))
                got = handle.get(key, torch.float32)
                error = (got - expected).abs().max().item()
                assert error < 1e-5, f"{key}: FedAvg mismatch {error:.3e}"


# --- the Flower round protocol ---------------------------------------------

HAS_FLWR = True
try:
    from flwr.common import Code, FitRes, Parameters, Status, ndarrays_to_parameters
    from flwr.server.client_proxy import ClientProxy
except ImportError:
    HAS_FLWR = False


class _FakeClient:
    def __init__(self, cid: str):
        self.cid = cid


class _FakeManager:
    """Enough ClientManager for configure_fit/configure_evaluate."""

    def __init__(self, count: int):
        self.clients = [_FakeClient(f"cid-{i}") for i in range(count)]

    def num_available(self) -> int:
        return len(self.clients)

    def sample(self, num_clients: int, min_num_clients: int | None = None):
        if min_num_clients is not None and len(self.clients) < min_num_clients:
            raise AssertionError(
                f"sample() would block: {len(self.clients)} available, "
                f"{min_num_clients} required"
            )
        return self.clients[:num_clients]


def _fit_res(metrics: dict, num_examples: int, arrays=None) -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=(
            ndarrays_to_parameters(arrays)
            if arrays is not None
            else Parameters(tensors=[], tensor_type=proto.BLOB_PARAMETERS_TYPE)
        ),
        num_examples=num_examples,
        metrics=metrics,
    )


if HAS_FLWR:

    @check("zero live replicas: the server starts without asking any client for a model")
    def _():
        """With min_clients=1 and blob transport, `initialize_parameters` must answer.

        Returning None is what makes Flower block sampling a client to ask for the
        architecture -- so every site being queued became a deadlock instead of a wait.
        """
        from pww.central.strategy import FedMom

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = GlobalState(root / "state")
            strategy = FedMom(
                transport=proto.TRANSPORT_BLOB, state=state,
                blob_root=root / "blobs", blob_url="http://vm:29512", run_id="t",
            )
            (root / "blobs").mkdir()
            empty = _FakeManager(0)
            params = strategy.initialize_parameters(empty)
            assert params is not None, "would block waiting for a client"
            assert params.tensor_type == proto.BLOB_PARAMETERS_TYPE
            assert strategy.merge_round == 0

    @check("cold start asks exactly one cluster to seed the global model")
    def _():
        from pww.central.strategy import FedMom

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "blobs").mkdir()
            state = GlobalState(root / "state")
            strategy = FedMom(
                transport=proto.TRANSPORT_BLOB, state=state,
                blob_root=root / "blobs", blob_url="http://vm:29512", run_id="t",
            )
            instructions = strategy.configure_fit(
                1, Parameters(tensors=[], tensor_type=proto.BLOB_PARAMETERS_TYPE),
                _FakeManager(2),
            )
            # Two available, but only one is asked: several concurrent uploads would
            # leave the global model depending on which PUT finished last.
            assert len(instructions) == 1, len(instructions)
            config = instructions[0][1].config
            assert config[proto.NEED_INIT] == "1"
            assert config[proto.INIT_BLOB] == proto.init_blob("t")
            assert config[proto.TRANSPORT] == proto.TRANSPORT_BLOB
            assert proto.GLOBAL_BLOB not in config

    @check("two clients reporting the same cluster id in one round are dropped, not merged")
    def _():
        """Closes a path the DARL coordinator cannot see.

        Registration now refuses a second live process on one cluster id, but a run can
        reach the aggregator without ever registering: `pww_qwen3_local` uses
        torchtitan's own dataloader, so a config with `flower.enable = true` and no DARL
        has no coordinator to refuse it.

        Left unchecked the result is not a lost round, it is a *wrong* one. Delta blobs
        are named (run, round, cluster), so both clients wrote the same object and one
        overwrote the other; two Contributions then point at one file and the merge sums
        `share_i * delta` over both, counting the survivor twice with the combined weight
        of both jobs. Nothing about the output looks wrong, which is why this has to be
        refused rather than warned about.
        """
        from pww.central.strategy import FedMom

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blobs = root / "blobs"
            blobs.mkdir()
            state = GlobalState(root / "state")
            write_state_dict(root / "init.pww", _model())
            state.initialise_from_file(root / "init.pww")
            strategy = FedMom(
                transport=proto.TRANSPORT_BLOB, state=state, blob_root=blobs,
                blob_url="http://vm:29512", run_id="t", keep_rounds=0,
            )

            # Two jobs at one site, both calling themselves "lumi", so both wrote this
            # one blob and only the last writer's content is in it.
            shared = proto.delta_blob("t", 0, "lumi")
            _delta(blobs / shared, 0.2, 0)
            # A third, correctly-named site in the same round.
            _delta(blobs / proto.delta_blob("t", 0, "snellius"), 0.3, 0)

            before = state.round
            params, metrics = strategy.aggregate_fit(
                1,
                [
                    (None, _fit_res({proto.CLUSTER: "lumi",
                                     proto.DELTA_BLOB: shared,
                                     proto.BASE_ROUND: 0}, 400)),
                    (None, _fit_res({proto.CLUSTER: "lumi",
                                     proto.DELTA_BLOB: shared,
                                     proto.BASE_ROUND: 0}, 500)),
                    (None, _fit_res({proto.CLUSTER: "snellius",
                                     proto.DELTA_BLOB: proto.delta_blob("t", 0, "snellius"),
                                     proto.BASE_ROUND: 0}, 600)),
                ],
                [],
            )

            # The round still happens -- snellius did nothing wrong and its tokens
            # should not be thrown away because another site was misconfigured.
            assert state.round == before + 1, state.round
            # But only snellius is in it. Both "lumi" entries are gone: with one file
            # and two token counts there is no way to know whose weights survived, so
            # there is no correct weight to give it.
            assert set(state.clusters) == {"snellius"}, sorted(state.clusters)
            assert state.clusters["snellius"].tokens_total == 600

    @check("a duplicated cluster id does not stop the other sites' round")
    def _():
        """And if the duplicated pair is *all* there is, the model must not move."""
        from pww.central.strategy import FedMom

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blobs = root / "blobs"
            blobs.mkdir()
            state = GlobalState(root / "state")
            write_state_dict(root / "init.pww", _model())
            state.initialise_from_file(root / "init.pww")
            strategy = FedMom(
                transport=proto.TRANSPORT_BLOB, state=state, blob_root=blobs,
                blob_url="http://vm:29512", run_id="t", keep_rounds=0,
            )
            shared = proto.delta_blob("t", 0, "lumi")
            _delta(blobs / shared, 0.2, 0)

            with state.open_global() as handle:
                original = {k: handle.get(k, torch.float32).clone() for k in handle.keys}

            strategy.aggregate_fit(
                1,
                [(None, _fit_res({proto.CLUSTER: "lumi", proto.DELTA_BLOB: shared,
                                  proto.BASE_ROUND: 0}, n)) for n in (400, 500)],
                [],
            )
            assert state.round == 0, "the model moved on a round with nothing valid in it"
            with state.open_global() as handle:
                for key, value in original.items():
                    assert torch.equal(handle.get(key, torch.float32), value), key

    @check("blob round: seed, merge, publish, then hand the new global to both sites")
    def _():
        from pww.central.strategy import FedMom

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blobs = root / "blobs"
            blobs.mkdir()
            state = GlobalState(root / "state")
            strategy = FedMom(
                transport=proto.TRANSPORT_BLOB, state=state, blob_root=blobs,
                blob_url="http://vm:29512", run_id="t", keep_rounds=0,
            )

            # Round 1: one cluster seeds and contributes.
            write_state_dict(blobs / proto.init_blob("t"), _model())
            _delta(blobs / proto.delta_blob("t", 0, "lumi"), 0.2, 0)
            params, metrics = strategy.aggregate_fit(
                1,
                [(None, _fit_res({
                    proto.CLUSTER: "lumi",
                    proto.DELTA_BLOB: proto.delta_blob("t", 0, "lumi"),
                    proto.BASE_ROUND: 0,
                    proto.UPLOADED_INIT: "1",
                }, 1000))],
                [],
            )
            assert state.round == 1, state.round
            assert metrics["merge_round"] == 1
            # The merged global is published under this round's name, so the next
            # configure_fit can point clusters at it.
            published = blobs / proto.global_blob("t", 1)
            assert published.is_file(), sorted(p.name for p in blobs.iterdir())

            # Round 2: both sites, and the newcomer is handed the current global.
            instructions = strategy.configure_fit(2, params, _FakeManager(2))
            assert len(instructions) == 2
            for _, ins in instructions:
                assert ins.config[proto.GLOBAL_BLOB] == proto.global_blob("t", 1)
                assert ins.config[proto.ROUND] == 1
                assert proto.NEED_INIT not in ins.config

            for cluster, value in (("lumi", 0.1), ("snellius", 0.3)):
                _delta(blobs / proto.delta_blob("t", 1, cluster), value, 1)
            strategy.aggregate_fit(
                2,
                [
                    (None, _fit_res({proto.CLUSTER: c,
                                     proto.DELTA_BLOB: proto.delta_blob("t", 1, c),
                                     proto.BASE_ROUND: 1}, n))
                    for c, n in (("lumi", 400), ("snellius", 600))
                ],
                [],
            )
            assert state.round == 2
            assert set(state.clusters) == {"lumi", "snellius"}
            # keep_rounds=0 means the round-1 global is gone once round 2 exists.
            assert (blobs / proto.global_blob("t", 2)).is_file()
            assert not (blobs / proto.global_blob("t", 1)).is_file()

    @check("a round in which every site trained nothing leaves the model untouched")
    def _():
        from pww.central.strategy import FedMom

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blobs = root / "blobs"
            blobs.mkdir()
            state = GlobalState(root / "state")
            state.initialise_from_file(write_state_dict(root / "init.pww", _model()))
            strategy = FedMom(
                transport=proto.TRANSPORT_BLOB, state=state, blob_root=blobs,
                blob_url="http://vm:29512", run_id="t",
            )
            before = state.open_global().get("norm.scale", torch.float32).clone()
            strategy.aggregate_fit(
                5,
                [(None, _fit_res({proto.CLUSTER: "lumi", proto.EXHAUSTED: True}, 0))],
                [],
            )
            assert state.round == 0, "an empty round must not advance the merge counter"
            assert torch.equal(state.open_global().get("norm.scale", torch.float32), before)

    @check("no results at all keeps the current global model rather than returning None")
    def _():
        from pww.central.strategy import FedMom

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blobs = root / "blobs"
            blobs.mkdir()
            state = GlobalState(root / "state")
            strategy = FedMom(
                transport=proto.TRANSPORT_BLOB, state=state, blob_root=blobs,
                blob_url="http://vm:29512", run_id="t",
            )
            params, _ = strategy.aggregate_fit(3, [], [RuntimeError("walltime")])
            assert params is not None
            assert state.round == 0

    @check("inline transport still works and keeps its arithmetic in float32")
    def _():
        from pww.central.strategy import FedMom

        weights = np.full((8,), 0.5, dtype=np.float16)
        strategy = FedMom(
            transport=proto.TRANSPORT_INLINE,
            initial_parameters=ndarrays_to_parameters([weights.copy()]),
            server_learning_rate=1.0, server_momentum=0.9,
        )
        strategy.configure_fit(
            1, ndarrays_to_parameters([weights.copy()]), _FakeManager(2)
        )
        aggregated, metrics = strategy.aggregate_fit(
            1,
            [
                (None, _fit_res({proto.CLUSTER: "lumi"}, 1_600_000,
                                arrays=[np.full((8,), 0.6, dtype=np.float16)])),
                (None, _fit_res({proto.CLUSTER: "snellius"}, 2_400_000,
                                arrays=[np.full((8,), 0.4, dtype=np.float16)])),
            ],
            [],
        )
        from flwr.common import parameters_to_ndarrays

        out = parameters_to_ndarrays(aggregated)[0].astype(np.float32)
        # Token counts times float16 weights overflow to inf through flwr's own
        # `aggregate`; normalised float32 shares do not.
        assert np.all(np.isfinite(out)), out
        assert np.allclose(out, 0.462, atol=2e-3), out
        assert strategy.v_vector[0].dtype == np.float32
        assert metrics["merge_round"] == 1

    @check("a failed round leaves at the wire dtype, so one crash cannot wedge the server")
    def _():
        # The regression this pins cost a whole run. A round that produced no results
        # returned the *authoritative float32* copy, which is twice the wire size; at
        # 0.6B that is 2.4 GiB against gRPC's hard 2 GiB per-message cap, so the next
        # configure_fit could not be sent, which failed the round, which took the same
        # exit again. Eighteen rounds of "0 results, 1 failure" from one crashed round.
        from flwr.common import parameters_to_ndarrays
        from pww.central.strategy import FedMom

        weights = np.full((512,), 0.5, dtype=np.float16)
        incoming = ndarrays_to_parameters([weights.copy()])
        wire_bytes = sum(len(t) for t in incoming.tensors)

        strategy = FedMom(
            transport=proto.TRANSPORT_INLINE,
            initial_parameters=ndarrays_to_parameters([weights.copy()]),
            server_learning_rate=0.7, server_momentum=0.9,
        )
        strategy.configure_fit(1, incoming, _FakeManager(1))

        # A crashed client: no results, one exception.
        out, _ = strategy.aggregate_fit(1, [], [RuntimeError("DTensor crash")])
        assert out is not None
        assert sum(len(t) for t in out.tensors) == wire_bytes, (
            f"a failed round must not grow the payload: {sum(len(t) for t in out.tensors)} "
            f"vs {wire_bytes} incoming"
        )
        assert parameters_to_ndarrays(out)[0].dtype == np.float16

        # initialize_parameters is the other exit that skipped the downcast, and it is
        # the one a restarted server takes.
        resumed = strategy.initialize_parameters(_FakeManager(1))
        assert sum(len(t) for t in resumed.tensors) == wire_bytes
        assert parameters_to_ndarrays(resumed)[0].dtype == np.float16

        # ...while the copy the outer step is computed on stays float32, because
        # round-tripping the momentum buffer through float16 quantises it every round.
        assert strategy._global_fp32[0].dtype == np.float32

    @check("blob transport refuses to start without the pieces it needs")
    def _():
        from pww.central.strategy import FedMom

        expect_raises(
            ValueError,
            lambda: FedMom(transport=proto.TRANSPORT_BLOB),
            contains="blob transport needs",
        )
        expect_raises(
            ValueError, lambda: FedMom(transport="carrier-pigeon"), contains="transport must be"
        )


# --- protocol ---------------------------------------------------------------


@check("blob names survive the store's own validation")
def _():
    for run in ("pww", "run/with/slashes", "run with spaces", ""):
        for cluster in ("lumi", "snellius-r0", "a b"):
            for name in (
                proto.global_blob(run, 12),
                proto.init_blob(run),
                proto.delta_blob(run, 3, cluster),
            ):
                # Generated names are fed straight to the store, so anything the
                # generator can emit has to be something the validator accepts.
                assert safe_name(name) == name, name


def main() -> int:
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if not HAS_FLWR:
        print("(flwr not installed -- strategy checks skipped)")
    if FAILED:
        print()
        for name, exc in FAILED:
            print(f"FAILED  {name}")
            print(f"        {type(exc).__name__}: {exc}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
