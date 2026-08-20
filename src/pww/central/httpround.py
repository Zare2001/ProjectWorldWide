"""HTTP round protocol: the same DiLoCo round, without gRPC on the wire.

Why this exists
---------------
A site whose compute nodes have no route to the internet reaches us only through an
HTTP forward proxy. OLCF documents this for Frontier -- "by default, the compute nodes
are closed off from the internet ... you can go through the proxy server" -- and the
proxy is provisioned for what that sentence describes: short outbound fetches, the
shape of "download a checkpoint or pre-trained model".

Two of our three components already fit that shape and have never failed from such a
site: the DARL coordinator (small request/response) and the blob store (discrete PUT
and GET, gigabytes at a time). The one that does not fit is Flower's transport, which
holds a single bidirectional HTTP/2 stream open for the entire job. Carried as a
CONNECT tunnel that stream is reaped mid-round, and a closed stream ends the run --
observed repeatedly on 2026-08-20, including once *after* a 1.07 GiB delta had already
uploaded successfully. The gigabytes get through; the small message that follows does
not.

So this module keeps flwr as a LIBRARY and drops it as a TRANSPORT. `PWWFedMom` is
untouched and still does every merge; `FitIns`/`FitRes`/`Parameters` are still flwr's
types. Only `Server.fit()`'s loop and its gRPC channel are replaced -- by a round
driver and an HTTP endpoint that clients poll, exactly the pattern DARL already proves
works from behind that proxy.

Blob transport only
-------------------
The weights are deliberately NOT carried here. Inline transport would put 1.32 GiB in
a control message, which is the thing this exists to avoid; `--transport blob` moves
them out of band through the blob store, and this plane carries only round config and
result metrics -- a few hundred bytes each way. Inline is refused at startup rather
than quietly working badly.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ..logging_utils import get_logger
from .. import fedproto as proto

logger = get_logger("pww.central.httpround")

# How long a client's poll is held open before returning "wait". Long enough that an
# idle site is not hammering the endpoint, short enough that no proxy along the way
# considers the request stalled -- the failure mode this whole module exists to avoid.
POLL_HOLD_S = 20.0


class HttpClientProxy:
    """Stands in for flwr's ClientProxy. The strategy only ever reads `.cid`."""

    def __init__(self, cluster_id: str):
        self.cid = cluster_id

    def __repr__(self) -> str:
        return f"HttpClientProxy({self.cid!r})"


class HttpClientManager:
    """Stands in for flwr's ClientManager: `sample` and `num_available`, nothing else.

    Membership is whoever has polled inside `liveness_s`. That is deliberately the same
    rule DARL uses for a cluster being live, so a site that vanishes drops out of both
    at roughly the same time instead of the two disagreeing about who is in the round.
    """

    def __init__(self, liveness_s: float = 120.0):
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self.liveness_s = liveness_s

    def touch(self, cluster_id: str) -> None:
        with self._lock:
            self._seen[cluster_id] = time.monotonic()

    def live(self) -> list[str]:
        now = time.monotonic()
        with self._lock:
            return sorted(c for c, t in self._seen.items() if now - t < self.liveness_s)

    def num_available(self) -> int:
        return len(self.live())

    def sample(self, num_clients: int, min_num_clients: int | None = None) -> list[HttpClientProxy]:
        chosen = self.live()[:num_clients]
        return [HttpClientProxy(c) for c in chosen]


class RoundState:
    """What each cluster is currently being asked to do, and what it has returned.

    One lock guards everything: the HTTP handler threads write results into it while
    the driver thread reads them, and the round boundary is exactly the moment both
    views have to agree.
    """

    def __init__(self) -> None:
        self.lock = threading.Condition()
        self.server_round = 0
        self.kind = ""                      # "fit" | "evaluate" | ""
        self.config: dict[str, Any] = {}
        self.expected: set[str] = set()
        self.results: dict[str, dict] = {}
        self.failures: dict[str, str] = {}
        # Set once the driver has run its last round. Without it a client polls a
        # finished server forever and the job burns its remaining walltime idle.
        self.finished = False
        # Who has actually collected that stop. The endpoint has to outlive the last
        # round long enough for every site to ask once more -- tearing it down the
        # instant rounds end gives each client a connection-refused, which it retries
        # and then raises, turning a clean finish into a failed job at every site.
        self.stopped: set[str] = set()

    def open(self, server_round: int, kind: str, config: dict, clusters: list[str]) -> None:
        with self.lock:
            self.server_round = server_round
            self.kind = kind
            self.config = config
            self.expected = set(clusters)
            self.results = {}
            self.failures = {}
            self.lock.notify_all()

    def close(self) -> None:
        with self.lock:
            self.kind = ""
            self.expected = set()
            self.lock.notify_all()

    def submit(self, cluster: str, server_round: int, payload: dict) -> bool:
        with self.lock:
            if server_round != self.server_round or cluster not in self.expected:
                # A late result from a round that has already closed. Dropping it is
                # correct: the merge it belonged to is done, and applying it now would
                # mix a delta measured against a superseded global model.
                return False
            if payload.get("status") == "ok":
                self.results[cluster] = payload
            else:
                self.failures[cluster] = str(payload.get("error", "client reported failure"))
            self.lock.notify_all()
            return True

    def wait_for_all(self, deadline: float, still_live=None) -> None:
        """Wait for every expected cluster, the deadline, or the last one going silent.

        `still_live` is what stops a dead site costing a full round_timeout: clients
        heartbeat while they train, so a cluster that has stopped heartbeating is gone
        rather than busy, and there is nothing left to wait for.
        """
        with self.lock:
            while time.monotonic() < deadline:
                answered = set(self.results) | set(self.failures)
                if answered >= self.expected:
                    return
                if still_live is not None:
                    outstanding = self.expected - answered
                    if outstanding and not (outstanding & set(still_live())):
                        return
                self.lock.wait(timeout=1.0)


def _jsonable(config: dict) -> dict:
    """Flower's Scalar allows bytes; JSON does not. Nothing in `fedproto` sends bytes,
    so this is a guard rather than an encoder -- if it ever trips, the fix is to stop
    putting bytes in the round config, not to base64 it here and hide the growth."""
    out = {}
    for k, v in config.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            raise TypeError(
                f"round config key {k!r} is {type(v).__name__}, which the HTTP round "
                f"protocol does not carry. Weights belong in the blob store."
            )
    return out


def make_handler(token: str, manager: HttpClientManager, state: RoundState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):        # noqa: A003 - silence per-request noise
            return

        def _send(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _authorised(self) -> bool:
            if not token:
                return True
            return self.headers.get("X-DARL-Token", "") == token

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):                          # noqa: N802 - BaseHTTPRequestHandler API
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "round": state.server_round})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):                         # noqa: N802
            if not self._authorised():
                self._send(401, {"error": "bad token"})
                return
            try:
                payload = self._body()
            except Exception as exc:               # noqa: BLE001
                self._send(400, {"error": f"bad json: {exc}"})
                return
            cluster = str(payload.get("cluster", ""))
            if not cluster:
                self._send(400, {"error": "cluster is required"})
                return

            if self.path.startswith("/join"):
                manager.touch(cluster)
                logger.info("cluster %r joined the HTTP round protocol (%d rank(s))",
                            cluster, int(payload.get("ranks", 0)))
                self._send(200, {"ok": True, "round": state.server_round})
                return

            if self.path.startswith("/poll"):
                # Heartbeat and instruction fetch are the same call on purpose: a site
                # that is polling is by definition still alive, so membership needs no
                # separate keepalive that could disagree with it.
                manager.touch(cluster)
                deadline = time.monotonic() + POLL_HOLD_S
                with state.lock:
                    while True:
                        if state.finished:
                            state.stopped.add(cluster)
                            state.lock.notify_all()
                            self._send(200, {"action": "stop"})
                            return
                        if state.kind and cluster in state.expected \
                                and cluster not in state.results \
                                and cluster not in state.failures:
                            self._send(200, {
                                "action": state.kind,
                                "round": state.server_round,
                                "config": _jsonable(state.config),
                            })
                            return
                        if time.monotonic() >= deadline:
                            self._send(200, {"action": "wait"})
                            return
                        state.lock.wait(timeout=1.0)

            if self.path.startswith("/result"):
                manager.touch(cluster)
                accepted = state.submit(cluster, int(payload.get("round", -1)), payload)
                self._send(200, {"ok": True, "accepted": accepted})
                return

            self._send(404, {"error": "not found"})

    return Handler


def run_rounds(strategy, manager: HttpClientManager, state: RoundState, *,
               num_rounds: int, round_timeout: float, min_clients: int) -> None:
    """Flower's `Server.fit()` loop, driven over HTTP instead of a gRPC stream.

    Deliberately the same shape and the same calls, so `PWWFedMom` cannot tell the
    difference: configure_fit -> collect -> aggregate_fit -> configure_evaluate ->
    collect -> aggregate_evaluate. Anything that behaves differently between the two
    transports would be a bug in here, not a feature.
    """
    from flwr.common import Code, EvaluateRes, FitRes, Parameters, Status

    def _collect(kind: str, server_round: int, ins) -> tuple[list, list]:
        clusters = [proxy.cid for proxy, _ in ins]
        configs = {proxy.cid: one.config for proxy, one in ins}
        first = configs[clusters[0]]
        if any(configs[c] != first for c in clusters):
            # The strategy sends one config to everyone today. If that ever changes,
            # fail loudly rather than silently broadcasting the first cluster's.
            raise RuntimeError("per-cluster round configs are not supported here yet")
        state.open(server_round, kind, first, clusters)
        state.wait_for_all(time.monotonic() + round_timeout, still_live=manager.live)
        with state.lock:
            got, bad = dict(state.results), dict(state.failures)
            missing = [c for c in clusters if c not in got and c not in bad]
        state.close()
        live_now = set(manager.live())
        for c in missing:
            bad[c] = ("stopped heartbeating mid-round" if c not in live_now
                      else f"no result within {round_timeout:.0f}s")
        ok = Status(code=Code.OK, message="")
        if kind == "fit":
            results = [
                (HttpClientProxy(c),
                 FitRes(status=ok,
                        parameters=Parameters(tensors=[], tensor_type=proto.BLOB_PARAMETERS_TYPE),
                        num_examples=int(p.get("num_examples", 0)),
                        metrics=p.get("metrics", {})))
                for c, p in got.items()
            ]
        else:
            results = [
                (HttpClientProxy(c),
                 EvaluateRes(status=ok,
                             loss=float(p.get("loss", 0.0)),
                             num_examples=int(p.get("num_examples", 0)),
                             metrics=p.get("metrics", {})))
                for c, p in got.items()
            ]
        failures = [RuntimeError(f"{c}: {why}") for c, why in bad.items()]
        return results, failures

    parameters = strategy.initialize_parameters(client_manager=manager)
    logger.info("HTTP round protocol ready: %d round(s), timeout %.0fs, min_clients %d",
                num_rounds, round_timeout, min_clients)

    for server_round in range(1, num_rounds + 1):
        while manager.num_available() < min_clients:
            time.sleep(2.0)

        fit_ins = strategy.configure_fit(server_round, parameters, manager)
        if not fit_ins:
            continue
        logger.info("[ROUND %d] asking %s to train", server_round,
                    [p.cid for p, _ in fit_ins])
        results, failures = _collect("fit", server_round, fit_ins)
        merged, _ = strategy.aggregate_fit(server_round, results, failures)
        if merged is not None:
            parameters = merged

        eval_ins = strategy.configure_evaluate(server_round, parameters, manager)
        if eval_ins:
            e_results, e_failures = _collect("evaluate", server_round, eval_ins)
            strategy.aggregate_evaluate(server_round, e_results, e_failures)

    with state.lock:
        state.finished = True
        state.lock.notify_all()
    logger.info("HTTP round protocol finished after %d round(s)", num_rounds)


def serve(strategy, *, host: str, port: int, token: str, num_rounds: int,
          round_timeout: float, min_clients: int) -> None:
    """Start the endpoint and drive the rounds. Blocks, like start_server does."""
    manager = HttpClientManager()
    state = RoundState()
    httpd = ThreadingHTTPServer((host, port), make_handler(token, manager, state))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    logger.info("HTTP round endpoint on http://%s:%d | token %s",
                host, port, "enforced" if token else "DISABLED")
    try:
        run_rounds(strategy, manager, state, num_rounds=num_rounds,
                   round_timeout=round_timeout, min_clients=min_clients)
        # Bounded: a site that has already died must not hold the node open, but one
        # that is mid-round deserves the chance to be told the run is over.
        grace = time.monotonic() + 3 * POLL_HOLD_S
        with state.lock:
            while time.monotonic() < grace:
                outstanding = [c for c in manager.live() if c not in state.stopped]
                if not outstanding:
                    break
                state.lock.wait(timeout=1.0)
            else:
                outstanding = [c for c in manager.live() if c not in state.stopped]
        if outstanding:
            logger.warning("shutting down without %s collecting the stop signal; "
                           "they will see a connection error rather than a clean end",
                           outstanding)
    finally:
        httpd.shutdown()
