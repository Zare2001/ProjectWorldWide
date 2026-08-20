"""Client side of the HTTP round protocol -- the loop `fl.client.start_client` replaces.

`DiLoCoFlowerClient` is reused verbatim. Flower only ever calls three of its methods,
and this calls the same three in the same order, so every piece of the round -- the
blob fetch, the rank broadcast, the inner steps, the delta upload, the metrics -- is
shared with the gRPC path rather than reimplemented beside it.

What changes is only how the instruction arrives: a short POST that returns, instead
of a message on a stream held open for the whole job. That matters at a site reachable
only through an HTTP forward proxy, where the long stream is reaped mid-round but
discrete request/response has never once failed -- the same reason the DARL client and
the blob store work there already.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from .logging_utils import get_logger

logger = get_logger("pww.http_round")


class HttpRoundClient:
    def __init__(self, url: str, cluster_id: str, *, token: str = "",
                 use_proxy: bool = False, retries: int = 6, backoff: float = 2.0):
        self.url = url.rstrip("/")
        self.cluster_id = cluster_id
        self.token = token
        self.retries = retries
        self.backoff = backoff
        # Same rule as the DARL client: bypass the proxy unless the site says it needs
        # one. A site that needs it for DARL needs it here too -- both are plain HTTP
        # to the same host.
        handlers = [] if use_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)

    def _post(self, route: str, payload: dict, *, timeout: float = 60.0) -> dict:
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-DARL-Token"] = self.token
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            request = urllib.request.Request(
                f"{self.url}{route}", data=body, headers=headers, method="POST"
            )
            try:
                with self._opener.open(request, timeout=timeout) as response:
                    return json.loads(response.read() or b"{}")
            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    raise RuntimeError(
                        "the round server rejected this token; pass the value the "
                        "central node was started with"
                    ) from exc
                last = exc
            except Exception as exc:                                   # noqa: BLE001
                # The failure this whole transport exists to survive: a dropped
                # connection is one retried request, not the end of the run.
                last = exc
            wait = self.backoff * attempt
            logger.warning("%s failed (%s), retry %d/%d in %.1fs",
                           route, last, attempt, self.retries, wait)
            time.sleep(wait)
        raise RuntimeError(f"{route} failed after {self.retries} attempts: {last}")

    def join(self, ranks: int) -> dict:
        return self._post("/join", {"cluster": self.cluster_id, "ranks": ranks})

    def start_heartbeat(self, ranks: int, every: float = 20.0) -> None:
        """Keep saying "still here" while the inner phase runs.

        Without this the server cannot tell a site that is training -- legitimately
        silent for the whole of H steps, minutes at a time -- from one that has died.
        Its only alternative would be to wait out round_timeout for every dead site,
        which at 1800s is half an hour of every other site idling per round.
        """
        def beat() -> None:
            while True:
                time.sleep(every)
                try:
                    self._post("/join", {"cluster": self.cluster_id, "ranks": ranks},
                               timeout=30.0)
                except Exception:                                    # noqa: BLE001
                    # Never fatal: the round itself is what matters, and a missed
                    # heartbeat costs at most one round of membership.
                    pass

        threading.Thread(target=beat, daemon=True).start()

    def poll(self) -> dict:
        # Longer than the server's hold, so a held poll returns normally rather than
        # timing out on this side and looking like a network fault.
        return self._post("/poll", {"cluster": self.cluster_id}, timeout=120.0)

    def report(self, server_round: int, kind: str, payload: dict) -> None:
        message = {"cluster": self.cluster_id, "round": server_round, "kind": kind}
        message.update(payload)
        self._post("/result", message)


def _scalars(metrics: dict) -> dict:
    """Metrics as JSON carries them. Anything else is dropped rather than crashing a
    round: a metric is telemetry, and losing one is never worth losing the merge."""
    out = {}
    for k, v in (metrics or {}).items():
        if isinstance(v, bool) or isinstance(v, (int, float, str)):
            out[k] = v
        else:
            logger.debug("dropping non-scalar metric %r (%s)", k, type(v).__name__)
    return out


def run_http_client(client, *, url: str, cluster_id: str, ranks: int,
                    token: str = "", use_proxy: bool = False) -> None:
    """Drive `client` from the central node's HTTP round endpoint until it says stop."""
    session = HttpRoundClient(url, cluster_id, token=token, use_proxy=use_proxy)
    session.join(ranks)
    session.start_heartbeat(ranks)
    logger.info("joined the HTTP round protocol at %s as %r", url, cluster_id)

    while True:
        reply = session.poll()
        action = str(reply.get("action", "wait"))
        if action == "wait":
            continue
        if action == "stop":
            logger.info("the central node has finished its rounds; disconnecting")
            return

        server_round = int(reply.get("round", 0))
        config = reply.get("config", {}) or {}
        try:
            if action == "fit":
                _, num_examples, metrics = client.fit([], config)
                session.report(server_round, "fit", {
                    "status": "ok",
                    "num_examples": int(num_examples),
                    "metrics": _scalars(metrics),
                })
            elif action == "evaluate":
                loss, num_examples, metrics = client.evaluate([], config)
                session.report(server_round, "evaluate", {
                    "status": "ok",
                    "loss": float(loss),
                    "num_examples": int(num_examples),
                    "metrics": _scalars(metrics),
                })
            else:
                logger.warning("unknown action %r from the round server", action)
        except Exception as exc:                                       # noqa: BLE001
            # Tell the server rather than going quiet: it can then close the round on
            # the spot instead of waiting out round_timeout for a result that is never
            # coming, which on a 1800s timeout is half an hour of every other site idle.
            logger.exception("round %d (%s) failed locally", server_round, action)
            session.report(server_round, action, {"status": "error", "error": str(exc)})
            raise
