"""Client side of DARL: RPCs, heartbeats, and the local view of held spans.

One `LeaseSession` per *cluster*, not per rank. The session runs on the site
leader (global rank 0 of the job) and everything it acquires is distributed to
the other ranks over the existing process group -- see `torch_data.py`. Ten
thousand ranks hammering a coordinator with per-rank requests is exactly the
"naive dynamic assignment" failure mode the design contrasts DARL against.

Four things here are load-bearing and easy to get wrong:

**Acquires carry an idempotency key.** Every other RPC is naturally idempotent,
so a lost reply can be retried. An acquire is not: retrying it blind after a
dropped response leaks a lease -- blocks marked LEASED that nobody is working on
until the TTL expires. So each acquire carries a `request_id` and the coordinator
replies with the cached grant if it sees the same key twice.

**One acquire in flight per session.** The prefetch runs on a background thread, so
it is easy for a phase's foreground acquire to overlap it -- and then two requests
from the same cluster are queued at the coordinator, competing for the same blocks.
At the tail of an epoch only one can win, the loser is told `drain`, and if the
loser is the foreground caller it waits on blocks its own prefetch just took. That
deadlocks: the thread that would consume and commit them is the one asleep, and the
heartbeat keeps renewing the lease so the TTL never breaks it either. Two things
prevent it -- `take_prefetched` waits for a request already in flight instead of
racing it, and `acquire` consumes a pending prefetch grant rather than sleeping on a
drain.

**The heartbeat reply is an instruction, not an acknowledgement.** It carries the
coordinator's authoritative `end` for each lease, which may have shrunk because
the tail was stolen, and `valid: false` for leases that were reaped while this
cluster was silent. `LeaseSession` applies both before the trainer is allowed to
draw another sample, which is what keeps stealing from duplicating work.

**Commit means durable, not consumed.** See `CommitPolicy`.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..logging_utils import get_logger
from .space import BlockSpace
from .table import HEARTBEAT_DIVISOR, MIN_TTL, compute_ttl

DEFAULT_URL = os.environ.get("PWW_DARL_URL", "")


class DarlError(RuntimeError):
    """Any coordinator interaction that failed and will not be retried."""


class ClusterBusy(DarlError):
    """Another live process is already registered under this cluster id.

    Two concurrent jobs sharing an id corrupt each other's leases and overwrite each
    other's weight deltas, so the coordinator refuses rather than letting it happen
    quietly. Temporary by construction -- it clears when the incumbent's leases expire --
    so the client retries before giving up. Give each job its own id to avoid the wait:
    `--replica a` / `--replica b` on scripts/titan/run_train.sh.
    """


class LeaseGone(DarlError):
    """The lease expired and its blocks went back to the pool.

    Raised on a commit for a lease the coordinator no longer knows about, which
    means this cluster was declared dead (heartbeats stopped for longer than the
    TTL -- a hung node, a paused job, a network partition). The samples are
    someone else's now: discard the span, do not count its work, acquire again.
    """


class CommitPolicy:
    """When a client tells the coordinator a span is done.

    CHECKPOINT   after the span's work is inside a durable model checkpoint. This
                 is the exact one: {theta, M_committed} is then a single
                 consistent unit, so a job that dies loses the same work from
                 both sides and the epoch stays exactly-once.

    CONSUMPTION  as soon as the samples have been fed to the model. One fewer
                 dependency between the dataloader and the checkpointer, and the
                 span is released for others sooner -- at the cost of an exactness
                 window: a crash after committing but before checkpointing loses
                 those samples from the epoch (a gap), and an expiry after
                 consuming but before committing gives another cluster the same
                 samples (a duplicate). Both are bounded by one lease.

    Pick CHECKPOINT for a run whose data coverage you intend to claim, CONSUMPTION
    when leases are much shorter than the checkpoint interval and you would rather
    not stall the pool.
    """

    CHECKPOINT = "checkpoint"
    CONSUMPTION = "consumption"
    ALL = (CHECKPOINT, CONSUMPTION)


@dataclass
class Span:
    """A lease as the client sees it.

    `end` and `valid` are owned by the coordinator and updated by heartbeats;
    `consumed` and `committed` are local watermarks. Everything the trainer reads
    goes through `remaining()` so a mid-phase revocation cannot be missed.
    """

    lease_id: str
    epoch: int
    start: int
    end: int
    ttl: float
    deadline: float
    consumed: int = 0            # positions handed to the dataloader
    committed: int = 0           # positions the coordinator has been told about
    valid: bool = True

    def __post_init__(self) -> None:
        self.consumed = max(self.consumed, self.start)
        self.committed = max(self.committed, self.start)

    @property
    def blocks(self) -> int:
        return max(0, self.end - self.start)

    def remaining(self) -> int:
        """Blocks still safe to consume. Zero once revoked."""
        return max(0, self.end - self.consumed) if self.valid else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id, "epoch": self.epoch, "start": self.start,
            "end": self.end, "consumed": self.consumed, "committed": self.committed,
            "valid": self.valid,
        }


@dataclass
class Acquisition:
    """Outcome of one acquire."""

    status: str                          # granted | drain | epoch_complete
    epoch: int
    spans: list[Span] = field(default_factory=list)
    retry_after: float = 0.0
    reason: str = ""

    @property
    def granted(self) -> bool:
        return self.status == "granted" and bool(self.spans)

    @property
    def epoch_complete(self) -> bool:
        return self.status == "epoch_complete"

    @property
    def blocks(self) -> int:
        return sum(span.blocks for span in self.spans)


class LeaseClient:
    """Thin, retrying JSON-over-HTTP client. Stdlib only, so it runs anywhere."""

    def __init__(
        self,
        url: str,
        cluster_id: str,
        *,
        token: str = "",
        timeout: float = 30.0,
        retries: int = 6,
        backoff: float = 1.5,
        use_proxy: bool = False,
    ):
        if not url:
            raise ValueError(
                "no coordinator URL. Pass --darl-url or set PWW_DARL_URL, e.g. "
                "http://int-node-1.snellius.surf.nl:8760"
            )
        self.url = url.rstrip("/")
        self.cluster_id = cluster_id
        self.token = token or os.environ.get("DARL_TOKEN", "")
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        self.rtt_s = 0.0
        self.calls = 0
        self.retried = 0
        # Identifies this *process*, where cluster_id identifies the logical stream.
        # The two have different lifetimes and conflating them is what let two
        # concurrent jobs at one site silently release each other's leases: a requeued
        # job must keep the cluster id (its measured throughput sizes its grants) but
        # must not be mistaken for its own predecessor. Random rather than derived,
        # because nothing in the environment is both stable across a requeue and unique
        # across concurrent jobs -- SLURM_JOB_ID changes on requeue, and the site name
        # does not change between jobs.
        self.incarnation = uuid.uuid4().hex
        # Compute nodes usually have http_proxy set so that they can reach the
        # internet through a slow gateway. The coordinator is normally *inside*
        # the facility, and routing to it through that proxy either fails or adds
        # seconds, so proxies are bypassed unless asked for. Cross-site clients
        # that genuinely need the proxy pass use_proxy=True.
        handlers = [] if use_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)

    # --- transport --------------------------------------------------------

    def _call(self, route: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """One RPC, retried on transport and 5xx failures with jittered backoff.

        Jitter matters more than it looks: without it, m clusters whose leases
        were granted at the same moment retry in lockstep after a coordinator
        restart, and the first thing the coordinator sees is a synchronised burst.
        """
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-DARL-Token"] = self.token
        log = get_logger()

        last_error: Exception | None = None
        busy = ""
        for attempt in range(self.retries):
            request = urllib.request.Request(f"{self.url}{route}", data=data,
                                             headers=headers,
                                             method="POST" if data is not None else "GET")
            t0 = time.monotonic()
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    body = json.loads(response.read() or b"{}")
                elapsed = time.monotonic() - t0
                # EWMA over successful calls only -- a timeout is not an RTT
                # measurement, and folding it in would inflate every TTL.
                self.rtt_s = elapsed if self.rtt_s <= 0 else 0.8 * self.rtt_s + 0.2 * elapsed
                self.calls += 1
                return body
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                try:
                    parsed = json.loads(detail)
                    message = parsed.get("error", detail)
                    code = parsed.get("code", "")
                except json.JSONDecodeError:
                    message, code = detail, ""
                if exc.code == 409 or code == "lease_gone":
                    raise LeaseGone(message) from None
                if code == "cluster_busy":
                    # A 5xx, so it falls through to the retry loop below -- which is
                    # what we want, since the incumbent's leases expire on a timer.
                    # Remembered so that exhausting the retries reports *this* rather
                    # than a generic "coordinator unreachable", which would send the
                    # next person looking at the network.
                    busy = message
                if exc.code == 401:
                    raise DarlError(
                        f"coordinator rejected the token. Set DARL_TOKEN (or "
                        f"--darl-token) to the value the coordinator was started with"
                    ) from None
                if 400 <= exc.code < 500:
                    raise DarlError(f"{route} rejected ({exc.code}): {message}") from None
                last_error = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    json.JSONDecodeError) as exc:
                last_error = exc

            self.retried += 1
            delay = min(60.0, self.backoff ** attempt) * (0.5 + random.random())
            log.warning("darl %s failed (%s), retry %d/%d in %.1fs",
                        route, last_error, attempt + 1, self.retries, delay)
            time.sleep(delay)

        if busy:
            raise ClusterBusy(
                f"{busy} (still refused after {self.retries} attempts)"
            )
        raise DarlError(
            f"coordinator unreachable at {self.url} after {self.retries} attempts: "
            f"{last_error}. The training job cannot get new data ranges; the span it "
            f"already holds is unaffected until its lease expires."
        )

    # --- RPCs -------------------------------------------------------------

    def register(self, space: BlockSpace, *, ranks: int = 0, epoch: int = 0) -> dict[str, Any]:
        """Announce this cluster and check that both sides agree on the space."""
        reply = self._call("/register", {
            "cluster": self.cluster_id,
            "digest": space.digest(epoch),
            "ranks": ranks,
            "num_samples": space.num_samples,
            "block_size": space.block_size,
            "incarnation": self.incarnation,
        })
        if reply.get("num_blocks") not in (None, space.num_blocks):
            raise DarlError(
                f"coordinator serves {reply['num_blocks']} blocks, this client computed "
                f"{space.num_blocks} -- check --num-samples and --block-size on both sides"
            )
        return reply

    def acquire(
        self,
        blocks: int,
        *,
        ttl: float = 0.0,
        max_spans: int = 4,
        macro_step_s: float = 0.0,
        request_id: str = "",
    ) -> Acquisition:
        reply = self._call("/acquire", {
            "cluster": self.cluster_id,
            "blocks": int(blocks),
            "ttl": float(ttl),
            "max_spans": int(max_spans),
            "macro_step_s": float(macro_step_s),
            "rtt_s": self.rtt_s,
            "request_id": request_id or uuid.uuid4().hex,
        })
        spans = [
            Span(lease_id=l["lease_id"], epoch=l["epoch"], start=l["start"], end=l["end"],
                 ttl=l["ttl"], deadline=l["deadline"])
            for l in reply.get("leases", [])
        ]
        return Acquisition(
            status=reply.get("status", "drain"),
            epoch=int(reply.get("epoch", 0)),
            spans=spans,
            retry_after=float(reply.get("retry_after", 0.0)),
            reason=reply.get("reason", ""),
        )

    def heartbeat(self, progress: dict[str, int], *, ttl: float = 0.0,
                  macro_step_s: float = 0.0) -> dict[str, Any]:
        return self._call("/heartbeat", {
            "cluster": self.cluster_id,
            "progress": progress,
            "ttl": float(ttl),
            "macro_step_s": float(macro_step_s),
            "rtt_s": self.rtt_s,
        })

    def commit(self, lease_id: str, through: int) -> dict[str, Any]:
        return self._call("/commit", {"cluster": self.cluster_id, "lease": lease_id,
                                      "through": int(through)})

    def release(self, lease_id: str | None = None) -> dict[str, Any]:
        # The incarnation matters on a release-everything: it is the one operation
        # whose blast radius is every lease a cluster id owns, so the coordinator
        # checks that the asker is the process that currently holds the id.
        return self._call("/release", {"cluster": self.cluster_id, "lease": lease_id,
                                       "incarnation": self.incarnation})

    def advance_epoch(self) -> dict[str, Any]:
        """Tell the coordinator to advance to the next epoch, recycling all blocks."""
        return self._call("/advance", None)

    def status(self) -> dict[str, Any]:
        return self._call("/status")


class LeaseSession:
    """Held spans, the heartbeat thread, and prefetch of the next span.

    Thread-safety: the heartbeat thread mutates `end`/`valid` on spans while the
    training thread reads them, so every access goes through `self._lock`. The
    training thread never blocks on the network -- heartbeats and prefetches are
    the background thread's job, which is what lets a cluster overlap staging the
    next span with computing on the current one.
    """

    def __init__(
        self,
        client: LeaseClient,
        space: BlockSpace,
        *,
        blocks_per_phase: int,
        ranks: int = 1,
        commit_policy: str = CommitPolicy.CHECKPOINT,
        max_spans: int = 4,
        heartbeat: bool = True,
        min_ttl: float = MIN_TTL,
    ):
        if commit_policy not in CommitPolicy.ALL:
            raise ValueError(f"commit_policy must be one of {CommitPolicy.ALL}")
        self.client = client
        self.space = space
        self.blocks_per_phase = max(1, int(blocks_per_phase))
        self.ranks = ranks
        self.commit_policy = commit_policy
        self.max_spans = max_spans
        self.min_ttl = float(min_ttl)
        # The TTL the coordinator actually granted, which is what the heartbeat
        # interval has to be derived from -- it may have clamped what we asked for.
        self.server_ttl = float(min_ttl)

        self.spans: dict[str, Span] = {}
        self.epoch = 0
        self.macro_step_s = 0.0
        self.blocks_committed = 0
        self.blocks_lost = 0
        self.epoch_complete = False

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prefetch: Acquisition | None = None
        self._prefetch_thread: threading.Thread | None = None
        # Set when a prefetch has stored its outcome, so a caller that wants the
        # next span can wait for the one already in flight instead of racing it
        # with a second acquire. See `take_prefetched`.
        self._prefetch_done = threading.Event()
        self._phase_started = 0.0

        reply = self.client.register(space, ranks=ranks)
        self.epoch = int(reply.get("epoch", 0))
        get_logger().info(
            "darl: registered %r with %s | epoch %d | %d blocks | %d blocks per phase",
            client.cluster_id, client.url, self.epoch, space.num_blocks,
            self.blocks_per_phase,
        )
        if heartbeat:
            self.start_heartbeat()

    # --- TTL and heartbeats -----------------------------------------------

    @property
    def ttl(self) -> float:
        """The TTL this cluster asks for, from its own measurements."""
        return compute_ttl(self.macro_step_s, self.client.rtt_s, floor=self.min_ttl)

    def note_phase_time(self, seconds: float) -> None:
        """Record how long one local phase took, so the TTL tracks reality.

        A cluster that gets slower -- more replicas per node, a degraded link,
        activation checkpointing switched on -- asks for a longer TTL on its next
        heartbeat, instead of being declared dead mid-phase.
        """
        if seconds <= 0:
            return
        with self._lock:
            self.macro_step_s = (
                seconds if self.macro_step_s <= 0 else 0.7 * self.macro_step_s + 0.3 * seconds
            )

    def start_heartbeat(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._heartbeat_loop, name="darl-heartbeat",
                                        daemon=True)
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        log = get_logger()
        while not self._stop.is_set():
            # Off the *granted* TTL, not the requested one: the coordinator clamps,
            # and beating slower than its clamp is how a healthy cluster loses a
            # lease it is actively working on.
            interval = max(1.0, min(self.server_ttl, self.ttl)) / HEARTBEAT_DIVISOR
            if self._stop.wait(interval):
                return
            try:
                self.heartbeat_once()
            except DarlError as exc:
                # Do not tear the run down: the span in hand is still valid until
                # its deadline, and the coordinator may well come back before then.
                log.warning("darl heartbeat failed: %s", exc)

    def heartbeat_once(self) -> dict[str, Any]:
        """Send progress, apply the reply. Safe to call from the training thread."""
        with self._lock:
            progress = {span.lease_id: span.consumed for span in self.spans.values()
                        if span.valid}
            ttl, macro = self.ttl, self.macro_step_s
        if not progress:
            return {}
        reply = self.client.heartbeat(progress, ttl=ttl, macro_step_s=macro)
        self._apply(reply)
        return reply

    def _apply(self, reply: dict[str, Any]) -> None:
        """Fold the coordinator's authoritative view into local state."""
        log = get_logger()
        with self._lock:
            self.epoch = int(reply.get("epoch", self.epoch))
            if reply.get("ttl"):
                self.server_ttl = float(reply["ttl"])
            for lease_id, info in (reply.get("leases") or {}).items():
                span = self.spans.get(lease_id)
                if span is None:
                    continue
                if not info.get("valid", False):
                    span.valid = False
                    lost = max(0, span.end - span.committed)
                    self.blocks_lost += lost
                    log.warning(
                        "darl: lease %s was reclaimed (%s) -- %d blocks of this phase "
                        "are no longer ours and their work must not be counted",
                        lease_id, info.get("reason", "expired"), lost,
                    )
                    continue
                new_end = int(info.get("end", span.end))
                if new_end < span.end:
                    log.info("darl: lease %s tail stolen, end %d -> %d (%d blocks)",
                             lease_id, span.end, new_end, span.end - new_end)
                    span.end = max(new_end, span.consumed)
                    if span.end < new_end:
                        # Cannot happen: the coordinator never cuts below the
                        # watermark it was told about. Loud if it ever does.
                        log.error("darl: steal cut below the consumed watermark")

    # --- acquiring ---------------------------------------------------------

    def acquire(self, blocks: int | None = None, *, wait: bool = True,
                timeout: float | None = None) -> Acquisition:
        """Get the next work vector, optionally waiting out a drained pool.

        `wait=True` is what a training loop wants: a `drain` reply means every
        remaining block is held by someone who is still working on it, so the
        right behaviour is to sleep until the next lease boundary and ask again,
        not to exit an epoch that is not finished.

        One `drain` must never be waited out, though: the one where the outstanding
        blocks are **this session's own**, won by a prefetch whose result nobody has
        collected yet. Sleeping there is a deadlock rather than patience -- the only
        thread that would ever commit those blocks is the one asleep in this loop,
        and the heartbeat keeps renewing the lease, so not even the TTL breaks it.
        The pool looks drained from the outside while the session is sitting on the
        very span it is waiting for. So a pending prefetch grant is adopted instead
        of slept on. Observed in practice: the tail of an epoch, where the last
        acquire and the prefetch of the phase before it are in flight together and
        only one of them can win the remaining blocks.
        """
        blocks = self.blocks_per_phase if blocks is None else blocks
        deadline = None if timeout is None else time.monotonic() + timeout
        request_id = uuid.uuid4().hex
        while True:
            with self._lock:
                ttl, macro = self.ttl, self.macro_step_s
            result = self.client.acquire(blocks, ttl=ttl, max_spans=self.max_spans,
                                         macro_step_s=macro, request_id=request_id)
            with self._lock:
                self.epoch = result.epoch
                if result.status == "epoch_complete":
                    self.epoch_complete = True
                elif result.granted:
                    # A grant proves the epoch is not over. Without this the flag
                    # stays set across an epoch boundary and anything keyed off it
                    # misreads the next epoch as already finished.
                    self.epoch_complete = False
                for span in result.spans:
                    self.spans[span.lease_id] = span
                    self.server_ttl = span.ttl or self.server_ttl
                self._phase_started = time.monotonic()
            if result.granted or result.epoch_complete or not wait:
                return result

            # A peek, not a wait: blocking here could overrun the caller's own
            # `timeout`, and it is not needed. A prefetch still in flight will have
            # landed by the next pass through this loop, which is at least a second
            # away, and the peek picks it up then.
            pending = self.take_prefetched(wait=0.0)
            if pending is not None and pending.granted:
                get_logger().info(
                    "darl: pool reported drained, but a prefetch of this session "
                    "already holds %d block(s) -- consuming those instead of waiting",
                    pending.blocks,
                )
                return pending

            if deadline is not None and time.monotonic() > deadline:
                return result
            delay = max(1.0, result.retry_after)
            get_logger().info("darl: pool drained (%s), retrying in %.0fs",
                              result.reason, delay)
            # A fresh id per attempt: this is a new request, not a retransmission.
            request_id = uuid.uuid4().hex
            time.sleep(delay)

    def start_prefetch(self, blocks: int | None = None) -> None:
        """Acquire the next work vector in the background.

        Called right after a phase's spans are in hand, so the RPC and any data
        staging overlap the GPU work of the current phase. By the time the outer
        step is reached, the next span is already known -- which is the difference
        between the coordinator's latency being invisible and it appearing once per
        phase in the critical path.
        """
        with self._lock:
            if self._prefetch is not None or (self._prefetch_thread
                                              and self._prefetch_thread.is_alive()):
                return
            self._prefetch_done.clear()

        def run() -> None:
            result: Acquisition | None = None
            try:
                result = self.acquire(blocks, wait=False)
            except DarlError as exc:
                get_logger().warning("darl prefetch failed: %s", exc)
            finally:
                # Signalled even on failure: a caller in `take_prefetched(wait=...)`
                # is waiting to find out whether a span is coming, and "it failed"
                # is an answer. Without the `finally` it would wait out its whole
                # bound for a thread that has already given up.
                with self._lock:
                    self._prefetch = result
                self._prefetch_done.set()

        self._prefetch_thread = threading.Thread(target=run, name="darl-prefetch",
                                                 daemon=True)
        self._prefetch_thread.start()

    def take_prefetched(self, *, wait: float | None = None) -> Acquisition | None:
        """Collect the prefetched work vector, waiting for one that is in flight.

        Waiting is the default, and it is what keeps a session to **one acquire at a
        time**. The alternative -- returning None while the prefetch RPC is
        outstanding, so the caller goes and issues its own acquire -- puts two
        requests from the same cluster in the coordinator's queue at once, and near
        the end of an epoch only one of them can win the remaining blocks. The loser
        is told `drain`, and if it is the foreground caller it then waits on blocks
        its own prefetch just won. `acquire` recovers from that now; not causing it
        is better, and it also saves the wasted round trip and the second lease.

        Bounded rather than unbounded, at one RPC timeout: a prefetch retrying
        against a coordinator that is not answering must not hold the training loop,
        and past the bound the caller's own acquire has the same prospects and better
        error reporting. `wait=0.0` is a non-blocking peek.
        """
        wait = self.client.timeout if wait is None else wait
        if wait > 0:
            with self._lock:
                inflight = (
                    self._prefetch is None
                    and self._prefetch_thread is not None
                    and self._prefetch_thread.is_alive()
                )
            if inflight:
                self._prefetch_done.wait(wait)
        with self._lock:
            result, self._prefetch = self._prefetch, None
            return result

    # --- progress and commits ---------------------------------------------

    def note_consumed(self, lease_id: str, through: int) -> None:
        """Move the local consumed watermark. Bounds what a thief may take."""
        with self._lock:
            span = self.spans.get(lease_id)
            if span is not None:
                span.consumed = max(span.consumed, min(through, span.end))

    def commit(self, lease_id: str, through: int | None = None) -> int:
        """Tell the coordinator a prefix of this span is durably processed.

        Returns blocks newly committed. A `LeaseGone` here is not fatal to the
        run: the span was reclaimed, someone else has it, and the caller must not
        count its work. It *is* a signal that this cluster's heartbeats are not
        getting through, which usually means a hung rank rather than a network
        problem.
        """
        with self._lock:
            span = self.spans.get(lease_id)
            if span is None:
                return 0
            through = span.consumed if through is None else min(through, span.end)
            if through <= span.committed or not span.valid:
                return 0
        try:
            reply = self.client.commit(lease_id, through)
        except LeaseGone as exc:
            with self._lock:
                span.valid = False
                self.blocks_lost += max(0, span.end - span.committed)
            get_logger().error("darl: commit of %s refused -- %s", lease_id, exc)
            return 0
        with self._lock:
            newly = max(0, through - span.committed)
            span.committed = through
            self.blocks_committed += newly
            self.epoch_complete = bool(reply.get("epoch_complete", False))
            if reply.get("lease_closed"):
                self.spans.pop(lease_id, None)
        return newly

    def commit_all(self) -> int:
        """Commit every span up to its consumed watermark.

        This is the call that pairs with a checkpoint write: do it *after* the
        checkpoint is on disk and {theta, M_committed} stay consistent under any
        crash. Do it before, and a crash in between silently drops those samples
        from the epoch.
        """
        with self._lock:
            targets = [(span.lease_id, span.consumed) for span in self.spans.values()
                       if span.valid and span.consumed > span.committed]
        return sum(self.commit(lease_id, through) for lease_id, through in targets)

    def release_all(self) -> int:
        """Hand back everything uncommitted. Wire this to SIGTERM.

        Slurm sends SIGTERM before it kills a job at walltime. Releasing there
        returns the tail in milliseconds rather than after a full TTL, which on a
        long TTL is the difference between the other clusters idling for a quarter
        of an hour and not idling at all.
        """
        with self._lock:
            if not self.spans:
                # Still drop a stale prefetch: `release` covers every lease this
                # cluster holds at the coordinator, so an uncollected grant here
                # describes blocks that are back in the pool.
                self._prefetch = None
                return 0
        try:
            reply = self.client.release(None)
        except DarlError as exc:
            get_logger().warning("darl: release failed: %s", exc)
            return 0
        with self._lock:
            self.spans.clear()
            # Must not survive the release. Whatever the prefetcher won was released
            # along with everything else, so handing it to a phase would train blocks
            # that another cluster is now free to lease -- a duplicate, and one that
            # no assertion downstream would catch.
            self._prefetch = None
        return int(reply.get("released", 0))

    # --- lifecycle --------------------------------------------------------

    def close(self, release: bool = True) -> None:
        self._stop.set()
        if release:
            self.release_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "LeaseSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close(release=exc[0] is not None or bool(self.spans))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cluster": self.client.cluster_id,
                "epoch": self.epoch,
                "blocks_committed": self.blocks_committed,
                "blocks_lost": self.blocks_lost,
                "held": [span.to_dict() for span in self.spans.values()],
                "macro_step_s": round(self.macro_step_s, 3),
                "rtt_ms": round(1000 * self.client.rtt_s, 2),
                "rpcs": self.client.calls,
                "retries": self.client.retried,
                "epoch_complete": self.epoch_complete,
            }


def main(argv: list[str] | None = None) -> None:
    """`python3 -m pww.darl.client --url http://host:8760 status`

    A read-only window into a running coordinator, for when a job is queued and
    you want to know whether the epoch is progressing.
    """
    import argparse

    from ..logging_utils import setup_logging
    from .server import format_status

    p = argparse.ArgumentParser(description="DARL coordinator client")
    p.add_argument("command", choices=("status", "json", "release"))
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--token", default=os.environ.get("DARL_TOKEN", ""))
    p.add_argument("--cluster", default="cli")
    p.add_argument("--watch", type=float, default=0.0, help="Repeat every N seconds")
    args = p.parse_args(argv)

    setup_logging(rank=0)
    client = LeaseClient(args.url, args.cluster, token=args.token)
    while True:
        if args.command == "release":
            print(json.dumps(client.release(None)))
            return
        status = client.status()
        print(json.dumps(status, indent=2) if args.command == "json"
              else format_status(status), flush=True)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
