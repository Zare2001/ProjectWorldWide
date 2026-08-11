"""The authoritative global model on the central node, and the FedMom merge.

Two problems this solves that the in-memory strategy could not.

**Durability, which is what makes membership elastic.** `LocalFedMom` kept the
global weights and the momentum buffer in Python attributes, so the central node
could not survive a restart and could not start a run before a cluster connected --
`initialize_parameters` returning None makes Flower block asking a random client for
the architecture. With the state on disk, the server holds the run: every cluster
can be sitting in a Slurm queue, the server waits, and whoever arrives first
receives the current global model. Zero live replicas becomes a normal state rather
than a failure.

**Memory, which is what makes >1B possible.** FedMom needs the global weights, the
weighted mean of the clusters' weights, and the momentum buffer. Held as dense
tensors that is three full copies of the model -- 84 GB at 7B in float32, 840 GB at
70B -- on a VM with tens of gigabytes of RAM. So `merge` streams **one tensor at a
time** through `pww.tensorio`, and peak memory is a small multiple of the *largest
single tensor* (the embedding) rather than of the model:

    7B    embedding 131328 x 4096 fp32 = 2.1 GiB  ->  ~9 GiB peak
    70B   embedding 131328 x 8192 fp32 = 4.3 GiB  ->  ~17 GiB peak

Disk is then the binding constraint, not RAM. See `disk_budget`.

Generation checking
-------------------
A delta is only meaningful against the global weights it was derived from. Each one
records `base_round`, and `merge` refuses any that does not match the current round.
That is the cheap equivalent of the `quorum_id`/`max_step` guard a peer-to-peer
scheme needs a consensus service for -- and it matters in exactly the situation this
whole design is built around: a cluster killed at walltime mid-round, requeued hours
later, uploading a delta computed against a global model that has since moved on.
Without the check that stale delta would be averaged in as though it were current.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from ..logging_utils import get_logger
from ..tensorio import TensorFile, TensorWriter, dtype_name, DTYPES

logger = get_logger("pww.central.globalstate")

META_NAME = "meta.json"
GLOBAL_NAME = "global.pww"
MOMENTUM_NAME = "momentum.pww"


@dataclass
class Contribution:
    """One cluster's upload for the current round."""

    cluster: str
    path: Path
    weight: float                # normalised share of this round's tokens
    tokens: int = 0
    steps: int = 0
    base_round: int = -1
    blob: str = ""


@dataclass
class ClusterRecord:
    """What the central node remembers about a cluster across its comings and goings."""

    first_seen_round: int = 0
    last_seen_round: int = 0
    rounds_contributed: int = 0
    tokens_total: int = 0
    stale_rejected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_seen_round": self.first_seen_round,
            "last_seen_round": self.last_seen_round,
            "rounds_contributed": self.rounds_contributed,
            "tokens_total": self.tokens_total,
            "stale_rejected": self.stale_rejected,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ClusterRecord":
        return cls(**{k: int(v) for k, v in raw.items() if k in cls.__annotations__})


class StaleContribution(ValueError):
    """A delta computed against a global model that is no longer current."""


class GlobalState:
    """Durable global weights, momentum buffer, round counter and membership.

    `round` is the number of *successful merges*, deliberately not Flower's round
    counter. A Flower round in which every cluster was killed at walltime consumes a
    round number but changes nothing, and conflating the two makes "how much training
    has actually happened" unanswerable from the logs.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        storage_dtype: torch.dtype = torch.float32,
    ):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.storage_dtype = storage_dtype
        self.round = 0
        self.global_step = 0
        self.keys: tuple[str, ...] = ()
        self.clusters: dict[str, ClusterRecord] = {}
        self.total_tokens = 0
        self.model_numel = 0
        self._load_meta()

    # --- paths ------------------------------------------------------------

    @property
    def global_path(self) -> Path:
        return self.dir / GLOBAL_NAME

    @property
    def momentum_path(self) -> Path:
        return self.dir / MOMENTUM_NAME

    @property
    def meta_path(self) -> Path:
        return self.dir / META_NAME

    @property
    def initialised(self) -> bool:
        return self.global_path.is_file() and bool(self.keys)

    # --- metadata ---------------------------------------------------------

    def _load_meta(self) -> None:
        if not self.meta_path.is_file():
            return
        raw = json.loads(self.meta_path.read_text())
        self.round = int(raw.get("round", 0))
        self.global_step = int(raw.get("global_step", 0))
        self.keys = tuple(raw.get("keys", ()))
        self.total_tokens = int(raw.get("total_tokens", 0))
        self.model_numel = int(raw.get("model_numel", 0))
        stored = raw.get("storage_dtype")
        if stored in DTYPES:
            self.storage_dtype = DTYPES[stored]
        self.clusters = {
            name: ClusterRecord.from_dict(record)
            for name, record in (raw.get("clusters") or {}).items()
        }
        logger.info(
            "resumed global state from %s: round %d, %d tensors, %s tokens, "
            "clusters %s",
            self.dir, self.round, len(self.keys), f"{self.total_tokens:,}",
            sorted(self.clusters) or "none yet",
        )

    def _save_meta(self) -> None:
        payload = {
            "round": self.round,
            "global_step": self.global_step,
            "storage_dtype": dtype_name(self.storage_dtype),
            "keys": list(self.keys),
            "model_numel": self.model_numel,
            "total_tokens": self.total_tokens,
            "updated_at": time.time(),
            "clusters": {name: rec.to_dict() for name, rec in self.clusters.items()},
        }
        staging = self.meta_path.with_suffix(".json.tmp")
        staging.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(staging, self.meta_path)

    # --- lifecycle --------------------------------------------------------

    def initialise_from_file(self, path: str | os.PathLike[str]) -> None:
        """Adopt the first arriving cluster's model as the global model.

        The central node cannot invent an architecture, so on a cold start the first
        cluster to connect defines it. Every later cluster is checked against these
        keys and shapes, which is how a run with mismatched `model.flavor` or a
        different tokenizer vocabulary fails loudly instead of averaging tensors that
        do not correspond.
        """
        with TensorFile(path) as source, TensorWriter(
            self.global_path, meta={"round": 0, "kind": "global"}
        ) as out:
            for key in source.keys:
                out.add(key, source.get(key, self.storage_dtype))
            self.keys = source.keys
            self.model_numel = source.numel()
        self.round = 0
        self.momentum_path.unlink(missing_ok=True)
        self._save_meta()
        logger.info(
            "initialised global model: %d tensors, %s parameters, %s on disk (%s)",
            len(self.keys), f"{self.model_numel:,}",
            _human(self.global_path.stat().st_size), dtype_name(self.storage_dtype),
        )
        self.log_disk_budget()

    def open_global(self) -> TensorFile:
        if not self.initialised:
            raise FileNotFoundError(
                f"no global model in {self.dir} -- it is adopted from the first "
                f"cluster that connects"
            )
        return TensorFile(self.global_path)

    # --- membership -------------------------------------------------------

    def note_seen(self, cluster: str) -> ClusterRecord:
        record = self.clusters.get(cluster)
        if record is None:
            record = ClusterRecord(
                first_seen_round=self.round, last_seen_round=self.round
            )
            self.clusters[cluster] = record
            logger.info(
                "cluster %r joined at round %d (%d cluster(s) known)",
                cluster, self.round, len(self.clusters),
            )
        record.last_seen_round = self.round
        return record

    def note_stale(self, cluster: str) -> None:
        self.note_seen(cluster).stale_rejected += 1

    # --- the merge --------------------------------------------------------

    def merge(
        self,
        contributions: Iterable[Contribution],
        *,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.9,
        strict_generation: bool = True,
    ) -> int:
        """Apply FedMom to the deltas in `contributions`; returns the new round.

        Streams key by key. `contributions` weights must already be normalised to
        sum to 1 -- see `normalise`.
        """
        items = list(contributions)
        if not items:
            raise ValueError("merge called with no contributions")

        fresh: list[Contribution] = []
        for item in items:
            if strict_generation and item.base_round != self.round:
                logger.warning(
                    "rejecting delta from %r: computed against round %d, current "
                    "round is %d. The cluster was almost certainly killed at "
                    "walltime and requeued; its next round will be current.",
                    item.cluster, item.base_round, self.round,
                )
                self.note_stale(item.cluster)
                continue
            fresh.append(item)

        if not fresh:
            raise StaleContribution(
                f"every delta for round {self.round} was stale; global model unchanged"
            )

        total = sum(item.weight for item in fresh)
        if total <= 0:
            raise ValueError("contribution weights sum to zero")
        # Normalised into a local list rather than by mutating the caller's objects.
        # Weights bounded by 1 are also what keeps the accumulation below in range --
        # multiplying by raw token counts (order 1e6) is how the inline path used to
        # overflow float16 to inf.
        shares = [item.weight / total for item in fresh]

        has_momentum = self.momentum_path.is_file() and server_momentum != 0.0
        new_global = self.dir / f"{GLOBAL_NAME}.next"
        new_momentum = self.dir / f"{MOMENTUM_NAME}.next"

        deltas = [TensorFile(item.path) for item in fresh]
        try:
            self._check_shapes(deltas, fresh)
            started = time.monotonic()
            peak_bytes = 0

            with self.open_global() as current, TensorWriter(
                new_global, meta={"round": self.round + 1, "kind": "global"}
            ) as out_global, TensorWriter(
                new_momentum, meta={"round": self.round + 1, "kind": "momentum"}
            ) as out_momentum:
                previous = TensorFile(self.momentum_path) if has_momentum else None
                try:
                    for key in current.keys:
                        w_t = current.get(key, torch.float32)

                        # accumulator <- sum_i p_i * delta_i, so
                        #   w_avg = w_t + accumulator
                        #   pseudo_gradient = w_t - w_avg = -accumulator
                        #   v_next = w_t - lr * pseudo_gradient = w_t + lr * accumulator
                        accumulator = torch.zeros_like(w_t)
                        for share, delta in zip(shares, deltas):
                            accumulator.add_(delta.get(key, torch.float32), alpha=share)
                        v_next = accumulator.mul_(server_learning_rate).add_(w_t)

                        if server_momentum == 0.0:
                            w_next = v_next
                        else:
                            # v_prev is w_t on the very first merge, which makes
                            # round 1 momentum-free -- there is no previous
                            # velocity to extrapolate from.
                            v_prev = previous.get(key, torch.float32) if previous else w_t.clone()
                            # w_next = v_next + beta*(v_next - v_prev)
                            #        = (1+beta)*v_next - beta*v_prev
                            # written into v_prev's buffer to avoid a fourth copy.
                            w_next = v_prev.mul_(-server_momentum).add_(
                                v_next, alpha=1.0 + server_momentum
                            )

                        out_global.add(key, w_next.to(self.storage_dtype))
                        out_momentum.add(key, v_next.to(self.storage_dtype))
                        peak_bytes = max(peak_bytes, w_t.numel() * 4 * 4)
                finally:
                    if previous is not None:
                        previous.close()
        finally:
            for handle in deltas:
                handle.close()

        os.replace(new_global, self.global_path)
        os.replace(new_momentum, self.momentum_path)

        # Membership is recorded against the round the clusters actually trained
        # against, before the counter advances -- for both first_seen and last_seen, so
        # the two are comparable. A cluster that first appears in the merge producing
        # round N trained against N-1: that is the global model it saw, and recording N
        # instead would misreport every join by one round.
        for item in fresh:
            record = self.note_seen(item.cluster)
            record.rounds_contributed += 1
            record.tokens_total += item.tokens
            self.total_tokens += item.tokens

        # Advance the global step by the LARGEST number of steps any cluster
        # took, not the token-weighted average.
        #
        # The average looks fairer and is the wrong quantity. It is strictly less than the
        # fast site's H, and `align_to_global_step` only moved a site forward -- so the
        # fast site outran the counter, its alignment became a permanent no-op, and the
        # slow site alone was pulled to a value neither of them was at. The two schedules
        # then diverged monotonically: with H=200/100 they are 265 steps apart after six
        # rounds and growing, which is exactly the "learning rate differs across clusters"
        # warning.
        #
        # A midpoint is a point where *no* replica is. max() is the only increment under
        # which every site can sit at the same place, which is the whole purpose of a
        # server-authoritative step -- and it matches what the run actually advanced by,
        # since the global model absorbed the fast site's work.
        #
        # With identical H this is exactly H, unchanged.
        step_increment = max((item.steps for item in fresh), default=0)
        if step_increment > 0:
            self.global_step += step_increment

        self.round += 1
        self._save_meta()

        logger.info(
            "round %d merged from %d cluster(s) in %.1fs (peak ~%s, lr=%g, momentum=%g, "
            "global_step=%d)",
            self.round, len(fresh), time.monotonic() - started,
            _human(peak_bytes), server_learning_rate, server_momentum,
            self.global_step,
        )
        return self.round

    def _check_shapes(self, deltas: list[TensorFile], items: list[Contribution]) -> None:
        """Refuse to average tensors that do not correspond.

        Two clusters running different `model.flavor`, or the same flavor against
        different tokenizers, produce parameter sets that are the same *shape of
        object* but mean different things. Averaging them elementwise produces a
        model that trains without erroring and never converges, which is a far worse
        outcome than a startup failure.
        """
        expected = set(self.keys)
        # Shapes come from the header, so checking every key costs no reads.
        with self.open_global() as current:
            want = {key: tuple(current.spec(key)["shape"]) for key in current.keys}

        for item, delta in zip(items, deltas):
            missing = expected - set(delta.keys)
            extra = set(delta.keys) - expected
            if missing or extra:
                raise ValueError(
                    f"delta from {item.cluster!r} does not match the global model: "
                    f"{len(missing)} missing key(s) {sorted(missing)[:3]}, "
                    f"{len(extra)} unexpected {sorted(extra)[:3]}. Both clusters must "
                    f"run the same model.flavor and the same tokenizer."
                )
            for key, shape in want.items():
                got = tuple(delta.spec(key)["shape"])
                if shape != got:
                    raise ValueError(
                        f"delta from {item.cluster!r}: {key} has shape {got}, global "
                        f"has {shape} -- mismatched model.flavor or vocab size"
                    )

    # --- reporting --------------------------------------------------------

    def disk_budget(self, sites: int = 2) -> dict[str, int]:
        """Bytes needed on this volume, and what is available.

        The global model and momentum buffer are permanent; one delta per site is
        transient but all of them exist at once during a merge.
        """
        element = torch.finfo(self.storage_dtype).bits // 8 if self.storage_dtype.is_floating_point else 4
        resident = self.model_numel * element * 2          # global + momentum
        transient = self.model_numel * 2 * max(1, sites)   # deltas, bfloat16
        return {
            "resident": resident,
            "transient": transient,
            "total": resident + transient,
            "free": shutil.disk_usage(self.dir).free,
        }

    def log_disk_budget(self, sites: int = 2) -> None:
        budget = self.disk_budget(sites)
        logger.info(
            "disk budget: %s resident (global + momentum) + %s transient "
            "(%d delta) = %s, %s free",
            _human(budget["resident"]), _human(budget["transient"]), sites,
            _human(budget["total"]), _human(budget["free"]),
        )
        if budget["total"] > budget["free"]:
            logger.error(
                "NOT ENOUGH DISK: this run needs %s but only %s is free on %s. "
                "Point --state-dir and the blob store at a larger volume before "
                "starting, or the merge will fail partway through a round.",
                _human(budget["total"]), _human(budget["free"]), self.dir,
            )

    def summary(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "global_step": self.global_step,
            "initialised": self.initialised,
            "tensors": len(self.keys),
            "parameters": self.model_numel,
            "total_tokens": self.total_tokens,
            "storage_dtype": dtype_name(self.storage_dtype),
            "clusters": {name: rec.to_dict() for name, rec in self.clusters.items()},
        }


def normalise(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        return [0.0 for _ in weights]
    return [value / total for value in weights]


def _human(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"
