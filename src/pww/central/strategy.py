# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Momentum (FedMom) [Huo et al., 2020], with elastic membership.

Paper: arxiv.org/abs/2002.02090

Three things this does that a stock server-momentum strategy does not, each driven by
what running across queued HPC allocations actually requires.

**Elastic membership.** Clusters arrive when Slurm lets them, not when the run wants
them. Every count of live replicas is a normal state:

  0 live   every site is queued. `initialize_parameters` answers from durable state
           instead of returning None, so the server never blocks asking a client for
           the architecture, and it simply waits in `client_manager` until someone
           connects. Nothing is lost, including across a restart of the server itself.
  1 live   one site trains alone. That is DiLoCo with k=1 -- correct, not degraded --
           and the momentum buffer keeps accumulating across the gap. The outer step
           takes eta=1 on such a round, because with nothing to average eta<1 would
           only discard that site's progress; see `_aggregate_inline`.
  N live   the ordinary case.

A cluster joining at round 400 needs no special handling: `configure_fit` hands every
participant the current global weights before it trains, so a newcomer cannot
contribute an update derived from a stale or freshly-initialised model.

**Generation checking.** A delta is only meaningful against the global model it was
computed from. Each carries its `base_round` and the merge rejects any that do not
match the current merge counter. This is the situation the whole design exists for: a
cluster killed at walltime mid-round and requeued hours later would otherwise have its
stale update averaged in as though it were current. It is the cheap equivalent of the
`quorum_id`/`max_step` guard a peer-to-peer scheme needs a consensus service to get.

**Two transports.** `inline` keeps weights in the gRPC message -- simple, and capped
by gRPC's 2 GiB per-message limit at roughly 1B parameters in float16. `blob` sends
only a name and moves the bytes over HTTP, with the merge streamed one tensor at a
time on the server (see `globalstate.py`), which is what makes 7B and beyond possible
at all.

On not using the forked FedMom
------------------------------
An earlier version of this module preferred `flwr.server.strategy.FedMom` from the
fedmom-strategy fork when it was importable, and fell back to a local class otherwise.
That is now a trap rather than a convenience: the fork has no durable state, no
generation checking, no blob transport, and does its arithmetic in whatever dtype
arrives -- so on a central node with the fork installed, every fix here would be
silently bypassed. The local implementation is therefore always used, and the presence
of the fork is logged and ignored.
"""

from __future__ import annotations

from collections.abc import Callable
from logging import ERROR, INFO, WARNING
from pathlib import Path
from typing import Any

from .. import fedproto as proto
from ..logging_utils import get_logger

logger = get_logger("pww.central.strategy")

HAS_FLWR = False
try:
    import flwr as fl  # noqa: F401
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False

FedMom = None

if HAS_FLWR:
    import numpy as np
    from flwr.common import (
        EvaluateIns,
        FitIns,
        FitRes,
        MetricsAggregationFn,
        NDArrays,
        Parameters,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.common.logger import log
    from flwr.server.client_manager import ClientManager
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy.fedavg import FedAvg

    try:
        from flwr.server.strategy import FedMom as _ForkedFedMom  # noqa: F401
        logger.info(
            "the forked flwr FedMom is installed but deliberately not used -- it has "
            "no durable state, generation checking or blob transport; see this "
            "module's docstring"
        )
    except ImportError:
        pass

    from .globalstate import Contribution, GlobalState, StaleContribution

    class PWWFedMom(FedAvg):
        """FedMom with durable state, elastic membership and a choice of transport."""

        def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 1,
            min_evaluate_clients: int = 1,
            min_available_clients: int = 1,
            evaluate_fn: (
                Callable[
                    [int, NDArrays, dict[str, Scalar]],
                    tuple[float, dict[str, Scalar]] | None,
                ]
                | None
            ) = None,
            on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
            on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
            accept_failures: bool = True,
            initial_parameters: Parameters | None = None,
            fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
            evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
            server_learning_rate: float = 1.0,
            server_momentum: float = 0.9,
            transport: str = proto.TRANSPORT_INLINE,
            state: GlobalState | None = None,
            blob_root: str | Path | None = None,
            blob_url: str = "",
            run_id: str = "pww",
            keep_rounds: int = 1,
            # Global-model checkpoints, two tiers. See `_checkpoint`.
            checkpoint_dir: str | Path | None = None,
            keep_ephemeral: int = 2,
            persist_every: int = 5,
            keep_persistent: int = 4,
            # eta = 1 for a round with a single contributor. See `_aggregate_inline`.
            solo_full_step: bool = True,
        ) -> None:
            super().__init__(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=on_fit_config_fn,
                on_evaluate_config_fn=on_evaluate_config_fn,
                accept_failures=accept_failures,
                initial_parameters=initial_parameters,
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            )
            if transport not in proto.TRANSPORTS:
                raise ValueError(
                    f"transport must be one of {proto.TRANSPORTS}, got {transport!r}"
                )
            self.server_learning_rate = server_learning_rate
            self.server_momentum = server_momentum
            self.transport = transport
            self.state = state
            self.blob_root = Path(blob_root) if blob_root else None
            self.blob_url = blob_url
            self.run_id = run_id
            self.keep_rounds = max(0, int(keep_rounds))

            if transport == proto.TRANSPORT_BLOB:
                if state is None or self.blob_root is None or not blob_url:
                    raise ValueError(
                        "blob transport needs state=, blob_root= and blob_url= -- the "
                        "server reads deltas straight off the local blob directory and "
                        "tells clients where to reach the store"
                    )

            # Inline-transport state. The authoritative global model is float32
            # regardless of what crosses the wire: momentum accumulates over hundreds
            # of rounds, and round-tripping it through float16 once per round would
            # quantise it a little further every time.
            self.v_vector: NDArrays | None = None
            self._global_fp32: NDArrays | None = None
            self._inline_round = 0
            self._inline_global_step = 0
            self.solo_full_step = bool(solo_full_step)
            self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
            self.keep_ephemeral = max(0, int(keep_ephemeral))
            self.persist_every = max(0, int(persist_every))
            self.keep_persistent = max(0, int(keep_persistent))
            if self.checkpoint_dir is not None:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # The dtype the clients put on the wire, remembered.
            #
            # It used to be a local inside `_aggregate_inline`, discovered from the
            # incoming layers and discarded, so the paths that leave this strategy
            # *without* a merge -- `_current_parameters` after a failed round, and
            # `initialize_parameters` -- had no way to know it and handed Flower the
            # float32 authoritative copy instead. At 0.6B that is 2.4 GiB against
            # gRPC's hard 2 GiB message cap, so the next configure_fit could not be
            # sent, which produced another failed round, which took the same exit
            # again: one crashed round wedged the server permanently.
            self._wire_dtype: Any = None

        def __repr__(self) -> str:
            return (
                f"PWWFedMom(transport={self.transport}, lr={self.server_learning_rate}, "
                f"momentum={self.server_momentum}, accept_failures={self.accept_failures})"
            )

        # --- round bookkeeping --------------------------------------------

        @property
        def merge_round(self) -> int:
            """How many times the global model has actually changed.

            Deliberately not Flower's `server_round`, which advances even when every
            cluster was killed at walltime and nothing was merged. Deltas are stamped
            against this.
            """
            return self.state.round if self.state is not None else self._inline_round

        @property
        def global_step(self) -> int:
            """Server-authoritative optimiser step counter.

            With per-site H (different inner_steps on each cluster), a client can no
            longer compute ``global_step = round * H`` locally.  The server tracks the
            token-weighted average of steps across all contributing clusters and
            broadcasts this so every site's LR schedule stays aligned.
            """
            return self.state.global_step if self.state is not None else self._inline_global_step

        @property
        def _has_global(self) -> bool:
            if self.transport == proto.TRANSPORT_BLOB:
                return self.state is not None and self.state.initialised
            return self._global_fp32 is not None

        # --- startup ------------------------------------------------------

        def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
            """Answer without needing a client, so a run can start with zero live sites.

            Returning None makes Flower block until a client connects and then ask it
            for the architecture. That is what made every site being queued a deadlock
            rather than a wait.
            """
            if self.transport == proto.TRANSPORT_BLOB:
                # Weights never travel in the message here, so there is nothing to
                # fetch from a client -- the architecture arrives via the init blob.
                if self._has_global:
                    log(INFO, "resuming at merge round %d from durable state", self.merge_round)
                else:
                    log(INFO, "cold start: the first cluster to connect will seed the global model")
                return Parameters(tensors=[], tensor_type=proto.BLOB_PARAMETERS_TYPE)

            if self._global_fp32 is not None:
                return self._on_wire(self._global_fp32)
            return self.initial_parameters

        # --- fit ----------------------------------------------------------

        def _round_config(self, server_round: int) -> dict[str, Scalar]:
            config: dict[str, Scalar] = {
                proto.TRANSPORT: self.transport,
                proto.ROUND: self.merge_round,
                proto.RUN_ID: self.run_id,
                proto.GLOBAL_STEP: self.global_step,
            }
            if self.on_fit_config_fn is not None:
                config.update(self.on_fit_config_fn(server_round))
            if self.transport != proto.TRANSPORT_BLOB:
                return config

            config[proto.BLOB_URL] = self.blob_url
            if self._has_global:
                config[proto.GLOBAL_BLOB] = proto.global_blob(self.run_id, self.merge_round)
            else:
                config[proto.NEED_INIT] = "1"
                config[proto.INIT_BLOB] = proto.init_blob(self.run_id)
            return config

        def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> list[tuple[ClientProxy, FitIns]]:
            self.initial_parameters = parameters
            if self.transport != proto.TRANSPORT_BLOB and self._global_fp32 is None:
                # Round 1 only: adopt whatever the initial parameters are, then own
                # them from here.
                adopted = parameters_to_ndarrays(parameters)
                # Record the wire dtype *before* upcasting, or it is gone. This is the
                # only place it is observable on a cold start, because the first merge
                # may never happen -- which is exactly the case that wedged the server.
                if self._wire_dtype is None and adopted:
                    self._wire_dtype = adopted[0].dtype
                self._global_fp32 = [layer.astype(np.float32) for layer in adopted]

            config = self._round_config(server_round)

            if self.transport == proto.TRANSPORT_BLOB and not self._has_global:
                # Exactly one cluster seeds the model. Letting several upload
                # concurrently would leave the global model depending on which HTTP
                # PUT finished last, and their deltas measured against different
                # starting points.
                clients = client_manager.sample(num_clients=1, min_num_clients=1)
                log(INFO, "cold start: asking %s to seed the global model",
                    [c.cid[:8] for c in clients])
            else:
                sample_size, minimum = self.num_fit_clients(client_manager.num_available())
                clients = client_manager.sample(
                    num_clients=sample_size, min_num_clients=minimum
                )

            outgoing = (
                Parameters(tensors=[], tensor_type=proto.BLOB_PARAMETERS_TYPE)
                if self.transport == proto.TRANSPORT_BLOB
                else parameters
            )
            return [(client, FitIns(outgoing, config)) for client in clients]

        def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> list[tuple[ClientProxy, EvaluateIns]]:
            if self.fraction_evaluate == 0.0 or not self._has_global:
                return []
            config: dict[str, Scalar] = {
                proto.TRANSPORT: self.transport,
                proto.ROUND: self.merge_round,
                proto.RUN_ID: self.run_id,
            }
            if self.on_evaluate_config_fn is not None:
                config.update(self.on_evaluate_config_fn(server_round))
            if self.transport == proto.TRANSPORT_BLOB:
                config[proto.BLOB_URL] = self.blob_url
                config[proto.GLOBAL_BLOB] = proto.global_blob(self.run_id, self.merge_round)
                parameters = Parameters(tensors=[], tensor_type=proto.BLOB_PARAMETERS_TYPE)

            sample_size, minimum = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=minimum
            )
            return [(client, EvaluateIns(parameters, config)) for client in clients]

        def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[tuple[ClientProxy, FitRes] | BaseException],
        ) -> tuple[Parameters | None, dict[str, Scalar]]:
            if failures:
                self._log_failures(server_round, failures)
            if not results:
                log(WARNING,
                    "Round %s: no results (%d failure(s)); global model unchanged at "
                    "merge round %d", server_round, len(failures), self.merge_round)
                return self._current_parameters(), {}
            if not self.accept_failures and failures:
                return self._current_parameters(), {}

            metrics = self._aggregate_metrics(results)
            if self.transport == proto.TRANSPORT_BLOB:
                return self._aggregate_blob(server_round, results, metrics)
            return self._aggregate_inline(server_round, results, metrics)

        # --- global-model checkpoints ---------------------------------------

        def _checkpoint(self, merge_round: int) -> Path | None:
            """Snapshot the merged global model and its momentum buffer.

            Two tiers, because they answer different questions:

            **ephemeral** -- written every merge, only the newest `keep_ephemeral`
            retained. This is crash recovery: if the aggregator dies, the run resumes
            from a round or two ago rather than from scratch. Frequent and cheap to
            discard.

            **persistent** -- every `persist_every` merges, retained separately. This
            is the one that matters for a poisoned run: it is a coarse history you can
            roll *back* to after discovering that the model went bad several rounds
            ago. An ephemeral-only scheme cannot do that, because by the time a human
            reads the loss the good state has already been evicted -- which is exactly
            what happened here: nan entered at round 9 and was noticed at round 18.

            Both are pruned oldest-first, so the newest N survive.

            Why this exists at all: with `transport=blob` the global model is written
            through `GlobalState`, but every `self.state` call is on the blob path, so
            **inline transport persisted nothing**. `--state-dir` was set, the log said
            "durable state in ...", and the directory stayed empty through eighteen
            successful merges. There was nothing to resume from and nothing to roll
            back to.

            Cost, stated because it is not small: weights and momentum are both
            float32, so one checkpoint is 2x the parameter count in bytes -- ~4.8 GiB
            at 0.6B. The defaults keep 2 ephemeral + 4 persistent, so ~29 GiB steady
            state. `keep_persistent=0` means unbounded, which over 200 rounds at
            `persist_every=5` is 40 checkpoints, ~192 GiB: check the filesystem first.
            """
            if self.checkpoint_dir is None or self._global_fp32 is None:
                return None
            persistent = (self.persist_every > 0
                          and merge_round % self.persist_every == 0)
            if not persistent and self.keep_ephemeral == 0:
                return None

            tier = "persistent" if persistent else "ephemeral"
            name = f"round-{merge_round:06d}-{tier}.npz"
            final = self.checkpoint_dir / name
            tmp = final.with_suffix(".npz.tmp")
            payload = {f"w{i}": layer for i, layer in enumerate(self._global_fp32)}
            if self.v_vector is not None:
                payload.update({f"v{i}": layer for i, layer in enumerate(self.v_vector)})
            payload["meta"] = np.array(
                [merge_round, len(self._global_fp32),
                 0 if self.v_vector is None else len(self.v_vector),
                 self._inline_global_step], dtype=np.int64)
            try:
                # Uncompressed on purpose: these are float32 weights, which compress
                # by a few percent for several minutes of CPU. Write to .tmp and
                # rename, so a kill mid-write cannot leave a half file that looks
                # loadable -- the same reason the DARL snapshot does it this way.
                with open(tmp, "wb") as handle:
                    np.savez(handle, **payload)
                tmp.replace(final)
            except OSError as exc:
                log(ERROR, "could not write checkpoint %s: %s", name, exc)
                tmp.unlink(missing_ok=True)
                return None

            size = final.stat().st_size
            logger.info("checkpoint %s (%s, %.2f GiB)", name, tier, size / 2 ** 30)
            self._prune_checkpoints(tier, self.keep_persistent if persistent
                                    else self.keep_ephemeral)
            return final

        def _prune_checkpoints(self, tier: str, keep: int) -> None:
            """Keep the newest `keep`; 0 means unbounded."""
            if self.checkpoint_dir is None or keep <= 0:
                return
            existing = sorted(self.checkpoint_dir.glob(f"round-*-{tier}.npz"))
            for stale in existing[:-keep]:
                try:
                    freed = stale.stat().st_size
                    stale.unlink()
                    logger.info("pruned %s (%s, freed %.2f GiB)",
                                stale.name, tier, freed / 2 ** 30)
                except OSError as exc:
                    log(WARNING, "could not prune %s: %s", stale.name, exc)

        def latest_checkpoint(self) -> Path | None:
            """Newest checkpoint of either tier, for a resume or a rollback."""
            if self.checkpoint_dir is None:
                return None
            found = list(self.checkpoint_dir.glob("round-*-ephemeral.npz"))
            found += list(self.checkpoint_dir.glob("round-*-persistent.npz"))
            # Sort by the round in the name, not mtime: a rollback may reinstate an
            # older round, and mtime would then pick the wrong one.
            return max(found, key=lambda p: p.name.split("-")[1], default=None)

        def resume_from_checkpoint(self) -> int | None:
            """Adopt the newest *finite* checkpoint, newest-first. None if there is none.

            Newest-first, but "newest that is not poisoned" rather than simply newest.
            That distinction is the whole reason for two tiers: when a run dies from a
            non-finite merge, the newest checkpoints are the dead ones, and blindly
            loading the latest would reinstate exactly the state you are trying to
            escape. Walking backwards past them lands on the last good round -- which,
            with ephemerals evicted, is typically a persistent one.

            Loading here rather than in `initialize_parameters` is deliberate: setting
            `_global_fp32` makes that method return these weights, so Flower never asks a
            client to seed the model and a resumed run cannot be re-seeded from whatever
            site happens to connect first.
            """
            if self.checkpoint_dir is None:
                return None
            candidates = sorted(
                list(self.checkpoint_dir.glob("round-*-ephemeral.npz"))
                + list(self.checkpoint_dir.glob("round-*-persistent.npz")),
                key=lambda p: p.name.split("-")[1],
                reverse=True,
            )
            for path in candidates:
                try:
                    merge_round = self.load_checkpoint(path)
                except Exception as exc:                              # noqa: BLE001
                    log(WARNING, "checkpoint %s is unreadable (%s); trying the one "
                        "before it", path.name, exc)
                    continue
                if all(np.all(np.isfinite(layer)) for layer in self._global_fp32 or []):
                    if path is not candidates[0]:
                        log(WARNING,
                            "skipped %d newer checkpoint(s) containing nan/inf; resumed "
                            "at merge round %d instead. The rounds after it trained on a "
                            "poisoned model and are not recoverable.",
                            candidates.index(path), merge_round)
                    return merge_round
                log(WARNING, "checkpoint %s contains nan/inf; trying the one before it",
                    path.name)
            self._global_fp32 = None
            self.v_vector = None
            self._inline_round = 0
            self._inline_global_step = 0
            if candidates:
                log(ERROR, "every checkpoint in %s contains nan/inf -- cold starting from "
                    "a client instead", self.checkpoint_dir)
            return None

        def load_checkpoint(self, path: str | Path) -> int:
            """Reinstate weights and momentum from a checkpoint. Returns its round."""
            with np.load(path) as data:
                meta = data["meta"]
                merge_round, n_w, n_v = int(meta[0]), int(meta[1]), int(meta[2])
                self._global_fp32 = [data[f"w{i}"] for i in range(n_w)]
                self.v_vector = [data[f"v{i}"] for i in range(n_v)] if n_v else None
                # global_step was added after the initial checkpoint format; older
                # checkpoints have only 3 elements in the meta array.
                self._inline_global_step = int(meta[3]) if len(meta) > 3 else 0
            self._inline_round = merge_round
            logger.info("restored global model from %s at merge round %d "
                        "(global_step=%d)",
                        Path(path).name, merge_round, self._inline_global_step)
            return merge_round

        def _log_failures(self, server_round: int, failures) -> None:
            """Say *why* a round produced no results.

            Flower reports failures as a count, and this strategy used to pass that
            count straight through -- so "received 0 results and 1 failures" repeated
            every round with nothing to act on. A failure is either a raised exception
            or a client reply the transport rejected, and in both cases the reason is
            sitting in the object being discarded. Losing it turns a five-second
            diagnosis into reading stack traces on a compute node at the other end of
            a WAN.
            """
            for failure in failures:
                if isinstance(failure, BaseException):
                    log(ERROR, "Round %s failure: %s: %s", server_round,
                        type(failure).__name__, failure)
                    continue
                # (ClientProxy, FitRes): the client answered, but with a status the
                # strategy could not use. `status` carries the client's own reason.
                try:
                    proxy, res = failure
                    status = getattr(res, "status", None)
                    log(ERROR, "Round %s failure from client %s: code=%s message=%r",
                        server_round, getattr(proxy, "cid", "?"),
                        getattr(status, "code", "?"), getattr(status, "message", ""))
                except Exception:                                     # noqa: BLE001
                    log(ERROR, "Round %s failure (unparseable): %r", server_round, failure)

        def _on_wire(self, arrays: NDArrays) -> Parameters:
            """Serialise for the wire, at the dtype the clients actually use.

            Every exit from this strategy goes through here, so a round that produced
            no merge hands back a payload the *same size* as one that did. The
            authoritative copy stays float32 -- see the comment on `v_vector` for why
            the momentum state must not be round-tripped through float16 -- but nothing
            that size ever goes into a gRPC message.
            """
            if self._wire_dtype is not None and self._wire_dtype != np.float32:
                arrays = [layer.astype(self._wire_dtype) for layer in arrays]
            params = ndarrays_to_parameters(arrays)
            # gRPC's cap is hard: 2**31 - 1 bytes, and no setting raises it. Exceeding
            # it fails the *send*, which Flower reports as an ordinary round failure
            # with no indication of the cause -- 18 rounds of "0 results, 1 failure"
            # with the reason nowhere in the log. If it is ever about to happen again,
            # say so here rather than leaving it to be inferred.
            total = sum(len(t) for t in params.tensors)
            if total >= 2_147_483_647:
                log(ERROR,
                    "outgoing parameters are %.2f GiB at dtype %s, at or above gRPC's "
                    "2 GiB per-message cap -- this send will fail and the round will be "
                    "reported as a plain failure. Lower flower.wire_dtype, or move this "
                    "run to transport=blob.",
                    total / 2 ** 30, self._wire_dtype)
            return params

        def _current_parameters(self) -> Parameters | None:
            if self.transport == proto.TRANSPORT_BLOB:
                return Parameters(tensors=[], tensor_type=proto.BLOB_PARAMETERS_TYPE)
            if self._global_fp32 is None:
                return self.initial_parameters
            return self._on_wire(self._global_fp32)

        def _aggregate_metrics(self, results) -> dict[str, Scalar]:
            if self.fit_metrics_aggregation_fn:
                return self.fit_metrics_aggregation_fn(
                    [(res.num_examples, res.metrics) for _, res in results]
                )
            return {}

        # --- blob transport ------------------------------------------------

        def _reject_duplicate_clusters(self, contributions, server_round):
            """Refuse a round where two clients claim the same cluster id.

            The DARL coordinator normally stops this at registration -- two live
            processes cannot hold one cluster id. But a run can reach here without ever
            registering: `pww_qwen3_local` uses torchtitan's own dataloader, so a config
            with `flower.enable = true` and no DARL has no coordinator to refuse it.

            Left unchecked the damage is not a lost round, it is a *wrong* one. Delta
            blobs are named (run, round, cluster), so both clients wrote to the same
            object and one overwrote the other. Two Contributions then point at the
            same file, and the merge sums `share_i * delta` over both -- so the
            surviving delta is counted twice, carrying the combined weight of both
            jobs, while the other's work is gone. Nothing about the result looks wrong.

            Dropping the whole duplicated set is deliberate: with one file and two token
            counts there is no way to tell which job's weights survived, so there is no
            correct weight to give it. Skipping the round costs H steps from those
            clusters; merging it corrupts the global model.
            """
            seen: dict[str, int] = {}
            for item in contributions:
                seen[item.cluster] = seen.get(item.cluster, 0) + 1
            duplicated = {name for name, count in seen.items() if count > 1}
            if not duplicated:
                return contributions

            for name in sorted(duplicated):
                log(ERROR,
                    "Round %s: %d clients reported as cluster %r. Their delta blobs "
                    "share a name, so one silently overwrote the other and the "
                    "survivor would be counted twice. Dropping every contribution "
                    "under that id. Give each concurrent job its own cluster id: "
                    "--replica a / --replica b on scripts/titan/run_train.sh.",
                    server_round, seen[name], name)
            return [item for item in contributions if item.cluster not in duplicated]

        def _aggregate_blob(self, server_round, results, metrics):
            assert self.state is not None and self.blob_root is not None

            if not self.state.initialised:
                if not self._seed_from_results(results):
                    return self._current_parameters(), metrics

            contributions: list[Contribution] = []
            for _, res in results:
                cluster = str(res.metrics.get(proto.CLUSTER, "unknown"))
                blob = str(res.metrics.get(proto.DELTA_BLOB, ""))
                if not blob:
                    log(WARNING, "cluster %r returned no delta blob; skipping", cluster)
                    continue
                if res.num_examples <= 0:
                    # Trained nothing -- DARL ran dry, or it was killed early. An
                    # honest zero, so it carries no weight rather than dragging the
                    # average toward its untouched weights.
                    log(INFO, "cluster %r reported 0 tokens; not merging its delta", cluster)
                    continue
                path = self.blob_root / blob
                if not path.is_file():
                    log(WARNING,
                        "delta %s from %r is not in the blob store -- the upload did "
                        "not complete", blob, cluster)
                    continue
                contributions.append(
                    Contribution(
                        cluster=cluster,
                        path=path,
                        weight=float(res.num_examples),
                        tokens=int(res.num_examples),
                        steps=int(res.metrics.get(proto.STEPS, 0)),
                        base_round=int(res.metrics.get(proto.BASE_ROUND, -1)),
                        blob=blob,
                    )
                )

            contributions = self._reject_duplicate_clusters(contributions, server_round)

            if not contributions:
                log(WARNING,
                    "Round %s: nothing mergeable; global model stays at merge round %d",
                    server_round, self.state.round)
                return self._current_parameters(), metrics

            try:
                new_round = self.state.merge(
                    contributions,
                    server_learning_rate=self.server_learning_rate,
                    server_momentum=self.server_momentum,
                )
            except StaleContribution as exc:
                log(WARNING, "Round %s: %s", server_round, exc)
                return self._current_parameters(), metrics

            self._publish_global(new_round)
            self._prune(new_round)
            metrics = dict(metrics)
            metrics["merge_round"] = new_round
            metrics["clusters_merged"] = len(contributions)
            return self._current_parameters(), metrics

        def _seed_from_results(self, results) -> bool:
            """Adopt the init blob a cold-start cluster uploaded."""
            expected = proto.init_blob(self.run_id)
            for _, res in results:
                if str(res.metrics.get(proto.UPLOADED_INIT, "")) != "1":
                    continue
                path = self.blob_root / expected
                if not path.is_file():
                    log(WARNING, "cluster claimed to seed %s but it is not present", expected)
                    continue
                self.state.initialise_from_file(path)
                self._publish_global(self.state.round)
                return True
            log(WARNING,
                "no cluster seeded the global model this round; nothing to merge into")
            return False

        def _publish_global(self, round_index: int) -> None:
            """Expose the current global model under this round's blob name.

            Hard-linked rather than copied: the file is up to 280 GiB at 70B, and the
            blob store lives on the same volume as the state directory precisely so
            this is free. A copy would double both the disk use and the round's
            wall-clock time.
            """
            name = proto.global_blob(self.run_id, round_index)
            target = self.blob_root / name
            if target.exists():
                return
            source = self.state.global_path
            try:
                import os

                os.link(source, target)
            except OSError:
                # Different filesystem, or a system without hard links. Correct but
                # expensive, so say so.
                import shutil

                log(WARNING,
                    "could not hard-link the global model into the blob store; "
                    "copying %s instead. Put --state-dir and --blob-root on the same "
                    "filesystem to avoid this.", name)
                shutil.copyfile(source, target)

        def _prune(self, current_round: int) -> None:
            """Drop blobs from finished rounds.

            Without this a 7B run leaves ~14 GiB per site per round on disk and fills
            the volume within an hour. `keep_rounds` retains a couple of recent
            globals so a cluster that was mid-download when the round advanced can
            still finish it.
            """
            keep = {
                proto.global_blob(self.run_id, current_round - offset)
                for offset in range(self.keep_rounds + 1)
                if current_round - offset >= 0
            }
            keep.add(proto.init_blob(self.run_id))
            removed = 0
            for entry in self.blob_root.iterdir():
                if not entry.is_file() or entry.name.startswith("."):
                    continue
                if entry.name in keep:
                    continue
                try:
                    entry.unlink()
                    removed += 1
                except OSError as exc:
                    log(WARNING, "could not prune %s: %s", entry.name, exc)
            if removed:
                logger.info("pruned %d blob(s) from finished rounds", removed)

        # --- inline transport ----------------------------------------------

        def _aggregate_inline(self, server_round, results, metrics):
            total_examples = sum(res.num_examples for _, res in results)
            if total_examples <= 0:
                log(WARNING,
                    "Round %s: all %d cluster(s) reported 0 examples; keeping the "
                    "current global model and aggregating nothing",
                    server_round, len(results))
                return self._current_parameters(), metrics

            # The weighted mean is computed here rather than through flwr's own
            # `aggregate`, and everything is float32 regardless of what arrived. Two
            # reasons, both load-bearing once clients send float16 to keep a 0.6B model
            # under the 2 GiB gRPC cap:
            #
            #   overflow  `aggregate` forms `layer * num_examples` before dividing.
            #             num_examples is a token count (order 1e6) and float16
            #             saturates at 65504, so every parameter would become inf on
            #             the first round. Normalised weights are bounded by 1.
            #   drift     momentum accumulates over hundreds of rounds; held in
            #             float16 it quantises each update toward zero.
            # A cluster whose weights contain nan or inf is dropped, not averaged.
            #
            # This is not defensive tidiness. A single poisoned contribution does not
            # degrade the average, it destroys it: nan propagates through the weighted
            # sum, into w_next, into `self._global_fp32`, and from there into every
            # future round via configure_fit -- so one bad round from one site ends the
            # run, and the logs keep reporting healthy token counts and "merge complete"
            # while doing it. Observed exactly once, and it cost 9 rounds across two
            # allocations before anyone read the loss: a site joined at round 9, its
            # first contribution was nan, and rounds 10-18 were arithmetic on nan.
            #
            # Dropping it degrades that to a skipped merge for one site, which is the
            # same handling as a stale delta and is already a supported state (k varies
            # by round in this implementation -- see the DiLoCo notes in the guide).
            wire_dtype = None
            weighted_sum: NDArrays | None = None
            usable: list[tuple[Any, Any]] = []
            for proxy, res in results:
                if res.num_examples <= 0:
                    continue
                layers = parameters_to_ndarrays(res.parameters)
                bad = next((i for i, layer in enumerate(layers)
                            if not np.all(np.isfinite(layer))), None)
                if bad is not None:
                    log(ERROR,
                        "Round %s: dropping cluster %r -- tensor %d of %d contains "
                        "nan/inf, and averaging it would poison the global model "
                        "permanently. Its %s tokens are not counted this round. A site "
                        "producing this is diverging locally: check its loss, its "
                        "learning rate, and for float16 overflow on the wire.",
                        server_round,
                        res.metrics.get(proto.CLUSTER, getattr(proxy, "cid", "?")),
                        bad, len(layers), f"{res.num_examples:,}")
                    continue
                usable.append((proxy, res))
                if wire_dtype is None and layers:
                    wire_dtype = layers[0].dtype
                share = res.num_examples / total_examples
                scaled = [layer.astype(np.float32) * share for layer in layers]
                if weighted_sum is None:
                    weighted_sum = scaled
                else:
                    for index, layer in enumerate(scaled):
                        weighted_sum[index] += layer

            if not usable:
                log(WARNING,
                    "Round %s: every cluster's contribution was non-finite; keeping the "
                    "current global model unchanged", server_round)
                return self._current_parameters(), metrics
            if len(usable) != len([r for r in results if r[1].num_examples > 0]):
                # Shares were computed against the original total, so renormalise over
                # what survived or the merge silently under-weights the good sites.
                kept = sum(res.num_examples for _, res in usable)
                rescale = total_examples / kept
                weighted_sum = [layer * rescale for layer in (weighted_sum or [])]
                log(WARNING, "Round %s: merging %d of %d cluster(s), reweighted over "
                    "%s surviving tokens", server_round, len(usable), len(results),
                    f"{kept:,}")
            fedavg_result: NDArrays = weighted_sum or []

            w_t: NDArrays = self._global_fp32 or [
                layer.astype(np.float32)
                for layer in parameters_to_ndarrays(self.initial_parameters)
            ]

            # eta = 1 when there is only one contributor, because with one contributor
            # there is nothing to average.
            #
            # w_avg is then that cluster's own weights, so the update reduces to
            #
            #     w_next = (1 - eta) * w + eta * w_local
            #
            # and eta < 1 is pure damping: it discards (1 - eta) of the round's local
            # progress in exchange for variance reduction across replicas that does not
            # exist here. At the paper's eta = 0.7 that is 30% of every solo round thrown
            # away. It is also not a hyperparameter question -- DiLoCo's eta is tuned for
            # the averaging case, and this is the degenerate one.
            #
            # It matters because solo rounds are the normal opening of a run: min-clients
            # is 1 precisely so the first site out of the queue starts training instead of
            # idling, and FEDERATION_GUIDE already promises that is "DiLoCo with k=1 --
            # correct, not degraded". Damping made that claim false.
            #
            # Momentum is deliberately left alone. Round 1 is momentum-free by
            # construction (v_prev = w), so this alone makes the opening round exactly
            # "adopt the local model", and changing beta as well would alter the behaviour
            # of a long run that merely loses a site for a while.
            #
            # solo_full_step = False restores the old behaviour. The reason to want it is
            # that eta < 1 bounds how far one cluster can drag the global model, which is
            # a real property if you do not trust a site -- but the non-finite check above
            # and the drift metric are the intended guards for that.
            solo = len(usable) == 1 and self.solo_full_step
            eta = 1.0 if solo else self.server_learning_rate
            if solo:
                log(INFO,
                    "Round %s: single contributor (%s), so applying the outer step at "
                    "eta=1.0 instead of %.2f -- with nothing to average, damping would "
                    "discard %.0f%% of this round for no benefit",
                    server_round,
                    usable[0][1].metrics.get(proto.CLUSTER, "?"),
                    self.server_learning_rate,
                    100.0 * (1.0 - self.server_learning_rate))

            pseudo_gradient = [w - w_avg for w, w_avg in zip(w_t, fedavg_result)]
            v_next = [w - eta * pg for w, pg in zip(w_t, pseudo_gradient)]
            v_prev: NDArrays = w_t if self.v_vector is None else self.v_vector
            w_next = [
                vn + self.server_momentum * (vn - vp)
                for vn, vp in zip(v_next, v_prev)
            ]

            self.v_vector = v_next
            self._global_fp32 = w_next
            self._inline_round += 1

            # Advance the server-authoritative global step by the token-weighted
            # average of steps each cluster completed.  With identical H this is
            # exactly H; with per-site H it places the schedule at the honest
            # midpoint.
            kept_tokens = sum(res.num_examples for _, res in usable)
            if kept_tokens > 0:
                step_increment = round(sum(
                    (res.num_examples / kept_tokens)
                    * int(res.metrics.get(proto.STEPS, 0))
                    for _, res in usable
                ))
                self._inline_global_step += max(1, step_increment) if step_increment > 0 else 0

            # Keep it, so a later round that fails to merge leaves at the same dtype.
            if wire_dtype is not None:
                self._wire_dtype = wire_dtype
            aggregated = self._on_wire(w_next)
            self.initial_parameters = aggregated
            # After the merge, so a checkpoint is always a state the run actually had.
            self._checkpoint(self._inline_round)

            metrics = dict(metrics)
            metrics["merge_round"] = self._inline_round
            metrics["global_step"] = self._inline_global_step
            logger.info(
                "round %s inline FedMom merge complete (%d cluster(s), %s tokens, "
                "global_step=%d)",
                server_round, len(results), f"{total_examples:,}",
                self._inline_global_step,
            )
            return aggregated, metrics

    FedMom = PWWFedMom
