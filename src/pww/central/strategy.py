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
           and the momentum buffer keeps accumulating across the gap.
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
from logging import INFO, WARNING
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
                return ndarrays_to_parameters(self._global_fp32)
            return self.initial_parameters

        # --- fit ----------------------------------------------------------

        def _round_config(self, server_round: int) -> dict[str, Scalar]:
            config: dict[str, Scalar] = {
                proto.TRANSPORT: self.transport,
                proto.ROUND: self.merge_round,
                proto.RUN_ID: self.run_id,
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
                self._global_fp32 = [
                    layer.astype(np.float32)
                    for layer in parameters_to_ndarrays(parameters)
                ]

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

        def _current_parameters(self) -> Parameters | None:
            if self.transport == proto.TRANSPORT_BLOB:
                return Parameters(tensors=[], tensor_type=proto.BLOB_PARAMETERS_TYPE)
            if self._global_fp32 is None:
                return self.initial_parameters
            return ndarrays_to_parameters(self._global_fp32)

        def _aggregate_metrics(self, results) -> dict[str, Scalar]:
            if self.fit_metrics_aggregation_fn:
                return self.fit_metrics_aggregation_fn(
                    [(res.num_examples, res.metrics) for _, res in results]
                )
            return {}

        # --- blob transport ------------------------------------------------

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
                        base_round=int(res.metrics.get(proto.BASE_ROUND, -1)),
                        blob=blob,
                    )
                )

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
            wire_dtype = None
            weighted_sum: NDArrays | None = None
            for _, res in results:
                if res.num_examples <= 0:
                    continue
                share = res.num_examples / total_examples
                layers = parameters_to_ndarrays(res.parameters)
                if wire_dtype is None and layers:
                    wire_dtype = layers[0].dtype
                scaled = [layer.astype(np.float32) * share for layer in layers]
                if weighted_sum is None:
                    weighted_sum = scaled
                else:
                    for index, layer in enumerate(scaled):
                        weighted_sum[index] += layer
            fedavg_result: NDArrays = weighted_sum or []

            w_t: NDArrays = self._global_fp32 or [
                layer.astype(np.float32)
                for layer in parameters_to_ndarrays(self.initial_parameters)
            ]

            pseudo_gradient = [w - w_avg for w, w_avg in zip(w_t, fedavg_result)]
            v_next = [
                w - self.server_learning_rate * pg
                for w, pg in zip(w_t, pseudo_gradient)
            ]
            v_prev: NDArrays = w_t if self.v_vector is None else self.v_vector
            w_next = [
                vn + self.server_momentum * (vn - vp)
                for vn, vp in zip(v_next, v_prev)
            ]

            self.v_vector = v_next
            self._global_fp32 = w_next
            self._inline_round += 1

            if wire_dtype is not None and wire_dtype != np.float32:
                on_wire = [layer.astype(wire_dtype) for layer in w_next]
            else:
                on_wire = w_next
            aggregated = ndarrays_to_parameters(on_wire)
            self.initial_parameters = aggregated

            metrics = dict(metrics)
            metrics["merge_round"] = self._inline_round
            logger.info(
                "round %s inline FedMom merge complete (%d cluster(s), %s tokens)",
                server_round, len(results), f"{total_examples:,}",
            )
            return aggregated, metrics

    FedMom = PWWFedMom
