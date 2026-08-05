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
"""Federated Momentum (FedMom) [Huo et al., 2020] strategy.

Paper: arxiv.org/abs/2002.02090
Using forked Flower repository: https://github.com/Zare2001/flower
"""

from __future__ import annotations

from collections.abc import Callable
from logging import WARNING
from typing import Any

from ..logging_utils import get_logger

logger = get_logger("pww.central.strategy")

HAS_FLWR = False
FedMom = None

try:
    import flwr as fl
    HAS_FLWR = True
    # First try importing FedMom directly from the forked Flower framework
    try:
        from flwr.server.strategy import FedMom as ForkedFedMom
        FedMom = ForkedFedMom
    except ImportError:
        try:
            from flwr.server.strategy.fedmom import FedMom as ForkedFedMom
            FedMom = ForkedFedMom
        except ImportError:
            FedMom = None
except ImportError:
    HAS_FLWR = False


if HAS_FLWR and FedMom is None:
    from flwr.common import (
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
    from flwr.server.strategy.aggregate import aggregate
    from flwr.server.strategy.fedavg import FedAvg

    class LocalFedMom(FedAvg):
        """Federated Momentum strategy from https://arxiv.org/abs/2002.02090."""

        def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
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
            self.server_learning_rate = server_learning_rate
            self.server_momentum = server_momentum
            self.v_vector: NDArrays | None = None

        def __repr__(self) -> str:
            return f"FedMom(accept_failures={self.accept_failures})"

        def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
            return self.initial_parameters

        def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> list[tuple[ClientProxy, FitIns]]:
            self.initial_parameters = parameters
            return super().configure_fit(server_round, parameters, client_manager)

        def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[tuple[ClientProxy, FitRes] | BaseException],
        ) -> tuple[Parameters | None, dict[str, Scalar]]:
            if not results:
                return None, {}
            if not self.accept_failures and failures:
                return None, {}
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]

            fedavg_result = aggregate(weights_results)

            if self.initial_parameters is None:
                raise ValueError(
                    "When using server-side optimization, model needs to be initialized."
                )
            w_t = parameters_to_ndarrays(self.initial_parameters)

            pseudo_gradient: NDArrays = [
                w - w_avg for w, w_avg in zip(w_t, fedavg_result)
            ]

            v_next: NDArrays = [
                w - self.server_learning_rate * pg
                for w, pg in zip(w_t, pseudo_gradient)
            ]

            v_prev: NDArrays = w_t if self.v_vector is None else self.v_vector

            w_next: NDArrays = [
                vn + self.server_momentum * (vn - vp)
                for vn, vp in zip(v_next, v_prev)
            ]

            parameters_aggregated = ndarrays_to_parameters(w_next)

            self.v_vector = v_next
            self.initial_parameters = parameters_aggregated

            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            logger.info(
                f"Round {server_round} FedMom aggregation complete ({len(results)} clusters)."
            )

            return parameters_aggregated, metrics_aggregated

    FedMom = LocalFedMom
