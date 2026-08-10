"""The Flower half of the cross-site outer step, over either transport.

One DiLoCo inner phase is one Flower `fit` call: take the global weights, run H local
steps, hand back what changed. `trainer.FederatedTrainer` runs the phase and knows
nothing about Flower; this module moves the weights.

Two transports, and the server picks per round
----------------------------------------------
``inline``  weights ride inside the gRPC message as numpy arrays. Simple, and hard
            capped by gRPC's 2 GiB per-message limit -- roughly 1B parameters in
            float16, and no setting raises it. `_check_transport_ceiling` runs that
            arithmetic at startup so an oversized model fails immediately with the
            numbers rather than as a truncated message mid-round.
``blob``    the message carries a blob *name*; the bytes go over HTTP to the central
            blob store, and both sides stream one tensor at a time. No size ceiling,
            and peak host memory is a small multiple of the largest single tensor
            instead of the whole model -- which is the difference between 7B working
            and not.

The client obeys whatever the round's `config` says, so switching transport is a
central-node decision and no cluster config has to agree with another.

Rank topology
-------------
Only global rank 0 speaks gRPC and HTTP. Every other rank waits on a broadcast that
says which collectives to enter next, and with which file paths. The weights themselves
never cross the process group: in blob mode every rank reads the same file from shared
scratch, which is cheaper than broadcasting a model that is already on local disk.

Staging files therefore have to live somewhere every rank of the job can read -- the
run's dump folder, which is on scratch. `_staging_dir`.

Reporting
---------
`num_examples` is the true token count and is never floored to 1. That floor is what
let a WikiText run report "1 sample, loss 0.0" for 23 consecutive rounds after DARL ran
dry, so FedMom kept averaging untouched weights while the log showed zero failures.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import flwr as fl
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from .. import fedproto as proto
from ..delta import BlobClient, stream_apply_global, stream_gather_delta, stream_write_full
from ..logging_utils import get_logger
from .params import (
    ParameterCodec,
    as_plain_tensor,
    gather_full_state,
    outer_agreement,
    scatter_full_state,
)
from .trainer import FederatedTrainer

logger = get_logger("pww.titan.flower_client")

CMD_FIT = "fit"
CMD_EVAL = "evaluate"
CMD_STOP = "stop"


class DiLoCoFlowerClient(fl.client.NumPyClient):
    """Flower client whose one `fit` call is one DiLoCo inner phase.

    `initial_state` is passed in rather than gathered here because gathering is a
    collective: every rank has to take part, but only rank 0 constructs this object.
    `train.py` does the gather on all ranks and hands the result down.
    """

    def __init__(
        self,
        federated: FederatedTrainer,
        initial_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.federated = federated
        self.trainer = federated.trainer
        cfg = federated.job_config.flower
        self.wire_dtype = cfg.wire_dtype
        self.declared_transport = cfg.transport
        self.cluster_id = (
            federated.job_config.darl.cluster_id or federated.job_config.darl.site or "pww"
        )

        # None under blob transport: the codec exists to fix a parameter *ordering* for
        # the inline wire format, and building it needs a full gather of the model --
        # the very operation blob transport exists to avoid.
        self.codec = (
            ParameterCodec.from_state_dict(initial_state, wire_dtype=self.wire_dtype)
            if initial_state is not None
            else None
        )
        self._inline_checked = False
        self._reference = initial_state

        self._blob: BlobClient | None = None
        self._blob_url = ""
        self._cached_global = ""
        self.done = False
        if self.codec is not None:
            logger.info(
                "model: %s tensors, %s parameters | cluster %r | transport %s",
                f"{len(self.codec.keys):,}", f"{self.codec.numel:,}",
                self.cluster_id, self.declared_transport,
            )
        else:
            logger.info(
                "cluster %r | transport %s (weights move out of band; the model is "
                "never gathered whole)", self.cluster_id, self.declared_transport,
            )

    def _require_transport(self, requested: str) -> None:
        """Fail loudly on a client/server transport mismatch.

        Not recoverable at this point: under blob transport the client deliberately
        never built the parameter ordering the inline path needs, and building it
        mid-round would be a surprise all-gather of the whole model.
        """
        if requested == self.declared_transport:
            return
        raise RuntimeError(
            f"transport mismatch: the central node selected {requested!r} but this "
            f"cluster was configured for {self.declared_transport!r}. Set "
            f"flower.transport to {requested!r} on every cluster, or restart the "
            f"server with --transport {self.declared_transport}."
        )

    # --- staging ----------------------------------------------------------

    def _staging_dir(self) -> Path:
        """Where downloaded globals and outgoing deltas live.

        Under the run's dump folder, which is on scratch, because every rank of the
        job reads these files -- including ranks on other nodes in a multi-node
        allocation. A node-local /tmp would work for one node and silently break for
        two.
        """
        path = Path(self.federated.job_config.job.dump_folder) / "blob-staging"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _client_for(self, config: dict) -> BlobClient:
        url = str(config.get(proto.BLOB_URL, "") or self._blob_url)
        if not url:
            raise RuntimeError(
                "the central node selected blob transport but sent no blob store URL; "
                "start the server with --blob-url"
            )
        if self._blob is None or url != self._blob_url:
            self._blob = BlobClient(
                url,
                token=self.federated.job_config.darl.token,
                use_proxy=self.federated.job_config.darl.use_proxy,
            )
            self._blob_url = url
        return self._blob

    def _check_transport_ceiling(self) -> None:
        """Only for inline transport, and only once."""
        if self._inline_checked:
            return
        self._inline_checked = True
        if self.codec is None:
            # Should be unreachable: _require_transport rejects the mismatch first.
            raise RuntimeError(
                "inline transport was requested but no parameter ordering was built. "
                "Set flower.transport = 'inline' so the client gathers the model at "
                "startup, or run the server with --transport blob."
            )
        limit = self.federated.job_config.flower.max_message_length
        needed = self.codec.wire_bytes
        logger.info(
            "inline transport: %.2f GiB per message (%s)", needed / 2**30, self.wire_dtype
        )
        if needed >= limit:
            raise RuntimeError(
                f"a full parameter set is {needed / 2**30:.2f} GiB in {self.wire_dtype}, "
                f"at or above the {limit / 2**30:.2f} GiB gRPC message cap. gRPC cannot "
                f"carry more than 2 GiB in one message no matter how "
                f"flower.max_message_length is set. Start the central node with "
                f"--transport blob to move weights out of band instead."
            )
        if needed > 0.8 * limit:
            logger.warning(
                "inline messages are at %.0f%% of the gRPC cap; a slightly larger model "
                "will not fit -- consider --transport blob", 100 * needed / limit,
            )

    # --- Flower NumPyClient interface -------------------------------------

    def get_parameters(self, config: dict) -> list:
        """Only reached under inline transport; blob mode seeds via the init blob."""
        self._broadcast(CMD_FIT, gather_only=True)
        self._check_transport_ceiling()
        return self.codec.encode(gather_full_state(self.trainer.model_parts))

    def set_parameters(self, parameters: list) -> dict[str, torch.Tensor]:
        state = self.codec.decode(parameters)
        scatter_full_state(self.trainer.model_parts, state)
        self._reference = state
        return state

    def fit(self, parameters: list, config: dict) -> tuple:
        transport = str(config.get(proto.TRANSPORT, proto.TRANSPORT_INLINE))
        self._require_transport(transport)
        if transport == proto.TRANSPORT_BLOB:
            return self._fit_blob(config)
        return self._fit_inline(parameters, config)

    def evaluate(self, parameters: list, config: dict) -> tuple:
        transport = str(config.get(proto.TRANSPORT, proto.TRANSPORT_INLINE))
        self._require_transport(transport)
        if transport == proto.TRANSPORT_BLOB:
            reference = self._fetch_global(config)
            self._broadcast(CMD_EVAL, reference=str(reference))
            stream_apply_global(self.trainer.model_parts, reference)
        else:
            self._broadcast(CMD_EVAL, has_parameters=bool(parameters))
            if parameters:
                self.set_parameters(parameters)

        loss, tokens = self.federated.validate()
        if math.isnan(loss):
            logger.warning(
                "validation produced NaN loss; reporting token count 1 to prevent "
                "server-side ZeroDivisionError in Flower aggregate_evaluate"
            )
            return 0.0, 1, {"eval_loss": float("nan"), "perplexity": float("nan")}
        if tokens <= 0:
            # Should not happen with validation enabled and steps > 0. Reported rather
            # than dropped, but flagged: the cross-site aggregate is only a token-weighted
            # mean if the weights are real token counts.
            logger.warning(
                "validation produced no token count; this cluster's held-out loss will "
                "carry weight 1 in the cross-site aggregate instead of its token count"
            )
            tokens = 1
        # `eval_loss` is what the central node aggregates; perplexity is derived there
        # from the pooled loss. Sent per-cluster too, for visibility -- but it must not be
        # what gets averaged. See central/server.py::aggregate_eval_metrics.
        return float(loss), int(tokens), {
            "eval_loss": float(loss),
            "perplexity": float(math.exp(min(20.0, loss))),
        }

    # --- blob transport ----------------------------------------------------

    def _fetch_global(self, config: dict) -> Path:
        """Download the round's global weights, or reuse the copy already on disk."""
        name = str(config.get(proto.GLOBAL_BLOB, ""))
        if not name:
            raise RuntimeError("blob transport round carried no global blob name")
        target = self._staging_dir() / name
        if name == self._cached_global and target.is_file():
            logger.info("reusing %s already staged at %s", name, target)
            return target
        self._client_for(config).get(name, target)
        self._cached_global = name
        return target

    def _fit_blob(self, config: dict) -> tuple:
        base_round = int(config.get(proto.ROUND, 0))
        run_id = str(config.get(proto.RUN_ID, "pww"))
        staging = self._staging_dir()

        if str(config.get(proto.NEED_INIT, "")) == "1":
            # Cold start: this cluster's freshly initialised weights become the global
            # model. Written before any training so the delta below is measured against
            # exactly what the central node adopts.
            init_name = str(config.get(proto.INIT_BLOB, proto.init_blob(run_id)))
            reference = staging / init_name
            self._broadcast(CMD_FIT, reference=str(reference), write_full=True)
            stream_write_full(
                self.trainer.model_parts, reference,
                meta={"cluster": self.cluster_id, "round": base_round},
            )
            self._client_for(config).put(init_name, reference)
            seeded = True
            logger.info("seeded the global model from this cluster's initial weights")
        else:
            reference = self._fetch_global(config)
            self._broadcast(CMD_FIT, reference=str(reference))
            stream_apply_global(self.trainer.model_parts, reference)
            seeded = False

        result = self.federated.run_inner_phase()

        delta_name = proto.delta_blob(run_id, base_round, self.cluster_id)
        delta_path = staging / delta_name
        _, drift = stream_gather_delta(
            self.trainer.model_parts, reference, delta_path,
            base_round=base_round, cluster=self.cluster_id,
            tokens=int(result["tokens"]),
        )

        metrics: dict[str, Any] = {
            proto.LOSS: float(result["loss"]),
            proto.STEPS: int(result["steps"]),
            proto.TOKENS: int(result["tokens"]),
            proto.EXHAUSTED: bool(result["exhausted"]),
            proto.CLUSTER: self.cluster_id,
            proto.BASE_ROUND: base_round,
            "tokens_per_s": float(result["tokens_per_s"]),
            "blocks_committed": int(result["blocks_committed"]),
            **{key: float(value) for key, value in drift.items()},
        }
        if seeded:
            metrics[proto.UPLOADED_INIT] = "1"

        if result["exhausted"]:
            self.done = True

        if result["steps"] == 0:
            # Honest zero. Uploading a delta of all zeros would waste a full transfer
            # to say nothing, and the server weights by tokens anyway.
            logger.warning(
                "round trained 0 steps (exhausted=%s); reporting 0 tokens and not "
                "uploading a delta", result["exhausted"],
            )
            delta_path.unlink(missing_ok=True)
            return [], 0, metrics

        self._client_for(config).put(delta_name, delta_path)
        metrics[proto.DELTA_BLOB] = delta_name
        # The staged delta is the server's now; keeping it would grow the site's
        # scratch by one model per round.
        delta_path.unlink(missing_ok=True)

        self._log_round(result, drift["drift_ratio"], base_round)
        return [], int(result["tokens"]), metrics

    # --- inline transport --------------------------------------------------

    def _fit_inline(self, parameters: list, config: dict) -> tuple:
        self._check_transport_ceiling()
        self._broadcast(CMD_FIT, has_parameters=bool(parameters))
        if parameters:
            self.set_parameters(parameters)

        result = self.federated.run_inner_phase()
        local = gather_full_state(self.trainer.model_parts)

        # Both sides are normalised, not just the gather. `local` comes from
        # gather_full_state, which unwraps already; `_reference` is either that same
        # gather (cold start) or a wire decode (every later round), and on torch 2.9
        # one of them can still arrive as a DTensor. Subtracting a DTensor from a
        # plain tensor raises rather than promoting, so a single stray entry fails the
        # whole outer round -- and the round-1 failure wedges the server, because its
        # no-results path puts float32 on the wire. Log which keys were wrapped so the
        # provenance is recorded rather than inferred.
        stray = [k for k in local if isinstance(self._reference.get(k), DTensor)]
        if stray:
            logger.warning(
                "reference held %d DTensor entries (e.g. %s); unwrapping before the "
                "delta. Expected plain tensors from gather_full_state/codec.decode",
                len(stray), ", ".join(stray[:3]),
            )
        delta = {
            key: as_plain_tensor(local[key]).to(torch.float32)
            - as_plain_tensor(self._reference[key]).to(torch.float32)
            for key in local
        }
        drift = outer_agreement(delta, self._reference)

        metrics: dict[str, Any] = {
            proto.LOSS: float(result["loss"]),
            proto.STEPS: int(result["steps"]),
            proto.TOKENS: int(result["tokens"]),
            proto.EXHAUSTED: bool(result["exhausted"]),
            proto.CLUSTER: self.cluster_id,
            proto.BASE_ROUND: int(config.get(proto.ROUND, 0)),
            "tokens_per_s": float(result["tokens_per_s"]),
            "blocks_committed": int(result["blocks_committed"]),
            **{key: float(value) for key, value in drift.items()},
        }
        if result["exhausted"]:
            self.done = True
        if result["steps"] == 0:
            logger.warning(
                "round trained 0 steps (exhausted=%s); reporting 0 tokens so this "
                "cluster carries no weight in the aggregate", result["exhausted"],
            )
            return self.codec.encode(local), 0, metrics

        self._log_round(result, drift["drift_ratio"], metrics[proto.BASE_ROUND])
        return self.codec.encode(local), int(result["tokens"]), metrics

    def _log_round(self, result: dict, drift: float, base_round: int) -> None:
        logger.info(
            "merge round %d: %d steps, %s tokens, loss %.4f (ppl %.2f), drift %.4f, "
            "%.0f tok/s, %d blocks committed",
            base_round, result["steps"], f"{result['tokens']:,}", result["loss"],
            math.exp(min(20.0, result["loss"])), drift, result["tokens_per_s"],
            result["blocks_committed"],
        )

    # --- rank coordination -------------------------------------------------

    def _broadcast(
        self,
        command: str,
        *,
        has_parameters: bool = False,
        gather_only: bool = False,
        reference: str = "",
        write_full: bool = False,
    ) -> None:
        """Tell the other ranks which collectives to enter, and over which file.

        Every flag here guards a collective. If rank 0 skipped a scatter on an empty
        parameter list, or read a file the workers did not know about, the job would
        hang in a mismatched collective rather than fail. Only this small tuple crosses
        the process group -- never the weights.
        """
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast_object_list(
                [(command, has_parameters, gather_only, reference, write_full)], src=0
            )

    def stop_workers(self) -> None:
        self._broadcast(CMD_STOP)


def broadcast_stop() -> None:
    """Release the workers when rank 0 failed before it had a client to do it.

    `DiLoCoFlowerClient.__init__` and the transport ceiling check can both raise, and by
    then every other rank is already blocked in `run_worker_loop`'s broadcast. Without
    this they wait until Slurm kills the job, which buries the real error under a
    walltime timeout.
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast_object_list([(CMD_STOP, False, False, "", False)], src=0)


def run_worker_loop(federated: FederatedTrainer) -> None:
    """What every rank but 0 does: mirror the leader's collectives, in the same order.

    The empty dict handed to `scatter_full_state` is the documented shape for a
    non-source rank under `broadcast_from_rank0`: rank 0 holds the full state, everyone
    else receives only their shard. In blob mode there is no such asymmetry -- every
    rank reads the same file and shards it locally -- so the workers call exactly the
    same functions rank 0 does, with `is_writer=False` where bytes would be produced.
    """
    while True:
        box: list[Any] = [None]
        dist.broadcast_object_list(box, src=0)
        command, has_parameters, gather_only, reference, write_full = box[0]

        if command == CMD_STOP:
            return

        if command == CMD_FIT:
            if gather_only:
                gather_full_state(federated.trainer.model_parts)
                continue
            if reference:
                if write_full:
                    stream_write_full(
                        federated.trainer.model_parts, reference, is_writer=False
                    )
                else:
                    stream_apply_global(federated.trainer.model_parts, reference)
                federated.run_inner_phase()
                stream_gather_delta(
                    federated.trainer.model_parts, reference, "",
                    base_round=0, cluster="", tokens=0, is_writer=False,
                )
            else:
                if has_parameters:
                    scatter_full_state(federated.trainer.model_parts, {})
                federated.run_inner_phase()
                gather_full_state(federated.trainer.model_parts)
        elif command == CMD_EVAL:
            if reference:
                stream_apply_global(federated.trainer.model_parts, reference)
            elif has_parameters:
                scatter_full_state(federated.trainer.model_parts, {})
            federated.validate()
        else:
            logger.error("unknown command %r from rank 0", command)
            return
