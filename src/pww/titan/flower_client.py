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
        control_group: Any | None = None,
    ) -> None:
        self.federated = federated
        self.trainer = federated.trainer
        # The gloo group the command broadcasts ride on. See _broadcast for why this
        # must not default to the NCCL process group.
        self._control_group = control_group
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
        if not math.isfinite(loss):
            # Report the real value, not 0.0.
            #
            # This used to return 0.0 as the loss while putting nan in the metrics, on the
            # grounds of avoiding a server-side ZeroDivisionError. Two things wrong with
            # that. The division in Flower's aggregate_evaluate is by sum(num_examples),
            # which is zero only when *every* client reports zero -- and that case is
            # guarded on the server -- so a fake loss was never needed to prevent it. And
            # this strategy does not override aggregate_evaluate, so Flower averages the
            # returned loss: 0.0 from a diverged cluster is reported as a *perfect* score.
            # nan is loud; 0.0 looks like success, which is the one thing a broken run must
            # not look like.
            #
            # isfinite rather than isnan, because inf is not nan: an infinite loss passed
            # the old check untouched and reached the pooled figure as a real number.
            logger.error(
                "validation produced a non-finite loss (%s); reporting it as-is. The "
                "central node excludes non-finite held-out losses from the pooled "
                "perplexity, so this will not corrupt the reported metric -- but this "
                "cluster is not producing a usable model and its training loss is where "
                "to look.", loss,
            )
            return float(loss), max(1, int(tokens)), {
                proto.CLUSTER: self.cluster_id,
                "eval_loss": float(loss),
                "perplexity": float("nan"),
            }
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
        # The cluster id travels with the eval metrics too, not just the fit ones. Without
        # it the server's cross-cluster spread warning can name neither side -- it printed
        # "?=9.32 vs ?=8.05", which is the one thing that warning exists to tell you.
        return float(loss), int(tokens), {
            proto.CLUSTER: self.cluster_id,
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
            self._broadcast(CMD_FIT, reference=str(reference), write_full=True,
                            global_step=self._global_step_for(config))
            stream_write_full(
                self.trainer.model_parts, reference,
                meta={"cluster": self.cluster_id, "round": base_round},
            )
            self._client_for(config).put(init_name, reference)
            seeded = True
            logger.info("seeded the global model from this cluster's initial weights")
        else:
            reference = self._fetch_global(config)
            self._broadcast(CMD_FIT, reference=str(reference), global_step=self._global_step_for(config))
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

        metrics = self._round_metrics(result, drift, base_round)
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
        self._broadcast(CMD_FIT, has_parameters=bool(parameters),
                        global_step=self._global_step_for(config))
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

        metrics = self._round_metrics(result, drift, int(config.get(proto.ROUND, 0)))
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

    def _round_metrics(
        self, result: dict[str, Any], drift: dict[str, float], base_round: int
    ) -> dict[str, Any]:
        """What this cluster reports for one round, built in exactly one place.

        The two transports used to assemble this dict independently, character for
        character, which is how the blob path ended up without a metric the inline
        path had. There is nothing transport-specific in it.

        `seq_len` and `dp_degree` are here because the central node cannot derive
        them and was guessing. It divided token counts by a literal 2048 to report a
        batch size, and mapped a cluster id to a rank count with
        `8 if "lumi" in cid else 4 if "snellius" in cid else 1` -- both correct only
        for the two sites this repo happens to ship configs for, at one seq_len, with
        no --replica suffix. A cluster knows its own geometry; it should say so.
        """
        return {
            proto.LOSS: float(result["loss"]),
            proto.STEPS: int(result["steps"]),
            proto.TOKENS: int(result["tokens"]),
            proto.EXHAUSTED: bool(result["exhausted"]),
            proto.CLUSTER: self.cluster_id,
            proto.BASE_ROUND: int(base_round),
            proto.SEQ_LEN: int(self.federated.job_config.training.seq_len),
            proto.DP_DEGREE: int(self.federated.dp_degree()),
            "tokens_per_s": float(result["tokens_per_s"]),
            "blocks_committed": int(result["blocks_committed"]),
            "seconds": float(result["seconds"]),
            # Hardware metrics, when the trainer could read them. Forwarded per cluster
            # rather than aggregated: MFU against an MI250X GCD and against an H100 are
            # not the same quantity, so a mean of the two would describe neither.
            **{key: float(result[key]) for key in
               ("mfu_pct", "tflops_per_rank", "peak_memory_gib", "peak_memory_pct", "lr", "grad_norm", "power_watts")
               if key in result},
            **{key: float(value) for key, value in drift.items()},
        }

    def _log_round(self, result: dict, drift: float, base_round: int) -> None:
        logger.info(
            "merge round %d: %d steps, %s tokens, loss %.4f (ppl %.2f), drift %.4f, "
            "%.0f tok/s, %d blocks committed",
            base_round, result["steps"], f"{result['tokens']:,}", result["loss"],
            math.exp(min(20.0, result["loss"])), drift, result["tokens_per_s"],
            result["blocks_committed"],
        )

    # --- rank coordination -------------------------------------------------

    def _global_step_for(self, config: dict) -> int:
        """Where the run is, in optimiser steps, and align this rank to it.

        The server broadcasts ``pww_global_step`` -- the token-weighted average
        of steps across all contributing clusters.  With per-site H (different
        ``darl.inner_steps`` on each cluster), each client can no longer compute
        ``global_step = round * H`` locally because H varies across sites.

        Falls back to ``merge_round * H`` when the server does not send the key,
        which keeps old servers compatible until they are restarted with the new
        strategy code.

        Rank 0 aligns here; the other ranks get the same number through the CMD_FIT
        broadcast and align in ``run_worker_loop``.
        """
        if proto.GLOBAL_STEP in config:
            global_step = int(config[proto.GLOBAL_STEP])
        else:
            global_step = int(config.get(proto.ROUND, 0)) * self.federated.inner_steps
        self.federated.align_to_global_step(global_step)
        return global_step

    def _broadcast(
        self,
        command: str,
        *,
        has_parameters: bool = False,
        gather_only: bool = False,
        reference: str = "",
        write_full: bool = False,
        global_step: int = -1,
    ) -> None:
        """Tell the other ranks which collectives to enter, and over which file.

        Every flag here guards a collective. If rank 0 skipped a scatter on an empty
        parameter list, or read a file the workers did not know about, the job would
        hang in a mismatched collective rather than fail. Only this small tuple crosses
        the process group -- never the weights.

        Over the GLOO control group, never NCCL, and this is load-bearing. The workers
        enter this broadcast the moment their phase ends and then wait for however long
        the rest of the federation takes -- the fast site's wait at the round barrier is
        the slow site's remaining phase plus the merge, which is unbounded by design.
        A NCCL collective under the CUDA watchdog cannot wait like that: the default
        process group's timeout is `comm.init_timeout_seconds` (300s), so a fast site
        idling more than five minutes at the barrier was SIGABRTed by its own watchdog.
        Observed on Snellius as `Watchdog caught collective operation timeout: BROADCAST,
        NumelIn=1 ... 300027ms` in run_worker_loop -- rounds 2..25 each survived by
        seconds, round 26 lost the race, and the whole job died mid-run. Commit 8d9a8e1
        raised the mesh groups' timeouts to 1800s for exactly this symptom, but
        broadcast_object_list without a group rides the DEFAULT group, which kept 300s.

        gloo has no watchdog, carries this CPU-side tuple without touching a GPU, and
        the group's own generous timeout still bounds a genuinely dead leader.
        """
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast_object_list(
                [(command, has_parameters, gather_only, reference, write_full, global_step)],
                src=0, group=self._control_group,
            )

    def stop_workers(self) -> None:
        self._broadcast(CMD_STOP)


def broadcast_stop(control_group: Any | None = None) -> None:
    """Release the workers when rank 0 failed before it had a client to do it.

    `DiLoCoFlowerClient.__init__` and the transport ceiling check can both raise, and by
    then every other rank is already blocked in `run_worker_loop`'s broadcast. Without
    this they wait until Slurm kills the job, which buries the real error under a
    walltime timeout.

    The tuple has to be the same arity `run_worker_loop` unpacks. It was one element
    short of the `global_step` field, so the workers raised `ValueError: not enough
    values to unpack` the moment this fired -- on the one path whose entire job is to
    let them exit cleanly. A wedged rank 0 then produced a wedged job anyway, with the
    unpack error burying the real cause.

    `control_group` must be the same group the workers are blocked on -- see
    `DiLoCoFlowerClient._broadcast` for why that is the gloo group, not the default.
    """
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.broadcast_object_list(
            [(CMD_STOP, False, False, "", False, -1)], src=0, group=control_group
        )


def run_worker_loop(
    federated: FederatedTrainer, control_group: Any | None = None
) -> None:
    """What every rank but 0 does: mirror the leader's collectives, in the same order.

    The empty dict handed to `scatter_full_state` is the documented shape for a
    non-source rank under `broadcast_from_rank0`: rank 0 holds the full state, everyone
    else receives only their shard. In blob mode there is no such asymmetry -- every
    rank reads the same file and shards it locally -- so the workers call exactly the
    same functions rank 0 does, with `is_writer=False` where bytes would be produced.
    """
    while True:
        box: list[Any] = [None]
        # On the gloo control group: this wait spans the whole round barrier -- the slow
        # site's remaining phase plus the merge -- and under NCCL the default group's
        # 300s watchdog SIGABRTed the fast site mid-run. See _broadcast.
        dist.broadcast_object_list(box, src=0, group=control_group)
        command, has_parameters, gather_only, reference, write_full, global_step = box[0]

        # Every rank owns an LR scheduler, so every rank has to be moved -- aligning only
        # rank 0 would leave the shards optimising at different learning rates within a
        # single cluster, which is worse than the cross-site skew this fixes.
        if global_step >= 0:
            federated.align_to_global_step(global_step)

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
