"""Out-of-band weight exchange: the cluster side.

The Flower connection carries the control plane; the weights travel beside it as a
file. This module is the client half -- streaming HTTP to the central blob store,
and a per-tensor gather/scatter that never materialises a whole model.

Deliberately free of any torchtitan dependency -- it needs only torch and
`pww.tensorio` -- so the transport can be tested without a model, and the central
node can reuse `BlobClient` without pulling in a trainer.

Why per-tensor and not `get_model_state_dict(full_state_dict=True)`
------------------------------------------------------------------
That call all-gathers the *entire* model onto **every** rank. In bfloat16 that is
14 GB per rank at 7B, so 112 GB of host RAM across a LUMI node -- survivable -- and
140 GB per rank at 70B, so 1.1 TB across the node, which is not.

`stream_gather_delta` instead walks the parameters one at a time, calling
`DTensor.full_tensor()` on each, so peak host memory is a small multiple of the
*largest single tensor* (the embedding: ~2 GiB at 7B, ~4 GiB at 70B). Symmetric with
how `central/globalstate.py` merges, and for the same reason.

Every rank calls `full_tensor()` on every parameter in the same order, because it is
a collective. Only rank 0 writes bytes.

Parameters, not buffers
-----------------------
Only `named_parameters()` is exchanged. Qwen3's buffers are RoPE frequency tables,
recomputed deterministically from the config at init -- averaging them across sites
would be a no-op, and one of them is complex-dtyped, which the file format does not
carry. A model with *learned* buffer state (BatchNorm running statistics, say) would
need them included, and that is exactly why the ResNet path pins outer momentum to
zero; Qwen3 is RMSNorm with learned weights and has no such state.

Do not "fix" this by widening it to `named_buffers()`. The inline path's equivalent
did include buffers on the *apply* side, and because the wire never carried them it
filled the RoPE table with uninitialised memory on every round -- silently zeroing
RoPE when the allocator handed back fresh pages, and producing a first-microbatch nan
when it handed back dirty ones. See `titan/params.py::keys_to_load`.

Pipeline parallelism is refused rather than half-supported: with PP each rank holds a
different subset of layers, so a naive walk over `model_parts` has ranks entering
different collectives and the job deadlocks instead of failing.
"""

from __future__ import annotations

import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn

from .logging_utils import get_logger
from .tensorio import TensorFile, TensorWriter

logger = get_logger("pww.delta")

CHUNK = 8 << 20


class BlobTransportError(RuntimeError):
    pass


class BlobClient:
    """Streaming HTTP client for the central blob store.

    Uploads and downloads are chunked through a fixed buffer, so a 14 GB transfer
    costs `CHUNK` bytes of RAM. Retries with jittered backoff, like the DARL client
    and for the same reason: a WAN hop from an HPC compute node to a cloud VM fails
    transiently often enough that a run without retries will not finish.
    """

    def __init__(
        self,
        url: str,
        token: str = "",
        *,
        timeout: float = 1800.0,
        retries: int = 5,
        backoff: float = 2.0,
        use_proxy: bool = False,
    ):
        if not url:
            raise ValueError("blob store URL is required for out-of-band transport")
        self.url = url.rstrip("/")
        self.token = token or os.environ.get("DARL_TOKEN", "")
        # Generous: a 14 GB upload over a shared WAN link is minutes, not seconds,
        # and a timeout mid-transfer costs a whole DiLoCo round of GPU time.
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.backoff = float(backoff)
        handlers = [] if use_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)
        self.bytes_up = 0
        self.bytes_down = 0

    def _headers(self) -> dict[str, str]:
        return {"X-DARL-Token": self.token} if self.token else {}

    def _attempt(self, description: str, action) -> Any:
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return action()
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:200]
                if 400 <= exc.code < 500 and exc.code != 429:
                    # A 4xx is our fault (bad name, bad token, disk full at 507) and
                    # retrying it just wastes the round.
                    raise BlobTransportError(
                        f"{description} rejected ({exc.code}): {body}"
                    ) from None
                last = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = exc
            delay = min(120.0, self.backoff**attempt) * (0.5 + random.random())
            logger.warning(
                "%s failed (%s), retry %d/%d in %.1fs",
                description, last, attempt + 1, self.retries, delay,
            )
            time.sleep(delay)
        raise BlobTransportError(f"{description} failed after {self.retries} attempts: {last}")

    def put(self, name: str, path: str | os.PathLike[str]) -> int:
        source = Path(path)
        size = source.stat().st_size
        self._preflight(name, size)

        def action() -> int:
            with source.open("rb") as body:
                request = urllib.request.Request(
                    f"{self.url}/blob/{name}", data=body, method="PUT",
                    headers={**self._headers(), "Content-Length": str(size),
                             "Content-Type": "application/octet-stream"},
                )
                with self._opener.open(request, timeout=self.timeout) as response:
                    response.read()
            return size

        started = time.monotonic()
        written = self._attempt(f"upload of {name}", action)
        elapsed = max(1e-6, time.monotonic() - started)
        self.bytes_up += written
        logger.info(
            "uploaded %s: %.2f GiB in %.0fs (%.0f MiB/s)",
            name, written / 2**30, elapsed, written / elapsed / 2**20,
        )
        return written

    def get(self, name: str, path: str | os.PathLike[str]) -> int:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_suffix(target.suffix + ".part")

        def action() -> int:
            request = urllib.request.Request(
                f"{self.url}/blob/{name}", headers=self._headers(), method="GET"
            )
            received = 0
            with self._opener.open(request, timeout=self.timeout) as response:
                with staging.open("wb") as out:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        out.write(block)
                        received += len(block)
            # Renamed only on a complete transfer, so a killed job cannot leave a
            # truncated global model that the next attempt happily loads.
            os.replace(staging, target)
            return received

        started = time.monotonic()
        try:
            received = self._attempt(f"download of {name}", action)
        finally:
            staging.unlink(missing_ok=True)
        elapsed = max(1e-6, time.monotonic() - started)
        self.bytes_down += received
        logger.info(
            "downloaded %s: %.2f GiB in %.0fs (%.0f MiB/s)",
            name, received / 2**30, elapsed, received / elapsed / 2**20,
        )
        return received

    def _preflight(self, name: str, size: int) -> None:
        """Check there is room before streaming gigabytes at the store.

        Without this, a store that refuses the upload does so *after* the client has
        started sending the body, and the client sees a broken pipe rather than the
        reason -- which for a once-per-round 16 GiB transfer is an expensive way to
        learn the disk is full. One small GET buys a usable error message.

        Deliberately advisory: a failure to pre-flight does not stop the upload, since
        the store enforces the same limit itself and a transient failure here should
        not cost a round.
        """
        try:
            usage = self.health().get("usage", {})
        except Exception as exc:  # noqa: BLE001
            logger.debug("could not pre-flight %s (%s); uploading anyway", name, exc)
            return
        free = int(usage.get("disk_free", 0) or 0)
        if free and size > free:
            raise BlobTransportError(
                f"cannot upload {name}: it is {size / 2**30:.2f} GiB and the blob "
                f"store has {free / 2**30:.2f} GiB free. The central node needs room "
                f"for the global model, the momentum buffer and one delta per site -- "
                f"see the disk budget logged at startup."
            )

    def exists(self, name: str) -> bool:
        request = urllib.request.Request(
            f"{self.url}/blob/{name}", headers=self._headers(), method="HEAD"
        )
        try:
            with self._opener.open(request, timeout=60.0):
                return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return False

    def health(self) -> dict[str, Any]:
        import json

        request = urllib.request.Request(f"{self.url}/health", method="GET")
        with self._opener.open(request, timeout=30.0) as response:
            return json.loads(response.read() or b"{}")


# --- per-tensor gather / scatter -------------------------------------------


def _named_parameters(model_parts: list[nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    """Every learned tensor, in an order identical on all ranks.

    Sorted, not insertion-ordered: `full_tensor()` is a collective, so a rank that
    walked the parameters in a different order would deadlock against the others. A
    sort makes the order a property of the names rather than of module construction.
    """
    if len(model_parts) != 1:
        raise BlobTransportError(
            f"out-of-band transport does not support pipeline parallelism "
            f"({len(model_parts)} model parts). With PP each rank holds different "
            f"layers, so a shared walk over parameters would have ranks entering "
            f"different collectives and deadlocking. Use "
            f"parallelism.pipeline_parallel_degree = 1."
        )
    named = dict(model_parts[0].named_parameters())
    for key in sorted(named):
        yield key, named[key]


def _full(tensor: torch.Tensor) -> torch.Tensor:
    """The unsharded value of one parameter. Collective when sharded."""
    full_tensor = getattr(tensor, "full_tensor", None)
    if full_tensor is None:
        return tensor.detach()
    return full_tensor().detach()


def _write_shard_into(param: torch.Tensor, full_value: torch.Tensor) -> None:
    """Copy the slice of `full_value` this rank owns into `param`, in place."""
    if not hasattr(param, "device_mesh"):
        param.detach().copy_(full_value.to(param.dtype))
        return
    from torch.distributed.tensor import distribute_tensor

    sharded = distribute_tensor(
        full_value.to(param.dtype), param.device_mesh, param.placements
    )
    param.detach().to_local().copy_(sharded.to_local())


def stream_apply_global(
    model_parts: list[nn.Module], path: str | os.PathLike[str]
) -> int:
    """Load a global model from a local file into the sharded model, tensor by tensor.

    Every rank reads the same local file -- no broadcast. Cheaper than one: the file
    is already on this node's disk after the download, and reading a tensor from page
    cache beats an all-ranks broadcast of the whole model.
    """
    applied = 0
    with TensorFile(path) as source:
        for key, param in _named_parameters(model_parts):
            if key not in source:
                raise BlobTransportError(
                    f"global model at {path} has no tensor {key!r} -- it was built "
                    f"from a different model.flavor or tokenizer"
                )
            _write_shard_into(param, source.get(key, torch.float32))
            applied += 1
    return applied


def stream_write_full(
    model_parts: list[nn.Module],
    out_path: str | os.PathLike[str],
    *,
    meta: dict[str, Any] | None = None,
    dtype: torch.dtype = torch.float32,
    is_writer: bool = True,
) -> Path | None:
    """Write the whole model, tensor by tensor. Used to seed a cold-start run.

    float32 by default, unlike the delta: this becomes the durable global model that
    every later round is built on, so it is the one artefact worth storing at higher
    precision than the compute dtype.
    """
    writer = TensorWriter(out_path, meta={"kind": "full", **(meta or {})}) if is_writer else None
    try:
        for key, param in _named_parameters(model_parts):
            full_value = _full(param)
            if writer is not None:
                writer.add(key, full_value.to(dtype))
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    return writer.close() if writer is not None else None


def stream_gather_delta(
    model_parts: list[nn.Module],
    reference_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    base_round: int,
    cluster: str,
    tokens: int,
    dtype: torch.dtype = torch.bfloat16,
    is_writer: bool = True,
) -> tuple[Path | None, dict[str, float]]:
    """Write `local - reference` per tensor. Returns (path on the writer, drift stats).

    `base_round` is stamped into the file and the central node refuses any delta whose
    base round is not current -- which is what stops a cluster killed at walltime and
    requeued hours later from contributing an update computed against a global model
    that has since moved on.

    bfloat16 by default: the delta is a *correction* to weights that are themselves
    trained in bfloat16, so storing it at higher precision buys nothing and doubles the
    WAN transfer.

    The drift statistics come out of this pass rather than a second one. `drift_ratio`
    -- the local update's norm over the weights' own norm -- is the quantity DiLoCo's H
    should be tuned against: once it approaches 1, replicas have diverged far enough
    that averaging them destroys progress instead of combining it.
    """
    writer: TensorWriter | None = None
    if is_writer:
        writer = TensorWriter(
            out_path,
            meta={
                "kind": "delta",
                "base_round": int(base_round),
                "cluster": cluster,
                "tokens": int(tokens),
                "dtype": str(dtype).replace("torch.", ""),
            },
        )
    delta_sq = 0.0
    base_sq = 0.0
    try:
        with TensorFile(reference_path) as reference:
            for key, param in _named_parameters(model_parts):
                # Collective on every rank, in the same order, regardless of who
                # writes -- skipping it on non-writers would deadlock.
                local_full = _full(param)
                if writer is None:
                    continue
                base = reference.get(key, torch.float32)
                difference = local_full.to(torch.float32) - base
                delta_sq += float(difference.pow(2).sum())
                base_sq += float(base.pow(2).sum())
                writer.add(key, difference.to(dtype))
    except BaseException:
        if writer is not None:
            writer.abort()
        raise

    delta_norm = delta_sq**0.5
    base_norm = base_sq**0.5
    stats = {
        "delta_norm": delta_norm,
        "param_norm": base_norm,
        "drift_ratio": delta_norm / base_norm if base_norm > 0 else 0.0,
    }
    if writer is None:
        return None, stats
    return writer.close(), stats


def blob_names(run_id: str, round_index: int, cluster: str = "") -> dict[str, str]:
    """Blob names for one round.

    Flat and predictable rather than content-addressed, so an operator can see at a
    glance what a round produced, and so a re-upload after a retry overwrites rather
    than accumulating. The blob store validates these against a strict allowlist, so
    they must stay `[A-Za-z0-9._-]`.
    """
    safe_run = "".join(c if c.isalnum() or c in "._-" else "-" for c in run_id) or "run"
    safe_cluster = "".join(c if c.isalnum() or c in "._-" else "-" for c in cluster)
    names = {"global": f"{safe_run}-global-r{round_index}.pww"}
    if safe_cluster:
        names["delta"] = f"{safe_run}-delta-r{round_index}-{safe_cluster}.pww"
    return names
