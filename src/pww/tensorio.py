"""A tensor file you can read and write one tensor at a time.

Used by both sides of the out-of-band weight exchange: clusters write parameter
deltas with `TensorWriter`, the central node merges them with `TensorFile` without
ever holding a whole model in memory.

Why not safetensors
-------------------
The format below is deliberately the same *shape* as safetensors -- length-prefixed
JSON header, then packed tensor data -- because that layout is exactly right for
this job. What safetensors' Python API lacks is an **incremental writer**:
`save_file` takes a complete dict, so producing a 140 GB file means holding 140 GB
in RAM. Reading is lazy, writing is not.

Since the writing side is the constraint (a cluster gathers weights it already has,
but the central node has to *build* a new global model), a format with a streaming
writer is the requirement, and it costs ~150 lines.

Why not torch.save / DCP
------------------------
`torch.save` is all-or-nothing in both directions. DCP does support per-tensor
access, but it writes a directory of files and expects a distributed context; this
has to be a single blob that one HTTP PUT can carry and one `os.replace` can make
atomic.

Layout
------
    [8 bytes  little-endian uint64  header length N]
    [N bytes  UTF-8 JSON header]
    [packed tensor data, in header order, each aligned to 64 bytes]

Every tensor is stored as raw bytes with its dtype recorded, so bfloat16
round-trips exactly. That matters because numpy has no bfloat16: anything that goes
through a numpy dtype (as the inline Flower path must) has to detour via float32 or
float16, and this path deliberately does not.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Iterator

import torch

MAGIC = "pww-tensors-v1"
ALIGN = 64
_HEADER_STRUCT = struct.Struct("<Q")

# Dtypes are recorded by name rather than by torch object so a file written by one
# torch version is readable by another.
DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "uint8": torch.uint8,
    "bool": torch.bool,
}
DTYPE_NAMES = {value: key for key, value in DTYPES.items()}


def dtype_name(dtype: torch.dtype) -> str:
    try:
        return DTYPE_NAMES[dtype]
    except KeyError:
        raise ValueError(
            f"dtype {dtype} is not supported by {MAGIC}; supported: {sorted(DTYPES)}"
        ) from None


def _align(offset: int) -> int:
    remainder = offset % ALIGN
    return offset if remainder == 0 else offset + (ALIGN - remainder)


class TensorWriter:
    """Append tensors one at a time; the header is written on close.

    The header has to precede the data but its contents are only known once every
    tensor has been written, so the data is staged to a temporary file and the two
    are joined on `close`. That copy is sequential and disk-bound -- the alternative,
    reserving header space up front, means guessing a bound on the key names and
    failing badly when the guess is wrong.

    Use as a context manager so a failed write leaves nothing behind.
    """

    def __init__(self, path: str | os.PathLike[str], meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.meta = dict(meta or {})
        self._data_path = self.path.with_suffix(self.path.suffix + ".data")
        self._handle = self._data_path.open("wb")
        self._entries: dict[str, dict[str, Any]] = {}
        self._offset = 0
        self._closed = False

    def add(self, key: str, tensor: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        if key in self._entries:
            raise ValueError(f"duplicate key {key!r}")

        padded = _align(self._offset)
        if padded > self._offset:
            self._handle.write(b"\0" * (padded - self._offset))
            self._offset = padded

        # .contiguous() first: a non-contiguous view's raw bytes are not the
        # logical tensor, so viewing as uint8 without it silently writes garbage.
        flat = tensor.detach().to("cpu").contiguous()
        raw = flat.view(torch.uint8).reshape(-1).numpy().data
        nbytes = flat.numel() * flat.element_size()
        self._handle.write(raw)
        self._entries[key] = {
            "dtype": dtype_name(flat.dtype),
            "shape": list(flat.shape),
            "offsets": [self._offset, self._offset + nbytes],
        }
        self._offset += nbytes

    def close(self) -> Path:
        if self._closed:
            return self.path
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()

        header = json.dumps(
            {"format": MAGIC, "meta": self.meta, "tensors": self._entries},
            separators=(",", ":"),
        ).encode()
        staging = self.path.with_suffix(self.path.suffix + ".tmp")
        with staging.open("wb") as out:
            out.write(_HEADER_STRUCT.pack(len(header)))
            out.write(header)
            with self._data_path.open("rb") as data:
                while True:
                    block = data.read(8 << 20)
                    if not block:
                        break
                    out.write(block)
            out.flush()
            os.fsync(out.fileno())
        os.replace(staging, self.path)
        self._data_path.unlink(missing_ok=True)
        self._closed = True
        return self.path

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self._handle.close()
        finally:
            self._data_path.unlink(missing_ok=True)
            self.path.with_suffix(self.path.suffix + ".tmp").unlink(missing_ok=True)
            self._closed = True

    def __enter__(self) -> "TensorWriter":
        return self

    def __exit__(self, exc_type, *_: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class TensorFile:
    """Random access to one tensor at a time, by key.

    Holds an open handle and seeks; nothing is read until `get` is called, so
    opening a 140 GB file is a couple of syscalls.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._handle = self.path.open("rb")
        raw_length = self._handle.read(_HEADER_STRUCT.size)
        if len(raw_length) < _HEADER_STRUCT.size:
            raise ValueError(f"{path} is too short to be a {MAGIC} file")
        (length,) = _HEADER_STRUCT.unpack(raw_length)
        if length <= 0 or length > (64 << 20):
            raise ValueError(f"{path}: implausible header length {length}")
        header = json.loads(self._handle.read(length))
        if header.get("format") != MAGIC:
            raise ValueError(
                f"{path}: format {header.get('format')!r}, expected {MAGIC!r}"
            )
        self.meta: dict[str, Any] = header.get("meta", {})
        self._entries: dict[str, dict[str, Any]] = header["tensors"]
        self._data_start = _HEADER_STRUCT.size + length

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def spec(self, key: str) -> dict[str, Any]:
        return self._entries[key]

    def numel(self) -> int:
        total = 0
        for entry in self._entries.values():
            count = 1
            for dim in entry["shape"]:
                count *= dim
            total += count
        return total

    def largest_bytes(self) -> int:
        """Biggest single tensor. Sets the memory floor for a streaming merge."""
        return max(
            (entry["offsets"][1] - entry["offsets"][0] for entry in self._entries.values()),
            default=0,
        )

    def get(self, key: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"{self.path}: no tensor {key!r}")
        start, end = entry["offsets"]
        self._handle.seek(self._data_start + start)
        raw = self._handle.read(end - start)
        if len(raw) != end - start:
            raise ValueError(
                f"{self.path}: truncated at {key!r} -- wanted {end - start} bytes, "
                f"got {len(raw)}"
            )
        source = DTYPES[entry["dtype"]]
        # bytearray, not bytes: frombuffer on an immutable buffer yields a tensor
        # that warns on any in-place op, and the merge path does plenty.
        tensor = torch.frombuffer(bytearray(raw), dtype=source).reshape(entry["shape"])
        return tensor if dtype is None or dtype == source else tensor.to(dtype)

    def items(self, dtype: torch.dtype | None = None) -> Iterator[tuple[str, torch.Tensor]]:
        for key in self._entries:
            yield key, self.get(key, dtype)

    def state_dict(self, dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
        """Everything at once. Only for models small enough to fit -- the point of
        this module is that the merge path never needs to call it."""
        return dict(self.items(dtype))

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "TensorFile":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def write_state_dict(
    path: str | os.PathLike[str],
    state: dict[str, torch.Tensor],
    *,
    meta: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
) -> Path:
    """Convenience for the side that already holds a full state dict in memory."""
    with TensorWriter(path, meta=meta) as writer:
        for key in sorted(state):
            tensor = state[key]
            writer.add(key, tensor if dtype is None else tensor.to(dtype))
    return Path(path)
