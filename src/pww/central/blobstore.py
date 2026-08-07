"""Blob store on the central node: where multi-gigabyte weight deltas travel.

Why this exists
---------------
The Flower connection carries the *control plane* -- which round it is, who is
participating, what the metrics were -- and that is all it should carry. gRPC caps
a single message at 2**31-1 bytes, so a full parameter set stops fitting somewhere
below 1B parameters (0.6B in float16 is 1.2 GiB; 1.7B is 3.4 GiB and cannot be
sent at all, at any setting). Past that the weights have to move out of band.

This is deliberately a dumb content store, not a smart service: PUT a file, GET it
back, DELETE it when it has been merged. All the aggregation logic lives in
`globalstate.py`. That split is what lets the same code path serve a 0.6B run and a
70B run -- only the byte counts change.

It is also the right shape for the problem rather than a workaround. DiLoCo's whole
premise is that the outer exchange is *rare*: at H=100 inner steps a site uploads
once per ~15 minutes of compute. Optimising that transfer for latency would be
optimising the wrong axis; what matters is that it can carry 14 GB at 7B and 140 GB
at 70B without holding any of it in memory.

Streaming, both directions
--------------------------
Every transfer is chunked through a fixed buffer and staged to a temporary file
that is `os.replace`d into position, so a 140 GB upload costs `CHUNK` bytes of RAM
and a reader can never observe a half-written blob. A server that read
`Content-Length` bytes into memory -- the obvious implementation -- would be killed
by the OOM killer on the first 7B round.

Security
--------
Same posture as the DARL coordinator: a shared token in a header, and no TLS. This
listens on a public VM, so:

* Blob names are validated against a strict allowlist before touching the
  filesystem. Client-supplied names reaching `os.path.join` unchecked is a
  directory traversal, and this service runs where the global model lives.
* Firewall the port to the participating sites' egress addresses.
* The token stops an unauthenticated third party from replacing the global model,
  which is the actual threat -- an attacker who can PUT can poison training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger, setup_logging

logger = get_logger("pww.central.blobstore")

DEFAULT_PORT = 29512

# 8 MiB: large enough that syscall overhead is irrelevant against a 14 GB body,
# small enough that a few concurrent uploads cannot exhaust a 1-2 GB VM.
CHUNK = 8 << 20

# Refuse an upload that would leave less than this free. A full volume on the
# central node takes the DARL coordinator down with it, so the store would rather
# fail one round than the whole run. Configurable because the right reserve for a
# 70B run and for a laptop test are different by four orders of magnitude.
DEFAULT_MIN_FREE = 1 << 30

# Blob names come from clients over the network and are turned into paths. One
# component, no separators, no dots-only names -- anything else is rejected before
# it can escape the root directory.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")


class BlobStoreError(RuntimeError):
    pass


def safe_name(name: str) -> str:
    """Validate a client-supplied blob name, or raise.

    Rejects rather than sanitises. Silently rewriting a name would mean a client
    thinks it uploaded `a/../../b` and got `a_.._.._b`, then cannot find it again --
    a confusing failure in place of a clear one.
    """
    if not _SAFE_NAME.match(name or ""):
        raise BlobStoreError(
            f"invalid blob name {name!r}: one path component matching "
            f"[A-Za-z0-9][A-Za-z0-9._-]* , at most 201 characters"
        )
    if name in (".", "..") or name.startswith("."):
        raise BlobStoreError(f"invalid blob name {name!r}")
    return name


class BlobStore:
    """Files under one directory, with atomic writes and size accounting."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp = self.root / ".incoming"
        self.tmp.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self.bytes_in = 0
        self.bytes_out = 0
        self.puts = 0
        self.gets = 0

    def path(self, name: str) -> Path:
        return self.root / safe_name(name)

    def exists(self, name: str) -> bool:
        return self.path(name).is_file()

    def size(self, name: str) -> int:
        target = self.path(name)
        return target.stat().st_size if target.is_file() else -1

    def open_read(self, name: str):
        target = self.path(name)
        if not target.is_file():
            raise FileNotFoundError(name)
        return target.open("rb")

    def receive(self, name: str, stream, length: int) -> int:
        """Stream `length` bytes from `stream` into blob `name`. Returns bytes written.

        Staged then renamed: a reader either sees the previous blob or the complete
        new one, never a prefix. `os.replace` is atomic within a filesystem, which
        is why the staging directory lives under the same root.
        """
        target = self.path(name)
        staging = self.tmp / f"{name}.{os.getpid()}.{threading.get_ident()}"
        written = 0
        try:
            with staging.open("wb") as out:
                remaining = length
                while remaining > 0:
                    block = stream.read(min(CHUNK, remaining))
                    if not block:
                        raise BlobStoreError(
                            f"client disconnected after {written} of {length} bytes"
                        )
                    out.write(block)
                    written += len(block)
                    remaining -= len(block)
                # The blob is about to become the input to an aggregation that
                # cannot be undone, so make sure it is really on the platter before
                # anyone is told it landed.
                out.flush()
                os.fsync(out.fileno())
            os.replace(staging, target)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
        with self._lock:
            self.bytes_in += written
            self.puts += 1
        return written

    def delete(self, name: str) -> bool:
        target = self.path(name)
        if target.is_file():
            target.unlink()
            return True
        return False

    def note_sent(self, count: int) -> None:
        with self._lock:
            self.bytes_out += count
            self.gets += 1

    def prune(self, keep: set[str], older_than_s: float = 0.0) -> int:
        """Delete blobs not in `keep`. Returns how many went.

        Called after a successful merge. Without it a 7B run leaves 14 GB per site
        per round on disk and fills the VM within an hour.
        """
        removed = 0
        now = time.time()
        for entry in self.root.iterdir():
            if entry.name.startswith(".") or not entry.is_file():
                continue
            if entry.name in keep:
                continue
            if older_than_s and (now - entry.stat().st_mtime) < older_than_s:
                continue
            try:
                entry.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("could not prune %s: %s", entry.name, exc)
        return removed

    def usage(self) -> dict[str, Any]:
        total = 0
        count = 0
        for entry in self.root.iterdir():
            if entry.is_file() and not entry.name.startswith("."):
                total += entry.stat().st_size
                count += 1
        free = shutil.disk_usage(self.root).free
        return {
            "blobs": count,
            "bytes": total,
            "disk_free": free,
            "puts": self.puts,
            "gets": self.gets,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
        }


def make_handler(store: BlobStore, token: str, min_free: int = DEFAULT_MIN_FREE):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "pww-blobstore/1"

        # --- helpers ---------------------------------------------------------

        def _authorised(self) -> bool:
            if not token:
                return True
            import hmac

            return hmac.compare_digest(self.headers.get("X-DARL-Token", ""), token)

        def _blob_name(self) -> str | None:
            path = self.path.split("?", 1)[0].strip("/")
            if not path.startswith("blob/"):
                return None
            return path[len("blob/"):]

        def _send(self, code: int, payload: dict | None = None,
                  close: bool = False) -> None:
            body = json.dumps(payload or {}).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if close:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            if close:
                self.close_connection = True

        def _fail(self, code: int, message: str, close: bool = False) -> None:
            self._send(code, {"error": message}, close=close)

        def log_message(self, fmt: str, *args: Any) -> None:
            # Default logs one line per request to stderr; a 140 GB run makes that
            # noise. Errors are logged explicitly where they happen instead.
            return

        # --- routes ----------------------------------------------------------

        def do_GET(self) -> None:
            route = self.path.split("?", 1)[0].strip("/")
            if route == "health":
                self._send(200, {"ok": True, "usage": store.usage()})
                return
            if not self._authorised():
                self._fail(401, "bad or missing X-DARL-Token")
                return
            if route == "usage":
                self._send(200, store.usage())
                return

            name = self._blob_name()
            if name is None:
                self._fail(404, f"no route {route!r}")
                return
            try:
                size = store.size(name)
            except BlobStoreError as exc:
                self._fail(400, str(exc))
                return
            if size < 0:
                self._fail(404, f"no blob {name!r}")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.end_headers()
            sent = 0
            try:
                with store.open_read(name) as handle:
                    while True:
                        block = handle.read(CHUNK)
                        if not block:
                            break
                        self.wfile.write(block)
                        sent += len(block)
            except (BrokenPipeError, ConnectionResetError):
                # A site killed at walltime mid-download. Not an error here; it
                # will pull again when it is requeued.
                logger.info("download of %s aborted by peer after %d bytes", name, sent)
                return
            store.note_sent(sent)

        def do_HEAD(self) -> None:
            if not self._authorised():
                self._fail(401, "bad or missing X-DARL-Token")
                return
            name = self._blob_name()
            try:
                size = -1 if name is None else store.size(name)
            except BlobStoreError as exc:
                self._fail(400, str(exc))
                return
            if size < 0:
                self._fail(404, "not found")
                return
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.end_headers()

        def do_PUT(self) -> None:
            # close=True on every pre-body rejection: the client is mid-upload, and a
            # kept-alive connection turns a clear 4xx into a broken pipe.
            if not self._authorised():
                self._fail(401, "bad or missing X-DARL-Token", close=True)
                return
            name = self._blob_name()
            if name is None:
                self._fail(404, "PUT only to /blob/<name>", close=True)
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if length < 0:
                # Chunked bodies would need de-chunking; every client here is ours
                # and sends a length, so reject rather than half-support it.
                self._fail(411, "Content-Length required (chunked upload unsupported)",
                           close=True)
                return

            free = shutil.disk_usage(store.root).free
            if length > free - min_free:
                # Refused before a byte is written. `close=True` matters: the client
                # is already streaming the body, and without closing the connection
                # it sees EPIPE on its next write instead of this message. Clients
                # also pre-flight /usage so they normally never get here.
                self._fail(
                    507,
                    f"upload of {length} bytes would leave under {min_free} bytes "
                    f"free ({free} available) -- prune old blobs, grow the volume, "
                    f"or lower --min-free-bytes",
                    close=True,
                )
                return

            try:
                written = store.receive(name, self.rfile, length)
            except BlobStoreError as exc:
                logger.warning("upload of %s failed: %s", name, exc)
                self._fail(400, str(exc))
                return
            except OSError as exc:
                logger.error("upload of %s failed: %s", name, exc)
                self._fail(500, str(exc))
                return
            logger.info("received %s (%.2f GiB)", name, written / 2**30)
            self._send(201, {"name": name, "bytes": written})

        def do_DELETE(self) -> None:
            if not self._authorised():
                self._fail(401, "bad or missing X-DARL-Token")
                return
            name = self._blob_name()
            if name is None:
                self._fail(404, "DELETE only to /blob/<name>")
                return
            try:
                gone = store.delete(name)
            except BlobStoreError as exc:
                self._fail(400, str(exc))
                return
            self._send(200 if gone else 404, {"deleted": gone})

    return Handler


def make_server(store: BlobStore, host: str = "0.0.0.0", port: int = DEFAULT_PORT,
                token: str = "", min_free: int = DEFAULT_MIN_FREE) -> ThreadingHTTPServer:
    handler = make_handler(store, token, min_free)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def content_hash(path: str | os.PathLike[str]) -> str:
    """Short digest of a file, for naming and for integrity checks."""
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PWW central blob store for weight deltas")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--root", required=True,
                   help="directory for blobs -- must be on a volume with room for "
                        "the global model, the momentum buffer and one delta per site")
    p.add_argument("--token", default=os.environ.get("DARL_TOKEN", ""),
                   help="shared secret; clients send it as X-DARL-Token")
    p.add_argument("--min-free-bytes", type=int, default=DEFAULT_MIN_FREE,
                   help="refuse an upload that would leave less than this free")
    args = p.parse_args(argv)

    setup_logging(rank=0)
    store = BlobStore(args.root)
    usage = store.usage()
    logger.info(
        "blob store at %s | %d blob(s), %.2f GiB used, %.2f GiB free",
        store.root, usage["blobs"], usage["bytes"] / 2**30, usage["disk_free"] / 2**30,
    )
    if not args.token:
        logger.warning(
            "no token set -- anyone who can reach this port can replace the global "
            "model. Set --token (or DARL_TOKEN) and firewall the port."
        )
    httpd = make_server(store, args.host, args.port, args.token, args.min_free_bytes)
    logger.info("listening on %s:%d", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
