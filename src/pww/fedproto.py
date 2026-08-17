"""The contract between the central node and a cluster, in one place.

Flower carries two free-form string-keyed dictionaries per round -- `config` on the
way out and `metrics` on the way back -- and this module names every key that travels
in them. Both sides import from here so a rename cannot desynchronise them: a typo in
a key name would otherwise show up as a cluster silently training against stale
weights, which is exactly the class of failure the generation check exists to catch.

Two transports
--------------
``inline``  parameters travel inside the Flower gRPC message as numpy arrays. Simple,
            one moving part, and hard-capped by gRPC's 2 GiB per-message limit --
            about 1B parameters in float16, and no setting raises it.
``blob``    the message carries only a blob *name*; the bytes move over plain HTTP to
            `pww.central.blobstore`. No size ceiling, at the cost of one more daemon.

The round, and why there are two of them
----------------------------------------
`PWW_ROUND` is the central node's **merge counter** -- how many times the global model
has actually changed -- and it is not Flower's `server_round`. Flower's counter
advances even for a round in which every cluster was killed at walltime and
contributed nothing. Deltas are stamped and validated against the merge counter,
because that is the thing a delta is actually derived from.
"""

from __future__ import annotations

import re

# --- config: central node -> cluster ---------------------------------------

TRANSPORT = "pww_transport"
"""'inline' or 'blob'."""

ROUND = "pww_round"
"""The merge counter the global model is currently at. A cluster stamps this into its
delta as `base_round`, and the merge refuses anything that does not match."""

RUN_ID = "pww_run_id"
"""Names the blobs for this run, so two runs sharing a blob store cannot collide."""

BLOB_URL = "pww_blob_url"
"""Base URL of the blob store, e.g. http://145.38.206.143:29512."""

GLOBAL_BLOB = "pww_global_blob"
"""Blob holding the current global weights. Absent on a cold start."""

NEED_INIT = "pww_need_init"
"""'1' when the central node has no global model yet and this cluster should upload
its freshly initialised weights to `INIT_BLOB` to become the starting point."""

INIT_BLOB = "pww_init_blob"
"""Where to put those initial weights."""

GLOBAL_STEP = "pww_global_step"
"""Server-authoritative optimiser step counter.

With per-site H (different ``darl.inner_steps`` on each cluster), a client can no
longer compute ``global_step = round * H`` locally because H varies across sites.
The server tracks the largest number of steps any contributing cluster took, and broadcasts this so every site's LR schedule stays aligned."""

INNER_STEPS = "pww_inner_steps"
"""How many inner steps THIS round should take, decided by the server per round.

Absent (or <= 0) means "use your own darl.inner_steps", which is every run before
the QSR schedule existed. When present, the client sets it on its FederatedTrainer
before the phase -- the DARL lease granularity stays whatever the config sized it
to, which is fine because the dataloader is one continuous iterator across leases
and commits are watermark-based, not lease-based (see darl_dataloader's module
docstring and LeaseSession.commit_all)."""

LOCAL_EVAL = "pww_local_eval"
"""'1' asks the cluster to validate its OWN phase endpoint before returning it.

This is the theta_i term of the Jensen gauge J = L(theta_bar) - mean_i L(theta_i):
the ordinary evaluate round scores the *merged* model, so without this the server
can see the harvest's minuend but never its subtrahend. Costs one extra validation
pass per round per site."""

# --- metrics: cluster -> central node --------------------------------------

DELTA_BLOB = "pww_delta_blob"
"""Blob the cluster wrote its parameter delta to."""

BASE_ROUND = "pww_base_round"
"""Merge counter the delta was computed against. Checked, not trusted."""

CLUSTER = "pww_cluster"
"""Which cluster produced it -- used for the membership record and for log lines."""

UPLOADED_INIT = "pww_uploaded_init"
"""'1' if this cluster wrote INIT_BLOB during a cold start."""

TOKENS = "tokens"
EXHAUSTED = "exhausted"
STEPS = "steps"
LOSS = "loss"

LOCAL_EVAL_LOSS = "local_eval_loss"
"""Held-out loss of this cluster's own phase endpoint (theta_i), measured just
before the endpoint is returned. Only sent when the round's config asked for it
with LOCAL_EVAL. Non-finite values are sent as-is; the server filters them."""

LOCAL_EVAL_TOKENS = "local_eval_tokens"
"""Tokens that loss was measured over -- the weight it carries in the cross-site
mean, for the same reason evaluate()'s num_examples is a real token count."""

SEQ_LEN = "seq_len"
"""Sequence length this cluster trained at, so the central node can convert a token
count into a batch of *sequences* without hardcoding one.

It divided by a literal 2048, which silently reports a wrong batch size for any other
seq_len and is one of the numbers a fair central-vs-DiLoCo comparison rests on."""

DP_DEGREE = "dp_degree"
"""Data-parallel ranks in this cluster.

The central node used to infer it from the cluster id -- 8 for anything containing
'lumi', 4 for 'snellius', 1 otherwise -- which is wrong for a partial allocation, a
third site, or a `--replica`-suffixed id, and is what per-rank throughput is divided
by."""

TRANSPORT_INLINE = "inline"
TRANSPORT_BLOB = "blob"
TRANSPORTS = (TRANSPORT_INLINE, TRANSPORT_BLOB)

# Empty-parameter placeholder type, so a mode mismatch is visible rather than being
# read as a zero-tensor model.
BLOB_PARAMETERS_TYPE = "pww-blob"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def sanitise(text: str, fallback: str = "x") -> str:
    """Reduce a name to what `blobstore.safe_name` accepts.

    Applied to run ids and cluster ids, which come from CLI flags and site names, so
    they are trusted-ish -- but they end up in a filesystem path on a public VM, and
    validating at both ends is cheaper than reasoning about which end was careless.
    """
    cleaned = _UNSAFE.sub("-", text or "").strip("-.")
    return cleaned or fallback


def global_blob(run_id: str, round_index: int) -> str:
    return f"{sanitise(run_id, 'run')}-global-r{int(round_index)}.pww"


def init_blob(run_id: str) -> str:
    return f"{sanitise(run_id, 'run')}-init.pww"


def delta_blob(run_id: str, round_index: int, cluster: str) -> str:
    """Named by round *and* cluster, so a retried upload overwrites rather than
    accumulating, and two clusters in the same round cannot collide."""
    return (
        f"{sanitise(run_id, 'run')}-delta-r{int(round_index)}-"
        f"{sanitise(cluster, 'cluster')}.pww"
    )
