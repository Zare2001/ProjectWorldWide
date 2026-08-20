"""The `[darl]` and `[flower]` sections added to torchtitan's JobConfig.

torchtitan merges this module's `JobConfig` with its own when a run sets
``job.custom_config_module = "pww.titan.config"`` (see
`torchtitan/config/manager.py::_merge_configs`). Extra top-level sections come
through as new fields, and tyro exposes every one on the CLI as
``--darl.<field>`` / ``--flower.<field>``, so anything here can be overridden per
site without touching the TOML.
"""

# Deliberately no `from __future__ import annotations`. torchtitan's
# ConfigManager._merge_configs inspects `dataclasses.fields(...)[i].type` and calls
# `is_dataclass()` on it to decide whether to merge a section recursively. With
# postponed evaluation those types are strings, `is_dataclass("DARL")` is False,
# and the merged config ends up with unresolvable string annotations -- which
# surfaces as `NameError: name 'DARL' is not defined` when tyro builds the CLI.
# torchtitan's own custom config module omits it for the same reason.

from dataclasses import dataclass, field


@dataclass
class DARL:
    """Dynamic dataset leasing: which tokens this cluster is allowed to train on."""

    url: str = ""
    """Coordinator base URL, e.g. http://145.38.206.143:29510. Required when
    training.dataset is 'pww_tokens'."""

    token: str = ""
    """Shared secret the coordinator was started with. Falls back to $DARL_TOKEN."""

    site: str = ""
    """Site name used to derive this cluster's coordinator identity (lumi, snellius)."""

    cluster_id: str = ""
    """Overrides the derived identity outright. Keep it stable across requeues --
    the coordinator sizes grants from a cluster's measured throughput, and a new
    id on every resubmission throws that history away."""

    block_size: int = 1024
    """Windows per leasable block. The leasing granularity: too small and the
    coordinator becomes the bottleneck, too large and the tail of an epoch is
    lumpy. At seq_len 2048 this is ~2M tokens per block."""

    blocks_per_phase: int = 0
    """Blocks leased per inner phase. 0 derives it from inner_steps and the batch
    geometry via darl.space.blocks_for_phase, which is what makes a lease boundary
    land exactly on an outer step. Set it explicitly only to debug."""

    inner_steps: int = 100
    """H: optimiser steps between outer aggregations. Also sizes the lease."""

    epochs: int = 1
    """Passes over the corpus. Reaching the end of the last one ends training."""

    space_seed: int = 42
    """Seeds the global block permutation. Must be identical on every site --
    BlockSpace.digest makes a mismatch a registration error rather than silent
    duplicate work."""

    commit_policy: str = "checkpoint"
    """When this cluster tells the coordinator a leased span is finished.

    'checkpoint' is the exact one: spans are committed only after the work is
    inside a durable checkpoint, so {weights, committed blocks} fail together and a
    crash loses the same work from both sides. The cost is that a cluster holds
    every lease since its last checkpoint, so `checkpoint.interval` should stay a
    small multiple of `inner_steps` -- otherwise a cluster accumulates uncommitted
    leases and can drain the pool for everyone else near the end of an epoch.

    'consumption' commits at the end of each phase instead. One fewer dependency
    and spans recycle sooner, at the cost of an exactness window of one lease: a
    crash after committing but before checkpointing drops those windows from the
    epoch. See darl.client.CommitPolicy."""

    shuffle: bool = True
    """False gives the identity permutation, so a lease maps to a readable window
    range. Debugging only."""

    use_proxy: bool = False
    """Route coordinator RPCs through http_proxy. Needed only when the coordinator
    is outside the facility and the site forces egress through a gateway."""


@dataclass
class Flower:
    """Cross-site outer step: Flower gRPC + the FedMom strategy on the central node."""

    enable: bool = False
    """False runs plain single-cluster torchtitan, DARL still leasing. Useful for
    validating a site before involving the WAN."""

    server_address: str = ""
    """host:port of the Flower server, e.g. 145.38.206.143:29511."""

    protocol: str = "grpc"
    """How the ROUND protocol reaches the central node: 'grpc' or 'http'.

    Orthogonal to `transport`, which is about the WEIGHTS. 'http' exists for a site
    whose compute nodes have no route out except an HTTP forward proxy: Flower's gRPC
    stream has to stay open for the whole job and such a proxy reaps it mid-round,
    while short request/response calls -- the same shape as the DARL client and the
    blob store, both of which work from those sites -- do not. Must match the central
    node's --protocol, and requires transport = "blob"."""

    transport: str = "inline"
    """Which weight transport this cluster expects: 'inline' or 'blob'.

    Must match what the central node was started with (`--transport`), and a mismatch
    is a startup error rather than something to paper over. It exists because the
    inline path needs a parameter ordering built from a full gather of the model, and
    that gather is precisely the operation blob transport avoids -- all-gathering a
    70B model onto every rank is 1.1 TB of host RAM across a node. Declaring the
    transport lets the client skip it entirely when the weights are going out of band."""

    rounds: int = 0
    """Outer rounds this client will participate in before disconnecting. 0 means
    follow the server, which is the normal case."""

    max_message_length: int = 2_147_483_647
    """gRPC message cap in bytes, and a protocol limit rather than a tunable -- raising
    this number does not raise the ceiling.

    A full parameter set crosses the wire each round, so it is the real ceiling on
    model size for the inline transport: 1,073,741,823 parameters at float16,
    536,870,911 at float32. Measured against the flavors this repo ships configs for
    (meta device, vocabulary padded to 131328), only 0.6B fits, and only at float16:

        0.6B     709,427,200 params    1.3 GiB fp16    2.6 GiB fp32
        1.7B   1,947,329,536 params    3.6 GiB fp16
        8B     8,021,914,624 params   14.9 GiB fp16

    Anything else needs `transport = "blob"`."""

    wire_dtype: str = "float16"
    """Dtype parameters are serialised in, for the inline transport only.

    Both 16-bit options halve WAN traffic against float32 and are exact for weights of
    normal magnitude. They differ in range, not precision, and the shipped configs all
    set **bfloat16** -- it is what `training.dtype` already is, so the round trip
    introduces no dtype the model was not already living in, and it cannot flush a small
    weight to zero. float16's smallest subnormal is 5.96e-08, so 3e-08 rounds up to it
    and anything at or below 2e-08 goes to zero: about 1 element in 819,200 of a randn
    tensor moves. That is harmless in practice and is pinned by a test, which is why this
    default is still float16 -- but prefer bfloat16 in a config, and note that the
    aggregator also falls back to bfloat16 when it has not yet seen what a client sent
    (`strategy._on_wire`).

    bfloat16 crosses the wire as uint16 bit patterns, because numpy has no bfloat16, so
    the decode is a reinterpretation rather than a cast -- see `strategy._from_wire` for
    what happens when that is got wrong.

    float32 is *not* simply the same thing at twice the bytes: it halves the parameter
    ceiling above, and at 2.6 GiB even the 0.6B flavor no longer fits in a gRPC
    message. Use blob transport instead of reaching for float32 here.

    Either way the central node holds its momentum state and the authoritative global
    model in float32 (see central/strategy.py)."""


@dataclass
class Titan:
    """Model-build settings that torchtitan's own sections have no field for."""

    pad_vocab_to_multiple_of: int = 256
    """Round the embedding up to a multiple of this after taking the vocabulary
    size from the tokenizer.

    Not cosmetic. OpenEuroLLM's 128k tokenizer reports 131073 ids (131072 plus one
    added token) -- an odd number, which misaligns every embedding and output-projection
    GEMM, and is not divisible by any tensor-parallel degree above 1, so
    `parallelism.tensor_parallel_degree > 1` would fail outright. Padding to 256
    gives 131328 here: 255 rows no token can ever index, ~0.2% of the embedding,
    against correct sharding and aligned matmuls.

    Set to 0 or 1 to use the tokenizer's count verbatim."""


@dataclass
class JobConfig:
    darl: DARL = field(default_factory=DARL)
    flower: Flower = field(default_factory=Flower)
    titan: Titan = field(default_factory=Titan)
