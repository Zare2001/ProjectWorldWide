# torchtitan path: environment requirements

This is the one thing that does not drop into the existing setup, so read it
before running anything under `scripts/titan/`.

## The version conflict

The rest of this repo pins **torch 2.7.1** on both sites deliberately, for
numerical comparability: the LUMI container ships `2.7.1+rocm6.2.4` and
`scripts/snellius/setup_venv.sh` installs `2.7.1+cu126` to match it. That pin is
the reason a LUMI result and a Snellius result are meaningfully the same number.

torchtitan at the pinned commit (`bb308da6`, 2025-11-04) needs **torch 2.9 or
newer**. It is built on FSDP2/DTensor APIs that 2.7.1 does not have, so it cannot
run in the existing environments. This is not a version-range annoyance to be
worked around with a looser pin -- the code genuinely is not there in 2.7.1.

So the torchtitan path needs its own environment on each site, and it breaks the
parity pin. That is a real cost of this approach, and the parity has to be
re-established at the new version rather than assumed: both sites should end up on
the **same torch minor version** (2.9.x against 2.9.x), even though one is ROCm and
one is CUDA.

The existing ResNet/CIFAR and HuggingFace LLM paths keep working on 2.7.1,
untouched. The two environments coexist; nothing is migrated.

## What is *not* needed

torchtitan can drive DiLoCo through **torchft** (`fault_tolerance.semi_sync_method
= "diloco"`) plus a Lighthouse process for quorum. This repo does not, because
Flower + FedMom already does the outer step, and running both would be two
implementations of the same thing fighting over the same weights.

That removes real work. torchft ships a compiled Rust extension and PyPI's
`torchft==0.1.1` predates DiLoCo, so using it would mean building a wheel from
source per site with `maturin`, matched to each image's glibc and Python version.
None of that applies here. torchtitan's `FTManager` short-circuits when
`fault_tolerance.enable = false` and never imports torchft (it is guarded by
`importlib.util.find_spec`), which is why every config under `configs/titan/` sets
that flag false.

## Snellius (CUDA, H100)

A second venv, alongside the 2.7.1 one:

```bash
./scripts/titan/setup_venv_snellius.sh
```

It installs torch 2.9.1+cu128 plus torchtitan's pure-Python dependencies. CUDA
12.8 runs on H100 through driver forward-compatibility.

## LUMI (ROCm, MI250X)

LUMI's maintained container is on torch 2.7.1, so this needs a new image built on
a newer ROCm/PyTorch base. `containers/titan-lumi.def` extends the LAIF
ROCm 6.4.4 / PyTorch 2.9.1 image with the pure-Python packages torchtitan needs.

```bash
sbatch scripts/lumi/build_titan_container.sh
```

The one trap, and it is worth stating explicitly: constrain the base image's torch
before installing anything else. `pip install torchdata` will
otherwise resolve `torch` against plain PyPI, which has no ROCm wheels, and
silently replace the ROCm build with a CUDA one. The symptom is
`rocm-smi` happily listing GPUs while `torch.cuda.device_count()` returns 0. The
`%post` section does this with `pip list --format=freeze` (not `pip freeze`, which
cannot match a wheel installed from a direct URL).

## Both sites

torchtitan is a **submodule**, never pip-installed. It reaches the interpreter
through `PYTHONPATH`, which `scripts/titan/run_train.sh` sets (and mirrors into
`SINGULARITYENV_PYTHONPATH`/`APPTAINERENV_PYTHONPATH`, since a container's own env
scripts would otherwise win). Initialise it first:

```bash
git submodule update --init --recursive
```

## Elastic membership: every count of live replicas is normal

HPC queues decide when a site trains, so the run has to tolerate all of it. This is
`min-clients: 1` in `configs/central_aggregator_titan.yaml` plus `--state-dir`, and
between them:

| Live sites | What happens |
|---|---|
| **0** — all queued | The server starts anyway, holds the run, and waits. Nothing is lost. |
| **0** — aggregator restarted | Durable state is re-read; the merge round, momentum buffer and membership survive. |
| **1** | That site trains alone. DiLoCo with k=1 — correct, not degraded — and momentum keeps accumulating across the gap. |
| **N** | The ordinary case. FedMom weights by tokens actually contributed. |
| **A site joins at round 400** | No special handling. It receives the current global model before it trains, so it cannot contribute an update derived from a stale or freshly-initialised model. |
| **A site is killed at walltime** | Its delta is rejected next time by the generation check rather than averaged in stale. DARL separately returns its uncommitted blocks to the pool. |

The trap worth naming: `min-clients` gates whether a round *starts*, and Flower blocks
in the client manager until that many are connected. `round-timeout` does **not** bound
that wait — it only bounds waiting for results once a round has begun. So
`min-clients: 2` means that while Snellius sits in the queue, LUMI connects and idles
and the run makes no progress at all. That is why the default here is 1.

**Generation checking** is what makes late arrival safe. Every delta records the merge
round it was computed against, and the merge refuses any that does not match:

```
rejecting delta from 'snellius': computed against round 12, current round is 47.
The cluster was almost certainly killed at walltime and requeued; its next round
will be current.
```

This is the cheap equivalent of the `quorum_id`/`max_step` guard a peer-to-peer scheme
needs a consensus service for. What is *not* implemented is torchft-style peer quorum,
because a central authority already exists: the server is the single source of truth
for what round it is and what the current weights are, so replicas never have to agree
among themselves. Revisit at 10+ sites, where the probability that someone is down at
any moment approaches 1 and automatic recovery starts earning its complexity.

Check membership at any time:

```bash
./scripts/central_node/status_central_services.sh
```

which reports the merge round and, per cluster, when it joined, when it was last seen,
how many rounds it contributed, how many tokens, and how many stale deltas were
rejected.

## Weight transport, and the >1B ceiling

Two transports. `TRANSPORT` on the start script picks it, and every cluster's
`flower.transport` must match or the client refuses the round with the mismatch spelled
out.

**`inline`** (default) — weights ride inside the Flower gRPC message. One moving part.
gRPC caps a single message at 2 GiB, and no setting raises it:

| Model | float16 | Fits inline? |
|---|---|---|
| 0.6B | 1.2 GiB | yes |
| 1.7B | 3.4 GiB | no |
| 8B+ | 16 GiB+ | no |

**`blob`** — the message carries a name; the bytes go over HTTP to
`pww.central.blobstore` on port 29512. No size ceiling. Both sides stream one tensor at
a time, so memory is a small multiple of the **largest single tensor** (the embedding)
rather than of the model:

```
                        gathered on a rank        peak on the central VM
8B   whole-model gather   14 GiB x 8 ranks         (would be 96 GiB)
8B   per-tensor stream    ~4 GiB                   ~17 GiB
```

That difference is the whole reason blob mode exists — a whole-model all-gather at 70B
is 1.1 TB of host RAM across a node, and the naive server-side merge is 840 GB.

Central node **disk** becomes the binding constraint instead of RAM:

```
resident   global model + momentum buffer   2 x params x 4 bytes (float32)
transient  one delta per site, during a merge   params x 2 bytes x sites

0.6B    5 GiB resident +  2 GiB transient  =   7 GiB
8B     64 GiB resident + 32 GiB transient  =  96 GiB
70B   560 GiB resident + 280 GiB transient = 840 GiB
```

The server logs this budget at startup and errors if the volume is too small, rather
than failing partway through a merge. `--storage-dtype bfloat16` halves the resident
figure at the cost of momentum precision accumulated over hundreds of rounds.

Per round the WAN carries one delta up and one global down per site. At 8B that is
~16 GiB up and ~32 GiB down, so **H is the knob that buys the overhead back** —
`configs/titan/qwen3_8b_c4_diloco.toml` uses `inner_steps = 400` rather than 100 for
exactly that reason. Watch `drift_ratio` in the round logs: it is the local update's
norm over the weights' own norm, and if it climbs toward 1 the replicas are diverging
faster than averaging can reconcile and H is too large.

Start the central node with out-of-band transport:

```bash
AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml TRANSPORT=blob \
NUM_SAMPLES=<windows> BLOCK_SIZE=1024 SEED=42 \
    ./scripts/central_node/start_central_services.sh
```

Firewall port 29512 to the sites' egress addresses and set `DARL_TOKEN` — anyone who
can PUT to the blob store can replace the global model.

## Order of operations for a real run

```bash
# once per site, login node (compute nodes have no internet)
scripts/titan/download_tokenizer.sh                       # OpenEuroLLM 128k
scripts/titan/tokenize_c4.sh --dataset c4 --seq-len 2048 --max-files 32

# once, on the central VM -- NUM_SAMPLES is the *window* count that
# tokenize_c4.sh printed, and these are environment variables, not flags.
# Add TRANSPORT=blob for any model above ~1B.
AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml \
NUM_SAMPLES=<windows> BLOCK_SIZE=1024 SEED=42 \
    ./scripts/central_node/start_central_services.sh

# per site
sbatch scripts/snellius/job_titan_diloco.sh
sbatch scripts/lumi/job_titan_diloco.sh

# the baseline it is measured against -- no central VM, its own throwaway
# coordinator on the same space_seed = 42 block space
sbatch scripts/snellius/job_titan_central.sh
```

Sites can be submitted in any order, and neither has to wait for the other: the
server holds the run while both are queued, trains with whichever arrives first, and
absorbs the second whenever it appears.

Validate one site on its own first -- `configs/titan/qwen3_0.6b_smoke.toml` runs the
whole local half (FSDP2, DARL leasing, checkpoint, commit ordering) with
`flower.enable = false`, so it needs no central node and no WAN.

**Starting over is an action, not the absence of one.** Every one of these resumes by
default, which is what an HPC job killed at walltime needs -- including the baseline,
which spawns a fresh coordinator every job and so looks stateless while its checkpoint
is not. `scripts/reset_run.sh --dry-run` lists what a reset would clear;
`PWW_FRESH_RUN=1` at submit time covers anything written between the reset and the job
starting. See [RUNBOOK.md](../../RUNBOOK.md) "Starting a genuinely fresh run".

**Reading the results.** The comparison is baseline versus the aggregator's own WandB run,
not versus a single site's, and the honest x-axis is `train/cum_tokens` rather than the
step -- equal steps is not equal work when the federation trains 12 ranks against the
baseline's 4. [FEDERATION_GUIDE.md](../../FEDERATION_GUIDE.md) section 5 has the key
reference; `configs/titan/qwen3_0.6b_c4_central.toml`'s header has the arithmetic.

## What is verified, and where

```bash
# no GPUs, no allocation, no torchtitan needed for the second one
python3 tests/test_titan.py        # shards, DARL leasing exactly-once, wire codec
python3 tests/test_federation.py   # blob store, elastic membership, both transports
```

`test_federation.py` covers the elasticity claims above directly: zero live replicas
starting a run, one replica merging alone, a late joiner recorded from the round it
actually arrived, a requeued site's stale delta being rejected rather than averaged in,
a mismatched-vocab delta refused, and the aggregator restarting mid-run without losing
the merge round or the momentum buffer.

Not covered on CPU, and therefore what the smoke config is for: FSDP2 wrapping, the
real Qwen3 forward pass, and the DTensor gather/scatter against a genuinely sharded
model at scale.
