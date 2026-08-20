# ProjectWorldWide

Distributed training infrastructure that runs the same code on **LUMI** (CSC,
AMD MI250X) and **Snellius** (SURF, NVIDIA). Starts with ResNet/CIFAR-10 as a
fast, cheap way to prove the whole stack works, and is structured so LLM
training reuses the same pieces rather than replacing them.

Machine-specific detail lives in `sites/<site>.sh` and `scripts/<site>/`.
Everything under `src/pww/` is site-independent: it reads the accelerator kind
and pinning variable from the environment rather than assuming ROCm or CUDA.

| | LUMI | Snellius |
|---|---|---|
| accelerator | AMD MI250X, ROCm/RCCL | NVIDIA H100 / A100, CUDA/NCCL |
| ranks per node | 8 (each MI250X = 2 GCDs) | 4 |
| pinning variable | `ROCR_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` |
| environment | prebuilt Singularity container | pip venv built by `setup_venv.sh` |
| torch | 2.7.1+rocm6.2.4 | 2.7.1+cu126 |
| partitions | `dev-g`, `small-g`, `standard-g` | `gpu_h100`, `gpu_a100` |
| interconnect | Slingshot (`hsn`) + aws-ofi-rccl | InfiniBand |
| MIOpen cache workaround | required | not applicable |

## How it works

### The problem

Two supercomputers, ~1,500 km apart, that between them have far more compute than
either alone. Ordinary data parallelism cannot use both: it all-reduces gradients
**every step**, so it needs a fat, low-latency path between every pair of ranks. Over a
WAN that path is the run.

Three separate things break, and each needs its own answer:

| what breaks | why | the answer here |
|---|---|---|
| gradient exchange every step | ~100 ms RTT and shared bandwidth, against a step of tens of ms | **DiLoCo** — exchange every `H` steps instead of every step |
| both sites train the same data | each site's dataloader shards *its own* ranks, so running the same code twice is two redundant runs, not one distributed one | **DARL** — a coordinator leases disjoint token ranges at run time |
| sites are not up at the same time | HPC queues. One site waits hours while the other idles | **elastic membership** — 0, 1 or N live sites are all normal states |

### The shape of it

One central VM with no GPUs coordinates; the two facilities never talk to each other.

```
                    +--------------------------------------------------+
                    |        CENTRAL VM  (145.38.206.143)              |
                    |        no GPUs -- it only coordinates            |
                    |                                                  |
                    |  DARL coordinator            :29510              |
                    |    the only map of which token blocks are        |
                    |    unassigned / leased / committed               |
                    |                                                  |
                    |  Flower + PWWFedMom          :29511              |
                    |    theta_global, momentum buffer, membership     |
                    |    -- all durable on disk, so a restart or an    |
                    |    empty queue loses nothing                     |
                    |                                                  |
                    |  blob store                  :29512              |
                    |    only for models above ~1B                     |
                    +---------^--------------------------^-------------+
                              |                          |
             leases, then     |                          |    leases, then
             weight deltas    |                          |    weight deltas
                              |                          |
        +---------------------+-----+      +-------------+----------------------+
        |  LUMI          (EuroHPC)  |      |  SNELLIUS              (SURF)      |
        |  8 x MI250X GCD, ROCm     |      |  4 x H100, CUDA                    |
        |  torchtitan + FSDP2       |      |  torchtitan + FSDP2                |
        |                           |      |                                    |
        |  H inner steps on the     |      |  H inner steps on the              |
        |  blocks IT leased         |      |  blocks IT leased                  |
        +---------------------------+      +------------------------------------+

                 no traffic between the two facilities, ever
```

Inside a facility, nothing is unusual: FSDP2 shards the model across that site's GPUs
and all-reduces every step over the local interconnect, which is what it is good at.
Only the *outer* loop crosses the WAN.

### The two loops

```
  inner loop  (per site, no WAN traffic)         outer loop  (H times less often)
  ---------------------------------------        ------------------------------------
  for h in 1..H:                                 delta_i = theta_i - theta_global
      x ~ the blocks this site leased                    (one file, streamed)
      loss = f(x, theta_i)                        --> central VM
      theta_i = AdamW(theta_i, grad)
                                                 theta_global = OuterOpt(
  H = 100 by default, so the WAN is                  theta_global,
  touched once per ~100 optimiser steps                sum_i p_i * delta_i)

                                                 <-- every site loads the new
                                                     theta_global and starts again
```

`OuterOpt` is Nesterov momentum at the paper's `lr` 0.7 / `momentum` 0.9. The FedMom
update the server applies is *algebraically* Nesterov, not merely momentum-flavoured —
pinned against `torch.optim.SGD(nesterov=True)` in `tests/test_federation.py`. Set
`momentum 0.0, lr 1.0` and it collapses to exact FedAvg, which is the control arm.

### One outer round, end to end

```
  SITE                                CENTRAL VM
  ----                                ----------
  1. POST /acquire  ----------------->  pick disjoint blocks, mark them LEASED
                    <-----------------  span [4096, 5120)          [DARL]
     Only rank 0 asks; it broadcasts the span to the other ranks, which each
     derive their own slice locally. A 512-rank job makes one RPC, not 512.

  2. load theta_global  <------------  whatever round the server is on now
     Unconditional, so a site joining at round 400 cannot contribute an
     update derived from stale weights.

  3. H inner steps .................  (no coordinator traffic at all)
     Heartbeats continue in the background; the reply is an *instruction* --
     it carries the authoritative lease end, which may have shrunk because a
     faster site stole the untouched tail.

  4. write checkpoint, THEN commit  ->  mark blocks COMMITTED
     That order is the exactly-once guarantee: {weights, committed blocks}
     fail together, so a crash loses the same work from both sides.

  5. delta = theta_local - theta_global
     PUT delta  ---------------------->  check base_round is current, then merge:
                                           v = w - eta*(w - sum_i p_i*theta_i)
                                           w = v + beta*(v - v_prev)
                                         publish theta_global, round += 1
```

Step 5's `base_round` check is what makes requeueing safe: a site killed at walltime
and restarted hours later computed its delta against a global model that has moved on,
and that delta is **rejected** rather than averaged in. Its next round is current.

### Who is "a site", exactly

A cluster id, and it is worth being careful about because two things with different
lifetimes were once the same field:

- **the logical stream** — `lumi`. Must be *stable across a requeue*, because the
  coordinator sizes a cluster's grants from its measured throughput and a new id on
  every resubmission throws that history away. `SLURM_JOB_ID` therefore cannot be part
  of it.
- **the process** — a random *incarnation* id, generated per session. Must be *unique
  among concurrent jobs*.

Nothing in the environment is both, which is why the two are separate. Running two jobs
at one facility — routine, since both sites allow partial-node allocations — needs
`--replica a` / `--replica b` to give each its own cluster id. Forget it and the
coordinator refuses the second one rather than letting two processes quietly release
each other's leases and overwrite each other's deltas. `FEDERATION_GUIDE.md` has the
three guards and what each catches.

### The operational workflow

The shape of it, in one view. **[RUNBOOK.md](RUNBOOK.md) is the copy with the actual
commands** — including first-time setup, which this omits.

```
  ONCE, offline (login node, no allocation)
    scripts/titan/download_tokenizer.sh     OpenEuroLLM 128k
    scripts/titan/stage_c4.sh               fetch C4 shards
    scripts/titan/tokenize_c4.sh            -> fixed-width memmap + manifest
                                            -> prints the WINDOW COUNT
    scp .../manifest.json <central>:/tmp/   carry that count in a file rather
                                            than by hand -- a few hundred bytes
  THEN, on the central VM
    start_central_services.sh               MANIFEST=/tmp/manifest.json, or the
                                            count directly as NUM_SAMPLES. Both
                                            sites must agree or registration is
                                            refused by the block-space digest
  THEN, at each site, independently
    sbatch scripts/{lumi,snellius}/job_titan_diloco.sh
                                            queue whenever; the server waits
  WHILE RUNNING
    status_central_services.sh              merge round + per-cluster membership
    tail -f runs/central/flower.log         rounds, loss, drift, tokens
  ENDS WHEN
    DARL runs out of blocks -> the dataloader stops rather than looping on
    empty work, which is what an earlier run did for 23 silent rounds
```

### Where each piece lives

| concern | code |
|---|---|
| which tokens a site may train on | [src/pww/darl/](src/pww/darl/) — `table.py` is the state machine, `server.py` the coordinator, `client.py` the site side |
| the inner training loop | [src/pww/titan/trainer.py](src/pww/titan/trainer.py) — wraps torchtitan's `Trainer` to run `H` steps at a time |
| the outer step | [src/pww/central/globalstate.py](src/pww/central/globalstate.py) — the streaming FedMom merge; [strategy.py](src/pww/central/strategy.py) — the round protocol |
| getting weights across | [src/pww/titan/params.py](src/pww/titan/params.py) inline, [src/pww/delta.py](src/pww/delta.py) out-of-band |
| single-site DiLoCo (no WAN) | [src/pww/diloco.py](src/pww/diloco.py) — same algorithm as a collective inside one allocation |
| which sites to submit, and when | [src/pww/plan/](src/pww/plan/) — `timeline.py` simulates the round barrier forward, `search.py` picks the submission plan ([PLANNER.md](PLANNER.md)) |

Step-by-step commands are in **[RUNBOOK.md](RUNBOOK.md)**, the reference behind them in
**[FEDERATION_GUIDE.md](FEDERATION_GUIDE.md)**, the torchtitan environment
requirements in **[scripts/titan/README.md](scripts/titan/README.md)**, and bringing a
third machine into the federation in **[ADDING_A_CLUSTER.md](ADDING_A_CLUSTER.md)**.
The arm that goes past plain DiLoCo — server-scheduled sync periods (QSR) plus a
Jensen-gauge dispersion controller — is **[DCLT_ARM.md](DCLT_ARM.md)**. Deciding which
sites to submit, at what shape and starting when, from the measured queue and the corpus
DARL has left, is **[PLANNER.md](PLANNER.md)**.

## Layout

```
env.sh                    site detection + shared config
sites/
  lumi.sh                 LUMI specifics
  snellius.sh             Snellius specifics
configs/                  YAML run configs
scripts/
  bootstrap.sh            one-time setup (run first)
  siteinfo.sh             report a machine's topology/partitions/modules
  download_data.sh        pre-fetch datasets -- LOGIN NODE ONLY
  task_wrapper.sh         per-rank entrypoint, site-agnostic
  darl_coordinator.sh     start/stop the lease coordinator -- LOGIN NODE ONLY
  reset_run.sh            clear a run's site state so the next submission starts
                          from step 0 -- checkpoints, blob staging, tb, and the
                          baseline jobs' throwaway coordinator dirs. --dry-run first
  lumi/                   job_smoke, job_cifar_debug, job_cifar_1node,
                          job_cifar_multinode, job_cifar_diloco
  snellius/               setup_venv.sh (run first), job_smoke, job_cifar_debug,
                          job_cifar_1node, job_cifar_multinode, job_cifar_diloco
src/pww/
  distributed.py          process group, GPU pinning, collective reductions
  parallel.py             DDP / FSDP wrapping, mixed precision, act. checkpointing
  diloco.py               DiLoCo: replica layout, outer optimizer, outer step
  checkpoint.py           consolidated + sharded save/load, atomic writes
  config.py               YAML-under-argparse, seeding, run dirs
  logging_utils.py        rank-aware logging, JSONL metrics
  smoke.py                infrastructure self-test
  models/resnet.py        ResNet-18/34/50, CIFAR stem
  data/cifar.py           CIFAR-10 loaders with DistributedSampler
  train_cifar.py          training entrypoint
  darl/                   dynamic dataset leasing across clusters
    space.py              index space, block permutation, digests
    table.py              lease state machine: TTL, expiry, stealing, invariants
    server.py             coordinator -- HTTP, one lock, write-ahead log
    client.py             cluster side -- RPCs, heartbeats, prefetch
    torch_data.py         spans -> per-rank sample lists
    simulate.py           m clusters end to end, with a coverage audit
tests/
  test_local.py           CPU-only, single process, no allocation needed
  test_diloco_gloo.py     CPU-only, multi-process gloo -- DiLoCo collectives
  test_darl.py            CPU-only, incl. a real coordinator on a socket
  test_titan.py           CPU-only, the torchtitan path (shards, leasing, wire)
```

There is a second, newer training path built on **torchtitan** -- Qwen3 flavors,
FSDP2, C4, and the OpenEuroLLM 128k tokenizer -- which keeps DARL and Flower/FedMom
but replaces the hand-rolled HuggingFace inner loop. It needs its own environment
(torch >= 2.9, which conflicts with this repo's deliberate 2.7.1 pin), so it is
documented separately in **[scripts/titan/README.md](scripts/titan/README.md)**.
Start there rather than here if that is what you are running.

```
third_party/torchtitan   submodule, pinned; never pip-installed (PYTHONPATH)
configs/titan/           TOML run configs for the torchtitan path
containers/titan-lumi.def  ROCm image for it (LUMI's own is on torch 2.7.1)
scripts/titan/           tokenizer download, C4 staging/tokenising, launcher
src/pww/
  fedproto.py            the config/metric keys the two sides agree on
  wandb_utils.py         the axis conventions all three run types share, and the
                         guard against WandB discarding a rewound step. No
                         torchtitan, so the central VM can import it
  tensorio.py            tensor file with a streaming writer, so a model can be
                         built or merged without ever holding it whole
  delta.py               out-of-band transport: streaming HTTP + per-tensor
                         DTensor gather/scatter
  central/
    blobstore.py         HTTP blob store daemon for multi-GB weight deltas
    globalstate.py       durable global model, momentum, membership; the
                         per-tensor FedMom merge
  titan/
    config.py            the [darl]/[flower]/[titan] JobConfig sections
    __init__.py          pww_qwen3 train spec; vocab_size from the tokenizer
    datasets.py          C4 registrations into torchtitan's DATASETS
    shards.py            pre-tokenised memmap format DARL leases over
    tokenize_corpus.py   the one-off offline tokenisation pass
    darl_dataloader.py   DARL-leased torchtitan BaseDataLoader
    trainer.py           torchtitan Trainer, one inner phase at a time
    params.py            whole-state gather/scatter and the inline wire codec
    flower_client.py     the cross-site outer round, over either transport
    wandb_metrics.py     the one metrics hook BOTH drivers install, so a baseline
                         and a DiLoCo run cannot compute train/loss, the token
                         axis or the held-out loss differently
    train.py             entrypoint (torchrun -m pww.titan.train)
tests/
  test_titan.py          shards, DARL leasing, wire codec
  test_federation.py     blob store, elastic membership, both transports
```

The central node runs three daemons: the DARL coordinator (29510), the Flower
aggregator (29511) and — only for models above ~1B, where weights no longer fit in a
gRPC message — the blob store (29512). Membership is elastic by default: zero live
replicas (every site queued), one, or several are all normal states, and the run
survives a restart of the aggregator itself. See
[scripts/titan/README.md](scripts/titan/README.md).

Code lives in `$HOME`; data, checkpoints and logs live on scratch, symlinked as
`data/` and `runs/`. Which scratch differs per site, and so do the limits:

| | LUMI | Snellius |
|---|---|---|
| `$HOME` | 20 GB, 100k inodes | 200 GiB, 1M inodes |
| scratch (`PWW_SCRATCH`) | `/scratch/project_462000226/$USER` | `/scratch-shared/$USER` (8 TiB, **purged ~14 days**) |
| venv | n/a (container) | `$HOME/venvs/pww-snellius` (~6 GB) |

## Getting started

```bash
cd ~/ProjectWorldWide
./scripts/bootstrap.sh            # dirs, symlinks, container check
./scripts/download_data.sh        # once, from a login node (slow: ~170 MB)

# validate before spending GPU hours
source env.sh && pww_run python3 tests/test_local.py
source env.sh && pww_run python3 tests/test_diloco_gloo.py
source env.sh && pww_run python3 tests/test_darl.py        # dataset leasing
sbatch scripts/lumi/job_smoke.sh
sbatch scripts/lumi/job_smoke.sh --diloco-replicas 2

# train
sbatch scripts/lumi/job_cifar_debug.sh                                     # 1 GCD, ~2 min
sbatch scripts/lumi/job_cifar_1node.sh --config configs/cifar10_resnet18.yaml
sbatch --nodes=4 scripts/lumi/job_cifar_multinode.sh
sbatch scripts/lumi/job_cifar_diloco.sh                                    # 2 replicas
```

Job logs land in `logs/<jobname>-<jobid>.out`. Per-run outputs (config snapshot,
`train.log`, `metrics.jsonl`, checkpoints) land in `runs/<run-name>/`.

Flags after the script name are forwarded to the trainer:

```bash
sbatch scripts/lumi/job_cifar_1node.sh --model resnet50 --epochs 50 --dtype bf16
sbatch scripts/lumi/job_cifar_1node.sh --parallel fsdp   # exercise the LLM path
sbatch scripts/lumi/job_cifar_1node.sh --resume auto     # continue newest checkpoint
```

## Environment

No container build is required. LUMI's maintained image already has everything
for both phases:

| | |
|---|---|
| container | `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif` |
| torch | 2.7.1+rocm6.2.4 |
| also inside | torchvision 0.22.1, transformers 4.55.3, tokenizers 0.21.4, datasets 4.0.0, accelerate 0.34.2, flash-attn 2.7.3, aws-ofi-rccl |

`sites/lumi.sh` deliberately contains no `module load`, so it behaves identically
in login shells, batch scripts and containers. It inlines the one variable that
`singularity-AI-bindings` sets; if LUMI changes those bindings, update it there.

### Container choice

Override per job rather than editing the site file:

```bash
PWW_CONTAINER=/path/to/other.sif sbatch scripts/lumi/job_smoke.sh
```

A richer image exists on project scratch:
`laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif`. Both were
benchmarked head to head on the same jobs:

| | LUMI official (default) | laif |
|---|---|---|
| torch / ROCm | 2.7.1 / 6.2.4 | 2.9.1 / 6.4.4 |
| CIFAR-10, 30 ep, 8 GCDs | 93.35% @ 38,400 img/s | 93.27% @ 37,700 img/s |
| all-reduce, 1 node | 123 GB/s | 123.4 GB/s |
| all-reduce, 2 nodes | 88 GB/s | 87.8 GB/s |
| `tests/test_local.py` (17 checks then, 28 now) | all pass | all pass |
| flash-attn | 2.7.3 | 2.8.0 |
| transformers | 4.55.3 | 4.57.3 |
| Transformer Engine | -- | 2.4.0 |
| DeepSpeed / apex | -- | 0.18.6 / yes |

**For the CIFAR phase there is no reason to switch** -- performance is identical
within noise. For LLM work the extra packages are worth having: DeepSpeed as a
ZeRO alternative to FSDP, apex `FusedAdam`, newer FlashAttention.

Three things to know before adopting it:

1. **fp8 does not work on LUMI, at all.** Transformer Engine's fp8 path asserts
   `Device arch gfx94x or gfx95x required`; MI250X is gfx90a. fp8 needs MI300X or
   newer. If fp8 is why you want TE, it will not pay off here -- use bf16.
2. **TE layers must not be wrapped in `torch.autocast`.** `te.Linear` under
   autocast fails with `Unable to find any suitable algorithms` on gfx90a. With
   explicit dtypes (fp32/bf16/fp16) it works. Set the layer dtype directly.
3. **It lives on scratch and is owned by another user.** LUMI scratch is purged
   after inactivity, and the file could be moved or deleted at any time, breaking
   every job that references it. The default container is under `/appl` and is
   system-maintained. If you adopt the laif image for real work, copy it into
   your own space first (13.5 GB against a 50 TB quota).

Verified working on GPU in the laif image: FlashAttention 2.8.0 forward,
`te.Linear` in fp32/bf16/fp16, apex `FusedAdam`.

## DiLoCo

Plain data parallelism all-reduces gradients on every step, so it needs a fat,
low-latency link between every pair of ranks. [DiLoCo](https://arxiv.org/abs/2311.08105)
replaces that with two nested loops, cutting inter-replica traffic by a factor
of `H`:

| paper | here | what it is |
|---|---|---|
| `k` | `--diloco-replicas` | model replicas, each on its own data shard |
| `H` | `--diloco-inner-steps` | inner steps between exchanges |
| `T` | derived | outer steps = total inner steps / `H` |
| `InnerOpt` | `--inner-optimizer` | AdamW in the paper; SGD+Nesterov for ResNet |
| `OuterOpt` | `--diloco-outer-optimizer` | Nesterov momentum, `lr` 0.7, `momentum` 0.9 |
| Δ⁽ᵗ⁾ | logged as `delta_norm` | `mean_i(θ⁽ᵗ⁻¹⁾ − θᵢ⁽ᵗ⁾)`, averaged then fed to `OuterOpt` |

Set `--diloco-outer-optimizer sgd --diloco-outer-lr 1.0 --diloco-outer-momentum 0`
and the outer step collapses to `θ⁽ᵗ⁾ = mean_i θᵢ⁽ᵗ⁾`, i.e. FederatedAveraging.
That identity is what the test suite checks against.

**The cross-site outer step is the same algorithm.** `diloco.py` runs `OuterOpt`
inside one allocation as a collective; across sites, `central/strategy.py` applies
FedMom on the central node instead. Those are not two different optimisers — FedMom's

```
v_next = w - eta*(w - w_avg);   w_next = v_next + beta*(v_next - v_prev)
```

is *algebraically* Nesterov momentum on the same Δ⁽ᵗ⁾. Substituting
`m = v_t − v_(t−1)` gives `m_next = beta*m − eta*Δ` with `Δ` evaluated at
`w = v + beta*m`, which is Nesterov's accelerated gradient in two-sequence form and
matches `torch.optim.SGD(momentum=beta, nesterov=True)` exactly, first step included.
`tests/test_federation.py` pins that equivalence against a heavy-ball control. Both
paths therefore default to the paper's `lr` 0.7 / `momentum` 0.9.

The cross-site path differs from the paper in two deliberate ways: deltas are
weighted by tokens contributed rather than uniformly by `1/k` (the sites do different
amounts of work per round, and this reduces to `1/k` when they do not), and `k`
varies between rounds because membership is elastic.

### Rank layout

One allocation is carved into `k` contiguous equal blocks, so `k` must divide the
rank count — 1, 2, 4 or 8 on a LUMI node, 1, 2 or 4 on a Snellius node. With
`k=2` on one LUMI node:

```
replica 0 = ranks 0-3        replica 1 = ranks 4-7
     inner group, DDP every step   inner group, DDP every step
            \____________  outer groups  ____________/
              {0,4} {1,5} {2,6} {3,7}  --  every H steps
```

Contiguous on purpose: SLURM numbers ranks node by node, so a replica stays
inside as few nodes as possible and the chatty inner all-reduce keeps the fast
local links. Every rank runs the outer step redundantly rather than electing a
leader per replica — DDP already makes the ranks of a replica bit-identical, so
pairing rank *j* of each replica with rank *j* of the others turns one `k`-way
exchange into four parallel `k`-way exchanges and leaves every rank already
holding `θ⁽ᵗ⁾` with no follow-up broadcast. The same layout is what makes FSDP
work: rank *j* holds shard *j*, and exchanges with the ranks holding shard *j*
elsewhere.

The layout DiLoCo is actually *for* is one replica per node, which removes
inter-node traffic except once every `H` steps:

```bash
sbatch --nodes=4 -p standard-g scripts/lumi/job_cifar_diloco.sh \
    --diloco-replicas 4 --diloco-inner-steps 100
```

### Choosing k and H, and reading the result

`H` is the whole trade. Watch `agreement` in `metrics.jsonl` — the norm of the
averaged Δ over the norm of this replica's own Δ:

- **near 1.0** — replicas still travelled in the same direction; `H` is safe and
  you could probably raise it.
- **near 1/√k** — replicas travelled in mutually orthogonal directions, so
  averaging is cancelling most of their progress. `H` is too large for this
  model and LR, and accuracy will suffer.

Because a step only synchronises within a replica, **the LR is scaled to the
replica batch, not the world batch**. `--batch-size 128` with `k=2` on 8 GCDs
gives a replica batch of 512 and LR 0.4, where plain DDP on the same node gives
1024 and 0.8. That is deliberate — each replica *is* a 4-rank data-parallel job —
but it means a DiLoCo run and a DDP run at the same `--batch-size` are not the
same experiment.

Evaluation and checkpoints report `θ`, not whichever replica rank 0 happens to
hold (`DiLoCo.global_model()`). Mid-inner-phase the replicas genuinely differ, so
a world-reduced metric on local weights would be an average over `k` different
models; `θ` is identical on every rank and lags by at most `H` steps. The final
partial inner phase is flushed before the last eval, so the last checkpoint is
the model you trained.

One asymmetry to know about: the checkpoint pairs `θ` with the *inner* optimizer
state of whichever rank wrote it, because inner optimizer state is per replica by
construction and there is no meaningful average of it. Resuming therefore restarts
each replica from `θ` carrying replica 0's inner momentum. The outer momentum,
which *is* shared, is restored exactly (`runs/<run>/diloco/outer_r0.pt`).

### What an outer step actually costs

Measured on one LUMI node, ResNet-18 (11.17M params), `k=2` x 4 GCDs
(`job_smoke.sh --diloco-replicas 2`):

| | |
|---|---|
| inner step (fwd + bwd + DDP all-reduce over 4 ranks) | 10.7 ms |
| outer step (Δ, one 2-way all-reduce, OuterOpt, re-dispatch) | 3.8 ms |
| overhead at `H`=10 / 100 / 500 | 3.6% / 0.36% / 0.07% |

The outer step being *cheaper* than an inner one is not a surprise on a single
node: it is one all-reduce of the parameters over 2 replicas across Infinity
Fabric, against a full forward, backward and 4-rank gradient all-reduce. Two
things change that ratio and neither is measured yet — replicas on separate nodes
push the exchange onto Slingshot, and a model large enough to need FSDP changes
both sides. Re-run the smoke test in whatever layout you actually intend to use;
it prints this table for that layout.

Memory cost is three extra fp32 copies of the parameters — `θ`, the outer
momentum, and one flat communication buffer. Under FSDP all three are sharded
with the model. `--diloco-outer-device cpu` moves two of them to host memory in
exchange for two host↔device copies every `H` steps.

### Measured: on a BatchNorm model, use FederatedAveraging

Four runs on one LUMI node, `k=2` x 4 GCDs, ResNet-18, 30 epochs, per-rank batch
128, everything identical except the outer optimizer and `H`. Plain DDP on the
same node gets **93.35%**:

| outer optimizer | `H` | `T` | best eval | vs DDP |
|---|---|---|---|---|
| `sgd` 1.0 / 0 (FederatedAveraging) | 25 | 58 | **92.41%** | −0.94 |
| `sgd` 1.0 / 0 (FederatedAveraging) | 100 | 15 | **92.17%** | −1.18 |
| `nesterov` 0.7 / 0.9 (paper defaults) | 25 | 58 | 90.04% | −3.31 |
| `nesterov` 0.7 / 0.9 (paper defaults) | 100 | 15 | 85.70% | −7.65 |

The two FedAvg rows were then reproduced on Snellius, `k=2` x 2 H100s at the
same replica batch of 512 and the same `T` (`--batch-size 256`, since 2 ranks
per replica there against LUMI's 4):

| | `H` | LUMI (8 GCDs) | Snellius (4 H100) | Δ |
|---|---|---|---|---|
| FederatedAveraging | 25 | 92.41% | **92.56%** | +0.15 |
| FederatedAveraging | 100 | 92.17% | **92.20%** | +0.03 |
| plain DDP, same global batch | -- | 93.35% | 93.55% | +0.20 |

Landing within 0.15 points on entirely different hardware, vendor stacks and
rank-to-replica layouts is the same result as the DDP cross-check: it says the
outer loop is correct, not merely running. `agreement` decayed 0.89 → 0.76 over
the H=100 run against a 1/√2 = 0.707 floor, so H=100 is close to the edge on
this model at this LR -- the same shape LUMI shows. The paper's Nesterov rows
were not re-run there; nothing suggests they would behave differently, since the
mechanism is BatchNorm, not hardware.

**FederatedAveraging at `H`=100 gives up 1.2 points of accuracy for a 100x
reduction in inter-replica communication, at identical throughput** (38,475 vs
38,400 img/s). That is DiLoCo working as advertised.

The paper's Nesterov defaults cost 6.5 points more than FedAvg here, and the
mechanism is visible in the logs. `θ⁽ᵗ⁾ = θ⁽ᵗ⁻¹⁾ − lr(1+momentum)·Δ` overshoots
*past* every replica's actual weights, so `θ` lands outside their convex hull —
whereas FedAvg's `mean_i θᵢ` lands inside it. BatchNorm running statistics were
collected at the replicas' weights, so they remain valid for an interpolation and
do not for an extrapolation. Hence eval losses of 223 and 32832 early on, decaying
back to sane values in exact step with `‖Δ‖` (18.5 → 0.38), while *training*
accuracy stayed healthy at 92%. It is not purely an eval artifact either — train
accuracy visibly regresses across an outer step (77.4% → 51.2% at outer step 5),
so the extrapolated `θ` really is a worse model, not just a badly-measured one.

Both knobs matter and the outer optimizer dominates. `T` matters only for the
momentum variants — halving `H` bought Nesterov 4.3 points and FedAvg 0.24, which
is what you would expect from an optimizer that has no momentum buffer to warm up.

None of this contradicts the paper: their models use LayerNorm, which keeps no
running statistics to invalidate. **`configs/cifar10_resnet18_diloco.yaml`
therefore ships FedAvg**, while the code default in `diloco.py` stays at the
paper's Nesterov — revisit that for the LLM phase, where the interaction goes away.

**One more thing to know:** `finish()` flushes a trailing partial inner phase, and
with momentum that step is mis-scaled — the buffer holds momentum accumulated from
full-length `H`-step deltas while `Δ` covers only the short remainder. Measured at
the last epoch: Nesterov lost 1.81 and 0.90 points to the flush, FedAvg gained
0.13 and 0.14. Either make the step count a multiple of `H`, or take the
second-to-last checkpoint, or use FedAvg.

### What is verified

| | |
|---|---|
| outer-step arithmetic, layout, state round-trip | `tests/test_local.py`, 27 checks, single process |
| group membership, cross-replica averaging, `θ(0)` alignment, DDP-in-replica reconvergence | `tests/test_diloco_gloo.py`, real process groups over gloo on CPU |
| the same on GPU with RCCL | **verified** on one LUMI node, `k=2 x 4` GCDs (job 20646477): group membership correct, outer step exactly `0.5000` |
| the same on GPU with NCCL | **verified** on one Snellius node, `k=2 x 2` and `k=4 x 1` H100s: group membership correct, outer step exactly `0.5000` and `1.5000` |
| **convergence** | **measured on both sites** — 92.41% / 92.56% at H=25 and 92.17% / 92.20% at H=100 (LUMI / Snellius), against 93.35% / 93.55% for plain DDP. See the tables above |
| DiLoCo + FSDP | code path exists (`--parallel fsdp`) and is sharding-aware, but **untested**; CIFAR is too small to shard meaningfully |

Note also that `k` replicas are only usefully *independent* while they share one
Slurm allocation here. Running one replica per cluster across LUMI and Snellius
would need an outer-gradient transport that does not exist yet: there is no
shared filesystem between the two sites and compute nodes reach the outside world
only through a slow proxy, so it would need a staging host both sides can reach.
[TODO.md](TODO.md) plans exactly that, using consolidated checkpoints as the
transport rather than a collective — which is the same outer step with a different
carrier, since `Δ` is just a difference of two checkpoints.
`diloco.py` is structured so that only the two `dist.all_reduce` calls in
`outer_step` would have to change.

## Dynamic dataset leasing (DARL)

DiLoCo says how the replicas exchange weights. It says nothing about how they
agree on *who trains on what*, and once the replicas are separate clusters that
question stops being trivial: each one starts when its own batch queue lets it,
runs at its own speed, and can be killed at walltime mid-step. The requirement is
still that the corpus is covered exactly once -- no sample twice, no sample
missed -- because a duplicate is invisible in the loss curve and a gap is
invisible in everything.

Splitting the corpus m ways in advance cannot deliver that. The fast cluster
drains its shard and idles; the slow one never finishes; a cluster that dies takes
its whole shard out of the epoch. Handing out work dynamically without a protocol
is worse -- two clusters race and silently train on the same samples.

So `src/pww/darl/` leases the index space, the way CockroachDB leases key ranges
and Flink assigns key-groups:

| | |
|---|---|
| unit | a **block** of `K` consecutive samples; `M = ceil(N/K)` blocks |
| order | blocks are permuted once from the seed, so a lease is a random sample of the corpus, not a run of consecutive text |
| lease | a contiguous run of positions, granted to one cluster under a heartbeat TTL |
| states | `UNASSIGNED` → `LEASED` → `COMMITTED`, plus `QUARANTINED` for a block that keeps failing |
| expiry | heartbeats stop → the uncommitted tail returns to the pool, no operator involved |
| stealing | an idle cluster takes the *unstarted* tail of whoever will finish last |
| commit | means "durably processed", i.e. covered by a checkpoint -- not "consumed" |

Three invariants, and `LeaseTable.verify()` asserts all three on every snapshot:
the union of committed blocks is the whole space (completeness), no two clusters
ever hold the same block (disjointness), and the per-cluster committed counts sum
to `M` (zero duplication).

### Running it

The coordinator is one small Python process on a login node -- not a batch job,
because it has to outlive every job that leases from it:

```bash
./scripts/darl_coordinator.sh start --num-samples 1000000 --block-size 10000
./scripts/darl_coordinator.sh status         # progress, per-cluster throughput
./scripts/darl_coordinator.sh url            # give this to the other site
```

Validate before spending queue time; both of these run on a login node:

```bash
source env.sh && pww_run python3 tests/test_darl.py        # must be 36/36
pww_run python3 -m pww.darl.simulate --clusters 4 --kill 1 --late 1
```

`simulate` is the one worth reading the output of. It starts a real coordinator
and four **separate processes** running the real client over real HTTP, gives them
throughputs an 8x spread apart, delays one behind a simulated queue, kills one
mid-lease without releasing anything, and then audits coverage from what the
processes themselves report rather than from the coordinator's own books:

```
cluster       committed  leased   lost    blk/s    (1,000 blocks, one epoch)
sim-0                18       0     12   31.469    <- crashed holding 12 blocks
sim-1               281       0      0    7.767
sim-2               141       0      0    3.942    <- slowest
sim-3               560       0      0   15.430    <- fastest, took 4x sim-2's share
blocks covered   1,000 of 1,000     duplicates 0     missing 0
PASS: coverage is exactly once
```

The crashed cluster's 12 blocks were reclaimed on TTL expiry and finished by
someone else, and the fastest cluster did four times the slowest one's work
without anyone configuring a ratio. That is the whole argument for leasing over
static partitioning, on one page.

### Granularity, and why it costs nothing

Lease one span per DiLoCo local phase (`blocks_for_phase(space, inner_steps=H,
batch_size=..., ranks=...)`, which is what `--darl-blocks-per-phase 0` derives).
Two consequences:

- **No coordinator traffic during the inner loop.** The only moment a lease
  boundary is observable is the outer step, where the ranks are synchronised
  anyway. The run above served 277 requests to cover 1,000 blocks across 4
  clusters -- for an LLM, where a phase is minutes, that is a handful of RPCs an
  hour and WAN latency is irrelevant.
- **One process per stream talks to the coordinator, not one per rank.** The
  stream leader acquires and broadcasts three integers per span over the process
  group the ranks already have; every rank then derives its own sample list from
  the shared seed. A 512-rank job makes the same number of RPCs as a 4-rank one.

TTL follows the design note, `Δt_TTL = α·t̄_step + β·RTT` with α=2.5, β=10, from
the client's own measured phase duration and RTT -- so a cluster that gets slower
asks for a longer lease instead of being declared dead mid-phase.

### Wiring it into a trainer

`train_cifar.py` does **not** use this yet (CIFAR is 50k samples on local disk, so
there is nothing to lease). For an LLM entrypoint it is four calls:

```python
space   = BlockSpace(num_samples=N, block_size=args.darl_block_size, seed=args.seed)
session = session_for_replica(space, replicas, url=args.darl_url, site=PWW_SITE,
                              batch_size=args.batch_size,
                              inner_steps=args.diloco_inner_steps)   # None off rank 0
data    = DARLDataSource.for_diloco(space, session, replicas, seed=args.seed)

while (phase := data.next_phase()) is not None:      # collective: every rank, same count
    sampler.set_indices(phase.indices)
    for batch in loader:
        ...                                          # inner steps as usual
    data.end_phase()                                 # feeds the TTL estimate
    save_checkpoint(...)                             # then, and only then:
    data.commit()                                    # blocks are now COMMITTED
```

The order of the last two lines is the exactly-once guarantee. Commit before the
checkpoint lands and a crash in between drops those samples from the epoch; commit
after and a crash re-runs them, which is correct, because the model rolled back
too.

### What is verified

| | |
|---|---|
| state machine -- expiry, stealing, quarantine, grant sizing, snapshot/restore, journal replay | `tests/test_darl.py`, 36 checks, injected clock |
| disjointness under real concurrency | same file: a real coordinator on a socket, 4 client threads, 400 blocks, 0 duplicates |
| exactly-once across processes, with a crash and a late joiner | `pww.darl.simulate`, output above -- **verified on a Snellius login node** |
| coordinator restart mid-epoch | snapshot + journal replay is unit-tested; a *live* restart under load is not |
| use in a real training loop | **not done** -- `torch_data.py` has its sharding maths tested with the broadcast stubbed, but no GPU job has consumed a lease yet |
| cross-site reachability | **not tested.** Compute nodes reach their own login nodes; whether a Snellius compute node can reach a LUMI login node is the open question in [TODO.md](TODO.md), and `scripts/darl_coordinator.sh` documents the `curl /health` check and the SSH-tunnel fallback |

### Things that are easy to get wrong here

**The heartbeat reply is an instruction, not an acknowledgement.** It carries the
coordinator's `end` for each lease, which may have shrunk because the tail was
stolen, and `valid: false` for a lease that was reclaimed. A client that trains to
its own remembered `end` breaks disjointness. `LeaseSession` applies the reply
before any more samples are drawn; anything built on `LeaseClient` directly has to
do the same.

**Every site must derive the same permutation.** It comes from the seed alone so
that no communication is needed, which means a Python or library skew between two
sites would hand two clusters the same samples under different position numbers.
Hence `BlockSpace.digest()` and the check at registration -- a mismatch is a
startup error, not a data-quality mystery three weeks later.

**One session per *stream*, not per job.** Under DiLoCo a stream is one replica:
k replicas in one allocation are k independent models and must see disjoint data.
`for_diloco` registers each replica as its own cluster.

**`persistent_workers=False` with `LeasedSampler`.** Persistent workers keep a
forked copy of the sampler from the first iteration and would replay the first
phase forever. Re-forking costs milliseconds against a phase of hundreds of steps.

**A single coordinator is a single point of failure.** While it is down, clusters
keep training on the span they hold (their leases are extended by
`--restore-grace` on restart) but cannot acquire. Replicating the state machine is
the one thing etcd would add, and the journal is exactly what Raft would
replicate; `Coordinator` is the only class that would have to change.

**Quarantine trades exactness for termination.** A block reclaimed
`--max-attempts` times is dropped from the epoch so a corrupt shard cannot hang it
forever. The status report and the completion message both say so loudly. Set
`--max-attempts 0` if a missing block is worse than a stalled epoch.

## Things that are easy to get wrong here

**One node = 8 ranks, not 4.** A LUMI-G node has 4 MI250X cards, but each is two
GCDs and ROCm sees each GCD as a separate GPU. Always
`--ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=7`.

**CPU binding is not optional.** `$PWW_CPU_BIND` maps each rank to the 7 cores
physically nearest its GCD. Dropping it, or getting the mask order wrong, costs
10-30% throughput with no error message. 56 of the 64 cores are usable; one core
per L3 group is reserved for the OS.

**`--mem=0`.** Requests all node memory. The default per-node share is far too
small for 8 ranks with dataloader workers.

**Download on a login node.** Compute nodes reach the internet only via a slow
proxy. `download=False` everywhere in job code.

**MIOpen cache.** Defaults to `$HOME` and is not multi-process safe -- concurrent
ranks corrupt it and die on SQLite errors. `task_wrapper.sh` gives each rank a
node-local cache. The first few steps of any new shape are slow (seconds) while
MIOpen autotunes kernels; this is normal and cached afterwards.

**Partitions**, and the choice is worth two minutes because `small-g` is a trap for a
full-node job. Measured 2026-08-12:

| | limit | nodes (alloc/idle/total) | allocation |
|---|---|---|---|
| `dev-g` | 3 h | 15 / 34 / 49 | shareable |
| `small-g` | 72 h | 195 / **1** / 199 | shareable, billed per GCD |
| `standard-g` | **48 h** | 2663 / 25 / 2728 | whole nodes, billed per node. **`MinNodes=1`** |

`standard-g` is **not** multi-node only — an earlier version of this line said so, and it
is what sends full-node jobs to the wrong queue. Since the titan job scripts ask for
`--gpus-per-node=8 --cpus-per-task=56 --mem=480G`, i.e. an entire LUMI-G node, `small-g`'s
node sharing is a benefit they never use — while `small-g` has ~200 nodes against
`standard-g`'s ~2,700 and routinely shows **one** idle. Observed cost of that mistake: a
10-hour queue wait for a job that `dev-g` would have started immediately.

So: `dev-g` for anything under 3 h (the 1k rehearsal fits — LUMI runs ~3 s/step at 8 GCDs,
so 1,000 steps is under an hour), `standard-g` for real runs up to 48 h, and `small-g` only
when a job genuinely needs **more than 48 h** or genuinely wants a partial node.

Check before waiting on a queue, and override the partition on any script rather than
editing it:

```bash
sinfo -p dev-g,small-g,standard-g -o "%.12P %.15l %.14F"   # A/Idle/Other/Total
sbatch -p dev-g scripts/lumi/job_cifar_1node.sh --config configs/cifar10_resnet18.yaml
```

**The CPU mask needs a full node.** `pww_cpu_bind` returns the optimal 8-rank
mask only when `--ntasks-per-node` equals 8, and falls back to `--cpu-bind=cores`
otherwise. Forcing the fixed mask on a partial allocation fails hard with
`CPU binding outside of job step allocation`. Affinity is therefore not optimal
on partial-node debug runs -- use a full node for anything you intend to measure.

**`MASTER_PORT` must not be a constant.** Every job script derives it from
`SLURM_JOB_ID` rather than pinning 29500, because two of your own jobs can land
on the same node -- routine on Snellius, where single-node jobs share a node by
default, and possible on LUMI's `small-g`. A fixed port makes the second job die
in the TCPStore rendezvous with "address already in use", which reads like a
network fault rather than a scheduling one.

**bf16, not fp16.** MI250X does bf16 natively and it has fp32's dynamic range, so
no loss scaler is needed. `--dtype fp16` is accepted by the parser but there is
**no `GradScaler` anywhere** -- neither the autocast path in `train_cifar.py` nor
FSDP's `MixedPrecision` scales gradients -- so fp16 training will quietly
underflow. It is untested on both sites. Use bf16, on H100 as well as on MI250X,
until someone wires a scaler in.

**DiLoCo: every rank must reach the outer step the same number of times.** The
outer step is a collective, triggered by a per-rank counter. `drop_last=True` on
the training sampler is therefore load-bearing: a ragged final batch makes ranks
disagree on the step count and the job hangs in the all-reduce rather than
failing. The same applies to any `continue`/`break` added to the inner loop.

**DiLoCo: `k` replicas must start from one `θ(0)`.** `set_seed` offsets the seed
by rank, so each rank builds a *different* random model, and replica-scoped DDP
only equalises within a replica. `DiLoCo.__init__` broadcasts `θ(0)` for exactly
this reason. Averaging deltas between models that were never the same model
trains to nothing while looking perfectly healthy.

## Running on Snellius

```bash
git clone <this repo> && cd ProjectWorldWide
./scripts/snellius/setup_venv.sh   # ONCE, from a login node. ~6 GB, a few minutes.
./scripts/bootstrap.sh
./scripts/download_data.sh         # once, from a login node

# validate on the login node first -- no allocation needed
source env.sh && pww_run python3 tests/test_local.py       # must be 28/28
source env.sh && pww_run python3 tests/test_diloco_gloo.py # must be 14/14

sbatch scripts/snellius/job_smoke.sh                       # must print SMOKE TEST PASSED
sbatch scripts/snellius/job_smoke.sh --diloco-replicas 2   # before any DiLoCo job
sbatch scripts/snellius/job_cifar_1node.sh --config configs/cifar10_resnet18.yaml
sbatch scripts/snellius/job_cifar_multinode.sh
sbatch scripts/snellius/job_cifar_diloco.sh
```

Budget a few minutes for the two CPU test suites: the Snellius login nodes are
heavily shared, and one ResNet-18 CPU step there has been measured at 68 s, which
makes `test_local.py` take around 7 minutes rather than the seconds it takes on an
idle machine. That is contention, not a hang.

`./scripts/siteinfo.sh` re-derives every machine-specific value in
`sites/snellius.sh` from the running system, which is how the values there were
obtained. Run it if Snellius changes underneath you.

Verified on Snellius, all on `gpu_h100` unless noted:

| check | result |
|---|---|
| `tests/test_local.py` | 28/28 pass |
| `tests/test_diloco_gloo.py` | 14/14 pass across k=2 and k=4 |
| `job_cifar_debug.sh` (1 GPU) | passes in 21 s |
| `job_smoke.sh` (1 node, 4 GPUs) | passes, 300.8 GB/s all-reduce |
| `job_smoke.sh --nodes=2` (8 GPUs) | passes, 133.1 GB/s all-reduce over InfiniBand |
| `job_smoke.sh --diloco-replicas 2` | passes, outer step exactly `0.5000`, 1.9 ms vs a 7.2 ms inner step |
| `job_smoke.sh --diloco-replicas 4` | passes, outer step exactly `1.5000` |
| `job_cifar_1node.sh` (DDP, fp32) | 93.55% eval acc, 56,184 img/s |
| `job_cifar_1node.sh --dtype bf16` | 93.35%, 67,776 img/s |
| `--parallel fsdp --dtype bf16`, 30 ep | 92.96%, 64,802 img/s |
| `job_cifar_multinode.sh` (2 nodes, 8 GPUs) | 93.20%, 85,220 img/s |
| `job_cifar_diloco.sh` (k=2, H=100) | 92.20% -- see the DiLoCo section |
| consolidated resume, 4 ranks -> 1 | works |
| `--sharded-checkpoint`, 4 ranks -> 2 | works |
| `gpu_a100` | **not yet run** -- the partition sits at 0 idle nodes |

The FSDP row is the one that matters for the LLM phase: it exercises
`init_device_mesh`, `FSDP(device_mesh=)` and
`torch.distributed.checkpoint.state_dict` -- three of the four APIs the
Snellius PyTorch module is missing, and the reason the venv exists.

The two resume rows are the ones that matter for federated training: a
checkpoint written by 4 ranks reloads onto a different world size, in both
formats. That is the claim [TODO.md](TODO.md) rests on, and it is now tested
rather than assumed.

### The environment is a venv, and it is not optional

This is the one real structural difference between the two sites. LUMI has a
maintained container with the whole AI stack in it. Snellius has no equivalent,
and its module tree cannot run this code at all:

- the **2023** tree has `PyTorch/2.1.2-foss-2023a-CUDA-12.1.1`, which predates
  every distributed API this codebase uses -- no
  `torch.distributed.device_mesh`, no `torch.distributed.checkpoint.state_dict`,
  no `device_id=` on `init_process_group`, no `device_mesh=` on FSDP. So
  `parallel.py` and all of `checkpoint.py` fail on import or first call.
- the **2024** tree's `ai/PyTorch` directory is **empty**.
- the **2025** tree has no `ai` modules at all.

`scripts/snellius/setup_venv.sh` therefore builds the environment itself: a venv
on `Python/3.12.3-GCCcore-13.3.0` with torch pinned to **2.7.1**, the same
version as the LUMI container. That pin is the point -- the two sites running
different torch generations would undermine the whole comparison. The pip wheels
bundle their own CUDA runtime, so nothing depends on the EasyBuild CUDA version.

It lives in `$HOME/venvs/pww-snellius` (override with `PWW_VENV`), deliberately
not on `/scratch-shared`, which is purged on file age -- an environment that
dissolves after two idle weeks is worse than no environment.

Every package in it is pinned to the LUMI container's version, `accelerate`
included -- left unpinned that one resolves to 1.x against LUMI's 0.34.2, which
is exactly the silent divergence the pinning exists to prevent. If your venv was
built before that pin landed, `pip install accelerate==0.34.2` inside it to
align.

`flash-attn` is the one thing the LUMI container has that this venv does not; it
has no prebuilt wheel for this combination and compiling it takes upwards of an
hour. Add it when the LLM phase actually needs it.

### Partitions, cost and queueing

| partition | nodes | GPUs/node | cores | RAM GiB | SBU/GPU-h |
|---|---|---|---|---|---|
| `gpu_h100` | 88 | 4x H100 | 64 (4x16) | 720 | 192 |
| `gpu_a100` | 63 | 4x A100 | 72 (4x18) | 480 | 128 |
| `gpu_vis` | 63 | 4x A100, 24 h cap | 72 | 480 | 128 |
| `gpu_mig` | 4 | 8x A100 MIG slice | 72 (8x9) | 480 | 64 |

There is **no partition called `gpu`** -- that name was a guess in the original
template and jobs using it are rejected at submit time.

The scripts default to `gpu_h100` at 16 cores per rank. For A100, override both
together, since it has 72 cores rather than 64:

```bash
sbatch -p gpu_a100 --cpus-per-task=18 scripts/snellius/job_cifar_1node.sh
```

Both GPU partitions routinely sit at **zero idle nodes**, so expect to queue.
Jobs asking for **1 hour or less** of walltime are routed to a reserved
short-job pool and start much sooner -- keep debug runs at or under
`--time=01:00:00`. H100 also bills 1.5x A100 per GPU-hour, so a run that is not
H100-bound is cheaper on `gpu_a100`.

**The lever that decides your queue time is GPUs per job, not cores.** A
single-node job may take a fraction of a node, and the smallest unit is 1 GPU +
16 cores + 180 GiB. That fraction is nearly always free even when the cluster is
fully allocated -- measured on a completely full `gpu_h100`:

| job shape | nodes that could start it immediately |
|---|---|
| 1 GPU + 16 cores | 23 of 88 |
| 2 GPUs + 32 cores | 6 of 88 |
| 4 GPUs + 64 cores | **0 of 88** |

So `scripts/snellius/job_cifar_debug.sh` (1 GPU) starts in seconds while a
full-node job waits over an hour. Use it for "does this change run at all".

Two consequences worth internalising:

- **Trimming `--cpus-per-task` on a 4-GPU job saves nothing.** Billing is
  `max(cpu, gpu, mem)` fraction, and 4 of 4 GPUs is already the whole node. You
  would leave cores idle on a node you have fully paid for, and starve the
  dataloader for no gain. The 64 cores come with the 4 GPUs; they are not
  separately priced.
- **There is no cheap multi-node run.** Snellius rejects a multi-node GPU job
  that asks for fewer than all 4 GPUs per node:
  *"You've requested less than the maximum amount of GPUs for your multi-node
  job. If that's intentional, use `--exclusive`. You will be charged for all
  GPUs, including the ones that you don't use."*
  Multi-node means whole nodes, and `--exclusive` bills you for them anyway.

### Other things that differ from LUMI

- **No `--mem=0`.** Snellius allocates memory in proportion to the GPUs
  requested (180 GiB per H100, 120 GiB per A100). Asking for all node memory is
  neither necessary nor accepted the way it is on LUMI.
- **CPU binding is simpler.** Both node types are 4 sockets with one GPU on
  each, so `--cpus-per-task=<cores per socket>` plus `--distribution=block:block`
  and `--cpu-bind=cores` gets the NUMA-correct placement. No hand-written mask,
  unlike LUMI where the GCD-to-core map is not something SLURM can infer.
- **`/scratch-shared` is purged at ~14 days** on file age. Fine for CIFAR and
  for run output you will read this week; point `PWW_SCRATCH` at a `/projects`
  space for anything you intend to keep.
- **Comparing results between sites.** A Snellius node has 4 ranks where LUMI
  has 8, so the same `--batch-size` gives half the global batch and a different
  scaled LR. Match the *global* batch when comparing, not the per-rank one.

The smoke test is the gate: it verifies each rank gets a distinct GPU, that
all-reduce is numerically correct, and that collective bandwidth is plausible.
Measured, 256 MiB all-reduce:

| | LUMI | Snellius (H100) |
|---|---|---|
| within a node | 123 GB/s | **300.8 GB/s** |
| across two nodes | 88 GB/s | **133.1 GB/s** |

**Single-digit GB/s across nodes means NCCL fell back to TCP** instead of
InfiniBand -- uncomment `NCCL_SOCKET_IFNAME` in the job script (the interfaces
here are `ibp*`/`mlx5`, not `ib0`). That did not happen on either measurement
above: NCCL found InfiniBand unaided, so the variable stays commented out.

### Comparing a Snellius run against a LUMI one

A Snellius node has 4 ranks where LUMI has 8, so the same `--batch-size` gives
half the global batch and a different scaled LR. Match the *global* batch, not the
per-rank one.

Under DiLoCo match the *replica* batch and `k` separately, because the LR is
scaled to the replica: `k=2` gives 2 ranks per replica here against 4 on LUMI, so
`--batch-size` has to double to keep the replica batch equal. `k` must also divide
the rank count, so a 4-GPU node allows `k` of 1, 2 or 4 — `k=8` is a LUMI-only
layout.

## Extending to LLM Training

Two client paths share the same central node. **Prefer torchtitan for new work** — it is
the only one that scales past ~1B parameters. To run it: **[RUNBOOK.md](RUNBOOK.md)** for
the steps, [FEDERATION_GUIDE.md](FEDERATION_GUIDE.md) for what the knobs mean,
[scripts/titan/README.md](scripts/titan/README.md) for the environment.

| | inner loop | model | config | torch |
|---|---|---|---|---|
| **torchtitan** | `pww.titan.train` | Qwen3, FSDP2, C4, OpenEuroLLM 128k | `configs/titan/*.toml` | >= 2.9 |
| **legacy HF** | `pww.train_llm_flower` | GPT-2 / LLaMA via `AutoModel` | `configs/llm_*.yaml` | 2.7.1 |

The rest of this section documents the legacy path, which still runs unchanged.

### System architecture

```
                +----------------------------------------------------------+
                |                 CENTRAL ORCHESTRATOR NODE                |
                |                (Ubuntu 24.04 @ 145.38.206.143)           |
                |  +---------------+  +----------------+  +-------------+  |
                |  | DARL          |  | Flower server  |  | Blob store  |  |
                |  | coordinator   |  | PWWFedMom      |  | (>1B only)  |  |
                |  | HTTP 29510    |  | gRPC 29511      |  | HTTP 29512  |  |
                |  +-------+-------+  +--------^-------+  +------^------+  |
                |          |                   |                 |         |
                |   durable global model, momentum buffer, membership      |
                |          |    (--state-dir: survives a restart)          |
                +----------|-------------------|-----------------|---------+
                           | lease spans       | deltas          | weights
              +------------+---------+         |                 |
              |                      |         |                 |
  +-----------v----------+  +--------v---------+-----------------+--+
  |   SNELLIUS CLUSTER   |  |            LUMI CLUSTER              |
  |  NVIDIA H100 (SURF)  |  |       AMD MI250X (EuroHPC)           |
  |  4 ranks/node        |  |       8 ranks/node (2 GCDs each)     |
  +----------------------+  +--------------------------------------+
```

Membership is **elastic**: zero live replicas (every site queued), one, or several
are all normal states, and the run survives a restart of the aggregator itself.

Weights move either inline in the gRPC message or out of band through the blob store.
The gRPC cap is 2,147,483,647 bytes exactly — a protocol limit, not a setting — which
puts the inline ceiling at 1,073,741,823 parameters in float16. Measured against the
Qwen3 flavors this repo ships configs for, with the 128k tokenizer's vocabulary padded
to 131,328 rows:

| flavor | actual parameters | fp16 wire | transport |
|---|---|---|---|
| 0.6B | 709,427,200 | 1.3 GiB | inline (float16 only — float32 is 2.6 GiB, over the cap) |
| 1.7B | 1,947,329,536 | 3.6 GiB | blob |
| 8B | 8,021,914,624 | 14.9 GiB | blob |
| 32B | 32,551,097,344 | 60.6 GiB | blob |

The names understate the sizes: at the 0.6B flavor the embedding and output projection
are ~38% of the model, because a 131,328-row vocabulary is far larger than Qwen's own
defaults assume. Blob transport streams one tensor at a time, so peak memory tracks
the largest tensor and stops growing once the embedding dimension does — 14B and 32B
have identical merge peaks (~10 GiB, against 271 GiB and 606 GiB held densely). Full
table in [FEDERATION_GUIDE.md](FEDERATION_GUIDE.md).

---

### How Single-Pass LLM Dataset Partitioning Works (DARL)

1. **Continuous 1D Token Stream**: The pre-tokenized corpus (e.g. FineWeb, C4, RedPajama) is sliced into fixed-length sequence windows (`seq_len=1024/2048/4096`).
2. **Dynamic 1-Epoch Leasing (`DARL_EPOCHS=1`)**: The dataset is divided into DARL token blocks (e.g., 100,000 tokens per block). All blocks start in the pool as `UNASSIGNED`.
3. **Zero Sample Duplication**: Before each inner phase, Snellius and LUMI lease fresh token block spans via lightweight 50-byte HTTP `/acquire` calls. Snellius and LUMI train on completely disjoint token blocks.
4. **Decoupled Rounds & Epochs**: A single 1-epoch corpus supports hundreds or thousands of Flower outer aggregation rounds ($H$ inner steps per round) without repeating a single token.

#### Note on tokenizers
This path loads tokenizers from HuggingFace via `AutoTokenizer.from_pretrained()` and
tokenises on the fly in [src/pww/data/text.py](src/pww/data/text.py). There is no
tokenizer-training utility in the repo; earlier revisions of this file described a
`src/pww/data/tokenizer.py` that was never written.

The torchtitan path works differently and deliberately so: tokenisation is a
**one-off offline pass** ([scripts/titan/tokenize_c4.sh](scripts/titan/tokenize_c4.sh))
that writes fixed-width memmap shards plus a manifest, and DARL leases over window
indices into those shards. That is what makes a lease mean the same thing at both
sites — the manifest digest covers the tokenizer hash and `seq_len`, so a
disagreement is refused at registration rather than silently producing two different
partitions of "the corpus".

---

### Inner Loop Optimization & Parallelism Stack

Inside each HPC cluster, the inner training loop is managed by HuggingFace `Trainer` + PyTorch FSDP:

* **Optimizer**: `AdamW` with cosine decay and `weight_decay=0.01/0.1`.
* **Gradient Accumulation & Clipping**: Micro-batch size accumulation with `max_grad_norm=1.0` gradient norm clipping.
* **Mixed Precision (`bf16`)**: Native bfloat16 computation on H100 and MI250X tensor cores.
* **FSDP (Fully Sharded Data Parallel)**: Shards model parameters, gradients, and optimizer states across all node GPUs, enabling 1B to 7B+ model training without VRAM OOM.
* **FlashAttention-2 / PyTorch SDPA**: Native scaled dot-product attention for high throughput.

---

### Quick Start Commands

#### Step 1: Launch Central Daemons
```bash
./scripts/central_node/start_central_services.sh
```

#### Step 2: Submit Slurm Jobs

* **Fast Integration Test (GPT-2 124M on WikiText-2)**:
  ```bash
  # On Snellius
  sbatch scripts/snellius/job_flower_diloco_llm.sh

  # On LUMI
  sbatch scripts/lumi/job_flower_diloco_llm.sh
  ```

* **Scaling Run (1B - 7B LLaMA-3 / Qwen-2.5)**:
  ```bash
  # On Snellius
  sbatch scripts/snellius/job_flower_diloco_llm.sh --config configs/llm_7b_diloco.yaml

  # On LUMI
  sbatch scripts/lumi/job_flower_diloco_llm.sh --config configs/llm_7b_diloco.yaml
  ```

#### Step 3: The baseline it is measured against

Needs no central VM — the job brings up its own throwaway DARL coordinator on the same
`space_seed = 42` block space:

```bash
sbatch scripts/snellius/job_titan_central.sh                    # step-matched
PWW_GLOBAL_BATCH=96 sbatch --export=ALL,PWW_GLOBAL_BATCH \
  scripts/snellius/job_titan_central.sh                         # compute-matched (either site)
```

#### Starting over

Both the DiLoCo job and the baseline resume from their config's `dump_folder`; neither is
stateless, and the baseline is the one that looks it. Clear both halves, or neither:

```bash
./scripts/reset_run.sh --dry-run              # what would go, and how big it is
./scripts/reset_run.sh                        # site half: checkpoints, staging, tb
./scripts/reset_run.sh --central              # + lease table and global model (on the VM)

# and at submit time, so nothing written in between is picked up
PWW_FRESH_RUN=1 sbatch --export=ALL,PWW_FRESH_RUN scripts/snellius/job_titan_central.sh
```

---

### Monitoring & Evaluation Metrics

Stream outer round metrics on the Central VM:
```bash
tail -f runs/central/flower.log
./scripts/central_node/status_central_services.sh   # merge round + per-cluster membership
```

Log outputs report:
* **`merge round`** — the count of *successful merges*, not Flower's round number. A
  round in which every site was killed at walltime consumes a round number and
  changes nothing.
* **Aggregated training loss** weighted by tokens, with perplexity alongside it.
  Perplexity is reported under its own name; an earlier version reported it in a
  field labelled `accuracy`.
* **Tokens per round**, which is 0 for a round that trained nothing and is not
  merged. Worth watching: an earlier WikiText run reported `loss 0.0` on 1 sample
  for 23 consecutive rounds, because a `max(1, ...)` floor turned "the corpus is
  exhausted" into "one sample" and the global model sat frozen while the log showed
  no failures.
* **`drift`** — `||local − global|| / ||global||` per round, as mean and **max**. The one
  number that says whether `H` was chosen sensibly, and it is the max that matters:
  two sites at 0.01 and 0.30 average to a reassuring 0.155.
* **Throughput and hardware, per cluster** — tokens/s, MFU, TFLOP/s per rank and peak
  memory, plus the round's wall time (the slowest site's inner phase, since everyone waits
  for the straggler before the merge). tokens/s is *summed* across sites because they train
  concurrently; MFU and memory are deliberately **not** averaged, because MFU is a ratio to
  a device's peak FLOPs and an MI250X GCD and an H100 have different peaks — a mean of the
  two would describe neither.
* **A dropped cluster** — a site whose weights contain nan/inf is named and excluded rather
  than averaged. One poisoned contribution used to propagate into the global model and end
  the run while the log still said `merge complete` every round.
* **The held-out figure is comparable across sites, not publishable.** It is measured on a
  staged slice of C4's real validation split when one is present (`run_train.sh` selects it
  automatically; see RUNBOOK Part 1 step 3), with the window *total* fixed so rank count
  cannot change what each site scores. Without the staged split it falls back — with a
  warning — to torchtitan's bundled fixture, which **overlaps the C4 training files** and
  biases a central-vs-DiLoCo comparison toward whichever arm trained more of the corpus.
  Judge progress from the training loss; use the held-out number to compare sites and arms.

#### The central baseline, and reading a chart with both runs on it

`scripts/{lumi,snellius}/job_titan_central.sh` runs the same config with the outer step
switched off (`flower.enable = false`), on a throwaway DARL coordinator seeded from the same
`space_seed = 42` block space — so the baseline draws from the identical permuted index space
the federated run does. `configs/titan/qwen3_0.6b_c4_central.toml` is its DiLoCo counterpart
with one line changed, and the two are meant to stay diffable.

**Equal steps is not equal work, and this is the thing to get right.** Both configs set
`steps = 20000`, so the baseline is *step-matched*. It is not token-matched, because the
federation trains every site concurrently — per global optimiser step,
`local_batch_size × seq_len × ranks`:

| | ranks | tokens/step | at 20,000 steps |
|---|---|---|---|
| baseline, Snellius | 4 | 65,536 | ~1.31 B |
| baseline, LUMI | 8 | 131,072 | ~2.62 B |
| **DiLoCo, both sites** | **12** | **196,608** | **~3.93 B** |

For the *compute-matched* baseline — same steps and same tokens per step, which is what
DiLoCo's own paper reports against — set `PWW_GLOBAL_BATCH=96` on **either** site:
`run_train.sh` derives the microbatch/accumulation split itself (Snellius 8×4×3, LUMI
6×8×2, both exactly 96) and marks the run name with `-gb96`. Both comparisons are worth
having and they answer different questions.

**WandB.** Off unless `PWW_WANDB=1`, `ENABLE_WANDB=1` or `WANDB_PROJECT` is set. Three runs
land in one project: `central-<site>`, `diloco-<site>` and `central-aggregator`. The default
x-axis is `train/cum_tokens` rather than the step, for the reason in the table above, and
`train/step` is logged alongside it. `train/*` and `eval/*` are produced by one shared
installer so a baseline and a DiLoCo run cannot compute them differently.

Two things that are easy to get wrong when reading these charts:

* **Compare a baseline against `central-aggregator`, not against a `diloco-<site>` run.** The
  aggregator's loss is token-weighted across every participant and its `train/cum_tokens` is
  the federation total; a site reports only its own share, so overlaying `diloco-snellius` on
  `central-snellius` by tokens makes the federated run look like it reached a loss on a third
  of the data it used.
* **There is no local-versus-global step ambiguity.** A site's `H` inner steps are a slice of
  the global count, not a separate clock: every site is realigned to the server-authoritative
  `pww_global_step` before each phase, and step 1500 means 1500 optimiser steps in all three
  runs. A `wandb step went from N to M` warning means a round was **not merged** and that
  site was pulled back — the guard keeps the data rather than letting WandB discard it.

Full key reference and axis semantics: [FEDERATION_GUIDE.md](FEDERATION_GUIDE.md) §5.

Three pieces of the metric arithmetic are non-obvious enough to be worth knowing, and
each is pinned by a test because all three produce plausible numbers when wrong:
perplexity is pooled through the **loss** rather than by averaging perplexities (`exp` is
convex, so averaging is always pessimistic — two sites at loss 2.0 and 4.0 report 31.0
against a true 20.1); accuracy deliberately *is* averaged, because it is linear; and
tokens and loss are all-reduced to **cluster** level, because `num_examples` is the
FedMom merge weight and a per-rank count would quietly turn token weighting back into
uniform `1/k`. See [FEDERATION_GUIDE.md](FEDERATION_GUIDE.md) §5.

See [FEDERATION_GUIDE.md](FEDERATION_GUIDE.md) §5 for annotated log excerpts and §6
for the failure table.

## Reading results

```bash
python3 -c "
import json, pathlib
for l in pathlib.Path('runs/<run>/metrics.jsonl').read_text().splitlines():
    r = json.loads(l)
    if r['split'] == 'eval': print(r['epoch'], round(r['acc'], 2))
"
```

A DiLoCo run adds `split == "diloco"` rows, one per outer step, carrying
`outer_step`, `delta_norm`, `avg_delta_norm` and `agreement`. Plot `agreement`
first — it is the one number that tells you whether `H` was chosen sensibly. Note
that `eval` rows repeat between outer steps, because they measure `θ`, which only
moves when an outer step happens.

Measured reference points, both verified, both ResNet-18 for 30 epochs on one
node at **the same global batch of 1024 and the same LR of 0.8** -- which is the
only way the two are comparable, since LUMI reaches 1024 as 8 x 128 and Snellius
as 4 x 256:

| | LUMI (8 GCDs, MI250X) | Snellius (4 GPUs, H100) |
|---|---|---|
| eval accuracy | 93.35% | **93.55%** |
| throughput | 38,400 img/s | **56,184 img/s** |
| per epoch | 1.3 s | 0.9 s |
| whole run | ~76 s | ~39 s |
| first epoch | ~1,500 img/s | 33,918 img/s |

The two accuracies landing within 0.2 points of each other, from the same source
at the same global batch on entirely different hardware and vendor stacks, is
the real result here -- it is what says the port is correct rather than merely
running.

Three caveats. LUMI's first epoch is ~25x slower than its steady state because
MIOpen autotunes kernels for shapes it has not seen; cuDNN does far less of
this, so Snellius only loses about a third on epoch one. Snellius run-to-run
variation of a few tenths of a point is normal, since nothing here is seeded
across sites. And 93-94% is below the ~94-95% usually quoted for ResNet-18 on
CIFAR-10 because those figures assume a much longer schedule at a smaller batch
-- 30 epochs at a global batch of 1024 is a deliberately short run. Raise
`epochs` if you want to close that gap.
