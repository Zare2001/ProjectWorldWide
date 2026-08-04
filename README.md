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
tests/
  test_local.py           CPU-only, single process, no allocation needed
  test_diloco_gloo.py     CPU-only, multi-process gloo -- DiLoCo collectives
```

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
| `tests/test_local.py` (17 checks then, 27 now) | all pass | all pass |
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
| **convergence** | **measured** — 92.41% against 93.35% for plain DDP, one LUMI node, `k=2`. See the table above |
| DiLoCo + FSDP | code path exists (`--parallel fsdp`) and is sharding-aware, but **untested**; CIFAR is too small to shard meaningfully |

Note also that `k` replicas are only usefully *independent* while they share one
Slurm allocation here. Running one replica per cluster across LUMI and Snellius
would need an outer-gradient transport that does not exist yet: there is no
shared filesystem between the two sites and compute nodes reach the outside world
only through a slow proxy, so it would need a staging host both sides can reach.
`diloco.py` is structured so that only the two `dist.all_reduce` calls in
`outer_step` would have to change.

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

**Partitions.** `dev-g` for a 3 h debug turnaround, `small-g` billed per-GPU with
node sharing, `standard-g` billed per whole node (multi-node only). `small-g` is
popular and frequently has zero idle nodes; check before waiting on a queue, and
override the partition on any script rather than editing it:

```bash
sinfo -p dev-g,small-g,standard-g -o "%.12P %.15l %.14F"   # A/Idle/Other/Total
sbatch -p dev-g scripts/lumi/job_cifar_1node.sh --config configs/cifar10_resnet18.yaml
```

**The CPU mask needs a full node.** `pww_cpu_bind` returns the optimal 8-rank
mask only when `--ntasks-per-node` equals 8, and falls back to `--cpu-bind=cores`
otherwise. Forcing the fixed mask on a partial allocation fails hard with
`CPU binding outside of job step allocation`. Affinity is therefore not optimal
on partial-node debug runs -- use a full node for anything you intend to measure.

**bf16, not fp16.** MI250X does bf16 natively and it has fp32's dynamic range, so
no loss scaler is needed.

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

source env.sh && pww_run python3 tests/test_local.py    # must be 17/17
sbatch scripts/snellius/job_smoke.sh                    # must print SMOKE TEST PASSED
sbatch scripts/snellius/job_cifar_1node.sh --config configs/cifar10_resnet18.yaml
sbatch scripts/snellius/job_cifar_multinode.sh
```

`./scripts/siteinfo.sh` re-derives every machine-specific value in
`sites/snellius.sh` from the running system, which is how the values there were
obtained. Run it if Snellius changes underneath you.

Verified on Snellius, all on `gpu_h100`:

| check | result |
|---|---|
| `tests/test_local.py` | 17/17 |
| `job_cifar_debug.sh` (1 GPU) | passes in 21 s |
| `job_smoke.sh` (1 node, 4 GPUs) | passes, 300.8 GB/s all-reduce |
| `job_smoke.sh --nodes=2` (8 GPUs) | passes, 133.1 GB/s all-reduce over InfiniBand |
| `job_cifar_1node.sh` (DDP, fp32) | 93.55% eval acc, 56,184 img/s |
| `--parallel fsdp --dtype bf16` | trains and checkpoints, 62,672 img/s |

The FSDP row is the one that matters for the LLM phase: it exercises
`init_device_mesh`, `FSDP(device_mesh=)` and
`torch.distributed.checkpoint.state_dict` -- three of the four APIs the
Snellius PyTorch module is missing, and the reason the venv exists.

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

## Extending to LLM training

The intent is that `train_cifar.py` is the only file replaced. What carries over
unchanged:

- `distributed.py` -- rank/device resolution is task-independent
- `parallel.py` -- `wrap_model(..., strategy="fsdp", transformer_layer_cls={YourDecoderLayer}, activation_checkpointing=True)`
- `checkpoint.py` -- switch to `--sharded-checkpoint`; it resumes onto a
  different world size, which matters when a run is requeued at a different scale
- `diloco.py` -- nothing in it knows what is being trained. The `--diloco-*`
  flags live in `config.add_common_args`, so a new entrypoint gets them by
  calling `build_replicas` / `DiLoCo` and `diloco.inner_step()` after
  `optimizer.step()`. This is where DiLoCo starts paying off: `H` in the hundreds
  is a rounding error against an LLM step, and `--inner-optimizer adamw` matches
  the paper
- `config.py`, `logging_utils.py`, all `scripts/` and `env.sh`

What needs writing:

1. `data/tokenizer.py` -- train or load your tokenizer; write it once from rank 0
   (see `distributed.master_first`)
2. `data/text.py` -- tokenise the corpus to a flat `uint16`/`uint32` memmap of
   token ids, then serve fixed-length blocks. Do the tokenisation as a separate
   preprocessing job, not in the dataloader; and prefer one big memmap over many
   small files, because Lustre handles a few large files far better than many
   small ones.
3. `models/transformer.py` -- or a `transformers` config. Pass the decoder layer
   class to FSDP as the wrap unit; without it the whole model becomes one shard
   and sharding buys nothing.
4. `train_llm.py` -- same loop shape, but step-based rather than epoch-based, with
   gradient accumulation, grad clipping, AdamW, and tokens/s instead of images/s.

Expect to need, in roughly this order: `--dtype bf16`, `--parallel fsdp`,
activation checkpointing, gradient accumulation for the target global batch, then
sharded checkpoints. `parallel.build_mesh` is where tensor parallelism would go
if a model outgrows pure FSDP.

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
