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
  lumi/                   job_smoke, job_cifar_debug, job_cifar_1node, job_cifar_multinode
  snellius/               setup_venv.sh (run first), job_smoke, job_cifar_debug,
                          job_cifar_1node, job_cifar_multinode
src/pww/
  distributed.py          process group, GPU pinning, collective reductions
  parallel.py             DDP / FSDP wrapping, mixed precision, act. checkpointing
  checkpoint.py           consolidated + sharded save/load, atomic writes
  config.py               YAML-under-argparse, seeding, run dirs
  logging_utils.py        rank-aware logging, JSONL metrics
  smoke.py                infrastructure self-test
  models/resnet.py        ResNet-18/34/50, CIFAR stem
  data/cifar.py           CIFAR-10 loaders with DistributedSampler
  train_cifar.py          training entrypoint
tests/test_local.py       CPU-only tests, no allocation needed
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
sbatch scripts/lumi/job_smoke.sh

# train
sbatch scripts/lumi/job_cifar_debug.sh                                     # 1 GCD, ~2 min
sbatch scripts/lumi/job_cifar_1node.sh --config configs/cifar10_resnet18.yaml
sbatch --nodes=4 scripts/lumi/job_cifar_multinode.sh
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
| `tests/test_local.py` | 17/17 | 17/17 |
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

## Extending to LLM training

The intent is that `train_cifar.py` is the only file replaced. What carries over
unchanged:

- `distributed.py` -- rank/device resolution is task-independent
- `parallel.py` -- `wrap_model(..., strategy="fsdp", transformer_layer_cls={YourDecoderLayer}, activation_checkpointing=True)`
- `checkpoint.py` -- switch to `--sharded-checkpoint`; it resumes onto a
  different world size, which matters when a run is requeued at a different scale
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
