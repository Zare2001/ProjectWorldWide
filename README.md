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
| accelerator | AMD MI250X, ROCm/RCCL | NVIDIA, CUDA/NCCL |
| ranks per node | 8 (each MI250X = 2 GCDs) | 4 |
| pinning variable | `ROCR_VISIBLE_DEVICES` | `CUDA_VISIBLE_DEVICES` |
| environment | prebuilt Singularity container | EasyBuild modules |
| interconnect | Slingshot (`hsn`) + aws-ofi-rccl | InfiniBand |
| MIOpen cache workaround | required | not applicable |
| status | **verified end to end** | **template, unverified** |

## Layout

```
env.sh                    site detection + shared config
sites/
  lumi.sh                 LUMI specifics (verified)
  snellius.sh             Snellius specifics (UNVERIFIED -- see below)
configs/                  YAML run configs
scripts/
  bootstrap.sh            one-time setup (run first)
  siteinfo.sh             report a machine's topology/partitions/modules
  download_data.sh        pre-fetch datasets -- LOGIN NODE ONLY
  task_wrapper.sh         per-rank entrypoint, site-agnostic
  lumi/                   job_smoke, job_cifar_debug, job_cifar_1node, job_cifar_multinode
  snellius/               job_smoke, job_cifar_1node, job_cifar_multinode
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

Code lives in `$HOME` (20 GB, 100k inodes). Data, checkpoints and logs live on
`/scratch/project_462000226/$USER`, symlinked as `data/` and `runs/`.

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

## Bringing up Snellius

The LUMI path is verified end to end. The Snellius path is a **template written
from SURF documentation and never executed** -- no Snellius access was available
when it was written. Treat every value marked `[VERIFY]` in `sites/snellius.sh`
as a guess until confirmed.

First run on Snellius:

```bash
git clone <this repo> ~/ProjectWorldWide && cd ~/ProjectWorldWide
./scripts/siteinfo.sh              # prints partitions, GPUs/node, cores, modules, filesystems
```

Use that output to correct, in `sites/snellius.sh`:

| value | how `siteinfo.sh` tells you |
|---|---|
| `PWW_GPUS_PER_NODE` | `Gres=gpu:N` on a GPU node |
| `PWW_CPUS_PER_TASK` | `CPUTot` / that N |
| `PWW_TORCH_MODULE` | the candidate PyTorch modules listed |
| `PWW_SCRATCH` | which of `/scratch-shared`, `/projects/0/...` exists and has room |
| `PWW_ACCOUNT` | the accounts section (may legitimately be empty) |

Then match the `#SBATCH` headers in `scripts/snellius/*.sh` to the same numbers
and partition name, and work up in the same order that was used on LUMI:

```bash
./scripts/bootstrap.sh
./scripts/download_data.sh
source env.sh && pww_run python3 tests/test_local.py    # must be 17/17
sbatch scripts/snellius/job_smoke.sh                    # must print SMOKE TEST PASSED
sbatch scripts/snellius/job_cifar_1node.sh --config configs/cifar10_resnet18.yaml
sbatch scripts/snellius/job_cifar_multinode.sh
```

The smoke test is the important gate: it verifies each rank gets a distinct GPU,
that all-reduce is numerically correct, and that collective bandwidth is
plausible. **Single-digit GB/s across nodes means NCCL fell back to TCP** instead
of InfiniBand -- uncomment `NCCL_SOCKET_IFNAME` in the job script. For reference,
LUMI measures 123 GB/s within a node and 88 GB/s across two.

Two things likely to need attention that the template cannot solve blind:

- **Package coverage.** LUMI's container bundles transformers, tokenizers,
  datasets and flash-attn. A Snellius PyTorch module may not. If `siteinfo.sh`
  reports any as MISSING, layer a venv on the module and point `PWW_VENV` at it:
  `python -m venv --system-site-packages $PWW_SCRATCH/venv`.
- **Comparing results between sites.** A Snellius node has 4 ranks where LUMI has
  8, so the same `--batch-size` gives half the global batch and a different scaled
  LR. Match the *global* batch when comparing, not the per-rank one.

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

Measured reference point (LUMI, verified): ResNet-18, 30 epochs, one node
(8 GCDs, global batch 1024, LR 0.8) -> **93.35% eval accuracy** at
**38,400 img/s**, 1.3 s/epoch, ~76 s wall clock for the whole run.

Two caveats on that number. The first epoch reports ~1,500 img/s rather than
38,000 because MIOpen is autotuning kernels for the shapes it has not seen
before; it is cached afterwards. And 93.35% is below the ~94-95% usually quoted
for ResNet-18 on CIFAR-10 because those figures assume a much longer schedule at
a smaller batch -- 30 epochs at a global batch of 1024 is a deliberately short
run. Raise `epochs` if you want to close that gap.
