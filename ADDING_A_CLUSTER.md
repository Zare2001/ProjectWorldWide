# Adding a cluster to the federation

How to bring a new HPC site into a running DiLoCo federation, from a fresh clone to
its first merged round.

The example throughout is [`configs/titan/qwen3_0.6b_c4_diloco.toml`](configs/titan/qwen3_0.6b_c4_diloco.toml)
— Qwen3 0.6B on C4 — but nothing in the procedure is specific to that model. §8 covers
what changes and what must not when you swap in a different LLM or corpus.

Related reading, not duplicated here: [FEDERATION_GUIDE.md](FEDERATION_GUIDE.md) §7 for the
short version and the design reasoning, [RUNBOOK.md](RUNBOOK.md) Part 0–1 for the
existing sites' exact commands, [scripts/titan/README.md](scripts/titan/README.md) for
why the torch environment is a container in one place and a venv in another.

---

## 0. What a "cluster" is here, and what is actually required

A site is **one file** — `sites/<name>.sh` — plus a detection branch and a job script.
There is no per-site deployment script to write and no application code to change.
`scripts/deploy.sh` is site-agnostic by construction: everything machine-specific comes
from the site file, which `env.sh` selects by detection.

Nothing on the central node changes. `min-clients: 1` means a new site is sampled as
soon as it connects, `configure_fit` hands it the current global model before it trains
a single step, and DARL partitions the index space over however many clusters have
registered.

The whole of the difficulty is in three things that are **not** per-site config:

| | where the time goes |
|---|---|
| **The torch ≥ 2.9 environment** | a container build (30–60 min batch job) or a pip venv |
| **The corpus** | compute nodes have no internet; shards and tokenizer must be on scratch first |
| **Reachability** | compute nodes must reach the central VM on the DARL and Flower ports |

And one hard invariant: the new site must compute the **same block-space digest** as
every other participant, or registration is refused. That is the guard working, not a
bug. §6 is the contract.

---

## 1. Survey the machine

On a **login node** of the new cluster, before writing anything:

```bash
git clone <repo-url> ~/ProjectWorldWide
cd ~/ProjectWorldWide
./scripts/siteinfo.sh
```

`siteinfo.sh` modifies nothing. It reports hostname, Slurm cluster name, partitions,
node shape, GPU count, cores per socket, filesystems and quotas — every value you need
for the site file. Read them off the machine; do not take them from documentation.

Note especially:

- **Ranks per node.** On AMD this is **GCDs, not cards** — LUMI's MI250X nodes are
  4 cards = 8 GCDs = 8 ranks. On NVIDIA it is cards: Snellius' H100 nodes are 4 = 4 ranks.
  Getting this wrong misplaces every process.
- **Which scratch survives.** Snellius' `/scratch-shared` is purged on file age (~14 days).
  Anything you want to keep belongs on a project filesystem.
- **Whether a partition exists at all.** `gpu` is not a partition name on Snellius;
  jobs naming it are rejected at submit time.

---

## 2. Write `sites/<name>.sh`

Copy whichever existing file the new machine resembles — [`sites/lumi.sh`](sites/lumi.sh)
for a container-based ROCm site, [`sites/snellius.sh`](sites/snellius.sh) for a
module-plus-venv CUDA site — and replace the values.

Every site file must define:

| | |
|---|---|
| `PWW_ACCOUNT` | accounting project for `sbatch` (may be empty where associations are implicit) |
| `PWW_SCRATCH` | writable, large, visible from compute nodes |
| `PWW_GPUS_PER_NODE` | ranks per full node — GCDs on AMD, cards on NVIDIA |
| `PWW_CPUS_PER_TASK` | cores per rank |
| `PWW_ACCELERATOR` | `rocm` or `cuda` |
| `PWW_GPU_VISIBLE_VAR` | `ROCR_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` |
| `PWW_LAUNCH` | bash **array**: the command prefix that enters the environment |
| `pww_cpu_bind()` | echoes the `--cpu-bind` value for an allocation |
| `pww_titan_env()` | echoes `kind<TAB>path<TAB>build-command`, so `deploy.sh` can check the torch ≥ 2.9 setup without knowing the machine |

`PWW_DATA_DIR`, `PWW_OUTPUT_DIR`, `PWW_TMPDIR` and `PWW_CACHE_DIR` are derived from
`PWW_SCRATCH` by `env.sh`; set them only to override.

### `PWW_LAUNCH` — container vs modules

This is the one line that differs most between sites. It is an array, and it prefixes
every Python invocation the repo makes.

```bash
# container site (LUMI): everything runs inside the image
PWW_LAUNCH=(singularity exec "${PWW_CONTAINER}")

# module/venv site (Snellius): the venv puts python on PATH, so no prefix
PWW_LAUNCH=()
```

If your cluster has a maintained AI container, prefer it — you inherit a tested
RCCL/NCCL and interconnect stack. If not, build a venv against the site's own Python
module. Do not mix: a venv built against the host Python and then used inside a
container will resolve the wrong libraries.

### `pww_cpu_bind()` — worth getting right

Wrong CPU placement silently costs 10–30% throughput. Two working patterns:

```bash
# Snellius: 4 sockets, one GPU each, so "rank N on socket N" falls out of
# --cpus-per-task + --distribution=block:block. No hand mask needed.
pww_cpu_bind() { echo "cores"; }

# LUMI: the GCD-to-core map is not something Slurm can infer, so it is explicit --
# and only valid for a full 8-rank node, hence the guard.
pww_cpu_bind() {
    if [[ "${SLURM_NTASKS_PER_NODE:-1}" -eq "${PWW_GPUS_PER_NODE}" ]]; then
        echo "${PWW_CPU_BIND}"
    else
        echo "cores"      # partial node: forcing a fixed mask fails outright
    fi
}
```

### `pww_titan_env()` — so `deploy.sh` can check without knowing the machine

```bash
# container
pww_titan_env() {
    printf 'container\t%s\tsbatch scripts/<name>/build_titan_container.sh\n' \
        "${PWW_SCRATCH}/containers/pww-titan.sif"
}

# venv
pww_titan_env() {
    printf 'venv\t%s\t./scripts/titan/setup_venv_<name>.sh\n' \
        "${PWW_TITAN_VENV:-${HOME}/venvs/pww-titan-<name>}"
}
```

`kind` may also be `none` — that is what the central VM uses, since it runs no model.

---

## 3. Add the detection branch

`env.sh`'s `pww_detect_site` picks the site file. Add a branch that matches something
unambiguous and cheap on the new machine — a facility-specific path is better than a
hostname pattern, since login and compute nodes often differ:

```bash
pww_detect_site() {
    if [[ -n "${PWW_SITE:-}" ]]; then
        echo "${PWW_SITE}"
    elif [[ -d /appl/local/containers/sif-images ]]; then
        echo lumi
    elif [[ -d /sw/arch ]] || [[ "$(hostname -f 2>/dev/null)" == *snellius* ]]; then
        echo snellius
    elif [[ -d /some/path/unique/to/your/site ]]; then      # <-- new
        echo <name>
    else
        echo central
    fi
}
```

`PWW_SITE=<name>` in the environment always wins, so you can test the site file before
detection is right.

Verify:

```bash
source env.sh && pww_summary
```

It must report your site, your scratch, and the derived paths.

---

## 4. Deploy

```bash
./scripts/deploy.sh --check      # report only, changes nothing
./scripts/deploy.sh             # pull, submodules, dirs, symlinks, verify
```

This is the same script on every machine, including the central VM, and re-running it
is the normal way to pick up a commit. It:

1. refuses to pull with modified tracked files
2. initialises submodules — `third_party/torchtitan` is pinned at a specific commit, not floating, and is never pip-installed: it reaches the interpreter through `PYTHONPATH`
3. creates `PWW_DATA_DIR`, `PWW_OUTPUT_DIR`, `PWW_TMPDIR`, `PWW_CACHE_DIR`, `logs/`
4. symlinks `runs/` and `data/` to the scratch locations
5. reports whether the torch ≥ 2.9 environment exists, via your `pww_titan_env`
6. runs `tests/test_darl.py` — the one suite needing neither torchtitan nor a GPU, so the cheapest signal the checkout is coherent
7. prints the next command for this site

It deliberately does **not** build the heavy environment. A container build is a 30–60
minute batch job, so it tells you the command rather than starting it behind your back.

> **Trap it guards:** if `runs/` or `data/` already exists as a real directory,
> `ln -sfn` would put the link *inside* it, producing `runs/runs`, and every documented
> path would quietly point at nothing. `deploy.sh` refuses and tells you to move it aside.

---

## 5. Build the torch ≥ 2.9 environment

Whatever `deploy.sh` step 4 reported as `TODO`. This is the bulk of the work and it is
a one-off.

```bash
# container site
sbatch scripts/<name>/build_titan_container.sh        # 30-60 min

# venv site
./scripts/titan/setup_venv_<name>.sh                  # login node
```

Then confirm, in the environment a real job will use:

```bash
source env.sh
export PWW_CONTAINER="$PWW_SCRATCH/containers/pww-titan.sif"   # container sites only
pww_run python3 tests/test_darl.py
pww_run python3 tests/test_titan.py     # imports torchtitan, so needs the 2.9 env
```

Run the suites **repeatedly**, not once. Two bugs found in this codebase were invisible
in a single run — a DARL lease deadlock at roughly 1 in 10, and a fixed rendezvous port
at 1 in 35. A green single run is not evidence of stability.

---

## 6. The corpus — where sites actually get out of sync

This is the part that fails at 3am if you rush it. Two files define agreement:
the **tokenizer** and the **shard manifest**.

### The contract

| Must agree across every site | Enforced by | Failure mode if it doesn't |
|---|---|---|
| `tokenizer.json` — byte-identical | `manifest.tokenizer_sha256` vs live fingerprint, in `shards.py verify_compatible` | startup `ValueError` |
| window count (`num_windows`) | both digests below | registration refused |
| `training.seq_len` | `manifest.seq_len` vs config, in `verify_compatible` | startup `ValueError` |
| `darl.block_size` | `BlockSpace.digest` at registration | registration refused |
| `darl.space_seed` | same | registration refused |
| `darl.epochs` | same | registration refused |

There are **two** digests, with different scopes. Both must match; they catch different
mistakes.

`Manifest.digest()` — corpus identity, printed by `tokenize_c4.sh`:

```
blake2b( format : seq_len : window : dtype : vocab_size : tokenizer_sha256 : num_windows )
```

Note what it deliberately **excludes**: the shard file list. Two sites may legitimately
stage the same corpus into a different number of files, as long as the resulting window
sequence is the same. So an `rsync` need not preserve file layout — only the windows.

`BlockSpace.digest(epoch)` — index-space identity, checked at DARL registration:

```
blake2b( num_samples : block_size : seed : shuffle : epoch )  + the permutation bytes when shuffling
```

A mismatch here means the two sides disagree about what position *p* refers to, so every
disjointness guarantee is void — which is why it is a hard refusal rather than a warning.

`Manifest.from_dict` also rejects an unrecognised `format` outright, so a manifest written
by an older version of the tokeniser is a clear error rather than a mis-slice.

The tokenizer check exists because a mismatch is otherwise **silent and total**: every
token id would mean a different piece of text while the index space still looked valid.
Note that regenerating a tokenizer under a different `transformers` version can change
the file, and therefore its sha256, even for the same repo id.

### The right way: tokenise once, copy the output

Do **not** regenerate the corpus at each site. It is deterministic given identical
inputs, but "identical" includes library versions you do not control.

```bash
# ---- on a site that already has the corpus ----
rsync -a --info=progress2 \
  "$PWW_DATA_DIR/tokenizers/tokenizer-128k" \
  "$PWW_DATA_DIR/c4-tokenizer-128k-2048" \
  <newsite>:/path/to/its/PWW_DATA_DIR/

# ---- on the new site, confirm agreement ----
source env.sh
python3 -c "
import json
m = json.load(open('$PWW_DATA_DIR/c4-tokenizer-128k-2048/manifest.json'))
print('format      ', m['format'])
print('num_windows ', m['num_windows'])       # -> NUM_SAMPLES on the central VM
print('seq_len     ', m['seq_len'], '/ window', m['window'])
print('dtype       ', m['dtype'])
print('vocab_size  ', m['vocab_size'])
print('tokenizer   ', m['tokenizer']['repo_id'])
print('tok sha256  ', m['tokenizer']['sha256'])
print('shards      ', len(m['shards']), 'file(s) -- may differ per site')"
```

Every line except the last must match an existing site exactly. The shard file *count*
may differ, because `Manifest.digest` excludes it.

Or let the code do the comparison, which also prints the digest itself:

```bash
PYTHONPATH=src pww_run python3 -c "
from pww.titan.shards import read_manifest
m = read_manifest('$PWW_DATA_DIR/c4-tokenizer-128k-2048')
print(m.describe()); print('digest', m.digest())"
```

### The other way: tokenise locally

Only if copying is impractical. **Login node** — compute nodes have no internet.

```bash
source env.sh
export PWW_CONTAINER="$PWW_SCRATCH/containers/pww-titan.sif"   # container sites only

# 1. tokenizer -- prints the vocab size that becomes the embedding
./scripts/titan/download_tokenizer.sh                       # openeurollm/tokenizer-128k
#   or: --repo-id Qwen/Qwen3-0.6B

# 2. tokenise, ONCE per (corpus, tokenizer, seq_len). Prints windows + digest.
./scripts/titan/tokenize_c4.sh --dataset c4 --seq-len 2048 --max-files 32

# 3. held-out validation split -- seconds, and NOT optional for a real run
./scripts/titan/stage_c4.sh --split validation --files 1 --out "$PWW_DATA_DIR/c4-validation"

# optional: keep the raw text so re-tokenising does not re-download hundreds of GB
./scripts/titan/stage_c4.sh --files 32
./scripts/titan/tokenize_c4.sh --dataset c4_local --seq-len 2048
```

`download_tokenizer.sh` fails loudly if the repo ships no `tokenizer.json` — torchtitan's
`build_hf_tokenizer` needs the fast format and will not accept a SentencePiece model.

Step 3 is not optional because the bundled `c4_test` fixture is **not held out** — it is
the head of `en/c4-train.00000`, the first file step 2 tokenises, so its windows are also
training windows. That biases the comparison in favour of whichever arm consumes more of
the corpus.

### What the central node needs — one small file

The central VM never opens the shards. It needs exactly one number, the window count:

```bash
scp "$PWW_DATA_DIR/c4-tokenizer-128k-2048/manifest.json" <central>:/tmp/
```

Then on the VM, `MANIFEST=/tmp/manifest.json` reads the count out of it, or
`NUM_SAMPLES=<windows>` states it directly (see [RUNBOOK.md](RUNBOOK.md) Part 2). If a
federation is already running, **the count is already fixed** — the new site must match
it, not the other way round.

---

## 7. Reachability, then the first job

Test from a **login node** before spending a queue slot. Compute nodes must reach the
central VM on the DARL port (29510 by default) and the Flower port (29511):

```bash
curl -s -m 5 http://<central-ip>:29510/health          # -> {"ok": true, "epoch": 0}
curl -s -m 5 -o /dev/null -w '%{http_code}\n' http://<central-ip>:29511
```

The central VM's default address is baked into `scripts/titan/run_train.sh` and the job
scripts; override with `PWW_CENTRAL_IP` or `CENTRAL_IP` rather than editing them. Sites
only ever dial **out**, so nothing needs opening on the cluster's side.

### Build `scripts/<name>/`, don't clone someone else's

Create the new cluster's own folder with only what that cluster needs. Do **not** copy
`scripts/lumi/` or `scripts/snellius/` wholesale — most of what is in them is either
irrelevant to a new site or actively wrong for it.

| File | Needed? | Why |
|---|---|---|
| `job_titan_diloco.sh` | **required** | the DiLoCo site job — this is the run |
| `build_titan_container.sh` *or* `setup_venv_<name>.sh` | **required**, one of the two | the torch ≥ 2.9 environment; whichever §5 chose |
| `job_smoke.sh` | recommended | smallest thing that exercises FSDP2 and the real forward pass; validates the site inside one short queue slot |
| `job_titan_central.sh` | optional | only if you want a single-site baseline at this cluster |
| `job_tokenize_c4.sh` | optional | only if this facility's login nodes can't do the tokenise pass (LUMI has one for this reason; Snellius doesn't need one) |
| `job_cifar_*.sh` | **no** | CIFAR-era scripts, unrelated to the LLM path |
| `job_flower_diloco*.sh` | **no** | the legacy HuggingFace path, superseded by the torchtitan one — see [FEDERATION_GUIDE.md](FEDERATION_GUIDE.md) §4 |

### The anatomy of a site job script

Read both existing `job_titan_diloco.sh` files side by side before writing yours. They are
139 and 167 lines and have the same four parts — but only part 4 is genuinely shared:

**1. `#SBATCH` header** — yours, from `siteinfo.sh`. One task per node and let `torchrun`
fork the ranks. Do not use `--ntasks-per-node=<gpus>`: torchtitan expects to own the
process topology (`LOCAL_RANK`, the rendezvous, the device mesh), and N independent srun
tasks each trying to be rank 0 of their own `torchrun` is the classic way to get N
one-GPU jobs that never form a mesh.

**2. Escalate to the torch ≥ 2.9 environment.** `env.sh` gives you the site's *default*
environment, which at both existing sites is the older torch. This part replaces it, and
it is where every site-specific quirk lives. The two existing sites show the classes of
problem to look for:

- *module-system interference* — Snellius' XALT wrapper injects an OpenSSL that conflicts with torch's, so the script unloads XALT, unsets `LD_PRELOAD` and strips `/opt/xalt` from `LD_LIBRARY_PATH`
- *library shadowing* — LUMI deliberately omits `singularity --rocm`, because binding the host's ROCm 6.2.4 over the image's 6.4 makes the older `librccl` shadow the newer one and `import torch` dies with `undefined symbol: ncclGroupSimulateEnd`
- *caches that must not live in `$HOME`* — MIOpen defaults its kernel cache to `$HOME`, which on LUMI is a small quota shared between concurrent jobs, and two jobs racing on one cache database is a known hang. One cache per job, on scratch.
- *the re-source trap* — `run_train.sh` sources `env.sh` again in its own shell. On Snellius that would re-activate the 2.7.1 venv and launch the ranks with a torch that has no FSDP2, so the job script points `PWW_VENV` at the titan venv to make the re-source a no-op.

> **The trap worth knowing before you hit it:** `export PWW_LAUNCH` looks like it works
> and cannot — **bash does not export arrays**. `run_train.sh` is a separate process that
> rebuilds `PWW_LAUNCH` from `PWW_CONTAINER`, so a container choice must travel as
> `PWW_CONTAINER`. Getting this wrong cost a real debugging session: every rank died on
> `ModuleNotFoundError: No module named 'tyro'` while the header printed the path of the
> image it was *supposed* to be using.

**3. Preflight the coordinator** before `torchrun` claims the GPUs, so a bad token fails
in seconds rather than well into a spent allocation. Copy this block as-is; only the
comment about *why* a `000` is non-fatal is site-specific:

```bash
darl_probe="$(curl -sS -m 15 -o /dev/null -w '%{http_code}' \
    -H "X-DARL-Token: ${DARL_TOKEN}" \
    "http://${CENTRAL_IP}:${PWW_DARL_PORT:-29510}/health" 2>/dev/null)" || true
[[ "${darl_probe}" =~ ^[0-9]{3}$ ]] || darl_probe=000
# 200 -> good; 401 -> exit 1, wrong token; 000 -> warn only, a proxy may still work
```

**4. Trap SIGTERM, then delegate.** This part is effectively identical everywhere — only
`--site` and the `--nproc` default change:

```bash
trap 'kill -TERM ${TRAIN_PID:-0} 2>/dev/null' TERM

"${PWW_ROOT}/scripts/titan/run_train.sh" \
    --config "${CONFIG}" --tokenizer "${TOKENIZER}" --shards "${SHARDS}" \
    --central "${CENTRAL_IP}" --site <name> --nproc "${SLURM_GPUS_PER_NODE:-N}" \
    -- "$@" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
```

The trap is not cosmetic. Slurm sends SIGTERM before SIGKILL, and forwarding it lets the
leader rank release its uncommitted DARL leases immediately instead of the other sites
idling out a full lease TTL. On a long run, walltime kill is the *normal* way a job ends.

Then submit:

```bash
DARL_TOKEN=<token from the central VM> \
  sbatch -J pww-<name>-titan-diloco \
  --export=ALL,DARL_TOKEN,PWW_FRESH_RUN=1,ENABLE_WANDB=1,WANDB_PROJECT=<project>,WANDB_RUN_NAME=diloco-<name> \
  --time=40:00:00 scripts/<name>/job_titan_diloco.sh
```

`run_train.sh` fills in the rest as CLI overrides, which is why the TOML carries no site
paths: `--model.hf_assets_path`, `--training.dataset_path`, `--job.dump_folder`,
`--darl.url`, `--darl.site`, `--flower.server_address`, the validation paths and the batch
geometry.

**First-minute signals**, in the site's `logs/` and in the central `flower.log`:

- `darl: corpus ... | space digest <hex>` — must equal the other sites' digest
- the new site's name appearing in the aggregator's `N cluster(s) [...]` line
- on a genuinely fresh federation, round 1 loss ≈ 9.9 (ppl ≈ 20,000). A low opening
  loss means something resumed that should not have

---

## 8. Generalising: any model, any corpus

The procedure above is model-agnostic. What changes, and what must not:

### Per-site — may differ freely

| | |
|---|---|
| `local_batch_size`, gradient accumulation, ranks | different hardware, same work — see below |
| `job.dump_folder` | **must** differ per concurrent job at a site, or two runs share a checkpoint dir |
| every path | derived from `PWW_SCRATCH` |
| `PWW_LAUNCH`, CPU binding, partition | that is what the site file is for |

**Equal steps is not equal work.** Match the *global* batch, not the local one:

```
Snellius:  8 local x 4 ranks x 3 accum = 96
LUMI:      6 local x 8 ranks x 2 accum = 96
```

`run_train.sh` will solve for the local size if you give it `PWW_GLOBAL_BATCH`.

### Federation-wide — must be identical everywhere

| | |
|---|---|
| the model — `[model] name`, `flavor` | a mismatched model is refused at merge |
| `training.seq_len` | checked against the manifest |
| the tokenizer, and therefore the shards | sha256-enforced |
| `darl.block_size`, `space_seed`, `epochs` | digest-enforced |
| `darl.inner_steps` (H) | the round boundary; sites would drift out of phase |
| `NUM_SAMPLES` on the central VM | must equal the shared window count |

### Swapping the model

`[model] name = "pww_qwen3"`, `flavor = "0.6B"` selects a torchtitan flavor. You do not
restate `vocab_size` or `eos_id` anywhere: `PWWQwen3ModelArgs` in
[`src/pww/titan/__init__.py`](src/pww/titan/__init__.py) reads them back off whatever
tokenizer `model.hf_assets_path` points at and rebuilds the model accordingly. That is
why the OpenEuroLLM 128k tokenizer (131,073 ids) works against a flavor whose own default
is Qwen3's 151,936 — and why changing tokenizer changes the embedding size for free, but
invalidates every shard built with the old one.

### Swapping the corpus

Re-run `tokenize_c4.sh` with the new dataset, then propagate the new `manifest.json` to
the central VM and restart it with the new `NUM_SAMPLES`. A corpus change is a new
federation, not a resumable one: the index space means something different.

### One thing that scales with the model

Transport. `inline` puts weights inside the gRPC message and is fine to about 1B
parameters; above that use `blob`, which brings up an out-of-band HTTP store on 29512 and
needs that port reachable too. See [FEDERATION_GUIDE.md](FEDERATION_GUIDE.md) §1 for the
measured per-flavor table.

---

## 9. Checklist

```
[ ] siteinfo.sh run on a login node; values recorded
[ ] sites/<name>.sh written; all nine definitions present
[ ] pww_detect_site branch added to env.sh
[ ] source env.sh && pww_summary reports the new site correctly
[ ] ./scripts/deploy.sh clean; runs/ and data/ are symlinks
[ ] torch >= 2.9 environment built; test_darl.py and test_titan.py pass repeatedly
[ ] tokenizer + shards present on scratch (copied, not regenerated)
[ ] num_windows / seq_len / vocab_size / tokenizer.sha256 match an existing site
[ ] Manifest.digest() matches an existing site
[ ] /health reachable from a login node on the DARL port
[ ] job script copied and its #SBATCH header adjusted
[ ] first job: space digest matches, site appears in the aggregator's cluster list
```

## 10. Troubleshooting the refusals

Each of these is a guard doing its job. None should be worked around.

| Symptom | Cause | Fix |
|---|---|---|
| registration refused, digest mismatch | `num_samples`, `block_size`, `space_seed`, `epochs` or the permutation differs | compare all four against a working site; copy the shard dir rather than regenerating |
| `ValueError` on tokenizer sha256 | shards were cut with a different `tokenizer.json` | copy the tokenizer *and* the shards together, as one unit |
| `token shards were cut for seq_len N` | config and manifest disagree | set `training.seq_len` to the manifest's value, or re-tokenise |
| `manifest window != seq_len + 1` | corrupt or hand-edited manifest | re-tokenise |
| `unsupported token shard format` | manifest written by an older tokeniser version | re-tokenise, then re-copy to every site |
| new site never appears in `N cluster(s)` | not reachable, or wrong token | `curl /health` from a login node; check `DARL_TOKEN` |
| `no python >= 3.10 in this environment` from `deploy.sh` | the torch env is not on `PATH` yet | build it, or export `PWW_CONTAINER` |
| `runs/ exists as a real directory` | a previous manual setup | `mv runs runs.bak` and re-run `deploy.sh` |
| site trains but merges never include it | joined after the round barrier, or `inner_steps` differs | check `darl.inner_steps` matches |

One that is **not** a failure: when a server closes at its configured final round, site
jobs die with `grpc ... Stream removed (Socket closed)` and a non-zero exit. That is the
normal end of a run.
