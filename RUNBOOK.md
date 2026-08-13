# Runbook

The ordered path to a running multi-site job, and what to do when one breaks.

**No explanation here on purpose.** Every step links to the section of
[FEDERATION_GUIDE.md](FEDERATION_GUIDE.md) that says *why*, so there is only one copy of
the reasoning to keep correct. If you want to understand the system rather than run it,
start with [README.md § How it works](README.md#how-it-works).

Three machines, and the order between them matters:

```
  LOGIN NODE (each site)          CENTRAL VM                  COMPUTE (each site)
  stage the corpus         -->    start the daemons     -->   sbatch
  (needs the internet)           (needs the window count)     (needs both of the above)
```

---

## Part 0 — first time only, per site

Skip to Part 1 if `pww_summary` already prints sane paths and the torch 2.9 environment
exists.

```bash
# 1. Clone, then let deploy.sh do the rest. Login node.
git clone <repo> ~/ProjectWorldWide && cd ~/ProjectWorldWide
./scripts/deploy.sh                             # submodules, dirs, symlinks, verify
```

`deploy.sh` is also how you **update** a site later — it pulls, re-syncs the submodule and
re-checks everything, and it is the same script on all three machines including the central
VM. `--check` reports without changing anything. It tells you whether the torch ≥ 2.9
environment exists and the one command that builds it, but never starts that build itself,
because on LUMI it is a 30–60 minute batch job.

For a **new** HPC there is no new script to write — a site is one file, `sites/<name>.sh`.
See [FEDERATION_GUIDE.md § 7](FEDERATION_GUIDE.md) for the contract it must satisfy.

Two things to check before moving on:

```bash
ls -ld runs data          # BOTH must be symlinks into scratch, not directories
```

If either already exists as a real directory, `ln -sfn` puts the link *inside* it — you
get `runs/runs` — and every path in Part 4 silently points at nothing. Move the directory
aside and re-run `bootstrap.sh`.

On the **central VM**, bootstrap exits 1 on its last step, `import torch, torchvision,
transformers`. That is expected and fine: the central node has no GPU stack, the
directories and symlinks are created before that check, and the aggregator runs from its
own venv under `runs/central/.venv`.

The torchtitan path needs **its own** torch >= 2.9 environment, separate from this repo's
2.7.1 pin. Read [scripts/titan/README.md](scripts/titan/README.md) before running either:

```bash
# 2a. Snellius: a second venv alongside the 2.7.1 one.
./scripts/titan/setup_venv_snellius.sh                 # -> ~/venvs/pww-titan-snellius

# 2b. LUMI: a container, since LUMI's own image is on 2.7.1. ~30-60 min, batch job.
sbatch scripts/lumi/build_titan_container.sh           # -> $PWW_SCRATCH/containers/pww-titan.sif
```

Verify before spending a queue slot. At a **site**, go through `pww_run` rather than a
bare `python3` — LUMI's login nodes are on python 3.6, which cannot parse the tests.
The central VM has no container and uses its own venv instead:

```bash
# at a site
source env.sh
pww_run python3 tests/test_darl.py

# test_titan.py imports torchtitan, so it needs the 2.9 environment, not the
# 2.7.1 one. On LUMI, replace PWW_LAUNCH with the container built above -- the
# same swap scripts/lumi/job_titan_diloco.sh makes.
PWW_LAUNCH=(singularity exec --bind "$PWW_SCRATCH" "$PWW_SCRATCH/containers/pww-titan.sif")
export PYTHONPATH="$PWD/src:$PWD/third_party/torchtitan"
export SINGULARITYENV_PYTHONPATH="$PYTHONPATH"
pww_run python3 tests/test_titan.py

# on the central VM -- no torchtitan, so use its own venv
PYTHONPATH="$PWD/src" runs/central/.venv/bin/python3 tests/test_federation.py   # 26
PYTHONPATH="$PWD/src" runs/central/.venv/bin/python3 tests/test_darl.py         # 46
```

Setting `PWW_CONTAINER` instead does **nothing here**, which is worth knowing before
it costs you an hour: `env.sh` reads it while building `PWW_LAUNCH`, so by the time
you have a shell it has already been consumed. It works only for a *child script*
that sources `env.sh` itself — which is exactly the Part 1 case below.

`test_darl.py` under the system python3 reports **42 passed, 1 skipped** instead — the
four `torch_data` checks need torch. That is not a failure.

---

## Part 1 — stage the corpus (login node, needs internet)

Compute nodes at both sites have **no internet**, so everything the run reads must be on
scratch first. Do this once per `(corpus, tokenizer, seq_len)`.

This part is **not** independent of Part 0 on LUMI: `tokenize_c4.sh` imports
`pww.titan`, which imports torchtitan, so it needs the 2.9 container. Only the
tokenizer download runs in the default 2.7.1 one. Export this for the rest of the
section. Each script below sources `env.sh` in its own shell, so it picks the
exported value up while building `PWW_LAUNCH`:

```bash
source env.sh
export PWW_CONTAINER="$PWW_SCRATCH/containers/pww-titan.sif"   # LUMI only
```

```bash
# 1. Tokenizer. Prints the vocab size that becomes the embedding.
./scripts/titan/download_tokenizer.sh                  # -> $PWW_DATA_DIR/tokenizers/tokenizer-128k

# 2. Tokenise. Pick ONE line.
./scripts/titan/tokenize_c4.sh --dataset c4_test --seq-len 2048        # fixture, seconds
./scripts/titan/tokenize_c4.sh --dataset c4 --seq-len 2048 --max-files 32   # real, ~5B tokens

# 3. The held-out validation split. One small file, seconds, NOT optional for a real run.
./scripts/titan/stage_c4.sh --split validation --files 1 --out "$PWW_DATA_DIR/c4-validation"

# optional: keep the raw training text so re-tokenising does not re-download
./scripts/titan/stage_c4.sh --files 32
./scripts/titan/tokenize_c4.sh --dataset c4_local --seq-len 2048
```

**Why step 3 exists:** the bundled `c4_test` fixture the configs default to is **not held
out** — it is the head of `en/c4-train.00000`, the first file step 2 tokenises, so its
windows are also training windows. Worse than a uniformly optimistic eval: the chance an
eval window was trained on grows with how much of the corpus a run consumes, so a fixture
eval favours the DiLoCo arm (which consumes ~70% of the corpus at 20k steps) over the
step-matched baseline (~23%). C4's validation split is disjoint from train by
construction. Once `$PWW_DATA_DIR/c4-validation` exists, `run_train.sh` uses it
automatically (`validation  : c4_local at ...` in the job log); with nothing staged it
falls back to the fixture and prints a warning.

**Both sites must stage the same file.** The eval walks the same window sequence in
lockstep at every site — that is what makes the cross-site spread check meaningful — and
that requires identical input bytes. `--files 1` gives file `00000-of-00008` everywhere;
confirm with:

```bash
sha256sum "$PWW_DATA_DIR"/c4-validation/c4-validation.00000-of-00008.json.gz   # match across sites
```

**Once, not per run.** It is a file on scratch: every subsequent run — the 1k rehearsal,
the 20k run, anything after — picks it up automatically. Stage it **before the 1k pair**,
not just before the 20k run: the rehearsal exists to exercise the exact configuration the
real run will use, and gate check 3 ([Part 3b](#gate-the-1k-pair-must-pass-before-the-20k-submission))
compares held-out losses that only mean something if the 1k pair evaluated on this split.
Staging it only for the 20k run means the `c4_local` path runs for the first time in hour
1 of a 72-hour job — the exact kind of gap the rehearsal is meant to close.

**Snellius purge caveat.** `/scratch-shared` deletes files after ~14 days of inactivity.
A running job reads the file every eval, which keeps it alive; a long gap between runs can
silently remove it, and the next job then falls back to the fixture (with a warning in its
log, but a warning nobody reads is a fixture eval nobody meant to run). The stage command
is idempotent and takes seconds — re-run it whenever the runs have been idle for a while,
same as the tokenised corpus itself.

**Write down two numbers the last command prints.** Both are needed later and a mismatch
is a startup failure, not a silent problem:

| | used for |
|---|---|
| **windows** | `NUM_SAMPLES` on the central VM |
| **manifest digest** | must match at both sites |

**Better than writing the window count down: copy `manifest.json` to the central VM.**

```bash
scp "$PWW_DATA_DIR/c4-tokenizer-128k-2048/manifest.json" <central>:/tmp/
```

That one file — a few hundred bytes — is all the central node needs from the corpus, and
`MANIFEST=` in Part 2 reads the count out of it. The central VM never opens the shards, so
there is nothing else to copy. It also removes the only step in the whole sequence whose
failure mode is a mistyped seven-digit number, which the digest guard catches, but not
until a site has already spent its queue wait trying to register.

Both sites must end up with the **same window count**. Either run the tokenisation
identically at each (it is deterministic given the same inputs) or run it once and
`rsync` the output directory across — the second is cheaper and removes the risk.

Run this at **both** sites before either job is queued, and compare the two lines. It
reads `manifest.json` directly and imports nothing from the repo, so it works on a bare
login node without `PYTHONPATH`, without the torchtitan submodule, and on a directory
that was `rsync`ed in:

```bash
python3 - "$PWW_DATA_DIR/c4-tokenizer-128k-2048" <<'PY'
import hashlib, json, sys, pathlib
raw = json.loads((pathlib.Path(sys.argv[1]) / "manifest.json").read_text())
h = hashlib.blake2b(digest_size=16)
h.update(f"{raw['format']}:{raw['seq_len']}:{raw['window']}:{raw['dtype']}:"
         f"{raw['vocab_size']}:{raw['tokenizer']['sha256']}:{raw['num_windows']}".encode())
print(f"windows {raw['num_windows']:,}  digest {h.hexdigest()}  "
      f"seq_len {raw['seq_len']}  vocab {raw['vocab_size']:,}")
PY
```

If the two sites print different digests, stop here — a job would be refused at
registration anyway, and finding out now costs seconds instead of a queue wait.

If a coordinator is **already running on a different corpus** — which it is, on the
wikitext-103 placeholder — it does not pick this count up by itself. Restart it with
`PWW_FRESH_RUN=1` before any real training: [Part 2, starting a genuinely fresh run](#part-2--start-the-central-vm).

---

## Part 2 — start the central VM

Runs on `145.38.206.143`. Must be up **before** either site's job starts training.

**The one thing this node needs from Part 1 is the window count.** It never reads the
corpus — no shards, no tokenizer, no GPU — so the only input that has to travel from a
site is that single number. Supply it either way:

| | |
|---|---|
| `MANIFEST=/path/to/manifest.json` | reads `num_windows` out of the file you copied in Part 1. Preferred — nothing to mistype. |
| `NUM_SAMPLES=<windows>` | the number itself, as printed by `tokenize_c4.sh`. Wins if both are given. |

```bash
cd ~/ProjectWorldWide

# Pick the transport by model size. 0.6B -> inline. Anything larger -> blob.
# FEDERATION_GUIDE.md #1 has the measured per-flavor table.

# <= ~1B parameters
MANIFEST=/tmp/manifest.json BLOCK_SIZE=1024 SEED=42 \
AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml \
  ./scripts/central_node/start_central_services.sh

# above ~1B -- also brings up the blob store on 29512
TRANSPORT=blob MANIFEST=/tmp/manifest.json BLOCK_SIZE=1024 SEED=42 \
AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml \
  ./scripts/central_node/start_central_services.sh

# or with the count typed in directly, if you have no manifest to hand
NUM_SAMPLES=<windows> BLOCK_SIZE=1024 SEED=42 \
AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml \
  ./scripts/central_node/start_central_services.sh
```

It echoes what it used, so this is checkable rather than assumed:

```
num_samples 2750000 read from /tmp/manifest.json
```

Then confirm, and take the token the sites need:

```bash
./scripts/central_node/status_central_services.sh     # ports, merge round, membership
cat runs/darl/token
```

`SEED` must equal `darl.space_seed` in the run's TOML, `BLOCK_SIZE` must equal
`darl.block_size`, and `NUM_SAMPLES` the window count from Part 1. Disagreement is refused
at registration by the block-space digest. Confirm the startup line agrees with the run
you meant to launch:

```
block space: 115,156 samples / 1,024 per block = 113 blocks | digest 7029d22cecd74aa3
darl coordinator on http://0.0.0.0:29510 | 113 blocks | epoch 0/1 | token yes
```

⚠️ **Those numbers are a wikitext-103 placeholder, not a target.** 115,156 windows /
113 blocks / digest `7029d22c` is what the coordinator was first brought up on, and any
site that tokenises C4 computes a different digest and is **refused at registration**.
The count has to come from your own Part 1 output. See below for the changeover.

`token yes` is the one to read. `token NO -- anyone can lease` means every endpoint on
that port is open, including on the public IP.

To stop:

```bash
./scripts/central_node/stop_central_services.sh
```

**Restarting resumes; it does not start over — and a bare restart is enough:**

```bash
./scripts/central_node/start_central_services.sh          # no variables needed
```

The script records what it launched with (`runs/darl/space.env`, `runs/central/launch.env`)
and reuses it, printing `block space: resumed from …` when it does. An explicit variable
still wins, so the commands above keep working unchanged.

Do not omit them on a *first* start of a new run. Without a recorded launch the fallbacks
are `NUM_SAMPLES=50000 BLOCK_SIZE=1000` and the **ResNet** aggregator config, which is not
a torchtitan run: it turns FedMom off (`server-momentum: 0.0` is FedAvg), sets
`min-clients: 2` so the run blocks until both sites are out of the queue, and cuts
`num-rounds` to 50 with a 300 s timeout. Check the line the server logs:

```
transport=inline | server_learning_rate=0.7, server_momentum=0.9 | min_clients=1 | num_rounds=200 (attempts), round_timeout=1800s
```

Both state dirs survive a stop, and the coordinator says which it did:

```
restored coordinator from .../snapshot.json (+0 journal entries): epoch 0, 0/113 blocks committed
```

If that line is absent, the lease table was **discarded** and already-trained windows are
free to be handed out again.

### Starting a genuinely fresh run

**`PWW_FRESH_RUN=1`, not `DARL_FRESH=1`.** There are two durable stores and they have to
agree:

| | holds |
|---|---|
| `runs/darl` | the lease table — which windows have been trained |
| `runs/central/global` | the global model, its momentum buffer, the checkpoints |

Resetting only the lease table re-issues windows the surviving model already trained on.
Resetting only the model merges a brand-new epoch into momentum accumulated against a
different corpus. **Both are silent.** So one switch does both, and `DARL_FRESH=1` on its
own now warns rather than being quietly honoured.

```bash
./scripts/central_node/stop_central_services.sh
PWW_FRESH_RUN=1 MANIFEST=/tmp/manifest.json BLOCK_SIZE=1024 SEED=42 \
AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml \
  ./scripts/central_node/start_central_services.sh
```

`MANIFEST=` reads the window count out of the file instead of you retyping it — the central
node never opens the corpus, so copying that one small file across is enough. `NUM_SAMPLES=`
still works and wins if both are given.

Confirm it discarded rather than resumed:

```
PWW_FRESH_RUN=1: discarding the lease table AND the global model.
--fresh: moved snapshot.json aside to snapshot.json.superseded
darl coordinator on http://0.0.0.0:29510 | 2692 blocks | epoch 0/1 | token yes
```

Old state is **renamed, not deleted** — `snapshot.json.superseded`, `journal.jsonl.superseded`,
`round-*.npz.superseded` — so a mistaken reset is recoverable for one generation.

**Switching corpus is the case that requires this.** Going from a placeholder space to real
C4 is exactly it: the old lease table is not stale, it is *meaningless*, because its
committed positions index a different corpus.

**The sites need the same variable.** torchtitan keeps its own checkpoint under each site's
dump folder and resumes from it, so pass `PWW_FRESH_RUN=1` at submit time as well:

```bash
PWW_FRESH_RUN=1 DARL_TOKEN="..." sbatch --export=ALL,PWW_FRESH_RUN,DARL_TOKEN \
  scripts/snellius/job_titan_diloco.sh
```

Only the model in that checkpoint is harmless — `configure_fit` overwrites it. The other
three are not:

| restored | effect on a fresh run |
|---|---|
| `OPTIMIZER` | stale AdamW moments paired with completely different weights |
| `LR_SCHEDULER` + `TRAIN_STATE` | resumes at the old global step, so a freshly seeded model **skips `warmup_steps`** and takes near-peak LR on its first step |
| `DATALOADER` | the DARL client's `epoch`/`phase_index`/`samples_seen`, while the coordinator is on a fresh epoch with every block unassigned |

`checkpoint.interval` equals `darl.inner_steps`, so a checkpoint is written every round — a
job that failed after one round has already left one behind.

**The central baseline resumes exactly the same way**, and this is the easy one to forget.
`scripts/{lumi,snellius}/job_titan_central.sh` has no Flower server and spawns a throwaway
coordinator that is always `--fresh`, which makes it *look* stateless. It is not:
torchtitan's checkpointer does not know or care whether Flower is involved, so
`configs/titan/qwen3_0.6b_c4_central*.toml` resume from `./outputs/qwen3-0.6b-c4-central*`
like anything else. A baseline resubmitted without `PWW_FRESH_RUN=1` continues the previous
one at its old LR-schedule position while reading a block space that just went back to
epoch 0 — a plausible curve of a run that is neither the old one nor a new one.

```bash
PWW_FRESH_RUN=1 sbatch --export=ALL,PWW_FRESH_RUN scripts/snellius/job_titan_central.sh
```

### `scripts/reset_run.sh` — clearing the site half up front

`PWW_FRESH_RUN=1` renames one generation aside (`checkpoint.superseded`) so a mistaken
reset is recoverable. When the intent is to actually reclaim the space, or to see what is
there before deciding, use the front end:

```bash
./scripts/reset_run.sh --dry-run     # list every path and its size, delete nothing
./scripts/reset_run.sh               # confirm, then delete
./scripts/reset_run.sh --yes         # no prompt
./scripts/reset_run.sh --config configs/titan/qwen3_0.6b_c4_diloco_1k.toml
./scripts/reset_run.sh --central     # ALSO the lease table + global model (on the VM)
```

With no `--config` it covers **every** shipped titan config, central and DiLoCo alike —
resetting one side of a comparison and not the other is its own kind of wrong answer. What
it clears, per config's `dump_folder`:

| | |
|---|---|
| `checkpoint`, `checkpoint.superseded` | model, optimiser moments, LR position, dataloader state |
| `blob-staging` | one full model per staged blob; harmless to correctness, expensive in quota |
| `tb` | tensorboard event files, one directory per launch |
| `outputs/darl_local_central_*` | throwaway coordinator state dirs, one per baseline job |

It deliberately does **not** touch the tokenised corpus (resetting it means re-tokenising
C4), `logs/`, or `wandb/` — a finished run's record is the only copy of it. Pass
`PWW_FRESH_RUN=1` at submit time as well, so a checkpoint written between the reset and the
job starting cannot be picked up, and give the new run a new `WANDB_RUN_NAME` or it overlays
the old one under the same name.

Since this reset, the baseline job scripts also delete their own
`outputs/darl_local_central_<jobid>` on exit unless the coordinator log mentions an error,
so they stop accumulating one directory per run.

### It never decides this for you

Nothing infers that a run is dead, and nothing should — resume is the default and is always
safe. So a **VM reboot** or **every site sitting in the queue** needs no flags and no
intervention:

| question | mechanism | who decides |
|---|---|---|
| is this lease still held? | TTL + heartbeats | automatic, seconds–minutes |
| is this process still alive? | `is_live()`, for the cluster-id conflict only | automatic, one TTL |
| **is this run dead?** | `PWW_FRESH_RUN=1` | **you, never inferred** |

A bare restart after a reboot resumes: DARL restores its snapshot, the geometry comes from
`space.env`, and the model resumes from the newest finite checkpoint. `min-clients: 1` makes
"nobody connected" a wait, not a failure, and leases held by a job that vanished expire on
their own TTL.

Changing `NUM_SAMPLES`/`BLOCK_SIZE`/`SEED` without the flag is **refused, not adapted to**
— the restore compares the block-space digest and raises rather than falling back to a
fresh table:

```
snapshot.json was written for block-space digest 7029d22cecd7... but this one is
a1b2c3d4e5f6...; the permutation changed, so committed positions no longer mean
the same samples
```

That is the guard working. It also means the coordinator does not come up at all until
you either pass `PWW_FRESH_RUN=1` or point `--state-dir` somewhere else.

### Global-model checkpoints

The aggregator snapshots the merged model to `runs/central/global/checkpoints/`, in two
tiers, and resumes from them automatically:

| | when | kept | for |
|---|---|---|---|
| **ephemeral** | every merge | `--keep-ephemeral 2` | crash recovery |
| **persistent** | every `--persist-every 5` merges | `--keep-persistent 4` | rolling **back** |

```
round-000005-persistent.npz   round-000010-persistent.npz
round-000011-ephemeral.npz    round-000012-ephemeral.npz
```

~4.8 GiB each at 0.6B (weights + momentum, both float32), so ~29 GiB steady state.
`--keep-persistent 0` is unbounded: 200 rounds at every-5 is ~192 GiB, so check the
filesystem first — it warns with the projected total.

**Resume picks the newest checkpoint that is finite, not simply the newest.** If a run died
from a bad merge, the newest checkpoints are the dead ones; it walks back past them and says
how many it skipped. `--fresh-model` opts out entirely, and is needed when the architecture
changes, since a checkpoint's tensor shapes must match the model the sites build.

The round in the filename is the **merge** round, so `round-000008-*.npz` is exactly the
model a site joining at round 9 receives.

See [FEDERATION_GUIDE.md § 2](FEDERATION_GUIDE.md).

---

## Part 3 — submit at each site

Check reachability **from each site's login node** — a firewall problem found here costs
seconds instead of a queue wait:

```bash
export DARL_TOKEN="<from cat runs/darl/token>"
curl -sS -H "X-DARL-Token: $DARL_TOKEN" http://145.38.206.143:29510/health
# {"ok": true, "epoch": 0}
nc -zv 145.38.206.143 29511
nc -zv 145.38.206.143 29512      # blob transport only
```

Two ways to misread the result:

* **Do not run this on the central VM itself.** `145.38.206.143` is a floating IP that is
  not on any of that host's interfaces, and there is no hairpin route, so the curl times
  out there even when both daemons are healthy and every site can reach them. Check the
  central side with `127.0.0.1` and `ss -tlnp` instead.
* **Omitting the token gives `401`, not a connection error.** That is the reachability
  check succeeding and the token being enforced.

Submit. Both scripts default to `configs/titan/qwen3_0.6b_c4_diloco.toml`:

```bash
# LUMI, 8 GCDs
DARL_TOKEN="$DARL_TOKEN" sbatch scripts/lumi/job_titan_diloco.sh

# Snellius, 4 H100s
DARL_TOKEN="$DARL_TOKEN" sbatch scripts/snellius/job_titan_diloco.sh
```

Override with environment variables rather than editing the scripts:

| variable | default |
|---|---|
| `CONFIG` | `configs/titan/qwen3_0.6b_c4_diloco.toml` |
| `SHARDS` | `$PWW_DATA_DIR/c4-tokenizer-128k-2048` |
| `TOKENIZER` | `$PWW_DATA_DIR/tokenizers/tokenizer-128k` |
| `CENTRAL_IP` | `145.38.206.143` |

```bash
CONFIG=configs/titan/qwen3_8b_c4_diloco.toml \
DARL_TOKEN="$DARL_TOKEN" sbatch scripts/snellius/job_titan_diloco.sh
```

**Two jobs at the same site at once** — pass a distinct `--replica` to each, or the second
is refused. This is a correctness flag, not a tuning one:

```bash
scripts/titan/run_train.sh --replica a --config ...
scripts/titan/run_train.sh --replica b --config ...
```

### Stop the faster site idling at the round barrier

A round cannot close until the slowest site delivers its delta — that is DiLoCo, not a
setting. But the fast site does **not** have to spend the wait doing nothing. Measured:
Snellius did 37 s of work inside a 154 s round and idled 118 s, **76% of every round**.

`PWW_GRAD_ACCUM=N` makes each of that site's H optimiser steps consume N microbatches, so
it trains N× the tokens in the same number of steps and finishes when the slow site does:

```bash
PWW_GRAD_ACCUM=2 DARL_TOKEN="<token>" \
  sbatch --export=ALL,PWW_GRAD_ACCUM,DARL_TOKEN scripts/snellius/job_titan_diloco.sh
```

Set it **only on the faster site**. It prints what it chose:

```
grad accum  : 2x -> global_batch_size 64 (8 local x 4 ranks x 2); H is unchanged, so
              drift and the LR schedule are unaffected
```

| | |
|---|---|
| **Start at 2** | halves the idle, ~1.5× throughput, and gives a clean comparison before 4 |
| **Watch `drift_ratio_max`** | it should be **unchanged**. Accumulation adds no optimiser steps, so if drift moves, something else is going on |
| **Judge by loss per *token***, not per round | a larger batch at the same LR is well-estimated but conservative, so per-round loss will look different for a reason that is not the model |

Why not simply raise `darl.inner_steps` on the fast site, which also fills the round:
drift scales with H and was already ~0.93 from a random init, where past 1 averaging
destroys progress. Accumulation fills the same time at the same drift. Full reasoning in
[FEDERATION_GUIDE.md § 3](FEDERATION_GUIDE.md).

Sites may be queued for hours; that is fine and needs no coordination. The server holds
the run and merges whoever is present.

---

## Part 3b — the central baseline, and making it a fair one

The baseline is the same config with the outer step switched off. It needs no central VM:
the job spawns its own throwaway DARL coordinator on `127.0.0.1`, seeded from the corpus
manifest with `space_seed = 42`, so it draws from the identical block-permuted index space
the DiLoCo run does.

```bash
# Snellius, 4 H100s
sbatch scripts/snellius/job_titan_central.sh
# LUMI, 8 GCDs
sbatch -A "$PWW_ACCOUNT" scripts/lumi/job_titan_central.sh
# the 1k short run, for validating the pair inside one queue slot
CONFIG=configs/titan/qwen3_0.6b_c4_central_1k.toml sbatch scripts/snellius/job_titan_central.sh
```

It deliberately does **not** lease from the shared coordinator. If it did, it would compete
with a live DiLoCo run for blocks: each would train on part of the corpus and neither would
be the run it claims to be.

`configs/titan/qwen3_0.6b_c4_central.toml` is `qwen3_0.6b_c4_diloco.toml` with one line
changed (`flower.enable = false`) plus `validation.freq = 100`, which exists because
`validation.freq` is *ignored* on the DiLoCo path — the central node evaluates once per
Flower round there, i.e. every `darl.inner_steps` steps, and 100 reproduces that cadence.
Diff the two before trusting any chart with both on it:

```bash
diff <(grep -vE '^\s*#|^\s*$' configs/titan/qwen3_0.6b_c4_central.toml) \
     <(grep -vE '^\s*#|^\s*$' configs/titan/qwen3_0.6b_c4_diloco.toml)
```

### Equal steps is not equal work

`steps = 20000` in both, so the baseline is **step-matched**. It is not token-matched, and
by default it cannot be, because the federation trains every site at once. Per global
optimiser step, `local_batch_size × seq_len × data-parallel ranks`:

| | ranks | tokens/step | at 20,000 steps |
|---|---|---|---|
| baseline, Snellius | 4 | 65,536 | ~1.31 B |
| baseline, LUMI | 8 | 131,072 | ~2.62 B |
| **DiLoCo, both sites** | **12** | **196,608** | **~3.93 B** |

So the two baselines are not even the same baseline as each other, and the federated run
has seen three times the Snellius baseline's tokens at the same step number. This is why
WandB's default x-axis here is `train/cum_tokens` and not the step.

For the **compute-matched** comparison — same steps *and* same tokens per step, which is the
baseline DiLoCo's own paper reports against — set the baseline's global batch to the
federation's 96 sequences. One variable, works on **both** sites, no config change:

```bash
# Snellius:  8 local × 4 ranks × 3 accum = 96
PWW_GLOBAL_BATCH=96 sbatch --export=ALL,PWW_GLOBAL_BATCH \
  scripts/snellius/job_titan_central.sh

# LUMI:      6 local × 8 ranks × 2 accum = 96
PWW_GLOBAL_BATCH=96 sbatch --export=ALL,PWW_GLOBAL_BATCH -A "$PWW_ACCOUNT" \
  scripts/lumi/job_titan_central.sh
```

`run_train.sh` derives the microbatch/accumulation split itself and prints it, lowering the
microbatch where the target does not divide — LUMI's 64/step cannot reach 96 by
whole-multiplier accumulation, which is why the older `PWW_GRAD_ACCUM=3` recipe was
Snellius-only. Accumulation is exact averaging, so the split does not change the
optimisation; the global batch is what the optimiser sees. The run's derived WandB name
gains a `-gb96` marker, and validation is untouched (`validation.local_batch_size` is its
own field and does not follow the training microbatch).

Two honest caveats. This matches the federation at **k=2**: rounds where a site was queued
or crashed ran the federation at 64 or 32 sequences, and no baseline setting mirrors
membership churn after the fact — `train/cum_tokens` absorbs it. And a matched baseline
consumes ~3× the corpus of a plain Snellius baseline (1.92M windows, ~70%), the same as the
DiLoCo run.

Both comparisons are worth having. Step-matched asks what one replica achieves in the same
number of steps; compute-matched asks what federating cost for the same amount of work.

### Turning WandB on

Off unless asked for, at every site and on the central VM. Any of `PWW_WANDB=1`,
`ENABLE_WANDB=1`, or simply setting `WANDB_PROJECT` switches it on:

```bash
# central VM -- start this BEFORE the sites so round 1 is captured
ENABLE_WANDB=1 WANDB_PROJECT=pww-diloco-1k \
AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml \
  ./scripts/central_node/start_central_services.sh

# each site
PWW_WANDB=1 WANDB_PROJECT=pww-diloco-1k WANDB_API_KEY="$WANDB_API_KEY" \
DARL_TOKEN="$DARL_TOKEN" \
  sbatch --export=ALL,PWW_WANDB,WANDB_PROJECT,WANDB_API_KEY,DARL_TOKEN \
    scripts/snellius/job_titan_diloco.sh
```

Run names are derived so a chart is readable without cross-referencing run ids:

| run | what it is |
|---|---|
| `central-<site>` | the single-node baseline |
| `diloco-<site>` | **one participant** of the federated run — its own tokens only |
| `central-aggregator` | the federated run **as a whole** |

**Compare a baseline against `central-aggregator`, not against a `diloco-<site>` run.** The
aggregator's `train/loss` is the token-weighted loss across every participant and its
`train/cum_tokens` is the federation total; a site reports only its own share, so overlaying
`diloco-snellius` on `central-snellius` by tokens makes the federated run look like it
reached a loss on a third of the data it actually used.

If you set `WANDB_ENTITY`, `run_train.sh` now also exports `WANDB_TEAM` (and vice versa) —
torchtitan's logger reads the latter and the aggregator the former, so setting only one used
to put the sites and the server in different entities, where they cannot share a chart at
all.

### Gate: the 1k pair must pass before the 20k submission

The `*_1k.toml` configs exist to buy this check cheaply: 10 outer rounds, one queue slot,
every moving part of the full run exercised. Submit the 1k DiLoCo pair (both sites) and the
1k baseline, and hold the 72-hour submission until **all four** of these pass. Each is a
grep against `runs/central/flower.log` on the central VM, so no dashboard is required.

Prerequisite: the held-out validation split is staged at **both** sites (Part 1 step 3) —
it is a one-time stage that every later run reuses, and check 3 below is meaningless
without it. A 1k pair run on the fixture rehearses a different eval path than the 20k run
will use.

**1. No non-finite contribution from either site.** This is the open blocker from a
previous run — one site's weights came back non-finite in the embedding tensor, which is
also the first tensor a scatter or vocab-padding bug at 8 ranks would hit. Its contribution
is dropped rather than merged, so the run *survives* it, but that site's compute is wasted
and the "comparison" is then one site training alone:

```bash
grep -iE "non-finite|excluded .* held-out" runs/central/flower.log   # must print NOTHING
grep -iE "NON-FINITE" logs/pww-*-titan-*.out                         # per site: nothing
```

If it fires, the site log's `step N: loss became NON-FINITE` line says which step — step 1
points at the weights the site was handed (the scatter path), a later step points at
optimisation. Do not submit the 20k run around it.

**2. Both sites merging, every round.** `merge round` advancing with 2 clusters and
nonzero tokens from each:

```bash
grep "cluster(s)" runs/central/flower.log | tail -5    # expect "2 cluster(s) [lumi, snellius]"
```

**3. Held-out spread quiet.** The two sites evaluate the same global weights on the same
staged validation windows, so their held-out losses must agree — the server warns above a
1-nat spread:

```bash
grep "held-out loss spread" runs/central/flower.log    # must print NOTHING
```

Also confirm both site logs printed `validation  : c4_local at ...` — if either fell back
to the fixture, the spread check compares different data and means nothing.

**4. Drift falling.** From a random init the opening rounds are legitimately large (1.69
then 0.93 measured); what matters is the trend across the 10 rounds:

```bash
grep -o "drift [0-9.]* (max [0-9.]*)" runs/central/flower.log
```

Read the max. Falling toward ~0.3 or below by round 10 is healthy; flat near 1 means the
outer average is destroying local progress and H or the outer momentum needs attention
before any 72-hour commitment ([FEDERATION_GUIDE § 3](FEDERATION_GUIDE.md)).

Then reset before the real run — the 1k pair leaves checkpoints, a spent lease table and a
seeded global model behind, all of which the 20k run must not inherit:

```bash
./scripts/reset_run.sh --yes                 # each site
# central VM:
./scripts/central_node/stop_central_services.sh
PWW_FRESH_RUN=1 ENABLE_WANDB=1 NUM_ROUNDS=400 WANDB_PROJECT=<20k project> \
  ./scripts/central_node/start_central_services.sh
```

---

## Part 4 — watch it

```bash
# central VM
tail -f runs/central/flower.log
tail -f runs/central/darl.log
./scripts/central_node/status_central_services.sh

# per site
tail -f logs/pww-lumi-titan-*.out
tail -f logs/pww-snellius-titan-*.out

# DARL coverage as JSON
curl -sS -H "X-DARL-Token: $(cat runs/darl/token)" \
  http://145.38.206.143:29510/status | jq .
```

Healthy looks like this — `merge round` advancing with nonzero tokens from every live
site:

```
  >> Training loss 2.9233 (ppl 18.60)  (2 cluster(s) [lumi, snellius], 19,660,800 tokens,
     drift 0.0068 (max 0.0071))
  >> Throughput 94,300 tok/s combined | lumi 41,500 tok/s, 31.4% MFU, 58.2 TFLOP/s/rank,
     48.3 GiB (76%); snellius 52,800 tok/s, 43.9% MFU, 434.6 TFLOP/s/rank, 61.2 GiB (77%)
  >> Round took 316s (slowest site's inner phase)
  >> Perplexity 18.79  (held-out loss 2.9333; per-cluster ppl [18.17, 19.11], 1,048,576 eval tokens)
round 138 merged from 2 cluster(s) in 41.3s
```

`MFU` and memory are per cluster and **not** averaged — an MFU against an MI250X GCD and
against an H100 are ratios to different peak FLOPs, so a mean of the two would describe
neither. `tok/s` *is* summed, because the sites train concurrently.

Four things to actually read, rather than everything:

| | |
|---|---|
| **merge round** | counts *successful merges*, not Flower attempts. If it is not advancing, nothing is training. |
| **tokens** | 0 means a site trained nothing and was not merged. |
| **drift (max)** | the number `H` is tuned against. Read the max, not the mean. |
| **ppl** | perplexity. Never appears under `accuracy`. |

[FEDERATION_GUIDE.md § 5](FEDERATION_GUIDE.md) explains each, and why the metric
arithmetic is less obvious than it looks.

---

## Part 5 — when it breaks

Full table in [FEDERATION_GUIDE.md § 6](FEDERATION_GUIDE.md). The five you will actually
hit:

| symptom | do this |
|---|---|
| server up, no rounds start | `min-clients` is above 1. Set it to 1 in the aggregator config. |
| client exits citing transport | `flower.transport` in the TOML disagrees with the server's `--transport`. Make them match. |
| `503 cluster_busy` at startup | another live process holds that cluster id. Pass `--replica`. If it is a requeue after a hard crash, it clears within one TTL by itself. |
| `digest mismatch` at registration | the two sites tokenised differently, or `SEED`/`NUM_SAMPLES` disagree. Re-check Part 1. |
| `every delta for round N was stale` | expected after a walltime kill. The next round is current. Nothing to do. |
| `401 bad or missing X-DARL-Token` | the site's `DARL_TOKEN` is not the value in `runs/darl/token`. Copy it across again. |

Nothing here needs the run to be restarted from scratch. A killed site requeues and
rejoins; a stopped aggregator restarts from `--state-dir` at the same merge round, and the
coordinator resumes its lease table from the same snapshot.

Those two are separate state dirs, and only one of them is guarded by the round counter,
so check both after any restart of the central services: the merge round in
`status_central_services.sh`, and the `restored coordinator` line in Part 2. A restart that
resumed the global model but wiped the lease table looks like a clean resume and quietly
trains the same windows twice.

---

## Sanity checks, no allocation needed

Run these after any change, and **repeatedly** — both bugs found in the last round of work
were invisible in a single run.

```bash
source env.sh
pww_run python3 tests/test_darl.py          # 46 checks
pww_run python3 tests/test_titan.py         # 19 -- needs the 2.9 env; on LUMI
                                            #    set PWW_CONTAINER, see Part 0
python3 tests/test_federation.py            # 26 -- central VM, no torchtitan needed
pww_run python3 tests/test_local.py         # 28
pww_run python3 tests/test_diloco_gloo.py   # 14
```

What these **cannot** cover, because it needs GPUs and a process group: FSDP2 wrapping,
the real Qwen3 forward pass, the DTensor gather/scatter, and the cluster-level metric
reduction. `configs/titan/qwen3_0.6b_smoke.toml` is the smallest thing that exercises
them; the open list is in [TODO.md](TODO.md) § 2.
