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
# 1. Clone and set up scratch/symlinks. Login node.
git clone <repo> ~/ProjectWorldWide && cd ~/ProjectWorldWide
git submodule update --init --recursive        # third_party/torchtitan
./scripts/bootstrap.sh                          # scratch dirs, logs/, data/ + runs/ symlinks
source env.sh && pww_summary                    # confirm site detection and paths
```

The torchtitan path needs **its own** torch >= 2.9 environment, separate from this repo's
2.7.1 pin. Read [scripts/titan/README.md](scripts/titan/README.md) before running either:

```bash
# 2a. Snellius: a second venv alongside the 2.7.1 one.
./scripts/titan/setup_venv_snellius.sh                 # -> ~/venvs/pww-titan-snellius

# 2b. LUMI: a container, since LUMI's own image is on 2.7.1. ~30-60 min, batch job.
sbatch scripts/lumi/build_titan_container.sh           # -> $PWW_SCRATCH/containers/pww-titan.sif
```

Verify before spending a queue slot:

```bash
PYTHONPATH="$PWD/src:$PWD/third_party/torchtitan" python3 tests/test_titan.py
PYTHONPATH="$PWD/src:$PWD/third_party/torchtitan" python3 tests/test_darl.py
```

---

## Part 1 — stage the corpus (login node, needs internet)

Compute nodes at both sites have **no internet**, so everything the run reads must be on
scratch first. Do this once per `(corpus, tokenizer, seq_len)`.

```bash
source env.sh

# 1. Tokenizer. Prints the vocab size that becomes the embedding.
./scripts/titan/download_tokenizer.sh                  # -> $PWW_DATA_DIR/tokenizers/tokenizer-128k

# 2. Tokenise. Pick ONE line.
./scripts/titan/tokenize_c4.sh --dataset c4_test --seq-len 2048        # fixture, seconds
./scripts/titan/tokenize_c4.sh --dataset c4 --seq-len 2048 --max-files 32   # real, ~5B tokens

# optional: keep the raw text so re-tokenising does not re-download
./scripts/titan/stage_c4.sh --files 32
./scripts/titan/tokenize_c4.sh --dataset c4_local --seq-len 2048
```

**Write down two numbers the last command prints.** Both are needed later and a mismatch
is a startup failure, not a silent problem:

| | used for |
|---|---|
| **windows** | `NUM_SAMPLES` on the central VM |
| **manifest digest** | must match at both sites |

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

---

## Part 2 — start the central VM

Runs on `145.38.206.143`. Must be up **before** either site's job starts training.

```bash
cd ~/ProjectWorldWide

# Pick the transport by model size. 0.6B -> inline. Anything larger -> blob.
# FEDERATION_GUIDE.md #1 has the measured per-flavor table.

# <= ~1B parameters
NUM_SAMPLES=<windows> BLOCK_SIZE=1024 SEED=42 \
AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml \
  ./scripts/central_node/start_central_services.sh

# above ~1B -- also brings up the blob store on 29512
TRANSPORT=blob NUM_SAMPLES=<windows> BLOCK_SIZE=1024 SEED=42 \
AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml \
  ./scripts/central_node/start_central_services.sh
```

Then confirm, and take the token the sites need:

```bash
./scripts/central_node/status_central_services.sh     # ports, merge round, membership
cat runs/darl/token
```

`SEED` must equal `darl.space_seed` in the run's TOML, and `NUM_SAMPLES` the window count
from Part 1. Disagreement is refused at registration by the block-space digest.

To stop:

```bash
./scripts/central_node/stop_central_services.sh
```

State under `--state-dir` survives that, so stopping and restarting the aggregator does
**not** lose the run. See [FEDERATION_GUIDE.md § 2](FEDERATION_GUIDE.md).

---

## Part 3 — submit at each site

Check reachability from the login node first — a firewall problem found here costs
seconds instead of a queue wait:

```bash
export DARL_TOKEN="<from cat runs/darl/token>"
curl -sS -H "X-DARL-Token: $DARL_TOKEN" http://145.38.206.143:29510/health
# {"ok": true, "epoch": 0}
nc -zv 145.38.206.143 29511
nc -zv 145.38.206.143 29512      # blob transport only
```

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

Sites may be queued for hours; that is fine and needs no coordination. The server holds
the run and merges whoever is present.

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
merge round 138: 100 steps, 1,048,576 tokens, loss 2.9143 (ppl 18.44), drift 0.0071
round 138 merged from 2 cluster(s) in 41.3s (peak ~2.1 GiB, lr=0.7, momentum=0.9)
```

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

Nothing here needs the run to be restarted from scratch. A killed site requeues and
rejoins; a stopped aggregator restarts from `--state-dir` at the same merge round.

---

## Sanity checks, no allocation needed

Run these after any change, and **repeatedly** — both bugs found in the last round of work
were invisible in a single run.

```bash
source env.sh
pww_run python3 tests/test_darl.py          # 46 checks
pww_run python3 tests/test_titan.py         # 19
python3 tests/test_federation.py            # 26 -- central VM, no torchtitan needed
pww_run python3 tests/test_local.py         # 28
pww_run python3 tests/test_diloco_gloo.py   # 14
```

What these **cannot** cover, because it needs GPUs and a process group: FSDP2 wrapping,
the real Qwen3 forward pass, the DTensor gather/scatter, and the cluster-level metric
reduction. `configs/titan/qwen3_0.6b_smoke.toml` is the smallest thing that exercises
them; the open list is in [TODO.md](TODO.md) § 2.
