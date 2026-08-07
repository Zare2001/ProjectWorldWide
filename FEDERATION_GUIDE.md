# Running a multi-site DiLoCo job (LUMI + Snellius + central node)

Operations guide for the cross-site outer step: what to start on the central VM
(`145.38.206.143`), what to submit at each site, and how to tell from the logs
whether it is actually training.

There are **two client paths** and they share the whole central node:

| | inner loop | model | config | when |
|---|---|---|---|---|
| **torchtitan** | `pww.titan.train` | Qwen3, FSDP2 | `configs/titan/*.toml` | new work; the only path that scales past ~1B |
| **legacy HF** | `pww.train_llm_flower` | GPT-2 / LLaMA via `AutoModel` | `configs/llm_*.yaml` | existing runs; torch 2.7.1 |

The torchtitan path needs its own environment (torch >= 2.9) — see
[scripts/titan/README.md](scripts/titan/README.md) and set that up first.

---

## 1. What runs on the central node

Three daemons. The third only starts when you ask for blob transport.

| service | protocol | port | needed for |
|---|---|---|---|
| DARL lease coordinator | HTTP/JSON | `29510` | always |
| Flower aggregator (`PWWFedMom`) | gRPC | `29511` | always |
| Blob store | HTTP | `29512` | models above ~1B only |

Security-group rules must open these to **subnets, not single IPs** — LUMI
`193.167.209.128/26`, Snellius `145.136.0.0/16` — because login and compute nodes
have different addresses within them.

### The aggregation strategy is in this repo

`src/pww/central/strategy.py` implements `PWWFedMom` directly on top of Flower's
`FedAvg`. It does **not** use the `FedMom` strategy from the
`Zare2001/flower@fedmom-strategy` fork: that one keeps the global weights and the
momentum buffer in Python attributes, which rules out surviving a restart, starting
a run before any site connects, and out-of-band weight transport. If the fork is
installed, the server logs that it is being ignored and carries on. Plain upstream
`flwr` is enough.

### Start, check, stop

```bash
cd ~/ProjectWorldWide

# inline transport: weights inside the gRPC message. Fine to ~1B parameters.
NUM_SAMPLES=<windows> BLOCK_SIZE=1024 SEED=42 \
AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml \
  ./scripts/central_node/start_central_services.sh

# blob transport: weights out of band over HTTP. Required above ~1B.
TRANSPORT=blob NUM_SAMPLES=<windows> BLOCK_SIZE=1024 SEED=42 \
AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml \
  ./scripts/central_node/start_central_services.sh

./scripts/central_node/status_central_services.sh   # ports, merge round, membership
./scripts/central_node/stop_central_services.sh
cat runs/darl/token                                 # the token the sites need
```

`NUM_SAMPLES` is the **window** count printed by `scripts/titan/tokenize_c4.sh`,
not a token or document count, and `SEED` must match `darl.space_seed` at every
site. A disagreement is refused at registration by `BlockSpace.digest` rather than
silently producing two different meanings for position *p*.

### Which transport, and why the choice exists

Inline transport all-gathers the full parameter set onto every rank and puts it in
one gRPC message, capped at **2,147,483,647 bytes** (`2**31 - 1`). That is a protocol
limit; no `flower.max_message_length` setting moves it. Dividing through, the ceiling
is exactly:

| wire dtype | ceiling |
|---|---|
| `float16` | 1,073,741,823 parameters |
| `float32` | 536,870,911 parameters |

Measured against the flavors this repo ships configs for, built on the meta device
with the 128k tokenizer's vocabulary padded to 131,328:

| flavor | actual parameters | largest tensor | fp16 wire | fp32 wire | inline? |
|---|---|---|---|---|---|
| 0.6B | 709,427,200 | (131328, 1024) | 1.3 GiB | 2.6 GiB | **fp16 only** |
| 1.7B | 1,947,329,536 | (131328, 2048) | 3.6 GiB | 7.3 GiB | no |
| 4B | 4,305,911,296 | (131328, 2560) | 8.0 GiB | 16.0 GiB | no |
| 8B | 8,021,914,624 | (131328, 4096) | 14.9 GiB | 29.9 GiB | no |
| 14B | 14,557,281,280 | (131328, 5120) | 27.1 GiB | 54.2 GiB | no |
| 32B | 32,551,097,344 | (131328, 5120) | 60.6 GiB | 121.3 GiB | no |

Two things there are worth knowing before you pick a transport:

- **Only the 0.6B flavor fits inline at all, and only in float16.** Its float32 wire
  is 2.6 GiB, over the cap — so `flower.wire_dtype = "float32"` is not a
  precision-for-bandwidth trade at this size, it simply does not fit.
- **The names understate the sizes.** The flavor called 0.6B is 709M parameters here
  and 1.7B is 1.95B, because a 131,328-row embedding and output projection are much
  larger than Qwen's own defaults intend. At 0.6B the embedding and output projection
  together are ~38% of the model.

Blob transport instead streams **one tensor at a time**: the client writes
`local - global` per tensor to a file and HTTP-PUTs it; the server merges per key.
Peak memory then tracks the largest single tensor — the embedding — not the model,
so it stops growing once the embedding dimension does:

| flavor | largest tensor, fp32 | streaming merge peak | dense equivalent (2 sites) |
|---|---|---|---|
| 0.6B | 513 MiB | ~2.0 GiB | 13.2 GiB |
| 8B | 2.0 GiB | ~8.0 GiB | 149.4 GiB |
| 14B | 2.5 GiB | ~10.0 GiB | 271.2 GiB |
| 32B | 2.5 GiB | ~10.0 GiB | 606.3 GiB |

14B and 32B share a largest tensor, so their streaming peak is identical — which is
the property that makes this approach scale at all. The dense column is what holding
the global model, the momentum buffer, the weighted mean and one delta per site whole
in float32 would cost; that is the wall this replaces.

Disk, not RAM, becomes the binding constraint. `GlobalState.log_disk_budget` prints
the requirement at startup and logs an error if the volume cannot hold it, because
the alternative is a merge that fails partway through a round.

Both sides must agree. Set `flower.transport` in the run's TOML to match the
server's `--transport`, or the client refuses the round and says so. This is a
startup error on purpose: the mismatch is unfixable mid-round, because the inline
path needs a parameter ordering built from exactly the gather that blob transport
exists to avoid.

---

## 2. Elastic membership: 0, 1 or N live sites

`min-clients: 1` in `configs/central_aggregator_titan.yaml`. This is the difference
between a run that survives HPC queues and one that does not, and it is worth being
precise about why.

`min_clients` gates whether a round *starts*, and Flower blocks inside the client
manager until that many are connected. `round-timeout` does **not** bound that wait
— it only bounds waiting for results once a round has begun. So `min-clients: 2`
means that while Snellius sits in the queue for six hours, LUMI connects and idles
and the run makes no progress at all.

With `min-clients: 1` and `--state-dir` set:

| live replicas | behaviour |
|---|---|
| **0** — everything queued | the server holds the run and waits. Nothing is lost, including across a restart of the aggregator process itself. |
| **1** | that site trains alone. This is DiLoCo with k=1 — correct, not degraded — and the momentum buffer keeps accumulating across the gap. |
| **N** | the ordinary case. |

Two mechanisms make joining and leaving safe:

- **`configure_fit` hands every participant the current global weights before it
  trains**, so a site joining at round 400 cannot contribute an update derived from
  a stale or freshly-initialised model. There is no special path for late joiners.
- **Generation checking.** Every delta records the `base_round` it was computed
  against, and the merge refuses any that is not current. A site killed at walltime
  mid-round, requeued hours later, has that first delta rejected rather than
  averaged in stale. Its next round is current and contributes normally.

On a cold start with no state on disk, the server asks exactly **one** site to
upload its initial weights and adopts them as the global model — the central node
cannot invent an architecture. Every later site is checked against those keys and
shapes, so a mismatched `model.flavor` or tokenizer vocabulary fails loudly instead
of averaging tensors that do not correspond.

### Quorum

There is deliberately no quorum service. Quorum exists in peer-to-peer schemes
(torchft's Lighthouse) because the participants must agree on a membership set
before entering a collective — with an all-reduce, a rank that thinks the group has
3 members when it has 2 hangs. Here the outer step is a merge on one node reading
files, not a collective: the server already knows exactly who uploaded, and
`base_round` plays the role `quorum_id` plays there. Adding consensus would buy
nothing and add a process that can fail.

---

## 3. The outer step

The two numbers in `configs/central_aggregator_titan.yaml` **are** DiLoCo's
`OuterOpt`, at the paper's own values (arXiv 2311.08105 §4):

```yaml
server-learning-rate: 0.7    # eta
server-momentum: 0.9         # beta
```

The update the server applies is

```
v_next = w - eta * (w - w_avg)                 # w - eta * Delta
w_next = v_next + beta * (v_next - v_prev)
```

which is not merely momentum-like — it is *algebraically* Nesterov momentum.
Substituting `m = v_t - v_(t-1)` gives `m_next = beta*m - eta*Delta` with `Delta`
evaluated at `w = v + beta*m`, i.e. Nesterov's accelerated gradient in two-sequence
form. It matches `torch.optim.SGD(momentum=beta, nesterov=True)` fed DiLoCo's outer
gradient, including PyTorch's first-step convention.

Measured rather than argued, both sides driven by an identical prescribed sequence of
outer gradients so that any divergence is the update rule alone:

| | max abs difference |
|---|---|
| ours vs `SGD(momentum=0.9, nesterov=True)` | **1.9e-06** |
| ours vs `SGD(momentum=0.9, nesterov=False)` — heavy ball | **4.9e-01** |

over 6 rounds on weights of magnitude ~4, with non-uniform token weights. The
heavy-ball row is the one that makes the first meaningful: without it, an
implementation where momentum did nothing at all would also "match".
`tests/test_federation.py` pins both, so this cannot drift silently.

Separately, the streaming per-tensor merge agrees with a dense reference
implementation to **4.8e-07** over 3 rounds — i.e. splitting the merge tensor by
tensor to bound memory costs nothing numerically.

`server-momentum: 0.0` with `server-learning-rate: 1.0` collapses the outer step to
plain FedAvg parameter averaging, `w_next = w_avg`. That is the control arm for any
DiLoCo-vs-FedAvg comparison, and it is what the earlier WikiText run was actually
doing while its config claimed FedMom.

Two deliberate departures from the paper:

- **Deltas are weighted by tokens contributed**, `p_i = tokens_i / sum tokens`, not
  uniformly by `1/k`. With 8 MI250X GCDs against 4 H100s the sites do different
  amounts of work per round, and uniform averaging would under-weight whichever did
  more. Reduces exactly to `1/k` when token counts are equal.
- **`k` varies between rounds**, per the elastic membership above. The paper assumes
  it fixed.

---

## 4. Submitting at each site

Both sites need `DARL_TOKEN` from the central node.

```bash
# check reachability from the login node first
export DARL_TOKEN="<token>"
curl -sS -H "X-DARL-Token: $DARL_TOKEN" http://145.38.206.143:29510/health
# {"ok": true, "epoch": 0}
nc -zv 145.38.206.143 29511
nc -zv 145.38.206.143 29512      # blob transport only
```

### torchtitan path

```bash
# LUMI
DARL_TOKEN="<token>" sbatch scripts/lumi/job_titan_diloco.sh
# Snellius
DARL_TOKEN="<token>" sbatch scripts/snellius/job_titan_diloco.sh
```

Both default to `configs/titan/qwen3_0.6b_c4_diloco.toml`. For a scaling run pass a
different config; `configs/titan/qwen3_8b_c4_diloco.toml` is already set to blob
transport. Any `[darl]` or `[flower]` field can be overridden on the command line as
`--darl.<field>` / `--flower.<field>` without editing the TOML.

Before the first run, stage the corpus once — the tokenisation pass is offline and
one-off:

```bash
./scripts/titan/download_tokenizer.sh     # OpenEuroLLM 128k
./scripts/titan/stage_c4.sh               # C4 shards
./scripts/titan/tokenize_c4.sh            # -> token shards + manifest + window count
```

That window count is the `NUM_SAMPLES` the central node needs.

### Legacy HuggingFace path

```bash
DARL_TOKEN="<token>" sbatch scripts/snellius/job_flower_diloco_llm.sh
DARL_TOKEN="<token>" sbatch scripts/lumi/job_flower_diloco_llm.sh
```

If Snellius compute nodes cannot reach the central VM directly, tunnel from the
login node:

```bash
ssh -f -N -g -L 29510:145.38.206.143:29510 \
              -L 29511:145.38.206.143:29511 \
              -L 29512:145.38.206.143:29512 145.38.206.143
```

---

## 5. Reading the logs

```bash
tail -f runs/central/flower.log     # outer rounds
tail -f runs/central/darl.log       # leasing
./scripts/central_node/status_central_services.sh
```

What to look for, and what each line actually tells you:

```
transport=blob | server_learning_rate=0.7, server_momentum=0.9 |
  min_clients=1 (elastic: trains with whoever is available) |
  num_rounds=200 (attempts), round_timeout=1800s
resumed global state from runs/central/global: round 137, 291 tensors,
  4,182,441,984 tokens, clusters ['lumi', 'snellius']
merge round 138: 100 steps, 1,048,576 tokens, loss 2.9143 (ppl 18.44),
  drift 0.0071, 24310 tok/s, 4 blocks committed
round 138 merged from 2 cluster(s) in 41.3s (peak ~2.1 GiB, lr=0.7, momentum=0.9)
```

- **`merge round` is not Flower's round number.** It counts *successful merges*. A
  Flower round in which every site was killed at walltime consumes a round number
  and changes nothing; conflating the two makes "how much training has actually
  happened" unanswerable from the logs.
- **`tokens` per round is the honest number.** A round that trained nothing reports
  0 and is not merged. This matters: the earlier WikiText run reported `loss 0.0`
  on 1 sample for 23 consecutive rounds because a `max(1, ...)` floor turned "the
  corpus is exhausted" into "one sample", and the global model sat frozen while the
  log showed no failures.
- **`drift`** is `||local - global|| / ||global||` per round. It is the number that
  tells you whether `H` is sensible: near zero means the inner loop is barely
  moving, and large means the replicas have diverged far enough that averaging them
  is losing information.
- **`ppl`** is the perplexity of the training loss. Evaluation reports perplexity
  under its own name, not as `accuracy` — the old path reported a perplexity of 30
  in a field labelled accuracy.

DARL coverage:

```bash
curl -sS -H "X-DARL-Token: $(cat runs/darl/token)" \
  http://145.38.206.143:29510/status | jq .
```

`unassigned` + `leased` + `committed` + `quarantined` must equal the block count at
all times; the coordinator asserts this internally.

---

## 6. Troubleshooting

### Running two jobs at the same site at once

Both facilities allow partial-node allocations, so submitting two jobs to one site can
put them on the same node. That is fine for the rendezvous — `run_train.sh` derives the
torchrun port from the Slurm job id on multi-node runs and asks the kernel for a free
one on single-node runs — but it is **not** fine for the DARL cluster id, and this is a
correctness issue rather than a performance one.

The cluster id defaults to the site name alone (`lumi`), deliberately: a job killed at
walltime and requeued must be recognised as the same cluster so it keeps the measured
throughput that sizes its grants. `SLURM_JOB_ID` changes on requeue, so it cannot be
part of the id — which means nothing in the environment is *both* stable across a
requeue *and* unique across concurrent jobs.

**So pass `--replica` when you run two jobs at one site:**

```bash
scripts/titan/run_train.sh --replica a --config ...   # cluster id "lumi-a"
scripts/titan/run_train.sh --replica b --config ...   # cluster id "lumi-b"
```

That is the path to be on. Forgetting it used to corrupt the run silently; it is now
refused instead, by three independent guards.

**1. Registration refuses a second live process.** Each session generates a random
*incarnation* id and sends it with `/register`. A register under a different
incarnation is judged by whether the incumbent is still alive — heartbeating, or
holding leases:

| incumbent | verdict |
|---|---|
| stale | a **requeue**. Take over, keep the record, so committed count, EWMA rate and cursor survive exactly as before |
| live | a **second concurrent job**. Refuse with `503 cluster_busy`, naming `--replica` |

503 rather than 409 because it clears on a timer, so the client's existing retry loop
keeps trying. The honest cost: after a *hard* crash (OOM kill, node failure — no
SIGTERM, no release) the requeued job waits out one TTL before it can take over,
because dead and merely slow are indistinguishable until the leases expire. A bounded
wait replacing silent duplicate training, and `--replica` avoids it entirely.

**2. `/release` with no lease id is scoped to the incarnation**, not just the id. It is
the one operation whose blast radius is every lease a name owns, so a process that
does not currently hold the id cannot hand back leases it never had.

**3. `aggregate_fit` drops a round's contributions if two clients claim one id.** This
closes a path the coordinator cannot see: `pww_qwen3_local` uses torchtitan's own
dataloader, so a config with `flower.enable = true` and no DARL never registers. There
the failure is not a lost round but a *wrong* one — delta blobs are named
`(run, round, cluster)`, so both clients wrote the same object and one overwrote the
other; two contributions then point at one file and the merge counts the survivor
twice with the combined weight of both. The other sites' round still proceeds; only
the duplicated id is dropped, because with one file and two token counts there is no
correct weight to give it.

All of it is pinned in `tests/test_darl.py` and `tests/test_federation.py`: second
concurrent refused, requeue after a clean exit, requeue after a hard crash, release
scoping, and the duplicated-id round.

### Failures

| symptom | cause | what to do |
|---|---|---|
| server waits, no rounds start | `min-clients: 2` in the aggregator config | set it to 1. Only require 2 if you specifically want to force both sites. |
| client refuses the round citing transport | `flower.transport` in the TOML disagrees with the server's `--transport` | make them match; restart whichever is wrong. |
| `every delta for round N was stale` | all participating sites were requeued and computed against an older global model | expected after a walltime kill. The next round is current. |
| a site hangs at the end of an epoch | was: a prefetched DARL lease nobody consumed, so the pool looked drained while the session held the missing blocks | fixed — `LeaseSession.acquire` now consumes a pending prefetch instead of waiting on it. Covered by `tests/test_darl.py`. |
| `blob store: 507` | the volume behind `--blob-root` is below its reserve | point `--state-dir`/`BLOB_ROOT` at a larger volume. `GlobalState.log_disk_budget` prints the requirement at startup. |
| lease expiry after a walltime kill | Slurm killed a site mid-epoch | self-healing: uncommitted blocks return to the pool on TTL expiry, and the surviving site finishes the epoch. Releasing on SIGTERM makes it milliseconds instead of a full TTL. |
| `Connection refused` on 29511 | daemons not running | `./scripts/central_node/start_central_services.sh` |
| `503 cluster_busy` at startup | another live process holds this cluster id — a second concurrent job at the same site, or a requeue whose predecessor died hard and whose leases have not expired yet | pass `--replica a` / `--replica b` to give each job its own id. If it is a requeue, it clears itself within one TTL. See above. |
| `N clients reported as cluster X` in the aggregator log | two Flower clients using one cluster id, on a run with no DARL coordinator to refuse it | same fix. That round's contributions under that id were dropped rather than merged wrongly. |

---

## 7. What is verified, and on what

CPU-only, no allocation needed. At a site, through the usual environment:

```bash
source env.sh
pww_run python3 tests/test_darl.py
pww_run python3 tests/test_titan.py        # needs third_party/torchtitan on the path
```

`test_federation.py` needs neither torchtitan nor GPUs — only `torch` and `flwr` —
so it runs on the central node directly:

```bash
python3 tests/test_federation.py
```

| suite | checks | covers |
|---|---|---|
| `test_darl.py` | 45 | lease state machine with an injected clock; a real coordinator over a socket; exactly-once coverage under concurrent clusters; the prefetch/acquire race; incarnation, requeue and release scoping |
| `test_titan.py` | 19 | the token shard format, the DARL dataloader's exactly-once coverage across ranks, the inline wire codec, config feasibility |
| `test_federation.py` | 23 | blob store over real HTTP; 0/1/N live replicas; restart durability; stale-delta rejection; mismatched-model refusal; duplicated cluster ids; the outer step against `SGD(nesterov=True)` |
| `test_local.py` | 28 | config parsing, checkpointing, the single-site pieces |
| `test_diloco_gloo.py` | 14 | the DiLoCo collectives over multi-process gloo, two replica layouts |

Run them **repeatedly**, not once. Both bugs fixed in this round of work were
invisible in a single run: the DARL lease deadlock surfaced in roughly 1 run in 10,
and a fixed rendezvous port in `test_diloco_gloo.py` in 1 in 35. A green single run is
not evidence of stability. The last full sweep was 50 runs (10 per suite) with 0
failures.

What these **cannot** cover, because it needs GPUs and a process group: FSDP2
wrapping, the real Qwen3 forward pass, and the DTensor gather/scatter in
`titan/params.py` and `delta.py`. `configs/titan/qwen3_0.6b_smoke.toml` is the
smallest thing that exercises those.
