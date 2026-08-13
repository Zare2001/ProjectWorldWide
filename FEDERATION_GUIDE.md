# Running a multi-site DiLoCo job (LUMI + Snellius + central node)

**Looking for the steps? [RUNBOOK.md](RUNBOOK.md).** That is the ordered command path,
first-time setup included. This file is the reference behind it: what each knob means,
why the choices are what they are, and how to read the output. Every runbook step links
back to a section here, so the reasoning lives in one place and the commands in the other.

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

### The central node needs one number from the corpus, and nothing else

It has no GPU, never opens a shard, and never loads a tokenizer. What it does need is the
window count, because that is the size of the index space DARL partitions — and it must be
the *same* space every site computes, or exactly-once leasing means nothing.

That makes the count a genuine cross-machine dependency, and the only one this node has on
the data. `BlockSpace.digest` covers `(num_samples, block_size, seed)`, so a mismatch is
caught — but at registration, which is after a site has queued, been scheduled and started.
The check is doing its job; the cost is a wasted queue wait.

So carry the number in a file rather than by hand. `MANIFEST=` reads `num_windows` out of
the manifest the tokenisation already wrote:

```bash
# once, from a site login node -- a few hundred bytes
scp "$PWW_DATA_DIR/c4-tokenizer-128k-2048/manifest.json" <central>:/tmp/

MANIFEST=/tmp/manifest.json BLOCK_SIZE=1024 SEED=42 \
AGGREGATOR_CONFIG=configs/central_aggregator_titan.yaml \
  ./scripts/central_node/start_central_services.sh
```

`NUM_SAMPLES` still wins if both are set, and the script echoes which it used. This is also
why the manifest is worth keeping rather than treating as a build artefact: it is the
authoritative record of what the block space *should* be, and the file whose digest both
sites are compared against.

Two notes on getting this wrong, since neither is obvious:

- **A shorter corpus is not caught by the digest** if `NUM_SAMPLES` was set to match it.
  The digest proves the sites agree with each other and with the coordinator; it cannot
  prove the number describes the corpus you meant. Reading it from the manifest is what
  closes that gap.
- **Re-tokenising changes the count.** `--max-files 32` and `--max-files 64` are different
  block spaces, so the coordinator must be restarted with `PWW_FRESH_RUN=1` and the new
  count. A resume would refuse, which is the good outcome; the bad one is passing the new
  count *and* the flag while a site still holds the old shards.

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

There are **two** durable state dirs, and only the model's is guarded by a round counter:
`runs/central/global` holds the global weights and the momentum buffer, `runs/darl` holds
the lease table. Restarting the central services resumes both, but they resume
independently, and a lease table that came back empty is not detectable from the model
side — the merge round continues from where it stopped either way, and the run trains
windows it has already trained. The coordinator prints `restored coordinator from
.../snapshot.json` when it resumed; that line is the check.

`PWW_FRESH_RUN=1` is what deliberately discards state, and it discards **both** stores
together. `DARL_FRESH=1` alone resets only the lease table and leaves the model and its
momentum buffer, which were trained against the previous block space -- so it warns. Neither
is the default, precisely because the failure they cause is silent.

Discarded state is renamed to `*.superseded` rather than deleted. That is not tidiness: both
flags used only to *skip loading* the old state while leaving it on disk, so a fresh
coordinator that died before its first snapshot came back resumed from the run it was meant
to replace.

Nothing infers that a run is dead, and nothing should. Resume is the default and is always
safe, which is what makes a VM reboot or every site sitting in the queue a non-event -- see
[RUNBOOK.md](RUNBOOK.md) "It never decides this for you" for the three timescales and which
of them are automatic.

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

### A round with one contributor takes the full step

With a single cluster there is nothing to average: `w_avg` is that cluster's own weights,
so the update reduces to

```
w_next = (1 - eta) * w + eta * w_local
```

and `eta < 1` is pure damping. It buys variance reduction across replicas, and there are
no replicas — at the paper's `eta = 0.7` it discards **30% of the round's local progress
for nothing**. So the outer step uses `eta = 1` when exactly one cluster contributed,
which makes such a round exactly "adopt that cluster's weights".

This is not a tuning choice, it is the degenerate case of the formula, and it matters
because solo rounds are the *normal opening* of a run: `min-clients: 1` exists so the
first site out of the queue starts training instead of idling, and a run frequently spends
its first hours with `k = 1`. It is also what makes the "DiLoCo with k=1 — correct, not
degraded" claim above actually true; before this, those rounds were degraded.

Momentum is deliberately left alone. Round 1 is momentum-free by construction
(`v_prev = w`), so this alone makes the opening round lossless, and changing `beta` too
would alter the behaviour of a long run that merely loses a site for a while.

`--no-solo-full-step` restores the old behaviour. The reason you might want it: `eta < 1`
bounds how far a single cluster can drag the global model, which is a real property if you
do not trust a site. The non-finite check and the drift metric are the intended guards for
that, so it is off by default.

Deliberate departures from the paper:

- **Deltas are weighted by tokens contributed**, `p_i = tokens_i / sum tokens`, not
  uniformly by `1/k`. With 8 MI250X GCDs against 4 H100s the sites do different
  amounts of work per round, and uniform averaging would under-weight whichever did
  more. Reduces exactly to `1/k` when token counts are equal.
- **`k` varies between rounds**, per the elastic membership above. The paper assumes
  it fixed.
- **Training starts from a random initialisation.** DiLoCo's Algorithm 1 takes an initial
  *pretrained* `theta^(0)`, and the outer values above were tuned in that setting — with
  `H = 500`, uniform `1/k` averaging and all replicas starting together. From scratch the
  first rounds behave differently: a measured opening round had
  `drift = ||local - global|| / ||global|| = 1.69`, i.e. the local update was larger than
  the weights themselves, which is what you would expect while the weights are still near
  initialisation. It fell to 0.93 by the next round. Worth knowing that `drift_ratio` is
  this repo's instrument and not the paper's, so there is no published threshold to
  compare it against — but an update exceeding the norm of the weights means averaging is
  not combining progress, so the early rounds are the ones to watch. The cheap levers are
  a smaller `darl.inner_steps` for the opening rounds, or `server-momentum: 0.0` until
  drift settles.
- **Gradient accumulation on the faster site, so nobody idles.** This is the preferred
  answer to unequal throughput, and it is worth understanding before the per-site H
  option below.

  **The barrier is not removable.** DiLoCo's outer step averages every participant's
  delta, so a round cannot close until the slowest site delivers. That is the algorithm.
  What *is* removable is the **idling** at that barrier.

  Measured on a real round: Snellius did 37 s of work inside a 154 s round and then sat
  still for 118 s — **76% of every round** — so hardware sustaining 256k tok/s produced a
  run averaging 101k tok/s.

  **What gradient accumulation does.** An optimiser step normally means one batch in,
  forward, backward, update. With `grad_accum = N` it means *N* smaller microbatches in,
  forward and backward on each, gradients **summed**, then **one** update:

  ```
  grad_accum = 1    [batch] -> fwd/bwd -> UPDATE                       1 update,  32 windows
  grad_accum = 4    [mb][mb][mb][mb] -> fwd/bwd each, sum -> UPDATE     1 update, 128 windows
  ```

  So the site consumes N× the data per optimiser step and takes N× as long per step —
  which is exactly what fills the wait. At `grad_accum = 4`, Snellius's 100 steps take
  ~146 s instead of 37 s and it arrives when LUMI does, having trained 4× the tokens.

  | | round | tokens merged | effective |
  |---|---|---|---|
  | `grad_accum = 1` (now) | 175 s | 19.7 M | 112k tok/s |
  | `grad_accum = 2` | 175 s | 26.2 M | ~150k tok/s |
  | `grad_accum = 4` | 175 s | **39.3 M** | **224k tok/s** |

  **Three things it does not disturb**, which is why it is preferred over raising H:

  | | |
  |---|---|
  | **drift** | unchanged. `drift = ‖local − global‖ / ‖global‖` measures how far the weights moved, and they move once per *optimiser* step — still H of them. Accumulation improves each step's gradient estimate; it does not take more steps or larger ones. Raising H multiplies drift, and drift was already ~0.93 from a random initialisation, where past 1 averaging destroys rather than combines progress. |
  | **the LR schedule** | unchanged. Identical H at every site means the schedule advances identically, so there is no per-site H and nothing to align. |
  | **peak memory** | unchanged. Microbatches run *sequentially*, so only one set of activations is live at a time. This is the entire reason accumulation exists. |

  **One thing it did disturb: the reported loss.** torchtitan wraps its loss function in
  `rescale_accumulated_loss(loss_fn, N)` so that summing N microbatch backwards produces a
  mean rather than a sum — which means every value `forward_backward_step` returns is
  **already divided by N**. torchtitan's own metrics path compensates with a `torch.sum`;
  this repo's per-round path took a *mean* over microbatches instead, and so reported the
  true loss divided by N. Silent at `grad_accum = 1` and wrong by exactly N otherwise.

  It looked like this, at `PWW_GRAD_ACCUM=5`:

  ```
  >> Training loss 1.9718 (ppl 7.18)   ...   8,192,000 tokens
  >> Perplexity 4225.88  (held-out loss 8.3490)
  ```

  A training loss 6.4 nats *below* the held-out loss on the same weights is not a
  generalisation gap; nothing generalises in that direction. The real figure was
  1.9718 × 5 = 9.859, which is exactly what 100 steps from a random initialisation should
  average. Fixed in `titan/trainer.py`, which now scales by
  `trainer.gradient_accumulation_steps` and logs the per-step mean rather than the last
  microbatch.

  Worth noting *why* it mattered beyond cosmetics: the two sites ran different
  `PWW_GRAD_ACCUM` values, so their reported losses were divided by different numbers and
  the cross-site comparison in §5 was meaningless while it lasted. Held-out loss was never
  affected — validation runs under `no_rescale()`.

  **How to set it.** Per site at submit time, because both sites share one TOML and the
  correct batch depends on the site's rank count:

  ```bash
  PWW_GRAD_ACCUM=2 DARL_TOKEN="..." sbatch --export=ALL,PWW_GRAD_ACCUM,DARL_TOKEN \
    scripts/snellius/job_titan_diloco.sh
  ```

  `run_train.sh` turns that into `--training.global_batch_size = N × local_batch_size ×
  nproc` — 64 for 2× on Snellius's 4 ranks, 128 for 2× on LUMI's 8 — and prints what it
  chose. Expressed as a multiplier rather than a batch size precisely because the batch
  that gives "2×" differs per site. DARL needs no change: `blocks_for_phase` already takes
  `grad_accum`, so each lease grows with the phase.

  **The cost, stated plainly.** A site running a larger effective batch at the *same*
  learning rate is under-using that batch — the linear-scaling rule would want a larger LR
  — so its steps are well estimated but conservatively small. That is an optimisation
  inefficiency, not an instability, and the way to see it is **loss per token** rather than
  loss per round. Raising only that site's LR to compensate would make the two sites
  optimise differently, which is a larger departure than the batch asymmetry it fixes.

  **Start at 2, not 4.** Half the idle, ~1.5× throughput, and a clean comparison first.
  Two numbers to watch: `drift_ratio_max` should be **unchanged** (if it moves, the
  reasoning above is wrong), and loss per token should not get worse.

  **What this is not.** It fills the wait; it does not remove the barrier. Removing it
  means asynchronous DiLoCo — sites pushing deltas whenever they finish, merged
  continuously — which is a different algorithm, and this implementation's generation check
  deliberately *rejects* any delta not computed against the current round.

- **Per-site H (inner steps vary across clusters).** Algorithm 1 has every replica run
  the same H. When clusters have very different throughput — e.g. Snellius at 2.73
  steps/s vs LUMI at 0.58 steps/s — fixing H = 100 on both means the Flower round
  barrier forces the fast site to idle for ~137 s out of every 194 s round, reducing
  effective throughput to 101k tok/s out of hardware that can sustain 256k tok/s
  instantaneously. Setting a higher H on the fast site (e.g. H = 200 on Snellius,
  H = 100 on LUMI) lets it do more work per round and halves the idle time.

  This is sound because the merge is already token-weighted (`p_i = tokens_i / Σtokens`),
  which is exactly the correction for unequal work. The `darl.inner_steps` config
  key is already per-site TOML, so the change is purely a configuration choice.

  **LR schedule alignment.** With a single H the client could compute
  `global_step = merge_round × H` locally. With per-site H that formula breaks — each
  site would place itself at a different point on the schedule. The central server now
  tracks a **server-authoritative `global_step`**, broadcast as `pww_global_step` in
  every `configure_fit` config dict. After each merge it advances by the **largest**
  number of steps any participating cluster took:

  ```
  Δglobal_step = max_i( H_i )
  ```

  When all sites use the same H this reduces exactly to H, preserving backward
  compatibility.

  **This was originally the token-weighted average `Σᵢ pᵢ × Hᵢ`, and that was wrong.**
  The average is strictly less than the fast site's H, and alignment only moved a site
  forward — so the fast site outran the counter, its alignment became a permanent no-op,
  and the slow site alone was pulled to a value neither of them occupied. The schedules
  then diverged *monotonically*: at H = 200/100 they are 265 steps apart after six rounds
  and growing, which surfaced as `learning rate differs across clusters: 1.515e-04,
  3.000e-04`. A midpoint is a step where no replica is. `max()` is the only increment
  under which every site can sit at the same place, and it matches what the run actually
  advanced by, since the global model absorbed the fast site's work.

  Alignment is correspondingly **authoritative in both directions** now: `LambdaLR`
  recomputes from `last_epoch`, so the client places the schedule exactly rather than
  replaying steps forward. Forward-only alignment is only safe when there is nothing to
  disagree about.

  The trade-off to know: with `max()`, the slow site's LR advances faster than its own step
  count — 200 schedule steps for 100 optimiser steps at H = 200/100. That is the intended
  reading (the schedule tracks the *run*, not one replica), but it means a slow site sees a
  decayed LR sooner than it would training alone. The global step is persisted in `meta.json` (blob transport) and in
  the npz checkpoint meta array (inline transport), so it survives aggregator restarts.

  **Drift caveat.** Drift scales with H. At H = 100 from scratch, `drift_ratio` was
  already ~0.93 on Snellius. A very aggressive H (e.g. H = 473 to fill the full round)
  would push drift well past 1, where averaging destroys rather than combines progress.
  The recommended approach is to start modest (e.g. H = 200/100), observe
  `drift_ratio_max` after each increase, and grow the fast site's H only once drift has
  settled below ~0.1.

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
- **`drift`** is `||local - global|| / ||global||` per round, reported as mean **and
  max**. It is the number `H` should be tuned against: near zero means the inner loop
  is barely moving and the WAN round trip is not earning its keep; large means the
  replicas diverged far enough that averaging them destroys rather than combines their
  progress. Read the **max** — it is the worst replica that decides that, and two sites
  at 0.01 and 0.30 average to a reassuring 0.155. Averaged unweighted across clusters
  on purpose: drift is a property of a trajectory, not of a token count, so weighting it
  by tokens would say a fast site drifted more merely by doing more work.
- **`ppl`** is the perplexity of the training loss. Evaluation reports perplexity under
  its own name, not as `accuracy` — the old path reported a perplexity of 30 in a field
  labelled accuracy.

### What the held-out number does and does not measure

`run_train.sh` picks the validation data, in this order: a staged copy of **C4's real
validation split** at `$PWW_DATA_DIR/c4-validation` (used automatically via the offline
`c4_local` loader — stage it once per site with
`scripts/titan/stage_c4.sh --split validation --files 1 --out $PWW_DATA_DIR/c4-validation`),
falling back to torchtitan's bundled `c4_test` fixture with a warning. Three things worth
keeping straight:

- **The fixture is not held out, which is why the staged split exists.** An earlier
  version of this section called the fixture's disjointness from the training slice
  "plausible but unverified"; it has since been verified **false** — the fixture is the
  head of `en/c4-train.00000`, the first file `tokenize_c4.sh` consumes, so its windows
  are training windows. And the bias is not even uniform: the chance an eval window has
  been trained on scales with how much of the corpus a run consumes, so a fixture eval
  favours the DiLoCo arm (~70% of the corpus at 20k steps) over the step-matched baseline
  (~23%). C4's validation split is disjoint from train by construction. A run that fell
  back to the fixture says so in its log; do not read a central-vs-DiLoCo `eval/loss`
  comparison off such a run.
- **It is comparable across sites, and that is its main job.** Every cluster evaluates the
  *same* global weights, so a disagreement is a bug rather than variance. `run_train.sh`
  fixes the window *total* (`PWW_VAL_WINDOWS`, default 512) rather than `validation.steps`,
  because windows scored = `steps x local_batch_size x nproc` — a fixed step count makes an
  8-rank site score twice as many windows as a 4-rank one. The validation iterator is
  stateful, so successive evals walk forward through the split in lockstep at every site —
  same windows, same union, comparable number — **provided every site staged the same
  file(s)**; `sha256sum` them. One honest caveat: a site that restarts mid-run rewinds its
  own validation stream to the beginning while the others continue, so its next few evals
  score different windows than theirs. Different 512-window slices of C4 differ by a few
  hundredths of a nat, far under the 1-nat spread warning, so the check stays meaningful —
  but a small spread after a requeue is expected, not a bug.
- **It is still not a publishable C4 perplexity.** 512 windows is ~1.05M tokens per eval,
  so a few percent of round-to-round wobble is sampling noise (raise `PWW_VAL_WINDOWS` to
  2048 if the curve matters; ~1-2% training-time overhead at freq 100), and perplexity
  under a 131k tokenizer is not on the same scale as any published number under a
  different one. Judge progress from the training loss; use the held-out figure for the
  cross-site check and the central-vs-DiLoCo comparison.

A **training** perplexity is also reported now, next to the training loss. That one is exact
rather than approximate: `exp()` of a token-weighted mean loss is the perplexity of the union
of those tokens, the same identity the held-out figure uses.

### Why the metric arithmetic is not obvious

Three things here would produce plausible-looking numbers if they were wrong, so they
are worth knowing about and are each pinned by a test.

**Perplexity is pooled through the loss, never by averaging perplexities.**
`ppl = exp(mean NLL per token)`, so the perplexity of the union of the sites' validation
tokens is `exp()` of their token-weighted mean loss — exactly, since each site reports
its own mean NLL per token. Averaging per-site perplexities computes `mean(exp(L))`, and
`exp` is convex, so by Jensen that is always ≥ `exp(mean(L))`. Two sites at loss 2.0 and
4.0 report **31.0 against a true 20.1**, and the error tracks how unevenly the sites are
running rather than the model — the worst property a training metric can have.

**Accuracy is *not* treated the same way**, and that is not an inconsistency: accuracy is
a mean of per-sample 0/1 outcomes, so a sample-weighted mean of per-site accuracies *is*
the pooled accuracy. Linear, unlike perplexity.

**Tokens and loss are cluster-level, not rank-level.** torchtitan keeps
`ntokens_seen` and the training loss per rank and globalises them only inside its own
logging. Both leave the process here, and `num_examples` **is the FedMom merge weight** —
so a per-rank count would silently collapse token weighting back into uniform `1/k`
averaging whenever two sites share a per-rank geometry. LUMI's 8 GCDs and Snellius's 4
report the same rank-0 count at the same `local_batch_size` and `seq_len`, while LUMI
trained twice the tokens. `titan/wandb_metrics.py::cluster_sum` all-reduces them over the dp
mesh once per phase, and lives there rather than in the trainer because the WandB token axis
needs the same number — two copies would be two chances for the merge weight and the chart
to disagree. On the metrics path it is only a fallback: torchtitan's own `train_step` already
`dist_sum`s the token count into `extra_metrics["n_tokens_seen"]`, which is what the axis
reads.

DARL coverage:

```bash
curl -sS -H "X-DARL-Token: $(cat runs/darl/token)" \
  http://145.38.206.143:29510/status | jq .
```

`unassigned` + `leased` + `committed` + `quarantined` must equal the block count at
all times; the coordinator asserts this internally.

### WandB: what is logged, and which axis to read

Three processes log to one project and are meant to be read on one chart:
`central-<site>` (the single-node baseline), `diloco-<site>` (one participant), and
`central-aggregator` (the federated run as a whole). The conventions live in
`src/pww/wandb_utils.py` so all three cannot drift apart.

**Two axes, and the default is tokens.**

| axis | meaning |
|---|---|
| `train/cum_tokens` | cumulative tokens trained on. **The default x-axis.** Per *run*: a site's own tokens on `diloco-<site>`, the federation total on `central-aggregator` |
| `train/step` | global optimiser steps |

Tokens is the default because equal steps is not equal work: a two-site federation at 8 + 4
ranks trains three times the tokens per step a 4-rank baseline does, so a per-step chart
alone makes the federated run look better than it is. Read tokens for the honest comparison
and steps for the optimisation behaviour.

**There is no local-versus-global step ambiguity, by construction.** A site's `H` inner
steps are a *slice* of the global count, never a separate clock. Every site is realigned to
the server-authoritative `pww_global_step` before each phase
(`FederatedTrainer.align_to_global_step`), and that counter advances by `max(steps)` over the
contributing clusters — so step 1500 means 1500 optimiser steps in every one of the three
runs, with the DiLoCo one having crossed 15 outer rounds to get there. `darl.inner_steps` is
not an axis anywhere, and the aggregator now accumulates the same counter rather than
reconstructing it as `merge_round × H`.

That realignment is authoritative in *both* directions, which is where WandB needed a guard.
When a contribution is dropped — non-finite weights, a delta refused by the generation
check, a round with no quorum — the global step does not advance while the site's own counter
already did, so the next round pulls it back and it re-logs steps it has already logged.
WandB **silently discards** a log call whose step is below the current one, so a repeated
round used to appear as a hole in every chart for that site.
`wandb_utils.MonotonicStep` advances the index artificially instead, and the true position
travels as `train/step`: nothing is dropped, and both axes still plot correctly. A
`wandb step went from N to M` warning in a site log is that guard firing, and it means a
round was not merged — look for the reason, not for the chart.

**Keys.** The `train/*` and `eval/*` sets are identical between a baseline and a DiLoCo run,
because one installer produces both (`pww/titan/wandb_metrics.py::install_metric_hooks`):

| key | on |
|---|---|
| `train/loss`, `train/perplexity` | all three. On the aggregator, token-weighted across participants |
| `train/cum_tokens`, `train/step`, `train/lr`, `train/grad_norm` | all three |
| `eval/loss`, `eval/perplexity` | all three. Pooled through the loss on the aggregator — see above |
| `train/drift_ratio_avg`, `train/drift_ratio_max` | aggregator only. Read the **max** |
| `train/global_batch_size` (sequences), `train/global_batch_tokens` | aggregator. Per *step*, from the `seq_len` the clusters report |
| `train/round_seconds`, `throughput/tokens_per_s_combined` | aggregator |
| `cluster/<id>/{tokens_per_s,tps_per_rank,dp_degree,mfu_pct,tflops_per_rank,peak_memory_gib,power_watts,grad_norm,drift_ratio}` | aggregator, per participant. Never averaged across sites — an MFU against an MI250X GCD and against an H100 are ratios to different peak FLOPs |
| `loss_metrics/*`, `throughput(tps)`, `mfu(%)`, `memory/*` | torchtitan's own, on the site runs only |

**Compare a baseline against `central-aggregator`, not against a `diloco-<site>` run.** A
site reports only its own share of the tokens, so overlaying `diloco-snellius` on
`central-snellius` by tokens makes the federated run look like it reached a loss on a third
of the data it actually used.

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

### The non-persistent buffer trap

Worth its own section because it produced two completely different symptoms from one
cause, and one of them was silent.

Applying incoming weights means writing a full state dict into an FSDP2-sharded model.
`params.scatter_full_state` does that by broadcasting each tensor from rank 0 and letting
every rank shard its own slice. The question it has to answer first is *which keys to
iterate*, and both obvious answers are wrong:

- **`state`, the incoming dict, cannot decide it.** Worker ranks are called with an empty
  dict and still have to reach every `broadcast` in the same order as rank 0.
- **`named_parameters() | named_buffers()` cannot decide it either**, and this is what the
  code used. `named_buffers()` reports **non-persistent** buffers. `state_dict()` — and
  therefore `get_model_state_dict`, the codec, and the wire — deliberately omits them.

On the Qwen3 0.6B flavor the difference is exactly one tensor: `rope_cache`, registered
`persistent=False`, holding the precomputed RoPE cos/sin table. **311** tensors cross the
wire; `rope_cache` is not among them. So the loop reached a key with nothing to load and
fell through to the worker-rank branch:

```python
value = torch.empty(full_shape, dtype=param.dtype, device="cuda")
dist.broadcast(value, src=0)      # a no-op at world_size 1
param.detach().copy_(value)       # uninitialised memory -> rope_cache
```

Every application of incoming weights destroyed the RoPE table, on every rank including
rank 0. What happened next depended on what the caching allocator handed back:

| the block was | RoPE becomes | symptom |
|---|---|---|
| freshly mapped (zeros) | `cos = sin = 0`, so RoPE annihilates q and k | **silent.** Attention goes uniform, the model keeps a finite loss and keeps descending — toward the unigram entropy rather than a language model. Nothing in any log says so. |
| reused and dirty | arbitrary float32 bit patterns | `step 1: loss became NON-FINITE (nan)` on the first microbatch, the site's weights rejected, its blocks released. |

Which one a site got was a matter of allocator history, not of hardware. The site that hit
the nan had received an `evaluate` message before its first `train` message — a full
validation pass, which allocates and frees, leaving dirty blocks to reuse.

Two things generalise from it:

- **A nan on the very first microbatch after a weight application is the weights.** Not the
  data, not the learning rate, not the accelerator. That is precisely why `_log_step` logs
  the first non-finite step and how many finite steps preceded it: `0 step(s) before it
  were finite` pointed straight here, and gradual divergence would have pointed elsewhere.
- **Prefer a rank-independent key list derived from the module.** `params.keys_to_load`
  now returns `sorted(part.state_dict())` filtered to keys the module owns distinctly —
  which excludes non-persistent buffers *and* the tied `output.weight` alias — and rank 0
  raises rather than substituting a buffer if a key it needs is genuinely missing.
  Defaulting was the real fault; the wrong key set was only how it got reached.

`rope_cache` is a pure function of `(head_dim, max_seq_len, rope_theta)`, built in the
constructor and rebuilt by `init_weights`. It must never be sent and never be written here.

### Failures

| symptom | cause | what to do |
|---|---|---|
| server waits, no rounds start | `min-clients: 2` in the aggregator config | set it to 1. Only require 2 if you specifically want to force both sites. |
| client refuses the round citing transport | `flower.transport` in the TOML disagrees with the server's `--transport` | make them match; restart whichever is wrong. |
| `every delta for round N was stale` | all participating sites were requeued and computed against an older global model | expected after a walltime kill. The next round is current. |
| a site hangs at the end of an epoch | was: a prefetched DARL lease nobody consumed, so the pool looked drained while the session held the missing blocks | fixed — `LeaseSession.acquire` now consumes a pending prefetch instead of waiting on it. Covered by `tests/test_darl.py`. |
| `dropping cluster X -- tensor N contains nan/inf` | that site's weights are non-finite, usually local divergence or float16 overflow on the wire | the round proceeds without it and the global model is untouched. Read that site's own training loss: this is a symptom, not the cause. |
| `step 1: loss became NON-FINITE`, `0 step(s) before it were finite`, on a site's **first** round | was: `scatter_full_state` overwrote the model's non-persistent buffers with uninitialised memory every time it applied incoming weights | fixed — see [the non-persistent buffer trap](#the-non-persistent-buffer-trap). A nan on the *first* microbatch after a weight application is always the weights, never the data or the learning rate. |
| training loss far *below* the held-out loss (e.g. 1.97 against 8.35) | was: the per-round loss summed torchtitan's already-rescaled microbatch losses, so it read low by exactly `PWW_GRAD_ACCUM` | fixed — see [§3, gradient accumulation](#3-the-outer-step). A training loss several nats under the held-out loss is a unit error, not a generalisation gap. |
| `Training loss nan` with `merge complete` | was: one non-finite contribution poisoned the global model, and every later round was arithmetic on nan | fixed — non-finite contributions are dropped before the merge. If it recurs, one site is diverging and the log now names it. |
| `Round N failure: ...` | a client raised, or the transport rejected its reply | the reason is now logged. Previously only the count was, which is why an 18-round failure loop had no diagnosable cause. |
| `held-out loss spread N nats across clusters` | either a site is not applying the weights it was sent, or the sites are not scoring the same data | check `validation.steps`/`PWW_VAL_WINDOWS` and the eval token counts first; equal counts mean the model is the suspect. |
| `outgoing parameters are N GiB ... gRPC's 2 GiB cap` | the wire dtype makes the message too large | lower `flower.wire_dtype`, or move to `transport=blob`. Previously this failed the send with no stated cause and wedged the run. |
| `blob store: 507` | the volume behind `--blob-root` is below its reserve | point `--state-dir`/`BLOB_ROOT` at a larger volume. `GlobalState.log_disk_budget` prints the requirement at startup. |
| lease expiry after a walltime kill | Slurm killed a site mid-epoch | self-healing: uncommitted blocks return to the pool on TTL expiry, and the surviving site finishes the epoch. Releasing on SIGTERM makes it milliseconds instead of a full TTL. |
| `Watchdog caught collective operation timeout: ... BROADCAST, NumelIn=1 ... 300000ms`, stack through `run_worker_loop`, whole site dies by SIGABRT mid-run | was: the worker ranks waited out the round barrier inside a NCCL broadcast on the **default** process group, whose CUDA watchdog runs at `comm.init_timeout_seconds` (300s). A fast site legitimately idles at the barrier for the slow site's remaining phase plus the merge — Snellius's ~4.5–5 min wait cleared 300s on round 26 of the first 20k run and its own watchdog killed it. An earlier fix raised the timeout for exactly this symptom, but `set_pg_timeouts` touches the mesh groups and the command broadcast rode the default group | fixed — control-plane broadcasts now ride a dedicated **gloo** group (no CUDA watchdog, no GPU involvement, own timeout at 4× the round timeout). Mitigation on a build without the fix: `-- --comm.init_timeout_seconds 1800` at submit time, which sets the default group's timeout directly. |
| `Connection refused` on 29511 | daemons not running | `./scripts/central_node/start_central_services.sh` |
| `503 cluster_busy` at startup | another live process holds this cluster id — a second concurrent job at the same site, or a requeue whose predecessor died hard and whose leases have not expired yet | pass `--replica a` / `--replica b` to give each job its own id. If it is a requeue, it clears itself within one TTL. See above. |
| `N clients reported as cluster X` in the aggregator log | two Flower clients using one cluster id, on a run with no DARL coordinator to refuse it | same fix. That round's contributions under that id were dropped rather than merged wrongly. |

---

## 7. Deploying, and adding a site

### Updating an existing site

```bash
./scripts/deploy.sh              # pull, submodules, dirs, symlinks, verify
./scripts/deploy.sh --check      # report only, change nothing
```

One script for every machine, including the central VM. It is site-agnostic by
construction: everything machine-specific comes from `sites/<site>.sh`, which
`env.sh` selects by detection. Re-running it is the normal way to pick up a
commit.

It deliberately does **not** build the heavy environment. A LUMI container build
is a 30–60 minute batch job, so `deploy.sh` reports what is missing and the one
command that creates it, rather than starting it behind your back.

It also refuses to turn `runs/` or `data/` into a symlink when one already exists
as a real directory. `ln -sfn` would put the link *inside* it — producing
`runs/runs` — and every path in the runbook would then quietly point at nothing.

### Adding a third site

There is no per-site deployment script to write, and no code to change. A site is
one file, `sites/<name>.sh`, which must define:

| | |
|---|---|
| `PWW_ACCOUNT` | accounting project for `sbatch` |
| `PWW_SCRATCH` | writable, large, visible from compute nodes |
| `PWW_GPUS_PER_NODE` | ranks per full node — **GCDs, not cards**, on AMD |
| `PWW_CPUS_PER_TASK` | cores per rank |
| `PWW_ACCELERATOR` | `rocm` or `cuda` |
| `PWW_GPU_VISIBLE_VAR` | `ROCR_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` |
| `PWW_LAUNCH` | bash array: command prefix that enters the environment (container `exec`, or empty for modules) |
| `pww_cpu_bind()` | echoes the `--cpu-bind` value for an allocation |
| `pww_titan_env()` | echoes `kind<TAB>path<TAB>build command` for the torch ≥ 2.9 environment, so `deploy.sh` can check it without knowing the machine |

Then add the detection branch in `env.sh`'s `pww_detect_site`, and copy a job
script from `scripts/lumi/` or `scripts/snellius/` — whichever resembles the new
machine — adjusting only the `#SBATCH` header and the environment quirks.

Three things that are **not** per-site, and where the time actually goes:

- **The torch ≥ 2.9 environment.** This is the whole of the work. Snellius has a
  usable module tree, so it is a second venv; LUMI's own image is on 2.7.1 with no
  2.9 module, so it is a container. A new machine is one or the other, and
  `scripts/titan/README.md` has the reasoning for both.
- **The corpus.** Compute nodes have no internet, so the tokenised shards and the
  exact `tokenizer.json` must be on that site's scratch before the first job.
  Copy them rather than regenerating: `tokenizer.sha256` feeds the manifest
  digest, so a file regenerated under a different `transformers` version is
  refused at registration. See [RUNBOOK.md](RUNBOOK.md) Part 1.
- **Reachability.** The compute nodes must reach the central VM on 29510 and
  29511. Test it from a login node before spending a queue slot.

Nothing on the central node changes. `min-clients: 1` means a third site is
sampled as soon as it connects, `configure_fit` hands it the current global model
before it trains, and DARL partitions the index space over however many clusters
register. The one hard requirement is that it computes the **same block-space
digest** — same window count, `block_size` and `space_seed` — or registration
refuses it, which is the guard working.

## 8. What is verified, and on what

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
| `test_titan.py` | 21 | the token shard format, the DARL dataloader's exactly-once coverage across ranks, the inline wire codec including the bfloat16 bit-pattern hop, the scatter key set that must not include non-persistent buffers, config feasibility |
| `test_federation.py` | 26 | blob store over real HTTP; 0/1/N live replicas; restart durability; stale-delta rejection; mismatched-model refusal; duplicated cluster ids; metric pooling; the outer step against `SGD(nesterov=True)` |
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
