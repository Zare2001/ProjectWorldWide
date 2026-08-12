# TODO: federated training across LUMI and Snellius

## Where this stands

Both sites run the same code independently, verified end to end:

| | LUMI (8x MI250X GCD) | Snellius (4x H100) |
|---|---|---|
| CIFAR-10, 30 ep, global batch 1024 | 93.35% @ 38,400 img/s | 93.55% @ 56,184 img/s |
| all-reduce, 1 node / 2 nodes | 123 / 88 GB/s | 300.8 / 133.1 GB/s |
| torch | 2.7.1+rocm6.2.4 | 2.7.1+cu126 |

The cross-site mechanism is built, CPU-verified, and **has now run on GPUs at both
sites simultaneously**: **DARL** partitions the corpus by lease so the sites cover it
exactly once, and **Flower + `PWWFedMom`** does the DiLoCo outer step on the central VM.

Where it actually stands, as of the last run:

- **Real C4, 5.65B tokens** (2,756,597 windows), digest-matched at both sites.
- **Rounds merge with both sites contributing.** A two-cluster round reported
  19,660,800 tokens and completed the FedMom merge.
- **One site's contribution came back non-finite**, and that is the live blocker. It is
  dropped rather than averaged now, so the run survives it — but that site contributes
  nothing until it is fixed. See § 4.

So the open work is no longer "make it run". It is: find out why one site diverges, and
close the verification gaps that a run which merges does not by itself close.

---

## Done, and what it replaced

The original plan here was checkpoint averaging driven over SSH from Snellius,
with an `src/pww/federated.py` merging consolidated checkpoints. That design is
**superseded**; recording the substitution because the shape of the code no longer
matches the plan it came from.

| original plan | what happened instead |
|---|---|
| `--site-index i` / `--site-count N` to pre-partition CIFAR-10 | DARL. The partition is decided at run time by a coordinator that knows who is alive, so a site that joins late or dies at walltime still yields exactly-once coverage. A static split cannot do that. `train_flower.py` leases like the LLM path. |
| `--round-epochs K` to bound a round without wrecking the LR schedule | a Flower round *is* the bound: `fit` runs H inner steps and returns. torchtitan's LR schedule is global and continues across rounds. |
| `average_checkpoints(paths, weights)` in a new `federated.py` | `central/strategy.py` (the round protocol) and `central/globalstate.py` (the durable global model and the streaming per-tensor merge). Averaging whole checkpoints was never going to reach 7B: three dense copies is 84 GB at 7B in float32 on a VM with tens of GB of RAM. |
| average BatchNorm buffers, handle `num_batches_tracked` | only `named_parameters()` is exchanged. Correct for Qwen3, which is RMSNorm with learned weights and no accumulated statistics; its buffers are RoPE tables recomputed at init. This is also exactly why the ResNet path pins outer momentum to 0.0 — a BatchNorm model *would* need its buffers, and momentum can push averaged weights outside the convex hull of the client weights. A model with learned buffer state needs this revisited. |
| SSH push/pull between sites, Snellius orchestrating | Flower gRPC on 29511, plus an HTTP blob store on 29512 for models above ~1B. Neither site talks to the other; both talk to the central VM. |
| "if one site's job dies, does the round stall or proceed?" | proceeds. `min-clients: 1`, and 0/1/N live replicas are all normal states. A requeued site's first stale delta is rejected by `base_round` rather than averaged in. |
| — (not foreseen) | **two jobs at one site.** Partial-node allocations make co-location routine, and the cluster id was the site name alone. Resolved with a per-session incarnation token judged by liveness — a stale incumbent means a requeue and is taken over, a live one means a second concurrent job and is refused. Plus `--replica` to avoid the refusal, release scoped to the incarnation, and a duplicated-id guard in `aggregate_fit` for runs that never register with DARL. |

Verified by measurement, CPU-only (`tests/test_darl.py`, `test_titan.py`,
`test_federation.py`):

- [x] consolidated and sharded checkpoints round-trip across differing world sizes
      (4 ranks -> 1 and 4 -> 2, on Snellius alone; loss continued at 1.31/1.45 from a
      cold start of ~2.5, so the weights really loaded)
- [x] exactly-once corpus coverage under concurrent clusters, expiry, stealing and
      quarantine
- [x] the streaming FedMom merge equals a dense reference to 4.8e-07 over 3 rounds
- [x] the outer step equals `torch.optim.SGD(momentum=0.9, nesterov=True)` on
      DiLoCo's outer gradient to 1.9e-06, and is distinguishable from heavy ball
- [x] 0 / 1 / N live replicas, restart durability, stale-delta rejection,
      mismatched-model refusal
- [x] two concurrent jobs at one site: refused at registration, requeue after a clean
      exit and after a hard crash, release scoped to the incarnation, and a duplicated
      cluster id in one aggregation round
- [x] per-tensor DTensor gather/scatter on 2 gloo ranks with genuinely sharded
      parameters

---

## Open

### 1. The torch 2.9 environments — done, except parity

torchtitan at the pinned commit needs torch >= 2.9; this repo pins 2.7.1 on both
sites for numerical comparability.

- [x] Snellius: venv built and in use — job logs show
      `activated torchtitan venv: ~/venvs/pww-titan-snellius`
- [x] LUMI: container built and in use — job logs show
      `singularity exec $PWW_SCRATCH/containers/pww-titan.sif`
- [ ] **Re-establish parity at the new version.** Both sites on the same torch
      *minor* version, and the CIFAR-10 comparison re-run there. The 0.2-point
      agreement between sites is the evidence that the port is correct, and it does
      not carry over from 2.7.1 by assumption.

### 2. GPU verification of what CPU tests cannot reach

All of this has now **executed** on hardware at both sites. Executing is not the same
as being correct, and the distinction is doing real work here: one site trains, merges,
and returns non-finite weights, so something in this list runs without being right.

- [x] FSDP2 wrapping of Qwen3 through torchtitan's `TrainSpec` — both sites train
- [x] a real Qwen3 forward/backward against the 128k-vocab embedding — losses of 10.9
      falling to 7.3 over 8 rounds on one site, so the model learns
- [ ] **the DTensor gather/scatter in `titan/params.py` and `delta.py` is the prime
      suspect for the non-finite contributions.** It has run on 8 GCDs and 4 H100s, and
      the evidence that it may be wrong at 8 ranks is that the site evaluating the *same*
      global weights scored 0.93 nats off an untrained model (`ln 131,328 = 11.79`), which
      means it was not running the weights it was sent. Verified on 2 gloo ranks only.
- [ ] **`bf16` numerics on MI250X vs H100** — now a live question rather than a
      precaution. Note the asymmetry that makes it sharp: local compute is bfloat16
      (range ~3.4e38) and the wire is float16 (max 65,504), so a weight that is ordinary
      locally becomes `inf` on the cast. Whether the non-finite values are produced by
      training or by that cast is unresolved and is one finiteness check away.
- [ ] DCP checkpoint save/load across the two sites' differing rank counts

### 3. Corpus staging at scale

- [x] **done.** Real C4, `--max-files 32`: **2,756,597 windows = 5.65B training tokens**
      at `seq_len 2048`, vocab 131,328. That is `training.dataset = "pww_tokens"` over the
      shard directory, 5,077x the `c4_test` fixture — the fixture is only the *validation*
      set, see below.
- [x] **done.** Manifest digest `f1bfffe94c71a9ab…` confirmed identical at both sites, and
      cross-checked three ways before either job was queued: the tokenizer.json sha256
      against the manifest's own field, the shard byte total
      (`2,756,597 x 2049 x 4 = 22,593,069,012`, exact), and all 22 shard boundaries walked
      through `ShardedTokenCorpus`. The coordinator's block-space digest
      (`20f69387b3d4ba09`, 2,692 blocks) was derived independently on both sides and
      matched — which is the check that actually gates registration.
- [x] **validation on a real held-out split.** The fixture's disjointness, previously
      "plausible rather than verified", was verified **false**: `c4_test` is the head of
      `en/c4-train.00000`, inside any `--max-files` training slice — and the bias is
      non-uniform, favouring the arm that consumes more of the corpus (DiLoCo ~70% vs the
      step-matched baseline ~23% at 20k steps). Resolved without touching the block space:
      `run_train.sh` automatically evaluates on a staged copy of C4's *validation* split
      (`stage_c4.sh --split validation --files 1 --out $PWW_DATA_DIR/c4-validation`,
      offline `c4_local` loader), falling back to the fixture with a warning. The
      per-site staging is the remaining manual step; the sha256 of the staged file must
      match across sites (RUNBOOK Part 1 step 3). The alternative — a held-out tail carved
      from the run's own shards, excluded from the DARL space — remains open only if the
      validation split's distribution shift from the train slice ever matters; it changes
      `num_samples` and the digest and so is a `PWW_FRESH_RUN=1` change.
- [ ] measure the tokenisation pass: it is offline and one-off, but it is also the
      only step whose cost scales with the corpus rather than the run

### 4. The first real cross-site run

- [x] **0.6B, inline transport, both sites, merging.** A two-cluster round reported
      19,660,800 tokens (13,107,200 from the 8-GCD site, 6,553,600 from the 4-GPU one,
      i.e. token weighting behaving as intended) and completed the merge. Throughput
      263,721 tok/s combined.
- [ ] **Find why one site returns non-finite weights. This is the blocker.** What is
      known: it starts clean (`PWW_FRESH_RUN=1` clears its checkpoint), its learning rate
      is *lower* than the other site's, its first held-out eval on the shared global model
      read 0.93 nats off an untrained model, and its first training contribution is
      non-finite in tensor 0 of 311 (the embedding). Two candidates, needing opposite
      fixes: the parameter scatter at 8 ranks (§ 2), or the bfloat16-local /
      float16-wire cast overflowing. The per-step loss trace added for this will say which
      — a first non-finite step of 1 points at the weights it was handed, step 40-odd
      points at optimisation.
- [ ] deliberately kill one site at walltime mid-round and confirm the survivor
      continues and the requeued site's stale delta is rejected once, then rejoins
- [ ] 8B, blob transport, and measure the actual WAN transfer time per round
      against the H that was chosen

### 5. Hyperparameters that are guesses until measured

- [ ] **H.** `inner_steps: 100`, against the paper's 500. Now partly measured rather
      than a guess: from a random initialisation, drift was **1.69** on the opening round
      and **0.93** on the next — i.e. the local update was larger than the norm of the
      weights themselves. That is expected while the weights are near initialisation and
      it is falling fast, so it may need nothing; the open question is whether it keeps
      falling. If it does not, the levers are a smaller H for the opening rounds or
      `server-momentum: 0.0` until it settles. Note this is a consequence of training from
      scratch, which the paper's Algorithm 1 does not assume — see FEDERATION_GUIDE § 3.
- [ ] **Outer LR.** Now 0.7 with momentum 0.9, the paper's values, and consistent
      with `src/pww/diloco.py`. Worth an A/B against `1.0 / 0.0` (exact FedAvg),
      which is the control arm.
- [ ] **Whether the federated batch should change the inner LR.** The effective
      batch across two sites is roughly twice one site's. Whether to scale for that
      is a research question, not an infrastructure one.

### 6. Costs we accepted, and one invariant that must not be tidied away

Not open work — decisions already made, recorded because the reasoning is not visible
from the code and someone will otherwise reverse them by accident.

#### A hard crash now costs a bounded requeue delay

Refusing a second live process on a cluster id (see the superseded-plans table) uses
liveness as its discriminator, and liveness is only knowable up to a TTL. A job that
exits cleanly releases on SIGTERM and its successor takes the id back immediately. A job
that dies **hard** — OOM kill, node failure, a hung rank — leaves leases on the books,
and the coordinator cannot tell that from "slow" until they expire. So the requeued job
is refused for up to one TTL, then takes over.

That is a real regression in requeue latency, taken deliberately in exchange for
eliminating silent duplicate training. It is bounded, it is logged with the reason, and
`--replica` avoids it entirely. Both paths are tested (`tests/test_darl.py`: requeue
after a clean exit, requeue after a hard crash).

- [ ] if the delay ever actually hurts, the lever is a shorter `min_ttl` for the
      *cluster* liveness window than for lease expiry — they are the same number today
      only because one predicate served both purposes
- [ ] a `--force-takeover` escape hatch is the other option, but it hands the operator a
      way to cause exactly the corruption this prevents, so it should not be added
      speculatively

#### Journal replay must record the incarnation without checking it

`Coordinator._replay` calls `register(..., check_conflict=False)`. Both halves of that
are load-bearing and they pull in opposite directions:

- **skip the check** — a replayed register was already authorised when it was served,
  and re-testing liveness against a historic timestamp is meaningless. Worse, it raises
  `ClusterBusy`, which is not in the replay handler's `except` clause, so recovery would
  crash.
- **still record the value** — a live client registers exactly once, at session start,
  and never again. Drop the incarnation on replay and a coordinator restart leaves the
  incumbent's field blank, which disables the concurrent-job guard for the remainder of
  the run. The failure it prevents is silent, so nothing would reveal that it had
  stopped working.

Pinned by "a coordinator restart keeps the incumbent's incarnation, from snapshot or
journal", which covers both recovery paths — the value can come from the snapshot or the
write-ahead journal depending on which side of the last snapshot the register fell — and
was verified to fail when the incarnation is dropped from the replay call.

### 7. Two sites can train on different corpora and nothing notices

The coordinator compares `BlockSpace.digest` at registration, which covers
`num_samples : block_size : seed : shuffle : epoch` plus the permutation. That proves the
clusters agree about what position *p* means **positionally**. It says nothing about what
the bytes at *p* are.

`Manifest.digest` is the one that covers corpus identity — `format : seq_len : window :
dtype : vocab_size : tokenizer_sha256 : num_windows` — and the coordinator never sees it.
The two digests overlap in exactly one field: `num_windows`, which arrives as
`num_samples`, stripped of everything that gives it meaning.

So this passes every check in the system:

> LUMI and Snellius have the same window count and **different tokenizers** — one
> regenerated under a different `transformers` version, so `tokenizer.json` hashes
> differently. Block-space digests match, so registration succeeds. Each site also passes
> its own `shards.verify_compatible`, because that compares a manifest against the files
> sitting next to it, and locally both are self-consistent. The sites lease disjoint
> positions over *different corpora*, train on different token ids, and the outer step
> averages the results.

It does not crash, and no log line is wrong. It shows up, if at all, as a loss curve that
is merely worse than it should be.

**The only defence today is manual**: comparing manifest digests across sites before
submitting (RUNBOOK.md Part 1) and `sha256sum tokenizer.json` against the manifest's
`tokenizer.sha256` after any transfer. Both are in the docs, neither is enforced, and the
window in which it matters is exactly when someone is moving 22 GiB between facilities
under time pressure.

The fix is to send the corpus digest at registration and have the coordinator pin it.
Prototyped and then reverted deliberately, because it touches the registration path of a
run that is about to start; the notes below are what the implementation has to get right.

- [ ] `LeaseTable.register(..., corpus_digest="")`: **first writer wins.** The coordinator
      cannot be given an expected value — it holds no shards, no tokenizer and no manifest
      by design, and [that is a property worth keeping](FEDERATION_GUIDE.md), so the first
      cluster to report one defines the run and later clusters are checked against it
- [ ] an empty digest must stay legal and unchecked. The legacy HuggingFace path
      (`train_llm_flower.py`) and `darl/simulate.py` have no manifest at all, and a
      mixed-version deployment must not break mid-run. Those clusters are simply not
      covered, and the status output should make that visible rather than implying they are
- [ ] persist it in `snapshot()`/`restore()`. This is the same trap as the incarnation in
      § 6: a live client registers **once**, at session start, so a coordinator restart that
      forgets the pinned value disables the guard for the remainder of the run — and the
      failure it prevents is silent, so nothing would reveal that it had stopped working
- [ ] thread it through `LeaseClient.register` → `LeaseSession` → `darl_dataloader`, which
      is the only caller that has a `Manifest` in scope
- [ ] `ValueError`, so it surfaces as `400 bad_request` exactly like the block-space
      mismatch, and the message must name both clusters and say to compare manifests and
      rsync the tokenizer rather than regenerate it
- [ ] tests: a second cluster with a differing corpus digest is refused; the same digest is
      accepted; an empty digest is accepted and does not pin; the pin survives a snapshot
      round-trip **and** journal replay. Verify each by reverting the guard and watching the
      test fail, not merely by watching it pass

Worth stating plainly, since it argues for doing this rather than relying on the runbook:
every other cross-site disagreement in this system is caught by a machine at startup. This
one is caught by a human remembering to run two commands.

### 8. Smaller things

- [ ] `--keep-rounds` prunes old blobs; confirm the pruning actually keeps disk
      bounded over a 200-round run rather than only in the test
- [x] **done.** `release_all()` is wired to SIGTERM on the leader rank
      (`darl_dataloader._release_on_sigterm`), so uncommitted spans go back to the pool in
      milliseconds instead of after a full lease TTL. Slurm sends SIGTERM before the
      walltime SIGKILL and a long DiLoCo run ends at walltime *normally*, so this is the
      common path.

      It was not merely missing before: both job scripts printed "SIGTERM -- releasing DARL
      leases" and then forwarded the signal, while no handler existed anywhere in
      `src/pww/titan/` — Python's default SIGTERM terminates without unwinding, so
      `LeaseSession.close()` never ran. The log asserted a release that never happened, and
      the shell messages now describe only what the shell does.

      Two limits, stated in the code: a signal arriving while the main thread is inside a
      C-level RCCL/NCCL collective is not handled until that collective returns, so a kill
      landing mid-all-reduce still falls back to TTL expiry; and the previous handler is
      chained rather than replaced, so termination is never swallowed into a hang.

- [ ] a third site is already supported with no special handling (see below), but it
      has never been tried

---

## Notes on the machines

- **Snellius multi-node must take whole nodes.** A multi-node GPU job asking for
  fewer than 4 GPUs/node is rejected; `--exclusive` bills for them anyway. So each
  federated site on Snellius is a whole node minimum.
- **Snellius single-node partial allocations start instantly** (1 GPU + 16 cores).
  `scripts/snellius/job_cifar_debug.sh` is the fast iteration loop.
- **`/scratch-shared` is purged at ~14 days.** Checkpoints that matter belong in
  `/projects`. This includes the central node's `--state-dir`: it holds the *only*
  copy of the global model.
- **Snellius priority is ~98% fairshare** with a 1-day decay half-life. A long
  federated sweep steadily lowers your own queue priority.
- **LUMI's first epoch is ~25x slower than steady state** because MIOpen autotunes
  for unseen shapes. Do not read anything into round 1 throughput.

---

## A third site joining mid-run

This needed a protocol when the transport was checkpoint averaging. It does not
now — the elastic membership path handles it, and there is no separate code for it.

A site joining at round *R* must start from the current global model
θ_global^(R), and it does, unavoidably: `configure_fit` sends the current global
weights to every participant before it trains, so a client cannot contribute a delta
derived from θ₀ or from a stale checkpoint even if it tried. That is what makes the
join safe rather than corrupting:

- **Exact alignment.** At inner step 0 of round *R* the new site holds identical
  weights to every other participant, so its delta θ_new^(H) − θ_global^(R) is a
  valid trajectory in the current loss basin.
- **Disjoint tokens.** It registers with DARL under its own cluster id and receives
  blocks nobody has trained on. It cannot re-process another site's tokens.
- **Proportional weight.** The merge weights each delta by tokens contributed, so a
  site that is initially slow contributes proportionally less rather than skewing
  the average.
- **A stale delta cannot slip in.** `base_round` is checked, and a mismatch is
  rejected with the reason logged, not averaged.

Checklist for adding one:

- [ ] match `model.flavor`, `training.seq_len`, the tokenizer, and
      `titan.pad_vocab_to_multiple_of` — mismatches are refused at the merge, by
      key and by shape
- [ ] match `darl.space_seed` and the corpus window count, or registration is
      refused by digest
- [ ] give it a **stable** `darl.cluster_id` across requeues: the coordinator sizes
      grants from a cluster's measured throughput, and a fresh id on every
      resubmission throws that history away
- [ ] set `flower.transport` to whatever the server is running
