# TODO: federated training across LUMI and Snellius

## Where this starts from

Both sites run the same code independently, verified end to end:

| | LUMI (8x MI250X GCD) | Snellius (4x H100) |
|---|---|---|
| CIFAR-10, 30 ep, global batch 1024 | 93.35% @ 38,400 img/s | 93.55% @ 56,184 img/s |
| all-reduce, 1 node / 2 nodes | 123 / 88 GB/s | 300.8 / 133.1 GB/s |
| torch | 2.7.1+rocm6.2.4 | 2.7.1+cu126 |

**Nothing below requires changing how either site runs.** This is additive.

The goal is federated: each site trains locally, sites exchange weights every K
epochs, checkpoints are the transport. No cross-site collectives.

## Three facts that make this tractable

1. **Checkpoints are portable between the sites.** Both run torch 2.7.1 (that
   pin was deliberate), and a consolidated checkpoint is plain CPU tensors.
2. **Consolidated checkpoints are world-size independent.** LUMI's 8 ranks and
   Snellius's 4 exchange freely; `set_model_state_dict` reshards on load.
3. **They are small.** ResNet-18 consolidated = **85 MiB** (model + SGD
   momentum). Trivial per round over a WAN. For a 7B LLM this becomes 28 GB
   (56 GB with optimizer state), which is what should drive the choice of K.

## Phase 1 -- trainer changes

Three concrete gaps. All testable on CPU, no allocation needed.

### 1.1 Sites train on identical data (`src/pww/data/cifar.py`)

`build_cifar10_loaders` hands every site the full CIFAR-10 and shards it with
`DistributedSampler` across *that site's* ranks only. Run it on both machines
and you get two redundant runs, not federated training.

- [ ] Add `--site-index i` / `--site-count N`, partition the dataset before the
      sampler sees it (`torch.utils.data.Subset` over a deterministic split).
- [ ] Keep the split reproducible from the seed alone, so both sites derive the
      same partition without communicating.
- [ ] Offset `set_seed` by site index too (`src/pww/config.py:74`), or the two
      sites apply identical augmentation to their shards.

### 1.2 No way to run K epochs without wrecking the LR schedule

The loop is `range(start_epoch, args.epochs)` (`train_cifar.py:223`), and
`args.epochs` *also* sets the cosine horizon (`build_scheduler`,
`train_cifar.py:65`). Capping a round by lowering `--epochs` makes the LR decay
to ~0 at every round boundary.

- [ ] Add `--round-epochs K`: train K epochs from `start_epoch`, then exit.
- [ ] Leave `--epochs` as the schedule horizon, untouched.

Already works in our favour: resume reads `epoch` from checkpoint meta and
fast-forwards the scheduler (`train_cifar.py:213-218`), so once this split
exists, a merged checkpoint continues the *global* schedule for free.

### 1.3 Verify cross-world-size resume actually round-trips -- DONE

- [x] Test consolidated save/load across differing world sizes.
- [x] Do it on Snellius alone first (4 ranks -> 1 rank via the debug job).

Both formats round-trip onto a different world size, tested on Snellius alone:

| written by | reloaded on | format | result |
|---|---|---|---|
| 4 ranks | 1 rank | consolidated | resumed at epoch 4, loss continued at 1.31 |
| 4 ranks | 2 ranks | sharded | resumed at epoch 4, loss continued at 1.45 |

The loss values are the check that matters: a cold ResNet-18 starts near 2.5, so
picking up at 1.3-1.4 means the weights really loaded rather than the run
silently restarting. `set_model_state_dict` reshards on load exactly as fact 2
above claims.

Still untested: the *cross-site* direction, LUMI's 8 ranks -> Snellius's 4. That
needs a checkpoint physically moved between the sites, so it belongs to Phase 4
rather than here.

## Phase 2 -- `src/pww/federated.py`

Nothing merges checkpoints today; `checkpoint.py` only saves and loads.

- [ ] `average_checkpoints(paths, weights) -> merged` weighted by samples seen
      per site (not uniform -- sites may run different numbers of steps).
- [ ] Average model weights **and BatchNorm buffers**. `get_model_state_dict`
      returns buffers, so `running_mean`/`running_var` are included; skipping
      them is a common and silent FedAvg bug.
- [ ] Handle `num_batches_tracked` (integer counter -- sum or take max, do not
      average into a float).
- [ ] Write merged output with `epoch` set to the round boundary so `--resume`
      continues the global schedule.
- [ ] Decide optimizer state: average momentum, or reset per round.
- [ ] Unit tests on CPU: averaging two identical checkpoints must be a no-op;
      averaging two known-different ones must give the exact midpoint.

## Phase 3 -- validate on Snellius alone, before touching LUMI

Simulate two sites as two Snellius jobs on disjoint shards. Same mechanism, zero
cross-site networking, and a bug costs one queue wait instead of two.

- [ ] Baseline: 1 job, 2 nodes, 8 GPUs, 30 epochs (the existing multinode run).
- [ ] Federated: 2 jobs x 4 GPUs, disjoint shards, sync every K epochs.
- [ ] Sweep K in {1, 2, 5, 10} and find where accuracy departs from baseline.
- [ ] Sanity check: K=1 should track the baseline closely; K=30 is just two
      independent runs averaged once, and should be clearly worse.

Only once this behaves should LUMI enter the picture.

## Phase 4 -- cross-site transport

- [ ] Confirm SSH key auth Snellius -> LUMI actually works. Verified so far:
      outbound TCP/22 to `lumi.csc.fi` is **open**, and `~/.ssh/id_ed25519`
      exists. Whether that key is authorised on LUMI is **not yet tested**.
- [ ] **Snellius orchestrates.** MFA makes automated inbound SSH to Snellius
      painful, so drive rounds from Snellius: it pushes and pulls checkpoints,
      and LUMI only ever runs `sbatch`.
- [ ] Round driver: submit both sites, wait for both checkpoints, merge,
      redistribute, repeat. Plain shell or Python on a login node -- it is
      waiting on Slurm, not computing.
- [ ] Decide failure behaviour: if one site's job dies, does the round stall,
      or proceed with the surviving site?
- [ ] Checksum checkpoints after transfer. A truncated 85 MiB file loads as a
      confusing shape error deep inside a queued job.

## Open decisions

| question | proposed default | rationale |
|---|---|---|
| average optimizer momentum? | no -- weights + BN buffers only | classic FedAvg, halves transfer |
| data split | disjoint shards | otherwise it is local-SGD, not federated |
| sync interval K | start at 1 epoch, sweep up | find where it breaks |
| LR scaling | keep each site's own global batch | the *federated* batch is 2x one site's; whether to scale for that is a research question, not an infra one |

## Notes on the machines

- **Snellius multi-node must take whole nodes.** A multi-node GPU job asking for
  fewer than 4 GPUs/node is rejected; `--exclusive` bills for them anyway. So
  each federated site on Snellius is a whole node minimum.
- **Snellius single-node partial allocations start instantly** (1 GPU + 16 cores).
  `scripts/snellius/job_cifar_debug.sh` is the fast iteration loop -- use it for
  every Phase 1 and 2 change before spending a full-node queue wait.
- **`/scratch-shared` is purged at ~14 days.** Round checkpoints that matter
  belong in `/projects`, or they will vanish mid-experiment.
- **Snellius priority is ~98% fairshare** with a 1-day decay half-life. A long
  federated sweep will steadily lower your own queue priority.
