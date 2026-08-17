# RUNBOOK — the DCLT arm

Operational procedure for the DCLT run: clean reset, start, per-phase checks,
recovery, teardown. The theory and the launch rationale are in
[DCLT_ARM.md](DCLT_ARM.md); the general campaign procedures in
[RUNBOOK.md](RUNBOOK.md). This file is only what you type and what you check.

Stack identity, fixed for the whole arm:

| what | value |
|---|---|
| output dir (VM) | `$S/runs-dclt` (`S=/data/thomasistriplet/zpalanciya`) |
| ports | DARL **29540**, Flower **29541**, blob 29542 |
| aggregator config | `configs/central_aggregator_dclt.yaml` |
| site config | `configs/titan/qwen3_0.6b_c4_dclt.toml` |
| wandb | project `pww-diloco-20k-elastic`; runs `central-aggregator-dclt`, `dclt-snellius`, `dclt-lumi` |
| job names | `pww-snellius-titan-dclt`, `pww-lumi-titan-dclt` |

---

## 0. Does this compose with FedMom / Nesterov?

Yes — the outer step is **untouched**. QSR and the Jensen controller change only
*when* merges happen and *how hot* the inner path runs; the merge itself is the
same `v_next = w − η·(w − w_avg); w_next = v_next + β(v_next − v_prev)` at
η=0.7, β=0.9 (η=1 on solo rounds, as always). `global_step` already advances by
max(steps) across contributors, which was built for per-site H and covers
per-round H for free. It also composes with plain FedAvg (`server-momentum 0`):
neither mechanism reads the outer optimiser.

Two real interactions to know about — both bounded, one genuinely novel:

1. **Outer updates grow as H grows.** A pseudo-gradient is the sum of H inner
   steps, so its magnitude scales like η·H; under QSR (H ∝ η⁻²) that is ∝ 1/η —
   the outer step gets LARGER late in the run, and Nesterov compounds updates
   across rounds. The `qsr-max: 500` cap bounds the growth at 5× the opening
   round, and three guards sit behind it: the weight-growth check
   (`MAX_WEIGHT_GROWTH`), the non-finite drop, and the Jensen fracture cut
   (J ≥ 0 → H×0.7). The QSR paper validated its rule with plain averaging, not
   outer Nesterov — **this combination is the untested part of the arm**, and
   `train/drift_ratio_max` late in the run is where it would show first.
2. **The momentum timescale is per-merge, not per-step.** β=0.9 smooths over
   ~10 merges; early that is ~1,000 optimiser steps, late (H=500) up to ~5,000.
   Expected and benign — it means the outer filter gets *slower* exactly when
   the inner path is coolest — but it is a difference from the full arm worth
   remembering when reading the last ~30 rounds.

If the late run misbehaves (drift climbing round over round, multiplier pinned
at 0.5): lower `qsr-max` to 300 in the yaml and restart the aggregator. Do NOT
lower `server-momentum` for this arm — it would break comparability with the
full arm, which is the entire point of holding OuterOpt fixed.

---

## 1. Prerequisites

- **Both sites on the new code.** `git pull --ff-only` on Snellius and LUMI.
  The client half (per-round H, endpoint eval, 8-field control broadcast) ships
  with the server half; an old client silently trains fixed H=100 and reports
  no endpoint losses — the arm runs but measures nothing.
- Corpus, tokenizer, venvs: identical to the full arm; nothing new to stage.
- `outputs` symlinked to scratch on both sites (home-quota lesson from the
  first campaign) — verify: `ls -ld ~/ProjectWorldWide*/ProjectWorldWide/outputs 2>/dev/null; ls -ld ~/ProjectWorldWide/outputs 2>/dev/null`.

## 2. Clean reset (order is load-bearing: scancel → stop → delete)

⚠️ **Never `scancel -u $USER` on Snellius** — production `rome` arrays.
⚠️ Everything below is scoped to `runs-dclt`. It must not touch `$S/runs`
(the full arm + its DARL state), `runs-churn`, or `runs-latejoin`.

```bash
# 1. sites first -- a live job would re-register against the fresh coordinator
squeue --me | grep dclt          # on each site; scancel only those job ids

# 2. stop THIS stack's services (stack-scoped via PWW_OUTPUT_DIR)
S=/data/thomasistriplet/zpalanciya
cd /data/thomasistriplet/ProjectWorldWide
PWW_OUTPUT_DIR=$S/runs-dclt ./scripts/central_node/stop_central_services.sh

# 3. only now delete the stack's state
rm -rf $S/runs-dclt

# 4. site dumps (inside each site's repo; PWW_FRESH_RUN=1 also does this)
rm -rf outputs/qwen3-0.6b-c4-dclt
```

## 3. Start the VM stack

```bash
S=/data/thomasistriplet/zpalanciya
cd /data/thomasistriplet/ProjectWorldWide
source $S/.pww-secrets

NUM_SAMPLES=$(. $S/runs/darl/space.env; echo $NUM_SAMPLES)   # 2756597 as of 2026-08
echo "NUM_SAMPLES=$NUM_SAMPLES"                              # must not be empty

PWW_OUTPUT_DIR=$S/runs-dclt \
DARL_PORT=29540 FLOWER_PORT=29541 BLOB_PORT=29542 \
AGGREGATOR_CONFIG=configs/central_aggregator_dclt.yaml \
NUM_SAMPLES=$NUM_SAMPLES BLOCK_SIZE=1024 SEED=42 \
ENABLE_WANDB=1 WANDB_PROJECT=pww-diloco-20k-elastic \
WANDB_RUN_NAME=central-aggregator-dclt \
  ./scripts/central_node/start_central_services.sh
```

**Six startup checks — all must pass before any sbatch:**

```bash
TOK_DCLT=$(cat $S/runs-dclt/darl/token); echo "TOK_DCLT=$TOK_DCLT"        # 1. token exists
PWW_OUTPUT_DIR=$S/runs-dclt ./scripts/central_node/status_central_services.sh  # 2. both pids live
ss -tlnp 2>/dev/null | grep -E '29540|29541'                              # 3. both ports listening
grep "QSR is ON" $S/runs-dclt/central/flower.log                          # 4. THE dclt-specific one
curl -sS -o /dev/null -w 'darl %{http_code}\n' \
  -H "X-DARL-Token: $TOK_DCLT" http://145.38.206.143:29540/health         # 5. 200
grep -E "min_clients=1.*momentum=0.9|server_learning_rate=0.7|Syncing run" \
  $S/runs-dclt/central/flower.log | head -3                               # 6. outer opt + wandb
```

No `QSR is ON` line = wrong yaml reached the server. Stop the stack, check
`AGGREGATOR_CONFIG`, start again — do not submit sites against it.

## 4. Submit the sites

Concrete values only — the placeholder-pasted-literally failure has happened
twice. Fill `TOK_DCLT` with the token check 1 printed.

```bash
# Snellius (~/ProjectWorldWideSnellius/ProjectWorldWide)
export WANDB_API_KEY=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
export TOK_DCLT=<paste the value check 1 printed>
curl -sS -o /dev/null -w 'darl says %{http_code}\n' \
  -H "X-DARL-Token: $TOK_DCLT" http://145.38.206.143:29540/health          # want 200

sbatch -J pww-snellius-titan-dclt --time=24:00:00 \
  --export=ALL,CONFIG=configs/titan/qwen3_0.6b_c4_dclt.toml,PWW_DARL_PORT=29540,PWW_FLOWER_PORT=29541,DARL_TOKEN="$TOK_DCLT",WANDB_API_KEY="$WANDB_API_KEY",PWW_FRESH_RUN=1,PWW_FRESH_DELETE=1,ENABLE_WANDB=1,WANDB_PROJECT=pww-diloco-20k-elastic,WANDB_RUN_NAME=dclt-snellius \
  scripts/snellius/job_titan_diloco.sh
squeue --me -p gpu_h100
```

```bash
# LUMI (~/ProjectWorldWide)
export WANDB_API_KEY=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
export TOK_DCLT=<paste the value check 1 printed>
curl -sS -o /dev/null -w 'darl says %{http_code}\n' \
  -H "X-DARL-Token: $TOK_DCLT" http://145.38.206.143:29540/health          # want 200

sbatch -A project_462000226 -p standard-g -J pww-lumi-titan-dclt --time=24:00:00 \
  --export=ALL,CONFIG=configs/titan/qwen3_0.6b_c4_dclt.toml,PWW_DARL_PORT=29540,PWW_FLOWER_PORT=29541,DARL_TOKEN="$TOK_DCLT",WANDB_API_KEY="$WANDB_API_KEY",PWW_FRESH_RUN=1,PWW_FRESH_DELETE=1,ENABLE_WANDB=1,WANDB_PROJECT=pww-diloco-20k-elastic,WANDB_RUN_NAME=dclt-lumi \
  scripts/lumi/job_titan_diloco.sh
squeue --me
```

Do NOT cancel `pww-lumi-central` (the LUMI baseline) while clearing anything.

**Auto-follow: submit LUMI the moment Snellius starts.** LUMI cannot see
Snellius's queue, but the coordinator sees Snellius the instant its job
registers — `scripts/follow_watch.sh` polls `/status` for that and fires once
(elastic_watch.sh's sibling: membership instead of a block threshold). Run it
on a LUMI login node INSTEAD of the manual sbatch above, from the shell where
the token and key are exported — the sbatch arguments are expanded at launch,
so check the log's first line echoes real values, not `$TOK_DCLT` literally:

```bash
cd ~/ProjectWorldWide
export WANDB_API_KEY=<from the secrets file>
export TOK_DCLT=<this stack's token>

nohup scripts/follow_watch.sh \
  --url http://145.38.206.143:29540 --token "$TOK_DCLT" --cluster snellius -- \
  sbatch -A project_462000226 -p standard-g -J pww-lumi-titan-dclt --time=24:00:00 \
    --export=ALL,CONFIG=configs/titan/qwen3_0.6b_c4_dclt.toml,PWW_DARL_PORT=29540,PWW_FLOWER_PORT=29541,DARL_TOKEN="$TOK_DCLT",WANDB_API_KEY="$WANDB_API_KEY",PWW_FRESH_RUN=1,PWW_FRESH_DELETE=1,ENABLE_WANDB=1,WANDB_PROJECT=pww-diloco-20k-elastic,WANDB_RUN_NAME=dclt-lumi \
    scripts/lumi/job_titan_diloco.sh \
  > watch-dclt-follow.log 2>&1 &

tail -f watch-dclt-follow.log     # first line must show the expanded command
```

An unreachable coordinator (VM reboot, firewall port not yet open) is retried
forever, so the watcher can be started BEFORE the ports are opened; a refused
token is fatal. If a login node reaps it, relaunch — a duplicate sbatch shows
up in squeue; scancel the extra one.

## 5. Checks per phase of the run

**First round of each site (minutes after it starts):**

```bash
# site log
grep -E "token accepted|Syncing run|server set this round" <newest .out> | head
# want: token accepted, Syncing run dclt-*, and NO "Failed to create WandB"
```

**First two-site merge (VM):**

```bash
grep -E "Training loss|cluster\(s\) \[" $S/runs-dclt/central/flower.log | tail -4
grep "Jensen gauge" $S/runs-dclt/central/flower.log | tail -3
```

Round-1 loss ≈ 9.9. The first `Jensen gauge` line appears on the first round
where BOTH sites contributed (solo rounds skip it — normal while one is
queued). Gap should be **negative** from early on; its magnitude will be small
(~−0.01 to −0.05 nats).

**Mid-run (~step 10,000 onward) — the QSR ramp:**

- wandb `qsr/next_inner_steps`: flat 100 until ~10,000, then a smooth climb.
  Log-side: `grep "server set this round's inner steps" <site .out> | tail`.
- `jensen/h_multiplier` near 1.0. Repeated `H multiplier ... -> 0.7` lines =
  fracture; see §0's response.
- Round cadence slows and checkpoints get sparser in wall-clock — expected,
  it's one checkpoint per round by design (`checkpoint.interval = 1`).

**Endgame (last ~30 rounds, H at the 500 cap):**

- Rounds take ~35–40 min. `round-timeout` is 5400 in the yaml; if you still
  see dropped rounds, the slow site is the problem, not the timeout.
- Watch `train/drift_ratio_max`: a monotone climb here is the
  QSR×Nesterov interaction from §0 — lower `qsr-max` next reset.

## 6. Recovery

| event | action |
|---|---|
| aggregator dies / VM reboot | rerun §3's start block unchanged. Weights, momentum, merge round, global step AND controller state (η_max, multiplier, Ĵ EMA) resume from the npz checkpoint. Do not pass FRESH_MODEL. |
| site hits walltime / requeued | just resubmit §4's block with the same token. Stale deltas are rejected by the generation check; the LR schedule realigns via `pww_global_step`. |
| one site queued for hours | nothing to do — solo rounds are DiLoCo k=1 with QSR's H; the controller freezes until both are back. |
| site diverges at the hot LR | once: the guards drop the round, DARL blocks are released, run continues. Twice: `optimizer.lr = 3.75e-4` in the toml, full reset (§2). |
| token 401 at submit | the shell holds a stale `TOK_DCLT` from a previous stack. Re-read it from `$S/runs-dclt/darl/token`, re-run the curl, resubmit. |

## 7. End of run

The clients stop themselves at `training.steps = 20000` (`exhausted`); the
aggregator logs a final merge and idles. Then:

1. Record final `eval/perplexity` and the full Ĵ trace.
2. Compare iso-token: DCLT vs full arm (39.59) vs `central-lumi-gb96` /
   `central-snellius-gb96`.
3. Decide mechanism VI by the Ĵ trace (see DCLT_ARM.md §5): clearly negative →
   build SparseLoCo-style EF compression next; ≈0 → skip it, push the
   temperature/schedule dials.
4. Teardown when the checkpoints are harvested:
   `PWW_OUTPUT_DIR=$S/runs-dclt ./scripts/central_node/stop_central_services.sh`
   — keep `$S/runs-dclt` on disk until the analysis is written up (31.7 GiB
   was the per-arm figure for the campaign stacks; this one is similar).
