# DCLT campaign — paste-ready command sheet

Written 2026-08-17 ~13:30 CEST, reflecting the ACTUAL state of the machines at
that moment. Explanations live in [DCLT_ARM.md](DCLT_ARM.md) and
[RUNBOOK_DCLT.md](RUNBOOK_DCLT.md); this sheet is only what to paste, in order.

**State right now**

| thing | state |
|---|---|
| VM full-arm stack (29510/11) | LIVE, restarted 13:26, resumed at merge round 27 / step 2700 — see §1 |
| full-arm site jobs | UNKNOWN — probably killed by the 13:25 aggregator outage; **check first, §1** |
| VM dclt stack (29540/41) | LIVE with QSR + pure-average eval; token `fnbLsi1FOb9KJfF6A2OBpOPz7FPaBre` |
| SURF firewall | 29540–42 open (verified from LUMI) |
| LUMI dclt watcher | armed on uan04 (pid 141016; the duplicate 116129 should be killed) |
| wandb | project `pww-diloco-20k-elastic` throughout |

Tokens: full arm `SyI7p1o8adYX4Reaq5DmEYJ2MVOXNeX`, dclt
`fnbLsi1FOb9KJfF6A2OBpOPz7FPaBre`. Same `WANDB_API_KEY` everywhere (in
`$S/.pww-secrets` on the VM).

---

## 1. FIRST — recover the full arm (the run that must not be lost)

The 13:25 incident: stopping the dclt stack swept ports 29510–12 and killed the
full arm's aggregator mid-round (root cause fixed in
`stop_central_services.sh`; state was durable and the VM side is fully
restored). The SITE jobs likely died with the broken gRPC stream. On each site:

```bash
squeue --me | grep titan-full
```

If a site's job is gone, resubmit — **RESUME, so NO `PWW_FRESH_RUN`/
`PWW_FRESH_DELETE`**: the site reloads its own step-2700 checkpoint and the
aggregator hands it the round-27 global model. Fresh flags here would wipe the
site back to step 0 against a server at step 2700.

```bash
# Snellius resume (only if the job is gone)
export WANDB_API_KEY=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
export TOK_FULL=SyI7p1o8adYX4Reaq5DmEYJ2MVOXNeX
curl -sS -m 8 -o /dev/null -w 'darl says %{http_code}\n' \
  -H "X-DARL-Token: $TOK_FULL" http://145.38.206.143:29510/health          # want 200
sbatch -J pww-snellius-titan-full --time=24:00:00 \
  --export=ALL,DARL_TOKEN="$TOK_FULL",WANDB_API_KEY="$WANDB_API_KEY",ENABLE_WANDB=1,WANDB_PROJECT=pww-diloco-20k-elastic,WANDB_RUN_NAME=diloco-snellius-full-run5 \
  scripts/snellius/job_titan_diloco.sh
```

```bash
# LUMI resume (only if the job is gone; do NOT touch pww-lumi-central)
export WANDB_API_KEY=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
export TOK_FULL=SyI7p1o8adYX4Reaq5DmEYJ2MVOXNeX
curl -sS -m 8 -o /dev/null -w 'darl says %{http_code}\n' \
  -H "X-DARL-Token: $TOK_FULL" http://145.38.206.143:29510/health          # want 200
sbatch -A project_462000226 -p standard-g -J pww-lumi-titan-full --time=24:00:00 \
  --export=ALL,DARL_TOKEN="$TOK_FULL",WANDB_API_KEY="$WANDB_API_KEY",ENABLE_WANDB=1,WANDB_PROJECT=pww-diloco-20k-elastic,WANDB_RUN_NAME=diloco-lumi-full-run5 \
  scripts/lumi/job_titan_diloco.sh
```

Verify on the VM: `grep "cluster(s) \[" $S/runs/central/flower.log | tail -3`
should resume showing merges within one round time.

## 2. LUMI — one watcher, exactly one

```bash
pgrep -af follow_watch.sh        # exactly ONE line (pid 141016)
# if 116129 is still there:  kill 116129
```

The watcher submits the LUMI dclt job by itself when Snellius registers.
Nothing else to do on LUMI.

## 3. Snellius — start the DCLT arm (this is the trigger)

Fresh flags ON — this is a new run. `git pull --ff-only` first: the sites need
the DCLT client code (the aggregator-side pure-eval addition needs no site
code, but per-round H and endpoint eval do).

```bash
cd ~/ProjectWorldWideSnellius/ProjectWorldWide && git pull --ff-only
export WANDB_API_KEY=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
export TOK_DCLT=fnbLsi1FOb9KJfF6A2OBpOPz7FPaBre
curl -sS -m 8 -o /dev/null -w 'darl says %{http_code}\n' \
  -H "X-DARL-Token: $TOK_DCLT" http://145.38.206.143:29540/health          # want 200

sbatch -J pww-snellius-titan-dclt --time=24:00:00 \
  --export=ALL,CONFIG=configs/titan/qwen3_0.6b_c4_dclt.toml,PWW_DARL_PORT=29540,PWW_FLOWER_PORT=29541,DARL_TOKEN="$TOK_DCLT",WANDB_API_KEY="$WANDB_API_KEY",PWW_FRESH_RUN=1,PWW_FRESH_DELETE=1,ENABLE_WANDB=1,WANDB_PROJECT=pww-diloco-20k-elastic,WANDB_RUN_NAME=dclt-snellius \
  scripts/snellius/job_titan_diloco.sh
squeue --me -p gpu_h100
```

## 4. Checks once DCLT is running (VM)

```bash
S=/data/thomasistriplet/zpalanciya
grep "QSR is ON" $S/runs-dclt/central/flower.log                    # schedule live
grep -E "Training loss|Jensen gauge" $S/runs-dclt/central/flower.log | tail -4
grep "PURE average" $S/runs-dclt/central/flower.log | tail -1       # every 5th merge
grep "server set this round" <site .out> | tail -2                  # per-round H reaching sites
```

Round-1 loss ≈ 9.9; Jensen lines appear on the first two-site round; the
`[PURE average, momentum excluded]` perplexity line appears on merge 5, 10, …
WandB: `qsr/next_inner_steps`, `jensen/gap` vs `jensen/pure_gap` (their
difference = the Nesterov bias), `jensen/h_multiplier`, per-cluster
`endpoint_eval_loss`.

## 5. After (or alongside, queue permitting) — the comparison suite

### 5a. Central gb96 baselines (one per site) — matched tokens/step

```bash
# Snellius
PWW_GLOBAL_BATCH=96 sbatch --export=ALL,PWW_GLOBAL_BATCH,PWW_FRESH_RUN=1,ENABLE_WANDB=1,WANDB_PROJECT=pww-diloco-20k-elastic,WANDB_RUN_NAME=central-snellius-gb96,WANDB_API_KEY="$WANDB_API_KEY" \
  -J pww-snellius-central --time=24:00:00 scripts/snellius/job_titan_central.sh

# LUMI
PWW_GLOBAL_BATCH=96 sbatch --export=ALL,PWW_GLOBAL_BATCH,PWW_FRESH_RUN=1,ENABLE_WANDB=1,WANDB_PROJECT=pww-diloco-20k-elastic,WANDB_RUN_NAME=central-lumi-gb96,WANDB_API_KEY="$WANDB_API_KEY" \
  -A project_462000226 -p standard-g -J pww-lumi-central --time=24:00:00 scripts/lumi/job_titan_central.sh
```

### 5b. central-hot — the decider ablation (one site is enough)

The DCLT schedule inside a single-site run
(`configs/titan/qwen3_0.6b_c4_central_hot.toml`). If it diverges or loses to
5a, the hot recipe was only survivable under averaging; if it matches DCLT, the
win was the schedule. Run it AFTER the gb96 pair so the queue isn't fighting
itself:

```bash
# Snellius (or LUMI with the -A/-p flags)
PWW_GLOBAL_BATCH=96 CONFIG=configs/titan/qwen3_0.6b_c4_central_hot.toml \
sbatch --export=ALL,PWW_GLOBAL_BATCH,CONFIG,PWW_FRESH_RUN=1,ENABLE_WANDB=1,WANDB_PROJECT=pww-diloco-20k-elastic,WANDB_RUN_NAME=central-snellius-gb96-hot,WANDB_API_KEY="$WANDB_API_KEY" \
  -J pww-snellius-central-hot --time=24:00:00 scripts/snellius/job_titan_central.sh
```

### 5c. 1k-step lr probe (6e-4) — informs DCLT run 2, not run 1

⚠️ **Not while the LUMI follow-watcher is armed** — a probe registering as
cluster `snellius` on the dclt coordinator would fire it. Run this after the
DCLT arm is underway (the watcher has fired and exited) or against a throwaway
stack. Then:

```bash
# Snellius, ~1h; uses the dclt stack once it is otherwise idle
export TOK_DCLT=fnbLsi1FOb9KJfF6A2OBpOPz7FPaBre
sbatch -J pww-snellius-probe --time=02:00:00 \
  --export=ALL,CONFIG=configs/titan/qwen3_0.6b_c4_dclt.toml,PWW_DARL_PORT=29540,PWW_FLOWER_PORT=29541,DARL_TOKEN="$TOK_DCLT",PWW_FRESH_RUN=1,PWW_FRESH_DELETE=1,ENABLE_WANDB=1,WANDB_PROJECT=pww-diloco-20k-elastic,WANDB_RUN_NAME=probe-lr6e4,WANDB_API_KEY="$WANDB_API_KEY" \
  scripts/snellius/job_titan_diloco.sh
# with these overrides appended inside CONFIG's run via run_train extra args --
# or simplest: copy the dclt toml, set lr = 6e-4, steps = 1000, warmup_steps = 100.
```

If 6e-4 survives 1,000 steps with finite loss and sane grad_norm, DCLT run 2
(and central-hot run 2) move to 6e-4.

## Order of operations, condensed

1. §1 — check/resume the full arm on both sites (resume = **no fresh flags**).
2. §2 — one watcher on LUMI.
3. §3 — submit Snellius DCLT; LUMI follows automatically.
4. §4 — verify QSR/Jensen/pure lines on the VM.
5. §5a when queues allow → §5b after → §5c last.
