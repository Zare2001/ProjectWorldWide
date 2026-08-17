# The DCLT arm — dispersion-controlled local training

The experiment that tries to **beat both baselines at the same 20,000-step,
same-token budget**: the DiLoCo full arm (held-out ppl 39.59 on the completed
campaign) and the single-site central baselines, all in the same WandB project.
It is DiLoCo plus the three mechanisms from the local-update literature that a
constant-H, conservative-LR run leaves on the table — each one implemented as a
**server-side dial**, so the clusters stay dumb and the whole recipe is
controlled from the VM.

| # | Mechanism | Source | Where it lives |
|---|---|---|---|
| 1 | Hot local recipe: lr 4.5e-4 (1.5×), decay stretched to the last 50% | Photon, arXiv:2411.02908 | `configs/titan/qwen3_0.6b_c4_dclt.toml` |
| 2 | QSR: per-round H = 100·(η_max/η)², capped at 500 | arXiv:2310.14423 | `strategy._next_inner_steps`, broadcast as `pww_inner_steps` |
| 3 | Jensen gauge + controller: J = L(merged) − mean L(endpoints), H trimmed to keep J in [−0.06, −0.015] nats | soup/SWAP line of work | `strategy._update_jensen`; clients answer `pww_local_eval` with one extra val pass |

Why averaging pays for the hotter LR: conditionally on the round's starting
point the two sites' endpoints are independent, so the merged iterate carries
1/k of the local noise — the local path runs hot, the global iterate stays
cool. QSR then grows the sync period exactly where the extra local steps buy
sharpness reduction (the decay phase) instead of drift. The gauge is the guard:
J ≥ 0 means the endpoints left a common linearly-connected basin and averaging
is paying a barrier penalty — the controller cuts H 30% and says so in the log.

Deliberately NOT changed, so the comparison stays readable: model, corpus,
tokenizer, DARL partitioning, `training.steps = 20000`, local batch, and the
outer optimiser (Nesterov, η=0.7, β=0.9). What varies is only how the same
20,000 steps are grouped into rounds and how hot the inner path runs.

Mechanisms from the same notes that are deliberately **absent**: cohort-
covariance preconditioning and re-sharding annealing both need N ≫ 2 replicas;
at two sites they are noise.

---

## 1. Deploy — BOTH sites need the new code

The client half (per-round H, endpoint eval, an 8-field control broadcast) is
in `src/pww/titan/flower_client.py` and `src/pww/fedproto.py`. **Pull on
Snellius and LUMI before submitting.** Version skew degrades safely but
pointlessly: an old client ignores `pww_inner_steps` and trains fixed H=100
while a new one follows QSR (per-site H is supported), and an old server sends
nothing so a new client falls back to `darl.inner_steps`. The run you want is
new everywhere.

```bash
# on each site, in the repo
git pull --ff-only
```

## 2. VM — a fourth stack, its own ports and output dir

Ports follow the campaign convention (full 2951x, churn 2952x, latejoin 2953x):
**dclt = 29540/29541/29542**.

```bash
S=/data/thomasistriplet/zpalanciya
cd /data/thomasistriplet/ProjectWorldWide
source $S/.pww-secrets

# NUM_SAMPLES must equal the sites' manifest num_windows -- same value every
# stack uses. Read it from the full stack's remembered geometry
# (2756597 as of 2026-08-17; re-derive rather than hardcode):
NUM_SAMPLES=$(. $S/runs/darl/space.env; echo $NUM_SAMPLES)
echo "NUM_SAMPLES=$NUM_SAMPLES"     # sanity: must be the corpus window count, not empty

PWW_OUTPUT_DIR=$S/runs-dclt \
DARL_PORT=29540 FLOWER_PORT=29541 BLOB_PORT=29542 \
AGGREGATOR_CONFIG=configs/central_aggregator_dclt.yaml \
NUM_SAMPLES=$NUM_SAMPLES BLOCK_SIZE=1024 SEED=42 \
ENABLE_WANDB=1 WANDB_PROJECT=pww-diloco-20k-elastic \
WANDB_RUN_NAME=central-aggregator-dclt \
  ./scripts/central_node/start_central_services.sh

TOK_DCLT=$(cat $S/runs-dclt/darl/token)
echo "TOK_DCLT=$TOK_DCLT"
grep -E "QSR is ON" $S/runs-dclt/central/flower.log   # must print the QSR banner
```

If the startup log does not say `QSR is ON`, the aggregator is running the
wrong yaml — stop the stack and check `AGGREGATOR_CONFIG`.

## 3. Sites

Same submit pattern as the full arm, three differences: the config, the ports,
and this stack's token.

```bash
# Snellius
export WANDB_API_KEY=<from $S/.pww-secrets>
export TOK_DCLT=<value printed above>
curl -sS -o /dev/null -w 'darl says %{http_code}\n' \
  -H "X-DARL-Token: $TOK_DCLT" http://145.38.206.143:29540/health   # want 200

sbatch -J pww-snellius-titan-dclt --time=24:00:00 \
  --export=ALL,CONFIG=configs/titan/qwen3_0.6b_c4_dclt.toml,PWW_DARL_PORT=29540,PWW_FLOWER_PORT=29541,DARL_TOKEN="$TOK_DCLT",WANDB_API_KEY="$WANDB_API_KEY",PWW_FRESH_RUN=1,PWW_FRESH_DELETE=1,ENABLE_WANDB=1,WANDB_PROJECT=pww-diloco-20k-elastic,WANDB_RUN_NAME=dclt-snellius \
  scripts/snellius/job_titan_diloco.sh
```

```bash
# LUMI
export WANDB_API_KEY=<from $S/.pww-secrets>
export TOK_DCLT=<value printed above>
curl -sS -o /dev/null -w 'darl says %{http_code}\n' \
  -H "X-DARL-Token: $TOK_DCLT" http://145.38.206.143:29540/health   # want 200

sbatch -A project_462000226 -p standard-g -J pww-lumi-titan-dclt --time=24:00:00 \
  --export=ALL,CONFIG=configs/titan/qwen3_0.6b_c4_dclt.toml,PWW_DARL_PORT=29540,PWW_FLOWER_PORT=29541,DARL_TOKEN="$TOK_DCLT",WANDB_API_KEY="$WANDB_API_KEY",PWW_FRESH_RUN=1,PWW_FRESH_DELETE=1,ENABLE_WANDB=1,WANDB_PROJECT=pww-diloco-20k-elastic,WANDB_RUN_NAME=dclt-lumi \
  scripts/lumi/job_titan_diloco.sh
```

Walltime note: the same 20,000 steps as the full arm plus ~140 extra validation
passes (seconds each) and FEWER merges — total wall-clock is at worst the full
arm's, usually a little under it.

## 4. What to watch

New WandB series, all on the aggregator run:

| series | healthy looks like |
|---|---|
| `qsr/next_inner_steps` | flat 100 to ~step 10,000, then a quadratic climb to 500 |
| `jensen/gap` (and `gap_ema`) | negative, drifting in [−0.06, −0.015]; each point is the round's model-soup harvest in nats |
| `jensen/h_multiplier` | near 1.0; sustained 0.5 means repeated fracture — see below |
| `cluster/*/endpoint_eval_loss` | slightly ABOVE `eval/loss` — that gap is the harvest |

Log-side, per round on the VM:

```bash
grep "Jensen gauge" $S/runs-dclt/central/flower.log | tail -5
grep "QSR is ON" $S/runs-dclt/central/flower.log
```

Solo rounds (one site queued) run QSR's H but freeze the controller — one
endpoint has no dispersion to measure. That is normal elastic behaviour.

## 5. Success criteria and the honest caveats

* **The headline**: `eval/perplexity` < 39.59 at step 20,000, same tokens, same
  outer optimiser. Also compare against the central baselines in the project —
  the interesting outcome is beating both.
* **Mechanism attribution**: if `jensen/gap` hovers near 0 all run, the soup
  harvest was negligible at k=2 and any win came from mechanisms 1–2; that is a
  finding, not a failure.
* **This is iso-token, not iso-FLOP**: the endpoint evals add ~1% compute.
  Photon's own headline hides a 1.9× compute gap; ours does not, but say
  "matched tokens" when reporting, not "matched everything".
* **k=2 truncates the theory**: the harvest factor (1−1/N) is 0.5 and the tail
  suppression exp(−N·I) is at its weakest. If DCLT wins here it should win more
  with a third site (see ADDING_A_CLUSTER.md).

## 6. If it misbehaves

| symptom | response |
|---|---|
| a site diverges in the first hot rounds (non-finite loss, aggregator drops its contribution) | once: let the guards eat the round. Twice: drop `optimizer.lr` to 3.75e-4 in the toml and reset the arm |
| `jensen/h_multiplier` pinned at 0.5 | endpoints keep fracturing: the hot LR + long H combination is too much late in the run. Lower `qsr-max` to 300 in the yaml and restart the aggregator (state survives) |
| H thrashing round to round | widen the band (`jensen-lo: 0.01`, `jensen-hi: 0.08`) — the gauge noise on ~1M val tokens is a few hundredths of a nat |
| rounds dropped at timeout late in the run | H=500 rounds are ~35–40 min; the yaml already sets `round-timeout: 5400`. If it still fires, the slow site is the problem, not the timeout |
| aggregator restarted mid-run | fine: weights, momentum, merge round, global step AND the controller state (η_max, multiplier, gauge EMA) all resume from the npz checkpoint |

Tests covering all of this: `tests/test_federation.py` (48 checks — QSR
schedule, controller law, cap interaction, restart persistence, old-checkpoint
compatibility) and `tests/test_darl.py` (47) — both green on the VM venv.
