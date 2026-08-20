# Planning a federated run

`src/pww/plan/` answers one question: **which sites to submit, at what shape, for how
long, starting when** — given the queue as `sbatch --test-only` currently sees it, the
corpus DARL has left, and the measured cost of a round.

It exists because a Flower round is a barrier. Every live site runs `H` inner AdamW
steps and the server merges whoever delivered; the round cannot close until the slowest
live site is done. So the inner phase is a **max** over the live sites and the
transport/merge/evaluate overhead is a **sum** over them. Three things follow that no
per-site capacity model can express:

- sites must overlap in time to federate at all — a site whose queue opens after the
  others' walltime has expired contributes nothing federated, however much it trains;
- adding a slow site lowers the round rate for everyone already in, so *is this site
  worth submitting* is a real question with a computable answer;
- gradient accumulation fills the barrier and costs DARL blocks linearly, so balancing,
  membership and shape are one decision, not three.

## What it decides

| variable | values |
|---|---|
| membership | which sites to submit at all, including none |
| shape | a **probed** `(partition, devices, walltime)`. Walltime is part of the key, never a knob attached afterwards |
| lanes | independent durable identities per site (`--replica`, own `PWW_DUMP`, own checkpoint). Concurrency comes from lanes |
| links | how many jobs a lane chains through, and how (`self`-resubmit, `singleton`, none) |
| begin | `--begin`, from `{now} ∪ {when each other site is predicted productive}` |
| balance | `PWW_GRAD_ACCUM` per site, solved jointly with the above |

The objective:

```
U = N_fed + alpha * N_solo + beta * (tokens / 1e9)      alpha default 0.25, beta default 0
```

`N_fed` counts merges with two or more distinct **sites** contributing (two lanes at one
site are two Flower clients, but they share the hardware, the queue, the WAN link and the
failure — that is not the measurement this campaign makes). Solo rounds are real work —
this campaign's own finding is that a centralized run beats the federated one at matched
tokens — but a run whose purpose is a federated measurement needs federated rounds, so
`alpha` is the exchange rate and every plan reports `alpha*`, the value at which the
recommendation changes.

Reported on every plan and never optimised: tokens, DARL blocks used, GPU-hours,
tokens per GPU-hour, barrier idle fraction, round period per membership regime, and the
split of each site's live hours into headstart / co-resident / tail. Re-rank by hand from
those if you disagree with `alpha`.

## Why the scanner's own `POST /plan` does not answer this

`slurm-scanner`'s planner is a waterfill for **embarrassingly parallel** work: capacity
is a per-site integral of `(horizon − wait) × rate`, sites never interact, and a site
whose wait exceeds the horizon simply gets zero units. There is no co-residency, no round
rate, and no notion of one site dragging another down. No parameter choice makes it
express a barrier — the model is wrong, not mistuned.

It is also blind to the walltime a shape was probed at: `candidates()` takes the GPU count
from `plan.json` and the walltime cap from a cluster-level `max_walltime_hours`, and never
reads `-t` from the probe's own `args`. Reproduced: with a probe for `h100_full_8h`
(`... -t 8:00:00`) and `max_walltime_hours: 120`, `/plan` returns that shape with
`walltime_hours: 12.4` — it quotes a wait measured for an 8 h job and tells you to submit
a 12.4 h one. A 40 h job does not start when a 4 h job would; that curve is the one thing
the probes measure.

So we read `GET /probes`, `GET /usage` and `GET /overview`, and ignore `POST /plan`. No
patch to the scanner, no new endpoint, no extra CSV column (extending `probes.csv` in
place corrupts every existing file — `append()` writes the header only when the file is
absent).

## The arithmetic

```
period(S) = H * max_i(a_i * step_i)  +  o_merge  +  sum_i o_i
step_i    = batch_i / tput_i
o_i       = xfer_i + eval_fix_i + V / (3 * tput_i)          V = PWW_VAL_WINDOWS = 512
a_i       = clamp(int(max_j(step_j)/step_i + 0.5), 1, 8)    when balancing, else 1
```

With the measured registry (`configs/site_throughput.env`): Snellius 32 seq/step at
89.8 seq/s → `step = 0.356 s`; LUMI 64 seq/step at 38.2 seq/s → `step = 1.675 s`.
The `/3` is forward-only validation: no backward pass, no optimiser step.

```
o_snellius@4 = 33.5 + 23.6 + 512/(3*89.8) =  59.0 s
o_lumi@8     = 64.2 + 34.7 + 512/(3*38.2) = 103.4 s
o_merge      = 17 s        fp32 weighted mean + Nesterov outer step + 5.29 GiB npz write

snellius@4 solo, H=100 :  100*0.356 + 17 + 59.0                = 111.6 s   (measured 113)
snellius@4 + lumi@8    :  100*1.675 + 17 + 59.0 + 103.4         = 346.9 s   (measured 353)
snellius@1 accum 5     :  100*1.639 + 17 + 64.1                 = 245.0 s   (measured 248)
```

`residuals()` recomputes all four calibration regimes; three are reproduced to −1.2 %,
−1.7 %, −1.2 %. The fourth (both sites at one device each) predicts 384.3 s against a
measured 363 s, +5.9 %, and is shipped tagged `extrapolated` — see *When to distrust*.

**The mistake this replaces is `round = H * step_time`.** The server's
`>> Round took Ns` log line and the wandb series `train/round_seconds` are
`max(metrics["seconds"])`, the slowest client's inner phase only. They exclude three
1.32 GiB WAN crossings per site per round, the merge, the checkpoint write and the whole
evaluate barrier. On a two-site round that is 179 s of the 347 s: a planner built on the
logged number overestimates round throughput by ~2.1x.

**Idle.** In that two-site round Snellius computes for 35.6 s and the round lasts 346.9 s.
The pre-balancing figure quoted elsewhere in this repo (74 %) counts only the phase; the
planner's `idle_fraction` counts the whole billed round, so it reads higher.

**Data.** One block = 1024 windows = 2,097,152 tokens (1024 x 2048).

```
blocks/round = H * sum_i(a_i * batch_i) / 1024
  unbalanced (1,1): 100*(32+64)/1024      =  9.375   ->  87 merges from 822 blocks
  balanced   (5,1): 100*(5*32+64)/1024    = 21.875   ->  37 merges from 822 blocks
```

Both consume the corpus; only the merge count differs. Simulated at a 48 h horizon with
today's 822 free blocks: balance off → 87 federated merges, 1.717 G tokens, run ends at
8.6 h; balance on → 37 federated merges, 1.704 G tokens, run ends at 3.9 h. **Under a
data cap, `PWW_BALANCE=1` costs 2.3x of the remaining federated merges for no extra
tokens.** Under a walltime cap it is free extra tokens at no extra round time. Which
budget binds is not knowable before simulating, so `balance="auto"` simulates both and
the report says which one bound.

Data at risk if a job dies mid-phase is one lease: 4 blocks for Snellius at H=100 accum 1
(16 at accum 5), 7 for LUMI — 0.15 % / 0.26 % of the corpus, returned by the SIGTERM
handler in milliseconds or by TTL on a hard kill.

**Budgets that bind before walltime does.** DARL blocks (`unassigned` from `GET /status`;
with `max_epochs = 1` exhaustion *ends the run*, `acquire` returns `epoch_complete`
forever) and Flower round attempts (`--num-rounds`; every started round consumes one,
solo included; zero live clients consumes none, because `sample()` blocks in `wait_for`).
The simulator debits both round by round and prints `recommended_num_rounds`.

**Traps.** A plan that federates zero times is degenerate rather than low-scoring, so it
is flagged before ranking and only returned if every alternative is also trapped. Two ways
in, with opposite fixes:

```
trap_no_overlap          lumi 4 h now + snellius quoted 20 h -> 0 federated, 214 solo
                         "lumi's last job ends at 4.0 h and snellius is not productive
                          until 20.0 h"                        fix: longer walltime or --begin
trap_corpus_exhausted    lumi 40 h full node now + snellius at 20 h
                         "lumi exhausted the corpus at 10.5 h, before snellius became
                          productive at 20.0 h"                fix: shorter headstart, or accum 1
```

The second is only reachable by a simulator that debits blocks round by round; no
closed-form headstart inequality finds it. Both closed forms in `search.py`
(`chain_breakeven_c`, `headstart_checks`) are computed **alongside** the simulator as
cross-checks and printed when they disagree with it. The simulator is authoritative.

**Chaining.** `c* = [t*w(T) − T*w(t)] / [(T − t) + w(T) − w(t)]`; chain links of length
`t` instead of one job of length `T` iff the per-job startup `c < c*`. If the queue is
insensitive to walltime, `w(t) = w(T)` and `c*` is negative: chaining only ever buys a
queue advantage, and paying per-job startup for one that does not exist is pure loss. The
asymmetry falls out per site against that site's own `w(T)` — the site with the bad queue
chains, the site with the good queue takes one long job, and they routinely get different
policies in the same plan. Chaining defaults to self-resubmission (`--begin=now+(T−eps)`
from inside the predecessor, `.stop` sentinel) rather than `--dependency`, because a
dependency-held job is not backfill-eligible, which forfeits exactly the advantage that
motivated short jobs.

**H.** `h_model="fixed"` is the full arm: `configs/central_aggregator_titan.yaml` has no
`qsr-h0` key, so `--qsr-h0` defaults to 0 and H is `darl.inner_steps = 100` for the whole
run. Applying the QSR formula there is wrong by up to 5x per round. `"qsr"` reimplements
`strategy._next_inner_steps` including the cap-before-multiplier ordering (so the
effective ceiling is `qsr_max/2 = 250` whenever the Jensen multiplier sits at its 0.5
floor) and the freeze that stops the controller moving while fewer than two clusters
report a loss. `"replay"` walks the measured DCLT trace.

## Inputs

| what | from | if missing |
|---|---|---|
| `w(T)` per probed shape | scanner `GET /probes?hours=168`, or `probes.csv` off disk | excluded, with the exact JSON to paste into `configs/slurm_probe/<site>.json` |
| `used_ratio` | scanner `GET /usage` | no discount, `used_ratio: null` printed |
| `tput`, `batch` per (site, devices) | `configs/site_throughput.env` | excluded, naming `scripts/titan/calibrate_throughput.sh` |
| blocks left, digest | DARL `GET /status` (token required on GETs) | no plan |
| `o_i`, `o_merge`, regimes | `configs/plan/federation.json` over the built-in table | built-in table |
| startup cost `c` | `federation.json`, flagged `lower_bound` | — |
| MaxSubmitJobs etc. | nowhere; there is no `sbatch`/`sacct`/`scontrol` on the aggregator VM | `assumed`, echoed as such |

Nothing is interpolated — not in walltime, not in device count, not in throughput. A
shape nobody probed produces an exclusion naming the shape, never a guessed `w(T)`:

```
shape_not_probed  lumi/standard-g 8 gpu 8 h
  no probe for lumi/standard-g 8 gpu 8 h; w(T) is never interpolated in walltime,
  device count or throughput
  fix: add {"name": "standardg_1node_8h", "args": ["-A", "project_462000226", "-p",
  "standard-g", "-N", "1", "--gpus-per-node", "8", "-t", "8:00:00"]} to
  configs/slurm_probe/lumi.json and restart the collector loop
```

Today the live scanner at `http://145.38.195.124:8000` probes **1 h and 8 h only**, and
the campaign submits at 24–40 h. PWW's own instance (`configs/slurm_probe/*.json`, which
does probe 1/4/8/24/40 h) is not up: `145.38.206.143:29513` refuses connections. Until it
is, the shapes actually used are unprobed and the planner will say so rather than
extrapolate.

## Running it

```bash
scripts/plan_campaign.sh                          # live sources, campaign defaults
scripts/plan_campaign.sh --dry-run tests/fixtures/plan/two-site.json
scripts/plan_campaign.sh show                     # every input, with provenance
scripts/plan_campaign.sh sbatch                   # the commands, nothing else
scripts/plan_campaign.sh --alpha 0 --json > plan.json
python3 -m pww.plan --help                        # every knob
```

`--scanner-url` defaults to the `server` field of `configs/slurm_probe/*.json`, i.e. the
instance this checkout's own collectors POST to. Upstream's deployment
(`145.38.185.196:8000`) does not answer from the aggregator VM and `:29513` is not up, so
neither is a usable default.

Exit status describes the PLAN: 0 recommended, 1 flagged as a trap or priced off an
unmeasured regime, 2 nothing admissible.

The API is also the interface, for a notebook or a test:

```python
from pww.plan import make_plan, PlanConfig
from pww.plan.inputs import (build_shapes, fetch_table, fetch_darl_status,
                             load_calibration, load_throughput)
from pww.plan.model import PlannerInputs, SiteInput

SCANNER = "http://145.38.195.124:8000"          # off disk instead of over HTTP:
DARL    = "http://145.38.206.143:29510"         #   inputs.read_csv_table(dir, c, "probes")
TOKEN   = open("/data/thomasistriplet/zpalanciya/runs/darl/token").read().strip()

cal, _  = load_calibration("configs/plan/federation.json")
geo, _  = load_throughput("configs/site_throughput.env")
darl    = fetch_darl_status(DARL, TOKEN)
startup = {"snellius": 108.0, "lumi": 216.0}    # federation.json, LOWER BOUNDS

sites = []
for c in ("snellius", "lumi"):
    shapes, waits, ex = build_shapes(
        c, fetch_table(SCANNER, "probes", c, 168), fetch_table(SCANNER, "usage", c, 720))
    sites.append(SiteInput(site=c, shapes=shapes, waits=waits,
                           geometries=geo.get(c, {}), startup_s=startup[c]))

plan = make_plan(PlannerInputs(sites=tuple(sites), calibration=cal, darl=darl),
                 PlanConfig(alpha=0.25, horizon_s=48 * 3600, num_rounds=400))
print(plan.describe(), plan.rankable, plan.traps)
```

`dataclasses.asdict(plan)` serialises the whole tree — every stage is a frozen dataclass
of builtins, so no custom encoder is needed. The package is pure except `inputs`: no
clock, no RNG, so identical inputs give a byte-identical plan.

Knobs worth naming: `alpha`, `beta`, `horizon_s`, `num_rounds`, `inner_steps`,
`h_model` (`fixed|qsr|replay`), `balance` (`auto|on|off`), `wait_quantile`,
`discount_strength`, `lanes_max`, `max_links_per_lane`, `chain_policies`,
`chain_wait_overlap`, `reserve_blocks`, `assume_overhead`.

Two defaults that cost you data if you forget them: `simulate(..., record_rounds=True)`
is required for `Timeline.rounds` to be populated (the counts are always right without
it), and `round_cost(..., explain=True)` for `RoundCost.arithmetic`. Both are off because
they were the top two entries in the profile.

## Reading the output

Every figure in this section is pasted from one hermetic run:

```bash
PYTHONPATH=src python3 -m pww.plan --dry-run tests/fixtures/plan/two-site.json
```

The fixture is a live capture (2026-08-19 14:22 CEST, 822 of 2692 DARL blocks free)
replayed at its capture time, so the output is identical run to run. Regenerate this
section by pasting from that command; never edit a figure in place. Hand-editing is how
the block below came to claim 18.2 M tokens/GPU-h — a figure from before GPU-hours
started counting the held allocation — and `tests/test_plan_report.py` now re-runs the
command and fails on any line quoted here that has drifted.
`tests/fixtures/scanner_snapshot.json` is the same scanner capture without the DARL and
throughput blocks: it plans identically but reads the live coordinator, so its corpus
figures move.

Section 5, the verdict. A line ending in `...` is quoted to its first words only, and a
bare `...` is a dropped line:

```
  federated merges   71
  solo merges        50
  tokens             1.72 G
  DARL blocks        822 used of 822 available   EXHAUSTED at 16.4 h
  round attempts     121 of 400
  GPU-hours          220.6   tokens/GPU-h 7.81 M   idle 86% of billed GPU-h
  ...
  run ends at        16.4 h (DARL exhausted; ...
  first federated    7.9 h    last federated 15.5 h

  U = N_fed + 0.25*N_solo + 0*Tok/1e9 = 71 + 0.25*50 + 0*1.72 = 83.50
  alpha* (the value at which the recommendation changes): 0.333
  search             exact+greedy, 221 plans, measured optimality gap 0.0% (exact, proved optimal)
  RECOMMENDED NUM_ROUNDS=152  (for scripts/central_node/start_central_services.sh)
```

**`GPU-h` is the allocation, not the run.** The run ends at 16.4 h and the jobs hold their
allocation to the 48 h horizon, so 126 of the 220.6 GPU-h are billed after there is
anything left to train on; the report warns in those words. Divide the same tokens by the
94.6 GPU-h that had corpus and tokens/GPU-h reads 18.2 M instead of 7.81 M — 2.3x in the
flattering direction, on a figure that is reported and never optimised, so nothing else
in the plan disagrees with it. Shorten the chain, or make the client exit when DARL is
exhausted.

Snellius' 8 h full-node shape quotes 13.27 h (p50, raw) → 7.71 h after the discount;
LUMI's `standard-g` full node quotes 0. Three plans exist and they are not equivalent:
submit LUMI now on a short walltime (it trains solo and dies as its partner appears —
zero federated rounds, and this is what "start ASAP, ask for the wait you were quoted"
produces); submit LUMI now on a long one (solo headstart, then co-residency); or delay
LUMI so both arrive together. The planner takes the third here, because a LUMI headstart
spends corpus the federated phase then does not have and at 822 free blocks the corpus is
what binds.

Section 3 is the presence bar, which exists because co-residency is a relation between
rows and no column can show it:

```
             0 h                                                          48 h
  lumi      |...........######:####............................................|
  snellius  |~~~~~~~~~~~############~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
  federated |           ^^^^^^ ^^^^                                            |
             legend: ~ queued (free)   : starting up (billed, idle)   # productive   ^ two sites live
             the run ENDED at 16.4 h; the jobs' remaining walltime past that is not drawn
```

`~` is free and `:` is not: the `:` in LUMI's row is a chain-link changeover — allocated,
billed, contributing nothing. Snellius' trailing `~` is a successor queued behind every
link; the hours those links go on to hold are the 126 GPU-h above, and the bar stops
drawing them at the run end rather than showing them as work.

Interval table, one row per stretch of constant membership (`period` is the mean over that
row's rounds):

```
    from     to  members                       H  period rounds  fed  blocks  stop
  ------------------------------------------------------------------------------
     7.7    7.9  snellius-l0                 100    529s      1    0     819  horizon  [derived_by_subtraction]
     7.9    8.7  snellius-l0,lumi-l0         100    347s      8    8     744  membership  [derived_by_subtraction]
     8.7    8.8  snellius-l0                 100    121s      4    0     731  horizon  [derived_by_subtraction]
  ...
    14.8   15.6  snellius-l0,lumi-l0         100    347s      9    9      69  walltime  [derived_by_subtraction]
    15.7   16.4  snellius-l0                 100    112s     22    0       0  darl_exhausted
```

- 529 s in the first row is one round, not a mean: LUMI connects during it, which charges
  the 378 s cold-join stall to the incumbent's phase and LUMI's evaluate share (39.2 s) to
  the overhead — 111.6 + 378 + 39.2. A joining lane is sampled by `configure_evaluate`
  before `configure_fit` ever sees it, so it pays the evaluate barrier and contributes
  nothing to that merge.
- 121 s solo rounds between LUMI links, against 112 s in the last row: three plain 111.6 s
  rounds and one 150.8 s round in which LUMI's next link has connected and is sampled into
  the evaluate barrier (+39.2 s). A rejoin inside a lane resumes that lane's own
  checkpoint, so it pays no stall — only the lane's first link is cold.
- Those 0.1 h solo stretches are LUMI's chain-link boundaries: 216 s of startup plus the
  round the departing link could not finish inside its walltime. Only 0.4 h of it is
  absence; through the rest LUMI is allocated and starting up, which is why the ledger
  below reports 0.4 h of BETW and not 0.9 h.
- `[derived_by_subtraction]` on every row LUMI is in: no LUMI-solo round exists in any
  log, so its overhead is a two-site measurement minus Snellius' share.
- Read `stop` on the **terminal** row. That is where `darl_exhausted` /
  `attempts_exhausted` appear; on earlier rows it is bookkeeping.

Per-site ledger:

```
  site      shape            queue  start   HEAD   BETW  CO-RES   TAIL   GAP   idle  merges    tokens  blocks   GPU-h  tok/GPU-h accum
  lumi      1node_1h          7.0h    29m   0.0h   0.0h    7.5h   0.0h  0.4h    56%   71( 71)  930.61 M   443.8    60.2    15.47 M [1]
  snellius  h100_full_8h     47.7h    11m   0.0h   0.4h    7.5h   0.7h  0.0h    86%  121( 71)  792.99 M   378.1   160.5     4.94 M [1]
```

- `queue` is summed over **links** and, under self-chaining, overlaps the predecessor's
  runtime: Snellius' 47.7 h is 7.71 h for the first job plus 5 x 8 h of successors queueing
  while their predecessor runs. It is not dead time.
- Four hour columns, not one: HEAD 0.0 h, BETW 0.4 h, CO-RES 7.5 h, TAIL 0.7 h. BETW is
  solo presence *between* two co-residency spells, which chaining makes routine and which
  a head/co-resident/tail split silently folds into one of the other three.
- `idle` is over the whole live window, so Snellius' 86 % is mostly the cost of waiting at
  a barrier set by LUMI's 1.675 s step. The headline `idle 86%` is the same quantity over
  the federation, weighted by device count.
- `accum [1]` at both sites: `balance="auto"` turned balancing off because DARL bound
  before walltime did.

Marginal ledger — *is this site worth submitting at all*, as a number:

```
  lumi kept at alpha=0.25: round rate 32.2/h -> 10.4/h, tokens/round 6.6M
    -> 19.7M, delta-U = +17.8. The verdict flips at alpha = 0.33
  snellius kept at alpha=0.25: round rate 12.5/h -> 10.4/h, tokens/round
    13.1M -> 19.7M, delta-U = +61.5. The verdict flips at alpha = 1.87
```

Adding LUMI cuts the round rate from 32.2/h to 10.4/h and triples tokens per round; at
`alpha = 0.25` that is worth +17.8, and at `alpha ≥ 0.33` it is not. Section 6 agrees: at
`alpha` 0.5 and 1.0 the winner is Snellius alone, 263 solo merges, zero federated. **The
default sits 0.08 away from flipping**, which is the single most important line in this
particular plan.

## The two ways the wait estimate is wrong

**1. `--test-only` is a pessimistic simulation of a queue that is not necessarily yours.**
It answers for the queue as it is right now and assumes every running and pending job
occupies its full requested walltime. `sacct` says how wrong that is: `used_ratio` on
`gpu_h100` is 0.1617 over 4169 jobs in a 48 h window — finished jobs consumed 16 % of what
they asked for. So the raw number is usually far too long, and by an amount nobody can
predict for a specific job. It is also conditioned on the **probing account's** fairshare,
QOS and priority: the snapshot above was probed by `douwew` (Snellius) and `vanderwal`
(LUMI), not by the account that submits our jobs, so it describes a different queue.
`probed_by_user` and `probe_age_s` are printed on every row for this reason, and a probe older than
`max_probe_age_s` (6 h) is excluded rather than used.

**2. The discount is a linear blend, not a model.**

```
w_eff = w_raw * (1 - strength * (1 - used_ratio))
      = 47772 s * (1 - 0.5 * (1 - 0.1617)) = 27749 s     # 13.27 h -> 7.71 h
```

It assumes the whole queue shrinks by one factor, which is wrong precisely when the
blocking jobs are the long ones — they are the jobs that do not finish early. And it
**silently no-ops** when `used_ratio` is null, ≤ 0 or > 1: a partition whose `sacct`
window found no finishing jobs produces no usage row at all, so the plan looks discounted
when it is not. The only tell is `WaitEstimate.discounted`, printed on every row.

Both `w_raw` and `w_eff` appear everywhere, and every plan is re-solved at
`discount_strength ∈ {0, 0.5, 1}`. In the worked example, `s = 0` (trust the pessimistic
probe) flips the recommendation to Snellius alone. Treat the wait as a range, not a
number: p50 prices the plan, p90 decides feasibility ("will A still be alive when B
arrives"), and a winner that changes between them is fragile, not resolved. It does not
change here: at p90 the plan still federates, at U 80.25 against 83.50 and 65 federated
merges against 71. The quantile is not what this recommendation is fragile to — `s = 0`
is.

## When to distrust the answer

Read `plan.traps` before `plan.selection`. Then, in order:

1. **`plan.rankable` is False.** The dominant round regime is not one of the measured
   ones (`quality: extrapolated`), so it is priced but not ranked. Either measure it or
   pass `assume_overhead=True` knowingly.
2. **A LUMI-solo round appears anywhere.** No LUMI-solo round exists in any log — LUMI
   never ran without Snellius — so `xfer_lumi` and `eval_fix_lumi` are a two-site
   measurement minus Snellius' share, flagged `derived_by_subtraction`. Any plan whose
   value sits in LUMI headstart hours rests on that subtraction.
3. **`alpha*` is close to `alpha`.** 0.33 against a default of 0.25 means the
   recommendation is one judgement call away from inverting. Say which side you are on.
4. **The `2c` sensitivity row changes the winner.** `c` is a *lower bound* at both sites:
   no job script prints a timestamp before `torchrun` and Slurm writes no start line into
   `logs/%x-%j.out`, so the shell prologue, `singularity exec` of a 13.5 GB sif and
   `import torch` are all unmeasured. A chain recommendation that flips between `c` and
   `2c` is provisional until `c` is measured.
5. **The plan chains.** `chain_wait_overlap=True` assumes the successor's queue wait runs
   concurrently with its `--begin` hold. If Slurm defers eligibility until the begin time
   instead, every link costs a further `w(T)`; set it False to reprice. Also: a successor
   registering under the same cluster id while the predecessor is still `is_live` is
   refused with DARL 503 `cluster_busy` after ~13–40 s of retries. Chains rely on the
   SIGTERM release path; the timeline emits a warning per overlapping link.
6. **`p50` and `p90` disagree on the winner.** As above — this is a statement about the
   queue's variance, not about the plan.
7. **Site limits.** `MaxSubmitJobs`, `MaxArraySize` and per-account running caps are not
   readable from the aggregator VM and are not in the repo; they are `assumed` and echoed.
   Snellius' priority is ~98 % fairshare with a one-day half-life, so many submissions do
   cost queue position, which the planner does not price.
8. **Which DARL coordinator you asked.** 29510 / 29520 / 29530 / 29540 are four arms with
   four different epochs and four different answers to "how much is left". The port is
   carried into `DarlState.source` and printed. `check_darl_digest` returns an exclusion
   when the coordinator disagrees with the toml, because every site would then be refused
   at registration and the plan would describe a run that cannot start.
9. **Probes older than the plan.** The collector has no retry and no queue; a failed POST
   is lost outright, so a stale row is a silent hole, not an error.

## Closing the open calibration item

The two-site 1-device regime predicts 384.3 s against a measured 363 s, and no additive
form reproduces it: LUMI@1's predicted evaluate compute alone approaches the whole
measured evaluate segment, so either that barrier is partly a max rather than a sum, or
`validation.local_batch_size` differs at reduced geometry. `n = 2` rounds; not resolvable
from existing logs. **One two-site 1-device round, with `PWW_VAL_WINDOWS` logged and the
period differenced merge-complete to merge-complete, closes it.** Until then reduced-
geometry plans are priced and tagged, not ranked.

Three other measurements the planner is waiting on, in the order they change answers:

```bash
# 1. c, per site. First command after the #SBATCH block of each job script:
date +%s > "${PWW_ROOT}/logs/jobstart-${SLURM_JOB_ID}"
#    then c = (first [titan] log timestamp) - that value. Or, on a login node:
sacct -j <jobid> -o JobID,Start,End,Elapsed,Timelimit

# 2. throughput at reduced geometry, so 1-device shapes stop being excluded:
scripts/titan/calibrate_throughput.sh --write        # on a 1-GPU / 1-GCD job log
#    -> PWW_TPUT_SNELLIUS_1 / PWW_BATCH_SNELLIUS_1 in configs/site_throughput.env

# 3. w(T) at the walltimes we actually submit. Stand up PWW's own scanner on
#    145.38.206.143:29513 (the collector configs already point there) and run the
#    collectors as the accounts that submit. Probes run serially with a 60 s timeout,
#    so every added shape is ~1 min of a 10-minute cron cycle.
```

## Status

Written and verified end to end: `model.py`, `rounds.py`, `timeline.py`, `search.py`,
`inputs.py`, `adapter.py`, `emit.py`, `report.py`, `cli.py`, `configs/plan/federation.json`,
`scripts/plan_campaign.sh`, `scripts/titan/job_chain_link.sh`, and the `run_train.sh`
change (`DUMP="${PWW_DUMP:-}"`) that makes `PWW_DUMP` reachable. `residuals()` reproduces
the three identified regimes; the simulator finds both traps; exact search over 221
selections runs in ~1 s and re-plans byte-identically.

Tests: no pytest, no network, one file per stage; each prints its own count. Run them
rather than trusting a total written here — the last total in this paragraph was wrong
the same day it was written.

```bash
for t in tests/test_plan_*.py; do PYTHONPATH=src python3 "$t"; done
```

`test_plan_emit.py` drives the real `job_chain_link.sh` with a stub `sbatch` on PATH,
because `--export=ALL` is a property of a command line and not of any Python object.
`test_plan_report.py` pins the presence bar column by column and re-runs the worked
example above.

Known gaps, in the order they will bite:

* **The two-site 1-device overhead cell is still `extrapolated`** (+5.9% against n=2
  measured rounds). Plans dominated by it are priced and NOT ranked without
  `--assume-overhead`. See *Closing the open calibration item*.
* **`c` is a LOWER BOUND at both sites.** No job script prints a timestamp before
  torchrun, so the c-sensitivity pass (c/2, c, 2c) is not decoration — a chain
  recommendation that moves across it is provisional. Fix by adding
  `date +%s > "${PWW_ROOT}/logs/jobstart-${SLURM_JOB_ID}"` as the first command after the
  `#SBATCH` block of both job scripts.
* **Site submission limits are assumed, not read.** There is no `sbatch`, `sacct` or
  `scontrol` on the aggregator VM; `scontrol show assoc_mgr` on a login node closes it.
* **Chained links assume the successor's queue wait overlaps its `--begin` hold.** If
  Slurm defers eligibility instead, each link costs a further `w(T)`; set
  `chain_wait_overlap=False` to reprice.
* **The short end of the `w(T)` curve is unprobed at the geometries the plan picks**, so
  the chain-vs-one-long-job cross-check reports `not evaluable` rather than a number. It
  needs a second walltime probed at the SAME device count.
