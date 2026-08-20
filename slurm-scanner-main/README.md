# slurm_scanner

Reports two things per HPC cluster, and keeps them separate:

1. **When a job would start** — `sbatch --test-only` for a fixed list of job
   shapes, every 10 minutes. Slurm answers with the start time its backfill
   scheduler would actually pick.
2. **How much of the requested walltime jobs really use** — `sacct` over one
   window, once a day. On Snellius's GPU partitions this is around 8–12%: most
   reserved GPU-time is handed back early, which is why (1) reads pessimistic.

Both are shown side by side, never blended into one number — see
[What the numbers mean](#what-the-numbers-mean).

A third thing is derived from them, in one place and reversibly: **where to send
work** — [`POST /plan`](#4-planning-a-job) splits X data units across the
clusters and says how long a job to ask for at each.

```
collector/slurm_probe.py   runs on each login node. stdlib only, single file.
server/app.py              ingest + query + dashboard
server/plan.py             the waterfill behind POST /plan
server/static/index.html   the dashboard, including the planner view
plan.example.json          per-cluster throughput, copy to plan.json
tests/                     86 tests, no cluster required
```

---

## 1. Server

One server serves every cluster, and runs at **145.38.185.196**. It needs to be
reachable from the login nodes over the network, and nothing else.

**Install.** The server wants Python **3.10+** (pydantic's floor).

```bash
git clone git@gitlab.surf.nl:douwe.vanderwal/slurm-scanner.git && cd slurm-scanner
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**Mint one token per cluster**, so a site can be revoked without touching the
rest:

```bash
mkdir -p ~/.slurm_scanner && chmod 700 ~/.slurm_scanner
for site in snellius lumi frontier etp; do
  printf '%s %s\n' "$site" "$(openssl rand -hex 16)" >> ~/.slurm_scanner/tokens
done
chmod 600 ~/.slurm_scanner/tokens
cat ~/.slurm_scanner/tokens        # copy these now, they are not recoverable
```

**Write `plan.json` into the data directory.** The server will not start without
it — see [section 4](#4-planning-a-job) for what goes in it. To run ingest and
the dashboard only, say so explicitly:

```bash
mkdir -p ~/.slurm_scanner/data
cp plan.example.json ~/.slurm_scanner/data/plan.json
echo '{"clusters": []}' > ~/.slurm_scanner/data/plan.json   # or this, to plan nowhere
```

**Run it.** Everything lives in `~/.slurm_scanner`: the tokens at the top, and
under `data/` one directory per cluster plus `plan.json`.

```bash
SLURM_SCANNER_DATA_DIR=$HOME/.slurm_scanner/data \
SLURM_SCANNER_TOKENS=$(awk '{print $2}' ~/.slurm_scanner/tokens | paste -sd,) \
  .venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 8000
```

| variable | default | meaning |
|---|---|---|
| `SLURM_SCANNER_DATA_DIR` | `./data` | where the CSVs and `plan.json` live |
| `SLURM_SCANNER_TOKENS` | *(none)* | comma-separated bearer tokens |
| `SLURM_SCANNER_PLAN_CONFIG` | `$SLURM_SCANNER_DATA_DIR/plan.json` | only if config must live apart from the data |

Tokens are read once at startup, so **adding or revoking one needs a restart.**

**The token is sent in clear over HTTP.** There is no hostname or certificate on
`145.38.185.196` yet, so the collector's `Authorization` header travels
unencrypted. That is acceptable while the token only guards write access to
queue statistics, but it means anyone on the path can capture it and post
fabricated numbers. When there is a DNS name for the host, bind uvicorn to
`127.0.0.1`, terminate TLS in nginx or Caddy, and switch the clients' `server`
to `https://`.

**Check it.**

```bash
curl -s localhost:8000/healthz                       # {"ok":true,"clusters":[]}
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/ingest/probe \
     -H "Authorization: Bearer <token>" -H 'Content-Type: application/json' -d '{}'
```

`400` means success — the token was accepted and only the empty body rejected.
`401` means the token is not in the list. `503` means no tokens are configured
at all: ingest fails closed rather than collecting anonymously, since an
unconfigured server must not silently gather numbers a scheduler will act on.

---

## 2. Each cluster

Repeat per cluster, on a login node. Two files and two cron lines; nothing to
install and no dependencies — it runs on whatever `python3` the login node
already has (standard library only, no modern syntax; tested on 3.9 and 3.11).

**Copy the collector over.** It is one self-contained file, so `scp` it rather
than cloning — sites that cannot reach GitLab work the same way:

```bash
scp collector/slurm_probe.py newsite:~/slurm_probe.py
```

**See what the partitions are called** before writing the config, since the
names differ per site:

```bash
ssh newsite
sinfo -o '%P %G' | grep -i gpu
```

**Write `~/.slurm_probe/config.json`.** This is the whole configuration —
there are no command-line options and no environment variables:

```bash
mkdir -p ~/.slurm_probe && cat > ~/.slurm_probe/config.json <<'EOF'
{
  "cluster": "snellius",
  "server": "http://145.38.185.196:8000",
  "token": "<the token minted for this cluster>",

  "partitions": ["gpu_h100", "gpu_a100"],
  "usage_hours": 48,

  "shapes": [
    {"name": "h100_1gpu_1h",  "args": ["-p", "gpu_h100", "--gpus-per-node", "1", "-t", "1:00:00"]},
    {"name": "h100_full_8h",  "args": ["-p", "gpu_h100", "--gpus-per-node", "4", "-t", "8:00:00"]},
    {"name": "h100_2node_8h", "args": ["-p", "gpu_h100", "-N", "2", "--gpus-per-node", "4", "-t", "8:00:00"]},
    {"name": "a100_1gpu_1h",  "args": ["-p", "gpu_a100", "--gpus-per-node", "1", "-t", "1:00:00"]}
  ]
}
EOF
chmod 600 ~/.slurm_probe/config.json
```

The `chmod` is not optional — the collector refuses to read a token from a file
anyone else can read, and says so.

| key | |
|---|---|
| `cluster` | Set it explicitly and never change it. Every row on the server is keyed on this string, and two machines reporting the same name merge into one series silently. |
| `server` | Leave it out to have the payload printed instead of posted — that is the dry run. |
| `partitions` | Which partitions `sacct` is filtered to. List only partitions you can actually submit to; waits on a partition you cannot use are not your waits. |
| `usage_hours` | The `sacct` window. Keep it longer than your longest typical job — see [below](#what-the-numbers-mean). |
| `shapes` | The job shapes to ask about. Plain `sbatch` arguments; `--wrap true` is appended for you. Names must be unique, and are what the graphs are keyed on. |

**Dry-run it.** Copy the config without the `server` key and both commands print
their payload instead of posting:

```bash
python3 -c "import json;c=json.load(open('$HOME/.slurm_probe/config.json'));c.pop('server',None);c.pop('token',None);json.dump(c,open('/tmp/dry.json','w'))"
SLURM_PROBE_CONFIG=/tmp/dry.json python3 ~/slurm_probe.py probe
SLURM_PROBE_CONFIG=/tmp/dry.json python3 ~/slurm_probe.py usage
```

Check that every shape came back `"ok": true`, and that `usage` reports a
plausible `n_jobs` per partition. A shape with `"ok": false` carries Slurm's
own refusal in `message` — usually a wrong partition name or a walltime over
the partition limit. Fix those now; they are stored as failures otherwise.

**Then deliver once for real, and confirm the server sees it:**

```bash
python3 ~/slurm_probe.py probe && python3 ~/slurm_probe.py usage
curl -s http://145.38.185.196:8000/clusters
```

**Install the cron lines.** `probe` is cheap; `usage` is one `sacct` query
(~1 s for 10k jobs) so it runs daily, at an offset minute so sites do not all
hit the server at once:

```cron
*/10 * * * *  python3 $HOME/slurm_probe.py probe >/dev/null
17   4 * * *  python3 $HOME/slurm_probe.py usage >/dev/null
```

There is no state file and no lock: each run recomputes its own window, so a
missed run leaves no gap to catch up on. A failed post is simply lost, which is
the right trade for a 10-minute cadence.

---

## 3. Using it

Open <http://SERVER_IP:8000/> for the dashboard: an overview table of
expected start times across all clusters, a **Plan a job** panel where you type a
number of data units and get the allocation drawn on a timeline, and per-cluster
graphs of both signals over time.

The JSON API behind it, for a scheduler to consume. Query endpoints need no
token:

| endpoint | returns |
|---|---|
| `POST /plan` | how many data units to send to each cluster, and for how long — [section 4](#4-planning-a-job) |
| `GET /overview` | latest estimate per cluster × shape, with that partition's walltime ratio alongside |
| `GET /probes?cluster=&hours=` | the estimated-wait time series |
| `GET /usage?cluster=&hours=` | the walltime-usage series per partition |
| `GET /clusters` | clusters, their shapes, and freshness |
| `GET /healthz` | liveness |

```bash
curl -s http://145.38.185.196:8000/overview | jq '.rows[] | {cluster, shape, estimated_wait_sec, used_ratio}'
```

Data lands as two append-only CSVs per cluster, which are meant to be read
directly if that is easier than the API:

```
<SLURM_SCANNER_DATA_DIR>/<cluster>/probes.csv     one row per shape per probe
<SLURM_SCANNER_DATA_DIR>/<cluster>/usage.csv      one row per partition per run
<SLURM_SCANNER_DATA_DIR>/plan.json                the planner's config
```

At a 10-minute cadence with four shapes this is about 16 MB per cluster per
year. Dropping a cluster is `rm -rf` on its directory — one directory per cluster,
so only `plan.json` sits at the top level and the cluster scan ignores it.

---

## 4. Planning a job

`POST /plan` answers the only question the measurements exist to answer: *I have
X data units — where do I put them?* It splits X across the clusters and says how
long a job to ask for at each. It reads the newest probe and usage rows, writes
nothing, and needs no token.

**Configure the clusters** in `plan.json`, in the data directory beside the
per-cluster CSVs — nothing needs to be set for it to be found, and it is the only
place the scheduler's assumptions live. Separate from the collector configs,
because these are the server's numbers, not a site's:

```json
{
  "defaults": {"reserve_units": 5, "min_units": 10, "discount_strength": 0.5,
               "max_probe_age_hours": 6, "max_gpus": 0},
  "clusters": [
    {"cluster": "snellius", "units_per_gpu_per_hour": 10,   "shapes": {"h100_1gpu_8h": 1, "h100_full_8h": 4}, "max_walltime_hours": 120},
    {"cluster": "frontier", "units_per_gpu_per_hour": 22.5, "shapes": {"mi250_8gpu_24h": 8},                  "max_walltime_hours": 24, "min_units": 40}
  ]
}
```

| key | |
|---|---|
| `units_per_gpu_per_hour` | Single-GPU processing speed on this cluster. |
| `shapes` | A mapping of available shape names to the number of GPUs they allocate. The planner evaluates all shapes (or respects `max_gpus` if set) and selects the best one for each cluster based on queue wait and throughput. (An optional `"shape"` key can pin a specific shape). |
| `max_walltime_hours` | The site's queue limit. Work that does not fit inside one job goes elsewhere. |
| `min_units` | Below this a cluster is not worth a job at all, and its share moves to the clusters that stay — as a longer job there. Per cluster, falling back to `defaults`. |

The file is read **once at startup**, and is **required**: a missing or malformed
one stops the server rather than letting it plan against numbers nobody meant. A
scheduler that silently has no clusters configured is worse than one that will
not boot, so running without planning is spelled `{"clusters": []}` — deliberate,
and visible in the file. Editing it needs a restart, like the tokens.

**Ask for a plan.** Only `units` is required; every other parameter defaults from
`defaults` and can be overridden per request:

```bash
curl -s -X POST http://145.38.185.196:8000/plan \
     -H 'Content-Type: application/json' -d '{"units": 2000, "max_gpus": 8}' | jq
```

```json
{"units": 2000, "assigned_units": 2000, "unassigned_units": 0, "feasible": true,
 "horizon_hours": 100.5, "latest_finish_hours": 101.3,
 "clusters": [
   {"cluster": "snellius", "shape": "h100_full_8h", "gpus": 4, "units": 982, "walltime_hours": 98.7, "start_hours": 2.2,
    "wait_raw_sec": 14400, "wait_eff_sec": 8006, "used_ratio": 0.112, ...},
   {"cluster": "frontier", "shape": "mi250_8gpu_24h", "gpus": 8, "units": 955, "walltime_hours": 24.0, "start_hours": 21.8, ...}],
 "excluded": [{"cluster": "lumi", "reason": "probe for 'mi250x_2node_8h' is 30.0 h old, limit is 6.0 h"}]}
```

**Or ask on the dashboard.** The *Plan a job* panel posts exactly this request:
type the units, press Plan, and the answer is drawn as one bar per cluster on a
shared time axis — grey for the queue wait, colour for the processing, faint for
the reserve tail — with a dashed line at `horizon_hours`, where the data is done.
Which clusters overlap, and by how much, is the thing a table of numbers hides.
*Parameters* reveals the knobs below; a field left blank uses the config default,
so the page never restates a default that could go stale. `excluded` is listed
under the chart, and the plan is recomputed on each refresh so it cannot sit
beside estimates newer than the ones it used.

**Read `excluded` every time.** A cluster silently missing from a plan is how a
wrong answer looks right; every configured cluster is either in `clusters` or in
`excluded` with the reason — stale probe, failed probe, no probe for its shape,
share under `min_units`, or simply not needed because its queue starts after the
work is already done elsewhere.

| parameter | default | |
|---|---|---|
| `units` | *(required)* | Data units to place. A positive integer; the split is integers summing to exactly this. |
| `reserve_units` | `0` | Spare capacity every job carries, **in data units, not hours**. Each cluster converts it at its own rate — `reserve_units / units_per_hour` — so all of them can absorb the same extra work while a fast one spends fewer hours doing it. It sits past `horizon_hours` and is unused when starts land as predicted. |
| `min_units` | `0` | Default floor per cluster, overridden per cluster in the config. |
| `discount_strength` | `0.5` | How much of `used_ratio` to apply to the queue estimate — see below. |
| `max_probe_age_hours` | `6` | Older than this and a cluster is excluded rather than planned on stale numbers. |
| `max_gpus` | `0` | Upper limit on GPUs per job / shape (0 = unlimited). Shapes requiring more GPUs are ignored during planning. |

**How the split is found.** A waterfill on a completion horizon: at horizon `T` a
cluster can process `T − its expected start` hours of work, capped by
`max_walltime_hours`, at `units_per_hour`. Bisect `T` until the clusters together
cover the request. A cluster whose queue starts after `T` gets nothing, which is
what makes it a waterfill and not a proportional split — for 60 units in the
config above, Snellius takes all of them and Frontier is not used. Then any
cluster under its `min_units` is dropped and the waterfill run again, so its work
becomes a longer job elsewhere; that is skipped when the remaining clusters
cannot cover the request without it, since a small job somewhere beats leaving
data unprocessed.

**Every cluster carries the reserve, each at its own cost.** A job's walltime is
`its work + reserve_units / units_per_hour`, so `reserve_units: 10` is an hour on
a 10 units/h cluster and 100 seconds on a 360 units/h one. 
The reserve does cost capacity. `max_walltime_hours` minus the reserve is what the
waterfill may fill, so a large reserve places less work per job, and a cluster
whose limit cannot hold it is excluded with that as the reason. And note what the
cushion is worth in wall-clock terms: 100 seconds tolerates almost no slip in the
start time, so on a fast cluster carrying most of the work, size `reserve_units`
from how late you think the queue can open rather than from how many units feels
like a small number — `reserve_hours` in the response is the number to read.

**`feasible: false` is a real answer, not an error.** When the clusters cannot
hold X inside one job each, the plan fills them and reports the shortfall in
`unassigned_units`. Raise `max_walltime_hours`, add a cluster, or send less.

---

## What the numbers mean

**`--test-only` is conditioned on the probing account.** Slurm answers using
*that* account's fairshare, QOS and priority. A well-placed account sees a
shorter queue than the site average, so the estimate is "when this account's
job would start", not a universal figure. Run the collector as whichever
account the eventual jobs will use.

**The estimate is pessimistic by construction.** The backfill simulation
assumes every running and pending job occupies its full requested walltime. The
usage ratio says how wrong that assumption is in aggregate — but how much of
that freed capacity *you* get depends on what else is queued, so the two are
reported side by side and never multiplied together. On a saturated partition
newly arriving jobs absorb most of it; on an idle one almost none.

**`/plan` is the one place the two are combined, because a scheduler has to
commit to a number.** It uses the weighted average
`wait_raw × (1 − s) + wait_raw × used_ratio × s`, where `s` is
`discount_strength`: at `0` the pessimistic estimate stands, at `1` the ratio is
believed in full, and `0.5` splits the difference. Nothing above makes `0.5`
right — it is a policy choice about how much of the handed-back capacity you
expect to win, and on a saturated partition it is too generous. The response
carries `wait_raw_sec`, `used_ratio` and `wait_eff_sec` together so any plan can
be re-derived by hand, and so lowering `s` is a visible knob rather than a code
change.

**The usage window must be longer than your jobs.** Jobs are selected on *end*
time, so a window shorter than a typical job over-samples short jobs — which
are exactly the ones that hand back the most walltime, biasing the ratio low.
48 h covers most GPU partitions. `sacct` itself refuses windows over ~7 days at
most sites.

**Usage rows are independent measurements, not increments.** Each run
recomputes its window from scratch, so consecutive runs overlap and describe
overlapping job sets. Never sum them. For a longer horizon, raise
`usage_hours`; do not add rows together.

**Failed probes are kept, not dropped.** A drained partition or a request over
a limit looks identical to an outage otherwise. `ok: false` rows carry Slurm's
own message.

---

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt && .venv/bin/pytest tests -q
```

86 tests, no cluster required — `sbatch` and `sacct` output is fixed text, and
the waterfill is pure arithmetic.
