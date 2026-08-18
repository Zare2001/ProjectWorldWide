# Adding a cluster

Bringing a new HPC site into the federation, from a fresh clone to its first merged
round. A site is one file — `sites/<name>.sh` — plus a detection branch and a job
script. No application code changes.

Steps 1–6 run entirely on your own machine. Step 7 is the first time you talk to the
central node: stop there and ping us, because a training campaign may be live and its
site membership is part of the measurement.

**Ask us for four things before step 7:** the DARL token, the ports of a test stack,
the reference `manifest.json`, and access to copy the tokenised corpus.

---

## 1. Survey

```bash
git clone <repo-url> ~/ProjectWorldWide
cd ~/ProjectWorldWide
./scripts/siteinfo.sh
```

Read the values off the machine. Two that are easy to get wrong:

- **Ranks per node.** GCDs on AMD (LUMI's MI250X node = 4 cards = 8 GCDs = 8 ranks),
  cards on NVIDIA (Snellius H100 node = 4).
- **Which scratch survives.** Snellius' `/scratch-shared` purges on file age.

## 2. Write `sites/<name>.sh`

Copy [`sites/lumi.sh`](sites/lumi.sh) (container/ROCm) or
[`sites/snellius.sh`](sites/snellius.sh) (module+venv/CUDA) and replace the values.

| | |
|---|---|
| `PWW_ACCOUNT` | accounting project for `sbatch` |
| `PWW_SCRATCH` | large, writable, visible from compute nodes |
| `PWW_GPUS_PER_NODE` | ranks per full node |
| `PWW_CPUS_PER_TASK` | cores per rank |
| `PWW_ACCELERATOR` | `rocm` or `cuda` |
| `PWW_GPU_VISIBLE_VAR` | `ROCR_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` |
| `PWW_LAUNCH` | bash array prefixing every python call |
| `pww_cpu_bind()` | echoes the `--cpu-bind` value |
| `pww_titan_env()` | echoes `kind<TAB>path<TAB>build-command` |

`PWW_DATA_DIR`, `PWW_OUTPUT_DIR`, `PWW_TMPDIR`, `PWW_CACHE_DIR` derive from
`PWW_SCRATCH` in `env.sh`.

```bash
PWW_LAUNCH=(singularity exec "${PWW_CONTAINER}")   # container site
PWW_LAUNCH=()                                      # venv site, python already on PATH

pww_cpu_bind() { echo "cores"; }                   # fine unless you need an explicit mask

pww_titan_env() {
    printf 'container\t%s\tsbatch scripts/<name>/build_titan_container.sh\n' \
        "${PWW_SCRATCH}/containers/pww-titan.sif"
}
```

Prefer a maintained site container if one exists — you inherit a tested RCCL/NCCL
stack. Don't mix: a venv built against the host python then used inside a container
resolves the wrong libraries.

## 3. Add the detection branch

In `env.sh`, `pww_detect_site()` — match a facility-specific path rather than a
hostname, since login and compute nodes differ:

```bash
elif [[ -d /some/path/unique/to/your/site ]]; then
    echo <name>
```

Check it:

```bash
source env.sh && pww_summary
```

## 4. Deploy

```bash
./scripts/deploy.sh
```

Pulls, inits submodules, creates the scratch dirs, and points `runs/` and `data/` at
them.

## 5. Build the torch >= 2.9 environment

Whatever `pww_titan_env` named. A container build is a 30–60 minute batch job:

```bash
sbatch scripts/<name>/build_titan_container.sh     # or ./scripts/titan/setup_venv_<name>.sh
```

## 6. Test, then a single-site run

```bash
python3 tests/test_darl.py
python3 tests/test_titan.py
python3 tests/test_federation.py
```

Run them a few times, not once — a couple of the failures they catch are
timing-dependent.

Then train locally with no federation, which exercises the whole stack:

```bash
scripts/titan/run_train.sh --config configs/titan/qwen3_0.6b_smoke.toml -- \
    --flower.enable false --training.steps 50
```

## 7. Corpus

**Do not re-tokenise.** The partitioning is keyed to an exact window count and
tokenizer hash; a locally regenerated corpus differs and is refused at registration.
Copy from an existing site:

```bash
rsync -a --info=progress2 <peer>:/path/to/data/c4-tokenizer-128k-2048/ "$PWW_DATA_DIR/c4-tokenizer-128k-2048/"
rsync -a <peer>:/path/to/data/tokenizers/tokenizer-128k/ "$PWW_DATA_DIR/tokenizers/tokenizer-128k/"
rsync -a <peer>:/path/to/data/c4-validation/ "$PWW_DATA_DIR/c4-validation/"
```

Then diff your manifest against the reference we send:

```bash
python3 -c "
import json; m=json.load(open('$PWW_DATA_DIR/c4-tokenizer-128k-2048/manifest.json'))
print('num_windows', m['num_windows'])
print('seq_len    ', m['seq_len'])
print('vocab      ', m['vocab_size'])
print('tok sha256 ', m['tokenizer']['sha256'])"
```

All four must match exactly.

## 8. Job script

Create `scripts/<name>/` with just what you need:

| file | |
|---|---|
| `job_titan_diloco.sh` | required — the DiLoCo site job |
| `build_titan_container.sh` or `setup_venv_<name>.sh` | required — one of the two |
| `job_smoke.sh` | recommended |

Copy an existing `job_titan_diloco.sh` and adjust the `#SBATCH` header from your
survey. One task per node; let `torchrun` fork the ranks — do **not** use
`--ntasks-per-node=<gpus>`, torchtitan owns the process topology.

## 9. First federated run

```bash
curl -s -m 5 http://<central-ip>:<darl-port>/health      # {"ok": true, ...}
```

Sites only dial out; nothing needs opening on your side.

```bash
DARL_TOKEN=<token> sbatch -J pww-<name>-titan-diloco --time=40:00:00 \
  --export=ALL,DARL_TOKEN,PWW_DARL_PORT=<port>,PWW_FLOWER_PORT=<port>,PWW_FRESH_RUN=1,ENABLE_WANDB=1,WANDB_PROJECT=<project>,WANDB_RUN_NAME=diloco-<name> \
  scripts/<name>/job_titan_diloco.sh
```

First-minute signals:

- `darl: corpus ... | space digest <hex>` — must equal the other sites' digest
- your site appears in the aggregator's `N cluster(s) [...]` line
- on a fresh federation, round-1 loss ≈ 9.9

## 10. When registration is refused

Each of these is a guard doing its job. Send us the message rather than working
around it.

| message | cause |
|---|---|
| space digest mismatch | different `num_samples`, `block_size` or `space_seed` |
| tokenizer sha256 mismatch | tokenizer directory is not byte-identical |
| `num_windows` mismatch | corpus was regenerated locally instead of copied |
| 401 | wrong or stale DARL token |
| transport mismatch | `flower.transport` differs from the server's |
