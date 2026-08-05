# Multi-Site DiLoCo Federated Training Guide (Snellius + LUMI + Central Node)

This guide provides step-by-step instructions for running distributed DiLoCo training across **Snellius** (SURF, NVIDIA GPUs), **LUMI** (EuroHPC, AMD GPUs), and a **Central Cloud Node** (`145.38.206.143`) using **DARL** for dynamic 1-epoch dataset partitioning and **Flower (`Zare2001/flower@fedmom-strategy`)** for `FedMom` outer step optimization over open ports `29510` and `29511`.

---

## 1. Architecture & Port Mapping

```
                  +-------------------------------------------------+
                  |            CENTRAL ORCHESTRATOR NODE            |
                  |            (Ubuntu 24.04 @ 145.38.206.143)     |
                  |  +-------------------+   +-------------------+  |
                  |  | DARL Coordinator  |   | Flower Server     |  |
                  |  | (HTTP Port 29510) |   | (FedMom - 29511)  |  |
                  |  +---------+---------+   +---------^---------+  |
                  +------------|-----------------------|------------+
                               | Lease Spans           | FedMom Weights
                               |                       |
                 +-------------+------------+          |
                 |                          |          |
     +-----------v-----------+  +-----------v----------++
     |    SNELLIUS CLUSTER   |  |     LUMI CLUSTER      |
     |  NVIDIA H100 (SURF)   |  |  AMD MI250X (EuroHPC) |
     |                       |  |                       |
     |  train_flower.py      |  |  train_flower.py      |
     |  (cluster-id: snellius|  |  (cluster-id: lumi)   |
     +-----------------------+  +-----------------------+
```

| Service | Protocol | Open Port | Security Group Rule |
| :--- | :--- | :--- | :--- |
| **DARL Lease Coordinator** | HTTP / REST | **`29510`** | `29510` open to LUMI (`193.167.209.128/26`) & Snellius (`145.136.63.192/32`) |
| **Flower Server (`FedMom`)** | gRPC | **`29511`** | `29511` open to LUMI (`193.167.209.128/26`) & Snellius (`145.136.63.192/32`) |

---

## 2. Central Node Setup & Execution (`145.38.206.143`)

The Central Node must be started **first** before submitting Slurm jobs on Snellius or LUMI.

### Step 1: Start Central Node Services
Log in to the central Ubuntu VM (`145.38.206.143`) and execute:

```bash
cd ~/ProjectWorldWide
./scripts/central_node/start_central_services.sh
```

> **Environment Note**: The startup script uses **`uv`** (or creates a dedicated isolated `.venv`) to install your forked Flower branch (`Zare2001/flower@fedmom-strategy`), avoiding Ubuntu 24.04 PEP 668 system-environment restrictions.

This launches both daemons in the background:
* **DARL Coordinator** on port `29510`
* **Flower Aggregator Server** (`FedMom` strategy) on port `29511`

### Step 2: Check Central Node Status
Verify both daemons are active and ports are listening:

```bash
./scripts/central_node/status_central_services.sh
```

### Step 3: Stop Central Node Services
To stop services after training completes:

```bash
./scripts/central_node/stop_central_services.sh
```

---

## 3. Snellius Cluster Execution (NVIDIA H100 / SURF)

### Step 1: Verify Connectivity from Snellius
Log in to the Snellius login node (`145.136.63.192`) and test connection to the Central Node:

```bash
curl -sS http://145.38.206.143:29510/health
# Expected output: {"status": "ok"}

nc -zv 145.38.206.143 29511
# Expected output: Connection to 145.38.206.143 29511 port [tcp/*] succeeded!
```

> **Note on Snellius Compute Nodes**: If Snellius GPU compute nodes are on a separate subnet blocked by the firewall, establish an SSH tunnel on the Snellius login node:
> ```bash
> ssh -f -N -g -L 29510:145.38.206.143:29510 -L 29511:145.38.206.143:29511 145.38.206.143
> ```

### Step 2: Submit Snellius Slurm Job
Submit the federated DiLoCo job:

```bash
cd ~/ProjectWorldWide
sbatch scripts/snellius/job_flower_diloco.sh
```

The script will automatically install the forked Flower branch (`Zare2001/flower@fedmom-strategy`) into your Snellius venv (`$HOME/venvs/pww-snellius`) if not already installed.

---

## 4. LUMI Cluster Execution (AMD MI250X / EuroHPC)

### Step 1: Verify Connectivity from LUMI
Log in to the LUMI login node and test reachability to the Central Node:

```bash
curl -sS http://145.38.206.143:29510/health
# Expected output: {"status": "ok"}

nc -zv 145.38.206.143 29511
# Expected output: Connection to 145.38.206.143 29511 port [tcp/*] succeeded!
```

### Step 2: Submit LUMI Slurm Job
Submit the federated DiLoCo job:

```bash
cd ~/ProjectWorldWide
sbatch scripts/lumi/job_flower_diloco.sh
```

The script will automatically install the forked Flower branch (`Zare2001/flower@fedmom-strategy`) inside the LUMI container environment.

---

## 5. Live Supervision & Monitoring Guide

You can monitor the whole multi-site training run live from the Central Node as well as from individual cluster logs.

### A. Monitor Flower Outer Rounds (`FedMom`)
On the **Central Node**, stream live outer round progress, connected cluster counts, and loss metrics:

```bash
tail -f runs/central/flower.log
```

*Key log indicators:*
* `Starting Flower Aggregator Server (FedMom) on 0.0.0.0:29511...`: Server ready.
* `Round 1: Aggregating outer step from 2 clusters`: Both Snellius and LUMI connected!
* `Round 1 complete: avg_cluster_loss=1.8421, pseudo_grad_norm=0.124501`: FedMom outer update applied.

---

### B. Monitor DARL Dataset Partitioning (1-Epoch Coverage)
On the **Central Node**, supervise block leases and verify exact 1-epoch data coverage:

```bash
# Stream DARL lease events live
tail -f runs/central/darl.log

# Query live JSON status summary
curl -sS http://145.38.206.143:29510/status | jq .
```

*Status output breakdown:*
* `unassigned_blocks`: Remaining dataset blocks waiting to be trained.
* `leased_blocks`: Blocks currently being processed by Snellius or LUMI.
* `committed_blocks`: Durably finished blocks.
* `clusters`: Breakdown of active leases per site (`snellius`, `lumi`).

---

### C. Monitor Cluster Slurm Execution

* **On Snellius**:
  ```bash
  tail -f logs/pww-snellius-flower-*.out
  ```
* **On LUMI**:
  ```bash
  tail -f logs/pww-lumi-flower-*.out
  ```

---

## 6. Troubleshooting & Edge Cases

| Issue / Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **Flower Server stuck waiting** | The server requires both Snellius AND LUMI to connect (`min_clients=2`). | Check `squeue` on both sites. Training starts automatically as soon as the second cluster enters `RUNNING` status. |
| **DARL Lease Expiry** | Slurm walltime expired mid-epoch on one cluster. | **Self-healing**: DARL automatically expires uncommitted leases via TTL and returns blocks to the pool so the surviving cluster completes the epoch. |
| **`Connection refused` on 29511** | Central Node services are not running. | Run `./scripts/central_node/start_central_services.sh` on the Central Node (`145.38.206.143`). |
