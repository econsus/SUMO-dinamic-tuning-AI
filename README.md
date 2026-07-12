# SUMO Dynamic Tuning with DQN

A reinforcement learning system that uses **Deep Q-Networks (DQN)** to automatically calibrate SUMO traffic simulation parameters so that simulated traffic matches real-world CCTV observations at the Suhat intersection in Malang.

Two parallel training environments are available:

| Env | Class | Parameter tuned | Actions |
|-----|-------|----------------|---------|
| **LCSumoEnv** | `src/envs/lc_env.py` | `lcCooperative`, `lcAssertive` | 9 (3×3 grid of ±0.2 deltas) |
| **SFSumoEnv** | `src/envs/sf_env.py` | `speedFactor` | 5 (deltas -0.2 to +0.2) |

The `lcCooperative`/`lcAssertive` parameters were found to have **zero effect** on north/south vehicle distribution at this intersection geometry. These files are kept for reference. The `speedFactor` environment is the actively developed version.

---

## Project Structure

```
root/
├── CCTV Data Remastered.xlsx            ← Source CCTV traffic data
├── README.md
├── DEVELOPMENT_GUIDE.md
├── testing/                             ← Diagnostic & legacy scripts
│   ├── stepA_counting_sanity.py
│   ├── stepB_warmup_contamination.py
│   ├── stepC_route_override_check.py
│   ├── stepD_vehicle_loop_mapping.py
│   ├── stepE_param_propagation_test.py
│   ├── main.py, main copy.py, SumoSimulation.py, run_recorder.py
│   └── logs/
├── data/sumo_files/                     ← SUMO network & config files
│   ├── map_suhat_edit.net.xml
│   ├── map_suhat_netedit.rou.xml
│   ├── induction_loop.xml
│   └── map_suhat_sumoconfig.sumocfg
├── src/
│   ├── logger.py                        ← TrainingLogger (generic params dict + action logging)
│   ├── data_repo.py                     ← File indexer singleton
│   ├── data/
│   │   ├── __init__.py
│   │   └── traffic_data.py              ← Excel parser
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── lc_env.py                    ← LCSumoEnv (lcCooperative/lcAssertive)
│   │   └── sf_env.py                    ← SFSumoEnv (speedFactor)
│   ├── agents/
│   │   ├── __init__.py
│   │   └── dqn_agent.py                 ← DQNAgent + QNetwork + ReplayBuffer
│   └── train/
│       ├── __init__.py
│       ├── train_lc.py                  ← Entry point for LCSumoEnv
│       └── train_sf.py                  ← Entry point for SFSumoEnv
└── logs/                                ← Created at runtime
    └── YYYY-MM-DD_HH-MM-SS/
        ├── config.json
        ├── metrics.csv
        ├── actions.csv
        ├── summary.json
        └── dqn_final.pt
```

---

## High-Level Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  1. DATA EXTRACTION                                              │
│     Parse CCTV Excel → extract SL, SPT, UL, UPT, OS, OU         │
│     Skip rows where OS=0 or OU=0                                │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────┴───────────────────────────────────────┐
│  2. ENVIRONMENT SETUP (per data point)                           │
│     Generate temp route XML with current SL/SPT/UL/UPT           │
│     Start SUMO → warm-up (no RL, remove warm-up vehicles)       │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────┴───────────────────────────────────────┐
│  3. AGENT DECISION (×4 per data point)                           │
│     Observe state [SL, SPT, UL, UPT] → pick action (ε-greedy)   │
│     Apply delta to current params (clamped to valid range)       │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────┴───────────────────────────────────────┐
│  4. SIMULATION & REWARD                                          │
│     Run period-minutes of SUMO with updated params               │
│     Read induction loop counts (north / south)                   │
│     Reward = -MAPE × reward_scale                                │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────┴───────────────────────────────────────┐
│  5. LEARNING & LOGGING                                           │
│     Store (s, a, r, s', done) in replay buffer                   │
│     Sample batch → Double DQN update → soft target update        │
│     Log step metrics + actions to CSV, iteration summary to JSON │
└──────────────────────────────────────────────────────────────────┘
```

### Two parameter flows (`--flow`)

| Flow | Flag | Behavior |
|------|------|----------|
| **A (reset)** | `--flow reset` | Params reset to default (1.0) at each `reset()`. Each data point starts fresh. |
| **B (persist)** | `--flow persist` (default) | Params carry across data points and iterations. The agent continuously fine-tunes. |

---

## Detailed Step Breakdown

### 1. Data Extraction

Same as before. The source data lives in **`CCTV Data Remastered.xlsx`**. Columns extracted from cells `I3:N30`:

| Key  | Meaning                              | Source                 |
|------|--------------------------------------|------------------------|
| SL   | South Lurus (straight) flow rate     | CCTV, hourly vehicles  |
| SPT  | South Putar Balik (U-turn) flow rate | CCTV, hourly vehicles  |
| UL   | Utara Lurus (straight) flow rate     | CCTV, hourly vehicles  |
| UPT  | Utara Putar Balik (U-turn) flow rate | CCTV, hourly vehicles  |
| OS   | Observed Southbound total            | CCTV, hourly vehicles  |
| OU   | Observed Northbound total            | CCTV, hourly vehicles  |

**Filtering**: rows where either `OS=0` or `OU=0` are skipped. Rows with zero input flows are kept.

### 2. Environment Setup (per data point)

Each `env.reset()` call:

1. **Stop prior SUMO** — force-kills any running `sumo.exe` / `sumo-gui.exe` via `taskkill /F /T` and PowerShell fallbacks.
2. **(Flow A only)** Reset params to defaults.
3. **Generate temp route XML** — injects the current SL/SPT/UL/UPT as `vehsPerHour` into 4 traffic flows, with the current parameter values as XML attributes on `<vType>`.
4. **Start SUMO** — launches SUMO with the project config, overriding route files with the temp XML.
5. **Warm-up** — runs `warmup-minutes` of simulation, then removes all vehicles. This lets traffic stabilise without polluting the RL reward window.

The 4 traffic flows and their mapping to induction loops:

| Flow ID | From edge          | To edge            | Loop contribution |
|---------|--------------------|--------------------|-------------------|
| f_sl    | `675098743#0`      | `675098743#2`      | northbound        |
| f_spt   | `675098743#0`      | `585303113#2`      | southbound        |
| f_ul    | `585303113#0`      | `585303113#2`      | southbound        |
| f_upt   | `585303113#0`      | `675098743#2`      | northbound        |

### 3. Agent Decision & Action Maps

At each RL step, the agent receives state `[SL, SPT, UL, UPT]` and selects an action.

**LCSumoEnv** (9 actions, parameter range `[0.0, 5.0]`):

| Action | Δ lcCooperative | Δ lcAssertive |
|--------|-----------------|---------------|
| 0      | -0.2            | -0.2          |
| 1      | -0.2            |  0.0          |
| 2      | -0.2            |  0.2          |
| 3      |  0.0            | -0.2          |
| 4      |  0.0            |  0.0          |
| 5      |  0.0            |  0.2          |
| 6      |  0.2            | -0.2          |
| 7      |  0.2            |  0.0          |
| 8      |  0.2            |  0.2          |

**SFSumoEnv** (5 actions, parameter range `[0.0, 2.0]`):

| Action | Δ speedFactor |
|--------|---------------|
| 0      | -0.2          |
| 1      | -0.1          |
| 2      |  0.0          |
| 3      | +0.1          |
| 4      | +0.2          |

Exploration uses **ε-greedy** with decay (`ε ← ε × epsilon_decay`, clamped at `epsilon_min`).

### 4. Simulation & Reward

SUMO runs for `period-minutes` (default 5 min). Vehicle counts are collected from 4 induction loops:

| Loop ID              | Lane               | Measures                |
|----------------------|--------------------|-------------------------|
| `total_north_left`   | `675098743#2_1`    | Northbound vehicles     |
| `total_north_right`  | `675098743#2_0`    | Northbound vehicles     |
| `total_south_left`   | `585303113#2_1`    | Southbound vehicles     |
| `total_south_right`  | `585303113#2_0`    | Southbound vehicles     |

```
sim_north = total_north_left + total_north_right
sim_south = total_south_left + total_south_right
```

The ground-truth observations `OS` and `OU` are hourly rates. They are divided by `(60 / period_minutes)` to get the window-expected count:

```
divisor = 60 / period_minutes      # 5 min → 12, 10 min → 6, etc.
expected_south = OS / divisor
expected_north = OU / divisor
```

Reward is the negative MAPE, optionally scaled:

```
mape = (|sim_south - expected_south| / expected_south +
        |sim_north - expected_north| / expected_north) / 2
reward = -mape × reward_scale
```

`reward_scale` (default 1.0) amplifies the reward magnitude for better log readability and stronger gradients — use `--reward-scale 10` to see clearer differences.

### 5. Learning & Logging

Each transition `(state, action, reward, next_state, done)` is pushed into a **ReplayBuffer** (circular deque, default capacity 10 000).

After each step, the agent replays **N batches** (`--replay-per-step`, default 5):

1. Sample batch from replay buffer
2. Compute current Q-values for the taken actions
3. Compute target Q-values using **Double DQN**
4. MSE loss → Adam optimizer step
5. **Soft update** target network every 100 steps: `θ_target ← τ·θ_online + (1-τ)·θ_target` with `τ = 0.005`

---

### Understanding the metrics: total, avg, and loss

| Term | What it is | What it tells you |
|------|-----------|-------------------|
| **Loss** | The neural network's **training error** — MSE between predicted Q-value and target Q-value (Double DQN). | Lower = the Q-network is learning to predict action values. Decreasing over iterations = converging. |
| **Total reward** | Sum of all step rewards across an entire iteration. Always ≤ 0 (MAPE is non-negative). | Less negative = better overall accuracy. Only comparable within the same run. |
| **Avg reward** | `total_reward / total_steps` — per-step mean. | Trending toward 0.0 = agent improving. Lower bound depends on data consistency. |

With `--reward-scale N`, all reward values are multiplied by N, making differences easier to spot in logs.

---

### Log files

Each run creates `logs/YYYY-MM-DD_HH-MM-SS/` with:

#### `config.json`

All CLI arguments saved at start. Example:
```json
{
  "iterations": 5,
  "sumo_binary": "sumo",
  "warmup_minutes": 5,
  "period_minutes": 5,
  "reward_scale": 10.0,
  "flow": "persist",
  "lr": 0.001,
  "gamma": 0.99,
  "batch_size": 32,
  "buffer_capacity": 10000,
  "replay_per_step": 5,
  "epsilon_start": 1.0,
  "epsilon_min": 0.1,
  "epsilon_decay": 0.7
}
```

#### `metrics.csv`

One row per RL step. Columns depend on which env is used — param columns are dynamic:

| Column | Example | Description |
|--------|---------|-------------|
| `iteration` | `1` | Outer training iteration (1‑indexed) |
| `data_point` | `0` | Index into the CCTV data array |
| `step_in_data` | `1` | Which of the 4 RL steps within this data point |
| `reward` | `-2.500` | Negative MAPE × reward_scale |
| `sim_south` | `37` | Simulated southbound vehicles in this window |
| `sim_north` | `41` | Simulated northbound vehicles in this window |
| `expected_south` | `44` | Expected southbound count (`OS / divisor`) |
| `expected_north` | `36` | Expected northbound count (`OU / divisor`) |
| `error_south` | `-7.000` | `sim_south − expected_south` |
| `error_north` | `5.000` | `sim_north − expected_north` |
| `speedFactor` (or `lcCooperative`, `lcAssertive`) | `1.200` | Parameter value used during this step |
| `epsilon` | `0.7000` | Exploration rate |
| `loss` | `0.042` | NN loss from last replay batch (blank until buffer fills) |

#### `actions.csv`

One row per RL step, logging which action was chosen and its delta:

| Column | Example | Description |
|--------|---------|-------------|
| `iteration` | `1` | Outer training iteration |
| `data_point` | `0` | Data point index |
| `step_in_data` | `1` | Step within data point |
| `action_index` | `3` | Which action (0-indexed) the agent chose |
| `delta_speedFactor` (or `delta_lcCooperative`, `delta_lcAssertive`) | `+0.100` | The delta applied to each parameter |

#### `summary.json`

Per-iteration aggregated metrics:
```json
{
  "iteration": 1,
  "total_reward": -228.5,
  "avg_reward": -28.562,
  "avg_loss": 0.038,
  "epsilon": 0.7,
  "total_steps": 12
}
```

#### `dqn_final.pt`

PyTorch checkpoint with `q_net_state_dict`, `target_net_state_dict`, `optimizer_state_dict`.

---

## Running

### LCSumoEnv (lcCooperative/lcAssertive — legacy)

```bash
python -m src.train.train_lc --iterations 5 --flow persist
python -m src.train.train_lc --iterations 5 --flow reset
```

### SFSumoEnv (speedFactor — active)

```bash
# Default parameters
python -m src.train.train_sf --iterations 5

# With reward scaling and GUI
python -m src.train.train_sf --iterations 10 --reward-scale 10 --sumo-binary sumo-gui

# Reset params per data point
python -m src.train.train_sf --iterations 5 --flow reset
```

### Common CLI arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | 5 | Number of outer training iterations |
| `--sumo-binary` | `sumo` | SUMO executable (`sumo` or `sumo-gui`) |
| `--warmup-minutes` | 5 | Warm-up duration in minutes |
| `--period-minutes` | 5 | RL step / reward window in minutes |
| `--reward-scale` | 1.0 | Multiplier for reward (mape × scale) |
| `--flow` | `persist` | `persist` (Flow B) or `reset` (Flow A) |
| `--lr` | 0.001 | Adam learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--batch-size` | 32 | Replay batch size |
| `--buffer-capacity` | 10000 | Replay buffer size |
| `--replay-per-step` | 5 | Replay batches per step |
| `--epsilon-start` | 1.0 | Initial exploration rate |
| `--epsilon-min` | 0.1 | Minimum exploration rate |
| `--epsilon-decay` | 0.7 | Epsilon decay per iteration |

---

## Script Reference

### Environments

| Script | Class | Params | Actions |
|--------|-------|--------|---------|
| `src/envs/lc_env.py` | `LCSumoEnv` | `lcCooperative`, `lcAssertive` | 9 (3×3 grid, range [0, 5]) |
| `src/envs/sf_env.py` | `SFSumoEnv` | `speedFactor` | 5 (deltas -0.2..+0.2, range [0, 2]) |

Both environments share the same interface:
- `reset()` → `(obs, info)` — stop SUMO, optionally reset params (Flow A), prepare next data point
- `step(action)` → `(obs, reward, terminated, truncated, info)` — apply delta, run SUMO window, compute reward
- `set_target_data(data)` — inject `(N, 6)` array
- `reset_data_pointer()` — restart data point index

### Training

| Script | Description |
|--------|-------------|
| `src/train/train_lc.py` | Entry point for LCSumoEnv. See CLI table above. |
| `src/train/train_sf.py` | Entry point for SFSumoEnv. Same CLI interface. |

### Agents

| Script | Description |
|--------|-------------|
| `src/agents/dqn_agent.py` | `DQNAgent` with ε-greedy, Double DQN, replay buffer, soft target updates. `QNetwork`: 4→64→64→N MLP. `ReplayBuffer`: fixed-capacity deque. |

### Data & Utilities

| Script | Description |
|--------|-------------|
| `src/data_repo.py` | Singleton file indexer for `data/sumo_files/`. |
| `src/data/traffic_data.py` | Excel parser for `CCTV Data Remastered.xlsx`. |
| `src/logger.py` | `TrainingLogger` — creates timestamped run dir, writes `metrics.csv`, `actions.csv`, `summary.json`, `config.json`. Generic param columns (any param name works). |
