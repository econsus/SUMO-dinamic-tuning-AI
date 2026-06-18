# SUMO Dynamic Tuning with DQN

A reinforcement learning system that uses **Deep Q-Networks (DQN)** to automatically calibrate SUMO traffic simulation parameters so that simulated traffic matches real-world CCTV observations at an intersection in Malang.

The agent learns to adjust two lane-changing parameters (`lcCooperative`, `lcAssertive`) by observing the traffic state (flow rates from 4 directions), running short simulation windows, and receiving a reward based on how closely the simulated vehicle counts match the ground-truth CCTV data.

---

## High-Level Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  1. DATA EXTRACTION                                              │
│     Parse CCTV Excel → extract SL, SPT, UL, UPT, OS, OU         │
│     Filter out all-zero rows                                     │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────┴───────────────────────────────────────┐
│  2. ENVIRONMENT SETUP (per data point)                           │
│     Generate temp route XML with current SL/SPT/UL/UPT           │
│     Start SUMO → warm-up (no RL, default params)                │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────┴───────────────────────────────────────┐
│  3. AGENT DECISION (×4 per data point)                           │
│     Observe state [SL, SPT, UL, UPT] → pick action (ε-greedy)   │
│     Apply Δ(lcCooperative, lcAssertive)                          │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────┴───────────────────────────────────────┐
│  4. SIMULATION & REWARD                                          │
│     Run 5 min SUMO with updated params                           │
│     Read induction loop counts (north / south)                   │
│     Reward = -MSE(simulated vs expected OS/12, OU/12)            │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────┴───────────────────────────────────────┐
│  5. LEARNING & LOGGING                                           │
│     Store (s, a, r, s', done) in replay buffer                   │
│     Sample batch → Double DQN update → soft target update        │
│     Log step metrics to CSV, iteration summary to JSON           │
└──────────────────────────────────────────────────────────────────┘
```

The outer loop runs **N iterations**. Each iteration loops through **all available data points**. Each data point yields **4 RL steps** (4 × `period-minutes` windows) after `warmup-minutes` of warm-up.

---

## Detailed Step Breakdown

### 1. Data Extraction

The source data lives in **`CCTV Data Remastered.xlsx`**, a spreadsheet containing CCTV vehicle counts at the Suhat intersection.

**Columns extracted** (cells `I3:N30`):

| Key  | Meaning                              | Source                 |
|------|--------------------------------------|------------------------|
| SL   | South Lurus (straight) flow rate     | CCTV, hourly vehicles  |
| SPT  | South Putar Balik (U-turn) flow rate | CCTV, hourly vehicles  |
| UL   | Utara Lurus (straight) flow rate     | CCTV, hourly vehicles  |
| UPT  | Utara Putar Balik (U-turn) flow rate | CCTV, hourly vehicles  |
| OS   | Observed Southbound total            | CCTV, hourly vehicles  |
| OU   | Observed Northbound total            | CCTV, hourly vehicles  |

Internally, the Excel cells use formulas (e.g., `=C3*6`) to multiply raw 10-minute counts by 6 to get hourly rates. The `openpyxl` library reads these with `data_only=True`, so only the last-saved cached values are available.

**Filtering**: rows where ALL 6 fields are zero are skipped entirely. Rows with non-zero flows but `OS=OU=0` are kept, but a warning is printed since those produce purely negative rewards (any simulated vehicle is penalised).

**Output**: a NumPy array of shape `(N, 6)` — the first 4 columns are the state (`SL, SPT, UL, UPT`), the last 2 are the ground truth for reward (`OS, OU`).

### 2. Environment Setup (per data point)

For each data point, `SUMOEnv.reset()` is called:

1. **Kill prior SUMO process** — force-kills any running `sumo.exe` / `sumo-gui.exe` via `taskkill /F /T` and PowerShell fallbacks. Resets TraCI internal state.
2. **Generate temp route XML** — injects the current SL/SPT/UL/UPT as `vehsPerHour` into 4 traffic flows (`f_sl`, `f_spt`, `f_ul`, `f_upt`), written to a temporary directory.
3. **Start SUMO** — launches `sumo` (or `sumo-gui`) with the project config file, overriding route files with the temp XML. Step length is 0.05 s.
4. **Warm-up** — runs 5 minutes of simulation with `lcCooperative=1.0, lcAssertive=1.0` (defaults). No RL interaction. This lets traffic stabilise on the network before the first RL step.

The 4 traffic flow routes map onto the road network:

| Flow ID | From edge          | To edge            | Direction           |
|---------|--------------------|--------------------|----------------------|
| f_sl    | `675098743#0`      | `675098743#2`      | South going straight  |
| f_spt   | `675098743#0`      | `585303113#2`      | South U-turning north |
| f_ul    | `585303113#0`      | `585303113#2`      | North going straight  |
| f_upt   | `585303113#0`      | `675098743#2`      | North U-turning south |

### 3. Agent Decision

At each RL step, the agent receives a **state vector** `[SL, SPT, UL, UPT]` (the same flow rates that were fed to SUMO — representing the current traffic demand).

The agent selects 1 of **9 discrete actions**, each specifying a delta to apply to `lcCooperative` and `lcAssertive`:

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

The agent explores using **ε-greedy**: with probability ε it picks a random action; otherwise it picks the action with the highest Q-value. ε decays each iteration (`ε ← ε × 0.7`, clamped at 0.1 minimum).

The parameter values are clamped to `[0.0, 5.0]` and applied to **all currently alive vehicles** plus the `DEFAULT_VEHTYPE` (so newly spawned vehicles inherit the same params).

### 4. Simulation & Reward

After adjusting the parameters, SUMO runs for **5 minutes** (6 000 steps at 0.05 s/step). The environment accumulates vehicle counts from 4 induction loops:

| Loop ID              | Lane               | Measures                |
|----------------------|---------------------|------------------------|
| `total_north_left`   | `675098743#2_1`     | Northbound vehicles     |
| `total_north_right`  | `675098743#2_0`     | Northbound vehicles     |
| `total_south_left`   | `585303113#2_1`     | Southbound vehicles     |
| `total_south_right`  | `585303113#2_0`     | Southbound vehicles     |

- **`sim_north`** = `total_north_left + total_north_right` (all vehicles going north)
- **`sim_south`** = `total_south_left + total_south_right` (all vehicles going south)

The ground-truth observations `OS` and `OU` are **hourly rates** (vehicles per hour). Since each RL step is a 5-minute window, they are divided by 12 to get the expected 5-minute count:

```
expected_south = OS / 12
expected_north = OU / 12
```

The reward is the negative Mean Squared Error between simulated and expected counts:

```
reward = -((sim_south - expected_south)² + (sim_north - expected_north)²) / 2
```

This rewards the agent when both directions simultaneously match the observed data.

After 4 such steps (4 × `period-minutes` of RL), the episode terminates and moves to the next data point.

### 5. Learning & Logging

Each transition `(state, action, reward, next_state, done)` is pushed into a **ReplayBuffer** (circular deque, default capacity 10 000).

After each step, the agent replays **5 random batches** of 32 transitions each:

1. Sample batch from replay buffer
2. Compute current Q-values for the taken actions
3. Compute target Q-values using **Double DQN**: action selection from the online network, value evaluation from the target network
4. MSE loss → Adam optimizer step
5. **Soft update** the target network every 100 steps: `θ_target ← τ·θ_online + (1-τ)·θ_target` with `τ = 0.005`

**Logging output** — each run creates a timestamped folder `logs/YYYY-MM-DD_HH-MM-SS/` containing:

---

#### `config.json`

All CLI arguments and hyperparameters saved at the start of the run. Example:

```json
{
  "iterations": 5,
  "sumo_binary": "sumo",
  "warmup_minutes": 5,
  "period_minutes": 5,
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

Use this to reproduce or compare runs.

---

#### `metrics.csv`

One row per RL step — the finest-grained record of the training process.

| Column | Example | Description |
|--------|---------|-------------|
| `iteration` | `1` | Outer training iteration (1‑indexed) |
| `data_point` | `0` | Index into the CCTV data array being simulated |
| `step_in_data` | `1` | Which of the 4 RL steps within this data point |
| `reward` | `-142.500` | Negative MSE against expected counts. `-((sim_south − OS/12)² + (sim_north − OU/12)²) / 2`. Lower magnitude = better match |
| `sim_south` | `37` | Simulated southbound vehicles counted in this `period-minutes` window (sum of both southbound induction loops) |
| `sim_north` | `41` | Simulated northbound vehicles in this window (sum of both northbound induction loops) |
| `expected_south` | `44.000` | Ground-truth expected southbound count in this window. Equals `OS / 12` since OS is an hourly rate and each window is `period-minutes` long |
| `expected_north` | `35.500` | Ground-truth expected northbound count. Equals `OU / 12` |
| `error_south` | `-7.000` | `sim_south − expected_south`. Negative = undershoot (too few vehicles simulated) |
| `error_north` | `5.500` | `sim_north − expected_north`. Positive = overshoot |
| `lcCooperative` | `0.800` | The agent's `lcCooperative` value used during this step (after applying the action delta) |
| `lcAssertive` | `1.200` | The agent's `lcAssertive` value used during this step |
| `epsilon` | `0.7000` | Exploration rate at this step. Higher = more random actions |
| `loss` | `0.042000` | Neural network loss from the last replay batch. Empty (blank) when the replay buffer hasn't yet reached `batch-size` samples |

---

#### `summary.json`

An array of per-iteration summary objects, one per outer iteration.

| Key | Example | Description |
|-----|---------|-------------|
| `iteration` | `1` | Iteration number (1‑indexed) |
| `total_reward` | `-2280.500` | Sum of all step rewards across all data points in this iteration |
| `avg_reward` | `-285.062` | Mean reward per step = `total_reward / (N_data × 4)` |
| `avg_loss` | `0.038` | Mean replay loss across all steps in this iteration (or `null` if no training occurred) |
| `epsilon` | `0.700` | Epsilon value at the *end* of this iteration (after decay has been applied for the next iteration) |
| `total_steps` | `8` | Total number of RL steps taken this iteration (= `N_data × 4`) |

---

#### `dqn_final.pt`

PyTorch model checkpoint saved after all iterations complete. Contains a dictionary with 3 keys:

- **`q_net_state_dict`** — Weights of the online Q-network. Load this for inference or further training.
- **`target_net_state_dict`** — Weights of the target network (used during Double DQN training for stable Q-targets).
- **`optimizer_state_dict`** — Adam optimizer state. Useful if you want to resume training from this checkpoint.

Load with:

```python
checkpoint = torch.load("dqn_final.pt", map_location="cpu", weights_only=True)
agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
```

---

## Script Documentation

### Training & Core

| Script | Description |
|--------|-------------|
| `src/train_dqn.py` | Entry point. Parses CLI args, extracts target data from Excel, creates the SUMO environment, DQN agent, and logger, then runs training iterations. |
| `src/envs/sumo_env.py` | Custom Gymnasium environment. Manages SUMO process lifecycle (start, force-kill, temp route files), runs configurable warm-up and RL steps, reads induction loops, computes reward. |
| `src/agents/dqn_agent.py` | DQN agent implementation. Contains `QNetwork` (4→64→64→9 MLP), `ReplayBuffer` (deque-based), and `DQNAgent` with ε-greedy action selection, Double DQN training loop, soft target updates, and save/load. |
| `src/logger.py` | `TrainingLogger` that creates timestamped run folders, writes per-step metrics to CSV and per-iteration summaries to JSON, and saves the final model checkpoint. |

### Data

| Script | Description |
|--------|-------------|
| `data_repo.py` | Singleton that scans the project directory for all data/config files. Provides typed accessors for SUMO config path, network path, route path, and induction loop path. Can parse XML configs, routes, edges, and induction loops into dictionaries. |
| `src/scripts/data/traffic_data.py` | Opens `CCTV Data Remastered.xlsx` with `openpyxl` (data-only mode), reads cells `I3:N30`, and populates the global `data_records` list with dicts mapping `{SL, SPT, UL, UPT, OS, OU}` to integer values. |

### Package Init

| Script | Description |
|--------|-------------|
| `src/__init__.py` | Marks `src/` as a Python package (empty). |
| `src/envs/__init__.py` | Marks `src/envs/` as a Python package (empty). |
| `src/agents/__init__.py` | Marks `src/agents/` as a Python package (empty). |

---

## Detailed Script Explanations

### `src/train_dqn.py`

The main training orchestrator. Called from the project root as:

```
python src/train_dqn.py --iterations 5 --sumo-binary sumo
```

**Flow**:
1. Adds project root and `src/` to `sys.path`, then loads `SUMO_HOME` for TraCI.
2. Calls `extract_target_data()` which invokes `import_records()` from `traffic_data.py`, iterates over `data_records`, builds a NumPy array of `[SL, SPT, UL, UPT, OS, OU]`, skipping fully-zero rows and warning on zero-observation rows.
3. Instantiates `DataRepo` (singleton) to locate the SUMO config file path.
4. Creates `SUMOEnv` with the config path and chosen SUMO binary.
5. Injects target data into the env, creates `DQNAgent` (state_dim=4, action_dim=9), and opens a `TrainingLogger`.
6. For each iteration:
   - Resets the data pointer, loops over all data points.
   - For each data point: `env.reset()` → 4× `agent.act()` → `env.step()` → `agent.remember()` → `agent.replay()` (×5 per step).
   - Logs every step to CSV, logs iteration summary to JSON.
   - Decays ε.
7. Saves the final model to `dqn_final.pt` and closes the environment.

**CLI Arguments**:

| Argument           | Default | Description                        |
|--------------------|---------|-----------------------------------|
| `--iterations`     | 5       | Number of outer training iterations |
| `--sumo-binary`    | `sumo`  | SUMO executable (`sumo` or `sumo-gui`) |
| `--warmup-minutes` | 5       | SUMO warm-up duration (default params, no RL) in minutes |
| `--period-minutes` | 5       | Duration of each RL step / reward window in minutes |
| `--lr`             | 0.001   | Learning rate for Adam            |
| `--gamma`          | 0.99    | Discount factor                   |
| `--batch-size`     | 32      | Replay batch size                 |
| `--buffer-capacity`| 10000   | Replay buffer maximum size        |
| `--replay-per-step`| 5       | Number of replay batches per step |
| `--epsilon-start`  | 1.0     | Initial exploration rate          |
| `--epsilon-min`    | 0.1     | Minimum exploration rate          |
| `--epsilon-decay`  | 0.7     | Epsilon multiplier per iteration  |

---

### `src/envs/sumo_env.py`

A `gymnasium.Env` subclass that wraps a SUMO traffic simulation into an RL environment.

**Key design decisions**:
- **Per-data-point lifecycle**: SUMO starts fresh for each data point (25 minutes), then is killed and restarted. This prevents cross-contamination between different traffic demand patterns.
- **4 steps per data point**: After `warmup-minutes` of warm-up, the agent gets 4 chances to adjust params, each `period-minutes` window gives one reward.
- **Force-kill on Windows**: `traci.close()` is never called (triggers "Close all views?" dialog in `sumo-gui`). Instead, processes are killed via `taskkill /F /T` and PowerShell `Stop-Process -Force`. TraCI's internal connection is reset to `None` so `traci.start()` works on the next call.
- **Temp route files**: Each data point gets a fresh route XML in a `TemporaryDirectory` with its specific flow rates.
- **Induction loops**: Pre-defined in `induction_loop.xml` (loaded via the SUMO config as an additional file). Two loops cover northbound traffic, two cover southbound.

**Public interface**:
- `set_target_data(data)` — Provide the `(N, 6)` array of data points.
- `reset()` — Kill SUMO, advance to next data point, start fresh SUMO, run warm-up, return first state.
- `step(action)` — Apply param delta, run 5 min, read loops, compute reward, return `(next_obs, reward, terminated, truncated, info)`.
- `close()` — Stop SUMO and clean up temp files.

---

### `src/agents/dqn_agent.py`

Three classes:

**`ReplayBuffer`** — Fixed-capacity circular buffer using `collections.deque`. Stores `(state, action, reward, next_state, done)` tuples. `sample(batch_size)` returns batched NumPy arrays.

**`QNetwork`** — Simple feedforward network with configurable hidden dimensions (default `[64, 64]`). Input: state_dim (4). Output: action_dim (9). ReLU activations between linear layers.

**`DQNAgent`** — The main agent class:
- Two Q-networks: `q_net` (online, trained) and `target_net` (for stable Q-targets).
- `act(state, epsilon)` — ε-greedy action selection.
- `remember(...)` — Push transition into replay buffer.
- `replay()` — Sample batch, compute Double DQN loss, gradient step, soft-update target every N steps. Returns loss value.
- `save(path)` / `load(path)` — PyTorch checkpoint (state dicts for both nets + optimizer).

Device auto-detection: CUDA if available, else CPU.

---

### `src/logger.py`

**`TrainingLogger`** creates a timestamped directory under `logs/` each time training starts.

| Method | Writes to   | Content |
|--------|-------------|---------|
| `log_step(...)`  | `metrics.csv` | One row per RL step: iteration, data point, step index, reward, sim_north, sim_south, expected, error, params, epsilon, loss |
| `log_iteration(...)` | `summary.json` | Aggregated per-iteration: total_reward, avg_reward, avg_loss, epsilon, total_steps |

The config is saved once at init as `config.json`. The final model checkpoint is copied into the run directory by `train_dqn.py` after training completes.

---

### `data_repo.py`

A **singleton** that discovers and indexes all files under the project directory at construction time. Scans `data/`, `config/`, and `cctv_data/` subtrees and maps relative paths (e.g., `data/sumo_files/map_suhat_sumoconfig.sumocfg`) to absolute `Path` objects.

**Key properties**:
- `sumo_config_path` — Path to the `.sumocfg` file.
- `net_path` — Path to the `.net.xml` road network.
- `route_path` — Path to the `.rou.xml` route definitions.
- `induction_loop_path` — Path to `induction_loop.xml`.

**Parsing methods** (`read_xml` + typed getters):
- `get_sumo_config()` — Returns net-file, route-files, additional-files from the SUMO config.
- `get_routes()` — Returns list of flow definitions (id, begin, from, to, end, vehsPerHour).
- `get_induction_loops()` — Returns list of induction loop definitions (id, lane, pos, freq).
- `get_edges()` — Returns all non-internal edges with their lane details.

---

### `src/scripts/data/traffic_data.py`

A simple module that loads the Excel workbook once at import time (`data_only=True`). The `import_records()` function reads rows 3–30 from columns I–N (which map to `SL, SPT, UL, UPT, OS, OU`), converts values to `int`, and appends each row as a list of 6 dicts to the global `data_records` list. Stops at the first completely empty row.

---

## Unused Scripts

These scripts are part of an earlier version of the project and are **not used** by the current DQN training pipeline:

- `src/scripts/main.py`
- `src/scripts/main copy.py`
- `src/scripts/SumoSimulation.py`
- `src/scripts/run_recorder.py`
