# Development Guide

A step-by-step guide to building this project from scratch: a DQN reinforcement learning system that calibrates SUMO traffic simulation parameters (`lcCooperative`, `lcAssertive`) to match real-world CCTV traffic data.

---

## Table of Contents

- [0. Project Architecture](#0-project-architecture)
- [1. Data Processing — Excel to Training Array](#1-data-processing--excel-to-training-array)
- [2. SUMO & TraCI — Simulation Backend](#2-sumo--traci--simulation-backend)
- [3. RL Environment — SUMOEnv](#3-rl-environment--sumoenv)
- [4. DQN Agent — Neural Network & Training Logic](#4-dqn-agent--neural-network--training-logic)
- [5. Logger — Metrics Recording](#5-logger--metrics-recording)
- [6. Training Loop — Connecting Everything](#6-training-loop--connecting-everything)
- [Appendix: Unused Scripts](#appendix-unused-scripts)

---

## 0. Project Architecture

### What This System Does

The system reads real CCTV traffic counts from an Excel spreadsheet, then uses Deep Q-Networks (DQN) to learn how to adjust two SUMO lane-changing parameters so that the simulated traffic matches the observed real-world counts.

### High-Level Data Flow

```
CCTV Excel ──> Data Processing ──> Training Array ──> SUMOEnv ──> DQNAgent ──> Logger
                    ▲                                            │
                    └────────────────────────────────────────────┘
                         (state, reward, next_state)
```

### Tech Stack

| Component | Technology | Role |
|-----------|-----------|------|
| Language  | Python 3.10+ | Everything |
| Simulation| SUMO 1.24+ with TraCI | Traffic simulation + control interface |
| RL        | Gymnasium 1.3+ | Environment interface (reset/step/reward) |
| ML        | PyTorch 2+ | Neural networks, optimiser, checkpointing |
| Data      | openpyxl | Read CCTV Excel files |
| Data      | NumPy | Array storage, replay buffer batching |
| Logging   | CSV + JSON (stdlib) | Metrics recording |

### File Tree (Active Scripts Only)

```
project_root/
├── CCTV Data Remastered.xlsx
├── data_repo.py                       ← File indexer singleton
├── data/sumo_files/
│   ├── map_suhat_edit.net.xml         ← Road network
│   ├── map_suhat_netedit.rou.xml      ← Route template (overridden at runtime)
│   ├── induction_loop.xml             ← Induction loop definitions
│   └── map_suhat_sumoconfig.sumocfg   ← SUMO config
├── src/
│   ├── train_dqn.py                   ← Entry point
│   ├── logger.py                      ← TrainingLogger
│   ├── envs/sumo_env.py               ← SUMOEnv (Gymnasium)
│   ├── agents/dqn_agent.py            ← DQNAgent + QNetwork + ReplayBuffer
│   └── scripts/data/traffic_data.py   ← Excel parser
└── logs/                              ← Created at runtime
    └── YYYY-MM-DD_HH-MM-SS/
        ├── config.json
        ├── metrics.csv
        ├── summary.json
        └── dqn_final.pt
```

---

## 1. Data Processing — Excel to Training Array

### 1.1 Concept

The CCTV data is an Excel spreadsheet with hourly vehicle flow rates for 4 directions (SL, SPT, UL, UPT) and 2 observed totals (OS, OU). These 6 values per row become:

- **State** (first 4): `[SL, SPT, UL, UPT]` — given to the agent to inform its decision.
- **Reward target** (last 2): `[OS, OU]` — compared against simulated counts to compute the reward.

### 1.2 High-Level Steps

1. Install `openpyxl` and load the workbook with `data_only=True`.
2. Read rows I3:N30, convert cell values to integers.
3. Stop reading at the first empty row.
4. Merge each row's 6 dicts into one flat list and filter out all-zero rows.
5. Convert to a NumPy array of shape `(N, 6)` with dtype `float32`.
6. Build a `DataRepo` singleton to locate all SUMO file paths.

### 1.3 Detailed Implementation

#### 1.3.1 Load the Workbook

```python
from openpyxl import load_workbook

data_path = "CCTV Data Remastered.xlsx"
wb = load_workbook(data_path, data_only=True)
```

`data_only=True` means formulas are read as their last-cached values rather than as formula strings.

#### 1.3.2 Define Keys and Read Rows

Define the 6 column keys and read cells I3 through N30:

```python
key_names = ["SL", "SPT", "UL", "UPT", "OS", "OU"]
data_records: list = []

ws = wb.active
for row in ws['I3':'N30']:
    values = []
    row_has_data = False
    for cell in row:
        if cell.value is not None and cell.value != '':
            try:
                values.append(int(cell.value))
                row_has_data = True
            except (ValueError, TypeError):
                values.append(None)
        else:
            values.append(None)
    if not row_has_data:
        break
    data_records.append(values)
```

At this point `data_records` is a list of rows, where each row is a list of 6 values (some may be `None`).

#### 1.3.3 Build the Target Data Array

The training script needs a clean `(N, 6)` float32 array. Wrap the logic in `extract_target_data()`:

```python
import numpy as np

def extract_target_data() -> np.ndarray:
    rows = []
    for row in data_records:
        sl, spt, ul, upt, os_val, ou_val = [
            int(v) if v is not None else 0 for v in row
        ]
        # Skip rows where EVERYTHING is zero
        if sl == 0 and spt == 0 and ul == 0 and upt == 0 and os_val == 0 and ou_val == 0:
            continue
        rows.append([sl, spt, ul, upt, os_val, ou_val])
    return np.array(rows, dtype=np.float32)
```

#### 1.3.4 Build `data_repo.py` — File Indexer

A singleton class that scans the project for SUMO files so you don't hardcode paths everywhere:

```python
import os
from pathlib import Path
from threading import Lock


class DataRepo:
    _instance = None
    _lock = Lock()
    BASE_DIR = Path(__file__).parent.resolve()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._files = {}
        self._scan()

    def _scan(self):
        for directory in (self.BASE_DIR / "data", self.BASE_DIR / "config"):
            if not directory.exists():
                continue
            for root, _dirs, files in os.walk(directory):
                for fname in files:
                    fpath = Path(root) / fname
                    rel = str(fpath.relative_to(self.BASE_DIR)).replace("\\", "/")
                    self._files[rel] = fpath

    @property
    def sumo_config_path(self):
        return self._files.get("data/sumo_files/map_suhat_sumoconfig.sumocfg")
```

#### 1.3.5 Verify

```python
from data_repo import DataRepo
repo = DataRepo()
print(repo.sumo_config_path)  # Should resolve to an existing file
```

---

## 2. SUMO & TraCI — Simulation Backend

### 2.1 Concept

SUMO is the traffic simulator. TraCI (Traffic Control Interface) is a TCP-based protocol that lets Python control SUMO at runtime — start/stop simulation, inject vehicles, read detector data, change parameters.

You need 4 SUMO files (assumed already provided):

| File | Purpose |
|------|---------|
| `map_suhat_edit.net.xml` | Road network (edges, lanes, connections) |
| `map_suhat_netedit.rou.xml` | Route template — defines 4 traffic flows |
| `induction_loop.xml` | 4 induction loop detectors (north/south, left/right) |
| `map_suhat_sumoconfig.sumocfg` | Config that references the three files above |

### 2.2 High-Level Steps

1. Create a route XML **template** with dynamic `vehsPerHour` placeholders.
2. Start TraCI, run a few steps, verify loop readings, then close.
3. Add Windows-specific force-kill logic (skip `traci.close()` to avoid GUI dialog).
4. Verify that you can start, read data, kill, and restart cleanly.

### 2.3 Detailed Implementation

#### 2.3.1 Route XML Template

The route file defines 4 flows corresponding to the 4 CCTV directions. At runtime the flow rates are injected from the current data point:

```python
ROU_XML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <flow id="f_sl" begin="0.00" from="675098743#0" to="675098743#2"
          end="3600.00" vehsPerHour="{sl}"/>
    <flow id="f_spt" begin="0.00" from="675098743#0" to="585303113#2"
          end="3600.00" vehsPerHour="{spt}"/>
    <flow id="f_ul" begin="0.00" from="585303113#0" to="585303113#2"
          end="3600.00" vehsPerHour="{ul}"/>
    <flow id="f_upt" begin="0.00" from="585303113#0" to="675098743#2"
          end="3600.00" vehsPerHour="{upt}"/>
</routes>"""
```

#### 2.3.2 Induction Loop Definitions

The induction loops sit on specific lanes at the intersection to count vehicles:

```xml
<inductionLoop id="total_north_left" lane="675098743#2_1" pos="6.22" period="60.00" file="output.xml"/>
<inductionLoop id="total_north_right" lane="675098743#2_0" pos="6.22" period="60.00" file="output.xml"/>
<inductionLoop id="total_south_left" lane="585303113#2_1" pos="7.75" period="60.00" file="output.xml"/>
<inductionLoop id="total_south_right" lane="585303113#2_0" pos="7.75" period="60.00" file="output.xml"/>
```

- `total_north_left + total_north_right` = all northbound vehicles
- `total_south_left + total_south_right` = all southbound vehicles

#### 2.3.3 Test Bare TraCI Connection

```python
import os
import tempfile
import traci

# Write a temp route file
rou_xml = ROU_XML_TEMPLATE.format(sl=390, spt=204, ul=384, upt=36)
tmpdir = tempfile.mkdtemp(prefix="sumo_test_")
rou_path = os.path.join(tmpdir, "flows.rou.xml")
with open(rou_path, "w") as f:
    f.write(rou_xml)

cmd = [
    "sumo",                              # or "sumo-gui" for visual
    "-c", "data/sumo_files/map_suhat_sumoconfig.sumocfg",
    "--route-files", os.path.abspath(rou_path),
    "--step-length", "0.05",
    "--no-warnings",
    "--start",
]

traci.start(cmd)

# Run 100 steps and read induction loops
for _ in range(100):
    traci.simulationStep()

north = sum(traci.inductionloop.getLastStepVehicleNumber(lid)
            for lid in ("total_north_left", "total_north_right"))
south = sum(traci.inductionloop.getLastStepVehicleNumber(lid)
            for lid in ("total_south_left", "total_south_right"))
print(f"North: {north}, South: {south}")
```

#### 2.3.4 Windows Force-Kill

On Windows, `traci.close()` sends a GUI close request that triggers a "Close all views?" dialog box when using `sumo-gui`. To avoid this:

1. **Never call `traci.close()`**
2. Kill the process directly with `taskkill /F /T`
3. Add PowerShell fallback
4. Reset TraCI's internal connection so `traci.start()` works next time

```python
import subprocess
import traci

def _force_kill_sumo():
    # Primary: taskkill with tree kill
    for exe in ("sumo.exe", "sumo-gui.exe", "sumo", "sumo-gui"):
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/IM", exe],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
    # Fallback: PowerShell Stop-Process
    try:
        subprocess.run(
            ["powershell", "-Command",
             "Get-Process sumo,sumo-gui -ErrorAction SilentlyContinue | Stop-Process -Force"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass
    # Last resort: kill whatever holds TraCI port 8813
    try:
        subprocess.run(
            ["powershell", "-Command",
             "Get-NetTCPConnection -LocalPort 8813 -ErrorAction SilentlyContinue | "
             "Select-Object -ExpandProperty OwningProcess | "
             "ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass
    # Reset traci so next traci.start() works
    try:
        traci._connection = None
    except AttributeError:
        pass
```

#### 2.3.5 Verify Restart Cleanly

After killing, you should be able to call `traci.start(cmd)` again without errors:

```python
_force_kill_sumo()
traci.start(cmd)  # Should work
# ... run some steps ...
_force_kill_sumo()
```

---

## 3. RL Environment — SUMOEnv

### 3.1 Concept

`SUMOEnv` wraps a SUMO session into a Gymnasium `Env`. Each **data point** gets its own SUMO process (25 minutes of simulation: warm-up + 4 RL steps). The agent sees 4 floats as state, picks 1 of 9 actions, and receives a negative-MSE reward.

### 3.2 High-Level Steps

1. Define the action map (9 discrete `(Δcoop, Δassert)` pairs).
2. Define observation and action spaces.
3. Implement `_start_sumo()` — temp dir, dynamic route XML, launch via TraCI.
4. Implement `_run_warmup()` — default params, step for `warmup_minutes`.
5. Implement `_apply_params()` — set on live vehicles + `DEFAULT_VEHTYPE`.
6. Implement `step()` — apply action, run `period_steps`, read loops, compute reward.
7. Implement `reset()` — kill SUMO, advance data pointer, fresh start.
8. Implement `_stop_sumo()` + `close()` — force-kill + cleanup.

### 3.3 Detailed Implementation

#### 3.3.1 Action Map

Each action is a delta to apply to `lcCooperative` and `lcAssertive`. The map covers all 9 combinations of `{-0.2, 0.0, +0.2}`:

```python
action_map = [
    {"lcCooperative": -0.2, "lcAssertive": -0.2},
    {"lcCooperative": -0.2, "lcAssertive":  0.0},
    {"lcCooperative": -0.2, "lcAssertive":  0.2},
    {"lcCooperative":  0.0, "lcAssertive": -0.2},
    {"lcCooperative":  0.0, "lcAssertive":  0.0},
    {"lcCooperative":  0.0, "lcAssertive":  0.2},
    {"lcCooperative":  0.2, "lcAssertive": -0.2},
    {"lcCooperative":  0.2, "lcAssertive":  0.0},
    {"lcCooperative":  0.2, "lcAssertive":  0.2},
]
```

#### 3.3.2 Observation & Action Spaces

```python
import gymnasium as gym
from gymnasium import spaces

self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
self.action_space = spaces.Discrete(len(action_map))
```

- **State**: `[SL, SPT, UL, UPT]` — hourly flow rates (same values fed to SUMO's flows).
- **Action**: integer 0–8 mapping to the action map above.

#### 3.3.3 `_start_sumo()`

```python
import tempfile

def _start_sumo(self, sl, spt, ul, upt):
    # Clean up previous temp dir
    if self._temp_dir is not None:
        self._temp_dir.cleanup()

    self._temp_dir = tempfile.TemporaryDirectory(prefix="sumo_env_")

    rou_xml = ROU_XML_TEMPLATE.format(sl=sl, spt=spt, ul=ul, upt=upt)
    rou_path = os.path.join(self._temp_dir.name, "flows.rou.xml")
    with open(rou_path, "w") as f:
        f.write(rou_xml)

    cmd = [
        self.sumo_binary,
        "-c", self.sumo_config_path,
        "--route-files", os.path.abspath(rou_path),
        "--step-length", str(self.step_length),
        "--no-warnings",
        "--start",
    ]

    traci.start(cmd)
    self._sim_active = True
```

Key points:
- A fresh temp directory is created per data point to avoid file conflicts.
- `--route-files` overrides the route file in the .sumocfg so the dynamic flows are used.
- The config's `induction_loop.xml` (referenced as an additional file) is still loaded.

#### 3.3.4 `_run_warmup()`

Run `warmup_minutes` of simulation with default params (1.0, 1.0). No RL interaction yet.

```python
def _run_warmup(self):
    traci.vehicletype.setParameter(
        "DEFAULT_VEHTYPE", "laneChangeModel.lcCooperative", "1.0"
    )
    traci.vehicletype.setParameter(
        "DEFAULT_VEHTYPE", "laneChangeModel.lcAssertive", "1.0"
    )
    for _ in range(self.warmup_steps):
        traci.simulationStep()
```

Where `warmup_steps = warmup_minutes × (60 / step_length)`.

#### 3.3.5 `_apply_params()`

After the agent chooses an action, apply the new params to every alive vehicle (immediate effect) and to `DEFAULT_VEHTYPE` (affects newly spawned vehicles):

```python
def _apply_params(self):
    for veh_id in traci.vehicle.getIDList():
        traci.vehicle.setParameter(
            veh_id, "laneChangeModel.lcCooperative",
            str(self.current_params["lcCooperative"])
        )
        traci.vehicle.setParameter(
            veh_id, "laneChangeModel.lcAssertive",
            str(self.current_params["lcAssertive"])
        )
    traci.vehicletype.setParameter(
        "DEFAULT_VEHTYPE", "laneChangeModel.lcCooperative",
        str(self.current_params["lcCooperative"])
    )
    traci.vehicletype.setParameter(
        "DEFAULT_VEHTYPE", "laneChangeModel.lcAssertive",
        str(self.current_params["lcAssertive"])
    )
```

Params are clamped to `[0.0, 5.0]` before applying.

#### 3.3.6 `step()`

This is the core RL interaction. For each step:

1. Apply the action delta to params (clamped).
2. Apply params to vehicles.
3. Run `period_steps` of simulation, accumulating loop counts.
4. Compute reward.
5. Return `(next_obs, reward, terminated, truncated, info)`.

```python
def step(self, action: int):
    delta = self.action_map[action]
    coop = self.current_params["lcCooperative"] + delta["lcCooperative"]
    assertive = self.current_params["lcAssertive"] + delta["lcAssertive"]
    self.current_params["lcCooperative"] = max(0.0, min(5.0, coop))
    self.current_params["lcAssertive"] = max(0.0, min(5.0, assertive))

    self._apply_params()

    # Run period_steps and count vehicles
    sim_south = 0
    sim_north = 0
    for _ in range(self.period_steps):
        traci.simulationStep()
        for lid in self.north_loop_ids:
            sim_north += traci.inductionloop.getLastStepVehicleNumber(lid)
        for lid in self.south_loop_ids:
            sim_south += traci.inductionloop.getLastStepVehicleNumber(lid)

    # Reward: negative MSE against expected 5-minute counts
    expected_south = self._current_row[4] / 12.0
    expected_north = self._current_row[5] / 12.0
    reward = -((sim_south - expected_south) ** 2 + (sim_north - expected_north) ** 2) / 2.0

    self.step_idx += 1
    terminated = self.step_idx >= self.steps_per_data

    if terminated:
        self.current_data_idx += 1
        if self.current_data_idx >= len(self.target_data):
            self.current_data_idx = 0

    next_obs = np.array(
        [self._current_row[0], self._current_row[1],
         self._current_row[2], self._current_row[3]],
        dtype=np.float32,
    )

    info = {
        "sim_south": sim_south,
        "sim_north": sim_north,
        "expected_south": expected_south,
        "expected_north": expected_north,
        "params": dict(self.current_params),
    }

    return next_obs, reward, terminated, False, info
```

**Why divide by 12?** `OS` and `OU` are hourly rates (vehicles per hour). Each RL step is `period_minutes` long (default 5). `5 min / 60 min = 1/12`, so `OS/12` gives the expected count for that window.

#### 3.3.7 `reset()`

```python
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)
    self._stop_sumo()

    row = self.target_data[self.current_data_idx]
    self._current_row = row
    self.step_idx = 0
    self.current_params = {"lcCooperative": 1.0, "lcAssertive": 1.0}

    self._start_sumo(int(row[0]), int(row[1]), int(row[2]), int(row[3]))
    self._run_warmup()

    obs = np.array([row[0], row[1], row[2], row[3]], dtype=np.float32)
    return obs, {"params": dict(self.current_params)}
```

#### 3.3.8 `_stop_sumo()` and `close()`

```python
def _stop_sumo(self):
    if self._sim_active:
        self._sim_active = False
        self._force_kill_sumo()
        try:
            traci._connection = None
        except AttributeError:
            pass
    if self._temp_dir is not None:
        try:
            self._temp_dir.cleanup()
        except PermissionError:
            pass
        self._temp_dir = None

def close(self):
    self._stop_sumo()
```

---

## 4. DQN Agent — Neural Network & Training Logic

### 4.1 Concept

The agent learns a mapping from traffic state `[SL, SPT, UL, UPT]` to Q-values for each of the 9 actions. It stores past experiences in a replay buffer and learns from them using Double DQN.

### 4.2 High-Level Steps

1. Build `ReplayBuffer` — stores `(s, a, r, s', done)` and samples random batches.
2. Build `QNetwork` — a small MLP: `Linear(4,64) → ReLU → Linear(64,64) → ReLU → Linear(64,9)`.
3. Build `DQNAgent` — wraps the two Q-networks (online + target), ε-greedy action selection, replay training with soft target updates, and save/load.

### 4.3 Detailed Implementation

#### 4.3.1 ReplayBuffer

```python
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
```

#### 4.3.2 QNetwork

```python
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

| Layer | Input → Output | Activation |
|-------|---------------|------------|
| Linear | 4 → 64 | ReLU |
| Linear | 64 → 64 | ReLU |
| Linear | 64 → 9 | (none) |

Output dimension = 9 (one Q-value per action).

#### 4.3.3 DQNAgent — Initialisation

```python
import torch
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 batch_size=64, buffer_capacity=50000, tau=0.005,
                 target_update_freq=100, hidden_dims=None, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.target_update_freq = target_update_freq
        self._step_counter = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_capacity)
```

#### 4.3.4 Action Selection (ε-Greedy)

```python
import random

def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
    if random.random() < epsilon:
        return random.randrange(self.action_dim)
    state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
    with torch.no_grad():
        q_values = self.q_net(state_t)
    return int(q_values.argmax(dim=1).item())
```

- With probability ε: return random action (exploration).
- With probability 1-ε: return action with highest Q-value (exploitation).

#### 4.3.5 Store Experience

```python
def remember(self, state, action, reward, next_state, done):
    self.replay_buffer.push(state, action, reward, next_state, done)
```

#### 4.3.6 Replay Training (Double DQN)

```python
def replay(self):
    if len(self.replay_buffer) < self.batch_size:
        return None

    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

    states_t = torch.from_numpy(states).to(self.device)
    actions_t = torch.from_numpy(actions).unsqueeze(1).to(self.device)
    rewards_t = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
    next_states_t = torch.from_numpy(next_states).to(self.device)
    dones_t = torch.from_numpy(dones).unsqueeze(1).to(self.device)

    # Current Q-values for the actions that were taken
    current_q = self.q_net(states_t).gather(1, actions_t)

    # Double DQN: select actions with online net, evaluate with target net
    with torch.no_grad():
        next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
        next_q = self.target_net(next_states_t).gather(1, next_actions)
        target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

    loss = self.loss_fn(current_q, target_q)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Soft target update
    self._step_counter += 1
    if self._step_counter % self.target_update_freq == 0:
        self._soft_update_target()

    return loss.item()

def _soft_update_target(self):
    for target_param, q_param in zip(self.target_net.parameters(), self.q_net.parameters()):
        target_param.data.copy_(
            self.tau * q_param.data + (1.0 - self.tau) * target_param.data
        )
```

**Double DQN**: instead of `max Q_target(s')`, we use `Q_target(s', argmax Q_online(s'))`. This reduces overestimation bias.

**Soft update**: slowly blend the target network toward the online network: `θ_target ← τ·θ_online + (1-τ)·θ_target` every N steps.

#### 4.3.7 Save / Load

```python
def save(self, path: str):
    torch.save({
        "q_net_state_dict": self.q_net.state_dict(),
        "target_net_state_dict": self.target_net.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
    }, path)

def load(self, path: str):
    checkpoint = torch.load(path, map_location=self.device, weights_only=True)
    self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
    self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

---

## 5. Logger — Metrics Recording

### 5.1 Concept

Every RL step produces numbers (reward, simulated counts, params, loss). The logger saves these to disk so you can inspect training progress, compare runs, and diagnose problems.

### 5.2 High-Level Steps

1. Create a timestamped run folder.
2. Save all hyperparameters as `config.json`.
3. Open a `metrics.csv` file and write one row per RL step.
4. Maintain a summary array and write `summary.json` after each iteration.

### 5.3 Detailed Implementation

#### 5.3.1 Initialisation

```python
import csv
import json
import os
from datetime import datetime
from pathlib import Path

class TrainingLogger:
    def __init__(self, log_dir: str = "logs", hyperparams: dict = None):
        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._run_dir = Path(log_dir) / folder_name
        os.makedirs(self._run_dir, exist_ok=True)

        # Save config
        if hyperparams:
            with open(self._run_dir / "config.json", "w") as f:
                json.dump(hyperparams, f, indent=2, default=str)

        # Open CSV
        csv_path = self._run_dir / "metrics.csv"
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "iteration", "data_point", "step_in_data",
            "reward", "sim_south", "sim_north",
            "expected_south", "expected_north",
            "error_south", "error_north",
            "lcCooperative", "lcAssertive",
            "epsilon", "loss",
        ])
        self._csv_file.flush()

        self._summary = []
```

The folder structure:

```
logs/
└── 2026-06-18_14-30-00/
    ├── config.json
    ├── metrics.csv
    ├── summary.json      <-- written at end (or incrementally)
    └── dqn_final.pt      <-- copied here by train_dqn.py
```

#### 5.3.2 Logging a Step

```python
def log_step(self, iteration, data_idx, step_in_data, reward,
             sim_south, sim_north, expected_south, expected_north,
             lcCooperative, lcAssertive, epsilon, loss=None):
    self._csv_writer.writerow([
        iteration, data_idx, step_in_data,
        f"{reward:.3f}",
        f"{sim_south:.0f}", f"{sim_north:.0f}",
        f"{expected_south:.3f}", f"{expected_north:.3f}",
        f"{sim_south - expected_south:.3f}",
        f"{sim_north - expected_north:.3f}",
        f"{lcCooperative:.3f}", f"{lcAssertive:.3f}",
        f"{epsilon:.4f}",
        f"{loss:.6f}" if loss is not None else "",
    ])
    self._csv_file.flush()
```

Flushing after every row ensures no data is lost if the training crashes mid-run.

#### 5.3.3 Logging an Iteration

```python
def log_iteration(self, iteration, total_reward, avg_reward,
                  avg_loss, epsilon, total_steps):
    entry = {
        "iteration": iteration,
        "total_reward": round(total_reward, 3),
        "avg_reward": round(avg_reward, 3),
        "avg_loss": round(avg_loss, 6) if avg_loss is not None else None,
        "epsilon": round(epsilon, 4),
        "total_steps": total_steps,
    }
    self._summary.append(entry)
    with open(self._run_dir / "summary.json", "w") as f:
        json.dump(self._summary, f, indent=2)
```

#### 5.3.4 Close

```python
def close(self):
    self._csv_file.close()
```

---

## 6. Training Loop — Connecting Everything

### 6.1 Concept

The training script ties all components together: it loads the data, creates the environment and agent, then runs N iterations of training, logging everything along the way.

### 6.2 High-Level Steps

1. Parse CLI arguments.
2. Load target data from Excel.
3. Create `SUMOEnv` with chosen binary and config.
4. Set target data on the environment.
5. Create `DQNAgent`.
6. Create `TrainingLogger`.
7. Outer loop (iterations):
   - Reset the environment's data pointer to start from data point 0.
   - Inner loop (data points):
     - `env.reset()` → starts SUMO for this data point, runs warm-up.
     - 4 times: `act → step → remember → replay → log`.
   - Decay epsilon.
   - Log iteration summary.
8. Save model checkpoint.
9. Clean up.

### 6.3 Detailed Implementation

#### 6.3.1 CLI Arguments

```python
import argparse

parser = argparse.ArgumentParser(description="DQN training for SUMO calibration")
parser.add_argument("--iterations", type=int, default=5)
parser.add_argument("--sumo-binary", default="sumo")
parser.add_argument("--warmup-minutes", type=int, default=5)
parser.add_argument("--period-minutes", type=int, default=5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--buffer-capacity", type=int, default=10000)
parser.add_argument("--replay-per-step", type=int, default=5)
parser.add_argument("--epsilon-start", type=float, default=1.0)
parser.add_argument("--epsilon-min", type=float, default=0.1)
parser.add_argument("--epsilon-decay", type=float, default=0.7)
args = parser.parse_args()
```

#### 6.3.2 Initialise Components

```python
# 1. Load data
target_data = extract_target_data()

# 2. Environment
repo = DataRepo()
env = SUMOEnv(
    sumo_config_path=str(repo.sumo_config_path),
    sumo_binary=args.sumo_binary,
    warmup_minutes=args.warmup_minutes,
    period_minutes=args.period_minutes,
)
env.set_target_data(target_data)

# 3. Agent
agent = DQNAgent(
    state_dim=4,
    action_dim=env.n_actions,
    lr=args.lr,
    gamma=args.gamma,
    batch_size=args.batch_size,
    buffer_capacity=args.buffer_capacity,
)

# 4. Logger
logger = TrainingLogger(log_dir="logs", hyperparams=vars(args))

epsilon = args.epsilon_start
```

#### 6.3.3 Training Loop

```python
n_data = len(target_data)

for iteration in range(args.iterations):
    env.reset_data_pointer()
    total_iter_reward = 0.0
    total_steps = 0
    iter_loss_sum = 0.0
    iter_loss_count = 0

    for data_idx in range(n_data):
        obs, _ = env.reset()
        data_reward = 0.0

        for step_in_data in range(4):
            action = agent.act(obs, epsilon)
            next_obs, reward, terminated, _, info = env.step(action)
            agent.remember(obs, action, reward, next_obs, terminated)

            step_loss = None
            for _ in range(args.replay_per_step):
                l = agent.replay()
                if l is not None:
                    step_loss = l
            if step_loss is not None:
                iter_loss_sum += step_loss
                iter_loss_count += 1

            logger.log_step(
                iteration=iteration + 1,
                data_idx=data_idx,
                step_in_data=step_in_data + 1,
                reward=reward,
                sim_south=info["sim_south"],
                sim_north=info["sim_north"],
                expected_south=info["expected_south"],
                expected_north=info["expected_north"],
                lcCooperative=info["params"]["lcCooperative"],
                lcAssertive=info["params"]["lcAssertive"],
                epsilon=epsilon,
                loss=step_loss,
            )

            obs = next_obs
            data_reward += reward
            total_steps += 1

        total_iter_reward += data_reward

    # Decay epsilon
    epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)

    # Log iteration summary
    avg_reward = total_iter_reward / (n_data * 4) if n_data > 0 else 0.0
    avg_loss = (iter_loss_sum / iter_loss_count) if iter_loss_count > 0 else None

    logger.log_iteration(
        iteration=iteration + 1,
        total_reward=total_iter_reward,
        avg_reward=avg_reward,
        avg_loss=avg_loss,
        epsilon=epsilon,
        total_steps=total_steps,
    )

    # Print progress
    loss_str = f"{avg_loss:.6f}" if avg_loss is not None else "N/A"
    print(f"  Iter {iteration + 1:2d}/{args.iterations}  "
          f"total={total_iter_reward:+9.1f}  "
          f"avg={avg_reward:+7.3f}  "
          f"loss={loss_str}  "
          f"eps={epsilon:.3f}  "
          f"buf={len(agent.replay_buffer):4d}  "
          f"steps={total_steps}")
```

#### 6.3.4 Save and Clean Up

```python
final_ckpt = logger.run_dir / "dqn_final.pt"
agent.save(str(final_ckpt))
logger.close()
env.close()
print(f"Done. Model saved to {final_ckpt}")
```

#### 6.3.5 Run

```bash
python src/train_dqn.py --iterations 5 --sumo-binary sumo
```

For visual debugging:

```bash
python src/train_dqn.py --iterations 2 --sumo-binary sumo-gui
```

---

## Appendix: Unused Scripts

The following scripts are part of an earlier version of the project and are **not used** by the current DQN training pipeline:

- `src/scripts/main.py`
- `src/scripts/main copy.py`
- `src/scripts/SumoSimulation.py`
- `src/scripts/run_recorder.py`
