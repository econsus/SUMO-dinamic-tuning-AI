# Comprehensive Project Plan: SUMO + DQN Traffic Auto-Calibration System

## Project Overview

This project implements a reinforcement learning system where a DQN (Deep Q-Network) agent acts as an **auto-tuner** for SUMO traffic simulation parameters. The system calibrates SUMO to match real-world CCTV data.

**Core Concept:**
- **Input:** Real CCTV data (vehicle counts per minute)
- **Agent Action:** Optimize driver behavior parameters
- **Reward:** Minimize error between SUMO output and CCTV data
- **Goal:** Make SUMO simulation match real-world traffic patterns

---

## 1. PROJECT STRUCTURE

### 1.1 Complete Folder Hierarchy

```
SUMO-dinamic-tuning-AI/
├── config/                          # Configuration files
│   ├── default.yaml                 # Default hyperparameters
│   ├── training.yaml                 # Training-specific config
│   ├── simulation.yaml              # SUMO simulation parameters
│   └── model.yaml                   # DQN model architecture
├── sumo_files/                      # SUMO simulation files
│   ├── networks/                    # Road network files (.net.xml)
│   │   └── uturn_road/
│   │       └── uturn_network.net.xml
│   ├── routes/                      # Traffic route files (.rou.xml)
│   │   ├── light_traffic.rou.xml
│   │   ├── heavy_traffic.rou.xml
│   │   └── mixed_traffic.rou.xml
│   ├── configs/                    # SUMO configuration files (.sumocfg)
│   │   └── default.sumocfg
│   └── vehicle_types/              # Vehicle type definitions
│       └── vehicle_params.xml
├── cctv_data/                      # Real CCTV data
│   ├── raw/                         # Raw CCTV data files
│   ├── processed/                  # Processed/normalized data
│   └── splits/
│       ├── train.csv                # Training data (e.g., 70%)
│       └── val.csv                  # Validation data (e.g., 30%)
├── src/                             # Source code
│   ├── sumo_env/                   # SUMO environment wrapper
│   │   ├── __init__.py
│   │   ├── simulation.py           # Main simulation class
│   │   ├── driver_params.py       # Driver behavior parameter controller
│   │   ├── metrics.py              # Data extraction utilities
│   │   ├── volume_collector.py   # Vehicle count per minute collector
│   │   └── rewards.py              # Reward functions
│   ├── agents/                     # DQN agent implementation
│   │   ├── __init__.py
│   │   ├── dqn_agent.py             # Main DQN class
│   │   ├── networks.py              # Neural network architectures
│   │   ├── replay_buffer.py         # Experience replay
│   │   └── exploration.py           # Epsilon-greedy strategies
│   ├── calibration/               # Calibration-specific modules
│   │   ├── __init__.py
│   │   ├── data_loader.py          # CCTV data loading
│   │   ├── error_calculator.py    # Error computation
│   │   └── parameter_optimizer.py # Parameter tuning logic
│   ├── integration/                # Integration layer
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # Main orchestration script
│   │   ├── data_pipeline.py        # Data flow management
│   │   └── checkpoint_manager.py    # Model persistence
│   ├── utils/                      # Utility functions
│   │   ├── __init__.py
│   │   ├── logger.py               # Logging configuration
│   │   ├── config_loader.py         # Config management
│   │   └── metrics_tracker.py      # Metrics tracking
│   └── visualization/             # Visualization tools
│       ├── __init__.py
│       ├── plotter.py               # Training plots
│       └── dashboard.py             # Real-time dashboard
├── data/                           # Data storage
│   ├── episodes/                   # Episode-level data (CSV)
│   │   └── ep_{episode_id}_{timestamp}.csv
│   ├── training_logs/              # Training metrics
│   │   └── training_log.csv
│   ├── models/                     # Saved model checkpoints
│   │   └── checkpoints/
│   └── results/                    # Final results
│       └── experiments/
├── logs/                           # Application logs
│   ├── training/
│   │   └── {timestamp}.log
│   └── simulation/
│       └── {timestamp}.log
├── scripts/                        # Execution scripts
│   ├── train.py                    # Training entry point
│   ├── evaluate.py                 # Evaluation script
│   ├── visualize.py                # Visualization script
│   ├── prepare_data.py             # Data splitting script
│   └── calibrate.py               # Standalone calibration script
├── tests/                          # Unit tests
│   ├── test_sumo_env/
│   ├── test_agent/
│   └── test_integration/
├── notebooks/                      # Jupyter notebooks
│   ├── exploration.ipynb
│   ├── analysis.ipynb
│   └── data_preparation.ipynb
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── pyproject.toml                  # Project metadata
├── README.md                       # Project documentation
└── LICENSE                         # License file
```

### 1.2 Naming Conventions

| Component | Convention | Example |
|-----------|------------|---------|
| Classes | PascalCase | `DQNAgent`, `SumoSimulation` |
| Functions | snake_case | `compute_error`, `load_cctv_data` |
| Constants | UPPER_SNAKE_CASE | `BUFFER_SIZE`, `NUM_ACTIONS` |
| Config keys | snake_case | `learning_rate`, `gamma` |
| Files | snake_case | `replay_buffer.py`, `dqn_agent.py` |

### 1.3 Configuration Management

```yaml
# config/default.yaml
simulation:
  sumo_home: "C:/Program Files/SUMO"
  gui: false
  time_step: 1
  max_steps: 3600          # 1 hour simulation
  network_file: "sumo_files/networks/uturn_road/uturn_network.net.xml"
  route_file: "sumo_files/routes/mixed_traffic.rou.xml"
  config_file: "sumo_files/configs/default.sumocfg"

data:
  cctv_data_path: "cctv_data/raw/cctv_data.csv"
  train_split: 0.7        # 70% training, 30% validation
  normalize: true
  time_window: 60         # 60-second windows for vehicle counts

agent:
  state_space:
    features: ["sim_uturn_count", "sim_straight_count", "sim_merged_count", "param_aggressiveness", "param_cooperativeness"]
    history_length: 5
  action_space:
    type: "discrete"
    num_actions: 125     # 5^3 = 125 parameter combinations (5 bins × 3 params)
  dqn:
    hidden_layers: [64, 128, 64]
    activation: "relu"
    optimizer: "adam"
    learning_rate: 0.001
    gamma: 0.95
    epsilon_start: 1.0
    epsilon_end: 0.01
    epsilon_decay: 0.995
    buffer_size: 50000
    batch_size: 32
    target_update_freq: 200
    gradient_clip: 1.0

driver_behavior:
  # Tunable driver parameters
  aggressiveness:
    min: 0.1
    max: 1.0
    bins: 5
    default: 0.5
  cooperativeness:
    min: 0.1
    max: 1.0
    bins: 5
    default: 0.5
  lane_change_threshold:
    min: 0.2
    max: 0.9
    bins: 5
    default: 0.5

training:
  num_episodes: 200
  episode_max_steps: 100
  save_freq: 20
  eval_freq: 10
  eval_episodes: 5
  early_stopping_patience: 30
  target_mse: 0.01       # Target MSE for stopping
```

---

## 2. COMPONENT ARCHITECTURE

### 2.A. SUMO Simulation Module

#### A.1 Scenario: U-Turn Volume Measurement

```
+=======================================================================+
|                                                                       |
|   Zone A          U-Turn Zone          Zone B         Zone C          |
|  (Before)      (Median Crossover)    (After U-turn)  (Merged)        |
|                                                                       |
|   ------>          [U-Turn]            ------       ------            |
|   Lane 1                              Lane 1        Lane 1           |
|                                                                       |
|   <------                            <------      <------            |
|   Lane 2          [U-Turn]            Lane 2        Lane 2           |
|                                                                       |
+=======================================================================+

Metrics Collected:
- Zone A: Vehicle count entering (per minute)
- U-Turn Zone: Vehicle count attempting U-turn (per minute)
- Zone B: Vehicle count completing U-turn (per minute)  
- Zone C: Vehicle count after merging to opposite lane (per minute)
- Straight: Vehicle count going straight without U-turn (per minute)
```

#### A.2 File Structure

**Network Files (.net.xml):**
- Road with U-turn crossovers
- Defined zones for vehicle counting

**Route Files (.rou.xml):**
```xml
<routes>
    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" maxSpeed="50"/>
    <route id="uturn_route" edges="edge_A lane1 uturn1 lane1 edge_B"/>
    <route id="straight_route" edges="edge_A lane1 edge_C"/>
    <vehicle id="veh0" route="uturn_route" depart="0" type="car"/>
</routes>
```

#### A.3 Volume Collector

```python
# src/sumo_env/volume_collector.py

class VolumeCollector:
    """
    Collects vehicle volume metrics per minute from SUMO simulation.
    Matches the format of CCTV data for comparison.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.time_window = config['data']['time_window']  # 60 seconds
        self.zones = {
            'zone_a': [],      # Entering the road
            'uturn_attempt': [], # Attempting U-turn
            'zone_b': [],      # Completed U-turn
            'zone_c': [],      # After merging
            'straight': []    # Going straight
        }
        self.current_window_vehicles = {
            'zone_a': set(),
            'uturn_attempt': set(),
            'zone_b': set(),
            'zone_c': set(),
            'straight': set()
        }
        self.edge_mapping = {
            'zone_a': ['edge_A'],
            'uturn_attempt': ['uturn_edge_1', 'uturn_edge_2'],
            'zone_b': ['edge_B'],
            'zone_c': ['edge_C'],
            'straight': ['straight_edge']
        }
        
    def reset(self):
        """Reset all counters."""
        for key in self.zones:
            self.zones[key] = []
            self.current_window_vehicles[key] = set()
            
    def update(self, step: int):
        """Update vehicle counts at current step."""
        # Check each zone for vehicles
        for zone_name, edges in self.edge_mapping.items():
            for edge in edges:
                vehicles = traci.edge.getLastStepVehicleIDs(edge)
                self.current_window_vehicles[zone_name].update(vehicles)
        
        # At end of time window, record counts
        if step % self.time_window == 0:
            for zone_name in self.zones:
                count = len(self.current_window_vehicles[zone_name])
                self.zones[zone_name].append(count)
                self.current_window_vehicles[zone_name] = set()
                
    def get_current_counts(self) -> dict:
        """Get current time window vehicle counts."""
        return {zone: len(vehicles) for zone, vehicles in self.current_window_vehicles.items()}
    
    def get_window_counts(self, window_idx: int) -> dict:
        """Get counts for specific time window."""
        if window_idx >= len(self.zones['zone_a']):
            return None
        return {zone: counts[window_idx] for zone, counts in self.zones.items()}
    
    def get_all_counts(self) -> pd.DataFrame:
        """Get all time windows as DataFrame."""
        return pd.DataFrame(self.zones)
```

#### A.4 Driver Behavior Parameter Controller

```python
# src/sumo_env/driver_params.py

class DriverBehaviorController:
    """Controls driver behavior parameters in SUMO for calibration."""
    
    def __init__(self, config: dict):
        self.config = config
        self.param_config = config['driver_behavior']
        self.current_params = self._get_default_params()
        
    def _get_default_params(self) -> dict:
        """Get default parameter values."""
        return {
            'aggressiveness': self.param_config['aggressiveness']['default'],
            'cooperativeness': self.param_config['cooperativeness']['default'],
            'lane_change_threshold': self.param_config['lane_change_threshold']['default']
        }
    
    def set_params(self, params: dict):
        """Set driver parameters."""
        self.current_params = params
        self._apply_to_simulation()
        
    def get_params(self) -> dict:
        """Get current parameters."""
        return self.current_params.copy()
    
    def action_to_params(self, action_idx: int) -> dict:
        """
        Convert discretized action to parameter values.
        
        With 5 bins × 3 params = 125 total actions
        """
        bins = {
            'aggressiveness': self.param_config['aggressiveness']['bins'],
            'cooperativeness': self.param_config['cooperativeness']['bins'],
            'lane_change_threshold': self.param_config['lane_change_threshold']['bins']
        }
        
        # Decode action index
        a = action_idx
        agg_idx = a % bins['aggressiveness']
        a //= bins['aggressiveness']
        coop_idx = a % bins['cooperativeness']
        a //= bins['cooperativeness']
        lc_idx = a % bins['lane_change_threshold']
        
        # Convert indices to values
        params = {
            'aggressiveness': self._index_to_value(
                agg_idx, bins['aggressiveness'],
                self.param_config['aggressiveness']['min'],
                self.param_config['aggressiveness']['max']
            ),
            'cooperativeness': self._index_to_value(
                coop_idx, bins['cooperativeness'],
                self.param_config['cooperativeness']['min'],
                self.param_config['cooperativeness']['max']
            ),
            'lane_change_threshold': self._index_to_value(
                lc_idx, bins['lane_change_threshold'],
                self.param_config['lane_change_threshold']['min'],
                self.param_config['lane_change_threshold']['max']
            )
        }
        
        self.current_params = params
        self._apply_to_simulation()
        return params
    
    def _index_to_value(self, idx: int, num_bins: int, min_val: float, max_val: float) -> float:
        """Map bin index to parameter value."""
        step = (max_val - min_val) / (num_bins - 1)
        return min_val + idx * step
    
    def _apply_to_simulation(self):
        """Apply current parameters to SUMO vehicles."""
        for veh_id in traci.vehicle.getIDList():
            traci.vehicle.setParameter(veh_id, "laneChangeModel.aggressiveness", 
                                     self.current_params['aggressiveness'])
            traci.vehicle.setParameter(veh_id, "laneChangeModel.cooperate", 
                                     self.current_params['cooperativeness'])
            traci.vehicle.setParameter(veh_id, "laneChangeModel.timeToImpatience",
                                     self.current_params['lane_change_threshold'])
```

#### A.5 Simulation Wrapper

```python
# src/sumo_env/simulation.py

class SumoCalibrationSimulation:
    """SUMO simulation wrapper for volume-based calibration."""
    
    def __init__(self, config: dict):
        self.config = config
        self.volume_collector = VolumeCollector(config)
        self.driver_controller = DriverBehaviorController(config)
        self.current_step = 0
        self.current_action = 0
        self.current_params = self.driver_controller._get_default_params()
        
    def start(self, net_file: str, route_file: str, config_file: str):
        """Initialize SUMO simulation."""
        sumo_cmd = [
            "sumo" if not self.config.get('gui') else "sumo-gui",
            "-c", config_file,
            "--start", "true"
        ]
        traci.start(sumo_cmd)
        
    def reset(self) -> np.ndarray:
        """Reset simulation and return initial state."""
        self.current_step = 0
        self.volume_collector.reset()
        # Apply default parameters
        self.current_params = self.driver_controller._get_default_params()
        self.driver_controller.set_params(self.current_params)
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one simulation step.
        
        In calibration mode, 'action' is applied for entire episode,
        then we compare output to CCTV data.
        """
        # Apply action (driver parameters) if changed
        if action != self.current_action:
            self.current_params = self.driver_controller.action_to_params(action)
            self.current_action = action
        
        # Advance simulation
        traci.simulationStep()
        self.current_step += 1
        
        # Collect volume metrics
        self.volume_collector.update(self.current_step)
        
        # Get state
        state = self._get_state()
        
        # For calibration, reward is computed after episode ends
        reward = 0.0
        
        # Check if episode is done
        max_steps = self.config['simulation']['max_steps']
        done = self.current_step >= max_steps
        
        return state, reward, done
    
    def compute_calibration_reward(self, cctv_data: np.ndarray) -> float:
        """
        Compute reward based on how well simulation matches CCTV data.
        
        Lower MSE = Higher reward
        """
        sim_counts = self.volume_collector.get_all_counts()
        
        # Align data lengths
        min_len = min(len(sim_counts), len(cctv_data))
        
        if min_len == 0:
            return -1000.0  # Heavy penalty for no data
        
        # Calculate MSE
        mse = np.mean((sim_counts[:min_len] - cctv_data[:min_len]) ** 2)
        
        # Convert to reward (lower error = higher reward)
        reward = -mse
        
        return reward
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        current_counts = self.volume_collector.get_current_counts()
        
        # State: [uturn_count, straight_count, merged_count, agg, coop, lc_threshold]
        state = np.array([
            current_counts.get('uturn_attempt', 0),
            current_counts.get('straight', 0),
            current_counts.get('zone_c', 0),
            self.current_params['aggressiveness'],
            self.current_params['cooperativeness'],
            self.current_params['lane_change_threshold']
        ])
        
        return state
    
    def get_simulated_volumes(self) -> pd.DataFrame:
        """Get simulated vehicle volumes for comparison."""
        return self.volume_collector.get_all_counts()
    
    def close(self):
        """Clean up resources."""
        traci.close()
```

---

### 2.B. Calibration Module

#### B.1 Data Loader

```python
# src/calibration/data_loader.py

class CCTVDataLoader:
    """Load and preprocess CCTV traffic data."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_path = config['data']['cctv_data_path']
        self.train_split = config['data']['train_split']
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load CCTV data and split into train/validation.
        
        Expected columns:
        - timestamp
        - uturn_count (vehicles attempting U-turn per minute)
        - straight_count (vehicles going straight per minute)
        - merged_count (vehicles after U-turn merge per minute)
        """
        df = pd.read_csv(self.data_path)
        
        # Validate required columns
        required_cols = ['uturn_count', 'straight_count', 'merged_count']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Shuffle data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split
        split_idx = int(len(df) * self.train_split)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        return train_df, val_df
    
    def get_training_batches(self, batch_size: int) -> List[np.ndarray]:
        """Generate training batches."""
        train_df, _ = self.load_data()
        
        # Create feature vectors
        train_data = train_df[['uturn_count', 'straight_count', 'merged_count']].values
        
        # Create batches
        batches = []
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            batches.append(batch)
        
        return batches
```

#### B.2 Error Calculator

```python
# src/calibration/error_calculator.py

class ErrorCalculator:
    """Calculate error metrics for calibration."""
    
    @staticmethod
    def mse(simulated: np.ndarray, actual: np.ndarray) -> float:
        """Mean Squared Error."""
        return np.mean((simulated - actual) ** 2)
    
    @staticmethod
    def rmse(simulated: np.ndarray, actual: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(ErrorCalculator.mse(simulated, actual))
    
    @staticmethod
    def mae(simulated: np.ndarray, actual: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(simulated - actual))
    
    @staticmethod
    def mape(simulated: np.ndarray, actual: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = actual != 0
        return np.mean(np.abs((simulated[mask] - actual[mask]) / actual[mask])) * 100
    
    @staticmethod
    def r2_score(simulated: np.ndarray, actual: np.ndarray) -> float:
        """R-squared (coefficient of determination)."""
        ss_res = np.sum((actual - simulated) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def calculate_all(simulated: pd.DataFrame, actual: pd.DataFrame) -> dict:
        """Calculate all error metrics."""
        metrics = {}
        for col in ['uturn_count', 'straight_count', 'merged_count']:
            if col in simulated.columns and col in actual.columns:
                sim = simulated[col].values
                act = actual[col].values
                
                min_len = min(len(sim), len(act))
                sim, act = sim[:min_len], act[:min_len]
                
                metrics[f'{col}_mse'] = ErrorCalculator.mse(sim, act)
                metrics[f'{col}_rmse'] = ErrorCalculator.rmse(sim, act)
                metrics[f'{col}_mae'] = ErrorCalculator.mae(sim, act)
                metrics[f'{col}_mape'] = ErrorCalculator.mape(sim, act)
        
        # Overall metrics
        metrics['overall_rmse'] = np.mean([metrics[f'{col}_rmse'] for col in ['uturn_count', 'straight_count', 'merged_count'] if f'{col}_rmse' in metrics])
        
        return metrics
```

---

### 2.C. DQN Agent Module

#### C.1 State Space Definition

| Feature | Description | Dimension |
|---------|-------------|-----------|
| Sim U-turn Count | Current U-turn volume | 1 |
| Sim Straight Count | Current straight volume | 1 |
| Sim Merged Count | Current merged volume | 1 |
| Aggressiveness | Current param value | 1 |
| Cooperativeness | Current param value | 1 |
| LC Threshold | Current param value | 1 |

**Total State Size:** 6 features × history (5) = 30

#### C.2 Action Space Definition

```
125 discrete actions (5 bins × 5 bins × 5 bins)

Action = Index 0-124
    |
    +-- aggressiveness: bins 0-4 (mapped to 0.1-1.0)
    +-- cooperativeness: bins 0-4 (mapped to 0.1-1.0)  
    +-- lane_change_threshold: bins 0-4 (mapped to 0.2-0.9)
```

#### C.3 Reward Function

```python
def compute_reward(self, simulated_counts: pd.DataFrame, cctv_counts: pd.DataFrame) -> float:
    """
    Reward = -MSE between simulated and CCTV vehicle counts
    
    Higher reward = better calibration
    """
    error_calc = ErrorCalculator()
    mse = error_calc.mse(
        simulated_counts.values,
        cctv_counts.values
    )
    
    # Negative MSE (lower error = higher reward)
    reward = -mse
    
    # Bonus for very low error
    if mse < 1.0:
        reward += 10.0
    elif mse < 5.0:
        reward += 5.0
    elif mse < 10.0:
        reward += 1.0
    
    return reward
```

---

### 2.D. Integration Layer

#### D.1 Main Orchestration

```python
# src/integration/orchestrator.py

class CalibrationOrchestrator:
    """Main orchestration for SUMO calibration with DQN."""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        
        # Initialize components
        self.simulation = SumoCalibrationSimulation(self.config['simulation'])
        self.data_loader = CCTVDataLoader(self.config['data'])
        self.error_calc = ErrorCalculator()
        
        # Load CCTV data
        self.train_data, self.val_data = self.data_loader.load_data()
        
        # Initialize DQN
        state_size = 6 * self.config['agent']['state_space']['history_length']
        action_size = self.config['agent']['action_space']['num_actions']
        self.agent = DQNAgent(state_size, action_size, self.config)
        
        # Logging
        self.logger = setup_logger('calibration')
        
    def train(self):
        """Main calibration training loop."""
        num_episodes = self.config['training']['num_episodes']
        
        best_error = float('inf')
        
        for episode in range(num_episodes):
            # Select parameter set (action) to test
            action = self.agent.select_action(
                self.simulation.reset(), 
                training=True
            )
            
            # Run simulation with these parameters
            self._run_episode(action)
            
            # Get simulated volumes
            sim_counts = self.simulation.get_simulated_volumes()
            
            # Compare with training data
            # For simplicity, compare mean values
            train_means = self.train_data[['uturn_count', 'straight_count', 'merged_count']].mean()
            
            # Compute reward
            reward = self._compute_reward(sim_counts, train_means)
            
            # Store experience
            state = self.simulation._get_state()
            next_state = state  # Episode ended
            self.agent.replay_buffer.push(state, action, reward, next_state, True)
            
            # Train
            if len(self.agent.replay_buffer) >= self.config['dqn']['batch_size']:
                loss = self.agent.train_step()
            
            # Logging
            error = -reward
            self.logger.info(f"Episode {episode}: Action={action}, Error={error:.4f}, Reward={reward:.4f}")
            
            # Validation
            if episode % self.config['training']['eval_freq'] == 0:
                val_error = self._validate()
                self.logger.info(f"Validation Error: {val_error:.4f}")
                
                if val_error < best_error:
                    best_error = val_error
                    self._save_checkpoint(episode, val_error)
            
            # Early stopping
            if best_error < self.config['training']['target_mse']:
                self.logger.info(f"Target MSE reached: {best_error}")
                break
    
    def _run_episode(self, action: int):
        """Run one full simulation episode."""
        self.simulation.reset()
        
        # Apply action (driver parameters)
        self.simulation.driver_controller.action_to_params(action)
        
        # Run simulation
        state = self.simulation._get_state()
        done = False
        
        while not done:
            # Just step through, no more action selection in episode
            state, _, done = self.simulation.step(action)
    
    def _compute_reward(self, sim_counts: pd.DataFrame, target_means: pd.Series) -> float:
        """Compute reward based on error."""
        # Calculate error between simulated and target
        sim_mean = sim_counts.mean()
        
        error = (
            (sim_mean.get('zone_a', 0) - target_means['uturn_count']) ** 2 +
            (sim_mean.get('straight', 0) - target_means['straight_count']) ** 2 +
            (sim_mean.get('zone_c', 0) - target_means['merged_count']) ** 2
        )
        
        return -error
    
    def _validate(self) -> float:
        """Validate on held-out data."""
        best_params = None
        best_error = float('inf')
        
        # Try all actions and find best
        for action in range(self.config['agent']['action_space']['num_actions']):
            self._run_episode(action)
            sim_counts = self.simulation.get_simulated_volumes()
            
            val_means = self.val_data[['uturn_count', 'straight_count', 'merged_count']].mean()
            error = self._compute_reward(sim_counts, val_means)
            
            if -error < best_error:
                best_error = -error
                best_params = action
        
        return best_error
```

---

## 3. IMPLEMENTATION PHASES

### Phase 1: Data Preparation & SUMO Setup

| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.1 | Receive CCTV data from user | Raw CCTV data file |
| 1.2 | Preprocess CCTV data | Cleaned CSV with required columns |
| 1.3 | Split data (70/30) | train.csv, val.csv |
| 1.4 | Verify SUMO installation | Working TraCI connection |
| 1.5 | Setup U-turn network | Network files ready |

### Phase 2: Simulation & Metrics Collection

| Task | Description | Deliverable |
|------|-------------|-------------|
| 2.1 | Implement VolumeCollector | Vehicle count per minute |
| 2.2 | Implement DriverBehaviorController | Parameter tuning |
| 2.3 | Implement SumoCalibrationSimulation | Full simulation wrapper |
| 2.4 | Test volume collection | Verify counts match expected |

### Phase 3: DQN Agent Implementation

| Task | Description | Deliverable |
|------|-------------|-------------|
| 3.1 | Define state/action space | Calibration-specific |
| 3.2 | Implement reward function | MSE-based reward |
| 3.3 | Build DQN network | Small network for fast training |
| 3.4 | Test with dummy data | Agent learns to minimize error |

### Phase 4: Integration & Training

| Task | Description | Deliverable |
|------|-------------|-------------|
| 4.1 | Connect SUMO to DQN | Full pipeline |
| 4.2 | Train on CCTV data | 200 episodes |
| 4.3 | Validate on held-out data | Validation error |
| 4.4 | Find optimal parameters | Best driver params |

### Phase 5: Testing & Validation

| Task | Description | Deliverable |
|------|-------------|-------------|
| 5.1 | Compare simulation vs CCTV | Error metrics |
| 5.2 | Analyze parameter sensitivity | Which params matter most |
| 5.3 | Final documentation | Results summary |

---

## 4. TECHNICAL SPECIFICATIONS

### 4.1 Data Format

**CCTV Data Expected Format:**
```csv
timestamp,uturn_count,straight_count,merged_count
2024-01-01 08:00:00,15,45,12
2024-01-01 08:01:00,18,42,15
...
```

**Training Split:**
- 70% training (3500 samples from 5000)
- 30% validation (1500 samples from 5000)

### 4.2 Error Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| MSE | Mean Squared Error | < 10 |
| RMSE | Root Mean Squared Error | < 3 |
| MAE | Mean Absolute Error | < 2 |
| R² | Coefficient of Determination | > 0.8 |

---

## 5. Clarifying Questions

### Answered:
1. ~~SUMO Installation~~ - User has SUMO installed
2. ~~Network Files~~ - User has existing network files (to be provided)
3. ~~Scenario~~ - U-turn road (NOT intersection)
4. ~~Parameters~~ - Driver behavior parameters
5. ~~Goal~~ - **Volume per minute matching (vehicle counts)**
6. ~~Data~~ - **CCTV data with 5000+ data points**
7. ~~Training approach~~ - **Supervised-style: compare simulation to real data**

### Remaining Questions:

8. **SUMO Installation Path**: What is your SUMO installation path?

9. **Network Files**: When can you provide the SUMO network files (.net.xml, .rou.xml, .sumocfg)?

10. **CCTV Data**: When can you provide the CCTV data file? What format is it in?

11. **GPU Availability**: Do you have a CUDA-capable GPU for training?

12. **Timeline**: What's your expected completion timeline?
