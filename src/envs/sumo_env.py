import os
import subprocess
import tempfile
from typing import Dict, List, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci


ROU_XML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <flow id="f_sl" begin="0.00" from="675098743#0" to="675098743#2" end="3600.00" vehsPerHour="{sl}"/>
    <flow id="f_spt" begin="0.00" from="675098743#0" to="585303113#2" end="3600.00" vehsPerHour="{spt}"/>
    <flow id="f_ul" begin="0.00" color="50,255,0" from="585303113#0" to="585303113#2" end="3600.00" vehsPerHour="{ul}"/>
    <flow id="f_upt" begin="0.00" color="255,53,0" from="585303113#0" to="675098743#2" end="3600.00" vehsPerHour="{upt}"/>
</routes>"""


class SUMOEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        sumo_config_path: str,
        sumo_binary: str = "sumo",
        step_length: float = 0.05,
        warmup_minutes: int = 5,
        period_minutes: int = 5,
        north_loop_ids: Optional[List[str]] = None,
        south_loop_ids: Optional[List[str]] = None,
        action_map: Optional[List[Dict[str, float]]] = None,
    ):
        super().__init__()

        self.sumo_config_path = sumo_config_path
        self.sumo_binary = sumo_binary
        self.step_length = step_length
        self.minute_steps = int(60.0 / step_length)
        self.warmup_minutes = warmup_minutes
        self.warmup_steps = self.minute_steps * self.warmup_minutes
        self.period_minutes = period_minutes
        self.period_steps = self.minute_steps * self.period_minutes
        self.steps_per_data = 4

        self.action_map = action_map or [
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
        self.n_actions = len(self.action_map)
        self.action_space = spaces.Discrete(self.n_actions)

        self.north_loop_ids = north_loop_ids or [
            "total_north_left", "total_north_right"
        ]
        self.south_loop_ids = south_loop_ids or [
            "total_south_left", "total_south_right"
        ]

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4,), dtype=np.float32
        )

        self.target_data: Optional[np.ndarray] = None
        self.current_data_idx = 0
        self.step_idx = 0
        self.current_params = {"lcCooperative": 1.0, "lcAssertive": 1.0}
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._sim_active = False

    def set_target_data(self, data: np.ndarray):
        self.target_data = data

    def reset_data_pointer(self):
        self.current_data_idx = 0

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

    def step(self, action: int):
        delta = self.action_map[action]
        coop = self.current_params["lcCooperative"] + delta["lcCooperative"]
        assertive = self.current_params["lcAssertive"] + delta["lcAssertive"]
        self.current_params["lcCooperative"] = max(0.0, min(5.0, coop))
        self.current_params["lcAssertive"] = max(0.0, min(5.0, assertive))

        self._apply_params()

        sim_south = 0
        sim_north = 0
        for _ in range(self.period_steps):
            traci.simulationStep()
            for lid in self.north_loop_ids:
                sim_north += traci.inductionloop.getLastStepVehicleNumber(lid)
            for lid in self.south_loop_ids:
                sim_south += traci.inductionloop.getLastStepVehicleNumber(lid)

        expected_south = self._current_row[4] / 12.0
        expected_north = self._current_row[5] / 12.0
        mape = (abs(sim_south - expected_south) / expected_south +
                abs(sim_north - expected_north) / expected_north) / 2.0
        reward = -mape

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

    def close(self):
        self._stop_sumo()

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

    @staticmethod
    def _force_kill_sumo():
        for exe in ("sumo.exe", "sumo-gui.exe", "sumo", "sumo-gui"):
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/IM", exe],
                    capture_output=True, timeout=5,
                )
            except Exception:
                pass
        try:
            subprocess.run(
                ["powershell", "-Command",
                 "Get-Process sumo,sumo-gui -ErrorAction SilentlyContinue | Stop-Process -Force"],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
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

    def _start_sumo(self, sl, spt, ul, upt):
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

    def _run_warmup(self):
        traci.vehicletype.setParameter(
            "DEFAULT_VEHTYPE", "laneChangeModel.lcCooperative", "1.0"
        )
        traci.vehicletype.setParameter(
            "DEFAULT_VEHTYPE", "laneChangeModel.lcAssertive", "1.0"
        )
        for _ in range(self.warmup_steps):
            traci.simulationStep()

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
