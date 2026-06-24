import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingLogger:
    def __init__(self, log_dir: str = "logs", hyperparams: Optional[Dict[str, Any]] = None):
        start_time = datetime.now()
        folder_name = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self._run_dir = Path(log_dir) / folder_name
        os.makedirs(self._run_dir, exist_ok=True)

        if hyperparams:
            config_path = self._run_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump(hyperparams, f, indent=2, default=str)

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

        self._summary: list = []
        self._iter_start_step: int = 0

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def log_step(
        self,
        iteration: int,
        data_idx: int,
        step_in_data: int,
        reward: float,
        sim_south: float,
        sim_north: float,
        expected_south: float,
        expected_north: float,
        lcCooperative: float,
        lcAssertive: float,
        epsilon: float,
        loss: Optional[float] = None,
    ):
        self._csv_writer.writerow([
            iteration, data_idx, step_in_data,
            f"{reward:.3f}",
            f"{sim_south:.0f}", f"{sim_north:.0f}",
            f"{expected_south:.0f}", f"{expected_north:.0f}",
            f"{sim_south - expected_south:.3f}",
            f"{sim_north - expected_north:.3f}",
            f"{lcCooperative:.3f}", f"{lcAssertive:.3f}",
            f"{epsilon:.4f}",
            f"{loss:.6f}" if loss is not None else "",
        ])
        self._csv_file.flush()

    def log_iteration(
        self,
        iteration: int,
        total_reward: float,
        avg_reward: float,
        avg_loss: Optional[float],
        epsilon: float,
        total_steps: int,
    ):
        entry = {
            "iteration": int(iteration),
            "total_reward": float(round(total_reward, 3)),
            "avg_reward": float(round(avg_reward, 3)),
            "avg_loss": float(round(avg_loss, 6)) if avg_loss is not None else None,
            "epsilon": float(round(epsilon, 4)),
            "total_steps": int(total_steps),
        }
        self._summary.append(entry)

        summary_path = self._run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(self._summary, f, indent=2)

    def close(self):
        self._csv_file.close()


__all__ = ["TrainingLogger"]
