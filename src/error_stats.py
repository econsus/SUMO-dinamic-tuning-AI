"""Helpers for per-data-point z-score reward normalization."""

import csv
import os
from typing import Dict, Tuple


def load_error_stats(csv_path: str) -> Dict[Tuple[int, int, int, int], Tuple[float, float]]:
    """Load error_stats.csv into {(sl, spt, ul, upt): (err_mean, err_std)}."""
    stats = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["sl"]), int(row["spt"]), int(row["ul"]), int(row["upt"]))
            stats[key] = (float(row["err_mean"]), float(row["err_std"]))
    return stats


def mape_from_info(info: dict) -> float:
    """MAPE from env info (period-scale independent, absolute error)."""
    sim_south = float(info["sim_south"])
    sim_north = float(info["sim_north"])
    expected_south = float(info["expected_south"])
    expected_north = float(info["expected_north"])
    return (abs(sim_south - expected_south) / expected_south +
            abs(sim_north - expected_north) / expected_north) / 2.0


def normalize_reward(mape: float, mean: float, std: float) -> float:
    """Z-score reward: (mean - mape) / std. Positive when error is below the
    data point's average error from the parameter impact test."""
    return (mean - mape) / std


def default_stats_path() -> str:
    return os.path.join("testing", "error_stats.csv")


__all__ = ["load_error_stats", "mape_from_info", "normalize_reward", "default_stats_path"]
