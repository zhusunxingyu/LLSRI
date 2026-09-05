"""Experiment and route generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .map_loader import sample_point_in_zone
from .map_loader import to_xy


ALTITUDES = [30, 60, 90, 120]
WIND_DIRECTIONS = ["tailwind", "headwind", "crosswind", "random"]
GUSTS = ["none", "weak", "medium", "strong"]
COMPLEXITIES = ["low", "medium", "high"]
DRONE_COUNTS = [1, 2, 5, 10, 20]
TIME_PERIODS = ["day", "evening", "night"]


def generate_task(experiment_id: int, runs: int, zones: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Generate one task; first half are special experiments, second half Monte Carlo."""

    experiment_type = "special" if experiment_id <= min(1000, runs // 2 if runs > 1000 else runs) else "monte_carlo"
    if experiment_type == "special":
        zone = zones.iloc[(experiment_id - 1) % len(zones)]
        altitude = ALTITUDES[((experiment_id - 1) // len(zones)) % len(ALTITUDES)]
        drone_count = DRONE_COUNTS[((experiment_id - 1) // 11) % len(DRONE_COUNTS)]
    else:
        zone = zones.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        altitude = float(rng.choice(ALTITUDES))
        drone_count = int(rng.choice(DRONE_COUNTS))
    start = sample_point_in_zone(zone, rng)
    end = sample_point_in_zone(zone, rng)
    sx, sy = to_xy(start[0], start[1])
    ex, ey = to_xy(end[0], end[1])
    actual_distance = float(np.hypot(ex - sx, ey - sy))
    return {
        "experiment_id": experiment_id,
        "experiment_type": experiment_type,
        "random_seed": int(rng.integers(1, 2**31 - 1)),
        "city_zone": zone["name"],
        "route_type": str(rng.choice(["direct", "corridor", "detour", "river_crossing"])),
        "start_lon": start[0],
        "start_lat": start[1],
        "end_lon": end[0],
        "end_lat": end[1],
        "initial_soc": float(rng.uniform(0.20, 1.00)),
        "battery_soh": float(rng.uniform(0.55, 1.00)),
        "payload_rate": float(rng.uniform(0, 1)),
        "wind_speed": float(rng.uniform(0, 12)),
        "wind_direction": str(rng.choice(WIND_DIRECTIONS)),
        "gust_level": str(rng.choice(GUSTS)),
        "gps_error": float(rng.uniform(0, 10)),
        "altitude": altitude,
        "route_distance": max(500.0, actual_distance),
        "route_complexity": str(rng.choice(COMPLEXITIES)),
        "drone_count": drone_count,
        "time_period": str(rng.choice(TIME_PERIODS)),
    }
