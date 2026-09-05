"""Second-level drone simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .map_feature_extractor import extract_features
from .map_loader import interpolate_route
from .risk_index import calculate_risks
from .warning_engine import apply_warnings
from .ai_predictor import add_ai_outputs


def _with_task_defaults(task: dict) -> dict:
    """Fill live web route tasks with the fields used by simulation and risk logic."""

    filled = dict(task)
    filled.setdefault("experiment_id", filled.get("task_id", 0))
    filled.setdefault("task_id", filled.get("experiment_id", 0))
    filled.setdefault("city_zone", "core_business")
    filled.setdefault("route_type", "direct")
    filled.setdefault("route_complexity", "medium")
    filled.setdefault("altitude", 90)
    filled.setdefault("cruise_speed", 10.0)
    filled.setdefault("route_distance", 1000.0)
    filled.setdefault("wind_speed", 3.0)
    filled.setdefault("wind_direction", "E")
    filled.setdefault("gust_level", "weak")
    filled.setdefault("time_period", "day")
    filled.setdefault("gps_error", 2.0)
    filled.setdefault("initial_soc", 0.86)
    filled.setdefault("battery_soh", 0.90)
    filled.setdefault("payload_rate", 0.35)
    filled.setdefault("drone_count", 3)
    filled.setdefault("hover_seconds", 10)
    filled.setdefault("experiment_type", "user_prediction")
    filled.setdefault("route_distance_band", "user_defined")
    filled.setdefault("temperature", 25.0)
    filled.setdefault("max_altitude", 120)
    return filled


def estimate_mission_seconds(task: dict, max_time_seconds: int | None = None) -> int:
    """Estimate real mission duration from distance, speed, hover, and return leg."""

    cruise_speed = max(3.0, float(task.get("cruise_speed", 8.0)))
    route_distance = max(1.0, float(task.get("route_distance", 1000.0)))
    route_factor = {
        "direct": 1.05,
        "corridor": 1.22,
        "detour": 1.45,
        "river_crossing": 1.32,
    }.get(str(task.get("route_type", "direct")), 1.18)
    complexity_factor = {
        "low": 1.00,
        "medium": 1.12,
        "high": 1.25,
    }.get(str(task.get("route_complexity", "medium")), 1.12)
    hover_seconds = int(task.get("hover_seconds", 10))
    takeoff_landing_buffer = 50
    # SRS task is out-and-back: outbound delivery, hover, return, landing.
    seconds = (2 * route_distance * route_factor * complexity_factor) / cruise_speed
    seconds += hover_seconds + takeoff_landing_buffer
    seconds = max(60, round(seconds))
    if max_time_seconds is not None and max_time_seconds > 0:
        seconds = min(seconds, max_time_seconds)
    return int(seconds)


def _fit_length(frame: pd.DataFrame, length: int) -> pd.DataFrame:
    """Pad or trim a route frame to an exact length."""

    if len(frame) == length:
        return frame.reset_index(drop=True)
    if len(frame) > length:
        return frame.iloc[:length].reset_index(drop=True)
    if frame.empty:
        raise ValueError("route segment cannot be empty")
    pad = pd.concat([frame.iloc[[-1]]] * (length - len(frame)), ignore_index=True)
    return pd.concat([frame, pad], ignore_index=True)


def build_mission_route(task: dict, seconds: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Build SRS-style outbound, hover, return, and landing mission route."""

    takeoff_n = max(6, int(seconds * 0.12))
    hover_n = max(8, int(task.get("hover_seconds", min(20, seconds * 0.10))))
    landing_n = max(6, int(seconds * 0.12))
    cruise_total = max(10, seconds - takeoff_n - hover_n - landing_n)
    outbound_n = max(5, cruise_total // 2)
    return_n = max(5, cruise_total - outbound_n)

    start = (task["start_lon"], task["start_lat"])
    end = (task["end_lon"], task["end_lat"])
    outbound = interpolate_route(start, end, takeoff_n + outbound_n, task["route_type"], task["route_complexity"], rng)
    hover = pd.concat([outbound.iloc[[-1]]] * hover_n, ignore_index=True)
    inbound = interpolate_route(end, start, return_n + landing_n, task["route_type"], task["route_complexity"], rng)
    route = pd.concat([outbound, hover, inbound], ignore_index=True)
    route = _fit_length(route, seconds)

    phases = (
        ["TAKEOFF"] * takeoff_n
        + ["CRUISE_OUTBOUND"] * outbound_n
        + ["HOVER"] * hover_n
        + ["CRUISE_RETURN"] * return_n
        + ["LANDING"] * landing_n
    )
    if len(phases) < seconds:
        phases.extend(["LANDING"] * (seconds - len(phases)))
    phases = np.array(phases[:seconds], dtype=object)
    return route, phases


def simulate_task(task: dict, zone: pd.Series, seconds: int, rng: np.random.Generator, buildings_cache=None, ai_model_dir=None) -> pd.DataFrame:
    """Simulate one drone task at one-second resolution."""

    task = _with_task_defaults(task)
    route, phase = build_mission_route(task, seconds, rng)
    t = np.arange(seconds)
    takeoff_mask = phase == "TAKEOFF"
    landing_mask = phase == "LANDING"
    altitude_curve = np.full(seconds, float(task["altitude"]))
    if takeoff_mask.any():
        altitude_curve[takeoff_mask] = np.linspace(0, task["altitude"], int(takeoff_mask.sum()))
    if landing_mask.any():
        altitude_curve[landing_mask] = np.linspace(task["altitude"], 0, int(landing_mask.sum()))
    base_speed = float(task.get("cruise_speed", 8 + task["wind_speed"] * 0.35))
    speed = np.clip(base_speed + task["wind_speed"] * 0.08 + rng.normal(0, 0.8, seconds), 2, 24)
    speed = np.where(phase == "HOVER", np.clip(rng.normal(0.4, 0.15, seconds), 0, 1.2), speed)
    speed = np.where((phase == "TAKEOFF") | (phase == "LANDING"), np.minimum(speed, base_speed * 0.65), speed)
    soc_drop = np.linspace(0, 0.05 + 0.18 * task["payload_rate"] + 0.10 * task["route_distance"] / 3000, seconds)
    df = route.copy()
    df["experiment_id"] = task["experiment_id"]
    df["timestamp"] = t
    df["city_zone"] = task["city_zone"]
    df["route_type"] = task["route_type"]
    df["route_distance"] = task.get("route_distance", 0)
    df["route_complexity"] = task.get("route_complexity", "medium")
    df["drone_count"] = task.get("drone_count", 1)
    df["payload_rate"] = task.get("payload_rate", 0.0)
    df["hover_seconds"] = task.get("hover_seconds", 10)
    df["task_experiment_type"] = task.get("experiment_type", "user_prediction")
    df["task_start_lon"] = task.get("start_lon", np.nan)
    df["task_start_lat"] = task.get("start_lat", np.nan)
    df["task_end_lon"] = task.get("end_lon", np.nan)
    df["task_end_lat"] = task.get("end_lat", np.nan)
    df["task_initial_soc"] = task.get("initial_soc", 0.86)
    df["task_payload_rate"] = task.get("payload_rate", 0.0)
    df["task_route_distance"] = task.get("route_distance", 0)
    df["task_route_complexity"] = task.get("route_complexity", "medium")
    df["task_drone_count"] = task.get("drone_count", 1)
    df["task_route_distance_band"] = task.get("route_distance_band", "user_defined")
    df["task_cruise_speed"] = task.get("cruise_speed", 8.0)
    df["task_hover_seconds"] = task.get("hover_seconds", 10)
    df["z"] = altitude_curve
    df["altitude"] = task["altitude"]
    df["speed"] = speed
    df["mission_phase"] = phase
    df["vx"] = np.gradient(df["x"].to_numpy())
    df["vy"] = np.gradient(df["y"].to_numpy())
    df["vz"] = np.gradient(df["z"].to_numpy())
    df["yaw"] = np.degrees(np.arctan2(df["vy"], df["vx"])).round(3)
    df["wind_speed"] = task["wind_speed"]
    df["wind_direction"] = task["wind_direction"]
    df["gust_level"] = task["gust_level"]
    df["visibility"] = np.clip(1 - task["wind_speed"] / 30 + rng.normal(0, 0.03, seconds), 0.35, 1)
    df["time_period"] = task["time_period"]
    df["soc"] = np.clip(task["initial_soc"] - soc_drop, 0, 1)
    df["battery_soh"] = task["battery_soh"]
    df["voltage"] = 44.4 * df["soc"] * (0.88 + 0.12 * task["battery_soh"])
    df["battery_voltage"] = df["voltage"]
    power = 180 + 2.6 * (df["speed"] ** 2) + 95 * task["payload_rate"] + 4.2 * (task["wind_speed"] ** 2) + 40 * np.maximum(df["vz"], 0)
    df["energy_consumption"] = power.cumsum()
    df["roll"] = rng.normal(0, 3 + task["wind_speed"] * 0.45, seconds)
    df["pitch"] = rng.normal(0, 2.5 + task["wind_speed"] * 0.35, seconds)
    df["gps_error"] = task["gps_error"]
    df["path_error"] = np.clip(task["gps_error"] * 1.8 + task["wind_speed"] * 1.2 + rng.normal(0, 4, seconds), 0, 80)
    df["obstacle_distance"] = np.nan
    features = extract_features(df, zone, task["altitude"], task["gps_error"], rng, buildings_cache=buildings_cache)
    df = pd.concat([df.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    df["obstacle_distance"] = df["nearest_building_distance"]
    count_pressure = task["drone_count"] / 20
    df["nearest_drone_distance"] = np.clip(260 - 230 * count_pressure + rng.normal(0, 35, seconds), 8, 500)
    df["conflict_count"] = np.clip(np.rint(count_pressure * 4 + rng.normal(0, 0.9, seconds)), 0, 8)
    df["near_miss_count"] = (df["nearest_drone_distance"] < 30).astype(int)
    df = calculate_risks(df, task)
    df = apply_warnings(df)
    df = add_ai_outputs(df, model_dir=ai_model_dir)
    df["mission_success"] = ~df["warning_level"].eq("红色").rolling(8, min_periods=1).sum().gt(5)
    df["failure_type"] = np.where(df["mission_success"], "", "risk_abort")
    return df


def summarize_task(task: dict, ts: pd.DataFrame) -> dict:
    """Summarize second-level records into one task row."""

    final_level = max(ts["warning_level"], key=lambda x: {"正常": 0, "蓝色": 1, "黄色": 2, "橙色": 3, "红色": 4}[x])
    return task | {
        "flight_time": int(ts.timestamp.max() + 1),
        "distance_flown": float(np.sqrt(ts.x.diff().fillna(0) ** 2 + ts.y.diff().fillna(0) ** 2 + ts.z.diff().fillna(0) ** 2).sum()),
        "average_speed": ts.speed.mean(),
        "average_path_error": ts.path_error.mean(),
        "max_path_error": ts.path_error.max(),
        "path_violation_count": int((ts.path_error > 20).sum()),
        "total_energy_consumption": ts.energy_consumption.max(),
        "final_soc": ts.soc.iloc[-1],
        "minimum_soc": ts.soc.min(),
        "minimum_obstacle_distance": ts.nearest_building_distance.min(),
        "minimum_drone_distance": ts.nearest_drone_distance.min(),
        "warning_count": int(ts.warning_level.ne("正常").sum()),
        "maximum_warning_level": final_level,
        "avg_environment_risk": ts.environment_risk.mean(),
        "max_environment_risk": ts.environment_risk.max(),
        "avg_drone_health_risk": ts.drone_health_risk.mean(),
        "max_drone_health_risk": ts.drone_health_risk.max(),
        "avg_mission_risk": ts.mission_risk.mean(),
        "max_mission_risk": ts.mission_risk.max(),
        "avg_airspace_risk": ts.airspace_risk.mean(),
        "max_airspace_risk": ts.airspace_risk.max(),
        "avg_social_risk": ts.social_risk.mean(),
        "max_social_risk": ts.social_risk.max(),
        "avg_compliance_risk": ts.compliance_risk.mean(),
        "max_compliance_risk": ts.compliance_risk.max(),
        "avg_llsri": ts.llsri.mean(),
        "max_llsri": ts.llsri.max(),
        "final_warning_level": final_level,
        "red_warning_count": int((ts.warning_level == "红色").sum()),
        "orange_warning_count": int((ts.warning_level == "橙色").sum()),
        "social_warning_count": int(ts.warning_reason.str.contains("SENSITIVE|GROUND|SOCIAL", regex=True).sum()),
        "compliance_warning_count": int(ts.warning_reason.str.contains("COMPLIANCE").sum()),
        "min_building_distance": ts.nearest_building_distance.min(),
        "max_building_density_300m": ts.building_density_300m.max(),
        "min_school_distance": ts.nearest_school_distance.min(),
        "min_hospital_distance": ts.nearest_hospital_distance.min(),
        "sensitive_area_flyover_count": int((ts.sensitive_area_risk > 0.55).sum()),
        "high_population_exposure_time": int((ts.population_exposure_risk > 0.70).sum()),
        "river_crossing_count": int(ts.river_crossing.sum()),
        "collision": bool((ts.nearest_drone_distance < 15).any()),
        "return_to_home": bool((ts.soc < 0.18).any()),
        "emergency_landing": bool((ts.soc < 0.10).any() or (ts.warning_level == "红色").sum() > 20),
        "mission_success": bool(ts.mission_success.iloc[-1]),
        "failure_type": "" if bool(ts.mission_success.iloc[-1]) else "risk_abort",
    }
