"""Risk formula implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ahp_ai_weighting import apply_weighted_llsri, calculate_fusion_weights


WARNING_ORDER = {"正常": 0, "蓝色": 1, "黄色": 2, "橙色": 3, "红色": 4}

SELECTED_AHP_FIRST_LEVEL_WEIGHTS = {
    "environment_risk": 0.1127,
    "drone_health_risk": 0.3159,
    "mission_risk": 0.2244,
    "airspace_risk": 0.1689,
    "social_risk": 0.0567,
    "compliance_risk": 0.1214,
}

AHP_SECOND_LEVEL_WEIGHTS = {
    "environment_risk": {
        "wind_speed_risk": 0.4008,
        "gust_level_risk": 0.2613,
        "visibility_risk": 0.1854,
        "gps_error_risk": 0.1034,
        "temperature_extreme_risk": 0.0492,
    },
    "drone_health_risk": {
        "battery_soc_risk": 0.4680,
        "battery_soh_risk": 0.1711,
        "payload_rate_risk": 0.0749,
        "power_margin_risk": 0.2860,
    },
    "mission_risk": {
        "route_distance_risk": 0.1055,
        "route_complexity_risk": 0.2472,
        "cruise_speed_risk": 0.0584,
        "altitude_margin_risk": 0.1708,
        "phase_risk": 0.4180,
    },
    "airspace_risk": {
        "drone_density_risk": 0.4056,
        "separation_margin_risk": 0.3373,
        "building_density_risk": 0.1115,
        "obstacle_collision_risk": 0.1457,
    },
    "social_risk": {
        "population_exposure_risk": 0.5268,
        "ground_casualty_risk": 0.2862,
        "property_damage_risk": 0.1286,
        "public_nuisance_risk": 0.0584,
    },
    "compliance_risk": {
        "altitude_compliance_risk": 0.2914,
        "zone_compliance_risk": 0.4730,
        "flight_permit_risk": 0.1698,
        "data_compliance_risk": 0.0658,
    },
}


def clip01(values):
    return np.clip(values, 0, 1)


def weighted_sum(parts: dict[str, pd.Series | np.ndarray | float], weights: dict[str, float]):
    return clip01(sum(parts[name] * weights[name] for name in weights))


def zone_population_score(zone_name: str) -> float:
    zone = str(zone_name)
    if any(key in zone for key in ["hospital_school", "医院", "学校", "transport", "交通", "core_business", "商务", "commercial"]):
        return 1.0
    if any(key in zone for key in ["residential", "居住", "居民"]):
        return 0.8
    if any(key in zone for key in ["industrial", "工业", "产业"]):
        return 0.4
    if any(key in zone for key in ["waterfront", "滨水", "跨江", "江岸"]):
        return 0.2
    if any(key in zone for key in ["suburban", "郊区"]):
        return 0.2
    return 0.6


def calculate_risks(df: pd.DataFrame, task: dict) -> pd.DataFrame:
    """Calculate normalized risk metrics and LLSRI."""

    altitude = task["altitude"]
    wind = task["wind_speed"]
    gps = task["gps_error"]
    gust_factor = {"none": 0.0, "weak": 0.08, "medium": 0.18, "strong": 0.30}[task["gust_level"]]
    visibility_raw = df["visibility"] if "visibility" in df else 10000
    visibility = visibility_raw * 10000 if np.nanmax(np.asarray(visibility_raw)) <= 1.5 else visibility_raw
    temperature = task.get("temperature", df["temperature"] if "temperature" in df else 25.0)
    temperature_extreme = clip01(np.abs(temperature - 25.0) / 35.0)
    population = pd.Series(zone_population_score(task.get("city_zone", "")), index=df.index)
    sensitive = clip01(0.50 * np.exp(-np.minimum(df.nearest_school_distance, df.nearest_hospital_distance) / 300) + 0.50 * np.minimum(df.sensitive_poi_count_300m / 5, 1))
    crash_probability = clip01(0.04 + 0.20 * (wind / 12) + 0.16 * (gps / 10) + 0.16 * (1 - task["battery_soh"]) + 0.12 * (task["payload_rate"]))
    ground_impact = clip01(crash_probability * population * (0.65 + 0.35 * task["payload_rate"]))
    noise = clip01((130 - altitude) / 120 * (0.45 + 0.55 * df.residential_exposure))
    privacy = clip01((90 - altitude) / 90 * df.residential_exposure)
    obstacle = clip01(0.55 * df.building_density_300m + 0.30 * (1 - df.nearest_building_distance / 120) + 0.15 * (gps / 15))

    environment_parts = {
        "wind_speed_risk": clip01(wind / 20),
        "gust_level_risk": clip01({"none": 0, "weak": 1, "medium": 3, "strong": 5}[task["gust_level"]] / 5),
        "visibility_risk": clip01((10000 - visibility) / 10000),
        "gps_error_risk": clip01(gps / 15),
        "temperature_extreme_risk": temperature_extreme,
    }
    health_parts = {
        "battery_soc_risk": clip01(1 - df.soc),
        "battery_soh_risk": clip01(1 - task["battery_soh"]),
        "payload_rate_risk": clip01(task["payload_rate"]),
        "power_margin_risk": clip01(1 - (df.soc - 0.18) / 0.82),
    }
    complexity = {"low": 0.20, "medium": 0.50, "high": 0.78}[task["route_complexity"]]
    phase_map = {"TAKEOFF": 0.85, "LANDING": 0.85, "HOVER": 0.55, "CRUISE_OUTBOUND": 0.28, "CRUISE_RETURN": 0.28}
    phase_risk = df["mission_phase"].map(phase_map).fillna(0.40) if "mission_phase" in df else 0.40
    mission_parts = {
        "route_distance_risk": clip01(task["route_distance"] / 15000),
        "route_complexity_risk": pd.Series(complexity, index=df.index),
        "cruise_speed_risk": clip01(float(task.get("cruise_speed", df["speed"].mean())) / 30),
        "altitude_margin_risk": clip01(altitude / float(task.get("max_altitude", 120))),
        "phase_risk": phase_risk,
    }
    airspace_parts = {
        "drone_density_risk": clip01((task["drone_count"] - 1) / 49),
        "separation_margin_risk": clip01((200 - df.nearest_drone_distance) / 200),
        "building_density_risk": clip01(df.building_density_300m),
        "obstacle_collision_risk": obstacle,
    }
    social_parts = {
        "population_exposure_risk": population,
        "ground_casualty_risk": clip01(population * task["payload_rate"]),
        "property_damage_risk": population,
        "public_nuisance_risk": clip01(0.50 * noise + 0.50 * privacy),
    }
    zone_compliance_base = clip01(1 - df.compliance_risk)
    compliance_parts = {
        "altitude_compliance_risk": clip01(np.maximum(0, altitude - 120) / 80),
        "zone_compliance_risk": clip01(1 - zone_compliance_base),
        "flight_permit_risk": clip01(df.compliance_risk),
        "data_compliance_risk": pd.Series(0.20, index=df.index),
    }

    environment = weighted_sum(environment_parts, AHP_SECOND_LEVEL_WEIGHTS["environment_risk"])
    health = weighted_sum(health_parts, AHP_SECOND_LEVEL_WEIGHTS["drone_health_risk"])
    mission = weighted_sum(mission_parts, AHP_SECOND_LEVEL_WEIGHTS["mission_risk"])
    airspace = weighted_sum(airspace_parts, AHP_SECOND_LEVEL_WEIGHTS["airspace_risk"])
    social = weighted_sum(social_parts, AHP_SECOND_LEVEL_WEIGHTS["social_risk"])
    compliance = weighted_sum(compliance_parts, AHP_SECOND_LEVEL_WEIGHTS["compliance_risk"])
    weights = task.get("risk_weights") or SELECTED_AHP_FIRST_LEVEL_WEIGHTS

    df["population_exposure_risk"] = population
    df["sensitive_area_risk"] = sensitive
    df["crash_probability"] = crash_probability
    df["ground_impact_risk"] = ground_impact
    df["noise_risk"] = noise
    df["privacy_risk"] = privacy
    for risk_parts in [environment_parts, health_parts, mission_parts, airspace_parts, social_parts, compliance_parts]:
        for name, values in risk_parts.items():
            df[name] = values
    df["environment_risk"] = environment
    df["drone_health_risk"] = health
    df["mission_risk"] = mission
    df["airspace_risk"] = airspace
    df["social_risk"] = social
    df["compliance_risk"] = compliance
    df["llsri"] = apply_weighted_llsri(df, weights)
    return df


def score_level(score: float) -> str:
    if score >= 0.80:
        return "红色"
    if score >= 0.60:
        return "橙色"
    if score >= 0.40:
        return "黄色"
    if score >= 0.20:
        return "蓝色"
    return "正常"
