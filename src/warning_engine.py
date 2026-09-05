"""Hard-rule warning engine."""

from __future__ import annotations

import pandas as pd

from .risk_index import WARNING_ORDER, score_level


def max_level(a: str, b: str) -> str:
    return a if WARNING_ORDER[a] >= WARNING_ORDER[b] else b


def apply_warnings(df: pd.DataFrame) -> pd.DataFrame:
    """Apply score and hard-threshold warning rules."""

    levels: list[str] = []
    reasons: list[str] = []
    for row in df.itertuples(index=False):
        level = score_level(float(row.llsri))
        reason: list[str] = []
        if row.soc < 0.20:
            level = max_level(level, "黄色")
            reason.append("LOW_BATTERY")
        if row.soc < 0.10:
            level = max_level(level, "红色")
            reason.append("CRITICAL_BATTERY")
        if row.path_error > 20:
            level = max_level(level, "黄色")
            reason.append("PATH_DEVIATION")
        if row.path_error > 40:
            level = max_level(level, "红色")
            reason.append("SEVERE_PATH_DEVIATION")
        if row.nearest_building_distance < 10:
            level = max_level(level, "橙色")
            reason.append("BUILDING_PROXIMITY")
        if row.nearest_drone_distance < 30:
            level = max_level(level, "黄色")
            reason.append("AIRSPACE_CONFLICT")
        if row.nearest_drone_distance < 15:
            level = max_level(level, "红色")
            reason.append("COLLISION_RISK")
        if min(row.nearest_school_distance, row.nearest_hospital_distance) < 300:
            level = max_level(level, "黄色")
            reason.append("SENSITIVE_AREA_FLYOVER")
        if min(row.nearest_school_distance, row.nearest_hospital_distance) < 300 and row.altitude < 50:
            level = max_level(level, "橙色")
            reason.append("LOW_ALTITUDE_SENSITIVE_AREA")
        if row.ground_impact_risk > 0.35 and row.population_exposure_risk > 0.70:
            level = max_level(level, "红色")
            reason.append("HIGH_GROUND_IMPACT_RISK")
        if row.compliance_risk > 0.82:
            level = max_level(level, "红色")
            reason.append("COMPLIANCE_RESTRICTED_ZONE")
        levels.append(level)
        reasons.append(";".join(reason) if reason else "LLSRI_SCORE")
    df["warning_level"] = levels
    df["warning_reason"] = reasons
    return df

