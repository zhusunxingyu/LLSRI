"""In-flight risk forecasting and active avoidance decision support."""

from __future__ import annotations

import pandas as pd

from .llm_agent import explain_risk


def forecast_flight_risk(ts: pd.DataFrame, current_second: int, horizon_seconds: int = 300) -> dict:
    """Forecast risk over a rolling future flight segment."""

    future = ts.loc[ts["timestamp"].between(current_second, current_second + horizon_seconds)].copy()
    if future.empty:
        future = ts.tail(1).copy()
    worst = future.sort_values(["ai_future_30s_risk", "llsri"], ascending=False).iloc[0]
    return {
        "horizon_seconds": int(horizon_seconds),
        "window_rows": int(len(future)),
        "max_predicted_risk": float(future["ai_future_30s_risk"].max()),
        "avg_predicted_risk": float(future["ai_future_30s_risk"].mean()),
        "max_current_llsri": float(future["llsri"].max()),
        "red_predicted_seconds": int((future["ai_future_30s_level"] == "红色").sum()),
        "orange_predicted_seconds": int((future["ai_future_30s_level"] == "橙色").sum()),
        "worst_timestamp": int(worst["timestamp"]),
        "worst_level": str(worst["ai_future_30s_level"]),
        "worst_reason": str(worst["warning_reason"]),
        "worst_row": worst,
    }


def recommend_avoidance_action(current: pd.Series, forecast: dict) -> dict:
    """Recommend active avoidance actions from current state and future forecast."""

    actions: list[str] = []
    triggers: list[str] = []
    severity = "保持监视"
    if forecast["red_predicted_seconds"] > 0 or forecast["max_predicted_risk"] >= 0.80:
        severity = "立即避险"
        triggers.append("后续航段预测红色风险")
    elif forecast["orange_predicted_seconds"] > 0 or forecast["max_predicted_risk"] >= 0.60:
        severity = "建议避险"
        triggers.append("后续航段预测橙色风险")

    if float(current.get("nearest_building_distance", 999)) < 20:
        actions.extend(["提高高度至安全高度层", "切换安全走廊"])
        triggers.append("建筑物接近")
    if float(current.get("nearest_drone_distance", 999)) < 30:
        actions.extend(["临时悬停等待", "切换高度层", "错峰通过冲突点"])
        triggers.append("多机冲突")
    if float(current.get("sensitive_area_risk", 0)) > 0.60 and float(current.get("altitude", 0)) < 90:
        actions.extend(["提高高度", "绕开敏感区缓冲范围"])
        triggers.append("敏感区低空暴露")
    if float(current.get("soc", 1)) < 0.20:
        actions.append("返航或就近备降")
        triggers.append("电量不足")
    if float(current.get("path_error", 0)) > 25:
        actions.extend(["降低速度", "重新入航"])
        triggers.append("航迹偏差过大")
    if float(current.get("wind_speed", 0)) >= 9:
        actions.extend(["降低巡航速度", "增加安全裕度"])
        triggers.append("风扰动偏强")

    if not actions and severity == "保持监视":
        actions.append("保持当前航线并继续滚动预测")
    actions = list(dict.fromkeys(actions))
    triggers = list(dict.fromkeys(triggers)) or ["未触发强避险条件"]
    return {
        "severity": severity,
        "triggers": triggers,
        "actions": actions,
        "explanation": explain_risk(current),
    }


def simulate_avoidance_effect(future: pd.DataFrame, actions: list[str]) -> pd.DataFrame:
    """Estimate risk after avoidance without mutating the original flight record."""

    adjusted = future.copy()
    multiplier = 1.0
    if "提高高度至安全高度层" in actions or "提高高度" in actions:
        adjusted["social_risk"] = adjusted["social_risk"] * 0.86
        adjusted["environment_risk"] = adjusted["environment_risk"] * 0.92
        multiplier *= 0.92
    if "切换安全走廊" in actions or "绕开敏感区缓冲范围" in actions:
        adjusted["social_risk"] = adjusted["social_risk"] * 0.78
        adjusted["compliance_risk"] = adjusted["compliance_risk"] * 0.82
        multiplier *= 0.88
    if "临时悬停等待" in actions or "错峰通过冲突点" in actions or "切换高度层" in actions:
        adjusted["airspace_risk"] = adjusted["airspace_risk"] * 0.65
        multiplier *= 0.86
    if "降低速度" in actions or "重新入航" in actions:
        adjusted["mission_risk"] = adjusted["mission_risk"] * 0.92
        adjusted["drone_health_risk"] = adjusted["drone_health_risk"] * 0.90
        multiplier *= 0.94
    if "返航或就近备降" in actions:
        multiplier *= 0.80
    adjusted["avoidance_predicted_risk"] = (adjusted["ai_future_30s_risk"] * multiplier).clip(0, 1)
    return adjusted


def active_avoidance_assessment(ts: pd.DataFrame, current_second: int, horizon_seconds: int = 300) -> dict:
    """Run rolling forecast, avoidance recommendation, and effect estimation."""

    current = ts.loc[ts["timestamp"].eq(current_second)]
    current_row = current.iloc[0] if not current.empty else ts.iloc[min(current_second, len(ts) - 1)]
    forecast = forecast_flight_risk(ts, current_second, horizon_seconds)
    recommendation = recommend_avoidance_action(current_row, forecast)
    future = ts.loc[ts["timestamp"].between(current_second, current_second + horizon_seconds)].copy()
    adjusted = simulate_avoidance_effect(future, recommendation["actions"])
    before = float(future["ai_future_30s_risk"].max()) if not future.empty else float(current_row["ai_future_30s_risk"])
    after = float(adjusted["avoidance_predicted_risk"].max()) if not adjusted.empty else before
    return {
        "forecast": forecast,
        "recommendation": recommendation,
        "adjusted_future": adjusted,
        "before_max_risk": before,
        "after_max_risk": after,
        "risk_reduction": max(0.0, before - after),
    }

