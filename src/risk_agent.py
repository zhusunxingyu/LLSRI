"""AI flight-risk monitoring agent powered by the trained XGBoost model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ai_predictor import add_ai_outputs
from .llm_agent import explain_risk
from .risk_index import score_level


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLATFORM_MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
LATEST_EXPERIMENT_MODEL_DIR = Path("E:/") / "小挑" / "新六大功能区仿真" / "models" / "xgboost_inflight_risk"


def risk_band(value: float) -> str:
    if value >= 0.56:
        return "极高风险"
    if value >= 0.42:
        return "高风险"
    if value >= 0.30:
        return "中风险"
    return "低风险"


def severity_from_risk(value: float) -> str:
    if value >= 0.60:
        return "立即干预"
    if value >= 0.42:
        return "重点监测"
    if value >= 0.30:
        return "加强巡检"
    return "常态监测"


def _safe_float(row: pd.Series, name: str, default: float = 0.0) -> float:
    try:
        return float(row.get(name, default))
    except Exception:
        return default


def _top_risk_factors(row: pd.Series) -> list[tuple[str, float]]:
    labels = {
        "environment_risk": "环境风险",
        "drone_health_risk": "无人机健康风险",
        "mission_risk": "任务运行风险",
        "airspace_risk": "空域运行风险",
        "social_risk": "社会安全风险",
        "compliance_risk": "合规约束风险",
    }
    values = [(label, _safe_float(row, col)) for col, label in labels.items()]
    return sorted(values, key=lambda item: item[1], reverse=True)[:3]


@dataclass
class FlightRiskAgent:
    """Monitor the current state and predict remaining in-flight risk."""

    model_dir: Path = DEFAULT_PLATFORM_MODEL_DIR

    def enrich_predictions(self, ts: pd.DataFrame) -> pd.DataFrame:
        frame = add_ai_outputs(ts, model_dir=self.model_dir)
        if "ai_operation_risk" not in frame.columns:
            frame["ai_operation_risk"] = frame["llsri"]
        if "ai_operation_level" not in frame.columns:
            frame["ai_operation_level"] = frame["ai_operation_risk"].map(lambda value: score_level(float(value)))
        return frame

    def assess(self, ts: pd.DataFrame, current_second: int, horizon_seconds: int = 600) -> dict:
        frame = self.enrich_predictions(ts)
        if frame.empty:
            return {
                "available": False,
                "message": "没有可用的飞行秒级数据，智能体无法监测。",
                "timeseries": frame,
                "events": pd.DataFrame(),
            }

        current_second = int(np.clip(current_second, int(frame.timestamp.min()), int(frame.timestamp.max())))
        current_rows = frame.loc[frame["timestamp"].eq(current_second)]
        current = current_rows.iloc[0] if not current_rows.empty else frame.iloc[(frame["timestamp"] - current_second).abs().argmin()]
        remaining = frame.loc[frame["timestamp"] >= current_second].copy()
        horizon = frame.loc[frame["timestamp"].between(current_second, current_second + int(horizon_seconds))].copy()
        if horizon.empty:
            horizon = remaining.head(1).copy()

        worst_remaining = remaining.sort_values(["ai_operation_risk", "llsri"], ascending=False).iloc[0]
        worst_horizon = horizon.sort_values(["ai_operation_risk", "llsri"], ascending=False).iloc[0]
        factors = _top_risk_factors(current)
        actions, triggers = self.recommend_actions(current, worst_remaining)

        events = self.build_event_log(frame, current_second, horizon_seconds)
        return {
            "available": True,
            "timeseries": frame,
            "current": current,
            "remaining": remaining,
            "horizon": horizon,
            "events": events,
            "current_predicted_risk": _safe_float(current, "ai_operation_risk"),
            "current_predicted_band": risk_band(_safe_float(current, "ai_operation_risk")),
            "remaining_max_predicted_risk": _safe_float(worst_remaining, "ai_operation_risk"),
            "remaining_max_predicted_band": risk_band(_safe_float(worst_remaining, "ai_operation_risk")),
            "horizon_max_predicted_risk": _safe_float(worst_horizon, "ai_operation_risk"),
            "horizon_max_predicted_band": risk_band(_safe_float(worst_horizon, "ai_operation_risk")),
            "worst_remaining_second": int(worst_remaining["timestamp"]),
            "worst_horizon_second": int(worst_horizon["timestamp"]),
            "severity": severity_from_risk(_safe_float(worst_remaining, "ai_operation_risk")),
            "top_factors": factors,
            "triggers": triggers,
            "actions": actions,
            "explanation": self.explain(current, worst_remaining, factors, actions),
            "model_type": str(current.get("ai_model_type", "unknown")),
        }

    def recommend_actions(self, current: pd.Series, worst_remaining: pd.Series) -> tuple[list[str], list[str]]:
        actions: list[str] = []
        triggers: list[str] = []
        predicted = _safe_float(worst_remaining, "ai_operation_risk")
        if predicted >= 0.60:
            actions.extend(["立即进入安全走廊", "降低任务优先级并准备返航或备降"])
            triggers.append("剩余航程预测存在橙色及以上风险")
        elif predicted >= 0.42:
            actions.extend(["提前切换安全高度层", "增加轨迹监测频率"])
            triggers.append("剩余航程预测达到高风险区间")
        elif predicted >= 0.30:
            actions.append("保持航线并缩短滚动评估间隔")
            triggers.append("剩余航程预测达到中风险区间")

        if _safe_float(current, "compliance_risk") >= 0.55:
            actions.extend(["检查禁限飞与敏感区缓冲约束", "必要时绕开合规高风险航段"])
            triggers.append("合规约束风险偏高")
        if _safe_float(current, "nearest_drone_distance", 999) < 35:
            actions.extend(["启动机间避让", "错峰通过潜在冲突点"])
            triggers.append("机间距离过近")
        if _safe_float(current, "nearest_building_distance", 999) < 20:
            actions.extend(["提高离障裕度", "修正航线避开建筑密集区"])
            triggers.append("建筑物距离不足")
        if _safe_float(current, "soc", 1.0) < 0.22:
            actions.extend(["评估返航电量", "选择就近安全降落点"])
            triggers.append("电池SOC偏低")
        if _safe_float(current, "path_error") > 25:
            actions.extend(["降低巡航速度", "重新入航并校正定位"])
            triggers.append("航迹偏差过大")
        if _safe_float(current, "wind_speed") >= 9:
            actions.extend(["降低速度", "提高抗风安全裕度"])
            triggers.append("风速扰动偏强")

        if not actions:
            actions.append("维持当前任务，继续秒级滚动监测")
            triggers.append("未触发强干预条件")
        return list(dict.fromkeys(actions)), list(dict.fromkeys(triggers))

    def build_event_log(self, frame: pd.DataFrame, current_second: int, horizon_seconds: int) -> pd.DataFrame:
        window = frame.loc[frame["timestamp"].between(current_second, current_second + horizon_seconds)].copy()
        if window.empty:
            return pd.DataFrame()
        anomaly_mask = window["ai_anomaly_label"].astype(str).ne("") if "ai_anomaly_label" in window.columns else False
        event_rows = window.loc[
            (window["ai_operation_risk"] >= 0.30)
            | anomaly_mask
            | (window.get("compliance_risk", 0) >= 0.50)
        ].copy()
        if event_rows.empty:
            event_rows = window.sort_values("ai_operation_risk", ascending=False).head(5).copy()
        event_rows["预测风险区间"] = event_rows["ai_operation_risk"].map(lambda value: risk_band(float(value)))
        event_rows["智能体处置级别"] = event_rows["ai_operation_risk"].map(lambda value: severity_from_risk(float(value)))
        columns = [
            "timestamp",
            "mission_phase",
            "llsri",
            "ai_operation_risk",
            "预测风险区间",
            "智能体处置级别",
            "compliance_risk",
            "social_risk",
            "airspace_risk",
            "soc",
            "path_error",
            "ai_anomaly_label",
        ]
        return event_rows[[col for col in columns if col in event_rows.columns]].head(300)

    def explain(self, current: pd.Series, worst_remaining: pd.Series, factors: list[tuple[str, float]], actions: list[str]) -> str:
        factor_text = "、".join(f"{name}{value:.2f}" for name, value in factors)
        return (
            f"AI智能体在第{int(current['timestamp'])}秒读取当前飞行状态，"
            f"XGBoost预测当前剩余全过程风险为{_safe_float(current, 'ai_operation_risk'):.3f}，"
            f"剩余航程最高预测风险为{_safe_float(worst_remaining, 'ai_operation_risk'):.3f}，"
            f"预计最危险时刻出现在第{int(worst_remaining['timestamp'])}秒。"
            f"当前主要风险贡献为{factor_text}。"
            f"建议执行：{'；'.join(actions)}。"
            f"{explain_risk(current)}"
        )


def run_flight_risk_agent(ts: pd.DataFrame, current_second: int, horizon_seconds: int = 600, model_dir: Path = DEFAULT_PLATFORM_MODEL_DIR) -> dict:
    return FlightRiskAgent(model_dir=model_dir).assess(ts, current_second=current_second, horizon_seconds=horizon_seconds)
