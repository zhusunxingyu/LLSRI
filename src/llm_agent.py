"""Local LLM-style risk explanation assistant."""

from __future__ import annotations

import pandas as pd

from .rag_system import retrieve_rules


def explain_risk(row: pd.Series) -> str:
    risks = {
        "环境": float(row.get("environment_risk", 0)),
        "健康": float(row.get("drone_health_risk", 0)),
        "任务": float(row.get("mission_risk", 0)),
        "空域": float(row.get("airspace_risk", 0)),
        "社会": float(row.get("social_risk", 0)),
        "合规": float(row.get("compliance_risk", 0)),
    }
    top = sorted(risks.items(), key=lambda item: item[1], reverse=True)[:3]
    rules = retrieve_rules(str(row.get("warning_reason", "")))
    return (
        f"当前预警等级为{row.get('warning_level')}，LLSRI={float(row.get('llsri', 0)):.3f}。"
        f"主要贡献来自{top[0][0]}风险({top[0][1]:.2f})、{top[1][0]}风险({top[1][1]:.2f})和{top[2][0]}风险({top[2][1]:.2f})。"
        f"建议：{rules[0]}"
    )


def mission_report(ts: pd.DataFrame) -> str:
    worst = ts.sort_values("llsri", ascending=False).iloc[0]
    anomaly_count = int(ts.get("ai_anomaly_label", pd.Series([""] * len(ts))).astype(str).ne("").sum())
    return (
        f"本任务最高风险出现在第{int(worst.timestamp)}秒，等级{worst.warning_level}，"
        f"原因{worst.warning_reason}。AI异常检测共发现{anomaly_count}个异常秒。"
        f"{explain_risk(worst)}"
    )

