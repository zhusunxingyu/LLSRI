"""Small local RAG-style rule base."""

from __future__ import annotations


KNOWLEDGE = [
    {"tag": "LOW_BATTERY", "text": "SOC低于20%时建议提前返航；低于10%时应触发红色告警并准备迫降。"},
    {"tag": "BUILDING_PROXIMITY", "text": "接近建筑物会提高障碍物碰撞和城市峡谷风风险，应提高高度或切换安全走廊。"},
    {"tag": "SENSITIVE_AREA_FLYOVER", "text": "飞越学校、医院等敏感区300m缓冲范围，应降低暴露时间并优先绕行。"},
    {"tag": "LOW_ALTITUDE_SENSITIVE_AREA", "text": "敏感区域低高度飞行会显著提高噪声、隐私和社会安全风险。"},
    {"tag": "AIRSPACE_CONFLICT", "text": "多机距离过近时建议减速、悬停等待或重新分配航路高度层。"},
    {"tag": "COMPLIANCE_RESTRICTED_ZONE", "text": "强合规约束区不应用于常规物流穿越，需重新规划航线。"},
    {"tag": "HIGH_GROUND_IMPACT_RISK", "text": "高人口暴露叠加坠落概率上升时，应避免继续飞越密集区域。"},
    {"tag": "WEATHER", "text": "风速、阵风、降水和高温会影响能耗、姿态稳定和电池安全裕度。"},
]


def retrieve_rules(reason: str, top_k: int = 4) -> list[str]:
    hits = [item["text"] for item in KNOWLEDGE if item["tag"] in reason]
    if "LLSRI" in reason or not hits:
        hits.append("综合风险升高时，应查看环境、健康、任务、空域和社会风险贡献，优先处理最高贡献项。")
    return hits[:top_k]

