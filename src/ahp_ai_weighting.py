"""AHP and data-driven risk-weight fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


RISK_COLUMNS = ["environment_risk", "drone_health_risk", "mission_risk", "airspace_risk", "social_risk", "compliance_risk"]
RISK_LABELS = {
    "environment_risk": "飞行环境风险",
    "drone_health_risk": "无人机健康风险",
    "mission_risk": "任务运行风险",
    "airspace_risk": "空域运行风险",
    "social_risk": "社会安全风险",
    "compliance_risk": "合规约束风险",
}


RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


@dataclass(frozen=True)
class AHPResult:
    weights: dict[str, float]
    lambda_max: float
    ci: float
    cr: float
    valid: bool


def default_pairwise_matrix() -> np.ndarray:
    """Expert prior matrix; airspace, social, and compliance risks are emphasized."""

    # Order: environment, drone health, mission, airspace, social, compliance.
    return np.array(
        [
            [1, 1, 2, 1 / 2, 1 / 2, 1 / 2],
            [1, 1, 2, 1 / 2, 1 / 2, 1 / 2],
            [1 / 2, 1 / 2, 1, 1 / 3, 1 / 3, 1 / 3],
            [2, 2, 3, 1, 1, 1],
            [2, 2, 3, 1, 1, 1],
            [2, 2, 3, 1, 1, 1],
        ],
        dtype=float,
    )


def calculate_ahp_weights(matrix: np.ndarray | None = None) -> AHPResult:
    matrix = matrix if matrix is not None else default_pairwise_matrix()
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_index = int(np.argmax(eigvals.real))
    lambda_max = float(eigvals[max_index].real)
    weights = np.abs(eigvecs[:, max_index].real)
    weights = weights / weights.sum()
    n = matrix.shape[0]
    ci = (lambda_max - n) / (n - 1)
    ri = RI_TABLE.get(n, 1.12)
    cr = 0.0 if ri == 0 else ci / ri
    return AHPResult(
        weights={col: float(weights[i]) for i, col in enumerate(RISK_COLUMNS)},
        lambda_max=lambda_max,
        ci=float(ci),
        cr=float(cr),
        valid=bool(cr < 0.1),
    )


def entropy_weights(frame: pd.DataFrame) -> dict[str, float]:
    """Entropy weighting gives higher weight to indicators with more information."""

    data = frame[RISK_COLUMNS].clip(0, 1).to_numpy(dtype=float)
    data = data + 1e-9
    normalized = data / data.sum(axis=0, keepdims=True)
    entropy = -(normalized * np.log(normalized)).sum(axis=0) / np.log(len(data))
    diversity = 1 - entropy
    if float(diversity.sum()) <= 1e-12:
        weights = np.ones(len(RISK_COLUMNS)) / len(RISK_COLUMNS)
    else:
        weights = diversity / diversity.sum()
    return {col: float(weights[i]) for i, col in enumerate(RISK_COLUMNS)}


def supervised_proxy_weights(frame: pd.DataFrame) -> dict[str, float]:
    """Estimate indicator importance from relation to severe outcomes."""

    severe = (
        frame.get("red_warning_count", 0).to_numpy(dtype=float)
        + 0.55 * frame.get("orange_warning_count", 0).to_numpy(dtype=float)
        + 25 * (1 - frame.get("mission_success", 1).astype(float).to_numpy())
    )
    if severe.std() < 1e-9:
        return {col: 1 / len(RISK_COLUMNS) for col in RISK_COLUMNS}
    scores = []
    for col in RISK_COLUMNS:
        values = frame[f"avg_{col.replace('_risk', '')}_risk"].to_numpy(dtype=float) if f"avg_{col.replace('_risk', '')}_risk" in frame else frame[col].to_numpy(dtype=float)
        corr = abs(np.corrcoef(values, severe)[0, 1]) if values.std() > 1e-9 else 0.0
        scores.append(0.0 if np.isnan(corr) else corr)
    scores = np.array(scores) + 1e-6
    scores = scores / scores.sum()
    return {col: float(scores[i]) for i, col in enumerate(RISK_COLUMNS)}


def summary_to_indicator_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """Normalize summary column names to risk indicator columns."""

    return pd.DataFrame(
        {
            "environment_risk": summary["avg_environment_risk"],
            "drone_health_risk": summary["avg_drone_health_risk"],
            "mission_risk": summary["avg_mission_risk"],
            "airspace_risk": summary["avg_airspace_risk"],
            "social_risk": summary["avg_social_risk"],
            "compliance_risk": summary["avg_compliance_risk"],
            "red_warning_count": summary.get("red_warning_count", 0),
            "orange_warning_count": summary.get("orange_warning_count", 0),
            "mission_success": summary.get("mission_success", True),
        }
    )


def fuse_weights(ahp: dict[str, float], entropy: dict[str, float], supervised: dict[str, float], alpha: float = 0.50, beta: float = 0.25) -> dict[str, float]:
    """Fuse expert prior, entropy information, and outcome relevance."""

    gamma = 1 - alpha - beta
    raw = {
        col: alpha * ahp[col] + beta * entropy[col] + gamma * supervised[col]
        for col in RISK_COLUMNS
    }
    total = sum(raw.values())
    return {col: float(value / total) for col, value in raw.items()}


def calculate_fusion_weights(summary: pd.DataFrame | None = None) -> dict:
    ahp_result = calculate_ahp_weights()
    if summary is None or summary.empty:
        entropy = {col: 1 / len(RISK_COLUMNS) for col in RISK_COLUMNS}
        supervised = entropy.copy()
    else:
        indicators = summary_to_indicator_frame(summary)
        entropy = entropy_weights(indicators)
        supervised = supervised_proxy_weights(indicators)
    fused = fuse_weights(ahp_result.weights, entropy, supervised)
    return {
        "ahp": ahp_result.weights,
        "entropy": entropy,
        "supervised_proxy": supervised,
        "fused": fused,
        "cr": ahp_result.cr,
        "ci": ahp_result.ci,
        "lambda_max": ahp_result.lambda_max,
        "valid": ahp_result.valid,
    }


def apply_weighted_llsri(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = sum(df[col] * weights[col] for col in RISK_COLUMNS)
    return score.clip(0, 1)
