"""XGBoost operation-process risk prediction and anomaly detection."""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .risk_index import WARNING_ORDER, score_level


AI_FEATURES = [
    "environment_risk",
    "drone_health_risk",
    "mission_risk",
    "airspace_risk",
    "social_risk",
    "compliance_risk",
    "soc",
    "path_error",
    "nearest_building_distance",
    "nearest_drone_distance",
    "wind_speed",
    "gps_error",
    "building_density_300m",
    "sensitive_area_risk",
    "population_exposure_risk",
    "roll",
    "pitch",
]

LEVEL_BY_ID = {value: key for key, value in WARNING_ORDER.items()}

NON_RISK_MODEL_FILENAME = "xgboost_non_risk_operation_risk.joblib"
OPERATION_MODEL_FILENAME = "xgboost_operation_risk.joblib"


def _default_feature_value(feature: str):
    """Default values for non-risk simulation parameters missing in live runs."""

    categorical_tokens = (
        "city_zone",
        "route_type",
        "mission_phase",
        "wind_direction",
        "gust_level",
        "time_period",
        "gis_building_source",
        "task_experiment_type",
        "task_route_complexity",
        "task_route_distance_band",
    )
    if feature in categorical_tokens:
        return "unknown"
    defaults = {
        "visibility": 0.85,
        "battery_soh": 0.88,
        "soc": 0.80,
        "task_initial_soc": 0.86,
        "task_payload_rate": 0.35,
        "task_drone_count": 3,
        "task_cruise_speed": 10.0,
        "task_hover_seconds": 10,
        "task_route_distance": 2500.0,
        "altitude": 90.0,
        "speed": 10.0,
        "gps_error": 2.0,
        "wind_speed": 3.0,
        "nearest_drone_distance": 180.0,
        "nearest_building_distance": 40.0,
    }
    return defaults.get(feature, 0.0)


def _mirror_task_feature(frame: pd.DataFrame, feature: str):
    """Fill task-level model features from equivalent second-level fields when present."""

    aliases = {
        "task_start_lon": "longitude",
        "task_start_lat": "latitude",
        "task_end_lon": "longitude",
        "task_end_lat": "latitude",
        "task_initial_soc": "soc",
        "task_payload_rate": "payload_rate",
        "task_route_distance": "route_distance",
        "task_route_complexity": "route_complexity",
        "task_drone_count": "drone_count",
        "task_cruise_speed": "speed",
        "task_hover_seconds": "hover_seconds",
    }
    source = aliases.get(feature)
    if source and source in frame.columns:
        return frame[source]
    if feature == "task_experiment_type":
        return pd.Series("user_prediction", index=frame.index)
    if feature == "task_route_distance_band":
        distance = frame["route_distance"] if "route_distance" in frame.columns else pd.Series(2500.0, index=frame.index)
        return pd.cut(
            pd.to_numeric(distance, errors="coerce").fillna(2500.0),
            bins=[0, 3000, 6500, float("inf")],
            labels=["last_mile", "urban", "cross_district"],
            include_lowest=True,
        ).astype(str)
    return pd.Series(_default_feature_value(feature), index=frame.index)


def _prepare_non_risk_features(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Prepare raw simulation parameters for the non-risk XGBoost model."""

    raw_features = list(bundle.get("raw_features", []))
    encoded_features = list(bundle.get("encoded_features", []))
    categorical_features = set(bundle.get("categorical_features", []))
    numeric_features = set(bundle.get("numeric_features", []))

    model_input = pd.DataFrame(index=frame.index)
    for feature in raw_features:
        if feature in frame.columns:
            model_input[feature] = frame[feature]
        elif feature.startswith("task_"):
            model_input[feature] = _mirror_task_feature(frame, feature)
        else:
            model_input[feature] = _default_feature_value(feature)

    for feature in numeric_features:
        if feature in model_input.columns:
            model_input[feature] = pd.to_numeric(model_input[feature], errors="coerce").fillna(_default_feature_value(feature))
    for feature in categorical_features:
        if feature in model_input.columns:
            model_input[feature] = model_input[feature].astype(str).fillna("unknown")

    encoded = pd.get_dummies(model_input, columns=[c for c in categorical_features if c in model_input.columns], dtype=float)
    for feature in encoded_features:
        if feature not in encoded.columns:
            encoded[feature] = 0.0
    return encoded[encoded_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _add_compatibility_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep older dashboard columns while the model target is operation risk."""

    frame["ai_future_30s_risk"] = frame["ai_operation_risk"]
    frame["ai_future_30s_level"] = frame["ai_operation_level"]
    return frame


def _fallback_predict_operation_risk(ts: pd.DataFrame) -> pd.DataFrame:
    """Fallback used before an XGBoost model exists."""

    frame = ts.copy()
    window = min(12, max(3, len(frame) // 8))
    trend = frame["llsri"].diff().rolling(window, min_periods=1).mean().fillna(0)
    remaining_pressure = frame["llsri"][::-1].cummax()[::-1]
    projected = (frame["llsri"] + 120 * trend).clip(0, 1)
    feature_pressure = (
        0.24 * frame["social_risk"]
        + 0.20 * frame["airspace_risk"]
        + 0.20 * frame["drone_health_risk"]
        + 0.20 * frame["compliance_risk"]
        + 0.16 * frame["environment_risk"]
    )
    frame["ai_operation_risk"] = (0.50 * remaining_pressure + 0.25 * projected + 0.25 * feature_pressure).clip(0, 1)
    frame["ai_operation_level"] = frame["ai_operation_risk"].map(lambda x: score_level(float(x)))
    frame["ai_model_type"] = "fallback_operation_risk_before_xgboost"
    return _add_compatibility_columns(frame)


def make_training_frame(timeseries: list[pd.DataFrame], horizon_seconds: int = 30) -> pd.DataFrame:
    """Build supervised samples whose target is whole remaining operation risk."""

    rows: list[pd.DataFrame] = []
    for ts in timeseries:
        if len(ts) < 30:
            continue
        frame = ts.copy()
        operation_risk = frame["llsri"][::-1].cummax()[::-1]
        frame["target_operation_risk"] = operation_risk.astype(float).to_numpy()
        frame["target_operation_level"] = frame["target_operation_risk"].map(lambda value: WARNING_ORDER[score_level(float(value))]).astype(int)
        rows.append(frame[AI_FEATURES + ["target_operation_level", "target_operation_risk"]])
    if not rows:
        return pd.DataFrame(columns=AI_FEATURES + ["target_operation_level", "target_operation_risk"])
    return pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0)


def train_xgboost_model(timeseries: list[pd.DataFrame], model_dir: Path, horizon_seconds: int = 30) -> dict:
    """Train and persist an XGBoost regressor for whole-operation risk prediction."""

    import joblib
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    model_dir.mkdir(parents=True, exist_ok=True)
    data = make_training_frame(timeseries, horizon_seconds)
    if len(data) < 100:
        return {"trained": False, "reason": "not_enough_samples", "sample_count": int(len(data))}

    x = data[AI_FEATURES]
    y = data["target_operation_risk"].astype(float)
    y_level = data["target_operation_level"].astype(int)
    stratify = y_level if y_level.nunique() > 1 and y_level.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test, level_train, level_test = train_test_split(
        x, y, y_level, test_size=0.2, random_state=20260729, stratify=stratify
    )
    model = XGBRegressor(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=20260729,
        n_jobs=2,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    pred = np.clip(pred, 0, 1)
    pred_level = pd.Series(pred).map(lambda value: WARNING_ORDER[score_level(float(value))]).astype(int)
    metrics = {
        "trained": True,
        "model_type": "XGBoostRegressor",
        "target": "whole_operation_remaining_max_llsri",
        "sample_count": int(len(data)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "level_accuracy": float(accuracy_score(level_test, pred_level)),
        "level_macro_f1": float(f1_score(level_test, pred_level, average="macro")),
        "horizon_seconds": 0,
        "feature_importance": {feature: float(value) for feature, value in zip(AI_FEATURES, model.feature_importances_)},
    }
    joblib.dump({"model": model, "features": AI_FEATURES, "metrics": metrics}, model_dir / "xgboost_operation_risk.joblib")
    pd.DataFrame([metrics | {f"importance_{k}": v for k, v in metrics["feature_importance"].items()}]).to_csv(
        model_dir / "xgboost_metrics.csv", index=False, encoding="utf-8-sig"
    )
    return metrics


def load_xgboost_model(model_dir: Path):
    try:
        import joblib
        try:
            from sklearn.exceptions import InconsistentVersionWarning
        except Exception:
            InconsistentVersionWarning = UserWarning

        for filename in [NON_RISK_MODEL_FILENAME, OPERATION_MODEL_FILENAME, "xgboost_future30.joblib"]:
            path = model_dir / filename
            if not path.exists():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InconsistentVersionWarning)
                return joblib.load(path)
    except Exception:
        return None
    return None


def predict_future_risk(ts: pd.DataFrame, model_dir: Path | None = None, horizon_seconds: int = 30) -> pd.DataFrame:
    """Predict whole-operation risk with trained XGBoost when available."""

    if model_dir is None:
        return _fallback_predict_operation_risk(ts)
    bundle = load_xgboost_model(model_dir)
    if not bundle:
        return _fallback_predict_operation_risk(ts)

    frame = ts.copy()
    if "raw_features" in bundle and "encoded_features" in bundle:
        x = _prepare_non_risk_features(frame, bundle)
        model = bundle["model"]
        operation_risk = np.clip(model.predict(x), 0, 1)
        frame["ai_operation_risk"] = operation_risk
        frame["ai_operation_level"] = frame["ai_operation_risk"].map(lambda value: score_level(float(value)))
        frame["ai_model_type"] = "xgboost_non_risk_simulation_parameters"
        return _add_compatibility_columns(frame)

    features = bundle["features"]
    categorical_features = set(bundle.get("categorical_features", []))
    for feature in features:
        if feature not in frame.columns:
            frame[feature] = "unknown" if feature in categorical_features else 0.0
    x = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = bundle["model"]
    metrics = bundle.get("metrics", {})
    target = str(metrics.get("target", ""))
    if target in {"whole_operation_remaining_max_llsri", "remaining_max_llsri_during_flight"} or not hasattr(model, "predict_proba"):
        operation_risk = np.clip(model.predict(x), 0, 1)
        frame["ai_operation_risk"] = operation_risk
        frame["ai_operation_level"] = frame["ai_operation_risk"].map(lambda value: score_level(float(value)))
        frame["ai_model_type"] = "xgboost_remaining_operation_risk"
        return _add_compatibility_columns(frame)

    level_id = model.predict(x).astype(int)
    proba = model.predict_proba(x)
    expected = np.zeros(len(frame), dtype=float)
    for cls_index, cls in enumerate(model.classes_):
        expected += proba[:, cls_index] * (float(cls) / 4.0)
    frame["ai_operation_risk"] = np.clip(expected, 0, 1)
    frame["ai_operation_level"] = [LEVEL_BY_ID.get(int(value), "正常") for value in level_id]
    frame["ai_model_type"] = "legacy_xgboost_future30_used_as_operation_proxy"
    return _add_compatibility_columns(frame)


def detect_anomalies(ts: pd.DataFrame) -> pd.DataFrame:
    """Detect abnormal GPS, battery, attitude, and airspace behavior."""

    frame = ts.copy()
    checks = {
        "GPS异常": frame["path_error"] > max(25, frame["path_error"].median() + 2.5 * frame["path_error"].std()),
        "电池异常": frame["soc"].diff().fillna(0) < -0.012,
        "姿态异常": (frame["roll"].abs() + frame["pitch"].abs()) > 32,
        "空域接近异常": frame["nearest_drone_distance"] < 25,
        "建筑接近异常": frame["nearest_building_distance"] < 12,
    }
    labels = []
    scores = []
    for idx in range(len(frame)):
        active = [name for name, mask in checks.items() if bool(mask.iloc[idx])]
        labels.append(";".join(active) if active else "")
        scores.append(min(1.0, len(active) / 3))
    frame["ai_anomaly_score"] = scores
    frame["ai_anomaly_label"] = labels
    return frame


def add_ai_outputs(ts: pd.DataFrame, model_dir: Path | None = None) -> pd.DataFrame:
    return detect_anomalies(predict_future_risk(ts, model_dir=model_dir))


def train_ai_profile(summary: pd.DataFrame, model_dir: Path | None = None) -> dict:
    """Create dashboard model profile from trained XGBoost metrics or summary fallback."""

    if model_dir is not None:
        non_risk_metrics_path = model_dir / "xgboost_metrics.csv"
        non_risk_importance_path = model_dir / "xgboost_non_risk_feature_importance.csv"
        if non_risk_metrics_path.exists():
            metrics = pd.read_csv(non_risk_metrics_path).iloc[0].to_dict()
            if str(metrics.get("input_design", "")).startswith("all_available_simulation_parameters"):
                importance: dict[str, float] = {}
                if non_risk_importance_path.exists():
                    raw_importance = pd.read_csv(non_risk_importance_path)
                    if {"feature", "importance"}.issubset(raw_importance.columns):
                        importance = {
                            str(row["feature"]): float(row["importance"])
                            for _, row in raw_importance.head(20).iterrows()
                        }
                return {
                    "sample_count": int(metrics.get("total_seconds_rows", metrics.get("sample_count", 0))),
                    "model_type": metrics.get("model_type", "XGBRegressor"),
                    "target": metrics.get("target", "remaining_max_llsri_during_flight"),
                    "accuracy": float(metrics.get("risk_level_accuracy", metrics.get("level_accuracy", 0))),
                    "macro_f1": float(metrics.get("risk_level_macro_f1", metrics.get("level_macro_f1", 0))),
                    "rmse": float(metrics.get("rmse", 0)),
                    "r2": float(metrics.get("r2", 0)),
                    "feature_importance": importance,
                }

        metrics_path = model_dir / "xgboost_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
            importance = {
                key.replace("importance_", ""): float(value)
                for key, value in metrics.items()
                if key.startswith("importance_")
            }
            return {
                "sample_count": int(metrics.get("sample_count", 0)),
                "model_type": metrics.get("model_type", "XGBoostRegressor"),
                "target": metrics.get("target", "whole_operation_remaining_max_llsri"),
                "accuracy": float(metrics.get("level_accuracy", metrics.get("accuracy", 0))),
                "macro_f1": float(metrics.get("level_macro_f1", metrics.get("macro_f1", 0))),
                "rmse": float(metrics.get("rmse", 0)),
                "r2": float(metrics.get("r2", 0)),
                "feature_importance": importance,
            }

    if summary.empty:
        return {"sample_count": 0, "feature_importance": {}}
    severity = summary["red_warning_count"] + 0.55 * summary["orange_warning_count"] + 20 * (1 - summary["mission_success"].astype(float))
    importance = {}
    columns = {
        "环境风险": "avg_environment_risk",
        "健康风险": "avg_drone_health_risk",
        "任务风险": "avg_mission_risk",
        "空域风险": "avg_airspace_risk",
        "社会风险": "avg_social_risk",
        "合规风险": "avg_compliance_risk",
    }
    for label, col in columns.items():
        values = summary[col]
        corr = abs(values.corr(severity)) if values.std() > 1e-9 else 0.0
        importance[label] = 0.0 if np.isnan(corr) else float(corr)
    total = sum(importance.values()) or 1
    importance = {key: value / total for key, value in importance.items()}
    return {"sample_count": int(len(summary)), "model_type": "summary_correlation_fallback", "feature_importance": importance}
