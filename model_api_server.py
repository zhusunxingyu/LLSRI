"""Local XGBoost model API for the standalone route-planning webpage."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import warnings

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "outputs" / "models" if (ROOT / "outputs" / "models").exists() else ROOT
HOST = "127.0.0.1"
PORT = 8765
NON_RISK_MODEL_FILENAME = "xgboost_non_risk_operation_risk.joblib"


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


def _default_feature_value(feature: str):
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


def _prepare_non_risk_features(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    raw_features = list(bundle.get("raw_features", []))
    encoded_features = list(bundle.get("encoded_features", []))
    categorical_features = set(bundle.get("categorical_features", []))
    numeric_features = set(bundle.get("numeric_features", []))

    model_input = pd.DataFrame(index=frame.index)
    for feature in raw_features:
        if feature in frame.columns:
            model_input[feature] = frame[feature]
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


def load_xgboost_model(model_dir: Path):
    try:
        import joblib
        try:
            from sklearn.exceptions import InconsistentVersionWarning
        except Exception:
            InconsistentVersionWarning = UserWarning

        for base in [model_dir, ROOT / "outputs" / "models", ROOT]:
            path = base / NON_RISK_MODEL_FILENAME
            if path.exists():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    return joblib.load(path)
    except Exception:
        return None
    return None


def _as_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _route_distance_band(distance_m: float) -> str:
    if distance_m <= 3000:
        return "last_mile"
    if distance_m <= 6500:
        return "urban"
    return "cross_district"


def _build_feature_frame(payload: dict) -> pd.DataFrame:
    points = payload.get("points") or []
    params = payload.get("params") or {}
    if not points:
        raise ValueError("points cannot be empty")

    frame = pd.DataFrame(points)
    frame["longitude"] = pd.to_numeric(frame.get("lon", frame.get("longitude")), errors="coerce")
    frame["latitude"] = pd.to_numeric(frame.get("lat", frame.get("latitude")), errors="coerce")
    frame["timestamp"] = pd.to_numeric(frame.get("timestamp", pd.Series(range(len(frame)))), errors="coerce").fillna(0)
    frame["altitude"] = pd.to_numeric(frame.get("altitude", 90.0), errors="coerce").fillna(90.0)
    frame["z"] = frame["altitude"]

    lon0 = float(frame["longitude"].iloc[0])
    lat0 = float(frame["latitude"].iloc[0])
    frame["x"] = (frame["longitude"] - lon0) * 111320 * np.cos(np.radians(lat0))
    frame["y"] = (frame["latitude"] - lat0) * 110540

    speed = _as_float(params.get("speed"), 10.0)
    wind = _as_float(params.get("wind"), 4.0)
    gps = _as_float(params.get("gps_error"), 3.0)
    payload_rate = _as_float(params.get("payload_rate"), 0.35)
    initial_soc = _as_float(params.get("initial_soc"), 0.86)
    battery_soh = _as_float(params.get("battery_soh"), 0.90)
    drone_count = int(round(_as_float(params.get("drone_count"), 3.0)))
    distance_m = _as_float(params.get("route_distance"), 2500.0)
    route_type = str(params.get("route_type") or "direct")
    route_complexity = str(params.get("route_complexity") or "medium")
    city_zone = str(params.get("city_zone") or "core_business")
    gust_level = str(params.get("gust_level") or "weak")
    wind_direction = str(params.get("wind_direction") or "E")
    time_period = str(params.get("time_period") or "day")
    hover_seconds = int(round(_as_float(params.get("hover_seconds"), 10.0)))

    n = len(frame)
    progress = np.linspace(0, 1, n)
    frame["speed"] = speed
    frame["mission_phase"] = np.select(
        [progress < 0.10, progress > 0.90, (progress > 0.45) & (progress < 0.55)],
        ["TAKEOFF", "LANDING", "HOVER"],
        default="CRUISE_OUTBOUND",
    )
    frame.loc[progress >= 0.55, "mission_phase"] = "CRUISE_RETURN"
    frame.loc[(progress > 0.45) & (progress < 0.55), "mission_phase"] = "HOVER"
    frame["vx"] = np.gradient(frame["x"].to_numpy())
    frame["vy"] = np.gradient(frame["y"].to_numpy())
    frame["vz"] = np.gradient(frame["z"].to_numpy())
    frame["yaw"] = np.degrees(np.arctan2(frame["vy"], frame["vx"])).round(3)
    frame["roll"] = np.sin(progress * np.pi * 8) * (1.5 + wind * 0.30)
    frame["pitch"] = np.cos(progress * np.pi * 7) * (1.2 + wind * 0.25)
    frame["p_rate"] = np.gradient(frame["roll"].to_numpy())
    frame["q_rate"] = np.gradient(frame["pitch"].to_numpy())
    frame["r_rate"] = np.gradient(frame["yaw"].to_numpy())

    rpm_base = 6400 + payload_rate * 900 + wind * 55
    for i, phase_shift in enumerate([0.0, 0.6, 1.2, 1.8], start=1):
        frame[f"motor_{i}_rpm"] = rpm_base + np.sin(progress * np.pi * 12 + phase_shift) * (120 + wind * 18)
    frame["total_thrust_n"] = 34 + payload_rate * 12 + wind * 0.35
    frame["torque_x_nm"] = frame["roll"] * 0.012
    frame["torque_y_nm"] = frame["pitch"] * 0.012
    frame["torque_z_nm"] = frame["r_rate"] * 0.004
    frame["desired_x"] = np.linspace(frame["x"].iloc[0], frame["x"].iloc[-1], n)
    frame["desired_y"] = np.linspace(frame["y"].iloc[0], frame["y"].iloc[-1], n)
    frame["desired_z"] = frame["z"].rolling(8, min_periods=1).mean()
    frame["path_error"] = np.sqrt((frame["x"] - frame["desired_x"]) ** 2 + (frame["y"] - frame["desired_y"]) ** 2).clip(0, 80)

    frame["city_zone"] = city_zone
    frame["route_type"] = route_type
    frame["route_distance"] = distance_m
    frame["route_complexity"] = route_complexity
    frame["drone_count"] = drone_count
    frame["payload_rate"] = payload_rate
    frame["hover_seconds"] = hover_seconds
    frame["wind_speed"] = wind
    frame["wind_direction"] = wind_direction
    frame["gust_level"] = gust_level
    frame["visibility"] = _as_float(params.get("visibility"), 0.85)
    frame["time_period"] = time_period
    frame["gps_error"] = gps
    frame["soc"] = np.clip(initial_soc - progress * (0.05 + payload_rate * 0.10 + distance_m / 60000), 0, 1)
    frame["battery_soh"] = battery_soh
    frame["voltage"] = 44.4 * frame["soc"] * (0.88 + 0.12 * battery_soh)
    frame["battery_voltage"] = frame["voltage"]
    frame["energy_consumption"] = (180 + speed ** 2 * 2.6 + payload_rate * 95 + wind ** 2 * 4.2) * (frame.index + 1)

    zone_density = {
        "核心商务区": 0.82,
        "交通枢纽区": 0.76,
        "医院学校敏感区": 0.68,
        "居住区": 0.52,
        "工业园区": 0.38,
        "滨水/跨江区": 0.30,
    }.get(city_zone, 0.50)
    frame["nearest_building_distance"] = np.clip(90 - zone_density * 60 + np.sin(progress * np.pi * 5) * 10, 8, 180)
    frame["building_density_300m"] = zone_density
    frame["nearest_school_distance"] = 120 if "医院学校" in city_zone else 900
    frame["nearest_hospital_distance"] = 140 if "医院学校" in city_zone else 900
    frame["sensitive_poi_count_300m"] = 4 if "医院学校" in city_zone else (2 if city_zone in {"核心商务区", "交通枢纽区"} else 0)
    frame["poi_density_300m"] = frame["sensitive_poi_count_300m"] / 5
    frame["residential_exposure"] = 0.75 if city_zone == "居住区" else (0.45 if city_zone == "核心商务区" else 0.20)
    frame["commercial_exposure"] = 0.85 if city_zone == "核心商务区" else 0.25
    frame["transport_hub_exposure"] = 0.85 if city_zone == "交通枢纽区" else 0.18
    frame["river_crossing"] = 1 if city_zone == "滨水/跨江区" else 0
    frame["gis_building_source"] = "web_route_proxy"
    frame["nearest_drone_distance"] = np.clip(260 - drone_count * 9 + np.cos(progress * np.pi * 3) * 25, 12, 500)
    frame["conflict_count"] = np.clip(np.rint((drone_count - 1) / 5 + np.sin(progress * np.pi * 4)), 0, 8)
    frame["near_miss_count"] = (frame["nearest_drone_distance"] < 30).astype(int)

    frame["task_experiment_type"] = "standalone_web_prediction"
    frame["task_start_lon"] = _as_float(params.get("start_lon"), lon0)
    frame["task_start_lat"] = _as_float(params.get("start_lat"), lat0)
    frame["task_end_lon"] = _as_float(params.get("end_lon"), float(frame["longitude"].iloc[-1]))
    frame["task_end_lat"] = _as_float(params.get("end_lat"), float(frame["latitude"].iloc[-1]))
    frame["task_initial_soc"] = initial_soc
    frame["task_payload_rate"] = payload_rate
    frame["task_route_distance"] = distance_m
    frame["task_route_complexity"] = route_complexity
    frame["task_drone_count"] = drone_count
    frame["task_route_distance_band"] = _route_distance_band(distance_m)
    frame["task_cruise_speed"] = speed
    frame["task_hover_seconds"] = hover_seconds
    return frame


def predict_payload(payload: dict) -> dict:
    bundle = load_xgboost_model(MODEL_DIR)
    if not bundle:
        raise RuntimeError(f"model not found in {MODEL_DIR}")
    frame = _build_feature_frame(payload)
    x = _prepare_non_risk_features(frame, bundle)
    pred = np.clip(bundle["model"].predict(x), 0, 1)
    levels = [score_level(float(value)) for value in pred]
    return {
        "model_type": "xgboost_non_risk_simulation_parameters",
        "model_file": str(MODEL_DIR / "xgboost_non_risk_operation_risk.joblib"),
        "predictions": [
            {"ai_operation_risk": float(value), "ai_operation_level": level}
            for value, level in zip(pred, levels)
        ],
    }


class ModelApiHandler(BaseHTTPRequestHandler):
    def _headers(self, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._headers()
        self.wfile.write(b"{}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path != "/health":
            self._headers(404)
            self.wfile.write(json.dumps({"ok": False, "error": "not_found"}, ensure_ascii=False).encode("utf-8"))
            return
        bundle = load_xgboost_model(MODEL_DIR)
        self._headers(200 if bundle else 503)
        self.wfile.write(
            json.dumps(
                {
                    "ok": bool(bundle),
                    "model_dir": str(MODEL_DIR),
                    "raw_feature_count": len(bundle.get("raw_features", [])) if bundle else 0,
                },
                ensure_ascii=False,
            ).encode("utf-8")
        )

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/predict":
            self._headers(404)
            self.wfile.write(json.dumps({"ok": False, "error": "not_found"}, ensure_ascii=False).encode("utf-8"))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = predict_payload(payload)
            self._headers(200)
            self.wfile.write(json.dumps({"ok": True, **result}, ensure_ascii=False).encode("utf-8"))
        except Exception as exc:
            self._headers(500)
            self.wfile.write(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), ModelApiHandler)
    print(f"XGBoost model API running at http://{HOST}:{PORT}")
    print(f"Health check: http://{HOST}:{PORT}/health")
    server.serve_forever()


if __name__ == "__main__":
    main()
