"""Weather lookup for Shanghai route risk prediction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class WeatherSnapshot:
    source: str
    observed_at: str
    temperature_c: float
    apparent_temperature_c: float
    relative_humidity_pct: float
    cloud_cover_pct: float
    surface_pressure_hpa: float
    wind_speed_ms: float
    wind_gust_ms: float
    wind_direction_deg: float
    precipitation_mm: float
    rain_mm: float
    showers_mm: float
    weather_code: int
    summary: str


WEATHER_CODES = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "强毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    80: "阵雨",
    81: "中等阵雨",
    82: "强阵雨",
    95: "雷暴",
}


def get_weather(latitude: float = 31.2304, longitude: float = 121.4737, label: str = "上海") -> WeatherSnapshot:
    """Fetch current weather for a coordinate; fall back to a hot summer default."""

    try:
        import requests

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join(
                [
                    "temperature_2m",
                    "apparent_temperature",
                    "relative_humidity_2m",
                    "cloud_cover",
                    "surface_pressure",
                    "precipitation",
                    "rain",
                    "showers",
                    "weather_code",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "wind_gusts_10m",
                ]
            ),
            "wind_speed_unit": "ms",
            "timezone": "Asia/Shanghai",
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        current = response.json()["current"]
        code = int(current.get("weather_code", 0))
        return WeatherSnapshot(
            source=f"Open-Meteo/{label}",
            observed_at=str(current.get("time", datetime.now().isoformat(timespec="minutes"))),
            temperature_c=float(current.get("temperature_2m", 32.0)),
            apparent_temperature_c=float(current.get("apparent_temperature", current.get("temperature_2m", 32.0))),
            relative_humidity_pct=float(current.get("relative_humidity_2m", 65.0)),
            cloud_cover_pct=float(current.get("cloud_cover", 50.0)),
            surface_pressure_hpa=float(current.get("surface_pressure", 1013.0)),
            wind_speed_ms=float(current.get("wind_speed_10m", 4.0)),
            wind_gust_ms=float(current.get("wind_gusts_10m", 6.0)),
            wind_direction_deg=float(current.get("wind_direction_10m", 0.0)),
            precipitation_mm=float(current.get("precipitation", 0.0)),
            rain_mm=float(current.get("rain", 0.0)),
            showers_mm=float(current.get("showers", 0.0)),
            weather_code=code,
            summary=WEATHER_CODES.get(code, f"天气代码 {code}"),
        )
    except Exception:
        return WeatherSnapshot(
            source=f"fallback/{label}",
            observed_at=datetime.now().isoformat(timespec="minutes"),
            temperature_c=36.0,
            apparent_temperature_c=38.0,
            relative_humidity_pct=65.0,
            cloud_cover_pct=35.0,
            surface_pressure_hpa=1013.0,
            wind_speed_ms=4.0,
            wind_gust_ms=7.0,
            wind_direction_deg=0.0,
            precipitation_mm=0.0,
            rain_mm=0.0,
            showers_mm=0.0,
            weather_code=0,
            summary="晴/高温回退值",
        )


def get_shanghai_weather() -> WeatherSnapshot:
    """Backward-compatible Shanghai weather helper."""

    return get_weather(31.2304, 121.4737, "上海")


def weather_to_task_adjustments(weather: WeatherSnapshot) -> dict:
    """Convert weather to simulator inputs."""

    gust_level = "none"
    if weather.wind_gust_ms >= 13 or weather.precipitation_mm >= 8:
        gust_level = "strong"
    elif weather.wind_gust_ms >= 9 or weather.precipitation_mm >= 3:
        gust_level = "medium"
    elif weather.wind_gust_ms >= 5:
        gust_level = "weak"
    weather_risk = min(
        1.0,
        0.38 * min(weather.wind_speed_ms / 12, 1)
        + 0.25 * min(weather.wind_gust_ms / 16, 1)
        + 0.20 * min(weather.precipitation_mm / 8, 1)
        + 0.10 * min(max(weather.relative_humidity_pct - 70, 0) / 30, 1)
        + 0.07 * min(weather.cloud_cover_pct / 100, 1),
    )
    return {
        "wind_speed": max(0.0, min(18.0, weather.wind_speed_ms)),
        "gust_level": gust_level,
        "weather_risk": weather_risk,
        "weather_note": (
            f"{weather.summary}，气温 {weather.temperature_c:.1f}°C，体感 {weather.apparent_temperature_c:.1f}°C，"
            f"湿度 {weather.relative_humidity_pct:.0f}%，降水 {weather.precipitation_mm:.1f}mm，"
            f"风速 {weather.wind_speed_ms:.1f}m/s，阵风 {weather.wind_gust_ms:.1f}m/s，风向 {weather.wind_direction_deg:.0f}°"
        ),
    }
