"""Map loading with local, online, and synthetic fallback layers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ZoneProfile:
    name: str
    label: str
    center_lon: float
    center_lat: float
    building_density: float
    poi_density: float
    residential: float
    commercial: float
    transport: float
    sensitive: float
    compliance: float
    water: float


ZONE_PROFILES: list[ZoneProfile] = [
    ZoneProfile("core_business", "核心商务区", 121.50, 31.24, 0.92, 0.95, 0.55, 0.96, 0.62, 0.30, 0.32, 0.20),
    ZoneProfile("residential", "居住区", 121.43, 31.21, 0.58, 0.62, 0.95, 0.45, 0.35, 0.35, 0.25, 0.10),
    ZoneProfile("hospital_school", "医院学校敏感区", 121.45, 31.20, 0.50, 0.72, 0.76, 0.50, 0.42, 0.98, 0.45, 0.06),
    ZoneProfile("transport_hub", "交通枢纽区", 121.46, 31.25, 0.70, 0.82, 0.45, 0.75, 0.98, 0.45, 0.86, 0.08),
    ZoneProfile("industrial_park", "工业园区", 121.62, 31.20, 0.42, 0.40, 0.25, 0.42, 0.35, 0.16, 0.18, 0.05),
    ZoneProfile("riverfront", "滨水/跨江区", 121.49, 31.23, 0.48, 0.58, 0.42, 0.70, 0.50, 0.22, 0.38, 0.95),
]

ZONE_ANCHORS: dict[str, list[tuple[float, float]]] = {
    "core_business": [(121.5020, 31.2363), (121.4752, 31.2288), (121.4567, 31.2297), (121.4906, 31.2417), (121.4375, 31.1883), (121.5146, 31.3037)],
    "residential": [(121.4300, 31.2100), (121.3797, 31.3450), (121.5120, 31.0830), (121.4096, 31.2287), (121.3932, 31.3158), (121.4740, 30.9180)],
    "hospital_school": [(121.5030, 31.2989), (121.4708, 31.2116), (121.4530, 31.1973), (121.4467, 31.2166), (121.5426, 31.2227), (121.4310, 31.0252)],
    "transport_hub": [(121.3279, 31.2000), (121.4559, 31.2495), (121.4302, 31.1546), (121.8052, 31.1434), (121.3350, 31.1979), (121.4007, 31.2628)],
    "industrial_park": [(121.5870, 31.2019), (121.6105, 31.2572), (121.4055, 31.1706), (121.9000, 30.8900), (121.2450, 31.3810), (121.3750, 31.0060)],
    "riverfront": [(121.5055, 31.2315), (121.5422, 31.2636), (121.4599, 31.1755), (121.3933, 31.2309), (121.5010, 31.3926), (121.9350, 30.9086)],
}


def load_map_context(map_dir: Path) -> pd.DataFrame:
    """Return zone profiles; read local metadata if present, otherwise fallback.

    The project is designed to accept real GIS files later. In this runnable
    version, the local-file check is kept explicit and synthetic proxy variables
    preserve the expected Shanghai spatial-risk behavior.
    """

    map_dir.mkdir(parents=True, exist_ok=True)
    required = ["buildings.geojson", "pois.geojson", "roads.geojson", "water.geojson", "city_zones.geojson"]
    mode = "local_files" if all((map_dir / name).exists() for name in required) else "synthetic_fallback"
    rows = [profile.__dict__ | {"map_mode": mode} for profile in ZONE_PROFILES]
    return pd.DataFrame(rows)


def sample_point_in_zone(zone: pd.Series, rng: np.random.Generator, radius_deg: float = 0.025) -> tuple[float, float]:
    """Sample a longitude/latitude near a zone center."""

    anchors = ZONE_ANCHORS.get(str(zone["name"]), [(float(zone.center_lon), float(zone.center_lat))])
    anchor_lon, anchor_lat = anchors[int(rng.integers(0, len(anchors)))]
    return (
        float(anchor_lon + rng.normal(0, radius_deg * 0.55)),
        float(anchor_lat + rng.normal(0, radius_deg * 0.40)),
    )


def to_xy(lon: float, lat: float, origin_lon: float = 121.4737, origin_lat: float = 31.2304) -> tuple[float, float]:
    """Approximate Shanghai lon/lat to meter offsets for simulation distances."""

    x = (lon - origin_lon) * 111_320 * np.cos(np.deg2rad(origin_lat))
    y = (lat - origin_lat) * 110_540
    return float(x), float(y)


def interpolate_route(
    start: tuple[float, float],
    end: tuple[float, float],
    steps: int,
    route_type: str = "direct",
    route_complexity: str = "medium",
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Create a route with one or more city-style waypoints."""

    rng = rng or np.random.default_rng(0)
    start_arr = np.array(start, dtype=float)
    end_arr = np.array(end, dtype=float)
    delta = end_arr - start_arr
    norm = float(np.linalg.norm(delta)) or 1e-6
    perpendicular = np.array([-delta[1], delta[0]]) / norm
    complexity_points = {"low": 1, "medium": 2, "high": 3}.get(route_complexity, 2)
    if route_type == "direct" and route_complexity == "low":
        waypoint_count = 1
        bend_scale = 0.20
    elif route_type == "detour":
        waypoint_count = complexity_points + 1
        bend_scale = 0.85
    elif route_type == "river_crossing":
        waypoint_count = complexity_points + 1
        bend_scale = 0.65
    elif route_type == "corridor":
        waypoint_count = complexity_points
        bend_scale = 0.42
    else:
        waypoint_count = complexity_points
        bend_scale = 0.30

    waypoints = [start_arr]
    for i in range(1, waypoint_count + 1):
        frac = i / (waypoint_count + 1)
        base = start_arr + delta * frac
        alternating = -1 if i % 2 == 0 else 1
        offset = perpendicular * norm * bend_scale * alternating * rng.uniform(0.35, 1.0)
        jitter = rng.normal(0, 0.004, 2)
        waypoints.append(base + offset + jitter)
    waypoints.append(end_arr)

    segment_lengths = [float(np.linalg.norm(waypoints[i + 1] - waypoints[i])) for i in range(len(waypoints) - 1)]
    total_length = sum(segment_lengths) or 1e-6
    counts = [max(2, int(round(steps * length / total_length))) for length in segment_lengths]
    while sum(counts) > steps + len(counts) - 1:
        counts[counts.index(max(counts))] -= 1

    lons: list[float] = []
    lats: list[float] = []
    for i, count in enumerate(counts):
        a = waypoints[i]
        b = waypoints[i + 1]
        include_endpoint = i == len(counts) - 1
        segment_lon = np.linspace(a[0], b[0], count, endpoint=include_endpoint)
        segment_lat = np.linspace(a[1], b[1], count, endpoint=include_endpoint)
        if i:
            segment_lon = segment_lon[1:]
            segment_lat = segment_lat[1:]
        lons.extend(segment_lon.tolist())
        lats.extend(segment_lat.tolist())

    if len(lons) < steps:
        lons.extend([end[0]] * (steps - len(lons)))
        lats.extend([end[1]] * (steps - len(lats)))
    lon = np.array(lons[:steps])
    lat = np.array(lats[:steps])
    xy = np.array([to_xy(a, b) for a, b in zip(lon, lat)])
    return pd.DataFrame({"longitude": lon, "latitude": lat, "x": xy[:, 0], "y": xy[:, 1]})
