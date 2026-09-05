"""Map proxy features sampled along a route."""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_features(route: pd.DataFrame, zone: pd.Series, altitude: float, gps_error: float, rng: np.random.Generator, buildings_cache=None) -> pd.DataFrame:
    """Attach synthetic map-risk features that behave like GIS-derived proxies."""

    n = len(route)
    progress = np.linspace(0, np.pi, n)
    noise = rng.normal(0, 0.035, n)
    density = np.clip(zone.building_density + 0.08 * np.sin(progress) + noise, 0, 1)
    poi_density = np.clip(zone.poi_density + 0.06 * np.cos(progress * 1.4) + rng.normal(0, 0.03, n), 0, 1)
    nearest_building = np.clip(85 - 70 * density - 1.5 * gps_error + rng.normal(0, 8, n), 4, 220)
    sensitive_base = 650 * (1 - zone.sensitive) + 40
    nearest_school = np.clip(sensitive_base + rng.normal(0, 60, n), 20, 1500)
    nearest_hospital = np.clip(sensitive_base * 1.15 + rng.normal(0, 75, n), 20, 1500)
    sensitive_count = np.clip(np.rint(zone.sensitive * 5 + rng.normal(0, 0.8, n)), 0, 8)
    river_crossing = ((zone.water > 0.6) & (np.sin(progress) > 0.15)).astype(int)
    low_altitude = max(0.0, (90 - altitude) / 90)
    features = pd.DataFrame(
        {
            "nearest_building_distance": nearest_building,
            "building_density_300m": density,
            "nearest_school_distance": nearest_school,
            "nearest_hospital_distance": nearest_hospital,
            "sensitive_poi_count_300m": sensitive_count,
            "poi_density_300m": poi_density,
            "residential_exposure": np.clip(zone.residential * (0.82 + 0.22 * low_altitude) + rng.normal(0, 0.025, n), 0, 1),
            "commercial_exposure": np.clip(zone.commercial * (0.85 + 0.15 * low_altitude) + rng.normal(0, 0.025, n), 0, 1),
            "transport_hub_exposure": np.clip(zone.transport * (0.85 + 0.18 * low_altitude) + rng.normal(0, 0.025, n), 0, 1),
            "river_crossing": river_crossing,
            "compliance_risk": np.clip(zone.compliance + 0.18 * river_crossing + rng.normal(0, 0.035, n), 0, 1),
            "gis_building_source": "synthetic_proxy",
        }
    )
    if buildings_cache is not None:
        try:
            from .gis_building_loader import extract_building_features

            real = extract_building_features(route, buildings_cache)
            if real is not None:
                features["nearest_building_distance"] = real["nearest_building_distance"].to_numpy()
                features["building_density_300m"] = real["building_density_300m"].to_numpy()
                features["gis_building_source"] = real["gis_building_source"].to_numpy()
        except Exception:
            pass
    return features
