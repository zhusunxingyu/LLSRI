"""Real GIS building loading and route feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BUILDING_COLUMNS = ["nearest_building_distance", "building_density_300m", "gis_building_source"]


def _deps_available() -> bool:
    try:
        import geopandas  # noqa: F401
        import shapely  # noqa: F401
        import pyproj  # noqa: F401
    except Exception:
        return False
    return True


def load_or_download_buildings(map_dir: Path, force: bool = False):
    """Load local building GeoJSON, otherwise try a bounded OSMnx download."""

    if not _deps_available():
        return None, "gis_dependencies_missing"
    import geopandas as gpd

    map_dir.mkdir(parents=True, exist_ok=True)
    path = map_dir / "buildings.geojson"
    if path.exists() and not force:
        try:
            buildings = gpd.read_file(path)
            return _prepare_buildings(buildings), "local_buildings_geojson"
        except Exception:
            pass

    try:
        import osmnx as ox

        # Core Shanghai bbox keeps the first real-GIS run practical. Users can
        # replace this cache with a full Shanghai buildings.geojson anytime.
        west, south, east, north = 121.32, 31.10, 121.66, 31.36
        try:
            buildings = ox.features_from_bbox((west, south, east, north), tags={"building": True})
        except TypeError:
            buildings = ox.features.features_from_bbox((west, south, east, north), tags={"building": True})
        buildings = buildings.reset_index(drop=True)
        if len(buildings) == 0:
            return None, "osm_buildings_empty"
        buildings.to_file(path, driver="GeoJSON")
        return _prepare_buildings(buildings), "osm_buildings_cached"
    except Exception as exc:
        return None, f"osm_buildings_failed:{type(exc).__name__}"


def _prepare_buildings(buildings):
    import geopandas as gpd

    if buildings.crs is None:
        buildings = buildings.set_crs("EPSG:4326")
    buildings = buildings[~buildings.geometry.is_empty & buildings.geometry.notna()].copy()
    buildings = buildings[buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings = buildings.to_crs("EPSG:3857")
    polygons = buildings[["geometry"]].copy()
    centroids = gpd.GeoDataFrame(geometry=buildings.geometry.centroid, crs=buildings.crs)
    return {"polygons": polygons, "centroids": centroids}


def extract_building_features(route: pd.DataFrame, buildings_cache) -> pd.DataFrame | None:
    """Calculate nearest building distance and 300m density from real polygons."""

    if buildings_cache is None or not _deps_available():
        return None
    import geopandas as gpd

    points = gpd.GeoDataFrame(route.copy(), geometry=gpd.points_from_xy(route.longitude, route.latitude), crs="EPSG:4326").to_crs("EPSG:3857")
    polygons = buildings_cache["polygons"]
    centroids = buildings_cache["centroids"]
    try:
        nearest = gpd.sjoin_nearest(points[["geometry"]], polygons, how="left", distance_col="dist_m")
        nearest_dist = nearest.groupby(level=0)["dist_m"].min().reindex(range(len(points))).fillna(220).to_numpy()
    except Exception:
        nearest_dist = np.full(len(points), 220.0)

    counts: list[int] = []
    spatial_index = centroids.sindex
    for point in points.geometry:
        candidates = list(spatial_index.query(point.buffer(300), predicate="intersects"))
        if not candidates:
            counts.append(0)
            continue
        subset = centroids.iloc[candidates]
        counts.append(int((subset.distance(point) <= 300).sum()))
    density = np.clip(np.array(counts, dtype=float) / 180.0, 0, 1)
    return pd.DataFrame(
        {
            "nearest_building_distance": np.clip(nearest_dist, 0, 300),
            "building_density_300m": density,
            "gis_building_source": "real_gis",
        }
    )
