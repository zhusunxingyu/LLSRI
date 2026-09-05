"""HTML map and lightweight chart generation."""

from __future__ import annotations

import html
import json
from pathlib import Path
import struct
import zlib

import pandas as pd


LEVEL_COLORS = {"正常": "#2e7d32", "蓝色": "#1976d2", "黄色": "#f9a825", "橙色": "#ef6c00", "红色": "#c62828"}


def write_route_map(ts: pd.DataFrame, out_dir: Path, name: str = "typical_route_map") -> Path:
    """Write a real online tile map with Leaflet; no Python map dependency required."""

    center_lat = float(ts.latitude.mean())
    center_lon = float(ts.longitude.mean())
    coords = ts[["latitude", "longitude", "llsri", "warning_level", "warning_reason"]].round(6).to_dict("records")
    line = [[row["latitude"], row["longitude"]] for row in coords]
    points = [
        {"lat": row["latitude"], "lon": row["longitude"], "level": row["warning_level"], "color": LEVEL_COLORS[row["warning_level"]], "risk": row["llsri"], "reason": row["warning_reason"]}
        for row in coords[:: max(1, len(coords) // 30)]
    ]
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>上海低空物流无人机轨迹地图</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>body{{margin:0;font-family:Arial,'Microsoft YaHei',sans-serif}}#map{{height:100vh}}.panel{{position:absolute;z-index:999;top:14px;left:14px;background:#fff;padding:12px 14px;border-radius:8px;box-shadow:0 8px 24px #0002;line-height:1.5}}</style>
</head>
<body>
<div id="map"></div>
<div class="panel"><b>上海真实底图接入版</b><br>OSM 实景瓦片底图 + 仿真轨迹风险点<br>任务 {html.escape(str(ts.experiment_id.iloc[0]))}</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map('map').setView([{center_lat:.6f}, {center_lon:.6f}], 12);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}}).addTo(map);
const line = {json.dumps(line, ensure_ascii=False)};
L.polyline(line, {{color:'#1565c0', weight:4, opacity:.85}}).addTo(map);
const points = {json.dumps(points, ensure_ascii=False)};
for (const p of points) {{
  L.circleMarker([p.lat, p.lon], {{radius:6, color:p.color, fillColor:p.color, fillOpacity:.85}})
    .bindPopup(`风险:${{p.risk.toFixed(3)}}<br>等级:${{p.level}}<br>${{p.reason}}`).addTo(map);
}}
</script>
</body></html>"""
    path = out_dir / f"{name}.html"
    path.write_text(page, encoding="utf-8")
    return path


def write_warning_overview_map(logs: pd.DataFrame, out_dir: Path, name: str = "all_warning_points_real_tile_map") -> Path:
    """Write an overview map sampled from warnings across all experiments."""

    sample = logs.copy()
    if len(sample) > 1800:
        sample = sample.sample(1800, random_state=20260728)
    center_lat = float(sample.latitude.mean()) if len(sample) else 31.2304
    center_lon = float(sample.longitude.mean()) if len(sample) else 121.4737
    points = [
        {
            "lat": float(row.latitude),
            "lon": float(row.longitude),
            "level": str(row.warning_level),
            "color": LEVEL_COLORS.get(str(row.warning_level), "#607d8b"),
            "risk": float(row.llsri),
            "zone": str(row.city_zone),
            "reason": str(row.warning_reason),
        }
        for row in sample.itertuples(index=False)
    ]
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>上海全任务预警点总览</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>body{{margin:0;font-family:Arial,'Microsoft YaHei',sans-serif}}#map{{height:100vh}}.panel{{position:absolute;z-index:999;top:14px;left:14px;background:#fff;padding:12px 14px;border-radius:8px;box-shadow:0 8px 24px #0002;line-height:1.5}}</style>
</head>
<body>
<div id="map"></div>
<div class="panel"><b>上海全任务预警点总览</b><br>抽样展示 {len(points)} 个预警点<br>颜色代表预警等级，底图为 OSM 实景瓦片</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map('map').setView([{center_lat:.6f}, {center_lon:.6f}], 10);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}}).addTo(map);
const points = {json.dumps(points, ensure_ascii=False)};
for (const p of points) {{
  L.circleMarker([p.lat, p.lon], {{radius:4, color:p.color, fillColor:p.color, fillOpacity:.55, weight:1}})
    .bindPopup(`功能区:${{p.zone}}<br>风险:${{p.risk.toFixed(3)}}<br>等级:${{p.level}}<br>${{p.reason}}`).addTo(map);
}}
</script>
</body></html>"""
    path = out_dir / f"{name}.html"
    path.write_text(page, encoding="utf-8")
    return path


def write_figures(summary: pd.DataFrame, out_dir: Path) -> None:
    """Write CSV-backed chart data and simple SVG charts."""

    charts = {
        "zone_avg_llsri.csv": summary.groupby("city_zone", as_index=False).avg_llsri.mean(),
        "zone_social_risk.csv": summary.groupby("city_zone", as_index=False).avg_social_risk.mean(),
        "altitude_social_risk.csv": summary.groupby("altitude", as_index=False).avg_social_risk.mean(),
        "drone_count_conflict.csv": summary.groupby("drone_count", as_index=False).red_warning_count.mean(),
        "warning_distribution.csv": summary.final_warning_level.value_counts().rename_axis("warning_level").reset_index(name="count"),
    }
    for name, data in charts.items():
        data.to_csv(out_dir / name, index=False, encoding="utf-8-sig")
    zone = charts["zone_avg_llsri.csv"].sort_values("avg_llsri", ascending=False)
    bars = []
    width = 720
    for i, row in enumerate(zone.itertuples(index=False)):
        bar_w = max(4, int(row.avg_llsri * 620))
        y = 34 + i * 34
        bars.append(f'<text x="10" y="{y + 16}" font-size="13">{html.escape(str(row.city_zone))}</text><rect x="165" y="{y}" width="{bar_w}" height="20" fill="#1565c0"/><text x="{170 + bar_w}" y="{y + 15}" font-size="12">{row.avg_llsri:.3f}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{70 + len(zone)*34}"><text x="10" y="24" font-size="18" font-weight="700">功能区平均 LLSRI</text>{"".join(bars)}</svg>'
    (out_dir / "zone_avg_llsri.svg").write_text(svg, encoding="utf-8")
    _write_bar_png(zone["avg_llsri"].tolist(), out_dir / "zone_avg_llsri.png")
    _write_bar_png(charts["warning_distribution.csv"]["count"].tolist(), out_dir / "warning_distribution.png", color=(198, 40, 40))


def _write_bar_png(values: list[float], path: Path, color: tuple[int, int, int] = (21, 101, 192)) -> None:
    """Write a tiny dependency-free PNG bar chart."""

    width, height = 900, 420
    pixels = bytearray([255, 255, 255] * width * height)
    max_value = max(values) if values else 1
    max_value = max(max_value, 1e-9)
    bar_gap = 18
    bar_h = max(18, int((height - 80) / max(1, len(values)) - bar_gap))
    for i, value in enumerate(values):
        y0 = 50 + i * (bar_h + bar_gap)
        x0 = 160
        x1 = min(width - 40, x0 + int((width - 220) * value / max_value))
        for y in range(y0, min(height - 30, y0 + bar_h)):
            for x in range(x0, x1):
                idx = (y * width + x) * 3
                pixels[idx : idx + 3] = bytes(color)
    raw = b"".join(b"\x00" + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)
