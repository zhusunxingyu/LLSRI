"""Output writers for summaries, second-level data, logs, and typical cases."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd


def ensure_outputs(root: Path) -> dict[str, Path]:
    paths = {
        "summary": root / "outputs" / "summary",
        "parquet": root / "outputs" / "timeseries_parquet",
        "typical": root / "outputs" / "timeseries_csv_typical",
        "figures": root / "outputs" / "figures",
        "maps": root / "outputs" / "maps",
        "logs": root / "outputs" / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_timeseries(ts: pd.DataFrame, out_dir: Path, experiment_id: int) -> Path:
    """Write Parquet if pyarrow is available; otherwise write compressed CSV with .parquet suffix note."""

    path = out_dir / f"task_{experiment_id:04d}.parquet"
    try:
        ts.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = out_dir / f"task_{experiment_id:04d}.csv.gz"
        ts.to_csv(fallback, index=False, encoding="utf-8-sig", compression="gzip")
        return fallback


def write_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    summary.to_csv(out_dir / "shanghai_map_experiment_summary.csv", index=False, encoding="utf-8-sig")
    try:
        summary.to_excel(out_dir / "shanghai_map_experiment_summary.xlsx", index=False)
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary.to_excel(out_dir / f"shanghai_map_experiment_summary_{stamp}.xlsx", index=False)


def write_warning_log(all_ts: list[pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    logs = pd.concat(
        [
            frame.loc[
                frame.warning_level.ne("正常"),
                ["experiment_id", "timestamp", "longitude", "latitude", "city_zone", "llsri", "warning_level", "warning_reason"],
            ]
            for frame in all_ts
        ],
        ignore_index=True,
    )
    logs.to_csv(out_dir / "warning_log.csv", index=False, encoding="utf-8-sig")
    return logs


def write_typical_cases(summary: pd.DataFrame, timeseries: dict[int, pd.DataFrame], out_dir: Path) -> list[int]:
    picks: list[int] = []
    bands = [
        ("low", summary.sort_values("avg_llsri").head(2)),
        ("medium", summary.iloc[(summary.avg_llsri - summary.avg_llsri.median()).abs().sort_values().index].head(2)),
        ("high", summary.sort_values("avg_llsri", ascending=False).head(2)),
    ]
    for label, rows in bands:
        for _, row in rows.iterrows():
            exp_id = int(row.experiment_id)
            if exp_id in timeseries:
                timeseries[exp_id].to_csv(out_dir / f"{label}_risk_task_{exp_id:04d}.csv", index=False, encoding="utf-8-sig")
                picks.append(exp_id)
    return picks
