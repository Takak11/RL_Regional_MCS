"""轨迹数据与调度点数据的读取与辅助函数。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import pandas as pd

from .geo import haversine_km


@dataclass
class TrajectoryPoint:
    timestamp: str
    lon: float
    lat: float


@dataclass
class Trajectory:
    vehicle_id: str
    points: List[TrajectoryPoint]


def load_dispatch_points(path: Path) -> List[Dict[str, float]]:
    """从CSV文件读取调度点与固定充电桩坐标。"""

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {
                "lon": float(row.get("lon") or row.get("longitude")),
                "lat": float(row.get("lat") or row.get("latitude")),
                "region": row.get("region", ""),
            }
            for row in reader
        ]


def load_vehicle_trajectories(root: Path, region_ids: Iterable[str] | None = None) -> List[Trajectory]:
    """从指定目录读取车辆轨迹CSV文件。"""

    trajectories: List[Trajectory] = []
    files = sorted(root.glob("*.csv"))
    selected_regions = set(region_ids) if region_ids else None
    for file in files:
        df = pd.read_csv(file)
        if selected_regions is not None and "region" in df.columns:
            df = df[df["region"].isin(selected_regions)]
        points = [
            TrajectoryPoint(timestamp=str(row["timestamp"]), lon=float(row["lon"]), lat=float(row["lat"]))
            for _, row in df.iterrows()
        ]
        trajectories.append(Trajectory(vehicle_id=file.stem, points=points))
    return trajectories


def estimate_energy_kwh(points: Sequence[TrajectoryPoint], energy_per_km: float) -> float:
    """估算经过给定轨迹段需要的能耗。"""

    total = 0.0
    for idx in range(1, len(points)):
        p1, p2 = points[idx - 1], points[idx]
        total += haversine_km(p1.lon, p1.lat, p2.lon, p2.lat) * energy_per_km
    return total


def iterate_trips(points: Sequence[TrajectoryPoint]) -> Iterator[tuple[TrajectoryPoint, TrajectoryPoint]]:
    """逐段返回相邻点对，表示每个5分钟移动。"""

    for idx in range(1, len(points)):
        yield points[idx - 1], points[idx]
