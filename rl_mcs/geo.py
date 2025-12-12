"""地理计算与Voronoi区域查找的工具函数。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import List, Optional, Tuple

from shapely.geometry import Point, shape


@dataclass
class RegionPolygon:
    """轻量化的Voronoi区域表示。"""

    region_id: str
    polygon: object  # shapely几何对象

    def contains(self, lon: float, lat: float) -> bool:
        return self.polygon.contains(Point(lon, lat))


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """计算两点间的大圆距离（公里）。"""

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def load_voronoi_regions(path: Path) -> List[RegionPolygon]:
    """从GeoJSON文件加载Voronoi多边形。"""

    regions: List[RegionPolygon] = []
    with path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)
    for feature in geojson["features"]:
        region_id = str(feature["properties"].get("id", feature["properties"].get("region")))
        polygon = shape(feature["geometry"])
        regions.append(RegionPolygon(region_id=region_id, polygon=polygon))
    return regions


def locate_region(regions: List[RegionPolygon], lon: float, lat: float) -> Optional[str]:
    """返回包含指定点的区域ID，若无则为None。"""

    point = Point(lon, lat)
    for region in regions:
        if region.polygon.contains(point):
            return region.region_id
    return None


def nearest_points_within(points: List[Tuple[float, float]], lon: float, lat: float, radius_km: float) -> List[Tuple[int, float]]:
    """返回半径范围内候选点的索引与距离。"""

    candidates: List[Tuple[int, float]] = []
    for idx, (lon2, lat2) in enumerate(points):
        dist = haversine_km(lon, lat, lon2, lat2)
        if dist <= radius_km:
            candidates.append((idx, dist))
    candidates.sort(key=lambda x: x[1])
    return candidates
