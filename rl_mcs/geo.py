"""Utilities for geographic calculations and Voronoi region lookup."""
from __future__ import annotations

import json
from dataclasses import dataclass
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import List, Optional, Tuple

from shapely.geometry import Point, shape


@dataclass
class RegionPolygon:
    """Lightweight Voronoi region representation."""

    region_id: str
    polygon: object  # shapely geometry

    def contains(self, lon: float, lat: float) -> bool:
        return self.polygon.contains(Point(lon, lat))


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Compute great-circle distance in kilometers between two points."""

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def load_voronoi_regions(path: Path) -> List[RegionPolygon]:
    """Load Voronoi polygons from a GeoJSON file."""

    regions: List[RegionPolygon] = []
    with path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)
    for feature in geojson["features"]:
        region_id = str(feature["properties"].get("id", feature["properties"].get("region")))
        polygon = shape(feature["geometry"])
        regions.append(RegionPolygon(region_id=region_id, polygon=polygon))
    return regions


def locate_region(regions: List[RegionPolygon], lon: float, lat: float) -> Optional[str]:
    """Return the ID of the region that contains the point, if any."""

    point = Point(lon, lat)
    for region in regions:
        if region.polygon.contains(point):
            return region.region_id
    return None


def nearest_points_within(points: List[Tuple[float, float]], lon: float, lat: float, radius_km: float) -> List[Tuple[int, float]]:
    """Return indices and distances of candidate points within a radius."""

    candidates: List[Tuple[int, float]] = []
    for idx, (lon2, lat2) in enumerate(points):
        dist = haversine_km(lon, lat, lon2, lat2)
        if dist <= radius_km:
            candidates.append((idx, dist))
    candidates.sort(key=lambda x: x[1])
    return candidates
