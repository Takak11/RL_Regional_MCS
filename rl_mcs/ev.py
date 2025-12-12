"""EV state machine and request generation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from .config import EVConfig
from .data import TrajectoryPoint, iterate_trips
from .geo import haversine_km


@dataclass
class EVState:
    vehicle_id: str
    soc: float  # normalized 0-1
    lon: float
    lat: float
    region_id: Optional[str]
    waiting: bool = False
    queue_start_ts: Optional[str] = None
    history: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.history.append(message)


@dataclass
class ChargeRequest:
    vehicle_id: str
    lon: float
    lat: float
    region_id: Optional[str]
    timestamp: str
    soc: float


class EVSimulator:
    """Drive vehicles along their trajectories and yield charge requests."""

    def __init__(self, ev_config: EVConfig):
        self.ev_config = ev_config

    def run_vehicle(self, vehicle_id: str, points: List[TrajectoryPoint], locate_region_fn) -> Iterator[ChargeRequest]:
        soc = 1.0
        region_id = locate_region_fn(points[0].lon, points[0].lat)
        state = EVState(vehicle_id=vehicle_id, soc=soc, lon=points[0].lon, lat=points[0].lat, region_id=region_id)
        for prev_pt, next_pt in iterate_trips(points):
            distance = haversine_km(prev_pt.lon, prev_pt.lat, next_pt.lon, next_pt.lat)
            energy_used = distance * self.ev_config.energy_kwh_per_km
            soc -= energy_used / self.ev_config.battery_capacity_kwh
            soc = max(soc, 0.0)
            state.soc = soc
            state.lon, state.lat = next_pt.lon, next_pt.lat
            state.region_id = locate_region_fn(next_pt.lon, next_pt.lat)
            if soc <= self.ev_config.soc_threshold:
                yield ChargeRequest(
                    vehicle_id=vehicle_id,
                    lon=next_pt.lon,
                    lat=next_pt.lat,
                    region_id=state.region_id,
                    timestamp=next_pt.timestamp,
                    soc=soc,
                )

    def stream_requests(self, trajectories, locate_region_fn) -> Iterator[ChargeRequest]:
        for traj in trajectories:
            if not traj.points:
                continue
            yield from self.run_vehicle(traj.vehicle_id, traj.points, locate_region_fn)
