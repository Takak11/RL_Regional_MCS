"""Region-level environment for MCS dispatch decisions."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import EdgeConfig, RegionSummary
from .ev import ChargeRequest
from .geo import nearest_points_within


@dataclass
class QueueItem:
    request: ChargeRequest
    wait_time: int = 0


@dataclass
class MCSState:
    """Mobile charging station representation."""

    lon: float
    lat: float
    available: bool = True
    battery_kwh: float = 80.0


@dataclass
class EdgeObservation:
    region_id: str
    pending_requests: int
    mean_wait: float
    max_wait: float
    available_mcs: int
    time_bin: int
    arrival_rate: float
    candidate_points: List[Tuple[float, float]]


class EdgeEnv:
    """Simplified queueing environment for a single region."""

    def __init__(self, region_id: str, config: EdgeConfig, dispatch_points: List[Tuple[float, float]]):
        self.region_id = region_id
        self.config = config
        self.dispatch_points = dispatch_points
        self.queue: List[QueueItem] = []
        self.time_step = 0
        self.arrivals_last_window = 0
        self.mcs_pool: List[MCSState] = [MCSState(lon=lon, lat=lat) for lon, lat in dispatch_points]

    def observe(self) -> EdgeObservation:
        waits = [item.wait_time for item in self.queue] or [0]
        mean_wait = sum(waits) / len(waits)
        max_wait = max(waits)
        time_bin = (self.time_step // 12) % 24  # coarse 2-hr bin if 5-min step
        arrival_rate = self.arrivals_last_window / max(1, self.time_step)
        return EdgeObservation(
            region_id=self.region_id,
            pending_requests=len(self.queue),
            mean_wait=mean_wait,
            max_wait=max_wait,
            available_mcs=sum(1 for m in self.mcs_pool if m.available),
            time_bin=time_bin,
            arrival_rate=arrival_rate,
            candidate_points=self.dispatch_points,
        )

    def add_request(self, request: ChargeRequest) -> None:
        if len(self.queue) >= self.config.max_queue_size:
            return
        self.queue.append(QueueItem(request=request))
        self.arrivals_last_window += 1

    def step(self, action_indices: Optional[List[int]]) -> Tuple[EdgeObservation, float, bool, Dict]:
        """Advance the queue by assigning requests to dispatch points.

        Args:
            action_indices: indices into candidate_points for each queued request.
        """

        reward = 0.0
        info: Dict[str, float] = {}

        # update waiting times
        for item in self.queue:
            item.wait_time += 1

        if action_indices:
            for item, idx in zip(self.queue, action_indices):
                if idx is None or idx >= len(self.dispatch_points):
                    continue
                point = self.dispatch_points[idx]
                nearest = nearest_points_within([point], item.request.lon, item.request.lat, self.config.region_radius_km)
                if nearest:
                    # pretend immediate service if available MCS exists
                    if self._assign_mcs(point):
                        reward += 1.0 - 0.01 * item.wait_time
                        item.wait_time = 0
                    else:
                        reward -= 0.1  # penalty for no MCS
                else:
                    reward -= 0.2  # too far

        # Remove served items
        self.queue = [item for item in self.queue if item.wait_time > 0]
        self.time_step += 1

        obs = self.observe()
        done = False
        return obs, reward, done, info

    def _assign_mcs(self, target_point: Tuple[float, float]) -> bool:
        for mcs in self.mcs_pool:
            if mcs.available:
                mcs.available = False
                # simplistic recharge cooldown
                mcs.battery_kwh -= 5
                if mcs.battery_kwh < 10:
                    mcs.available = False
                else:
                    mcs.available = True
                mcs.lon, mcs.lat = target_point
                return True
        return False

    def build_summary(self) -> RegionSummary:
        success_rate = 0.0
        average_wait = 0.0
        if self.queue:
            waits = [item.wait_time for item in self.queue]
            average_wait = sum(waits) / len(waits)
        return RegionSummary(
            region_id=self.region_id,
            success_rate=success_rate,
            average_wait=average_wait,
            arrival_rate=self.arrivals_last_window / max(1, self.time_step),
            available_mcs=sum(1 for m in self.mcs_pool if m.available),
            queue_length=len(self.queue),
        )

    def reset_window(self) -> None:
        self.arrivals_last_window = 0
