"""Cloud-level environment coordinating region allocations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import CloudConfig, RegionSummary


@dataclass
class CloudObservation:
    summaries: List[RegionSummary]


class CloudEnv:
    """Simplified environment for allocating MCS counts across regions."""

    def __init__(self, config: CloudConfig, region_ids: List[str]):
        self.config = config
        self.region_ids = region_ids
        self.allocations: Dict[str, int] = {rid: 1 for rid in region_ids}
        self.time_step = 0

    def observe(self, summaries: List[RegionSummary]) -> CloudObservation:
        return CloudObservation(summaries=summaries)

    def step(self, action: Dict[str, int], summaries: List[RegionSummary]) -> Tuple[CloudObservation, float, bool, Dict]:
        """Apply allocation deltas and compute a reward based on summaries."""

        reward = 0.0
        info: Dict[str, float] = {}
        for rid, delta in action.items():
            new_alloc = max(0, self.allocations.get(rid, 0) + delta)
            transfer_cost = abs(delta) * 0.1
            reward -= transfer_cost
            self.allocations[rid] = new_alloc

        # reward high success rate and low wait
        for summary in summaries:
            reward += summary.success_rate * 2.0
            reward -= summary.average_wait * 0.05
            info[f"wait_{summary.region_id}"] = summary.average_wait

        self.time_step += 1
        obs = self.observe(summaries)
        done = False
        return obs, reward, done, info

    def greedy_action(self, summaries: List[RegionSummary]) -> Dict[str, int]:
        """Simple baseline: shift resources from low-utilization to high-wait regions."""

        if not summaries:
            return {}
        sorted_regions = sorted(summaries, key=lambda s: s.average_wait, reverse=True)
        action: Dict[str, int] = {}
        for idx, summary in enumerate(sorted_regions):
            if idx == 0:
                action[summary.region_id] = min(self.config.max_transfer_per_interval, 1)
            elif idx == len(sorted_regions) - 1:
                action[summary.region_id] = -1
        return action
