"""云端环境，用于跨区域协调MCS配额。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import CloudConfig, RegionSummary


@dataclass
class CloudObservation:
    summaries: List[RegionSummary]


class CloudEnv:
    """跨区域分配MCS数量的简化环境。"""

    def __init__(self, config: CloudConfig, region_ids: List[str]):
        self.config = config
        self.region_ids = region_ids
        self.allocations: Dict[str, int] = {rid: 1 for rid in region_ids}
        self.time_step = 0

    def observe(self, summaries: List[RegionSummary]) -> CloudObservation:
        return CloudObservation(summaries=summaries)

    def step(self, action: Dict[str, int], summaries: List[RegionSummary]) -> Tuple[CloudObservation, float, bool, Dict]:
        """根据区域统计调整配额，并计算对应奖励。"""

        reward = 0.0
        info: Dict[str, float] = {}
        for rid, delta in action.items():
            new_alloc = max(0, self.allocations.get(rid, 0) + delta)
            transfer_cost = abs(delta) * 0.1
            reward -= transfer_cost
            self.allocations[rid] = new_alloc

        # 对成功率高、等待低的区域给予更高奖励
        for summary in summaries:
            reward += summary.success_rate * 2.0
            reward -= summary.average_wait * 0.05
            info[f"wait_{summary.region_id}"] = summary.average_wait

        self.time_step += 1
        obs = self.observe(summaries)
        done = False
        return obs, reward, done, info

    def greedy_action(self, summaries: List[RegionSummary]) -> Dict[str, int]:
        """简单基线：将资源从低等待区域迁移至高等待区域。"""

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
