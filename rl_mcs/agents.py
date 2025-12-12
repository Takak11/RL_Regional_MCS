"""边缘与云端策略的轻量级占位实现。"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .cloud_env import CloudObservation
from .config import CloudConfig, EdgeConfig
from .edge_env import EdgeObservation


@dataclass
class EdgePolicy:
    config: EdgeConfig

    def act(self, obs: EdgeObservation) -> List[int]:
        """为每个排队请求选择调度点索引，当前直接选最近候选。"""

        actions: List[int] = []
        num_requests = obs.pending_requests
        for _ in range(num_requests):
            if not obs.candidate_points:
                actions.append(None)  # type: ignore  # 无可选调度点时输出空动作
                continue
            actions.append(0)
        return actions

    def update(self, batch) -> Dict[str, float]:  # 占位接口
        return {"loss": 0.0}


@dataclass
class CloudPolicy:
    config: CloudConfig

    def act(self, obs: CloudObservation) -> Dict[str, int]:
        """将更多MCS分配给平均等待时间较高的区域。"""

        if not obs.summaries:
            return {}
        sorted_regions = sorted(obs.summaries, key=lambda s: s.average_wait, reverse=True)
        action: Dict[str, int] = {}
        for idx, summary in enumerate(sorted_regions):
            if idx == 0:
                action[summary.region_id] = min(self.config.max_transfer_per_interval, 1)
            elif idx == len(sorted_regions) - 1:
                action[summary.region_id] = -1
        return action

    def update(self, batch) -> Dict[str, float]:
        return {"loss": 0.0}
