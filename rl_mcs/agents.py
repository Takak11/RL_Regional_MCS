"""Lightweight agent stubs for edge and cloud policies."""
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
        """Select dispatch point indices for each pending request.

        Here we simply choose the nearest candidate; replace with RL inference.
        """

        actions: List[int] = []
        num_requests = obs.pending_requests
        for _ in range(num_requests):
            if not obs.candidate_points:
                actions.append(None)  # type: ignore
                continue
            actions.append(0)
        return actions

    def update(self, batch) -> Dict[str, float]:  # placeholder
        return {"loss": 0.0}


@dataclass
class CloudPolicy:
    config: CloudConfig

    def act(self, obs: CloudObservation) -> Dict[str, int]:
        """Allocate more MCS to regions with larger wait times."""

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
