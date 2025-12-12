"""云边协同训练的调度器。"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .agents import CloudPolicy, EdgePolicy
from .cloud_env import CloudEnv
from .config import SimulationConfig, TrainingSchedule
from .edge_env import EdgeEnv

logger = logging.getLogger(__name__)


@dataclass
class RolloutBuffer:
    trajectories: List[dict] = field(default_factory=list)

    def add(self, **kwargs) -> None:
        self.trajectories.append(kwargs)

    def clear(self) -> None:
        self.trajectories.clear()


class Trainer:
    """利用给定策略协调边缘与云端的采样与更新。"""

    def __init__(self, sim_config: SimulationConfig, schedule: TrainingSchedule):
        self.sim_config = sim_config
        self.schedule = schedule
        self.edge_policies: Dict[str, EdgePolicy] = {}
        self.edge_envs: Dict[str, EdgeEnv] = {}
        self.cloud_policy = CloudPolicy(config=sim_config.cloud)
        self.cloud_env = CloudEnv(config=sim_config.cloud, region_ids=list(sim_config.region_ids or []))
        self.edge_buffers: Dict[str, RolloutBuffer] = defaultdict(RolloutBuffer)
        self.cloud_buffer = RolloutBuffer()

    def register_region(self, region_id: str, env: EdgeEnv, policy: EdgePolicy) -> None:
        self.edge_envs[region_id] = env
        self.edge_policies[region_id] = policy

    def edge_rollout(self, region_id: str) -> None:
        env = self.edge_envs[region_id]
        policy = self.edge_policies[region_id]
        obs = env.observe()
        actions = policy.act(obs)
        new_obs, reward, done, info = env.step(actions)
        self.edge_buffers[region_id].add(obs=obs, actions=actions, reward=reward, new_obs=new_obs, done=done, info=info)

    def cloud_rollout(self) -> None:
        summaries = [env.build_summary() for env in self.edge_envs.values()]
        cloud_obs = self.cloud_env.observe(summaries)
        action = self.cloud_policy.act(cloud_obs)
        new_obs, reward, done, info = self.cloud_env.step(action, summaries)
        self.cloud_buffer.add(obs=cloud_obs, action=action, reward=reward, new_obs=new_obs, done=done, info=info)

    def train(self) -> None:
        logger.info("Starting training for %d iterations", self.schedule.max_iterations)
        for step in range(self.schedule.max_iterations):
            for region_id in self.edge_envs:
                self.edge_rollout(region_id)
            if step % self.sim_config.cloud.allocation_interval == 0:
                self.cloud_rollout()

            if step % self.schedule.cloud_update_every == 0 and step > 0:
                self._update_cloud()
            if step % self.schedule.edge_sync_every == 0 and step > 0:
                self._update_edges()

            if step % self.schedule.evaluation_interval == 0 and step > 0:
                logger.info("Evaluation checkpoint at step %d", step)
            if step % self.schedule.save_interval == 0 and step > 0:
                logger.info("Saving checkpoint at step %d", step)

    def _update_edges(self) -> None:
        for region_id, buffer in self.edge_buffers.items():
            metrics = self.edge_policies[region_id].update(buffer.trajectories)
            logger.debug("Edge policy update %s: %s", region_id, metrics)
            buffer.clear()

    def _update_cloud(self) -> None:
        metrics = self.cloud_policy.update(self.cloud_buffer.trajectories)
        logger.debug("Cloud policy update: %s", metrics)
        self.cloud_buffer.clear()
