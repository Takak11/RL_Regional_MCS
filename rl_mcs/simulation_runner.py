"""Executable scaffolding to wire EV simulator, edge envs, and cloud trainer."""
from __future__ import annotations

import logging
from typing import List, Tuple

from .agents import EdgePolicy
from .config import SimulationConfig, TrainingSchedule
from .data import load_dispatch_points, load_vehicle_trajectories
from .edge_env import EdgeEnv
from .ev import EVSimulator
from .geo import load_voronoi_regions, locate_region
from .trainer import Trainer

logger = logging.getLogger(__name__)


def build_region_dispatch_points(dispatch_rows, region_id: str) -> List[Tuple[float, float]]:
    return [
        (row["lon"], row["lat"])
        for row in dispatch_rows
        if not row.get("region") or row.get("region") == region_id
    ]


def build_trainer(sim_config: SimulationConfig, schedule: TrainingSchedule) -> Trainer:
    dispatch_rows = load_dispatch_points(sim_config.edge.dispatch_points_path)
    regions = load_voronoi_regions(sim_config.edge.fcs_regions_path)
    region_ids = sim_config.region_ids or [r.region_id for r in regions]
    trainer = Trainer(sim_config, schedule)

    def locate(lon: float, lat: float):
        return locate_region(regions, lon, lat)

    for region_id in region_ids:
        points = build_region_dispatch_points(dispatch_rows, region_id)
        env = EdgeEnv(region_id=region_id, config=sim_config.edge, dispatch_points=points)
        policy = EdgePolicy(config=sim_config.edge)
        trainer.register_region(region_id, env, policy)

    return trainer


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    sim_config = SimulationConfig()
    schedule = TrainingSchedule()
    trainer = build_trainer(sim_config, schedule)

    logger.info("Loading trajectories from %s", sim_config.trajectory_root)
    trajectories = load_vehicle_trajectories(sim_config.trajectory_root, sim_config.region_ids)
    ev_sim = EVSimulator(sim_config.ev)
    regions = load_voronoi_regions(sim_config.edge.fcs_regions_path)

    def locate(lon: float, lat: float):
        return locate_region(regions, lon, lat)

    # stream requests into edge environments before training
    for request in ev_sim.stream_requests(trajectories, locate):
        region_env = trainer.edge_envs.get(request.region_id)
        if region_env:
            region_env.add_request(request)

    trainer.train()


if __name__ == "__main__":
    main()
