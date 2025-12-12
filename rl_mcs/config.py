"""Shared configuration dataclasses for the RL simulation and training loops."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class EVConfig:
    """Configuration for EV energy consumption and charge request rules."""

    energy_kwh_per_km: float = 0.18
    soc_threshold: float = 0.2
    battery_capacity_kwh: float = 80.0
    timestep_minutes: int = 5


@dataclass
class EdgeConfig:
    """Hyperparameters for region-level environments and agents."""

    region_radius_km: float = 2.0
    max_queue_size: int = 50
    dispatch_points_path: Path = Path("dataset/dispatch_points_400.csv")
    fcs_regions_path: Path = Path("dataset/fcs_voronoi_regions.geojson")
    replay_buffer_size: int = 50000
    batch_size: int = 128
    gamma: float = 0.99
    learning_rate: float = 3e-4
    model_hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])


@dataclass
class CloudConfig:
    """Configuration for cloud-level training and coordination."""

    allocation_interval: int = 12  # in edge timesteps (e.g., hourly if timestep is 5 minutes)
    max_transfer_per_interval: int = 5
    ppo_clip: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    learning_rate: float = 1e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95


@dataclass
class SimulationConfig:
    """Paths and knobs for orchestrating a multi-region simulation."""

    trajectory_root: Path = Path("dataset/traj_data")
    region_ids: Optional[Iterable[str]] = None
    log_dir: Path = Path("logs")
    random_seed: int = 42
    max_steps: Optional[int] = None
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    ev: EVConfig = field(default_factory=EVConfig)


@dataclass
class TrainingSchedule:
    """High-level training schedule for cloud-edge coordination."""

    cloud_update_every: int = 50
    edge_sync_every: int = 500
    evaluation_interval: int = 1000
    save_interval: int = 2000
    max_iterations: int = 10000
    checkpoint_dir: Path = Path("checkpoints")


@dataclass
class RegionSummary:
    """Statistics uploaded from edge to cloud after a rollout window."""

    region_id: str
    success_rate: float
    average_wait: float
    arrival_rate: float
    available_mcs: int
    queue_length: int
    extra_metrics: Dict[str, float] = field(default_factory=dict)
