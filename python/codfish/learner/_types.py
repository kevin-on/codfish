from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path

import torch


@dataclass(slots=True, frozen=True)
class ModelIOShape:
    input_channels: int
    policy_size: int

    def __post_init__(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.policy_size <= 0:
            raise ValueError("policy_size must be positive")


@dataclass(slots=True)
class TrainerConfig:
    learning_rate: float
    optimizer_momentum: float
    optimizer_weight_decay: float
    value_loss_weight: float

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.optimizer_momentum < 0:
            raise ValueError("optimizer_momentum must be non-negative")
        if self.optimizer_weight_decay < 0:
            raise ValueError("optimizer_weight_decay must be non-negative")
        if self.value_loss_weight < 0:
            raise ValueError("value_loss_weight must be non-negative")


@dataclass(slots=True)
class TrainStepMetrics:
    total_loss: float
    policy_loss: float
    wdl_loss: float


@dataclass(slots=True)
class TrainIterationReport:
    iteration: int
    starting_global_step: int
    ending_global_step: int
    num_updates: int
    new_sample_count: int
    replay_sample_count: int
    mean_total_loss: float
    mean_policy_loss: float
    mean_wdl_loss: float
    latest_checkpoint_path: Path
    snapshot_path: Path


@dataclass(slots=True)
class ModelSpec:
    name: str
    config: dict[str, object]
    factory: Callable[[], torch.nn.Module]


@dataclass(slots=True)
class WandbConfig:
    project: str
    entity: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.project:
            raise ValueError("project must be non-empty")


@dataclass(slots=True)
class LearnerRunnerConfig:
    device: str | torch.device
    checkpoint_dir: str | os.PathLike[str]
    resume: bool
    wandb: WandbConfig | None = None
