from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ._api import get_model_io_shape
from ._replay import ReplayBuffer, ReplayBufferConfig
from ._trainer import Trainer
from ._types import (
    LearnerRunnerConfig,
    ModelSpec,
    TrainIterationReport,
    TrainerConfig,
)
from ._wandb import WandbSession


class LearnerRunner:
    def __init__(
        self,
        model_spec: ModelSpec,
        trainer_config: TrainerConfig,
        replay_buffer_config: ReplayBufferConfig,
        config: LearnerRunnerConfig,
    ) -> None:
        self.model_spec = model_spec
        self.trainer_config = trainer_config
        self.replay_buffer_config = replay_buffer_config
        self.config = config

        _validate_model_spec_shape(model_spec)
        self.model = model_spec.factory()
        self.trainer = Trainer(
            self.model,
            trainer_config,
            device=config.device,
            checkpoint_dir=config.checkpoint_dir,
        )
        self.trainer.set_model_metadata(
            model_name=model_spec.name,
            model_config=model_spec.config,
        )
        self.replay_buffer = ReplayBuffer(replay_buffer_config)
        self._wandb_session: WandbSession | None = None

        if config.resume:
            latest_path = Path(config.checkpoint_dir) / "latest.pt"
            if not latest_path.exists():
                raise FileNotFoundError(
                    f"resume requested but checkpoint does not exist: {latest_path}"
                )
            self.trainer.load_checkpoint(latest_path)

        if config.wandb is not None:
            self._wandb_session = WandbSession(
                config.wandb,
                model_spec=model_spec,
                trainer_config=trainer_config,
                replay_buffer_config=replay_buffer_config,
                resume=config.resume,
            )

    def run_iteration(
        self, chunk_paths: list[str | os.PathLike[str]], iteration: int
    ) -> TrainIterationReport:
        new_sample_count = self.replay_buffer.ingest_chunk_files(chunk_paths)
        train_report = self.trainer.train_iteration(
            self.replay_buffer, new_sample_count, iteration
        )
        if self._wandb_session is not None:
            self._wandb_session.log_iteration(train_report)
        return train_report

    def close(self) -> None:
        if self._wandb_session is not None:
            self._wandb_session.close()

    def __enter__(self) -> LearnerRunner:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        self.close()


def _validate_model_spec_shape(model_spec: ModelSpec) -> None:
    native_shape = get_model_io_shape()
    input_channels = model_spec.config.get("input_channels")
    policy_size = model_spec.config.get("policy_size")
    if not isinstance(input_channels, int):
        raise ValueError("model_spec.config must contain integer input_channels")
    if not isinstance(policy_size, int):
        raise ValueError("model_spec.config must contain integer policy_size")
    if input_channels != native_shape.input_channels:
        raise ValueError(
            "model_spec.config input_channels does not match native learner shape"
        )
    if policy_size != native_shape.policy_size:
        raise ValueError(
            "model_spec.config policy_size does not match native learner shape"
        )
