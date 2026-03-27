from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ._api import get_model_io_shape, read_raw_chunk_file
from ._replay import ReplayBuffer, ReplayBufferConfig
from ._trainer import Trainer
from ._types import (
    LearnerRunnerConfig,
    ModelSpec,
    TrainerConfig,
    TrainIterationReport,
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
        self._replay_rng = np.random.default_rng(replay_buffer_config.seed)
        self._wandb_session: WandbSession | None = None

        if config.resume:
            latest_path = Path(config.checkpoint_dir) / "latest.pt"
            if not latest_path.exists():
                raise FileNotFoundError(
                    f"resume requested but checkpoint does not exist: {latest_path}"
                )
            self.trainer.load_checkpoint(latest_path)
            if self.trainer.replay_sampler_rng_state is not None:
                self._replay_rng.bit_generator.state = (
                    self.trainer.replay_sampler_rng_state
                )

        if config.wandb is not None:
            self._wandb_session = WandbSession(
                config.wandb,
                model_spec=model_spec,
                trainer_config=trainer_config,
                replay_buffer_config=replay_buffer_config,
                resume=config.resume,
            )

    def run_iteration(
        self,
        historical_chunk_paths: Sequence[str | os.PathLike[str]],
        new_chunk_paths: Sequence[str | os.PathLike[str]],
        iteration: int,
    ) -> TrainIterationReport:
        new_sample_count_hint = sum(
            _count_chunk_samples(chunk_path) for chunk_path in new_chunk_paths
        )
        history_sample_budget = max(
            0, self.replay_buffer_config.sample_capacity - new_sample_count_hint
        )
        replay_buffer = ReplayBuffer(
            self.replay_buffer_config,
            rng=self._replay_rng,
        )
        replay_history_chunk_paths = _select_replay_history_tail(
            historical_chunk_paths, history_sample_budget
        )
        if replay_history_chunk_paths:
            replay_buffer.ingest_chunk_files(replay_history_chunk_paths)
        new_sample_count = replay_buffer.ingest_chunk_files(new_chunk_paths)
        train_report = self.trainer.train_iteration(
            replay_buffer, new_sample_count, iteration
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


def _select_replay_history_tail(
    chunk_paths: Sequence[str | os.PathLike[str]], sample_budget: int
) -> list[str | os.PathLike[str]]:
    if sample_budget <= 0:
        return []

    selected_chunk_paths: list[str | os.PathLike[str]] = []
    selected_sample_count = 0
    for chunk_path in reversed(chunk_paths):
        selected_chunk_paths.append(chunk_path)
        selected_sample_count += _count_chunk_samples(chunk_path)
        if selected_sample_count >= sample_budget:
            break

    selected_chunk_paths.reverse()
    return selected_chunk_paths


def _count_chunk_samples(chunk_path: str | os.PathLike[str]) -> int:
    path_str = os.fspath(chunk_path)
    try:
        chunk = read_raw_chunk_file(path_str)
    except RuntimeError as exc:
        raise RuntimeError(f"{path_str}: {exc}") from exc
    return sum(len(game.plies) for game in chunk.games)
