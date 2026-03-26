from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from typing import Sequence

import numpy as np

from ._api import (
    EncodedGameSamples,
    GameResult,
    RawGame,
    encode_raw_game,
    read_raw_chunk_file,
)


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ReplayBufferConfig:
    sample_capacity: int
    batch_size: int
    replay_ratio: float
    seed: int = 0

    def __post_init__(self) -> None:
        if self.sample_capacity <= 0:
            raise ValueError("sample_capacity must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.replay_ratio < 0:
            raise ValueError("replay_ratio must be non-negative")


@dataclass(slots=True)
class ReplayBufferIngestReport:
    requested_chunk_count: int
    loaded_chunk_count: int
    skipped_chunk_count: int
    skipped_chunk_errors: list[str]
    new_game_count: int
    new_sample_count: int
    trimmed_sample_count: int
    total_sample_count: int


class ReplayBuffer:
    def __init__(self, config: ReplayBufferConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)

        empty = encode_raw_game(
            RawGame(initial_fen=None, game_result=GameResult.DRAW, plies=[])
        )
        self._inputs = empty.inputs
        self._policy_targets = empty.policy_targets
        self._wdl_targets = empty.wdl_targets
        self._input_channels = empty.input_channels
        self._policy_size = empty.policy_size

    @property
    def sample_count(self) -> int:
        return int(self._inputs.shape[0])

    @property
    def input_channels(self) -> int:
        return self._input_channels

    @property
    def policy_size(self) -> int:
        return self._policy_size

    def ingest_chunk_files(
        self, chunk_paths: Sequence[str | os.PathLike[str]]
    ) -> ReplayBufferIngestReport:
        report = ReplayBufferIngestReport(
            requested_chunk_count=len(chunk_paths),
            loaded_chunk_count=0,
            skipped_chunk_count=0,
            skipped_chunk_errors=[],
            new_game_count=0,
            new_sample_count=0,
            trimmed_sample_count=0,
            total_sample_count=self.sample_count,
        )

        for chunk_path in chunk_paths:
            path_str = os.fspath(chunk_path)
            try:
                chunk = read_raw_chunk_file(path_str)
                chunk_inputs: list[np.ndarray] = []
                chunk_policy_targets: list[np.ndarray] = []
                chunk_wdl_targets: list[np.ndarray] = []
                chunk_game_count = 0
                chunk_sample_count = 0

                for raw_game in chunk.games:
                    chunk_game_count += 1
                    encoded = encode_raw_game(raw_game)
                    if encoded.sample_count == 0:
                        continue
                    chunk_inputs.append(encoded.inputs)
                    chunk_policy_targets.append(encoded.policy_targets)
                    chunk_wdl_targets.append(encoded.wdl_targets)
                    chunk_sample_count += encoded.sample_count

            except RuntimeError as exc:
                message = f"{path_str}: {exc}"
                LOGGER.warning("Skipping chunk file %s", message)
                report.skipped_chunk_count += 1
                report.skipped_chunk_errors.append(message)
                continue

            report.loaded_chunk_count += 1
            report.new_game_count += chunk_game_count
            report.new_sample_count += chunk_sample_count
            if chunk_inputs:
                self._inputs = np.concatenate(
                    (self._inputs, np.concatenate(chunk_inputs, axis=0)), axis=0
                )
                self._policy_targets = np.concatenate(
                    (
                        self._policy_targets,
                        np.concatenate(chunk_policy_targets, axis=0),
                    ),
                    axis=0,
                )
                self._wdl_targets = np.concatenate(
                    (self._wdl_targets, np.concatenate(chunk_wdl_targets, axis=0)),
                    axis=0,
                )

                overflow = max(0, self.sample_count - self._config.sample_capacity)
                if overflow:
                    self._inputs = self._inputs[overflow:]
                    self._policy_targets = self._policy_targets[overflow:]
                    self._wdl_targets = self._wdl_targets[overflow:]
                    report.trimmed_sample_count += overflow

        report.total_sample_count = self.sample_count
        return report

    def sample_minibatch(self) -> EncodedGameSamples:
        if self.sample_count == 0:
            raise RuntimeError("replay buffer is empty")

        indices = self._rng.integers(
            low=0, high=self.sample_count, size=self._config.batch_size
        )
        return EncodedGameSamples(
            sample_count=self._config.batch_size,
            input_channels=self._input_channels,
            policy_size=self._policy_size,
            inputs=self._inputs[indices],
            policy_targets=self._policy_targets[indices],
            wdl_targets=self._wdl_targets[indices],
        )

    def compute_update_steps(self, new_sample_count: int) -> int:
        if new_sample_count < 0:
            raise ValueError("new_sample_count must be non-negative")
        if new_sample_count == 0:
            return 0
        return math.ceil(
            self._config.replay_ratio
            * new_sample_count
            / self._config.batch_size
        )
