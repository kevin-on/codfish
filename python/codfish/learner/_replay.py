from __future__ import annotations

from dataclasses import dataclass
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
    ) -> int:
        pending_inputs: list[np.ndarray] = []
        pending_policy_targets: list[np.ndarray] = []
        pending_wdl_targets: list[np.ndarray] = []
        new_sample_count = 0

        for chunk_path in chunk_paths:
            path_str = os.fspath(chunk_path)
            try:
                chunk = read_raw_chunk_file(path_str)
                for raw_game in chunk.games:
                    encoded = encode_raw_game(raw_game)
                    if encoded.sample_count == 0:
                        continue
                    pending_inputs.append(encoded.inputs)
                    pending_policy_targets.append(encoded.policy_targets)
                    pending_wdl_targets.append(encoded.wdl_targets)
                    new_sample_count += encoded.sample_count
            except RuntimeError as exc:
                raise RuntimeError(f"{path_str}: {exc}") from exc

        if not pending_inputs:
            return 0

        inputs = np.concatenate((self._inputs, np.concatenate(pending_inputs, axis=0)), axis=0)
        policy_targets = np.concatenate(
            (self._policy_targets, np.concatenate(pending_policy_targets, axis=0)),
            axis=0,
        )
        wdl_targets = np.concatenate(
            (self._wdl_targets, np.concatenate(pending_wdl_targets, axis=0)),
            axis=0,
        )
        overflow = max(0, inputs.shape[0] - self._config.sample_capacity)
        if overflow:
            inputs = inputs[overflow:]
            policy_targets = policy_targets[overflow:]
            wdl_targets = wdl_targets[overflow:]

        self._inputs = inputs
        self._policy_targets = policy_targets
        self._wdl_targets = wdl_targets
        return new_sample_count

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
