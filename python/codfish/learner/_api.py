from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import numpy as np

from . import _native
from ._types import ModelIOShape

GameResult = _native.GameResult


@dataclass(slots=True)
class RawPolicyEntry:
    move_uci: str
    prob: float


@dataclass(slots=True)
class RawPly:
    selected_move_uci: str
    policy: list[RawPolicyEntry]


@dataclass(slots=True)
class RawGame:
    initial_fen: Optional[str]
    game_result: GameResult
    plies: list[RawPly]


@dataclass(slots=True)
class RawChunkFile:
    version: int
    games: list[RawGame]


@dataclass(slots=True)
class EncodedGameSamples:
    sample_count: int
    input_channels: int
    policy_size: int
    inputs: np.ndarray
    policy_targets: np.ndarray
    wdl_targets: np.ndarray


def _from_native_policy_entry(entry: _native.RawPolicyEntry) -> RawPolicyEntry:
    return RawPolicyEntry(move_uci=entry.move_uci, prob=entry.prob)


def _to_native_policy_entry(entry: RawPolicyEntry) -> _native.RawPolicyEntry:
    native = _native.RawPolicyEntry()
    native.move_uci = entry.move_uci
    native.prob = entry.prob
    return native


def _from_native_ply(ply: _native.RawPly) -> RawPly:
    return RawPly(
        selected_move_uci=ply.selected_move_uci,
        policy=[_from_native_policy_entry(entry) for entry in ply.policy],
    )


def _to_native_ply(ply: RawPly) -> _native.RawPly:
    native = _native.RawPly()
    native.selected_move_uci = ply.selected_move_uci
    native.policy = [_to_native_policy_entry(entry) for entry in ply.policy]
    return native


def _from_native_raw_game(raw_game: _native.RawGame) -> RawGame:
    return RawGame(
        initial_fen=raw_game.initial_fen,
        game_result=raw_game.game_result,
        plies=[_from_native_ply(ply) for ply in raw_game.plies],
    )


def _to_native_raw_game(raw_game: RawGame) -> _native.RawGame:
    native = _native.RawGame()
    if raw_game.initial_fen is not None:
        native.initial_fen = raw_game.initial_fen
    native.game_result = raw_game.game_result
    native.plies = [_to_native_ply(ply) for ply in raw_game.plies]
    return native


def _from_native_raw_chunk(chunk: _native.RawChunkFile) -> RawChunkFile:
    return RawChunkFile(
        version=chunk.version,
        games=[_from_native_raw_game(game) for game in chunk.games],
    )


def _from_native_encoded_samples(
    samples: _native.EncodedGameSamples,
) -> EncodedGameSamples:
    return EncodedGameSamples(
        sample_count=samples.sample_count,
        input_channels=samples.input_channels,
        policy_size=samples.policy_size,
        inputs=samples.inputs,
        policy_targets=samples.policy_targets,
        wdl_targets=samples.wdl_targets,
    )


def _from_native_model_io_shape(shape: _native.ModelIOShape) -> ModelIOShape:
    return ModelIOShape(
        input_channels=shape.input_channels,
        policy_size=shape.policy_size,
    )


def read_raw_chunk_file(path: str | os.PathLike[str]) -> RawChunkFile:
    return _from_native_raw_chunk(_native.read_raw_chunk_file(os.fspath(path)))


def encode_raw_game(raw_game: RawGame) -> EncodedGameSamples:
    if not isinstance(raw_game, RawGame):
        raise TypeError("encode_raw_game expects codfish.learner.RawGame")
    return _from_native_encoded_samples(
        _native.encode_raw_game(_to_native_raw_game(raw_game))
    )


def get_model_io_shape() -> ModelIOShape:
    return _from_native_model_io_shape(_native.get_model_io_shape())
