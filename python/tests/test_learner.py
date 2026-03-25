from __future__ import annotations

import pathlib
import struct
import tempfile
import unittest

import numpy as np

from codfish.learner import (
    GameResult,
    RawGame,
    RawPly,
    RawPolicyEntry,
    encode_raw_game,
    read_raw_chunk_file,
)


CHUNK_MAGIC = b"CFRG"
CHUNK_VERSION = 1
STORED_MOVE_UCI_BYTES = 5


def _encode_move_uci(move_uci: str) -> bytes:
    if len(move_uci) < 4 or len(move_uci) > STORED_MOVE_UCI_BYTES:
        raise ValueError("bad move uci")
    return move_uci.encode("ascii").ljust(STORED_MOVE_UCI_BYTES, b"\0")


def _serialize_raw_game(raw_game: RawGame) -> bytes:
    payload = bytearray()
    payload.extend(struct.pack("<B", 1 if raw_game.initial_fen is not None else 0))
    payload.extend(struct.pack("<B", raw_game.game_result.value))
    initial_fen_bytes = (
        raw_game.initial_fen.encode("utf-8")
        if raw_game.initial_fen is not None
        else b""
    )
    payload.extend(struct.pack("<I", len(initial_fen_bytes)))
    payload.extend(initial_fen_bytes)
    payload.extend(struct.pack("<I", len(raw_game.plies)))
    for ply in raw_game.plies:
        payload.extend(_encode_move_uci(ply.selected_move_uci))
        payload.extend(struct.pack("<I", len(ply.policy)))
        for entry in ply.policy:
            payload.extend(_encode_move_uci(entry.move_uci))
            payload.extend(struct.pack("<f", entry.prob))
    return struct.pack("<I", len(payload)) + payload


def _write_chunk_file(path: pathlib.Path, raw_games: list[RawGame]) -> None:
    data = bytearray(CHUNK_MAGIC)
    data.extend(struct.pack("<I", CHUNK_VERSION))
    for raw_game in raw_games:
        data.extend(_serialize_raw_game(raw_game))
    path.write_bytes(bytes(data))


class LearnerBindingsTest(unittest.TestCase):
    def test_read_raw_chunk_file_returns_dataclasses(self) -> None:
        raw_game = RawGame(
            initial_fen="4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
            game_result=GameResult.WHITE_WON,
            plies=[
                RawPly(
                    selected_move_uci="e2e4",
                    policy=[
                        RawPolicyEntry(move_uci="e2e4", prob=0.75),
                        RawPolicyEntry(move_uci="e2e3", prob=0.25),
                    ],
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = pathlib.Path(tmp_dir) / "games-000001.bin"
            _write_chunk_file(chunk_path, [raw_game])
            chunk = read_raw_chunk_file(chunk_path)

        self.assertEqual(chunk.version, CHUNK_VERSION)
        self.assertEqual(len(chunk.games), 1)
        self.assertEqual(chunk.games[0], raw_game)

    def test_encode_raw_game_returns_numpy_arrays(self) -> None:
        raw_game = RawGame(
            initial_fen="4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
            game_result=GameResult.WHITE_WON,
            plies=[
                RawPly(
                    selected_move_uci="e2e4",
                    policy=[
                        RawPolicyEntry(move_uci="e2e4", prob=0.75),
                        RawPolicyEntry(move_uci="e2e3", prob=0.25),
                    ],
                )
            ],
        )

        samples = encode_raw_game(raw_game)

        self.assertEqual(samples.sample_count, 1)
        self.assertEqual(samples.inputs.dtype, np.uint8)
        self.assertEqual(samples.inputs.shape, (1, 118, 8, 8))
        self.assertTrue(samples.inputs.flags["C_CONTIGUOUS"])
        self.assertTrue(samples.inputs.flags["OWNDATA"])

        self.assertEqual(samples.policy_targets.dtype, np.float32)
        self.assertEqual(samples.policy_targets.shape, (1, samples.policy_size))
        self.assertTrue(samples.policy_targets.flags["C_CONTIGUOUS"])
        self.assertTrue(samples.policy_targets.flags["OWNDATA"])

        self.assertEqual(samples.wdl_targets.dtype, np.float32)
        self.assertEqual(samples.wdl_targets.shape, (1, 3))
        self.assertTrue(samples.wdl_targets.flags["C_CONTIGUOUS"])
        self.assertTrue(samples.wdl_targets.flags["OWNDATA"])
        np.testing.assert_array_equal(
            samples.wdl_targets,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        )

    def test_encode_raw_game_surfaces_cpp_errors(self) -> None:
        raw_game = RawGame(
            initial_fen=None,
            game_result=GameResult.UNDECIDED,
            plies=[],
        )

        with self.assertRaises(RuntimeError):
            encode_raw_game(raw_game)

    def test_imported_game_result_enum_round_trips(self) -> None:
        self.assertEqual(GameResult.WHITE_WON.name, "WHITE_WON")
        self.assertEqual(GameResult.DRAW.name, "DRAW")


if __name__ == "__main__":
    unittest.main()
