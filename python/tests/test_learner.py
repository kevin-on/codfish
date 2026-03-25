from __future__ import annotations

import hashlib
import os
import pathlib
import subprocess
import tempfile
import unittest

import numpy as np

from codfish.learner import (
    EncodedGameSamples,
    GameResult,
    RawChunkFile,
    RawGame,
    RawPly,
    RawPolicyEntry,
    encode_raw_game,
    read_raw_chunk_file,
)


CHUNK_VERSION = 1
INPUT_SHA256 = "b94b6afd88212cd22cd797703ce8388f062dde5491a197f1ecfa92fb8d46fc24"
NONZERO_POLICY_INDICES = [317, 322]
NONZERO_POLICY_VALUES = [0.25, 0.75]


def _canonical_raw_game() -> RawGame:
    return RawGame(
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


def _canonical_raw_chunk() -> RawChunkFile:
    return RawChunkFile(version=CHUNK_VERSION, games=[_canonical_raw_game()])


def _chunk_writer_executable() -> str:
    try:
        return os.environ["CODFISH_LEARNER_GOLDEN_CHUNK_WRITER"]
    except KeyError as exc:
        raise RuntimeError(
            "CODFISH_LEARNER_GOLDEN_CHUNK_WRITER is not set for learner_python_test"
        ) from exc


def _write_canonical_chunk(path: pathlib.Path) -> None:
    subprocess.run([_chunk_writer_executable(), os.fspath(path)], check=True)


def _input_sha256(samples: EncodedGameSamples) -> str:
    return hashlib.sha256(samples.inputs.tobytes()).hexdigest()


class LearnerBindingsTest(unittest.TestCase):
    def test_read_raw_chunk_file_matches_cxx_golden_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = pathlib.Path(tmp_dir) / "golden.bin"
            _write_canonical_chunk(chunk_path)
            chunk = read_raw_chunk_file(chunk_path)

        self.assertEqual(chunk, _canonical_raw_chunk())

    def test_encode_raw_game_matches_golden_expectations(self) -> None:
        samples = encode_raw_game(_canonical_raw_game())

        self.assertEqual(samples.sample_count, 1)

        self.assertEqual(samples.inputs.dtype, np.uint8)
        self.assertEqual(samples.inputs.shape, (1, 118, 8, 8))
        self.assertTrue(samples.inputs.flags["C_CONTIGUOUS"])
        self.assertTrue(samples.inputs.flags["OWNDATA"])
        self.assertEqual(_input_sha256(samples), INPUT_SHA256)

        self.assertEqual(samples.policy_targets.dtype, np.float32)
        self.assertEqual(samples.policy_targets.shape, (1, samples.policy_size))
        self.assertTrue(samples.policy_targets.flags["C_CONTIGUOUS"])
        self.assertTrue(samples.policy_targets.flags["OWNDATA"])
        nonzero_idx = np.flatnonzero(samples.policy_targets[0]).tolist()
        self.assertEqual(nonzero_idx, NONZERO_POLICY_INDICES)
        np.testing.assert_allclose(
            samples.policy_targets[0, nonzero_idx],
            np.array(NONZERO_POLICY_VALUES, dtype=np.float32),
        )

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
