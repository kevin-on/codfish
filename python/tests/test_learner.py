from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from codfish.learner import (
    EncodedGameSamples,
    GameResult,
    ReplayBuffer,
    ReplayBufferConfig,
    RawChunkFile,
    RawGame,
    RawPly,
    RawPolicyEntry,
    TrainIterationReport,
    TrainStepMetrics,
    Trainer,
    TrainerConfig,
    encode_raw_game,
    read_raw_chunk_file,
)
from codfish.learner._checkpoint import load_training_checkpoint


CHUNK_MAGIC = b"CFRG"
CHUNK_VERSION = 1
INPUT_SHA256 = "b94b6afd88212cd22cd797703ce8388f062dde5491a197f1ecfa92fb8d46fc24"
NONZERO_POLICY_INDICES = [317, 322]
NONZERO_POLICY_VALUES = [0.25, 0.75]
STORED_MOVE_UCI_BYTES = 5


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


def _alternate_raw_game() -> RawGame:
    return RawGame(
        initial_fen="4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        game_result=GameResult.DRAW,
        plies=[
            RawPly(
                selected_move_uci="e2e3",
                policy=[RawPolicyEntry(move_uci="e2e3", prob=1.0)],
            )
        ],
    )


def _chunk_writer_executable() -> str:
    try:
        return os.environ["CODFISH_LEARNER_GOLDEN_CHUNK_WRITER"]
    except KeyError as exc:
        raise RuntimeError(
            "CODFISH_LEARNER_GOLDEN_CHUNK_WRITER is not set for learner_python_test"
        ) from exc


def _write_canonical_chunk(path: pathlib.Path) -> None:
    subprocess.run([_chunk_writer_executable(), os.fspath(path)], check=True)


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


def _input_sha256(samples: EncodedGameSamples) -> str:
    return hashlib.sha256(samples.inputs.tobytes()).hexdigest()


class TrainerTest(unittest.TestCase):
    class _ToyModel(torch.nn.Module):
        def __init__(self, policy_size: int) -> None:
            super().__init__()
            self.last_input_dtype = None
            self.last_input_shape = None
            self.policy_head = torch.nn.Linear(118 * 8 * 8, policy_size)
            self.wdl_head = torch.nn.Linear(118 * 8 * 8, 3)

        def forward(
            self, inputs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.last_input_dtype = inputs.dtype
            self.last_input_shape = tuple(inputs.shape)
            flat = inputs.reshape(inputs.shape[0], -1)
            return self.policy_head(flat), self.wdl_head(flat)

    class _BadShapeModel(torch.nn.Module):
        def __init__(self, policy_size: int) -> None:
            super().__init__()
            self.policy_size = policy_size
            self._dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(
            self, inputs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            batch = inputs.shape[0]
            return (
                torch.zeros((batch, self.policy_size - 1), dtype=torch.float32),
                torch.zeros((batch, 3), dtype=torch.float32),
            )

    def _build_buffer(self, chunk_paths: list[pathlib.Path]) -> ReplayBuffer:
        buffer = ReplayBuffer(
            ReplayBufferConfig(
                sample_capacity=8, batch_size=2, replay_ratio=1.5, seed=7
            )
        )
        buffer.ingest_chunk_files(chunk_paths)
        return buffer

    def _build_trainer(
        self, checkpoint_dir: pathlib.Path, policy_size: int
    ) -> tuple[Trainer, torch.nn.Module]:
        model = self._ToyModel(policy_size)
        trainer = Trainer(
            model,
            TrainerConfig(
                learning_rate=0.01,
                optimizer_momentum=0.9,
                optimizer_weight_decay=1e-4,
                value_loss_weight=1.25,
            ),
            device="cpu",
            checkpoint_dir=checkpoint_dir,
        )
        return trainer, model

    def test_train_step_updates_model_and_casts_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = pathlib.Path(tmp_dir) / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            buffer = self._build_buffer([chunk_path])
            trainer, model = self._build_trainer(
                pathlib.Path(tmp_dir), policy_size=buffer.policy_size
            )

            before = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
            }
            metrics = trainer.train_step(buffer.sample_minibatch())

        self.assertIsInstance(metrics, TrainStepMetrics)
        self.assertTrue(np.isfinite(metrics.total_loss))
        self.assertTrue(np.isfinite(metrics.policy_loss))
        self.assertTrue(np.isfinite(metrics.wdl_loss))
        self.assertEqual(trainer.global_learner_step, 1)
        self.assertEqual(model.last_input_dtype, torch.float32)
        self.assertEqual(model.last_input_shape, (2, 118, 8, 8))
        self.assertTrue(
            any(
                not torch.equal(before[name], param.detach())
                for name, param in model.named_parameters()
            )
        )

    def test_train_step_raises_on_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = pathlib.Path(tmp_dir) / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            buffer = self._build_buffer([chunk_path])
            trainer = Trainer(
                self._BadShapeModel(buffer.policy_size),
                TrainerConfig(
                    learning_rate=0.01,
                    optimizer_momentum=0.9,
                    optimizer_weight_decay=1e-4,
                    value_loss_weight=1.25,
                ),
                device="cpu",
                checkpoint_dir=tmp_dir,
            )

            with self.assertRaisesRegex(ValueError, "policy logits shape"):
                trainer.train_step(buffer.sample_minibatch())

    def test_train_iteration_runs_expected_updates_and_writes_checkpoints(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = pathlib.Path(tmp_dir) / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            buffer = self._build_buffer([chunk_path])
            trainer, _ = self._build_trainer(
                pathlib.Path(tmp_dir), policy_size=buffer.policy_size
            )

            with mock.patch.object(
                buffer, "compute_update_steps", wraps=buffer.compute_update_steps
            ) as wrapped_steps:
                report = trainer.train_iteration(
                    replay_buffer=buffer, new_sample_count=257, iteration=7
                )

            latest_path = pathlib.Path(tmp_dir) / "latest.pt"
            snapshots_dir = pathlib.Path(tmp_dir) / "snapshots"

            self.assertIsInstance(report, TrainIterationReport)
            wrapped_steps.assert_called_once_with(257)
            self.assertEqual(report.iteration, 7)
            self.assertEqual(report.starting_global_step, 0)
            self.assertEqual(report.ending_global_step, report.num_updates)
            self.assertEqual(report.num_updates, 193)
            self.assertEqual(report.replay_sample_count, 1)
            self.assertTrue(np.isfinite(report.mean_total_loss))
            self.assertTrue(np.isfinite(report.mean_policy_loss))
            self.assertTrue(np.isfinite(report.mean_wdl_loss))
            self.assertEqual(report.latest_checkpoint_path, latest_path)
            self.assertTrue(latest_path.exists())
            self.assertEqual(report.snapshot_path.parent, snapshots_dir)
            self.assertTrue(report.snapshot_path.exists())

    def test_checkpoint_round_trip_restores_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = pathlib.Path(tmp_dir)
            chunk_path = checkpoint_dir / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            buffer = self._build_buffer([chunk_path])
            trainer, model = self._build_trainer(
                checkpoint_dir, policy_size=buffer.policy_size
            )
            trainer.train_iteration(buffer, new_sample_count=1, iteration=3)
            latest_path = checkpoint_dir / "latest.pt"
            saved_params = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
            }

            restored_model = self._ToyModel(buffer.policy_size)
            restored_trainer = Trainer(
                restored_model,
                TrainerConfig(
                    learning_rate=0.01,
                    optimizer_momentum=0.9,
                    optimizer_weight_decay=1e-4,
                    value_loss_weight=1.25,
                ),
                device="cpu",
                checkpoint_dir=checkpoint_dir,
            )
            restored_trainer.load_checkpoint(latest_path)

        self.assertEqual(
            restored_trainer.global_learner_step, trainer.global_learner_step
        )
        self.assertEqual(restored_trainer.last_checkpoint_iteration, 3)
        self.assertTrue(restored_trainer.optimizer.state_dict()["state"])
        for name, param in restored_model.named_parameters():
            self.assertTrue(torch.equal(param.detach(), saved_params[name]))

    def test_load_checkpoint_rejects_trainer_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = pathlib.Path(tmp_dir)
            chunk_path = checkpoint_dir / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            buffer = self._build_buffer([chunk_path])
            trainer, _ = self._build_trainer(
                checkpoint_dir, policy_size=buffer.policy_size
            )
            trainer.train_iteration(buffer, new_sample_count=1, iteration=3)
            latest_path = checkpoint_dir / "latest.pt"

            restored_model = self._ToyModel(buffer.policy_size)
            restored_trainer = Trainer(
                restored_model,
                TrainerConfig(
                    learning_rate=0.01,
                    optimizer_momentum=0.9,
                    optimizer_weight_decay=1e-4,
                    value_loss_weight=2.0,
                ),
                device="cpu",
                checkpoint_dir=checkpoint_dir,
            )

            with self.assertRaisesRegex(ValueError, "trainer_config"):
                restored_trainer.load_checkpoint(latest_path)

        self.assertEqual(restored_trainer.global_learner_step, 0)
        self.assertIsNone(restored_trainer.last_checkpoint_iteration)

    def test_load_checkpoint_rejects_snapshot_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = pathlib.Path(tmp_dir)
            chunk_path = checkpoint_dir / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            buffer = self._build_buffer([chunk_path])
            trainer, _ = self._build_trainer(
                checkpoint_dir, policy_size=buffer.policy_size
            )
            report = trainer.train_iteration(buffer, new_sample_count=1, iteration=2)

            restored_trainer, _ = self._build_trainer(
                checkpoint_dir, policy_size=buffer.policy_size
            )

            with self.assertRaisesRegex(ValueError, "training checkpoint"):
                restored_trainer.load_checkpoint(report.snapshot_path)

        self.assertEqual(restored_trainer.global_learner_step, 0)
        self.assertIsNone(restored_trainer.last_checkpoint_iteration)

    def test_load_training_checkpoint_rejects_unsupported_format_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "bad.pt"
            torch.save({"format_version": 999}, path)

            with self.assertRaisesRegex(ValueError, "unsupported checkpoint"):
                load_training_checkpoint(path, map_location=torch.device("cpu"))

    def test_snapshot_payload_excludes_optimizer_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = pathlib.Path(tmp_dir) / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            buffer = self._build_buffer([chunk_path])
            trainer, _ = self._build_trainer(
                pathlib.Path(tmp_dir), policy_size=buffer.policy_size
            )
            report = trainer.train_iteration(buffer, new_sample_count=1, iteration=2)
            payload = torch.load(report.snapshot_path, map_location="cpu")

            self.assertIn("model_state_dict", payload)
            self.assertIn("global_learner_step", payload)
            self.assertIn("iteration", payload)
            self.assertNotIn("optimizer_state_dict", payload)

    def test_failed_checkpoint_save_keeps_latest_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = pathlib.Path(tmp_dir)
            chunk_path = checkpoint_dir / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            buffer = self._build_buffer([chunk_path])
            trainer, _ = self._build_trainer(
                checkpoint_dir, policy_size=buffer.policy_size
            )
            trainer.train_iteration(buffer, new_sample_count=1, iteration=3)

            latest_path = checkpoint_dir / "latest.pt"
            previous_path = checkpoint_dir / "previous.pt"
            expected_latest = latest_path.read_bytes()

            with mock.patch(
                "codfish.learner._checkpoint.torch.save",
                side_effect=RuntimeError("disk full"),
            ):
                with self.assertRaisesRegex(RuntimeError, "disk full"):
                    trainer.save_checkpoint(iteration=4)
            self.assertEqual(latest_path.read_bytes(), expected_latest)
            self.assertFalse(previous_path.exists())

    def test_zero_update_iteration_still_writes_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = pathlib.Path(tmp_dir) / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            buffer = self._build_buffer([chunk_path])
            trainer, _ = self._build_trainer(
                pathlib.Path(tmp_dir), policy_size=buffer.policy_size
            )

            report = trainer.train_iteration(buffer, new_sample_count=0, iteration=9)

            self.assertEqual(report.num_updates, 0)
            self.assertEqual(report.starting_global_step, 0)
            self.assertEqual(report.ending_global_step, 0)
            self.assertEqual(report.mean_total_loss, 0.0)
            self.assertEqual(report.mean_policy_loss, 0.0)
            self.assertEqual(report.mean_wdl_loss, 0.0)
            self.assertTrue(report.latest_checkpoint_path.exists())
            self.assertTrue(report.snapshot_path.exists())


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


class ReplayBufferTest(unittest.TestCase):
    def test_ingest_chunk_files_reports_counts_and_matches_encoded_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = pathlib.Path(tmp_dir) / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])

            buffer = ReplayBuffer(
                ReplayBufferConfig(
                    sample_capacity=8, batch_size=2, replay_ratio=1.5, seed=7
                )
            )
            report = buffer.ingest_chunk_files([chunk_path])

        self.assertEqual(report.requested_chunk_count, 1)
        self.assertEqual(report.loaded_chunk_count, 1)
        self.assertEqual(report.skipped_chunk_count, 0)
        self.assertEqual(report.skipped_chunk_errors, [])
        self.assertEqual(report.new_game_count, 1)
        self.assertEqual(report.new_sample_count, 1)
        self.assertEqual(report.trimmed_sample_count, 0)
        self.assertEqual(report.total_sample_count, 1)
        self.assertEqual(buffer.sample_count, 1)

        expected = encode_raw_game(_canonical_raw_game())
        batch = buffer.sample_minibatch()
        np.testing.assert_array_equal(batch.inputs, np.repeat(expected.inputs, 2, axis=0))
        np.testing.assert_array_equal(
            batch.policy_targets, np.repeat(expected.policy_targets, 2, axis=0)
        )
        np.testing.assert_array_equal(
            batch.wdl_targets, np.repeat(expected.wdl_targets, 2, axis=0)
        )

    def test_ingest_chunk_files_trims_exactly_to_sample_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_chunk = pathlib.Path(tmp_dir) / "first.bin"
            second_chunk = pathlib.Path(tmp_dir) / "second.bin"
            _write_chunk_file(first_chunk, [_canonical_raw_game()])
            _write_chunk_file(second_chunk, [_alternate_raw_game()])

            buffer = ReplayBuffer(
                ReplayBufferConfig(sample_capacity=1, batch_size=1, replay_ratio=1.0)
            )
            first_report = buffer.ingest_chunk_files([first_chunk])
            second_report = buffer.ingest_chunk_files([second_chunk])

        self.assertEqual(first_report.total_sample_count, 1)
        self.assertEqual(second_report.trimmed_sample_count, 1)
        self.assertEqual(second_report.total_sample_count, 1)
        self.assertEqual(buffer.sample_count, 1)

        batch = buffer.sample_minibatch()
        expected = encode_raw_game(_alternate_raw_game())
        np.testing.assert_array_equal(batch.inputs, expected.inputs)
        np.testing.assert_array_equal(batch.policy_targets, expected.policy_targets)
        np.testing.assert_array_equal(batch.wdl_targets, expected.wdl_targets)

    def test_ingest_chunk_files_skips_malformed_chunk_and_reports_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_chunk = pathlib.Path(tmp_dir) / "bad.bin"
            bad_chunk.write_bytes(b"CFRG\x01\x00")

            buffer = ReplayBuffer(
                ReplayBufferConfig(sample_capacity=4, batch_size=1, replay_ratio=1.0)
            )
            with self.assertLogs("codfish.learner._replay", level="WARNING") as logs:
                report = buffer.ingest_chunk_files([bad_chunk])

        self.assertEqual(report.requested_chunk_count, 1)
        self.assertEqual(report.loaded_chunk_count, 0)
        self.assertEqual(report.skipped_chunk_count, 1)
        self.assertEqual(report.new_game_count, 0)
        self.assertEqual(report.new_sample_count, 0)
        self.assertEqual(report.total_sample_count, 0)
        self.assertEqual(len(report.skipped_chunk_errors), 1)
        self.assertIn(os.fspath(bad_chunk), report.skipped_chunk_errors[0])
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].levelno, logging.WARNING)

    def test_sample_minibatch_raises_when_buffer_is_empty(self) -> None:
        buffer = ReplayBuffer(
            ReplayBufferConfig(sample_capacity=4, batch_size=2, replay_ratio=1.0)
        )
        with self.assertRaises(RuntimeError):
            buffer.sample_minibatch()

    def test_compute_update_steps_uses_replay_ratio(self) -> None:
        buffer = ReplayBuffer(
            ReplayBufferConfig(sample_capacity=4, batch_size=128, replay_ratio=2.5)
        )
        self.assertEqual(buffer.compute_update_steps(0), 0)
        self.assertEqual(buffer.compute_update_steps(257), 6)


if __name__ == "__main__":
    unittest.main()
