from __future__ import annotations

import hashlib
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
    LearnerRunner,
    LearnerRunnerConfig,
    ModelSpec,
    ReplayBuffer,
    ReplayBufferConfig,
    RawChunkFile,
    RawGame,
    RawPly,
    RawPolicyEntry,
    SmallAlphaZeroResNet,
    SmallAlphaZeroResNetConfig,
    TrainIterationReport,
    TrainStepMetrics,
    Trainer,
    TrainerConfig,
    WandbConfig,
    encode_raw_game,
    make_small_alphazero_resnet_spec,
    read_raw_chunk_file,
)
from codfish.learner._api import get_model_io_shape
from codfish.learner._checkpoint import (
    load_snapshot_checkpoint,
    load_training_checkpoint,
)


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


def _model_shape() -> tuple[int, int]:
    shape = get_model_io_shape()
    return shape.input_channels, shape.policy_size


def _input_channels() -> int:
    return _model_shape()[0]


def _policy_size() -> int:
    return _model_shape()[1]


def _small_model_config() -> SmallAlphaZeroResNetConfig:
    return SmallAlphaZeroResNetConfig(
        input_channels=_input_channels(),
        policy_size=_policy_size(),
    )


class _FakeWandbRun:
    def __init__(self) -> None:
        self.define_metric_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.log_calls: list[dict[str, object]] = []
        self.finish_calls = 0
        self.log_artifact_calls = 0

    def define_metric(self, *args: object, **kwargs: object) -> None:
        self.define_metric_calls.append((args, dict(kwargs)))

    def log(self, payload: dict[str, object]) -> None:
        self.log_calls.append(dict(payload))

    def finish(self) -> None:
        self.finish_calls += 1

    def log_artifact(self, *args: object, **kwargs: object) -> None:
        self.log_artifact_calls += 1


class _FakeWandbModule:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, object]] = []
        self.run = _FakeWandbRun()
        self.artifact_calls = 0

    def init(self, **kwargs: object) -> _FakeWandbRun:
        self.init_calls.append(dict(kwargs))
        return self.run

    def Artifact(self, *args: object, **kwargs: object) -> None:
        self.artifact_calls += 1


class SmallAlphaZeroResNetTest(unittest.TestCase):
    def test_forward_returns_policy_and_wdl_logits(self) -> None:
        config = _small_model_config()
        model = SmallAlphaZeroResNet(config)

        policy_logits, wdl_logits = model(
            torch.zeros((2, config.input_channels, 8, 8), dtype=torch.float32)
        )

        self.assertEqual(tuple(policy_logits.shape), (2, config.policy_size))
        self.assertEqual(tuple(wdl_logits.shape), (2, 3))

    def test_forward_rejects_invalid_input_shape(self) -> None:
        config = _small_model_config()
        model = SmallAlphaZeroResNet(config)

        with self.assertRaisesRegex(ValueError, "expected \\[B,"):
            model(torch.zeros((2, config.input_channels + 1, 8, 8), dtype=torch.float32))

    def test_make_small_alphazero_resnet_spec_exposes_full_model_config(
        self,
    ) -> None:
        config = SmallAlphaZeroResNetConfig(
            input_channels=117,
            policy_size=1857,
            trunk_channels=16,
            num_blocks=2,
            policy_channels=3,
            value_channels=5,
            value_hidden=32,
        )

        spec = make_small_alphazero_resnet_spec(config)
        model = spec.factory()

        self.assertEqual(spec.name, "small_alphazero_resnet")
        self.assertEqual(
            spec.config,
            {
                "input_channels": 117,
                "policy_size": 1857,
                "trunk_channels": 16,
                "num_blocks": 2,
                "policy_channels": 3,
                "value_channels": 5,
                "value_hidden": 32,
            },
        )
        self.assertIsInstance(model, SmallAlphaZeroResNet)


class TrainerTest(unittest.TestCase):
    class _ToyModel(torch.nn.Module):
        def __init__(self, input_channels: int, policy_size: int) -> None:
            super().__init__()
            self.input_channels = input_channels
            self.policy_size = policy_size
            self.last_input_dtype = None
            self.last_input_shape = None
            flat_size = input_channels * 8 * 8
            self.policy_head = torch.nn.Linear(flat_size, policy_size)
            self.wdl_head = torch.nn.Linear(flat_size, 3)

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
        self, checkpoint_dir: pathlib.Path, input_channels: int, policy_size: int
    ) -> tuple[Trainer, torch.nn.Module]:
        model = self._ToyModel(input_channels, policy_size)
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
                pathlib.Path(tmp_dir), buffer.input_channels, buffer.policy_size
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
        self.assertEqual(model.last_input_shape, (2, buffer.input_channels, 8, 8))
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
                pathlib.Path(tmp_dir), buffer.input_channels, buffer.policy_size
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
                checkpoint_dir, buffer.input_channels, buffer.policy_size
            )
            trainer.train_iteration(buffer, new_sample_count=1, iteration=3)
            latest_path = checkpoint_dir / "latest.pt"
            saved_params = {
                name: param.detach().clone()
                for name, param in model.named_parameters()
            }

            restored_model = self._ToyModel(buffer.input_channels, buffer.policy_size)
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
        self.assertIsNotNone(restored_trainer.replay_sampler_rng_state)
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
                checkpoint_dir, buffer.input_channels, buffer.policy_size
            )
            trainer.train_iteration(buffer, new_sample_count=1, iteration=3)
            latest_path = checkpoint_dir / "latest.pt"

            restored_model = self._ToyModel(buffer.input_channels, buffer.policy_size)
            restored_trainer = Trainer(
                restored_model,
                TrainerConfig(
                    learning_rate=0.02,
                    optimizer_momentum=0.4,
                    optimizer_weight_decay=2e-4,
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
                checkpoint_dir, buffer.input_channels, buffer.policy_size
            )
            report = trainer.train_iteration(buffer, new_sample_count=1, iteration=2)

            restored_trainer, _ = self._build_trainer(
                checkpoint_dir, buffer.input_channels, buffer.policy_size
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
                pathlib.Path(tmp_dir), buffer.input_channels, buffer.policy_size
            )
            report = trainer.train_iteration(buffer, new_sample_count=1, iteration=2)
            payload = torch.load(report.snapshot_path, map_location="cpu")

            self.assertIn("model_state_dict", payload)
            self.assertIn("global_learner_step", payload)
            self.assertIn("iteration", payload)
            self.assertNotIn("optimizer_state_dict", payload)

    def test_load_training_checkpoint_accepts_legacy_flat_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "legacy_training.pt"
            torch.save(
                {
                    "format_version": 1,
                    "model_state_dict": {},
                    "optimizer_state_dict": {},
                    "global_learner_step": 12,
                    "iteration": 5,
                    "model_name": "legacy-model",
                    "model_config": {"kind": "legacy"},
                    "trainer_config": {
                        "learning_rate": 0.01,
                        "optimizer_momentum": 0.9,
                        "optimizer_weight_decay": 1e-4,
                        "value_loss_weight": 1.25,
                    },
                },
                path,
            )

            checkpoint = load_training_checkpoint(
                path,
                map_location=torch.device("cpu"),
            )

        self.assertEqual(checkpoint.model_name, "legacy-model")
        self.assertEqual(checkpoint.model_config, {"kind": "legacy"})
        self.assertIsNone(checkpoint.replay_sampler_rng_state)

    def test_load_snapshot_checkpoint_accepts_legacy_flat_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = pathlib.Path(tmp_dir) / "legacy_snapshot.pt"
            torch.save(
                {
                    "format_version": 1,
                    "model_state_dict": {},
                    "global_learner_step": 12,
                    "iteration": 5,
                    "model_name": "legacy-model",
                    "model_config": {"kind": "legacy"},
                },
                path,
            )

            checkpoint = load_snapshot_checkpoint(
                path,
                map_location=torch.device("cpu"),
            )

        self.assertEqual(checkpoint.model_name, "legacy-model")
        self.assertEqual(checkpoint.model_config, {"kind": "legacy"})

    def test_failed_checkpoint_save_keeps_latest_intact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = pathlib.Path(tmp_dir)
            chunk_path = checkpoint_dir / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            buffer = self._build_buffer([chunk_path])
            trainer, _ = self._build_trainer(
                checkpoint_dir, buffer.input_channels, buffer.policy_size
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
                pathlib.Path(tmp_dir), buffer.input_channels, buffer.policy_size
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


class LearnerRunnerTest(unittest.TestCase):
    class _ToyModel(torch.nn.Module):
        def __init__(self, input_channels: int, policy_size: int) -> None:
            super().__init__()
            flat_size = input_channels * 8 * 8
            self.policy_head = torch.nn.Linear(flat_size, policy_size)
            self.wdl_head = torch.nn.Linear(flat_size, 3)

        def forward(
            self, inputs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            flat = inputs.reshape(inputs.shape[0], -1)
            return self.policy_head(flat), self.wdl_head(flat)

    def _trainer_config(self) -> TrainerConfig:
        return TrainerConfig(
            learning_rate=0.01,
            optimizer_momentum=0.9,
            optimizer_weight_decay=1e-4,
            value_loss_weight=1.25,
        )

    def _replay_config(self) -> ReplayBufferConfig:
        return ReplayBufferConfig(
            sample_capacity=8,
            batch_size=2,
            replay_ratio=1.5,
            seed=7,
        )

    def _model_spec(self) -> ModelSpec:
        return ModelSpec(
            name="toy-model",
            config={
                "kind": "toy",
                "input_channels": _input_channels(),
                "policy_size": _policy_size(),
            },
            factory=lambda: self._ToyModel(_input_channels(), _policy_size()),
        )

    def _runner_config(
        self,
        checkpoint_dir: pathlib.Path,
        *,
        resume: bool = False,
        wandb: WandbConfig | None = None,
    ) -> LearnerRunnerConfig:
        return LearnerRunnerConfig(
            device="cpu",
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            wandb=wandb,
        )

    def test_runner_init_builds_model_and_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = LearnerRunner(
                self._model_spec(),
                self._trainer_config(),
                self._replay_config(),
                self._runner_config(pathlib.Path(tmp_dir)),
            )

        self.assertIsInstance(runner.model, self._ToyModel)
        self.assertIsInstance(runner.trainer, Trainer)
        self.assertEqual(runner.trainer.model_name, "toy-model")
        self.assertEqual(
            runner.trainer.model_config,
            {
                "kind": "toy",
                "input_channels": _input_channels(),
                "policy_size": _policy_size(),
            },
        )
        self.assertFalse(hasattr(runner, "replay_buffer"))

    def test_resume_with_existing_latest_restores_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = pathlib.Path(tmp_dir)
            chunk_path = checkpoint_dir / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            first_runner = LearnerRunner(
                self._model_spec(),
                self._trainer_config(),
                self._replay_config(),
                self._runner_config(checkpoint_dir),
            )
            first_report = first_runner.run_iteration([], [chunk_path], iteration=4)
            first_runner.close()

            resumed_runner = LearnerRunner(
                self._model_spec(),
                self._trainer_config(),
                self._replay_config(),
                self._runner_config(checkpoint_dir, resume=True),
            )

        self.assertEqual(
            resumed_runner.trainer.global_learner_step,
            first_report.ending_global_step,
        )
        self.assertEqual(resumed_runner.trainer.last_checkpoint_iteration, 4)
        self.assertEqual(resumed_runner.trainer.model_name, "toy-model")
        self.assertEqual(
            resumed_runner.trainer.model_config,
            {
                "kind": "toy",
                "input_channels": _input_channels(),
                "policy_size": _policy_size(),
            },
        )
        self.assertTrue(resumed_runner.trainer.optimizer.state_dict()["state"])
        self.assertEqual(
            resumed_runner._replay_rng.bit_generator.state,
            first_runner._replay_rng.bit_generator.state,
        )

    def test_runner_rejects_model_config_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch(
                "codfish.learner._runner.get_model_io_shape",
                return_value=mock.Mock(input_channels=7, policy_size=11),
            ):
                with self.assertRaisesRegex(ValueError, "input_channels"):
                    LearnerRunner(
                        self._model_spec(),
                        self._trainer_config(),
                        self._replay_config(),
                        self._runner_config(pathlib.Path(tmp_dir)),
                    )

    def test_resume_with_missing_latest_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(FileNotFoundError):
                LearnerRunner(
                    self._model_spec(),
                    self._trainer_config(),
                    self._replay_config(),
                    self._runner_config(pathlib.Path(tmp_dir), resume=True),
                )

    def test_run_iteration_returns_train_iteration_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = pathlib.Path(tmp_dir) / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            runner = LearnerRunner(
                self._model_spec(),
                self._trainer_config(),
                self._replay_config(),
                self._runner_config(pathlib.Path(tmp_dir)),
            )

            with mock.patch.object(
                runner.trainer,
                "train_iteration",
                wraps=runner.trainer.train_iteration,
            ) as wrapped_train_iteration:
                report = runner.run_iteration([], [chunk_path], iteration=6)

        self.assertIsInstance(report, TrainIterationReport)
        wrapped_train_iteration.assert_called_once()
        replay_buffer_arg, new_sample_count_arg, iteration_arg = (
            wrapped_train_iteration.call_args.args
        )
        self.assertIsInstance(replay_buffer_arg, ReplayBuffer)
        self.assertEqual(new_sample_count_arg, report.new_sample_count)
        self.assertEqual(iteration_arg, 6)
        self.assertEqual(report.new_sample_count, 1)
        self.assertEqual(replay_buffer_arg.sample_count, 1)

    def test_run_iteration_rebuilds_replay_from_scratch_each_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_chunk = pathlib.Path(tmp_dir) / "first.bin"
            second_chunk = pathlib.Path(tmp_dir) / "second.bin"
            _write_chunk_file(first_chunk, [_canonical_raw_game()])
            _write_chunk_file(second_chunk, [_alternate_raw_game()])
            runner = LearnerRunner(
                self._model_spec(),
                self._trainer_config(),
                self._replay_config(),
                self._runner_config(pathlib.Path(tmp_dir)),
            )
            observed_sample_counts: list[int] = []
            original_train_iteration = runner.trainer.train_iteration

            def wrapped_train_iteration(
                replay_buffer: ReplayBuffer,
                new_sample_count: int,
                iteration: int,
            ) -> TrainIterationReport:
                observed_sample_counts.append(replay_buffer.sample_count)
                return original_train_iteration(replay_buffer, new_sample_count, iteration)

            with mock.patch.object(
                runner.trainer,
                "train_iteration",
                side_effect=wrapped_train_iteration,
            ):
                runner.run_iteration([], [first_chunk], iteration=1)
                runner.run_iteration([], [second_chunk], iteration=2)

        self.assertEqual(observed_sample_counts, [1, 1])

    def test_run_iteration_preserves_replay_sampler_state_across_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_chunk = pathlib.Path(tmp_dir) / "first.bin"
            second_chunk = pathlib.Path(tmp_dir) / "second.bin"
            _write_chunk_file(
                first_chunk,
                [_canonical_raw_game(), _alternate_raw_game()],
            )
            _write_chunk_file(
                second_chunk,
                [_canonical_raw_game(), _alternate_raw_game()],
            )
            runner = LearnerRunner(
                self._model_spec(),
                self._trainer_config(),
                self._replay_config(),
                self._runner_config(pathlib.Path(tmp_dir)),
            )
            observed_rng_states: list[dict[str, object]] = []
            original_train_iteration = runner.trainer.train_iteration

            def wrapped_train_iteration(
                replay_buffer: ReplayBuffer,
                new_sample_count: int,
                iteration: int,
            ) -> TrainIterationReport:
                observed_rng_states.append(dict(replay_buffer._rng.bit_generator.state))
                return original_train_iteration(replay_buffer, new_sample_count, iteration)

            with mock.patch.object(
                runner.trainer,
                "train_iteration",
                side_effect=wrapped_train_iteration,
            ):
                runner.run_iteration([], [first_chunk], iteration=1)
                runner.run_iteration([], [second_chunk], iteration=2)

        self.assertEqual(len(observed_rng_states), 2)
        self.assertNotEqual(observed_rng_states[0], observed_rng_states[1])

    def test_run_iteration_selects_recent_history_tail_before_new_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_chunk = pathlib.Path(tmp_dir) / "first.bin"
            second_chunk = pathlib.Path(tmp_dir) / "second.bin"
            third_chunk = pathlib.Path(tmp_dir) / "third.bin"
            new_chunk = pathlib.Path(tmp_dir) / "new.bin"
            _write_chunk_file(first_chunk, [_canonical_raw_game()])
            _write_chunk_file(
                second_chunk,
                [_canonical_raw_game(), _alternate_raw_game()],
            )
            _write_chunk_file(third_chunk, [_alternate_raw_game()])
            _write_chunk_file(new_chunk, [_canonical_raw_game()])
            runner = LearnerRunner(
                self._model_spec(),
                self._trainer_config(),
                ReplayBufferConfig(
                    sample_capacity=3,
                    batch_size=2,
                    replay_ratio=1.5,
                    seed=7,
                ),
                self._runner_config(pathlib.Path(tmp_dir)),
            )
            ingest_calls: list[list[pathlib.Path]] = []
            original_ingest = ReplayBuffer.ingest_chunk_files

            def wrapped_ingest(
                replay_buffer: ReplayBuffer,
                chunk_paths: list[str | os.PathLike[str]],
            ) -> int:
                ingest_calls.append([pathlib.Path(path) for path in chunk_paths])
                return original_ingest(replay_buffer, chunk_paths)

            with mock.patch.object(
                ReplayBuffer,
                "ingest_chunk_files",
                new=wrapped_ingest,
            ):
                runner.run_iteration(
                    [first_chunk, second_chunk, third_chunk],
                    [new_chunk],
                    iteration=3,
                )

        self.assertEqual(ingest_calls, [[second_chunk, third_chunk], [new_chunk]])

    def test_run_iteration_is_atomic_when_any_chunk_is_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = pathlib.Path(tmp_dir)
            first_chunk = checkpoint_dir / "first.bin"
            second_chunk = checkpoint_dir / "second.bin"
            bad_chunk = checkpoint_dir / "bad.bin"
            _write_chunk_file(first_chunk, [_canonical_raw_game()])
            _write_chunk_file(second_chunk, [_alternate_raw_game()])
            bad_chunk.write_bytes(b"CFRG\x01\x00")
            runner = LearnerRunner(
                self._model_spec(),
                self._trainer_config(),
                self._replay_config(),
                self._runner_config(checkpoint_dir),
            )
            first_report = runner.run_iteration([], [first_chunk], iteration=1)

            with self.assertRaisesRegex(RuntimeError, os.fspath(bad_chunk)):
                runner.run_iteration([first_chunk], [second_chunk, bad_chunk], iteration=2)

        self.assertEqual(
            runner.trainer.global_learner_step, first_report.ending_global_step
        )

    def test_checkpoints_include_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = pathlib.Path(tmp_dir)
            chunk_path = checkpoint_dir / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            runner = LearnerRunner(
                self._model_spec(),
                self._trainer_config(),
                self._replay_config(),
                self._runner_config(checkpoint_dir),
            )
            report = runner.run_iteration([], [chunk_path], iteration=8)

            training_checkpoint = load_training_checkpoint(
                report.latest_checkpoint_path,
                map_location=torch.device("cpu"),
            )
            snapshot_checkpoint = load_snapshot_checkpoint(
                report.snapshot_path,
                map_location=torch.device("cpu"),
            )

        self.assertEqual(training_checkpoint.model_name, "toy-model")
        self.assertEqual(
            training_checkpoint.model_config,
            {
                "kind": "toy",
                "input_channels": _input_channels(),
                "policy_size": _policy_size(),
            },
        )
        self.assertIsNotNone(training_checkpoint.replay_sampler_rng_state)
        self.assertEqual(snapshot_checkpoint.model_name, "toy-model")
        self.assertEqual(
            snapshot_checkpoint.model_config,
            {
                "kind": "toy",
                "input_channels": _input_channels(),
                "policy_size": _policy_size(),
            },
        )

    def test_wandb_disabled_path_does_not_import_wandb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = pathlib.Path(tmp_dir) / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            with mock.patch(
                "codfish.learner._wandb.importlib.import_module",
                side_effect=AssertionError("wandb should not be imported"),
            ):
                runner = LearnerRunner(
                    self._model_spec(),
                    self._trainer_config(),
                    self._replay_config(),
                    self._runner_config(pathlib.Path(tmp_dir)),
                )
                runner.run_iteration([], [chunk_path], iteration=3)

    def test_wandb_enabled_logs_metrics_and_finishes_run(self) -> None:
        fake_wandb = _FakeWandbModule()
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = pathlib.Path(tmp_dir) / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])
            with mock.patch(
                "codfish.learner._wandb.importlib.import_module",
                return_value=fake_wandb,
            ):
                runner = LearnerRunner(
                    self._model_spec(),
                    self._trainer_config(),
                    self._replay_config(),
                    self._runner_config(
                        pathlib.Path(tmp_dir),
                        wandb=WandbConfig(
                            project="codfish-test",
                            entity="codfish",
                            name="runner",
                        ),
                    ),
                )
                report = runner.run_iteration([], [chunk_path], iteration=11)
                runner.close()

        self.assertEqual(len(fake_wandb.init_calls), 1)
        init_kwargs = fake_wandb.init_calls[0]
        self.assertEqual(init_kwargs["project"], "codfish-test")
        self.assertEqual(init_kwargs["entity"], "codfish")
        self.assertEqual(init_kwargs["name"], "runner")
        self.assertEqual(
            init_kwargs["config"],
            {
                "model": {
                    "name": "toy-model",
                    "config": {
                        "kind": "toy",
                        "input_channels": _input_channels(),
                        "policy_size": _policy_size(),
                    },
                },
                "trainer": {
                    "learning_rate": 0.01,
                    "optimizer_momentum": 0.9,
                    "optimizer_weight_decay": 1e-4,
                    "value_loss_weight": 1.25,
                },
                "replay": {
                    "sample_capacity": 8,
                    "batch_size": 2,
                    "replay_ratio": 1.5,
                    "seed": 7,
                },
                "runner": {"resume": False},
            },
        )
        self.assertEqual(
            fake_wandb.run.define_metric_calls,
            [
                (("global_step",), {}),
                (("iteration",), {}),
                (("train/*",), {"step_metric": "global_step"}),
            ],
        )
        self.assertEqual(len(fake_wandb.run.log_calls), 1)
        self.assertEqual(
            fake_wandb.run.log_calls[0],
            {
                "global_step": report.ending_global_step,
                "iteration": 11,
                "train/total_loss": report.mean_total_loss,
                "train/policy_loss": report.mean_policy_loss,
                "train/wdl_loss": report.mean_wdl_loss,
            },
        )
        self.assertEqual(fake_wandb.run.finish_calls, 1)
        self.assertEqual(fake_wandb.run.log_artifact_calls, 0)
        self.assertEqual(fake_wandb.artifact_calls, 0)

    def test_wandb_requested_without_dependency_raises_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch(
                "codfish.learner._wandb.importlib.import_module",
                side_effect=ModuleNotFoundError("No module named 'wandb'"),
            ):
                with self.assertRaisesRegex(RuntimeError, "wandb is not installed"):
                    LearnerRunner(
                        self._model_spec(),
                        self._trainer_config(),
                        self._replay_config(),
                        self._runner_config(
                            pathlib.Path(tmp_dir),
                            wandb=WandbConfig(project="codfish-test"),
                        ),
                    )

    def test_runner_context_manager_calls_finish(self) -> None:
        fake_wandb = _FakeWandbModule()
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch(
                "codfish.learner._wandb.importlib.import_module",
                return_value=fake_wandb,
            ):
                with LearnerRunner(
                    self._model_spec(),
                    self._trainer_config(),
                    self._replay_config(),
                    self._runner_config(
                        pathlib.Path(tmp_dir),
                        wandb=WandbConfig(project="codfish-test"),
                    ),
                ):
                    pass

        self.assertEqual(fake_wandb.run.finish_calls, 1)


class LearnerBindingsTest(unittest.TestCase):
    def test_get_model_io_shape_matches_encoded_sample_metadata(self) -> None:
        shape = get_model_io_shape()
        samples = encode_raw_game(_canonical_raw_game())

        self.assertGreater(shape.input_channels, 0)
        self.assertGreater(shape.policy_size, 0)
        self.assertEqual(shape.input_channels, samples.input_channels)
        self.assertEqual(shape.policy_size, samples.policy_size)

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
    def test_ingest_chunk_files_returns_new_sample_count_and_matches_encoded_sample(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            chunk_path = pathlib.Path(tmp_dir) / "canonical.bin"
            _write_chunk_file(chunk_path, [_canonical_raw_game()])

            buffer = ReplayBuffer(
                ReplayBufferConfig(
                    sample_capacity=8, batch_size=2, replay_ratio=1.5, seed=7
                )
            )
            new_sample_count = buffer.ingest_chunk_files([chunk_path])

        self.assertEqual(new_sample_count, 1)
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
            first_new_sample_count = buffer.ingest_chunk_files([first_chunk])
            second_new_sample_count = buffer.ingest_chunk_files([second_chunk])

        self.assertEqual(first_new_sample_count, 1)
        self.assertEqual(second_new_sample_count, 1)
        self.assertEqual(buffer.sample_count, 1)

        batch = buffer.sample_minibatch()
        expected = encode_raw_game(_alternate_raw_game())
        np.testing.assert_array_equal(batch.inputs, expected.inputs)
        np.testing.assert_array_equal(batch.policy_targets, expected.policy_targets)
        np.testing.assert_array_equal(batch.wdl_targets, expected.wdl_targets)

    def test_ingest_chunk_files_is_atomic_when_any_chunk_is_malformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            first_chunk = pathlib.Path(tmp_dir) / "first.bin"
            second_chunk = pathlib.Path(tmp_dir) / "second.bin"
            bad_chunk = pathlib.Path(tmp_dir) / "bad.bin"
            _write_chunk_file(first_chunk, [_canonical_raw_game()])
            _write_chunk_file(second_chunk, [_alternate_raw_game()])
            bad_chunk.write_bytes(b"CFRG\x01\x00")

            buffer = ReplayBuffer(
                ReplayBufferConfig(sample_capacity=4, batch_size=1, replay_ratio=1.0)
            )
            buffer.ingest_chunk_files([first_chunk])
            with self.assertRaisesRegex(RuntimeError, os.fspath(bad_chunk)):
                buffer.ingest_chunk_files([second_chunk, bad_chunk])

        self.assertEqual(buffer.sample_count, 1)
        batch = buffer.sample_minibatch()
        expected = encode_raw_game(_canonical_raw_game())
        np.testing.assert_array_equal(batch.inputs, expected.inputs)
        np.testing.assert_array_equal(batch.policy_targets, expected.policy_targets)
        np.testing.assert_array_equal(batch.wdl_targets, expected.wdl_targets)

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
