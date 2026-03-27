from __future__ import annotations

import copy
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from codfish.learner import (
    GameResult,
    LearnerRunner,
    LearnerRunnerConfig,
    ModelSpec,
    RawGame,
    RawPly,
    RawPolicyEntry,
    ReplayBufferConfig,
    TrainIterationReport,
    TrainerConfig,
    WandbConfig,
)
from codfish.learner._api import get_model_io_shape, read_raw_chunk_file
from codfish.learner._checkpoint import (
    atomic_torch_save,
    build_training_checkpoint_payload,
    load_training_checkpoint,
)
from codfish.orchestrator import (
    NativeSelfPlayLauncher,
    SelfPlayConfig,
    _partial_iteration_dir,
    run_selfplay_update_loop,
)


CHUNK_MAGIC = b"CFRG"
CHUNK_VERSION = 1
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(data))


def _model_shape() -> tuple[int, int]:
    shape = get_model_io_shape()
    return shape.input_channels, shape.policy_size


def _toy_model_spec() -> ModelSpec:
    input_channels, policy_size = _model_shape()

    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            flat_size = input_channels * 8 * 8
            self.policy_head = torch.nn.Linear(flat_size, policy_size)
            self.wdl_head = torch.nn.Linear(flat_size, 3)

        def forward(
            self, inputs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            flat = inputs.reshape(inputs.shape[0], -1)
            return self.policy_head(flat), self.wdl_head(flat)

    return ModelSpec(
        name="toy-model",
        config={
            "kind": "toy",
            "input_channels": input_channels,
            "policy_size": policy_size,
        },
        factory=ToyModel,
    )


def _trainer_config() -> TrainerConfig:
    return TrainerConfig(
        learning_rate=0.05,
        optimizer_momentum=0.0,
        optimizer_weight_decay=0.0,
        value_loss_weight=1.0,
    )


def _replay_config() -> ReplayBufferConfig:
    return ReplayBufferConfig(
        sample_capacity=8,
        batch_size=2,
        replay_ratio=1.0,
        seed=7,
    )


def _runner_config(run_root: pathlib.Path, *, resume: bool = False) -> LearnerRunnerConfig:
    return LearnerRunnerConfig(
        device="cpu",
        checkpoint_dir=run_root / "learner",
        resume=resume,
        wandb=None,
    )


def _selfplay_config() -> SelfPlayConfig:
    return SelfPlayConfig(
        num_workers=1,
        num_games=1,
        raw_chunk_max_bytes=128 * 1024 * 1024,
        num_action=4,
        num_simulation=8,
        c_puct=1.0,
        c_visit=1.0,
        c_scale=1.0,
    )


def _write_minimal_checkpoint(
    learner_dir: pathlib.Path,
    *,
    iteration: int,
    trainer_config: TrainerConfig,
    wandb_run_id: str | None = None,
    replay_sampler_rng_state: dict[str, object] | None = None,
) -> None:
    payload = build_training_checkpoint_payload(
        model_state_dict={},
        optimizer_state_dict={},
        global_learner_step=0,
        iteration=iteration,
        model_name="toy-model",
        model_config={
            "kind": "toy",
            "input_channels": _model_shape()[0],
            "policy_size": _model_shape()[1],
        },
        trainer_config=trainer_config,
        wandb_run_id=wandb_run_id,
        replay_sampler_rng_state=replay_sampler_rng_state,
    )
    atomic_torch_save(payload, learner_dir / "latest.pt")


class _FakeWandbRun:
    def __init__(self, run_id: str = "fake-wandb-run-id") -> None:
        self.id = run_id
        self.define_metric_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.log_calls: list[dict[str, object]] = []
        self.finish_calls = 0

    def define_metric(self, *args: object, **kwargs: object) -> None:
        self.define_metric_calls.append((args, dict(kwargs)))

    def log(self, payload: dict[str, object]) -> None:
        self.log_calls.append(dict(payload))

    def finish(self) -> None:
        self.finish_calls += 1


class _FakeWandbModule:
    def __init__(self, run_id: str = "fake-wandb-run-id") -> None:
        self.init_calls: list[dict[str, object]] = []
        self.run = _FakeWandbRun(run_id)

    def init(self, **kwargs: object) -> _FakeWandbRun:
        self.init_calls.append(dict(kwargs))
        return self.run


class _FakeLearnerRunner:
    instances: list["_FakeLearnerRunner"] = []

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
        self.trainer = type("FakeTrainer", (), {"wandb_run_id": None})()
        self.run_calls: list[tuple[list[pathlib.Path], list[pathlib.Path], int]] = []
        self.closed = False
        type(self).instances.append(self)

    def run_iteration(
        self,
        historical_chunk_paths: list[str | os.PathLike[str]],
        new_chunk_paths: list[str | os.PathLike[str]],
        iteration: int,
    ) -> TrainIterationReport:
        historical = [pathlib.Path(path) for path in historical_chunk_paths]
        new = [pathlib.Path(path) for path in new_chunk_paths]
        self.run_calls.append((historical, new, iteration))
        return TrainIterationReport(
            iteration=iteration,
            starting_global_step=iteration - 1,
            ending_global_step=iteration,
            num_updates=1,
            new_sample_count=1,
            replay_sample_count=len(historical) + len(new),
            mean_total_loss=1.0,
            mean_policy_loss=0.5,
            mean_wdl_loss=0.5,
            latest_checkpoint_path=pathlib.Path(f"/tmp/latest-{iteration}.pt"),
            snapshot_path=pathlib.Path(f"/tmp/snapshot-{iteration}.pt"),
        )

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "_FakeLearnerRunner":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class OrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeLearnerRunner.instances = []

    def test_run_selfplay_update_loop_fresh_uses_ephemeral_runners(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(output_dir / "games-000001.bin", [_canonical_raw_game()])

            with mock.patch(
                "codfish.orchestrator.LearnerRunner",
                _FakeLearnerRunner,
            ), mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                new=fake_run_selfplay,
            ):
                reports = run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=2,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=False),
                )

        self.assertEqual([report.iteration for report in reports], [1, 2])
        self.assertEqual(len(_FakeLearnerRunner.instances), 2)
        self.assertEqual(
            [instance.config.resume for instance in _FakeLearnerRunner.instances],
            [False, True],
        )
        self.assertTrue(all(instance.closed for instance in _FakeLearnerRunner.instances))
        self.assertEqual(
            [instance.run_calls[0][2] for instance in _FakeLearnerRunner.instances],
            [1, 2],
        )

    def test_run_selfplay_update_loop_resume_uses_latest_iteration_plus_one(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            learner_dir = run_root / "learner"
            learner_dir.mkdir(parents=True, exist_ok=True)
            _write_minimal_checkpoint(
                learner_dir,
                iteration=5,
                trainer_config=_trainer_config(),
            )
            for iteration in range(1, 6):
                (run_root / "selfplay" / f"iter_{iteration:06d}").mkdir(
                    parents=True,
                    exist_ok=True,
                )

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(output_dir / "games-000001.bin", [_canonical_raw_game()])

            with mock.patch(
                "codfish.orchestrator.LearnerRunner",
                _FakeLearnerRunner,
            ), mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                new=fake_run_selfplay,
            ):
                reports = run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=True),
                )

        self.assertEqual([report.iteration for report in reports], [6])
        self.assertEqual(_FakeLearnerRunner.instances[0].config.resume, True)

    def test_run_selfplay_update_loop_rejects_missing_latest_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "latest.pt"):
                run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=True),
                )

    def test_run_selfplay_update_loop_rejects_missing_historical_dir_on_resume(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            learner_dir = run_root / "learner"
            learner_dir.mkdir(parents=True, exist_ok=True)
            _write_minimal_checkpoint(
                learner_dir,
                iteration=2,
                trainer_config=_trainer_config(),
            )
            (run_root / "selfplay" / "iter_000001").mkdir(parents=True, exist_ok=True)

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(output_dir / "games-000001.bin", [_canonical_raw_game()])

            with mock.patch(
                "codfish.orchestrator.LearnerRunner",
                _FakeLearnerRunner,
            ), mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                new=fake_run_selfplay,
            ), self.assertRaisesRegex(FileNotFoundError, "iter_000002"):
                run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=True),
                )

    def test_run_selfplay_update_loop_discovers_sorted_chunk_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            learner_dir = run_root / "learner"
            learner_dir.mkdir(parents=True, exist_ok=True)
            _write_minimal_checkpoint(
                learner_dir,
                iteration=2,
                trainer_config=_trainer_config(),
            )
            _write_chunk_file(
                run_root / "selfplay" / "iter_000001" / "games-000002.bin",
                [_canonical_raw_game()],
            )
            _write_chunk_file(
                run_root / "selfplay" / "iter_000001" / "games-000001.bin",
                [_canonical_raw_game()],
            )
            _write_chunk_file(
                run_root / "selfplay" / "iter_000002" / "games-000002.bin",
                [_alternate_raw_game()],
            )
            _write_chunk_file(
                run_root / "selfplay" / "iter_000002" / "games-000001.bin",
                [_alternate_raw_game()],
            )

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(output_dir / "games-000002.bin", [_canonical_raw_game()])
                _write_chunk_file(output_dir / "games-000001.bin", [_canonical_raw_game()])

            with mock.patch(
                "codfish.orchestrator.LearnerRunner",
                _FakeLearnerRunner,
            ), mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                new=fake_run_selfplay,
            ):
                run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=True),
                )

        historical, new, iteration = _FakeLearnerRunner.instances[0].run_calls[0]
        self.assertEqual(iteration, 3)
        self.assertEqual(
            historical,
            [
                run_root / "selfplay" / "iter_000001" / "games-000001.bin",
                run_root / "selfplay" / "iter_000001" / "games-000002.bin",
                run_root / "selfplay" / "iter_000002" / "games-000001.bin",
                run_root / "selfplay" / "iter_000002" / "games-000002.bin",
            ],
        )
        self.assertEqual(
            new,
            [
                run_root / "selfplay" / "iter_000003" / "games-000001.bin",
                run_root / "selfplay" / "iter_000003" / "games-000002.bin",
            ],
        )

    def test_orchestrator_wandb_logs_once_per_iteration(self) -> None:
        fake_wandb = _FakeWandbModule()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(output_dir / "games-000001.bin", [_canonical_raw_game()])

            with mock.patch(
                "codfish.orchestrator.LearnerRunner",
                _FakeLearnerRunner,
            ), mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                new=fake_run_selfplay,
            ), mock.patch(
                "importlib.import_module",
                return_value=fake_wandb,
            ):
                run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=2,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=False),
                    wandb_config=WandbConfig(project="proj", name="run"),
                )

        self.assertEqual(len(fake_wandb.init_calls), 1)
        self.assertEqual(len(fake_wandb.run.log_calls), 2)
        self.assertEqual(fake_wandb.run.finish_calls, 1)
        self.assertNotIn("id", fake_wandb.init_calls[0])
        self.assertNotIn("resume", fake_wandb.init_calls[0])

    def test_orchestrator_resume_reuses_checkpointed_wandb_run_id(self) -> None:
        fake_wandb = _FakeWandbModule(run_id="resumed-run-id")
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            learner_dir = run_root / "learner"
            learner_dir.mkdir(parents=True, exist_ok=True)
            _write_minimal_checkpoint(
                learner_dir,
                iteration=2,
                trainer_config=_trainer_config(),
                wandb_run_id="persisted-run-id",
            )
            for iteration in range(1, 3):
                (run_root / "selfplay" / f"iter_{iteration:06d}").mkdir(
                    parents=True,
                    exist_ok=True,
                )

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(output_dir / "games-000001.bin", [_canonical_raw_game()])

            with mock.patch(
                "codfish.orchestrator.LearnerRunner",
                _FakeLearnerRunner,
            ), mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                new=fake_run_selfplay,
            ), mock.patch(
                "importlib.import_module",
                return_value=fake_wandb,
            ):
                run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=True),
                    wandb_config=WandbConfig(project="proj", name="resume-run"),
                )

        self.assertEqual(len(fake_wandb.init_calls), 1)
        self.assertEqual(fake_wandb.init_calls[0]["id"], "persisted-run-id")
        self.assertEqual(fake_wandb.init_calls[0]["resume"], "must")

    def test_orchestrator_persists_wandb_run_id_to_checkpoint(self) -> None:
        fake_wandb = _FakeWandbModule(run_id="persisted-run-id")
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(output_dir / "games-000001.bin", [_canonical_raw_game()])

            with mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                new=fake_run_selfplay,
            ), mock.patch(
                "importlib.import_module",
                return_value=fake_wandb,
            ):
                run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=False),
                    wandb_config=WandbConfig(project="proj", name="persist-run"),
                )

            checkpoint = load_training_checkpoint(
                run_root / "learner" / "latest.pt",
                map_location=torch.device("cpu"),
            )

        self.assertEqual(checkpoint.wandb_run_id, "persisted-run-id")

    def test_run_selfplay_update_loop_cleans_partial_dir_after_selfplay_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            launcher_calls = 0

            def failing_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                output_dir: pathlib.Path,
            ) -> None:
                nonlocal launcher_calls
                launcher_calls += 1
                _write_chunk_file(output_dir / "games-000001.bin", [_canonical_raw_game()])
                raise RuntimeError("injected self-play failure")

            with mock.patch(
                "codfish.orchestrator.LearnerRunner",
                _FakeLearnerRunner,
            ), mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                new=failing_run_selfplay,
            ), self.assertRaisesRegex(RuntimeError, "injected self-play failure"):
                run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=False),
                )

            iteration_dir = run_root / "selfplay" / "iter_000001"
            self.assertFalse(iteration_dir.exists())
            self.assertFalse(_partial_iteration_dir(iteration_dir).exists())
            self.assertEqual(launcher_calls, 1)

            def successful_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                output_dir: pathlib.Path,
            ) -> None:
                nonlocal launcher_calls
                launcher_calls += 1
                _write_chunk_file(output_dir / "games-000001.bin", [_canonical_raw_game()])

            with mock.patch(
                "codfish.orchestrator.LearnerRunner",
                _FakeLearnerRunner,
            ), mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                new=successful_run_selfplay,
            ):
                reports = run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=False),
                )

        self.assertEqual([report.iteration for report in reports], [1])
        self.assertEqual(launcher_calls, 2)

    def test_run_selfplay_update_loop_reuses_existing_iteration_dir_after_learner_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            launcher_calls = 0

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                output_dir: pathlib.Path,
            ) -> None:
                nonlocal launcher_calls
                launcher_calls += 1
                _write_chunk_file(output_dir / "games-000001.bin", [_canonical_raw_game()])

            class FailingLearnerRunner(_FakeLearnerRunner):
                def run_iteration(
                    self,
                    historical_chunk_paths: list[str | os.PathLike[str]],
                    new_chunk_paths: list[str | os.PathLike[str]],
                    iteration: int,
                ) -> TrainIterationReport:
                    raise RuntimeError("injected learner failure")

            with mock.patch(
                "codfish.orchestrator.LearnerRunner",
                FailingLearnerRunner,
            ), mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                new=fake_run_selfplay,
            ), self.assertRaisesRegex(RuntimeError, "injected learner failure"):
                run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=False),
                )

            iteration_dir = run_root / "selfplay" / "iter_000001"
            self.assertTrue(iteration_dir.is_dir())
            self.assertFalse(_partial_iteration_dir(iteration_dir).exists())
            self.assertEqual(launcher_calls, 1)

            with mock.patch(
                "codfish.orchestrator.LearnerRunner",
                _FakeLearnerRunner,
            ), mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                side_effect=AssertionError("existing self-play dir should be reused"),
            ):
                reports = run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=False),
                )

        self.assertEqual([report.iteration for report in reports], [1])

    def test_resume_restores_replay_sampler_rng_state_on_recreated_runner(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            model_spec = _toy_model_spec()
            trainer_config = _trainer_config()
            replay_buffer_config = _replay_config()

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                output_dir: pathlib.Path,
            ) -> None:
                iteration = int(output_dir.name.removesuffix(".partial").split("_")[1])
                if iteration == 1:
                    _write_chunk_file(
                        output_dir / "games-000001.bin",
                        [_canonical_raw_game(), _alternate_raw_game()],
                    )
                else:
                    _write_chunk_file(
                        output_dir / "games-000001.bin",
                        [_alternate_raw_game()],
                    )

            with mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                new=fake_run_selfplay,
            ):
                first_reports = run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=model_spec,
                    trainer_config=trainer_config,
                    replay_buffer_config=replay_buffer_config,
                    runner_config=_runner_config(run_root, resume=False),
                )

            self.assertEqual(first_reports[0].iteration, 1)
            first_checkpoint = load_training_checkpoint(
                run_root / "learner" / "latest.pt",
                map_location=torch.device("cpu"),
            )
            self.assertIsNotNone(first_checkpoint.replay_sampler_rng_state)

            class RecordingLearnerRunner(LearnerRunner):
                observed_rng_states: list[dict[str, object]] = []

                def __init__(self, *args, **kwargs) -> None:
                    super().__init__(*args, **kwargs)
                    type(self).observed_rng_states.append(
                        copy.deepcopy(self._replay_rng.bit_generator.state)
                    )

            with mock.patch(
                "codfish.orchestrator.LearnerRunner",
                RecordingLearnerRunner,
            ), mock.patch.object(
                NativeSelfPlayLauncher,
                "run_selfplay",
                new=fake_run_selfplay,
            ):
                second_reports = run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=model_spec,
                    trainer_config=trainer_config,
                    replay_buffer_config=replay_buffer_config,
                    runner_config=_runner_config(run_root, resume=True),
                )

        self.assertEqual(second_reports[0].iteration, 2)
        self.assertEqual(
            RecordingLearnerRunner.observed_rng_states[0],
            first_checkpoint.replay_sampler_rng_state,
        )

    def test_orchestrator_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir) / "run"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "codfish.orchestrator",
                    "--run-root",
                    os.fspath(run_root),
                    "--num-iterations",
                    "1",
                    "--learning-rate",
                    "0.01",
                    "--optimizer-momentum",
                    "0.0",
                    "--optimizer-weight-decay",
                    "0.0",
                    "--value-loss-weight",
                    "1.0",
                    "--sample-capacity",
                    "32",
                    "--batch-size",
                    "32",
                    "--replay-ratio",
                    "0.25",
                    "--num-workers",
                    "1",
                    "--num-games",
                    "1",
                    "--num-action",
                    "4",
                    "--num-simulation",
                    "8",
                    "--c-puct",
                    "1.0",
                    "--c-visit",
                    "1.0",
                    "--c-scale",
                    "1.0",
                ],
                check=True,
            )

            self.assertTrue((run_root / "learner" / "latest.pt").exists())
            self.assertTrue(sorted((run_root / "selfplay" / "iter_000001").glob("*.bin")))
