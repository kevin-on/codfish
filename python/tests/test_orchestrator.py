from __future__ import annotations

import copy
import importlib
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock

import torch

from codfish import _run_layout
from codfish._run_layout import partial_path
from codfish.artifacts import (
    AOTI_ARTIFACT_FORMAT_VERSION,
    AOTI_MANIFEST_FILE,
    AOTI_PACKAGE_FILE,
    AOTI_RUNTIME,
    AOTI_TARGET_DEVICE,
)
from codfish.learner import (
    GameResult,
    LearnerRunner,
    LearnerRunnerConfig,
    ModelSpec,
    RawGame,
    RawPly,
    RawPolicyEntry,
    ReplayBufferConfig,
    TrainerConfig,
    TrainIterationReport,
    WandbConfig,
)
from codfish.learner._api import get_model_io_shape
from codfish.learner._checkpoint import (
    atomic_torch_save,
    build_training_checkpoint_payload,
    load_training_checkpoint,
)
from codfish.orchestrator import (
    EvalConfig,
    NativeSelfPlayLauncher,
    SelfPlayConfig,
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

        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def _counter_model_spec() -> ModelSpec:
    input_channels, policy_size = _model_shape()
    init_counter = 0

    class CounterModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            nonlocal init_counter
            init_counter += 1
            flat_size = input_channels * 8 * 8
            self.policy_head = torch.nn.Linear(flat_size, policy_size)
            self.wdl_head = torch.nn.Linear(flat_size, 3)
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter.fill_(float(init_counter))

        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            flat = inputs.reshape(inputs.shape[0], -1)
            return self.policy_head(flat), self.wdl_head(flat)

    return ModelSpec(
        name="counter-model",
        config={
            "kind": "counter",
            "input_channels": input_channels,
            "policy_size": policy_size,
        },
        factory=CounterModel,
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


def _runner_config(
    run_root: pathlib.Path, *, resume: bool = False
) -> LearnerRunnerConfig:
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


def _write_fake_artifact(
    artifact_dir: pathlib.Path,
    *,
    model_name: str,
    model_config: dict[str, object],
    iteration: int,
    global_learner_step: int,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / AOTI_PACKAGE_FILE).write_bytes(b"fake-pt2-package")
    (artifact_dir / AOTI_MANIFEST_FILE).write_text(
        json.dumps(
            {
                "format_version": AOTI_ARTIFACT_FORMAT_VERSION,
                "runtime": AOTI_RUNTIME,
                "target_device": AOTI_TARGET_DEVICE,
                "package_file": AOTI_PACKAGE_FILE,
                "model_name": model_name,
                "model_config": dict(model_config),
                "input_channels": model_config["input_channels"],
                "policy_size": model_config["policy_size"],
                "iteration": iteration,
                "global_learner_step": global_learner_step,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


class _FakeWandbRun:
    def __init__(self, run_id: str = "fake-wandb-run-id") -> None:
        self.id = run_id
        self.define_metric_calls: list[
            tuple[tuple[object, ...], dict[str, object]]
        ] = []
        self.log_calls: list[dict[str, object]] = []
        self.finish_calls = 0

    def define_metric(self, *args: object, **kwargs: object) -> None:
        self.define_metric_calls.append((args, dict(kwargs)))

    def log(self, payload: dict[str, object]) -> None:
        self.log_calls.append(dict(payload))

    def finish(self) -> None:
        self.finish_calls += 1


class _FakeWandbTable:
    def __init__(self, *, columns: list[str], data: list[list[object]]) -> None:
        self.columns = list(columns)
        self.data = [list(row) for row in data]


class _FakeWandbLinePlot:
    def __init__(
        self,
        *,
        table: _FakeWandbTable,
        x: str,
        y: str,
        stroke: str | None = None,
        title: str = "",
        split_table: bool = False,
    ) -> None:
        self.table = table
        self.x = x
        self.y = y
        self.stroke = stroke
        self.title = title
        self.split_table = split_table


class _FakeWandbPlotNamespace:
    @staticmethod
    def line(
        *,
        table: _FakeWandbTable,
        x: str,
        y: str,
        stroke: str | None = None,
        title: str = "",
        split_table: bool = False,
    ) -> _FakeWandbLinePlot:
        return _FakeWandbLinePlot(
            table=table,
            x=x,
            y=y,
            stroke=stroke,
            title=title,
            split_table=split_table,
        )


class _FakeWandbModule:
    def __init__(self, run_id: str = "fake-wandb-run-id") -> None:
        self.init_calls: list[dict[str, object]] = []
        self.run = _FakeWandbRun(run_id)
        self.Table = _FakeWandbTable
        self.plot = _FakeWandbPlotNamespace()

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
        self.model = model_spec.factory()
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
        checkpoint_dir = pathlib.Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_checkpoint_path = _run_layout.latest_checkpoint_path(checkpoint_dir)
        snapshot_path = _run_layout.snapshot_path(
            checkpoint_dir,
            iteration=iteration,
            global_step=iteration,
        )
        payload = build_training_checkpoint_payload(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict={},
            global_learner_step=iteration,
            iteration=iteration,
            model_name=self.model_spec.name,
            model_config=self.model_spec.config,
            trainer_config=self.trainer_config,
            wandb_run_id=self.trainer.wandb_run_id,
            replay_sampler_rng_state=None,
        )
        atomic_torch_save(payload, latest_checkpoint_path)
        atomic_torch_save(payload, snapshot_path)
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
            latest_checkpoint_path=latest_checkpoint_path,
            snapshot_path=snapshot_path,
        )

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "_FakeLearnerRunner":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _clone_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _eval_config(
    *,
    snapshot_interval: int = 1,
    match_games: int = 2,
    num_workers: int = 1,
    num_action: int = 4,
    num_simulation: int = 8,
    c_puct: float = 1.0,
    c_visit: float = 1.0,
    c_scale: float = 1.0,
    window_offsets: tuple[int, ...] = (1, 3, 10),
) -> EvalConfig:
    return EvalConfig(
        snapshot_interval=snapshot_interval,
        match_games=match_games,
        num_workers=num_workers,
        num_action=num_action,
        num_simulation=num_simulation,
        c_puct=c_puct,
        c_visit=c_visit,
        c_scale=c_scale,
        window_offsets=window_offsets,
    )


def _wandb_import_side_effect(fake_wandb: _FakeWandbModule):
    real_import_module = importlib.import_module

    def side_effect(name: str, package: str | None = None):
        if name == "wandb":
            return fake_wandb
        return real_import_module(name, package)

    return side_effect


class OrchestratorTest(unittest.TestCase):
    def setUp(self) -> None:
        _FakeLearnerRunner.instances = []

    def _fake_export_artifact(
        self,
        *,
        model: torch.nn.Module,
        model_name: str,
        model_config: dict[str, object],
        artifact_dir_path: str | os.PathLike[str],
        iteration: int,
        global_learner_step: int,
    ) -> pathlib.Path:
        del model
        artifact_path = pathlib.Path(artifact_dir_path)
        _write_fake_artifact(
            artifact_path,
            model_name=model_name,
            model_config=model_config,
            iteration=iteration,
            global_learner_step=global_learner_step,
        )
        return artifact_path

    def _fake_regenerate_artifact(
        self,
        *,
        model_spec: ModelSpec,
        checkpoint_path: str | os.PathLike[str],
        artifact_dir_path: str | os.PathLike[str],
        expected_iteration: int | None = None,
    ) -> pathlib.Path:
        checkpoint = load_training_checkpoint(
            checkpoint_path,
            map_location=torch.device("cpu"),
        )
        if expected_iteration is not None:
            self.assertEqual(checkpoint.iteration, expected_iteration)
        artifact_path = pathlib.Path(artifact_dir_path)
        _write_fake_artifact(
            artifact_path,
            model_name=model_spec.name,
            model_config=model_spec.config,
            iteration=checkpoint.iteration,
            global_learner_step=checkpoint.global_learner_step,
        )
        return artifact_path

    @contextmanager
    def _patched_aoti(self):
        with (
            mock.patch(
                "codfish.orchestrator.torch.cuda.is_available",
                return_value=True,
            ),
            mock.patch(
                "codfish.orchestrator.export_model_to_aoti_artifact",
                side_effect=self._fake_export_artifact,
            ),
            mock.patch(
                "codfish.orchestrator.regenerate_aoti_artifact_from_checkpoint",
                side_effect=self._fake_regenerate_artifact,
            ),
        ):
            yield

    def test_run_selfplay_update_loop_fresh_uses_ephemeral_runners(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            seen_artifact_names: list[str] = []

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                seen_artifact_names.append(artifact_dir.name)
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    _FakeLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
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

            self.assertTrue(
                (run_root / "artifacts" / "iter_000000" / AOTI_MANIFEST_FILE).exists()
            )
            self.assertTrue(
                (run_root / "artifacts" / "iter_000001" / AOTI_MANIFEST_FILE).exists()
            )
            self.assertTrue(
                (run_root / "artifacts" / "iter_000002" / AOTI_MANIFEST_FILE).exists()
            )
            self.assertEqual(seen_artifact_names, ["iter_000000", "iter_000001"])

        self.assertEqual([report.iteration for report in reports], [1, 2])
        self.assertEqual(len(_FakeLearnerRunner.instances), 2)
        self.assertEqual(
            [instance.config.resume for instance in _FakeLearnerRunner.instances],
            [False, True],
        )
        self.assertTrue(
            all(instance.closed for instance in _FakeLearnerRunner.instances)
        )
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
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                self.assertEqual(artifact_dir.name, "iter_000005")
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    _FakeLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
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
            with (
                self._patched_aoti(),
                self.assertRaisesRegex(FileNotFoundError, "latest.pt"),
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

    def test_run_selfplay_update_loop_rejects_missing_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=False),
                )

    def test_run_selfplay_update_loop_rejects_mismatched_bootstrap_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            _write_fake_artifact(
                run_root / "artifacts" / "iter_000000",
                model_name="wrong-model",
                model_config={
                    "kind": "toy",
                    "input_channels": _model_shape()[0],
                    "policy_size": _model_shape()[1],
                },
                iteration=0,
                global_learner_step=0,
            )

            with (
                mock.patch(
                    "codfish.orchestrator.torch.cuda.is_available",
                    return_value=True,
                ),
                self.assertRaisesRegex(ValueError, "artifact model_name"),
            ):
                run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=False),
                )

    def test_run_selfplay_update_loop_regenerates_missing_resume_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            learner_dir = run_root / "learner"
            learner_dir.mkdir(parents=True, exist_ok=True)
            _write_minimal_checkpoint(
                learner_dir,
                iteration=2,
                trainer_config=_trainer_config(),
            )
            for iteration in range(1, 3):
                (run_root / "selfplay" / f"iter_{iteration:06d}").mkdir(
                    parents=True,
                    exist_ok=True,
                )

            regenerate_calls: list[tuple[pathlib.Path, pathlib.Path]] = []

            def recording_regenerate(
                *,
                model_spec: ModelSpec,
                checkpoint_path: str | os.PathLike[str],
                artifact_dir_path: str | os.PathLike[str],
                expected_iteration: int | None = None,
            ) -> pathlib.Path:
                regenerate_calls.append(
                    (pathlib.Path(checkpoint_path), pathlib.Path(artifact_dir_path))
                )
                return self._fake_regenerate_artifact(
                    model_spec=model_spec,
                    checkpoint_path=checkpoint_path,
                    artifact_dir_path=artifact_dir_path,
                    expected_iteration=expected_iteration,
                )

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                self.assertEqual(artifact_dir.name, "iter_000002")
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            with (
                mock.patch(
                    "codfish.orchestrator.torch.cuda.is_available",
                    return_value=True,
                ),
                mock.patch(
                    "codfish.orchestrator.export_model_to_aoti_artifact",
                    side_effect=self._fake_export_artifact,
                ),
                mock.patch(
                    "codfish.orchestrator.regenerate_aoti_artifact_from_checkpoint",
                    side_effect=recording_regenerate,
                ),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    _FakeLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
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

            self.assertEqual([report.iteration for report in reports], [3])
            self.assertEqual(
                regenerate_calls,
                [
                    (
                        learner_dir / "latest.pt",
                        run_root / "artifacts" / "iter_000002",
                    )
                ],
            )
            self.assertTrue(
                (run_root / "artifacts" / "iter_000002" / AOTI_MANIFEST_FILE).exists()
            )

    def test_run_selfplay_update_loop_rejects_mismatched_existing_resume_artifact(
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
            for iteration in range(1, 3):
                (run_root / "selfplay" / f"iter_{iteration:06d}").mkdir(
                    parents=True,
                    exist_ok=True,
                )
            _write_fake_artifact(
                run_root / "artifacts" / "iter_000002",
                model_name="toy-model",
                model_config={
                    "kind": "wrong-kind",
                    "input_channels": _model_shape()[0],
                    "policy_size": _model_shape()[1],
                },
                iteration=2,
                global_learner_step=0,
            )

            with (
                mock.patch(
                    "codfish.orchestrator.torch.cuda.is_available",
                    return_value=True,
                ),
                self.assertRaisesRegex(ValueError, "artifact model_config"),
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

    def test_fresh_run_bootstrap_artifact_matches_iteration_one_start_weights(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            model_spec = _counter_model_spec()
            bootstrap_state: dict[str, torch.Tensor] | None = None

            def recording_export(
                *,
                model: torch.nn.Module,
                model_name: str,
                model_config: dict[str, object],
                artifact_dir_path: str | os.PathLike[str],
                iteration: int,
                global_learner_step: int,
            ) -> pathlib.Path:
                nonlocal bootstrap_state
                artifact_path = pathlib.Path(artifact_dir_path)
                _write_fake_artifact(
                    artifact_path,
                    model_name=model_name,
                    model_config=model_config,
                    iteration=iteration,
                    global_learner_step=global_learner_step,
                )
                if iteration == 0:
                    bootstrap_state = _clone_state_dict(model.state_dict())
                return artifact_path

            class RecordingLearnerRunner(_FakeLearnerRunner):
                observed_initial_state: dict[str, torch.Tensor] | None = None

                def run_iteration(
                    self,
                    historical_chunk_paths: list[str | os.PathLike[str]],
                    new_chunk_paths: list[str | os.PathLike[str]],
                    iteration: int,
                ) -> TrainIterationReport:
                    type(self).observed_initial_state = _clone_state_dict(
                        self.model.state_dict()
                    )
                    return super().run_iteration(
                        historical_chunk_paths,
                        new_chunk_paths,
                        iteration,
                    )

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            with (
                mock.patch(
                    "codfish.orchestrator.torch.cuda.is_available",
                    return_value=True,
                ),
                mock.patch(
                    "codfish.orchestrator.export_model_to_aoti_artifact",
                    side_effect=recording_export,
                ),
                mock.patch(
                    "codfish.orchestrator.regenerate_aoti_artifact_from_checkpoint",
                    side_effect=self._fake_regenerate_artifact,
                ),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    RecordingLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
            ):
                reports = run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=1,
                    selfplay_config=_selfplay_config(),
                    model_spec=model_spec,
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=False),
                )

        self.assertEqual([report.iteration for report in reports], [1])
        self.assertIsNotNone(bootstrap_state)
        self.assertIsNotNone(RecordingLearnerRunner.observed_initial_state)
        self.assertEqual(
            bootstrap_state.keys(),
            RecordingLearnerRunner.observed_initial_state.keys(),
        )
        for key, value in bootstrap_state.items():
            self.assertTrue(
                torch.equal(value, RecordingLearnerRunner.observed_initial_state[key]),
                msg=f"bootstrap and runner state differ for {key}",
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
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    _FakeLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
                self.assertRaisesRegex(FileNotFoundError, "iter_000002"),
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
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(
                    output_dir / "games-000002.bin", [_canonical_raw_game()]
                )
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    _FakeLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
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
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    _FakeLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
                mock.patch(
                    "codfish.orchestrator.importlib.import_module",
                    side_effect=_wandb_import_side_effect(fake_wandb),
                ),
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
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    _FakeLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
                mock.patch(
                    "codfish.orchestrator.importlib.import_module",
                    side_effect=_wandb_import_side_effect(fake_wandb),
                ),
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
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            with (
                self._patched_aoti(),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
                mock.patch(
                    "codfish.orchestrator.importlib.import_module",
                    side_effect=_wandb_import_side_effect(fake_wandb),
                ),
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

    def test_eval_phase_runs_matches_rebuilds_ratings_and_logs_wandb(self) -> None:
        fake_wandb = _FakeWandbModule()
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            match_calls: list[tuple[str, str, pathlib.Path]] = []
            ordo_calls: list[list[str]] = []

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            def fake_run_match(
                model_package_path_a: str | os.PathLike[str],
                model_package_path_b: str | os.PathLike[str],
                *,
                player_name_a: str,
                player_name_b: str,
                input_channels: int,
                policy_size: int,
                output_pgn_path: str | os.PathLike[str],
                num_workers: int,
                num_games: int,
                num_action: int,
                num_simulation: int,
                c_puct: float,
                c_visit: float,
                c_scale: float,
            ) -> None:
                del (
                    model_package_path_a,
                    model_package_path_b,
                    input_channels,
                    policy_size,
                    num_workers,
                    num_games,
                    num_action,
                    num_simulation,
                    c_puct,
                    c_visit,
                    c_scale,
                )
                output_path = pathlib.Path(output_pgn_path)
                match_calls.append((player_name_a, player_name_b, output_path))
                output_path.write_text(
                    (
                        f'[Event "Eval"]\n[White "{player_name_a}"]\n'
                        f'[Black "{player_name_b}"]\n[Result "1-0"]\n\n1-0\n\n'
                        f'[Event "Eval"]\n[White "{player_name_b}"]\n'
                        f'[Black "{player_name_a}"]\n[Result "0-1"]\n\n0-1\n'
                    ),
                    encoding="utf-8",
                )

            def fake_ordo_run(
                cmd: list[str],
                *,
                check: bool,
                capture_output: bool,
                text: bool,
            ) -> subprocess.CompletedProcess[str]:
                self.assertTrue(check)
                self.assertTrue(capture_output)
                self.assertTrue(text)
                ordo_calls.append(list(cmd))
                all_games_path = pathlib.Path(cmd[cmd.index("-p") + 1])
                ratings_path = pathlib.Path(cmd[cmd.index("-o") + 1])
                player_names = sorted(
                    {
                        line.split('"')[1]
                        for line in all_games_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if line.startswith("[White ") or line.startswith("[Black ")
                    }
                )
                rows = []
                for index, player_name in enumerate(player_names, start=1):
                    iteration = int(player_name.split("_")[1])
                    rows.append(f"{index:3d} {player_name} : {float(iteration):.1f}")
                ratings_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    _FakeLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
                mock.patch(
                    "codfish.orchestrator.run_aoti_match",
                    side_effect=fake_run_match,
                ),
                mock.patch(
                    "codfish.orchestrator.subprocess.run",
                    side_effect=fake_ordo_run,
                ),
                mock.patch(
                    "codfish.orchestrator.importlib.import_module",
                    side_effect=_wandb_import_side_effect(fake_wandb),
                ),
            ):
                run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=4,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=False),
                    wandb_config=WandbConfig(project="proj", name="eval"),
                    eval_config=_eval_config(
                        snapshot_interval=2,
                        match_games=2,
                        window_offsets=(1,),
                    ),
                )

            match_path = (
                run_root
                / "eval"
                / "matches"
                / "iter_000002_step_000000002__iter_000004_step_000000004.pgn"
            )
            all_games_path = run_root / "eval" / "ordo" / "all_games.pgn"
            ratings_csv_path = run_root / "eval" / "ordo" / "ratings.csv"
            self.assertEqual(
                [call[:2] for call in match_calls],
                [
                    (
                        "iter_000002_step_000000002",
                        "iter_000004_step_000000004",
                    )
                ],
            )
            self.assertTrue(match_path.exists())
            self.assertTrue(ratings_csv_path.exists())
            self.assertEqual(
                all_games_path.read_text(encoding="utf-8"),
                match_path.read_text(encoding="utf-8") + "\n",
            )
            self.assertEqual(len(ordo_calls), 1)
            eval_logs = [
                payload
                for payload in fake_wandb.run.log_calls
                if "eval/ratings_curve" in payload or "eval/ratings_table" in payload
            ]
            self.assertEqual(len(eval_logs), 1)
            self.assertNotIn("iteration", eval_logs[0])
            self.assertNotIn("global_step", eval_logs[0])
            ratings_table = eval_logs[0]["eval/ratings_table"]
            self.assertEqual(
                ratings_table.columns,
                ["snapshot", "iteration", "global_step", "rating"],
            )
            self.assertEqual(
                ratings_table.data,
                [
                    ["iter_000002_step_000000002", 2, 2, 2.0],
                    ["iter_000004_step_000000004", 4, 4, 4.0],
                ],
            )
            ratings_curve = eval_logs[0]["eval/ratings_curve"]
            self.assertIs(ratings_curve.table, ratings_table)
            self.assertEqual(ratings_curve.x, "iteration")
            self.assertEqual(ratings_curve.y, "rating")
            self.assertEqual(ratings_curve.title, "Eval Ratings")

    def test_eval_phase_skips_existing_match_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            existing_match_path = (
                run_root
                / "eval"
                / "matches"
                / "iter_000002_step_000000002__iter_000004_step_000000004.pgn"
            )
            existing_match_path.parent.mkdir(parents=True, exist_ok=True)
            existing_match_path.write_text(
                '[Event "Eval"]\n[White "iter_000002_step_000000002"]\n'
                '[Black "iter_000004_step_000000004"]\n[Result "1/2-1/2"]\n\n1/2-1/2\n',
                encoding="utf-8",
            )

            def fake_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            def fake_ordo_run(
                cmd: list[str],
                *,
                check: bool,
                capture_output: bool,
                text: bool,
            ) -> subprocess.CompletedProcess[str]:
                del check, capture_output, text
                ratings_path = pathlib.Path(cmd[cmd.index("-o") + 1])
                ratings_path.write_text(
                    "  1 iter_000002_step_000000002 : 2.0\n"
                    "  2 iter_000004_step_000000004 : 4.0\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    _FakeLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
                mock.patch(
                    "codfish.orchestrator.run_aoti_match",
                    side_effect=AssertionError(
                        "existing eval match should have been skipped"
                    ),
                ),
                mock.patch(
                    "codfish.orchestrator.subprocess.run",
                    side_effect=fake_ordo_run,
                ),
            ):
                run_selfplay_update_loop(
                    run_root=run_root,
                    num_iterations=4,
                    selfplay_config=_selfplay_config(),
                    model_spec=_toy_model_spec(),
                    trainer_config=_trainer_config(),
                    replay_buffer_config=_replay_config(),
                    runner_config=_runner_config(run_root, resume=False),
                    eval_config=_eval_config(
                        snapshot_interval=2,
                        match_games=2,
                        window_offsets=(1,),
                    ),
                )

            self.assertTrue((run_root / "eval" / "ordo" / "ratings.csv").exists())

    def test_run_selfplay_update_loop_cleans_partial_dir_after_selfplay_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = pathlib.Path(tmp_dir)
            launcher_calls = 0

            def failing_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                nonlocal launcher_calls
                launcher_calls += 1
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )
                raise RuntimeError("injected self-play failure")

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    _FakeLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=failing_run_selfplay,
                ),
                self.assertRaisesRegex(RuntimeError, "injected self-play failure"),
            ):
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
            self.assertFalse(partial_path(iteration_dir).exists())
            self.assertEqual(launcher_calls, 1)

            def successful_run_selfplay(
                launcher: NativeSelfPlayLauncher,
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                nonlocal launcher_calls
                launcher_calls += 1
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    _FakeLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=successful_run_selfplay,
                ),
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
                artifact_dir: pathlib.Path,
                output_dir: pathlib.Path,
            ) -> None:
                nonlocal launcher_calls
                launcher_calls += 1
                _write_chunk_file(
                    output_dir / "games-000001.bin", [_canonical_raw_game()]
                )

            class FailingLearnerRunner(_FakeLearnerRunner):
                def run_iteration(
                    self,
                    historical_chunk_paths: list[str | os.PathLike[str]],
                    new_chunk_paths: list[str | os.PathLike[str]],
                    iteration: int,
                ) -> TrainIterationReport:
                    raise RuntimeError("injected learner failure")

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    FailingLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
                self.assertRaisesRegex(RuntimeError, "injected learner failure"),
            ):
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
            self.assertFalse(partial_path(iteration_dir).exists())
            self.assertEqual(launcher_calls, 1)

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    _FakeLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    side_effect=AssertionError(
                        "existing self-play dir should be reused"
                    ),
                ),
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
                artifact_dir: pathlib.Path,
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

            with (
                mock.patch.object(
                    NativeSelfPlayLauncher, "run_selfplay", new=fake_run_selfplay
                ),
                self._patched_aoti(),
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

            with (
                self._patched_aoti(),
                mock.patch(
                    "codfish.orchestrator.LearnerRunner",
                    RecordingLearnerRunner,
                ),
                mock.patch.object(
                    NativeSelfPlayLauncher,
                    "run_selfplay",
                    new=fake_run_selfplay,
                ),
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

    @unittest.skipUnless(torch.cuda.is_available(), "AOTI self-play requires CUDA")
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
                    "--device",
                    "cuda",
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
            self.assertTrue(
                (run_root / "artifacts" / "iter_000000" / AOTI_MANIFEST_FILE).exists()
            )
            self.assertTrue(
                (run_root / "artifacts" / "iter_000001" / AOTI_MANIFEST_FILE).exists()
            )
            self.assertTrue(
                sorted((run_root / "selfplay" / "iter_000001").glob("*.bin"))
            )
