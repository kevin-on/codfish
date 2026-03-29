from __future__ import annotations

import argparse
import csv
import importlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import torch

from . import _run_layout
from .artifacts import (
    export_model_to_aoti_artifact,
    read_aoti_artifact_manifest,
    regenerate_aoti_artifact_from_checkpoint,
    validate_aoti_artifact,
)
from .learner import (
    LearnerRunner,
    LearnerRunnerConfig,
    ReplayBufferConfig,
    SmallAlphaZeroResNetConfig,
    TrainerConfig,
    TrainIterationReport,
    WandbConfig,
    make_small_alphazero_resnet_spec,
)
from .learner._api import get_model_io_shape, run_aoti_match, run_aoti_selfplay
from .learner._checkpoint import (
    _trainer_config_payload,
    atomic_torch_save,
    build_snapshot_payload,
    build_training_checkpoint_payload,
    load_snapshot_checkpoint,
    load_training_checkpoint,
)
from .learner._types import ModelSpec


@dataclass(slots=True)
class SelfPlayConfig:
    num_workers: int
    num_games: int
    raw_chunk_max_bytes: int
    num_action: int
    num_simulation: int
    c_puct: float
    c_visit: float
    c_scale: float

    def __post_init__(self) -> None:
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.num_games <= 0:
            raise ValueError("num_games must be positive")
        if self.raw_chunk_max_bytes <= 0:
            raise ValueError("raw_chunk_max_bytes must be positive")
        if self.num_action <= 0:
            raise ValueError("num_action must be positive")
        if self.num_simulation <= 0:
            raise ValueError("num_simulation must be positive")
        if self.c_puct < 0:
            raise ValueError("c_puct must be non-negative")
        if self.c_visit < 0:
            raise ValueError("c_visit must be non-negative")
        if self.c_scale < 0:
            raise ValueError("c_scale must be non-negative")


@dataclass(slots=True)
class EvalConfig:
    snapshot_interval: int
    match_games: int
    num_workers: int
    num_action: int
    num_simulation: int
    c_puct: float
    c_visit: float
    c_scale: float
    window_offsets: tuple[int, ...] = (1, 3, 10)

    def __post_init__(self) -> None:
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        if self.match_games <= 0:
            raise ValueError("match_games must be positive")
        if self.match_games % 2 != 0:
            raise ValueError("match_games must be even")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.num_action <= 0:
            raise ValueError("num_action must be positive")
        if self.num_simulation <= 0:
            raise ValueError("num_simulation must be positive")
        if self.c_puct < 0:
            raise ValueError("c_puct must be non-negative")
        if self.c_visit < 0:
            raise ValueError("c_visit must be non-negative")
        if self.c_scale < 0:
            raise ValueError("c_scale must be non-negative")
        if not self.window_offsets:
            raise ValueError("window_offsets must be non-empty")
        if any(offset <= 0 for offset in self.window_offsets):
            raise ValueError("window_offsets must contain only positive values")
        if len(set(self.window_offsets)) != len(self.window_offsets):
            raise ValueError("window_offsets must be unique")


class NativeSelfPlayLauncher:
    def __init__(self, config: SelfPlayConfig) -> None:
        self.config = config

    def run_selfplay(
        self,
        artifact_dir_path: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
    ) -> None:
        artifact_path = Path(artifact_dir_path)
        manifest = read_aoti_artifact_manifest(artifact_path)
        model_package_path = artifact_path / manifest.package_file
        if not model_package_path.is_file():
            raise FileNotFoundError(
                f"artifact package file does not exist: {model_package_path}"
            )

        output_path = Path(output_dir)
        if output_path.exists():
            raise FileExistsError(f"self-play output dir already exists: {output_path}")
        output_path.mkdir(parents=True, exist_ok=False)
        run_aoti_selfplay(
            model_package_path,
            input_channels=manifest.input_channels,
            policy_size=manifest.policy_size,
            raw_output_dir=output_path,
            num_workers=self.config.num_workers,
            num_games=self.config.num_games,
            raw_chunk_max_bytes=self.config.raw_chunk_max_bytes,
            num_action=self.config.num_action,
            num_simulation=self.config.num_simulation,
            c_puct=self.config.c_puct,
            c_visit=self.config.c_visit,
            c_scale=self.config.c_scale,
        )


class NativeEvalLauncher:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def run_match(
        self,
        artifact_dir_path_a: str | os.PathLike[str],
        artifact_dir_path_b: str | os.PathLike[str],
        *,
        player_name_a: str,
        player_name_b: str,
        output_pgn_path: str | os.PathLike[str],
    ) -> None:
        artifact_path_a = Path(artifact_dir_path_a)
        artifact_path_b = Path(artifact_dir_path_b)
        manifest_a = read_aoti_artifact_manifest(artifact_path_a)
        manifest_b = read_aoti_artifact_manifest(artifact_path_b)
        if manifest_a.input_channels != manifest_b.input_channels:
            raise ValueError("eval artifacts must agree on input_channels")
        if manifest_a.policy_size != manifest_b.policy_size:
            raise ValueError("eval artifacts must agree on policy_size")

        model_package_path_a = artifact_path_a / manifest_a.package_file
        model_package_path_b = artifact_path_b / manifest_b.package_file
        if not model_package_path_a.is_file():
            raise FileNotFoundError(
                f"artifact package file does not exist: {model_package_path_a}"
            )
        if not model_package_path_b.is_file():
            raise FileNotFoundError(
                f"artifact package file does not exist: {model_package_path_b}"
            )

        output_path = Path(output_pgn_path)
        if output_path.exists():
            raise FileExistsError(f"eval output PGN already exists: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        run_aoti_match(
            model_package_path_a,
            model_package_path_b,
            player_name_a=player_name_a,
            player_name_b=player_name_b,
            input_channels=manifest_a.input_channels,
            policy_size=manifest_a.policy_size,
            output_pgn_path=output_path,
            num_workers=self.config.num_workers,
            num_games=self.config.match_games,
            num_action=self.config.num_action,
            num_simulation=self.config.num_simulation,
            c_puct=self.config.c_puct,
            c_visit=self.config.c_visit,
            c_scale=self.config.c_scale,
        )


@dataclass(slots=True, frozen=True)
class _EvalSnapshot:
    path: Path
    stem: str
    iteration: int
    global_step: int


@dataclass(slots=True)
class _EvalRatingsResult:
    current_snapshot: _EvalSnapshot
    ratings_rows: list[dict[str, object]]
    ratings_csv_path: Path


_SNAPSHOT_STEM_RE = re.compile(r"^iter_(\d+)_step_(\d+)$")
_ORDO_RATING_LINE_RE = re.compile(r"^\s*\d+\s+(.+?)\s+:\s+(-?\d+(?:\.\d+)?)\b")


class _OrchestratorWandbSession:
    def __init__(
        self,
        config: WandbConfig,
        *,
        run_root: Path,
        num_iterations: int,
        model_spec: ModelSpec,
        trainer_config: TrainerConfig,
        replay_buffer_config: ReplayBufferConfig,
        runner_config: LearnerRunnerConfig,
        selfplay_config: SelfPlayConfig,
        eval_config: EvalConfig | None,
        run_id: str | None,
        resume: bool,
    ) -> None:
        try:
            wandb = importlib.import_module("wandb")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "W&B logging requested but wandb is not installed"
            ) from exc

        init_kwargs: dict[str, object] = {
            "project": config.project,
            "entity": config.entity,
            "name": config.name,
            "config": {
                "orchestrator": {
                    "run_root": os.fspath(run_root),
                    "num_iterations": num_iterations,
                },
                "model": {
                    "name": model_spec.name,
                    "config": dict(model_spec.config),
                },
                "trainer": _trainer_config_payload(trainer_config),
                "replay": asdict(replay_buffer_config),
                "runner": {
                    "device": str(runner_config.device),
                    "resume": runner_config.resume,
                },
                "selfplay": asdict(selfplay_config),
            },
        }
        if eval_config is not None:
            init_kwargs["config"]["eval"] = asdict(eval_config)
        if run_id is not None:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "must" if resume else "allow"

        self._wandb = wandb
        self._run = wandb.init(**init_kwargs)
        self._run.define_metric("global_step")
        self._run.define_metric("iteration")
        self._run.define_metric("train/*", step_metric="global_step")
        self._run.define_metric("eval/*", step_metric="iteration")
        self._closed = False

    @property
    def run_id(self) -> str:
        run_id = getattr(self._run, "id", None)
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("wandb.init() did not return a run with a valid id")
        return run_id

    def log_iteration(self, report: TrainIterationReport) -> None:
        self._run.log(
            {
                "global_step": report.ending_global_step,
                "iteration": report.iteration,
                "train/total_loss": report.mean_total_loss,
                "train/policy_loss": report.mean_policy_loss,
                "train/wdl_loss": report.mean_wdl_loss,
            }
        )

    def log_eval_ratings(
        self,
        *,
        ratings_rows: list[dict[str, object]],
    ) -> None:
        payload: dict[str, object] = {}

        table_ctor = getattr(self._wandb, "Table", None)
        if callable(table_ctor):
            ratings_table = table_ctor(
                columns=["snapshot", "iteration", "global_step", "rating"],
                data=[
                    [
                        row["snapshot"],
                        row["iteration"],
                        row["global_step"],
                        row["rating"],
                    ]
                    for row in ratings_rows
                ],
            )
            payload["eval/ratings_table"] = ratings_table
            plot_namespace = getattr(self._wandb, "plot", None)
            line_plot = getattr(plot_namespace, "line", None)
            if callable(line_plot):
                payload["eval/ratings_curve"] = line_plot(
                    table=ratings_table,
                    x="iteration",
                    y="rating",
                    title="Eval Ratings",
                )
        else:
            payload["eval/ratings"] = [dict(row) for row in ratings_rows]
        self._run.log(payload)

    def close(self) -> None:
        if self._closed:
            return
        self._run.finish()
        self._closed = True


def run_selfplay_update_loop(
    *,
    run_root: str | os.PathLike[str],
    num_iterations: int,
    selfplay_config: SelfPlayConfig,
    model_spec: ModelSpec,
    trainer_config: TrainerConfig,
    replay_buffer_config: ReplayBufferConfig,
    runner_config: LearnerRunnerConfig,
    wandb_config: WandbConfig | None = None,
    eval_config: EvalConfig | None = None,
) -> list[TrainIterationReport]:
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "AOTI self-play requires CUDA, but torch.cuda.is_available() is false"
        )

    root_path = Path(run_root)
    artifacts_root = _run_layout.artifacts_dir(root_path)
    learner_dir = _run_layout.learner_dir(root_path)
    selfplay_root = _run_layout.selfplay_dir(root_path)
    _validate_runner_config(runner_config, learner_dir)
    if not runner_config.resume:
        _ensure_initial_bootstrap_state(
            artifacts_root=artifacts_root,
            learner_dir=learner_dir,
            model_spec=model_spec,
            trainer_config=trainer_config,
        )

    start_iteration, restored_wandb_run_id = _resume_state(
        learner_dir, runner_config.resume
    )
    launcher = NativeSelfPlayLauncher(selfplay_config)
    eval_launcher = NativeEvalLauncher(eval_config) if eval_config is not None else None
    wandb_session = (
        _OrchestratorWandbSession(
            wandb_config,
            run_root=root_path,
            num_iterations=num_iterations,
            model_spec=model_spec,
            trainer_config=trainer_config,
            replay_buffer_config=replay_buffer_config,
            runner_config=runner_config,
            selfplay_config=selfplay_config,
            eval_config=eval_config,
            run_id=restored_wandb_run_id,
            resume=runner_config.resume,
        )
        if wandb_config is not None
        else None
    )
    reports: list[TrainIterationReport] = []

    try:
        for offset in range(num_iterations):
            iteration = start_iteration + offset
            input_artifact_dir = _ensure_selfplay_input_artifact(
                artifacts_root=artifacts_root,
                learner_dir=learner_dir,
                model_spec=model_spec,
                trainer_config=trainer_config,
                selfplay_iteration=iteration,
            )
            iteration_dir = _run_layout.selfplay_iteration_dir(selfplay_root, iteration)
            _ensure_iteration_selfplay_output(
                launcher, input_artifact_dir, iteration_dir
            )
            new_chunk_paths = _discover_iteration_chunk_paths(iteration_dir)
            if not new_chunk_paths:
                raise RuntimeError(
                    f"self-play run produced no chunk files: {iteration_dir}"
                )
            historical_chunk_paths = _discover_historical_chunk_paths(
                selfplay_root, iteration
            )
            current_runner_config = replace(
                runner_config,
                checkpoint_dir=learner_dir,
                resume=runner_config.resume or offset > 0,
                wandb=None,
            )
            with LearnerRunner(
                model_spec,
                trainer_config,
                replay_buffer_config,
                current_runner_config,
            ) as runner:
                if not runner_config.resume and iteration == 1:
                    _load_runner_from_checkpoint(
                        runner, _run_layout.initial_checkpoint_path(learner_dir)
                    )
                if wandb_session is not None:
                    runner.trainer.wandb_run_id = wandb_session.run_id
                report = runner.run_iteration(
                    historical_chunk_paths,
                    new_chunk_paths,
                    iteration,
                )
                export_model_to_aoti_artifact(
                    model=runner.model,
                    model_name=model_spec.name,
                    model_config=model_spec.config,
                    artifact_dir_path=_run_layout.artifact_iteration_dir(
                        artifacts_root, iteration
                    ),
                    iteration=iteration,
                    global_learner_step=report.ending_global_step,
                )
            eval_result = (
                _run_eval_phase(
                    run_root=root_path,
                    learner_dir=learner_dir,
                    artifacts_root=artifacts_root,
                    report=report,
                    eval_config=eval_config,
                    eval_launcher=eval_launcher,
                )
                if eval_config is not None and eval_launcher is not None
                else None
            )
            if wandb_session is not None:
                wandb_session.log_iteration(report)
                if eval_result is not None:
                    wandb_session.log_eval_ratings(
                        ratings_rows=eval_result.ratings_rows,
                    )
            reports.append(report)
    finally:
        if wandb_session is not None:
            wandb_session.close()

    return reports


def _validate_runner_config(
    runner_config: LearnerRunnerConfig, learner_dir: Path
) -> None:
    if runner_config.wandb is not None:
        raise ValueError("runner_config.wandb must be None; orchestrator owns W&B")
    if Path(runner_config.checkpoint_dir) != learner_dir:
        raise ValueError("runner_config.checkpoint_dir must match run_root / 'learner'")


def _resume_state(learner_dir: Path, resume: bool) -> tuple[int, str | None]:
    if not resume:
        return 1, None

    latest_path = _run_layout.latest_checkpoint_path(learner_dir)
    if not latest_path.exists():
        raise FileNotFoundError(
            f"resume requested but checkpoint does not exist: {latest_path}"
        )
    checkpoint = load_training_checkpoint(latest_path, map_location=torch.device("cpu"))
    return checkpoint.iteration + 1, checkpoint.wandb_run_id


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _ensure_iteration_selfplay_output(
    launcher: NativeSelfPlayLauncher,
    input_artifact_dir: Path,
    iteration_dir: Path,
) -> None:
    partial_dir = _run_layout.partial_path(iteration_dir)
    if iteration_dir.exists():
        if not iteration_dir.is_dir():
            raise FileExistsError(
                f"self-play iteration path already exists and is not a directory: {iteration_dir}"
            )
        if partial_dir.exists():
            _remove_path(partial_dir)
        return

    if partial_dir.exists():
        _remove_path(partial_dir)
    try:
        launcher.run_selfplay(input_artifact_dir, partial_dir)
    except Exception:
        _remove_path(partial_dir)
        raise
    partial_dir.rename(iteration_dir)


def _selfplay_input_artifact_iteration(selfplay_iteration: int) -> int:
    if selfplay_iteration <= 0:
        raise ValueError("selfplay_iteration must be positive")
    return selfplay_iteration - 1


def _ensure_selfplay_input_artifact(
    *,
    artifacts_root: Path,
    learner_dir: Path,
    model_spec: ModelSpec,
    trainer_config: TrainerConfig,
    selfplay_iteration: int,
) -> Path:
    artifact_iteration = _selfplay_input_artifact_iteration(selfplay_iteration)
    input_artifact_dir = _run_layout.artifact_iteration_dir(
        artifacts_root, artifact_iteration
    )
    if input_artifact_dir.exists():
        validate_aoti_artifact(
            input_artifact_dir,
            model_spec=model_spec,
            expected_iteration=artifact_iteration,
        )
        return input_artifact_dir

    if artifact_iteration == 0:
        _ensure_initial_bootstrap_state(
            artifacts_root=artifacts_root,
            learner_dir=learner_dir,
            model_spec=model_spec,
            trainer_config=trainer_config,
        )
        return input_artifact_dir

    latest_checkpoint_path = _run_layout.latest_checkpoint_path(learner_dir)
    if not latest_checkpoint_path.exists():
        raise FileNotFoundError(
            "artifact is missing and learner checkpoint does not exist: "
            f"{latest_checkpoint_path}"
        )
    regenerate_aoti_artifact_from_checkpoint(
        model_spec=model_spec,
        checkpoint_path=latest_checkpoint_path,
        artifact_dir_path=input_artifact_dir,
        expected_iteration=artifact_iteration,
    )
    return input_artifact_dir


def _ensure_initial_bootstrap_state(
    *,
    artifacts_root: Path,
    learner_dir: Path,
    model_spec: ModelSpec,
    trainer_config: TrainerConfig,
) -> Path:
    initial_checkpoint_path = _run_layout.initial_checkpoint_path(learner_dir)
    bootstrap_artifact_dir = _run_layout.artifact_iteration_dir(artifacts_root, 0)

    if bootstrap_artifact_dir.exists():
        validate_aoti_artifact(
            bootstrap_artifact_dir,
            model_spec=model_spec,
            expected_iteration=0,
        )
        if not initial_checkpoint_path.is_file():
            raise FileNotFoundError(
                "bootstrap artifact exists but initial checkpoint is missing: "
                f"{initial_checkpoint_path}"
            )
        _validate_checkpoint_matches_model_spec(
            initial_checkpoint_path,
            model_spec=model_spec,
            expected_iteration=0,
        )
        _ensure_initial_snapshot(
            learner_dir=learner_dir,
            initial_checkpoint_path=initial_checkpoint_path,
            model_spec=model_spec,
        )
        return initial_checkpoint_path

    if initial_checkpoint_path.is_file():
        _validate_checkpoint_matches_model_spec(
            initial_checkpoint_path,
            model_spec=model_spec,
            expected_iteration=0,
        )
        regenerate_aoti_artifact_from_checkpoint(
            model_spec=model_spec,
            checkpoint_path=initial_checkpoint_path,
            artifact_dir_path=bootstrap_artifact_dir,
            expected_iteration=0,
        )
        _ensure_initial_snapshot(
            learner_dir=learner_dir,
            initial_checkpoint_path=initial_checkpoint_path,
            model_spec=model_spec,
        )
        return initial_checkpoint_path

    model = model_spec.factory()
    _write_initial_checkpoint(
        initial_checkpoint_path=initial_checkpoint_path,
        model=model,
        model_spec=model_spec,
        trainer_config=trainer_config,
    )
    export_model_to_aoti_artifact(
        model=model,
        model_name=model_spec.name,
        model_config=model_spec.config,
        artifact_dir_path=bootstrap_artifact_dir,
        iteration=0,
        global_learner_step=0,
    )
    _ensure_initial_snapshot(
        learner_dir=learner_dir,
        initial_checkpoint_path=initial_checkpoint_path,
        model_spec=model_spec,
    )
    return initial_checkpoint_path


def _write_initial_checkpoint(
    *,
    initial_checkpoint_path: Path,
    model: torch.nn.Module,
    model_spec: ModelSpec,
    trainer_config: TrainerConfig,
) -> None:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=trainer_config.learning_rate,
        momentum=trainer_config.optimizer_momentum,
        weight_decay=trainer_config.optimizer_weight_decay,
    )
    payload = build_training_checkpoint_payload(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        global_learner_step=0,
        iteration=0,
        model_name=model_spec.name,
        model_config=model_spec.config,
        trainer_config=trainer_config,
        wandb_run_id=None,
        replay_sampler_rng_state=None,
    )
    atomic_torch_save(payload, initial_checkpoint_path)


def _ensure_initial_snapshot(
    *,
    learner_dir: Path,
    initial_checkpoint_path: Path,
    model_spec: ModelSpec,
) -> Path:
    snapshot_path = _run_layout.snapshot_path(learner_dir, iteration=0, global_step=0)
    if snapshot_path.is_file():
        _validate_snapshot_matches_model_spec(
            snapshot_path,
            model_spec=model_spec,
            expected_iteration=0,
            expected_global_step=0,
        )
        return snapshot_path

    checkpoint = load_training_checkpoint(
        initial_checkpoint_path, map_location=torch.device("cpu")
    )
    if checkpoint.global_learner_step != 0:
        raise ValueError(
            "initial checkpoint global_learner_step does not match expected value: "
            f"{checkpoint.global_learner_step} != 0"
        )
    payload = build_snapshot_payload(
        model_state_dict=checkpoint.model_state_dict,
        global_learner_step=checkpoint.global_learner_step,
        iteration=checkpoint.iteration,
        model_name=checkpoint.model_name,
        model_config=checkpoint.model_config,
    )
    atomic_torch_save(payload, snapshot_path)
    return snapshot_path


def _load_runner_from_checkpoint(
    runner: LearnerRunner,
    checkpoint_path: str | os.PathLike[str],
) -> None:
    trainer = getattr(runner, "trainer", None)
    load_checkpoint = getattr(trainer, "load_checkpoint", None)
    if callable(load_checkpoint):
        load_checkpoint(checkpoint_path)
        return

    checkpoint = load_training_checkpoint(
        checkpoint_path, map_location=torch.device("cpu")
    )
    runner.model.load_state_dict(checkpoint.model_state_dict)


def _validate_checkpoint_matches_model_spec(
    checkpoint_path: str | os.PathLike[str],
    *,
    model_spec: ModelSpec,
    expected_iteration: int,
) -> None:
    checkpoint = load_training_checkpoint(
        checkpoint_path, map_location=torch.device("cpu")
    )
    if checkpoint.iteration != expected_iteration:
        raise ValueError(
            "checkpoint iteration does not match expected iteration: "
            f"{checkpoint.iteration} != {expected_iteration}"
        )
    if checkpoint.model_name != model_spec.name:
        raise ValueError("checkpoint model_name does not match current ModelSpec")
    if checkpoint.model_config != model_spec.config:
        raise ValueError("checkpoint model_config does not match current ModelSpec")


def _validate_snapshot_matches_model_spec(
    checkpoint_path: str | os.PathLike[str],
    *,
    model_spec: ModelSpec,
    expected_iteration: int,
    expected_global_step: int,
) -> None:
    checkpoint = load_snapshot_checkpoint(
        checkpoint_path, map_location=torch.device("cpu")
    )
    if checkpoint.iteration != expected_iteration:
        raise ValueError(
            "snapshot iteration does not match expected iteration: "
            f"{checkpoint.iteration} != {expected_iteration}"
        )
    if checkpoint.global_learner_step != expected_global_step:
        raise ValueError(
            "snapshot global_learner_step does not match expected value: "
            f"{checkpoint.global_learner_step} != {expected_global_step}"
        )
    if checkpoint.model_name != model_spec.name:
        raise ValueError("snapshot model_name does not match current ModelSpec")
    if checkpoint.model_config != model_spec.config:
        raise ValueError("snapshot model_config does not match current ModelSpec")


def _discover_iteration_chunk_paths(iteration_dir: Path) -> list[Path]:
    if not iteration_dir.is_dir():
        raise FileNotFoundError(f"missing self-play iteration dir: {iteration_dir}")
    return sorted(
        path
        for path in iteration_dir.iterdir()
        if path.is_file() and path.suffix == ".bin"
    )


def _discover_historical_chunk_paths(selfplay_root: Path, iteration: int) -> list[Path]:
    chunk_paths: list[Path] = []
    for historical_iteration in range(1, iteration):
        historical_dir = _run_layout.selfplay_iteration_dir(
            selfplay_root, historical_iteration
        )
        if not historical_dir.is_dir():
            raise FileNotFoundError(
                f"missing self-play iteration dir: {historical_dir}"
            )
        chunk_paths.extend(_discover_iteration_chunk_paths(historical_dir))
    return chunk_paths


def _parse_eval_snapshot_path(snapshot_path: Path) -> _EvalSnapshot:
    match = _SNAPSHOT_STEM_RE.fullmatch(snapshot_path.stem)
    if match is None:
        raise ValueError(f"invalid eval snapshot filename: {snapshot_path.name}")
    return _EvalSnapshot(
        path=snapshot_path,
        stem=snapshot_path.stem,
        iteration=int(match.group(1)),
        global_step=int(match.group(2)),
    )


def _discover_eval_snapshots(
    learner_dir: Path, snapshot_interval: int
) -> list[_EvalSnapshot]:
    snapshots_dir = _run_layout.snapshot_dir(learner_dir)
    if not snapshots_dir.exists():
        return []

    snapshots: list[_EvalSnapshot] = []
    for path in sorted(snapshots_dir.glob("*.pt")):
        snapshot = _parse_eval_snapshot_path(path)
        if snapshot.iteration % snapshot_interval != 0:
            continue
        snapshots.append(snapshot)
    snapshots.sort(key=lambda snapshot: snapshot.iteration)
    return snapshots


def _eval_artifact_dir(artifacts_root: Path, snapshot: _EvalSnapshot) -> Path:
    artifact_dir = _run_layout.artifact_iteration_dir(
        artifacts_root, snapshot.iteration
    )
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"missing eval artifact dir: {artifact_dir}")
    return artifact_dir


def _run_eval_phase(
    *,
    run_root: Path,
    learner_dir: Path,
    artifacts_root: Path,
    report: TrainIterationReport,
    eval_config: EvalConfig,
    eval_launcher: NativeEvalLauncher,
) -> _EvalRatingsResult | None:
    if report.iteration % eval_config.snapshot_interval != 0:
        return None
    if not report.snapshot_path.is_file():
        raise FileNotFoundError(
            f"eval snapshot checkpoint does not exist: {report.snapshot_path}"
        )

    snapshots = _discover_eval_snapshots(learner_dir, eval_config.snapshot_interval)
    current_snapshot = _parse_eval_snapshot_path(report.snapshot_path)
    try:
        current_index = next(
            index
            for index, snapshot in enumerate(snapshots)
            if snapshot.stem == current_snapshot.stem
        )
    except StopIteration as exc:
        raise RuntimeError(
            f"current eval snapshot is missing from discovered snapshot set: {current_snapshot.stem}"
        ) from exc
    current_snapshot = snapshots[current_index]

    for offset in eval_config.window_offsets:
        prior_index = current_index - offset
        if prior_index < 0:
            continue
        _ensure_eval_match(
            run_root=run_root,
            artifacts_root=artifacts_root,
            eval_launcher=eval_launcher,
            snapshot_a=current_snapshot,
            snapshot_b=snapshots[prior_index],
        )

    return _rebuild_eval_ratings(
        run_root=run_root,
        learner_dir=learner_dir,
        current_snapshot=current_snapshot,
        snapshot_interval=eval_config.snapshot_interval,
    )


def _ensure_eval_match(
    *,
    run_root: Path,
    artifacts_root: Path,
    eval_launcher: NativeEvalLauncher,
    snapshot_a: _EvalSnapshot,
    snapshot_b: _EvalSnapshot,
) -> None:
    ordered = sorted((snapshot_a, snapshot_b), key=lambda snapshot: snapshot.stem)
    player_a, player_b = ordered
    match_path = _run_layout.eval_match_path(run_root, player_a.stem, player_b.stem)
    if match_path.exists():
        return

    partial_match_path = _run_layout.partial_path(match_path)
    _remove_path(partial_match_path)
    try:
        eval_launcher.run_match(
            _eval_artifact_dir(artifacts_root, player_a),
            _eval_artifact_dir(artifacts_root, player_b),
            player_name_a=player_a.stem,
            player_name_b=player_b.stem,
            output_pgn_path=partial_match_path,
        )
        partial_match_path.rename(match_path)
    except Exception:
        _remove_path(partial_match_path)
        raise


def _rebuild_eval_ratings(
    *,
    run_root: Path,
    learner_dir: Path,
    current_snapshot: _EvalSnapshot,
    snapshot_interval: int,
) -> _EvalRatingsResult | None:
    all_games_pgn = _rebuild_all_games_pgn(run_root)
    if all_games_pgn is None:
        return None

    ratings_text_path = _run_layout.eval_ratings_text_path(run_root)
    _run_ordo(all_games_pgn, ratings_text_path)

    ratings_rows = _write_eval_ratings_csv(
        learner_dir=learner_dir,
        snapshot_interval=snapshot_interval,
        ratings_text_path=ratings_text_path,
        ratings_csv_path=_run_layout.eval_ratings_csv_path(run_root),
    )
    return _EvalRatingsResult(
        current_snapshot=current_snapshot,
        ratings_rows=ratings_rows,
        ratings_csv_path=_run_layout.eval_ratings_csv_path(run_root),
    )


def _rebuild_all_games_pgn(run_root: Path) -> Path | None:
    matches_dir = _run_layout.eval_matches_dir(run_root)
    match_paths = sorted(path for path in matches_dir.glob("*.pgn") if path.is_file())
    if not match_paths:
        return None

    output_path = _run_layout.eval_all_games_pgn_path(run_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output_path = _run_layout.partial_path(output_path)
    _remove_path(partial_output_path)
    try:
        with partial_output_path.open("wb") as stream:
            for match_path in match_paths:
                payload = match_path.read_bytes()
                stream.write(payload)
                if payload and not payload.endswith(b"\n"):
                    stream.write(b"\n")
                stream.write(b"\n")
        partial_output_path.rename(output_path)
    except Exception:
        _remove_path(partial_output_path)
        raise
    return output_path


def _run_ordo(all_games_pgn_path: Path, ratings_text_path: Path) -> None:
    ratings_text_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ordo",
                "-a",
                "0",
                "-p",
                os.fspath(all_games_pgn_path),
                "-o",
                os.fspath(ratings_text_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ordo executable not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown error"
        raise RuntimeError(f"ordo failed: {stderr}") from exc


def _write_eval_ratings_csv(
    *,
    learner_dir: Path,
    snapshot_interval: int,
    ratings_text_path: Path,
    ratings_csv_path: Path,
) -> list[dict[str, object]]:
    snapshots_by_stem = {
        snapshot.stem: snapshot
        for snapshot in _discover_eval_snapshots(learner_dir, snapshot_interval)
    }
    ratings_rows: list[dict[str, object]] = []
    for line in ratings_text_path.read_text(encoding="utf-8").splitlines():
        match = _ORDO_RATING_LINE_RE.match(line)
        if match is None:
            continue
        snapshot = snapshots_by_stem.get(match.group(1).strip())
        if snapshot is None:
            continue
        ratings_rows.append(
            {
                "snapshot": snapshot.stem,
                "iteration": snapshot.iteration,
                "global_step": snapshot.global_step,
                "rating": float(match.group(2)),
            }
        )

    ratings_rows.sort(key=lambda row: (int(row["iteration"]), str(row["snapshot"])))
    ratings_csv_path.parent.mkdir(parents=True, exist_ok=True)
    partial_csv_path = _run_layout.partial_path(ratings_csv_path)
    _remove_path(partial_csv_path)
    try:
        with partial_csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["snapshot", "iteration", "global_step", "rating"],
            )
            writer.writeheader()
            for row in ratings_rows:
                writer.writerow(row)
        partial_csv_path.rename(ratings_csv_path)
    except Exception:
        _remove_path(partial_csv_path)
        raise
    return ratings_rows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run codfish self-play/update loop")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--num-iterations", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--optimizer-momentum", required=True, type=float)
    parser.add_argument("--optimizer-weight-decay", required=True, type=float)
    parser.add_argument("--value-loss-weight", required=True, type=float)

    parser.add_argument("--sample-capacity", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--replay-ratio", required=True, type=float)
    parser.add_argument("--replay-seed", default=0, type=int)

    parser.add_argument("--trunk-channels", default=32, type=int)
    parser.add_argument("--num-blocks", default=4, type=int)
    parser.add_argument("--policy-channels", default=4, type=int)
    parser.add_argument("--value-channels", default=8, type=int)
    parser.add_argument("--value-hidden", default=64, type=int)

    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--num-games", default=1, type=int)
    parser.add_argument("--raw-chunk-max-bytes", default=128 * 1024 * 1024, type=int)
    parser.add_argument("--num-action", default=8, type=int)
    parser.add_argument("--num-simulation", default=16, type=int)
    parser.add_argument("--c-puct", default=1.0, type=float)
    parser.add_argument("--c-visit", default=1.0, type=float)
    parser.add_argument("--c-scale", default=1.0, type=float)

    parser.add_argument("--eval-snapshot-interval", default=0, type=int)
    parser.add_argument("--eval-match-games", default=200, type=int)
    parser.add_argument("--eval-num-workers", default=1, type=int)
    parser.add_argument("--eval-num-action", default=8, type=int)
    parser.add_argument("--eval-num-simulation", default=16, type=int)
    parser.add_argument("--eval-c-puct", default=1.0, type=float)
    parser.add_argument("--eval-c-visit", default=1.0, type=float)
    parser.add_argument("--eval-c-scale", default=1.0, type=float)
    parser.add_argument("--eval-window-offset", nargs="+", type=int, default=[1, 3, 10])

    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_root = Path(args.run_root)
    shape = get_model_io_shape()
    model_spec = make_small_alphazero_resnet_spec(
        SmallAlphaZeroResNetConfig(
            input_channels=shape.input_channels,
            policy_size=shape.policy_size,
            trunk_channels=args.trunk_channels,
            num_blocks=args.num_blocks,
            policy_channels=args.policy_channels,
            value_channels=args.value_channels,
            value_hidden=args.value_hidden,
        )
    )
    trainer_config = TrainerConfig(
        learning_rate=args.learning_rate,
        optimizer_momentum=args.optimizer_momentum,
        optimizer_weight_decay=args.optimizer_weight_decay,
        value_loss_weight=args.value_loss_weight,
    )
    replay_buffer_config = ReplayBufferConfig(
        sample_capacity=args.sample_capacity,
        batch_size=args.batch_size,
        replay_ratio=args.replay_ratio,
        seed=args.replay_seed,
    )
    runner_config = LearnerRunnerConfig(
        device=args.device,
        checkpoint_dir=_run_layout.learner_dir(run_root),
        resume=args.resume,
        wandb=None,
    )
    selfplay_config = SelfPlayConfig(
        num_workers=args.num_workers,
        num_games=args.num_games,
        raw_chunk_max_bytes=args.raw_chunk_max_bytes,
        num_action=args.num_action,
        num_simulation=args.num_simulation,
        c_puct=args.c_puct,
        c_visit=args.c_visit,
        c_scale=args.c_scale,
    )
    wandb_config = (
        WandbConfig(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
        )
        if args.wandb_project is not None
        else None
    )
    eval_config = (
        EvalConfig(
            snapshot_interval=args.eval_snapshot_interval,
            match_games=args.eval_match_games,
            num_workers=args.eval_num_workers,
            num_action=args.eval_num_action,
            num_simulation=args.eval_num_simulation,
            c_puct=args.eval_c_puct,
            c_visit=args.eval_c_visit,
            c_scale=args.eval_c_scale,
            window_offsets=tuple(args.eval_window_offset),
        )
        if args.eval_snapshot_interval > 0
        else None
    )

    try:
        run_selfplay_update_loop(
            run_root=run_root,
            num_iterations=args.num_iterations,
            selfplay_config=selfplay_config,
            model_spec=model_spec,
            trainer_config=trainer_config,
            replay_buffer_config=replay_buffer_config,
            runner_config=runner_config,
            wandb_config=wandb_config,
            eval_config=eval_config,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
