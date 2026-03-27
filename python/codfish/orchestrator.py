from __future__ import annotations

import argparse
import importlib
from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
import shutil
import sys
from typing import Sequence

import torch

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
from .learner._api import get_model_io_shape, run_mock_selfplay
from .learner._checkpoint import _trainer_config_payload, load_training_checkpoint
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


class NativeSelfPlayLauncher:
    def __init__(self, config: SelfPlayConfig) -> None:
        self.config = config

    def run_selfplay(self, output_dir: str | os.PathLike[str]) -> None:
        output_path = Path(output_dir)
        if output_path.exists():
            raise FileExistsError(f"self-play output dir already exists: {output_path}")
        output_path.mkdir(parents=True, exist_ok=False)
        run_mock_selfplay(
            output_path,
            num_workers=self.config.num_workers,
            num_games=self.config.num_games,
            raw_chunk_max_bytes=self.config.raw_chunk_max_bytes,
            num_action=self.config.num_action,
            num_simulation=self.config.num_simulation,
            c_puct=self.config.c_puct,
            c_visit=self.config.c_visit,
            c_scale=self.config.c_scale,
        )


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
        if run_id is not None:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "must" if resume else "allow"

        self._run = wandb.init(**init_kwargs)
        self._run.define_metric("global_step")
        self._run.define_metric("iteration")
        self._run.define_metric("train/*", step_metric="global_step")
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
) -> list[TrainIterationReport]:
    if num_iterations <= 0:
        raise ValueError("num_iterations must be positive")

    root_path = Path(run_root)
    learner_dir = root_path / "learner"
    selfplay_root = root_path / "selfplay"
    _validate_runner_config(runner_config, learner_dir)

    start_iteration, restored_wandb_run_id = _resume_state(
        learner_dir, runner_config.resume
    )
    launcher = NativeSelfPlayLauncher(selfplay_config)
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
            iteration_dir = _iteration_dir(selfplay_root, iteration)
            _ensure_iteration_selfplay_output(launcher, iteration_dir)
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
                if wandb_session is not None:
                    runner.trainer.wandb_run_id = wandb_session.run_id
                report = runner.run_iteration(
                    historical_chunk_paths,
                    new_chunk_paths,
                    iteration,
                )
            if wandb_session is not None:
                wandb_session.log_iteration(report)
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
        raise ValueError(
            "runner_config.checkpoint_dir must match run_root / 'learner'"
        )


def _resume_state(learner_dir: Path, resume: bool) -> tuple[int, str | None]:
    if not resume:
        return 1, None

    latest_path = learner_dir / "latest.pt"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"resume requested but checkpoint does not exist: {latest_path}"
        )
    checkpoint = load_training_checkpoint(latest_path, map_location=torch.device("cpu"))
    return checkpoint.iteration + 1, checkpoint.wandb_run_id


def _iteration_dir(selfplay_root: Path, iteration: int) -> Path:
    return selfplay_root / f"iter_{iteration:06d}"


def _partial_iteration_dir(iteration_dir: Path) -> Path:
    return iteration_dir.with_name(f"{iteration_dir.name}.partial")


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def _ensure_iteration_selfplay_output(
    launcher: NativeSelfPlayLauncher,
    iteration_dir: Path,
) -> None:
    partial_dir = _partial_iteration_dir(iteration_dir)
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
        launcher.run_selfplay(partial_dir)
    except Exception:
        _remove_path(partial_dir)
        raise
    partial_dir.rename(iteration_dir)


def _discover_iteration_chunk_paths(iteration_dir: Path) -> list[Path]:
    if not iteration_dir.is_dir():
        raise FileNotFoundError(f"missing self-play iteration dir: {iteration_dir}")
    return sorted(
        path for path in iteration_dir.iterdir() if path.is_file() and path.suffix == ".bin"
    )


def _discover_historical_chunk_paths(
    selfplay_root: Path, iteration: int
) -> list[Path]:
    chunk_paths: list[Path] = []
    for historical_iteration in range(1, iteration):
        historical_dir = _iteration_dir(selfplay_root, historical_iteration)
        if not historical_dir.is_dir():
            raise FileNotFoundError(
                f"missing self-play iteration dir: {historical_dir}"
            )
        chunk_paths.extend(_discover_iteration_chunk_paths(historical_dir))
    return chunk_paths


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
        checkpoint_dir=run_root / "learner",
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
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
