from __future__ import annotations

from dataclasses import asdict
import importlib

from ._checkpoint import _trainer_config_payload
from ._replay import ReplayBufferConfig
from ._types import (
    ModelSpec,
    TrainIterationReport,
    TrainerConfig,
    WandbConfig,
)


class WandbSession:
    def __init__(
        self,
        config: WandbConfig,
        *,
        model_spec: ModelSpec,
        trainer_config: TrainerConfig,
        replay_buffer_config: ReplayBufferConfig,
        resume: bool,
    ) -> None:
        try:
            wandb = importlib.import_module("wandb")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "W&B logging requested but wandb is not installed"
            ) from exc

        self._run = wandb.init(
            project=config.project,
            entity=config.entity,
            name=config.name,
            config={
                "model": {
                    "name": model_spec.name,
                    "config": dict(model_spec.config),
                },
                "trainer": _trainer_config_payload(trainer_config),
                "replay": asdict(replay_buffer_config),
                "runner": {"resume": resume},
            },
        )
        self._run.define_metric("global_step")
        self._run.define_metric("iteration")
        self._run.define_metric("train/*", step_metric="global_step")
        self._closed = False

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
