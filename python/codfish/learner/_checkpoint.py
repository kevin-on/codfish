from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from .._dict_validation import (
    optional_require_dict,
    optional_require_str,
    require_dict,
    require_int,
    require_real,
    require_str,
)
from ._types import TrainerConfig

CHECKPOINT_FORMAT_VERSION = 1


@dataclass(slots=True)
class TrainingCheckpoint:
    format_version: int
    model_state_dict: dict[str, object]
    optimizer_state_dict: dict[str, object]
    global_learner_step: int
    iteration: int
    model_name: str
    model_config: dict[str, object]
    trainer_config: TrainerConfig
    wandb_run_id: str | None
    replay_sampler_rng_state: dict[str, object] | None


@dataclass(slots=True)
class SnapshotCheckpoint:
    format_version: int
    model_state_dict: dict[str, object]
    global_learner_step: int
    iteration: int
    model_name: str
    model_config: dict[str, object]


def build_training_checkpoint_payload(
    *,
    model_state_dict: dict[str, object],
    optimizer_state_dict: dict[str, object],
    global_learner_step: int,
    iteration: int,
    model_name: str,
    model_config: dict[str, object],
    trainer_config: TrainerConfig,
    wandb_run_id: str | None = None,
    replay_sampler_rng_state: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "global_learner_step": global_learner_step,
        "iteration": iteration,
        "model_name": model_name,
        "model_config": dict(model_config),
        "trainer_config": _trainer_config_payload(trainer_config),
        "wandb_run_id": wandb_run_id,
        "replay_sampler_rng_state": (
            dict(replay_sampler_rng_state)
            if replay_sampler_rng_state is not None
            else None
        ),
    }


def build_snapshot_payload(
    *,
    model_state_dict: dict[str, object],
    global_learner_step: int,
    iteration: int,
    model_name: str,
    model_config: dict[str, object],
) -> dict[str, object]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model_state_dict,
        "global_learner_step": global_learner_step,
        "iteration": iteration,
        "model_name": model_name,
        "model_config": dict(model_config),
    }


def load_training_checkpoint(
    path: str | os.PathLike[str], map_location: torch.device
) -> TrainingCheckpoint:
    payload = _load_payload(path, map_location=map_location)
    return _parse_training_checkpoint(payload)


def load_snapshot_checkpoint(
    path: str | os.PathLike[str], map_location: torch.device
) -> SnapshotCheckpoint:
    payload = _load_payload(path, map_location=map_location)
    return _parse_snapshot_checkpoint(payload)


def _load_payload(
    path: str | os.PathLike[str], map_location: torch.device
) -> dict[str, object]:
    payload = torch.load(os.fspath(path), map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dict")

    format_version = payload.get("format_version")
    if not isinstance(format_version, int):
        raise ValueError("checkpoint payload is missing integer format_version")
    if format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "unsupported checkpoint format_version "
            f"{format_version}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
    return payload


def _parse_training_checkpoint(payload: dict[str, object]) -> TrainingCheckpoint:
    return TrainingCheckpoint(
        format_version=require_int(payload, "format_version", "training checkpoint"),
        model_state_dict=require_dict(
            payload, "model_state_dict", "training checkpoint"
        ),
        optimizer_state_dict=require_dict(
            payload, "optimizer_state_dict", "training checkpoint"
        ),
        global_learner_step=require_int(
            payload, "global_learner_step", "training checkpoint"
        ),
        iteration=require_int(payload, "iteration", "training checkpoint"),
        model_name=require_str(payload, "model_name", "training checkpoint"),
        model_config=require_dict(payload, "model_config", "training checkpoint"),
        trainer_config=_parse_trainer_config(
            require_dict(payload, "trainer_config", "training checkpoint")
        ),
        wandb_run_id=optional_require_str(
            payload,
            "wandb_run_id",
            "training checkpoint",
        ),
        replay_sampler_rng_state=optional_require_dict(
            payload,
            "replay_sampler_rng_state",
            "training checkpoint",
        ),
    )


def _parse_snapshot_checkpoint(payload: dict[str, object]) -> SnapshotCheckpoint:
    return SnapshotCheckpoint(
        format_version=require_int(payload, "format_version", "snapshot"),
        model_state_dict=require_dict(payload, "model_state_dict", "snapshot"),
        global_learner_step=require_int(payload, "global_learner_step", "snapshot"),
        iteration=require_int(payload, "iteration", "snapshot"),
        model_name=require_str(payload, "model_name", "snapshot"),
        model_config=require_dict(payload, "model_config", "snapshot"),
    )


def _parse_trainer_config(payload: dict[str, object]) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=require_real(payload, "learning_rate", "trainer_config"),
        optimizer_momentum=require_real(
            payload, "optimizer_momentum", "trainer_config"
        ),
        optimizer_weight_decay=require_real(
            payload, "optimizer_weight_decay", "trainer_config"
        ),
        value_loss_weight=require_real(payload, "value_loss_weight", "trainer_config"),
    )


def _trainer_config_payload(trainer_config: TrainerConfig) -> dict[str, object]:
    return {
        "learning_rate": trainer_config.learning_rate,
        "optimizer_momentum": trainer_config.optimizer_momentum,
        "optimizer_weight_decay": trainer_config.optimizer_weight_decay,
        "value_loss_weight": trainer_config.value_loss_weight,
    }


def atomic_torch_save(
    payload: dict[str, object],
    path: str | os.PathLike[str],
    *,
    previous_path: str | os.PathLike[str] | None = None,
) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        torch.save(payload, tmp_path)
        if previous_path is not None and target_path.exists():
            previous_target = Path(previous_path)
            previous_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_path, previous_target)
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return target_path
