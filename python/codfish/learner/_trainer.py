from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import _run_layout
from ._api import EncodedGameSamples
from ._checkpoint import (
    atomic_torch_save,
    build_snapshot_payload,
    build_training_checkpoint_payload,
    load_training_checkpoint,
)
from ._replay import ReplayBuffer
from ._types import TrainerConfig, TrainIterationReport, TrainStepMetrics


class Trainer:
    def __init__(
        self,
        model: "torch.nn.Module",
        config: TrainerConfig,
        *,
        device: str | torch.device,
        checkpoint_dir: str | os.PathLike[str],
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model.to(self.device)
        self.model_name = type(model).__name__
        self.model_config: dict[str, object] = {}
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.optimizer_momentum,
            weight_decay=config.optimizer_weight_decay,
        )
        self.global_learner_step = 0
        self.last_checkpoint_iteration: int | None = None
        self.wandb_run_id: str | None = None
        self.replay_sampler_rng_state: dict[str, object] | None = None

    def set_model_metadata(
        self,
        *,
        model_name: str,
        model_config: dict[str, object],
    ) -> None:
        self.model_name = model_name
        self.model_config = dict(model_config)

    def train_step(self, batch: EncodedGameSamples) -> TrainStepMetrics:
        inputs = torch.from_numpy(batch.inputs).to(
            device=self.device, dtype=torch.float32
        )
        policy_targets = torch.from_numpy(batch.policy_targets).to(
            device=self.device, dtype=torch.float32
        )
        wdl_targets = torch.from_numpy(batch.wdl_targets).to(
            device=self.device, dtype=torch.float32
        )

        self.model.train()
        policy_logits, wdl_logits = self.model(inputs)

        if tuple(policy_logits.shape) != tuple(policy_targets.shape):
            raise ValueError(
                "model policy logits shape "
                f"{tuple(policy_logits.shape)} does not match targets "
                f"{tuple(policy_targets.shape)}"
            )
        if tuple(wdl_logits.shape) != tuple(wdl_targets.shape):
            raise ValueError(
                "model wdl logits shape "
                f"{tuple(wdl_logits.shape)} does not match targets "
                f"{tuple(wdl_targets.shape)}"
            )

        policy_loss = (
            -(policy_targets * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
        )
        wdl_loss = -(wdl_targets * F.log_softmax(wdl_logits, dim=-1)).sum(dim=-1).mean()
        total_loss = policy_loss + self.config.value_loss_weight * wdl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.global_learner_step += 1

        return TrainStepMetrics(
            total_loss=float(total_loss.detach().cpu().item()),
            policy_loss=float(policy_loss.detach().cpu().item()),
            wdl_loss=float(wdl_loss.detach().cpu().item()),
        )

    def train_iteration(
        self, replay_buffer: ReplayBuffer, new_sample_count: int, iteration: int
    ) -> TrainIterationReport:
        starting_global_step = self.global_learner_step
        num_updates = replay_buffer.compute_update_steps(new_sample_count)
        total_losses: list[float] = []
        policy_losses: list[float] = []
        wdl_losses: list[float] = []

        for _ in range(num_updates):
            metrics = self.train_step(replay_buffer.sample_minibatch())
            total_losses.append(metrics.total_loss)
            policy_losses.append(metrics.policy_loss)
            wdl_losses.append(metrics.wdl_loss)

        latest_checkpoint_path = self.save_checkpoint(
            iteration,
            replay_sampler_rng_state=replay_buffer.rng_state,
        )
        snapshot_path = self.save_snapshot(iteration)

        if total_losses:
            mean_total_loss = sum(total_losses) / len(total_losses)
            mean_policy_loss = sum(policy_losses) / len(policy_losses)
            mean_wdl_loss = sum(wdl_losses) / len(wdl_losses)
        else:
            mean_total_loss = 0.0
            mean_policy_loss = 0.0
            mean_wdl_loss = 0.0

        return TrainIterationReport(
            iteration=iteration,
            starting_global_step=starting_global_step,
            ending_global_step=self.global_learner_step,
            num_updates=num_updates,
            new_sample_count=new_sample_count,
            replay_sample_count=replay_buffer.sample_count,
            mean_total_loss=mean_total_loss,
            mean_policy_loss=mean_policy_loss,
            mean_wdl_loss=mean_wdl_loss,
            latest_checkpoint_path=latest_checkpoint_path,
            snapshot_path=snapshot_path,
        )

    def save_checkpoint(
        self,
        iteration: int,
        *,
        replay_sampler_rng_state: dict[str, object] | None = None,
    ) -> Path:
        latest_path = _run_layout.latest_checkpoint_path(self.checkpoint_dir)
        previous_path = _run_layout.previous_checkpoint_path(self.checkpoint_dir)
        payload = build_training_checkpoint_payload(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            global_learner_step=self.global_learner_step,
            iteration=iteration,
            model_name=self.model_name,
            model_config=self.model_config,
            trainer_config=self.config,
            wandb_run_id=self.wandb_run_id,
            replay_sampler_rng_state=replay_sampler_rng_state,
        )
        atomic_torch_save(payload, latest_path, previous_path=previous_path)
        self.last_checkpoint_iteration = iteration
        self.replay_sampler_rng_state = (
            dict(replay_sampler_rng_state)
            if replay_sampler_rng_state is not None
            else None
        )
        return latest_path

    def save_snapshot(self, iteration: int) -> Path:
        snapshot_path = _run_layout.snapshot_path(
            self.checkpoint_dir,
            iteration=iteration,
            global_step=self.global_learner_step,
        )
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload = build_snapshot_payload(
            model_state_dict=self.model.state_dict(),
            global_learner_step=self.global_learner_step,
            iteration=iteration,
            model_name=self.model_name,
            model_config=self.model_config,
        )
        atomic_torch_save(payload, snapshot_path)
        return snapshot_path

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        checkpoint = load_training_checkpoint(path, map_location=self.device)
        if checkpoint.trainer_config != self.config:
            raise ValueError(
                "checkpoint trainer_config does not match current TrainerConfig"
            )
        self.model.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        self.global_learner_step = checkpoint.global_learner_step
        self.last_checkpoint_iteration = checkpoint.iteration
        self.model_name = checkpoint.model_name
        self.model_config = checkpoint.model_config
        self.wandb_run_id = checkpoint.wandb_run_id
        self.replay_sampler_rng_state = checkpoint.replay_sampler_rng_state
        self.model.to(self.device)
