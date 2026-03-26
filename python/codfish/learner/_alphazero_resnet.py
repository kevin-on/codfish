from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._types import ModelSpec


@dataclass(slots=True, frozen=True)
class SmallAlphaZeroResNetConfig:
    input_channels: int = 118
    policy_size: int = 1858
    trunk_channels: int = 32
    num_blocks: int = 4
    policy_channels: int = 4
    value_channels: int = 8
    value_hidden: int = 64

    def __post_init__(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if self.policy_size <= 0:
            raise ValueError("policy_size must be positive")
        if self.trunk_channels <= 0:
            raise ValueError("trunk_channels must be positive")
        if self.num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if self.policy_channels <= 0:
            raise ValueError("policy_channels must be positive")
        if self.value_channels <= 0:
            raise ValueError("value_channels must be positive")
        if self.value_hidden <= 0:
            raise ValueError("value_hidden must be positive")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class SmallAlphaZeroResNet(nn.Module):
    def __init__(self, config: SmallAlphaZeroResNetConfig) -> None:
        super().__init__()
        self.config = config

        channels = config.trunk_channels
        self.stem_conv = nn.Conv2d(
            config.input_channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(channels)
        self.trunk = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(config.num_blocks)]
        )

        self.policy_conv = nn.Conv2d(
            channels,
            config.policy_channels,
            kernel_size=1,
            bias=False,
        )
        self.policy_bn = nn.BatchNorm2d(config.policy_channels)
        self.policy_fc = nn.Linear(config.policy_channels * 8 * 8, config.policy_size)

        self.value_conv = nn.Conv2d(
            channels,
            config.value_channels,
            kernel_size=1,
            bias=False,
        )
        self.value_bn = nn.BatchNorm2d(config.value_channels)
        self.value_fc1 = nn.Linear(config.value_channels * 8 * 8, config.value_hidden)
        self.value_fc2 = nn.Linear(config.value_hidden, 3)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_shape = (self.config.input_channels, 8, 8)
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected_shape:
            raise ValueError(
                f"expected [B, {self.config.input_channels}, 8, 8], "
                f"got {tuple(inputs.shape)}"
            )

        x = F.relu(self.stem_bn(self.stem_conv(inputs)), inplace=True)
        x = self.trunk(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        policy = torch.flatten(policy, start_dim=1)
        policy_logits = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        value = torch.flatten(value, start_dim=1)
        value = F.relu(self.value_fc1(value), inplace=True)
        wdl_logits = self.value_fc2(value)

        return policy_logits, wdl_logits


def make_small_alphazero_resnet_spec(
    config: SmallAlphaZeroResNetConfig,
) -> ModelSpec:
    return ModelSpec(
        name="small_alphazero_resnet",
        config=asdict(config),
        factory=lambda: SmallAlphaZeroResNet(config),
    )
