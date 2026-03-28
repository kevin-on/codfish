from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

from ._dict_validation import (
    require_dict,
    require_int,
    require_positive_int,
    require_str,
)
from ._run_layout import partial_path
from .learner._checkpoint import load_training_checkpoint
from .learner._types import ModelSpec

AOTI_ARTIFACT_FORMAT_VERSION = 1
AOTI_RUNTIME = "aoti"
AOTI_TARGET_DEVICE = "cuda"
AOTI_PACKAGE_FILE = "model.pt2"
AOTI_MANIFEST_FILE = "manifest.json"


@dataclass(slots=True, frozen=True)
class AotiArtifactManifest:
    format_version: int
    runtime: str
    target_device: str
    package_file: str
    model_name: str
    model_config: dict[str, object]
    input_channels: int
    policy_size: int
    iteration: int
    global_learner_step: int


class _InferenceExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits, wdl_logits = self.model(inputs)
        return policy_logits, torch.softmax(wdl_logits, dim=-1)


def read_aoti_artifact_manifest(
    artifact_dir_path: str | os.PathLike[str],
) -> AotiArtifactManifest:
    artifact_dir = Path(artifact_dir_path)
    manifest_path = artifact_dir / AOTI_MANIFEST_FILE
    with manifest_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("artifact manifest payload must be a dict")

    manifest = AotiArtifactManifest(
        format_version=require_int(
            payload,
            "format_version",
            "artifact manifest",
            distinguish_missing=False,
        ),
        runtime=require_str(
            payload, "runtime", "artifact manifest", distinguish_missing=False
        ),
        target_device=require_str(
            payload, "target_device", "artifact manifest", distinguish_missing=False
        ),
        package_file=require_str(
            payload, "package_file", "artifact manifest", distinguish_missing=False
        ),
        model_name=require_str(
            payload, "model_name", "artifact manifest", distinguish_missing=False
        ),
        model_config=require_dict(
            payload, "model_config", "artifact manifest", distinguish_missing=False
        ),
        input_channels=require_int(
            payload, "input_channels", "artifact manifest", distinguish_missing=False
        ),
        policy_size=require_int(
            payload, "policy_size", "artifact manifest", distinguish_missing=False
        ),
        iteration=require_int(
            payload, "iteration", "artifact manifest", distinguish_missing=False
        ),
        global_learner_step=require_int(
            payload,
            "global_learner_step",
            "artifact manifest",
            distinguish_missing=False,
        ),
    )
    if manifest.format_version != AOTI_ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            "unsupported artifact format_version "
            f"{manifest.format_version}; expected {AOTI_ARTIFACT_FORMAT_VERSION}"
        )
    if manifest.runtime != AOTI_RUNTIME:
        raise ValueError(
            f"artifact runtime must be {AOTI_RUNTIME}, got {manifest.runtime}"
        )
    if manifest.target_device != AOTI_TARGET_DEVICE:
        raise ValueError(
            "artifact target_device must be "
            f"{AOTI_TARGET_DEVICE}, got {manifest.target_device}"
        )
    if manifest.package_file != AOTI_PACKAGE_FILE:
        raise ValueError(
            f"artifact package_file must be {AOTI_PACKAGE_FILE}, got {manifest.package_file}"
        )
    return manifest


def validate_aoti_artifact(
    artifact_dir_path: str | os.PathLike[str],
    *,
    model_spec: ModelSpec,
    expected_iteration: int,
) -> AotiArtifactManifest:
    artifact_dir = Path(artifact_dir_path)
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"artifact dir does not exist: {artifact_dir}")

    manifest = read_aoti_artifact_manifest(artifact_dir)
    if manifest.iteration != expected_iteration:
        raise ValueError(
            "artifact iteration does not match expected iteration: "
            f"{manifest.iteration} != {expected_iteration}"
        )
    if manifest.model_name != model_spec.name:
        raise ValueError("artifact model_name does not match current ModelSpec")
    if manifest.model_config != model_spec.config:
        raise ValueError("artifact model_config does not match current ModelSpec")

    model_package_path = artifact_dir / manifest.package_file
    if not model_package_path.is_file():
        raise FileNotFoundError(
            f"artifact package file does not exist: {model_package_path}"
        )
    return manifest


def export_model_to_aoti_artifact(
    *,
    model: torch.nn.Module,
    model_name: str,
    model_config: dict[str, object],
    artifact_dir_path: str | os.PathLike[str],
    iteration: int,
    global_learner_step: int,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "AOTI export requires CUDA, but torch.cuda.is_available() is false"
        )

    artifact_dir = Path(artifact_dir_path)
    if artifact_dir.exists():
        raise FileExistsError(f"artifact dir already exists: {artifact_dir}")

    partial_dir = partial_path(artifact_dir)
    if partial_dir.exists():
        _remove_path(partial_dir)
    partial_dir.mkdir(parents=True, exist_ok=False)

    was_training = model.training
    try:
        export_model = model.to(device="cuda", dtype=torch.float32)
        export_model.eval()
        wrapper = _InferenceExportWrapper(export_model).to(device="cuda")
        input_channels = require_positive_int(
            model_config, "input_channels", "model_config"
        )
        policy_size = require_positive_int(model_config, "policy_size", "model_config")
        # torch.export can specialize a named dynamic batch dim when the
        # example input uses batch size 1. Export from batch size 2 to keep the
        # leading dimension truly dynamic for inference batching.
        example_input = torch.zeros(
            (2, input_channels, 8, 8),
            dtype=torch.float32,
            device="cuda",
        )
        dynamic_batch = torch.export.Dim("batch", min=2, max=65535)
        with torch.no_grad():
            exported_program = torch.export.export(
                wrapper,
                (example_input,),
                dynamic_shapes={"inputs": {0: dynamic_batch}},
            )
            torch._inductor.aoti_compile_and_package(
                exported_program,
                package_path=os.fspath(partial_dir / AOTI_PACKAGE_FILE),
            )

        manifest = AotiArtifactManifest(
            format_version=AOTI_ARTIFACT_FORMAT_VERSION,
            runtime=AOTI_RUNTIME,
            target_device=AOTI_TARGET_DEVICE,
            package_file=AOTI_PACKAGE_FILE,
            model_name=model_name,
            model_config=dict(model_config),
            input_channels=input_channels,
            policy_size=policy_size,
            iteration=iteration,
            global_learner_step=global_learner_step,
        )
        _write_manifest(partial_dir / AOTI_MANIFEST_FILE, manifest)
        partial_dir.rename(artifact_dir)
    except Exception:
        _remove_path(partial_dir)
        raise
    finally:
        model.train(was_training)

    return artifact_dir


def regenerate_aoti_artifact_from_checkpoint(
    *,
    model_spec: ModelSpec,
    checkpoint_path: str | os.PathLike[str],
    artifact_dir_path: str | os.PathLike[str],
    expected_iteration: int | None = None,
) -> Path:
    checkpoint = load_training_checkpoint(
        checkpoint_path, map_location=torch.device("cpu")
    )
    if expected_iteration is not None and checkpoint.iteration != expected_iteration:
        raise ValueError(
            "checkpoint iteration does not match requested artifact iteration: "
            f"{checkpoint.iteration} != {expected_iteration}"
        )
    if checkpoint.model_name != model_spec.name:
        raise ValueError("checkpoint model_name does not match current ModelSpec")
    if checkpoint.model_config != model_spec.config:
        raise ValueError("checkpoint model_config does not match current ModelSpec")

    model = model_spec.factory()
    model.load_state_dict(checkpoint.model_state_dict)
    return export_model_to_aoti_artifact(
        model=model,
        model_name=checkpoint.model_name,
        model_config=checkpoint.model_config,
        artifact_dir_path=artifact_dir_path,
        iteration=checkpoint.iteration,
        global_learner_step=checkpoint.global_learner_step,
    )


def _write_manifest(path: Path, manifest: AotiArtifactManifest) -> None:
    payload = {
        "format_version": manifest.format_version,
        "runtime": manifest.runtime,
        "target_device": manifest.target_device,
        "package_file": manifest.package_file,
        "model_name": manifest.model_name,
        "model_config": dict(manifest.model_config),
        "input_channels": manifest.input_channels,
        "policy_size": manifest.policy_size,
        "iteration": manifest.iteration,
        "global_learner_step": manifest.global_learner_step,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()
