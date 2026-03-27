from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import torch

from codfish.artifacts import (
    AOTI_ARTIFACT_FORMAT_VERSION,
    AOTI_MANIFEST_FILE,
    AOTI_PACKAGE_FILE,
    AOTI_RUNTIME,
    AOTI_TARGET_DEVICE,
    export_model_to_aoti_artifact,
    read_aoti_artifact_manifest,
)


class ArtifactManifestTest(unittest.TestCase):
    def test_read_aoti_artifact_manifest_round_trips_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = pathlib.Path(tmp_dir) / "iter_000123"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / AOTI_PACKAGE_FILE).write_bytes(b"package")
            (artifact_dir / AOTI_MANIFEST_FILE).write_text(
                json.dumps(
                    {
                        "format_version": AOTI_ARTIFACT_FORMAT_VERSION,
                        "runtime": AOTI_RUNTIME,
                        "target_device": AOTI_TARGET_DEVICE,
                        "package_file": AOTI_PACKAGE_FILE,
                        "model_name": "toy-model",
                        "model_config": {
                            "kind": "toy",
                            "input_channels": 3,
                            "policy_size": 7,
                        },
                        "input_channels": 3,
                        "policy_size": 7,
                        "iteration": 123,
                        "global_learner_step": 456,
                    }
                ),
                encoding="utf-8",
            )

            manifest = read_aoti_artifact_manifest(artifact_dir)

        self.assertEqual(manifest.format_version, AOTI_ARTIFACT_FORMAT_VERSION)
        self.assertEqual(manifest.runtime, AOTI_RUNTIME)
        self.assertEqual(manifest.target_device, AOTI_TARGET_DEVICE)
        self.assertEqual(manifest.package_file, AOTI_PACKAGE_FILE)
        self.assertEqual(manifest.model_name, "toy-model")
        self.assertEqual(manifest.model_config["kind"], "toy")
        self.assertEqual(manifest.input_channels, 3)
        self.assertEqual(manifest.policy_size, 7)
        self.assertEqual(manifest.iteration, 123)
        self.assertEqual(manifest.global_learner_step, 456)

    def test_read_aoti_artifact_manifest_rejects_wrong_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = pathlib.Path(tmp_dir) / "iter_000000"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / AOTI_MANIFEST_FILE).write_text(
                json.dumps(
                    {
                        "format_version": AOTI_ARTIFACT_FORMAT_VERSION,
                        "runtime": "not-aoti",
                        "target_device": AOTI_TARGET_DEVICE,
                        "package_file": AOTI_PACKAGE_FILE,
                        "model_name": "toy-model",
                        "model_config": {
                            "kind": "toy",
                            "input_channels": 3,
                            "policy_size": 7,
                        },
                        "input_channels": 3,
                        "policy_size": 7,
                        "iteration": 0,
                        "global_learner_step": 0,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "runtime must be aoti"):
                read_aoti_artifact_manifest(artifact_dir)


@unittest.skipUnless(torch.cuda.is_available(), "AOTI export requires CUDA")
class ArtifactExportSmokeTest(unittest.TestCase):
    def test_export_model_to_aoti_artifact_creates_manifest_and_package(self) -> None:
        class TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.policy = torch.nn.Linear(4 * 8 * 8, 9)
                self.wdl = torch.nn.Linear(4 * 8 * 8, 3)

            def forward(
                self, inputs: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                flat = inputs.reshape(inputs.shape[0], -1)
                return self.policy(flat), self.wdl(flat)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifact_dir = pathlib.Path(tmp_dir) / "iter_000001"
            path = export_model_to_aoti_artifact(
                model=TinyModel(),
                model_name="tiny-model",
                model_config={
                    "kind": "tiny",
                    "input_channels": 4,
                    "policy_size": 9,
                },
                artifact_dir_path=artifact_dir,
                iteration=1,
                global_learner_step=17,
            )

            manifest = read_aoti_artifact_manifest(path)
            self.assertTrue((artifact_dir / AOTI_PACKAGE_FILE).is_file())
            self.assertEqual(path, artifact_dir)
            self.assertEqual(manifest.model_name, "tiny-model")
            self.assertEqual(manifest.iteration, 1)
            self.assertEqual(manifest.global_learner_step, 17)
