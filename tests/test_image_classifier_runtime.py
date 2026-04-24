from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.models.runtime import ModelArtifactBundle
from app.services.image_classifier_runtime import TorchvisionImageClassifierRuntime


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.ndim = len(shape)


class _FakeTorch:
    def __init__(self, checkpoint: dict) -> None:
        self._checkpoint = checkpoint

    def load(self, path: Path, map_location: str = "cpu") -> dict:
        del path
        del map_location
        return self._checkpoint


class _MissingDependencyRuntime(TorchvisionImageClassifierRuntime):
    def _import_torch_stack(self):  # type: ignore[override]
        return None


class _AwaitingLabelsRuntime(TorchvisionImageClassifierRuntime):
    def _import_torch_stack(self):  # type: ignore[override]
        checkpoint = {
            "model_state_dict": {
                "classifier.1.weight": _FakeTensor((5, 1280)),
            }
        }
        return SimpleNamespace(
            torch=_FakeTorch(checkpoint),
            torch_nn=object(),
            torchvision_models=object(),
        )


class ImageClassifierRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tempdir.name)
        self.checkpoint_path = self.base_path / "best_model.pt"
        self.checkpoint_path.write_bytes(b"fake-weights")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_runtime_reports_missing_dependency_before_prediction(self) -> None:
        runtime = _MissingDependencyRuntime()
        bundle = ModelArtifactBundle(
            model_key="crop_classifier",
            model_dir=str(self.base_path),
            runtime_ready=True,
            weight_files=[str(self.checkpoint_path)],
            supported_labels=["Tomato", "Potato"],
        )

        prediction = runtime.predict(
            image_path=str(self.checkpoint_path),
            bundle=bundle,
            task_name="crop classification",
        )

        self.assertFalse(prediction.model_ready)
        self.assertEqual(prediction.runtime_stage, "missing_dependency")
        self.assertIn("torch", prediction.note.lower())

    def test_runtime_requires_labels_before_named_prediction(self) -> None:
        runtime = _AwaitingLabelsRuntime()
        bundle = ModelArtifactBundle(
            model_key="crop_classifier",
            model_dir=str(self.base_path),
            runtime_ready=True,
            weight_files=[str(self.checkpoint_path)],
            supported_labels=[],
        )

        prediction = runtime.predict(
            image_path=str(self.checkpoint_path),
            bundle=bundle,
            task_name="crop classification",
        )

        self.assertFalse(prediction.model_ready)
        self.assertEqual(prediction.runtime_stage, "awaiting_labels")
        self.assertIn("5 output classes", prediction.note)

    def test_runtime_prefers_efficientnet_candidates_when_manifest_hints_it(self) -> None:
        runtime = TorchvisionImageClassifierRuntime()

        architectures = runtime._resolve_candidate_architectures(  # noqa: SLF001
            {
                "checkpoint_observations": {
                    "architecture_hint": "torchvision-efficientnet-like classifier backbone",
                }
            }
        )

        self.assertIn("efficientnet_b0", architectures)
        self.assertIn("efficientnet_v2_s", architectures)


if __name__ == "__main__":
    unittest.main()
