from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.models.runtime import PredictionCandidate, ServicePrediction
from app.services.crop_predictor import CropPredictorService
from app.services.disease_predictor import DiseasePredictorService
from app.services.disease_router import DiseaseModelRouter
from app.services.image_classifier_runtime import ImageClassifierRuntime
from app.services.model_registry import ModelRegistry


class FakeRuntime(ImageClassifierRuntime):
    def predict(
        self,
        image_path: str,
        bundle,
        *,
        task_name: str,
    ) -> ServicePrediction:
        del image_path
        del task_name
        top_k = [
            PredictionCandidate(label=bundle.supported_labels[0], confidence=0.91),
            PredictionCandidate(label=bundle.supported_labels[-1], confidence=0.09),
        ]
        return ServicePrediction(
            label=bundle.supported_labels[0],
            confidence=0.91,
            top_k=top_k,
            model_ready=True,
            model_key=bundle.model_key,
            artifact_count=len(bundle.weight_files),
            supported_labels=bundle.supported_labels,
            provider="test-runtime",
            runtime_stage="predicted",
            note="Predicted by the fake runtime.",
        )


class PredictionInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tempdir.name)
        self.crop_dir = self.base_path / "models" / "crop"
        self.disease_dir = self.base_path / "models" / "disease"
        self.metadata_dir = self.base_path / "models" / "metadata"
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.disease_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.registry = ModelRegistry(
            crop_model_dir=self.crop_dir,
            disease_model_dir=self.disease_dir,
            metadata_dir=self.metadata_dir,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_crop_bundle_detects_weights_and_labels(self) -> None:
        (self.crop_dir / "model.onnx").write_bytes(b"fake-weights")
        (self.crop_dir / "labels.txt").write_text(
            "Tomato\nPotato\nRice\n",
            encoding="utf-8",
        )

        bundle = self.registry.crop_bundle()

        self.assertTrue(bundle.runtime_ready)
        self.assertEqual(bundle.model_key, "crop_classifier")
        self.assertEqual(bundle.supported_labels, ["Tomato", "Potato", "Rice"])

    def test_disease_router_prefers_metadata_route_when_available(self) -> None:
        tomato_dir = self.disease_dir / "tomato"
        tomato_dir.mkdir(parents=True, exist_ok=True)
        (tomato_dir / "model.pt").write_bytes(b"weights")
        (self.metadata_dir / "disease_routes.json").write_text(
            json.dumps(
                {
                    "default_model_key": "generic",
                    "routes": [
                        {
                            "crop_name": "Tomato",
                            "aliases": ["tomato leaf"],
                            "model_key": "tomato",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        router = DiseaseModelRouter(self.registry, self.metadata_dir)
        route = router.resolve("Tomato")

        self.assertTrue(route.route_found)
        self.assertEqual(route.resolved_model_key, "tomato")
        self.assertEqual(route.strategy, "metadata_match")

    def test_disease_router_falls_back_to_generic_model(self) -> None:
        (self.disease_dir / "model.pt").write_bytes(b"generic-weights")

        router = DiseaseModelRouter(self.registry, self.metadata_dir)
        route = router.resolve("Unknown Crop")

        self.assertTrue(route.route_found)
        self.assertEqual(route.resolved_model_key, "generic")
        self.assertEqual(route.strategy, "generic_fallback")

    def test_prediction_interfaces_report_runtime_metadata(self) -> None:
        tomato_dir = self.disease_dir / "tomato"
        tomato_dir.mkdir(parents=True, exist_ok=True)
        (tomato_dir / "model.pt").write_bytes(b"weights")
        (tomato_dir / "labels.json").write_text(
            json.dumps(["early_blight", "late_blight"]),
            encoding="utf-8",
        )
        (self.crop_dir / "crop_classifier.pt").write_bytes(b"weights")
        (self.crop_dir / "labels.txt").write_text(
            "Tomato\nPotato\n",
            encoding="utf-8",
        )

        runtime = FakeRuntime()
        crop_service = CropPredictorService(self.registry, runtime=runtime)
        disease_service = DiseasePredictorService(self.registry, runtime=runtime)

        crop_prediction = crop_service.predict("sample.png")
        disease_prediction = disease_service.predict(
            "sample.png",
            crop_name="Tomato",
            model_key="tomato",
        )

        self.assertTrue(crop_prediction.model_ready)
        self.assertEqual(crop_prediction.model_key, "crop_classifier")
        self.assertEqual(crop_prediction.artifact_count, 1)
        self.assertEqual(crop_prediction.runtime_stage, "predicted")
        self.assertEqual(crop_prediction.provider, "test-runtime")

        self.assertTrue(disease_prediction.model_ready)
        self.assertEqual(disease_prediction.model_key, "tomato")
        self.assertEqual(disease_prediction.supported_labels, ["early_blight", "late_blight"])
        self.assertEqual(disease_prediction.runtime_stage, "predicted")


if __name__ == "__main__":
    unittest.main()
