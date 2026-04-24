from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.services.hierarchical_bundle_runtime import HierarchicalBundleRuntime


class _PredictingBundleRuntime(HierarchicalBundleRuntime):
    def _load_stack(self):  # type: ignore[override]
        return object()

    def _extract_features(self, image_path, artifacts, stack):  # type: ignore[override]
        del image_path
        del artifacts
        del stack
        return np.zeros((1, 4), dtype=np.float32)

    def _predict_final_probabilities(self, features, artifacts):  # type: ignore[override]
        del features
        del artifacts
        return np.asarray([0.10, 0.25, 0.65], dtype=np.float64)


class _CrossCropBundleRuntime(HierarchicalBundleRuntime):
    def _load_stack(self):  # type: ignore[override]
        return object()

    def _extract_features(self, image_path, artifacts, stack):  # type: ignore[override]
        del image_path
        del artifacts
        del stack
        return np.zeros((1, 4), dtype=np.float32)

    def _predict_final_probabilities(self, features, artifacts):  # type: ignore[override]
        del features
        del artifacts
        return np.asarray([0.05, 0.10, 0.85], dtype=np.float64)


class _MissingDependencyBundleRuntime(HierarchicalBundleRuntime):
    def _load_stack(self):  # type: ignore[override]
        return None


class HierarchicalBundleRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tempdir.name)
        self.bundles_dir = self.base_path / "models" / "bundles"
        self.bundle_root = self.bundles_dir / "crop_disease_detection_model_bundle_20260422"
        self.metadata_dir = self.bundle_root / "metadata"
        self.models_dir = self.bundle_root / "models"
        self.per_crop_heads_dir = self.models_dir / "per_crop_heads"
        self.per_crop_heads_dir.mkdir(parents=True, exist_ok=True)
        self._write_bundle_files()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_artifacts_are_discovered_from_bundle_manifest(self) -> None:
        runtime = HierarchicalBundleRuntime(self.bundles_dir)

        self.assertTrue(runtime.artifacts_detected())
        self.assertEqual(runtime.available_model_keys(), ["hierarchical_bundle"])
        route = runtime.route_payload("tomato")
        self.assertTrue(route["route_found"])
        self.assertEqual(route["resolved_model_key"], "hierarchical_bundle")

    def test_missing_dependency_returns_runtime_blocked_predictions(self) -> None:
        runtime = _MissingDependencyBundleRuntime(self.bundles_dir)

        crop_prediction = runtime.predict_crop("example.png")
        disease_prediction = runtime.predict_disease("example.png")

        self.assertFalse(crop_prediction.model_ready)
        self.assertEqual(crop_prediction.runtime_stage, "missing_dependency")
        self.assertIn("torch", crop_prediction.note.lower())
        self.assertFalse(disease_prediction.model_ready)
        self.assertEqual(disease_prediction.runtime_stage, "missing_dependency")

    def test_prediction_bundle_maps_final_class_to_crop_and_disease_outputs(self) -> None:
        runtime = _PredictingBundleRuntime(self.bundles_dir)

        crop_prediction = runtime.predict_crop("example.png")
        disease_prediction = runtime.predict_disease("example.png")

        self.assertTrue(crop_prediction.model_ready)
        self.assertEqual(crop_prediction.label, "tomato")
        self.assertAlmostEqual(crop_prediction.confidence, 0.65)
        self.assertEqual(disease_prediction.label, "early_blight")
        self.assertAlmostEqual(disease_prediction.confidence, 1.0)
        self.assertEqual(disease_prediction.top_k[0].label, "early_blight")

    def test_predict_disease_can_be_constrained_to_selected_crop_head(self) -> None:
        runtime = _CrossCropBundleRuntime(self.bundles_dir)

        unconstrained_prediction = runtime.predict_disease("example.png")
        constrained_prediction = runtime.predict_disease("example.png", crop_name="maize")

        self.assertEqual(unconstrained_prediction.label, "early_blight")
        self.assertEqual(constrained_prediction.label, "leaf_blight")
        self.assertAlmostEqual(constrained_prediction.confidence, 2.0 / 3.0)
        self.assertIn("selected crop head", constrained_prediction.note.lower())

    def _write_bundle_files(self) -> None:
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        (self.models_dir / "best_model.pt").write_bytes(b"checkpoint")
        (self.models_dir / "crop_router_refinement.joblib").write_bytes(b"router")
        (self.per_crop_heads_dir / "maize_disease_head.joblib").write_bytes(b"head")
        (self.per_crop_heads_dir / "tomato_disease_head.joblib").write_bytes(b"head")

        manifest = {
            "bundle_name": "crop_disease_detection_model_bundle_20260422",
            "artifacts": {
                "backbone_checkpoint": "models/best_model.pt",
                "crop_router": "models/crop_router_refinement.joblib",
                "per_crop_heads_dir": "models/per_crop_heads",
                "id_to_label": "metadata/id_to_label.json",
                "label_to_id": "metadata/label_to_id.json",
                "crop_order": "metadata/crop_order.json",
                "preprocessing_config": "metadata/preprocessing_config.json",
                "pipeline_summary": "metadata/pipeline_summary.json",
                "crop_router_refinement_summary": "metadata/crop_router_refinement_summary.json",
            },
            "selected_inference_settings": {
                "crop_scale": 0.4,
                "blend_weight": 0.0,
            },
        }
        (self.metadata_dir / "bundle_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        (self.metadata_dir / "id_to_label.json").write_text(
            json.dumps(
                {
                    "0": "maize::healthy",
                    "1": "maize::leaf_blight",
                    "2": "tomato::early_blight",
                }
            ),
            encoding="utf-8",
        )
        (self.metadata_dir / "label_to_id.json").write_text(
            json.dumps(
                {
                    "maize::healthy": 0,
                    "maize::leaf_blight": 1,
                    "tomato::early_blight": 2,
                }
            ),
            encoding="utf-8",
        )
        (self.metadata_dir / "crop_order.json").write_text(
            json.dumps(["maize", "tomato"]),
            encoding="utf-8",
        )
        (self.metadata_dir / "preprocessing_config.json").write_text(
            json.dumps(
                {
                    "image_size": 224,
                    "resize_shorter_side": 255,
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                }
            ),
            encoding="utf-8",
        )
        (self.metadata_dir / "pipeline_summary.json").write_text("{}", encoding="utf-8")
        (self.metadata_dir / "crop_router_refinement_summary.json").write_text(
            "{}",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
