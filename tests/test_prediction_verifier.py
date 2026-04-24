from __future__ import annotations

import unittest

from app.services.prediction_verifier import PredictionVerificationService


class PredictionVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = PredictionVerificationService()

    def test_verify_blocks_low_quality_requests(self) -> None:
        result = self.service.verify(
            {
                "image_ok": True,
                "quality_gate_status": "stop",
                "is_leaf": True,
                "is_healthy": False,
            }
        )

        self.assertEqual(result.status, "blocked_quality")
        self.assertFalse(result.verified)
        self.assertIn("quality", result.summary.lower())

    def test_verify_accepts_healthy_screen_branch(self) -> None:
        result = self.service.verify(
            {
                "image_ok": True,
                "quality_gate_status": "proceed",
                "is_leaf": True,
                "is_healthy": True,
            }
        )

        self.assertEqual(result.status, "healthy_screen_skip")
        self.assertTrue(result.verified)
        self.assertEqual(result.issues, [])

    def test_verify_waits_for_runtime_when_predictions_are_placeholders(self) -> None:
        result = self.service.verify(
            {
                "image_ok": True,
                "quality_gate_status": "proceed",
                "is_leaf": True,
                "is_healthy": False,
                "crop_prediction": {
                    "label": "runtime_not_connected",
                    "model_ready": False,
                    "supported_labels": [],
                },
                "disease_route": {"route_found": False, "strategy": "no_route"},
                "disease_prediction": {
                    "label": "model_not_uploaded",
                    "model_ready": False,
                    "supported_labels": [],
                },
                "rag": {"knowledge_base_ready": False, "documents": []},
            }
        )

        self.assertEqual(result.status, "awaiting_runtime")
        self.assertFalse(result.verified)
        self.assertIn("waiting", result.summary.lower())

    def test_verify_marks_aligned_prediction_as_verified(self) -> None:
        result = self.service.verify(
            {
                "image_ok": True,
                "quality_gate_status": "proceed",
                "is_leaf": True,
                "is_healthy": False,
                "crop_prediction": {
                    "label": "Tomato",
                    "model_ready": True,
                    "supported_labels": ["Tomato", "Potato"],
                },
                "disease_route": {
                    "route_found": True,
                    "resolved_model_key": "tomato",
                    "strategy": "metadata_match",
                },
                "disease_prediction": {
                    "label": "Early Blight",
                    "model_ready": True,
                    "supported_labels": ["Early Blight", "Leaf Mold"],
                },
                "rag": {
                    "knowledge_base_ready": True,
                    "documents": [
                        {
                            "title": "Tomato - Early Blight",
                            "metadata": {
                                "crop": "Tomato",
                                "disease": "Early Blight",
                                "aliases": ["Alternaria leaf spot"],
                            },
                        }
                    ],
                },
            }
        )

        self.assertEqual(result.status, "verified")
        self.assertTrue(result.verified)
        self.assertGreaterEqual(result.matched_evidence_count, 1)


if __name__ == "__main__":
    unittest.main()
