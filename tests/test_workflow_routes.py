from __future__ import annotations

import unittest

from app.graph.workflow import (
    route_after_crop_gate,
    route_after_disease_gate,
    route_after_leaf_detection,
    route_after_quality_check,
    route_after_validation,
)


class WorkflowRouteTests(unittest.TestCase):
    def test_invalid_input_routes_to_verification(self) -> None:
        self.assertEqual(
            route_after_validation({"image_ok": False}),
            "verify_prediction",
        )

    def test_quality_stop_routes_to_verification(self) -> None:
        self.assertEqual(
            route_after_quality_check({"image_ok": True, "quality_gate_status": "stop"}),
            "verify_prediction",
        )

    def test_non_leaf_routes_to_verification(self) -> None:
        self.assertEqual(
            route_after_leaf_detection({"is_leaf": False}),
            "verify_prediction",
        )

    def test_healthy_crop_gate_routes_to_verification(self) -> None:
        self.assertEqual(
            route_after_crop_gate({"is_healthy": True}),
            "verify_prediction",
        )

    def test_healthy_crop_gate_uses_bundle_route_when_bundle_prediction_is_ready(self) -> None:
        self.assertEqual(
            route_after_crop_gate(
                {
                    "is_healthy": True,
                    "crop_prediction": {
                        "model_key": "hierarchical_bundle",
                        "model_ready": True,
                    },
                }
            ),
            "route_disease_model",
        )

    def test_crop_gate_routes_when_concrete_crop_label_exists_even_if_gate_would_stop(self) -> None:
        self.assertEqual(
            route_after_crop_gate(
                {
                    "crop_gate_status": "stop",
                    "crop_prediction": {
                        "label": "wheat",
                        "model_ready": True,
                    },
                }
            ),
            "route_disease_model",
        )

    def test_disease_stop_routes_to_verification(self) -> None:
        self.assertEqual(
            route_after_disease_gate({"disease_gate_status": "stop"}),
            "verify_prediction",
        )

    def test_disease_gate_routes_to_rag_when_concrete_disease_label_exists(self) -> None:
        self.assertEqual(
            route_after_disease_gate(
                {
                    "disease_gate_status": "stop",
                    "disease_prediction": {
                        "label": "Wheat blast",
                        "model_ready": True,
                    },
                }
            ),
            "build_rag_query",
        )


if __name__ == "__main__":
    unittest.main()
