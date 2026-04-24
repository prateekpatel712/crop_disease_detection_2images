from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import create_application


class PredictRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(create_application())

    @patch("app.api.routes.invoke_dual_image_pipeline")
    @patch("app.api.routes.UploadStorageService.save", new_callable=AsyncMock)
    def test_predict_accepts_crop_and_diseased_images(
        self,
        mock_save: AsyncMock,
        mock_invoke_dual_image_pipeline,
    ) -> None:
        crop_path = Path("C:/tmp/reference-crop.jpg")
        diseased_path = Path("C:/tmp/diseased-leaf.jpg")
        mock_save.side_effect = [
            (crop_path, 111),
            (diseased_path, 222),
        ]
        mock_invoke_dual_image_pipeline.return_value = {
            "request_id": "req-123",
            "crop_image_path": str(crop_path),
            "diseased_image_path": str(diseased_path),
            "image_path": str(diseased_path),
            "disease_prediction": {"label": "Early blight"},
            "final_answer": "Predicted disease: Early blight. Precautions: remove infected leaves.",
        }

        response = self.client.post(
            "/api/v1/predict",
            files={
                "crop_image": ("crop.jpg", io.BytesIO(b"crop-bytes"), "image/jpeg"),
                "diseased_image": ("disease.jpg", io.BytesIO(b"disease-bytes"), "image/jpeg"),
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "request_id": "req-123",
                "image_path": str(diseased_path),
                "disease_name": "Early blight",
                "final_answer": "Predicted disease: Early blight. Precautions: remove infected leaves.",
            },
        )
        self.assertEqual(mock_save.await_count, 2)
        mock_invoke_dual_image_pipeline.assert_called_once()

        crop_state, diseased_state = mock_invoke_dual_image_pipeline.call_args.args
        self.assertEqual(crop_state["image_path"], str(crop_path))
        self.assertEqual(crop_state["crop_image_path"], str(crop_path))
        self.assertEqual(crop_state["diseased_image_path"], str(diseased_path))
        self.assertEqual(diseased_state["image_path"], str(diseased_path))
        self.assertEqual(diseased_state["crop_image_path"], str(crop_path))
        self.assertEqual(diseased_state["diseased_image_path"], str(diseased_path))

    def test_predict_requires_both_images(self) -> None:
        response = self.client.post(
            "/api/v1/predict",
            files={
                "crop_image": ("crop.jpg", io.BytesIO(b"crop-bytes"), "image/jpeg"),
            },
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
