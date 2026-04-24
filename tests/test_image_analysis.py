from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.core.config import get_settings
from app.services.image_analysis import ImageAnalysisService


class ImageAnalysisServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ImageAnalysisService(get_settings())
        self.tempdir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _save_image(self, name: str, image: Image.Image) -> Path:
        path = self.base_path / name
        image.save(path)
        return path

    def test_dark_low_resolution_image_is_rejected_by_quality_gate(self) -> None:
        image = Image.new("RGB", (120, 120), (10, 10, 10))
        path = self._save_image("dark.png", image)

        result = self.service.analyze(path)

        self.assertFalse(result.image_ok)
        self.assertEqual(result.quality.status, "stop")
        self.assertIn("The image appears too dark for reliable analysis.", result.quality.issues)

    def test_green_leaf_like_image_is_detected_and_screened_as_healthy(self) -> None:
        image = Image.new("RGB", (512, 512), (35, 35, 35))
        draw = ImageDraw.Draw(image)
        draw.ellipse((96, 48, 416, 464), fill=(36, 165, 62))
        path = self._save_image("healthy_leaf.png", image)

        result = self.service.analyze(path)

        self.assertTrue(result.image_ok)
        self.assertEqual(result.quality.status, "proceed")
        self.assertIs(result.leaf.detected, True)
        self.assertIs(result.health.is_healthy, True)

    def test_leaf_with_large_lesion_pattern_is_screened_as_stressed(self) -> None:
        image = Image.new("RGB", (512, 512), (35, 35, 35))
        draw = ImageDraw.Draw(image)
        draw.ellipse((96, 48, 416, 464), fill=(40, 150, 58))
        draw.ellipse((130, 150, 260, 300), fill=(150, 95, 40))
        draw.ellipse((250, 210, 360, 350), fill=(195, 175, 55))
        path = self._save_image("diseased_leaf.png", image)

        result = self.service.analyze(path)

        self.assertTrue(result.image_ok)
        self.assertIs(result.leaf.detected, True)
        self.assertIs(result.health.is_healthy, False)


if __name__ == "__main__":
    unittest.main()
