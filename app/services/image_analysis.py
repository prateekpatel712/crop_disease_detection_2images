from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import Settings
from app.models.runtime import (
    HealthAssessment,
    ImageAnalysisResult,
    ImageMetadata,
    ImageQualityAssessment,
    LeafAssessment,
)


class ImageAnalysisService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def analyze(self, image_path: str | Path) -> ImageAnalysisResult:
        path = Path(image_path)

        try:
            with Image.open(path) as raw_image:
                image = ImageOps.exif_transpose(raw_image).convert("RGB")
                image_format = raw_image.format or ""
                original_mode = raw_image.mode or ""
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return ImageAnalysisResult(
                image_ok=False,
                metadata=ImageMetadata(),
                quality=ImageQualityAssessment(
                    status="stop",
                    score=0.0,
                    issues=["The uploaded file could not be decoded as a valid image."],
                ),
                leaf=LeafAssessment(
                    detected=None,
                    note="Leaf detection skipped because the image could not be opened.",
                ),
                health=HealthAssessment(
                    is_healthy=None,
                    note="Health screening skipped because the image could not be opened.",
                ),
                warnings=[f"Image decode failure: {exc.__class__.__name__}"],
            )

        rgb = np.asarray(image, dtype=np.float32)
        grayscale = self._to_grayscale(rgb)
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32)

        metadata = ImageMetadata(
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            format=image_format,
            mode=original_mode,
        )

        quality = self._assess_quality(rgb, grayscale, metadata)
        leaf = self._detect_leaf(hsv, metadata, quality)
        health = self._screen_health(hsv, leaf)

        warnings = list(dict.fromkeys([*quality.issues, leaf.note, health.note]))
        warnings = [warning for warning in warnings if warning]

        return ImageAnalysisResult(
            image_ok=quality.status != "stop",
            metadata=metadata,
            quality=quality,
            leaf=leaf,
            health=health,
            warnings=warnings,
        )

    @staticmethod
    def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
        return (
            0.299 * rgb[..., 0]
            + 0.587 * rgb[..., 1]
            + 0.114 * rgb[..., 2]
        ).astype(np.float32)

    @staticmethod
    def _variance_of_laplacian(grayscale: np.ndarray) -> float:
        center = grayscale[1:-1, 1:-1]
        up = grayscale[:-2, 1:-1]
        down = grayscale[2:, 1:-1]
        left = grayscale[1:-1, :-2]
        right = grayscale[1:-1, 2:]
        laplacian = (-4.0 * center) + up + down + left + right
        return float(np.var(laplacian)) if laplacian.size else 0.0

    @staticmethod
    def _colorfulness(rgb: np.ndarray) -> float:
        rg = np.abs(rgb[..., 0] - rgb[..., 1])
        yb = np.abs(0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2])
        std_root = np.sqrt(np.var(rg) + np.var(yb))
        mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        return float(std_root + (0.3 * mean_root))

    def _assess_quality(
        self,
        rgb: np.ndarray,
        grayscale: np.ndarray,
        metadata: ImageMetadata,
    ) -> ImageQualityAssessment:
        brightness_mean = float(np.mean(grayscale))
        contrast_std = float(np.std(grayscale))
        sharpness_score = self._variance_of_laplacian(grayscale)
        colorfulness = self._colorfulness(rgb)

        issues: list[str] = []
        critical_issues = 0

        if metadata.width < 256 or metadata.height < 256:
            issues.append("The image resolution is low. Use at least a 256x256 crop leaf image.")
            critical_issues += 1

        if brightness_mean < 45.0:
            issues.append("The image appears too dark for reliable analysis.")
            critical_issues += 1
        elif brightness_mean > 225.0:
            issues.append("The image appears overexposed.")

        if contrast_std < 22.0:
            issues.append("The image has low contrast.")

        if sharpness_score < 25.0:
            issues.append("The image appears blurry.")
            critical_issues += 1

        if colorfulness < 12.0:
            issues.append("The image has very weak color separation, which may reduce leaf detection quality.")

        resolution_score = min(1.0, min(metadata.width, metadata.height) / 512.0)
        brightness_score = max(0.0, 1.0 - abs(brightness_mean - 128.0) / 128.0)
        contrast_score = min(1.0, contrast_std / 64.0)
        sharpness_norm = min(1.0, np.log1p(max(sharpness_score, 0.0)) / 6.0)
        colorfulness_score = min(1.0, colorfulness / 80.0)

        score = float(
            np.clip(
                (0.30 * resolution_score)
                + (0.20 * brightness_score)
                + (0.20 * contrast_score)
                + (0.20 * sharpness_norm)
                + (0.10 * colorfulness_score),
                0.0,
                1.0,
            )
        )

        if critical_issues >= 2 or score < 0.35:
            status = "stop"
        elif issues or score < 0.60:
            status = "caution"
        else:
            status = "proceed"

        return ImageQualityAssessment(
            status=status,
            score=round(score, 4),
            issues=issues,
            brightness_mean=round(brightness_mean, 2),
            contrast_std=round(contrast_std, 2),
            sharpness_score=round(sharpness_score, 2),
            colorfulness=round(colorfulness, 2),
        )

    def _detect_leaf(
        self,
        hsv: np.ndarray,
        metadata: ImageMetadata,
        quality: ImageQualityAssessment,
    ) -> LeafAssessment:
        hue = hsv[..., 0] * (360.0 / 255.0)
        sat = hsv[..., 1]
        val = hsv[..., 2]

        green_mask = (
            (hue >= 45.0)
            & (hue <= 170.0)
            & (sat >= 35.0)
            & (val >= 25.0)
        )

        center_slice_y = slice(metadata.height // 4, max(metadata.height * 3 // 4, 1))
        center_slice_x = slice(metadata.width // 4, max(metadata.width * 3 // 4, 1))
        center_mask = green_mask[center_slice_y, center_slice_x]

        green_ratio = float(np.mean(green_mask)) if green_mask.size else 0.0
        center_green_ratio = float(np.mean(center_mask)) if center_mask.size else 0.0
        mean_saturation = float(np.mean(sat))

        green_score = min(1.0, green_ratio / 0.22)
        center_score = min(1.0, center_green_ratio / 0.18)
        saturation_score = min(1.0, mean_saturation / 120.0)
        confidence = float(
            np.clip(
                (0.50 * green_score) + (0.35 * center_score) + (0.15 * saturation_score),
                0.0,
                1.0,
            )
        )

        note = ""
        detected: bool | None

        if quality.status == "stop":
            detected = None
            note = "Leaf detection skipped because the image quality is too weak."
        elif green_ratio >= 0.10 and center_green_ratio >= 0.08 and confidence >= 0.45:
            detected = True
            note = "Leaf presence was inferred using color and saturation heuristics."
        elif green_ratio <= 0.03 and center_green_ratio <= 0.02:
            detected = False
            note = "The image does not show enough leaf-like green structure."
        else:
            detected = None
            note = "Leaf detection is uncertain. A tighter crop around a single leaf would help."

        return LeafAssessment(
            detected=detected,
            confidence=round(confidence, 4),
            green_ratio=round(green_ratio, 4),
            center_green_ratio=round(center_green_ratio, 4),
            mean_saturation=round(mean_saturation, 2),
            note=note,
        )

    def _screen_health(
        self,
        hsv: np.ndarray,
        leaf: LeafAssessment,
    ) -> HealthAssessment:
        if leaf.detected is not True:
            return HealthAssessment(
                is_healthy=None,
                confidence=0.0,
                lesion_ratio=0.0,
                note="Health screening skipped because the leaf signal is not strong enough.",
            )

        hue = hsv[..., 0] * (360.0 / 255.0)
        sat = hsv[..., 1]
        val = hsv[..., 2]

        green_mask = (
            (hue >= 45.0)
            & (hue <= 170.0)
            & (sat >= 35.0)
            & (val >= 25.0)
        )
        yellow_mask = (
            (hue >= 20.0)
            & (hue < 50.0)
            & (sat >= 40.0)
            & (val >= 45.0)
        )
        brown_mask = (
            (hue >= 5.0)
            & (hue < 30.0)
            & (sat >= 30.0)
            & (val >= 20.0)
            & (val <= 190.0)
        )

        leaf_like_mask = green_mask | yellow_mask | brown_mask
        lesion_mask = yellow_mask | brown_mask
        leaf_pixels = int(np.count_nonzero(leaf_like_mask))

        if leaf_pixels == 0:
            return HealthAssessment(
                is_healthy=None,
                confidence=0.0,
                lesion_ratio=0.0,
                note="Health screening is uncertain because no leaf-like region was isolated.",
            )

        lesion_ratio = float(np.count_nonzero(lesion_mask) / leaf_pixels)

        if lesion_ratio >= 0.12:
            confidence = float(np.clip(0.55 + ((lesion_ratio - 0.12) / 0.25), 0.0, 1.0))
            return HealthAssessment(
                is_healthy=False,
                confidence=round(confidence, 4),
                lesion_ratio=round(lesion_ratio, 4),
                note="The leaf shows a noticeable warm-color lesion pattern, which can indicate stress or disease.",
            )

        if lesion_ratio <= 0.05 and leaf.green_ratio >= 0.14:
            confidence = float(
                np.clip(
                    0.55 + (((0.05 - lesion_ratio) / 0.05) * 0.35),
                    0.0,
                    1.0,
                )
            )
            return HealthAssessment(
                is_healthy=True,
                confidence=round(confidence, 4),
                lesion_ratio=round(lesion_ratio, 4),
                note="The leaf appears predominantly green without a strong lesion signal.",
            )

        confidence = float(np.clip(0.35 + abs(lesion_ratio - 0.10), 0.0, 0.7))
        return HealthAssessment(
            is_healthy=None,
            confidence=round(confidence, 4),
            lesion_ratio=round(lesion_ratio, 4),
            note="Health screening is inconclusive. The real disease model should make the final decision.",
        )
