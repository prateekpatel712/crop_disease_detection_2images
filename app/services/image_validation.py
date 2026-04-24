from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import Settings


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    image_ok: bool
    issues: list[str] = field(default_factory=list)


class ImageValidationService:
    _allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def validate(
        self,
        *,
        filename: str,
        content_type: str | None,
        size_bytes: int,
    ) -> ValidationOutcome:
        issues: list[str] = []
        image_ok = True

        extension = Path(filename).suffix.lower()

        if not extension or extension not in self._allowed_extensions:
            image_ok = False
            issues.append(
                "Unsupported file extension. Use JPG, PNG, or WEBP images."
            )

        if content_type and content_type not in self.settings.allowed_content_types:
            image_ok = False
            issues.append(
                "Unsupported content type. Upload a JPEG, PNG, or WEBP image."
            )

        if size_bytes <= 0:
            image_ok = False
            issues.append("The uploaded image is empty.")

        if size_bytes > self.settings.max_upload_size_bytes:
            image_ok = False
            issues.append(
                f"Image exceeds the {self.settings.max_upload_size_bytes} byte limit."
            )

        return ValidationOutcome(image_ok=image_ok, issues=issues)
