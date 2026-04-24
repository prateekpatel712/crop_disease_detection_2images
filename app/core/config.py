from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    project_name: str = "Crop Disease Detection API"
    project_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    allowed_content_types: tuple[str, ...] = (
        "image/jpeg",
        "image/png",
        "image/webp",
    )
    min_crop_confidence: float = 0.80
    caution_crop_confidence: float = 0.60
    min_disease_confidence: float = 0.75
    caution_disease_confidence: float = 0.55
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    @property
    def uploads_dir(self) -> Path:
        return self.base_dir / "uploads"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def prediction_audit_log_path(self) -> Path:
        return self.logs_dir / "prediction_events.jsonl"

    @property
    def feedback_log_path(self) -> Path:
        return self.logs_dir / "feedback_events.jsonl"

    @property
    def crop_model_dir(self) -> Path:
        return self.base_dir / "models" / "crop"

    @property
    def disease_model_dir(self) -> Path:
        return self.base_dir / "models" / "disease"

    @property
    def model_metadata_dir(self) -> Path:
        return self.base_dir / "models" / "metadata"

    @property
    def model_bundles_dir(self) -> Path:
        return self.base_dir / "models" / "bundles"

    @property
    def rag_raw_dir(self) -> Path:
        return self.base_dir / "rag_data" / "raw"

    @property
    def rag_processed_dir(self) -> Path:
        return self.base_dir / "rag_data" / "processed"

    @property
    def rag_vector_store_dir(self) -> Path:
        return self.base_dir / "rag_data" / "vector_store"

    def ensure_runtime_dirs(self) -> None:
        runtime_dirs = (
            self.uploads_dir,
            self.logs_dir,
            self.crop_model_dir,
            self.disease_model_dir,
            self.model_metadata_dir,
            self.model_bundles_dir,
            self.rag_raw_dir,
            self.rag_processed_dir,
            self.rag_vector_store_dir,
        )

        for directory in runtime_dirs:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
