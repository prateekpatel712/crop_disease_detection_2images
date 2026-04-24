from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value[:1] == value[-1:] and value[:1] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value

    return values


def _get_str(env: dict[str, str], key: str, default: str) -> str:
    return env.get(key, default).strip()


def _get_bool(env: dict[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _get_int(env: dict[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None:
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


def _get_float(env: dict[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if value is None:
        return default
    try:
        return float(value.strip())
    except ValueError:
        return default


def _get_csv_tuple(
    env: dict[str, str],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    value = env.get(key)
    if value is None:
        return default
    parts = [item.strip() for item in value.split(",")]
    cleaned = tuple(item for item in parts if item)
    return cleaned or default


@dataclass(frozen=True, slots=True)
class Settings:
    project_name: str = "Crop Disease Detection API"
    project_version: str = "0.1.0"
    environment: str = "development"
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
    enable_api_docs: bool = False
    docs_url: str | None = None
    redoc_url: str | None = None
    openapi_url: str | None = None
    api_key: str = ""
    require_api_key: bool = False
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    runtime_data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    @property
    def uploads_dir(self) -> Path:
        return self.runtime_data_dir / "uploads"

    @property
    def logs_dir(self) -> Path:
        return self.runtime_data_dir / "logs"

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
        return self.runtime_data_dir / "rag_data" / "vector_store"

    def ensure_runtime_dirs(self) -> None:
        runtime_dirs = (
            self.uploads_dir,
            self.logs_dir,
            self.rag_vector_store_dir,
        )

        for directory in runtime_dirs:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    base_dir = Path(__file__).resolve().parents[2]
    env_file_values = _load_env_file(base_dir / ".env")
    env = {**env_file_values, **os.environ}
    runtime_data_dir_value = _get_str(env, "APP_DATA_DIR", "")
    runtime_data_dir = (
        Path(runtime_data_dir_value).expanduser()
        if runtime_data_dir_value
        else base_dir
    )

    api_key = _get_str(env, "API_KEY", "")
    require_api_key = _get_bool(env, "REQUIRE_API_KEY", bool(api_key))
    if require_api_key and not api_key:
        raise RuntimeError(
            "REQUIRE_API_KEY is enabled, but API_KEY is missing. Add it to `.env` "
            "or the deployment environment before starting the app."
        )

    enable_api_docs = _get_bool(env, "ENABLE_API_DOCS", False)

    return Settings(
        project_name=_get_str(env, "PROJECT_NAME", "Crop Disease Detection API"),
        project_version=_get_str(env, "PROJECT_VERSION", "0.1.0"),
        environment=_get_str(env, "APP_ENV", "development"),
        api_prefix=_get_str(env, "API_PREFIX", "/api/v1"),
        max_upload_size_bytes=_get_int(env, "MAX_UPLOAD_SIZE_BYTES", 10 * 1024 * 1024),
        allowed_content_types=_get_csv_tuple(
            env,
            "ALLOWED_CONTENT_TYPES",
            (
                "image/jpeg",
                "image/png",
                "image/webp",
            ),
        ),
        min_crop_confidence=_get_float(env, "MIN_CROP_CONFIDENCE", 0.80),
        caution_crop_confidence=_get_float(env, "CAUTION_CROP_CONFIDENCE", 0.60),
        min_disease_confidence=_get_float(env, "MIN_DISEASE_CONFIDENCE", 0.75),
        caution_disease_confidence=_get_float(env, "CAUTION_DISEASE_CONFIDENCE", 0.55),
        enable_api_docs=enable_api_docs,
        docs_url="/docs" if enable_api_docs else None,
        redoc_url="/redoc" if enable_api_docs else None,
        openapi_url="/openapi.json" if enable_api_docs else None,
        api_key=api_key,
        require_api_key=require_api_key,
        base_dir=base_dir,
        runtime_data_dir=runtime_data_dir,
    )
