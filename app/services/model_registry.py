from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.models.runtime import ModelArtifactBundle


class ModelRegistry:
    artifact_suffixes = {".pt", ".pth", ".onnx", ".joblib", ".pkl", ".safetensors"}
    label_filenames = (
        "labels.json",
        "label_map.json",
        "classes.json",
        "class_names.json",
        "labels.txt",
        "classes.txt",
    )
    manifest_filenames = (
        "model.json",
        "metadata.json",
        "manifest.json",
        "config.json",
    )

    def __init__(
        self,
        *,
        crop_model_dir: Path,
        disease_model_dir: Path,
        metadata_dir: Path,
    ) -> None:
        self.crop_model_dir = crop_model_dir
        self.disease_model_dir = disease_model_dir
        self.metadata_dir = metadata_dir

    @staticmethod
    def normalize_key(value: str | None) -> str:
        if not value:
            return ""
        return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")

    def crop_bundle(self) -> ModelArtifactBundle:
        return self._build_bundle(
            model_key="crop_classifier",
            base_dir=self.crop_model_dir,
            recursive=True,
        )

    def disease_bundles(self) -> dict[str, ModelArtifactBundle]:
        bundles: dict[str, ModelArtifactBundle] = {}

        generic_bundle = self._build_bundle(
            model_key="generic",
            base_dir=self.disease_model_dir,
            recursive=False,
        )
        if generic_bundle.runtime_ready:
            bundles[generic_bundle.model_key] = generic_bundle

        for child_dir in sorted(self.disease_model_dir.iterdir()):
            if not child_dir.is_dir():
                continue
            model_key = self.normalize_key(child_dir.name)
            if not model_key:
                continue
            bundle = self._build_bundle(
                model_key=model_key,
                base_dir=child_dir,
                recursive=True,
            )
            if bundle.runtime_ready:
                bundles[model_key] = bundle

        return bundles

    def disease_bundle(self, model_key: str | None) -> ModelArtifactBundle:
        normalized_key = self.normalize_key(model_key)
        bundles = self.disease_bundles()

        if normalized_key and normalized_key in bundles:
            return bundles[normalized_key]

        if normalized_key == "generic":
            return self._build_bundle(
                model_key="generic",
                base_dir=self.disease_model_dir,
                recursive=False,
            )

        if normalized_key:
            candidate_dir = self.disease_model_dir / normalized_key
            if candidate_dir.exists():
                return self._build_bundle(
                    model_key=normalized_key,
                    base_dir=candidate_dir,
                    recursive=True,
                )

        return ModelArtifactBundle(
            model_key=normalized_key or "generic",
            model_dir=str(
                (self.disease_model_dir / normalized_key)
                if normalized_key and normalized_key != "generic"
                else self.disease_model_dir
            ),
            runtime_ready=False,
            note="No disease model artifacts were detected for the requested route.",
        )

    def available_disease_model_keys(self) -> list[str]:
        return sorted(self.disease_bundles().keys())

    def routing_config_path(self) -> Path:
        return self.metadata_dir / "disease_routes.json"

    def routing_config_detected(self) -> bool:
        return self.routing_config_path().is_file()

    def _build_bundle(
        self,
        *,
        model_key: str,
        base_dir: Path,
        recursive: bool,
    ) -> ModelArtifactBundle:
        if not base_dir.exists():
            return ModelArtifactBundle(
                model_key=model_key,
                model_dir=str(base_dir),
                runtime_ready=False,
                note="Model directory does not exist yet.",
            )

        weight_files = self._find_files(
            base_dir=base_dir,
            recursive=recursive,
            predicate=lambda path: path.suffix.lower() in self.artifact_suffixes,
        )
        label_map_path = self._find_named_file(
            base_dir=base_dir,
            recursive=recursive,
            names=self.label_filenames,
        )
        manifest_path = self._find_named_file(
            base_dir=base_dir,
            recursive=recursive,
            names=self.manifest_filenames,
        )
        manifest = self._read_manifest(manifest_path)
        supported_labels = self._read_labels(label_map_path)

        note = (
            "Model artifacts detected and ready for runtime integration."
            if weight_files
            else "No model weights detected yet."
        )

        return ModelArtifactBundle(
            model_key=model_key,
            model_dir=str(base_dir),
            runtime_ready=bool(weight_files),
            weight_files=[str(path) for path in weight_files],
            label_map_path=str(label_map_path) if label_map_path else None,
            manifest_path=str(manifest_path) if manifest_path else None,
            manifest=manifest,
            supported_labels=supported_labels,
            note=note,
        )

    def _find_files(
        self,
        *,
        base_dir: Path,
        recursive: bool,
        predicate,
    ) -> list[Path]:
        iterator = base_dir.rglob("*") if recursive else base_dir.glob("*")
        return sorted(
            (
                path
                for path in iterator
                if path.is_file() and predicate(path)
            ),
            key=lambda path: str(path),
        )

    def _find_named_file(
        self,
        *,
        base_dir: Path,
        recursive: bool,
        names: tuple[str, ...],
    ) -> Path | None:
        normalized_names = {name.lower() for name in names}
        iterator = base_dir.rglob("*") if recursive else base_dir.glob("*")

        for path in sorted(iterator, key=lambda candidate: str(candidate)):
            if path.is_file() and path.name.lower() in normalized_names:
                return path
        return None

    def _read_labels(self, label_map_path: Path | None) -> list[str]:
        if not label_map_path or not label_map_path.is_file():
            return []

        try:
            if label_map_path.suffix.lower() == ".txt":
                return [
                    line.strip()
                    for line in label_map_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]

            payload = json.loads(label_map_path.read_text(encoding="utf-8"))
            return self._extract_labels_from_json(payload)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return []

    def _read_manifest(self, manifest_path: Path | None) -> dict[str, Any]:
        if not manifest_path or not manifest_path.is_file():
            return {}

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return {}

        return payload if isinstance(payload, dict) else {}

    def _extract_labels_from_json(self, payload: Any) -> list[str]:
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]

        if isinstance(payload, dict):
            for key in ("labels", "classes", "class_names"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [
                        str(item).strip()
                        for item in value
                        if str(item).strip()
                    ]

            int_like_keys = []
            string_values = []

            for key, value in payload.items():
                if isinstance(value, str) and value.strip():
                    string_values.append(value.strip())
                    if str(key).isdigit():
                        int_like_keys.append((int(str(key)), value.strip()))

            if int_like_keys:
                return [value for _, value in sorted(int_like_keys)]

            if string_values:
                return sorted(dict.fromkeys(string_values))

        return []
