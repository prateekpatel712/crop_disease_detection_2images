from __future__ import annotations

import json
from pathlib import Path

from app.models.runtime import DiseaseRouteSelection
from app.services.model_registry import ModelRegistry


class DiseaseModelRouter:
    def __init__(self, model_registry: ModelRegistry, metadata_dir: Path) -> None:
        self.model_registry = model_registry
        self.metadata_dir = metadata_dir

    def resolve(self, crop_name: str | None) -> DiseaseRouteSelection:
        available = self.model_registry.available_disease_model_keys()
        normalized_crop = self.model_registry.normalize_key(crop_name)
        requested_crop = crop_name or ""

        if not normalized_crop or normalized_crop in {
            "unknown",
            "model_not_uploaded",
            "runtime_not_connected",
        }:
            return self._fallback_route(
                requested_crop=requested_crop,
                normalized_crop=normalized_crop,
                available=available,
                note="Crop prediction is not available yet, so disease routing falls back to the generic path.",
            )

        configured = self._resolve_from_metadata(
            requested_crop=requested_crop,
            normalized_crop=normalized_crop,
            available=available,
        )
        if configured is not None:
            return configured

        if normalized_crop in available:
            return DiseaseRouteSelection(
                requested_crop=requested_crop,
                normalized_crop=normalized_crop,
                resolved_model_key=normalized_crop,
                route_found=True,
                strategy="directory_match",
                available_model_keys=available,
                note="Disease routing matched a crop-specific model directory.",
            )

        return self._fallback_route(
            requested_crop=requested_crop,
            normalized_crop=normalized_crop,
            available=available,
            note="No crop-specific disease route was found, so a fallback route was selected.",
        )

    def _resolve_from_metadata(
        self,
        *,
        requested_crop: str,
        normalized_crop: str,
        available: list[str],
    ) -> DiseaseRouteSelection | None:
        config_path = self.model_registry.routing_config_path()
        if not config_path.is_file():
            return None

        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return DiseaseRouteSelection(
                requested_crop=requested_crop,
                normalized_crop=normalized_crop,
                resolved_model_key="",
                route_found=False,
                strategy="invalid_metadata",
                available_model_keys=available,
                note="The disease route metadata file exists but could not be parsed.",
            )

        routes = payload.get("routes", []) if isinstance(payload, dict) else []
        default_model_key = ""
        if isinstance(payload, dict):
            default_model_key = self.model_registry.normalize_key(
                str(payload.get("default_model_key", ""))
            )

        for route in routes:
            if not isinstance(route, dict):
                continue
            candidate_names = [route.get("crop_name", "")]
            aliases = route.get("aliases", [])
            if isinstance(aliases, list):
                candidate_names.extend(str(alias) for alias in aliases)

            normalized_names = {
                self.model_registry.normalize_key(name)
                for name in candidate_names
                if isinstance(name, str) and name.strip()
            }

            if normalized_crop not in normalized_names:
                continue

            model_key = self.model_registry.normalize_key(str(route.get("model_key", "")))
            if model_key in available:
                return DiseaseRouteSelection(
                    requested_crop=requested_crop,
                    normalized_crop=normalized_crop,
                    resolved_model_key=model_key,
                    route_found=True,
                    strategy="metadata_match",
                    available_model_keys=available,
                    note="Disease routing matched a metadata-defined crop route.",
                )

            if default_model_key and default_model_key in available:
                return DiseaseRouteSelection(
                    requested_crop=requested_crop,
                    normalized_crop=normalized_crop,
                    resolved_model_key=default_model_key,
                    route_found=True,
                    strategy="metadata_default_fallback",
                    available_model_keys=available,
                    note="Disease routing metadata matched the crop, but its target model was missing, so the metadata default route was used.",
                )

            return DiseaseRouteSelection(
                requested_crop=requested_crop,
                normalized_crop=normalized_crop,
                resolved_model_key="",
                route_found=False,
                strategy="metadata_missing_model",
                available_model_keys=available,
                note="Disease routing metadata matched the crop, but the referenced model artifacts were not found.",
            )

        if default_model_key and default_model_key in available:
            return DiseaseRouteSelection(
                requested_crop=requested_crop,
                normalized_crop=normalized_crop,
                resolved_model_key=default_model_key,
                route_found=True,
                strategy="metadata_default_fallback",
                available_model_keys=available,
                note="Disease routing used the default model declared in metadata.",
            )

        return None

    def _fallback_route(
        self,
        *,
        requested_crop: str,
        normalized_crop: str,
        available: list[str],
        note: str,
    ) -> DiseaseRouteSelection:
        if "generic" in available:
            return DiseaseRouteSelection(
                requested_crop=requested_crop,
                normalized_crop=normalized_crop,
                resolved_model_key="generic",
                route_found=True,
                strategy="generic_fallback",
                available_model_keys=available,
                note=note,
            )

        if available:
            first_available = available[0]
            return DiseaseRouteSelection(
                requested_crop=requested_crop,
                normalized_crop=normalized_crop,
                resolved_model_key=first_available,
                route_found=True,
                strategy="first_available_fallback",
                available_model_keys=available,
                note=note,
            )

        return DiseaseRouteSelection(
            requested_crop=requested_crop,
            normalized_crop=normalized_crop,
            resolved_model_key="",
            route_found=False,
            strategy="no_route",
            available_model_keys=available,
            note="No disease model artifacts are available yet.",
        )
