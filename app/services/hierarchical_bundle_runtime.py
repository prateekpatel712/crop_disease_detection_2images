from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.models.runtime import PredictionCandidate, ServicePrediction


@dataclass(slots=True)
class _HierarchicalStack:
    torch: Any
    torch_nn: Any
    torchvision_models: Any
    torchvision_transforms: Any


@dataclass(slots=True)
class _BundleArtifacts:
    bundle_name: str
    bundle_root: Path
    backbone_checkpoint: Path
    crop_router_path: Path
    per_crop_heads_dir: Path
    id_to_label_path: Path
    label_to_id_path: Path
    crop_order_path: Path
    preprocessing_config_path: Path
    pipeline_summary_path: Path
    crop_router_refinement_summary_path: Path
    crop_scale: float
    blend_weight: float
    manifest: dict[str, Any]

    @property
    def model_key(self) -> str:
        return "hierarchical_bundle"

    @property
    def artifact_count(self) -> int:
        return 2 + len(list(self.per_crop_heads_dir.glob("*_disease_head.joblib")))


@dataclass(slots=True)
class _HierarchicalPredictionBundle:
    crop_prediction: ServicePrediction
    disease_prediction: ServicePrediction
    final_label: str = ""
    final_confidence: float = 0.0
    final_probabilities: np.ndarray | None = None


class HierarchicalBundleRuntime:
    def __init__(self, bundles_dir: Path) -> None:
        self.bundles_dir = bundles_dir
        self._artifacts_cache: _BundleArtifacts | None = None
        self._stack_cache: _HierarchicalStack | None = None
        self._model_cache: Any | None = None
        self._router_payload: dict[str, Any] | None = None
        self._head_payloads: dict[str, dict[str, Any]] | None = None
        self._id_to_label: dict[int, str] | None = None
        self._crop_order: list[str] | None = None
        self._preprocess_config: dict[str, Any] | None = None
        self._prediction_cache: dict[str, _HierarchicalPredictionBundle] = {}

    def artifacts_detected(self) -> bool:
        return self._load_artifacts() is not None

    def available_model_keys(self) -> list[str]:
        return ["hierarchical_bundle"] if self.artifacts_detected() else []

    def route_payload(self, crop_name: str | None = None) -> dict[str, Any]:
        artifacts = self._load_artifacts()
        if artifacts is None:
            return {
                "requested_crop": crop_name or "",
                "normalized_crop": (crop_name or "").strip().lower(),
                "resolved_model_key": "",
                "route_found": False,
                "strategy": "no_route",
                "available_model_keys": [],
                "note": "No hierarchical crop-disease bundle is registered yet.",
            }

        return {
            "requested_crop": crop_name or "",
            "normalized_crop": (crop_name or "").strip().lower(),
            "resolved_model_key": artifacts.model_key,
            "route_found": True,
            "strategy": "hierarchical_bundle",
            "available_model_keys": [artifacts.model_key],
            "note": (
                "Using the shared hierarchical crop-disease bundle with one backbone, "
                "one crop router, and per-crop disease heads."
            ),
        }

    def predict_crop(self, image_path: str) -> ServicePrediction:
        bundle = self._predict_bundle(image_path)
        return bundle.crop_prediction

    def predict_disease(self, image_path: str, crop_name: str | None = None) -> ServicePrediction:
        bundle = self._predict_bundle(image_path)
        if crop_name:
            constrained_prediction = self._build_constrained_disease_prediction(
                bundle=bundle,
                crop_name=crop_name,
            )
            if constrained_prediction is not None:
                return constrained_prediction
        return bundle.disease_prediction

    def _predict_bundle(self, image_path: str) -> _HierarchicalPredictionBundle:
        normalized_key = str(Path(image_path).resolve())
        cached = self._prediction_cache.get(normalized_key)
        if cached is not None:
            return cached

        artifacts = self._load_artifacts()
        if artifacts is None:
            bundle = self._missing_artifact_bundle()
            self._prediction_cache[normalized_key] = bundle
            return bundle

        stack = self._load_stack()
        if stack is None:
            note = (
                "The hierarchical bundle requires `torch` and `torchvision`, but they "
                "are not installed in the current Python environment."
            )
            bundle = self._runtime_blocked_bundle(
                artifacts=artifacts,
                runtime_stage="missing_dependency",
                note=note,
            )
            self._prediction_cache[normalized_key] = bundle
            return bundle

        try:
            features = self._extract_features(Path(image_path), artifacts, stack)
            final_probabilities = self._predict_final_probabilities(features, artifacts)
            bundle = self._build_prediction_bundle(final_probabilities, artifacts)
        except Exception as exc:
            bundle = self._runtime_blocked_bundle(
                artifacts=artifacts,
                runtime_stage="runtime_error",
                note=f"The hierarchical bundle failed during inference: {exc}",
            )

        self._prediction_cache[normalized_key] = bundle
        return bundle

    def _load_artifacts(self) -> _BundleArtifacts | None:
        if self._artifacts_cache is not None:
            return self._artifacts_cache

        if not self.bundles_dir.exists():
            return None

        candidates = sorted(
            (
                path.parent.parent
                for path in self.bundles_dir.glob("*/metadata/bundle_manifest.json")
            ),
            key=lambda path: path.name,
            reverse=True,
        )
        for bundle_root in candidates:
            manifest_path = bundle_root / "metadata" / "bundle_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue

            artifacts_payload = manifest.get("artifacts", {})
            try:
                artifacts = _BundleArtifacts(
                    bundle_name=str(manifest.get("bundle_name", bundle_root.name)),
                    bundle_root=bundle_root,
                    backbone_checkpoint=bundle_root / str(artifacts_payload["backbone_checkpoint"]),
                    crop_router_path=bundle_root / str(artifacts_payload["crop_router"]),
                    per_crop_heads_dir=bundle_root / str(artifacts_payload["per_crop_heads_dir"]),
                    id_to_label_path=bundle_root / str(artifacts_payload["id_to_label"]),
                    label_to_id_path=bundle_root / str(artifacts_payload["label_to_id"]),
                    crop_order_path=bundle_root / str(artifacts_payload["crop_order"]),
                    preprocessing_config_path=bundle_root / str(artifacts_payload["preprocessing_config"]),
                    pipeline_summary_path=bundle_root / str(artifacts_payload["pipeline_summary"]),
                    crop_router_refinement_summary_path=bundle_root / str(
                        artifacts_payload["crop_router_refinement_summary"]
                    ),
                    crop_scale=float(
                        manifest.get("selected_inference_settings", {}).get("crop_scale", 1.0)
                    ),
                    blend_weight=float(
                        manifest.get("selected_inference_settings", {}).get("blend_weight", 0.0)
                    ),
                    manifest=manifest,
                )
            except KeyError:
                continue

            required_paths = (
                artifacts.backbone_checkpoint,
                artifacts.crop_router_path,
                artifacts.per_crop_heads_dir,
                artifacts.id_to_label_path,
                artifacts.crop_order_path,
                artifacts.preprocessing_config_path,
                artifacts.crop_router_refinement_summary_path,
            )
            if not all(path.exists() for path in required_paths):
                continue

            self._artifacts_cache = artifacts
            return artifacts

        return None

    def _load_stack(self) -> _HierarchicalStack | None:
        if self._stack_cache is not None:
            return self._stack_cache

        try:
            torch_module = import_module("torch")
            torch_nn = import_module("torch.nn")
            torchvision_models = import_module("torchvision.models")
            torchvision_transforms = import_module("torchvision.transforms")
        except ModuleNotFoundError:
            return None

        self._stack_cache = _HierarchicalStack(
            torch=torch_module,
            torch_nn=torch_nn,
            torchvision_models=torchvision_models,
            torchvision_transforms=torchvision_transforms,
        )
        return self._stack_cache

    def _load_model(self, artifacts: _BundleArtifacts, stack: _HierarchicalStack) -> Any:
        if self._model_cache is not None:
            return self._model_cache

        id_to_label = self._load_id_to_label(artifacts)

        class EfficientNetFeatureModel(stack.torch_nn.Module):
            def __init__(self, num_classes: int) -> None:
                super().__init__()
                self.model = stack.torchvision_models.efficientnet_b0(weights=None)
                in_features = self.model.classifier[1].in_features
                self.model.classifier = stack.torch_nn.Sequential(
                    stack.torch_nn.Dropout(p=0.30, inplace=True),
                    stack.torch_nn.Linear(in_features, num_classes),
                )

            def forward(self, x: Any) -> tuple[Any, Any]:
                x = self.model.features(x)
                x = self.model.avgpool(x)
                features = stack.torch.flatten(x, 1)
                logits = self.model.classifier(features)
                return features, logits

        model = EfficientNetFeatureModel(num_classes=len(id_to_label))
        checkpoint = stack.torch.load(artifacts.backbone_checkpoint, map_location="cpu")
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self._model_cache = model
        return model

    def _load_router_payload(self, artifacts: _BundleArtifacts) -> dict[str, Any]:
        if self._router_payload is None:
            self._router_payload = joblib.load(artifacts.crop_router_path)
        return self._router_payload

    def _load_head_payloads(self, artifacts: _BundleArtifacts) -> dict[str, dict[str, Any]]:
        if self._head_payloads is None:
            self._head_payloads = {}
            for path in sorted(artifacts.per_crop_heads_dir.glob("*_disease_head.joblib")):
                crop_name = path.name.replace("_disease_head.joblib", "")
                self._head_payloads[crop_name] = joblib.load(path)
        return self._head_payloads

    def _load_id_to_label(self, artifacts: _BundleArtifacts) -> dict[int, str]:
        if self._id_to_label is None:
            payload = json.loads(artifacts.id_to_label_path.read_text(encoding="utf-8"))
            self._id_to_label = {int(key): str(value) for key, value in payload.items()}
        return self._id_to_label

    def _load_crop_order(self, artifacts: _BundleArtifacts) -> list[str]:
        if self._crop_order is None:
            payload = json.loads(artifacts.crop_order_path.read_text(encoding="utf-8"))
            self._crop_order = [str(item) for item in payload]
        return self._crop_order

    def _load_preprocess_config(self, artifacts: _BundleArtifacts) -> dict[str, Any]:
        if self._preprocess_config is None:
            self._preprocess_config = json.loads(
                artifacts.preprocessing_config_path.read_text(encoding="utf-8")
            )
        return self._preprocess_config

    def _extract_features(
        self,
        image_path: Path,
        artifacts: _BundleArtifacts,
        stack: _HierarchicalStack,
    ) -> np.ndarray:
        preprocess_config = self._load_preprocess_config(artifacts)
        transform = stack.torchvision_transforms.Compose(
            [
                stack.torchvision_transforms.Resize(
                    int(preprocess_config["resize_shorter_side"])
                ),
                stack.torchvision_transforms.CenterCrop(
                    int(preprocess_config["image_size"])
                ),
                stack.torchvision_transforms.ToTensor(),
                stack.torchvision_transforms.Normalize(
                    preprocess_config["normalize_mean"],
                    preprocess_config["normalize_std"],
                ),
            ]
        )

        image_module = import_module("PIL.Image")
        with image_module.open(image_path) as image:
            image_tensor = transform(image.convert("RGB")).unsqueeze(0)

        model = self._load_model(artifacts, stack)
        with stack.torch.no_grad():
            features_tensor, _ = model(image_tensor)
        return features_tensor.cpu().numpy()

    def _predict_final_probabilities(
        self,
        features: np.ndarray,
        artifacts: _BundleArtifacts,
    ) -> np.ndarray:
        router_payload = self._load_router_payload(artifacts)
        head_payloads = self._load_head_payloads(artifacts)
        crop_to_ids = self._build_crop_to_ids(self._load_id_to_label(artifacts))
        crop_log_probs = self._predict_crop_log_probs(features, router_payload)

        if artifacts.blend_weight != 0.0:
            raise RuntimeError(
                "This integration expects `blend_weight` 0.0 for the selected hierarchical bundle."
            )

        scores = self._build_hierarchical_scores(
            features=features,
            crop_log_probs=crop_log_probs,
            crop_to_ids=crop_to_ids,
            num_classes=len(self._load_id_to_label(artifacts)),
            crop_names=self._resolve_crop_names(router_payload, artifacts),
            crop_model_payloads=head_payloads,
            crop_scale=artifacts.crop_scale,
        )
        shifted = scores[0] - np.max(scores[0])
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        return probabilities

    def _predict_crop_log_probs(self, features: np.ndarray, payload: dict[str, Any]) -> np.ndarray:
        x_eval = features
        scaler = payload.get("scaler")
        if scaler is not None:
            x_eval = scaler.transform(x_eval)
        classifier = payload["classifier"]
        return classifier.predict_log_proba(x_eval)

    def _build_hierarchical_scores(
        self,
        *,
        features: np.ndarray,
        crop_log_probs: np.ndarray,
        crop_to_ids: dict[str, list[int]],
        num_classes: int,
        crop_names: list[str],
        crop_model_payloads: dict[str, dict[str, Any]],
        crop_scale: float,
    ) -> np.ndarray:
        scores = np.full((features.shape[0], num_classes), -1e9, dtype=np.float64)
        crop_log_prob_map = {
            crop_names[index]: crop_log_probs[:, index]
            for index in range(min(len(crop_names), crop_log_probs.shape[1]))
        }

        for crop, class_ids in crop_to_ids.items():
            payload = crop_model_payloads[crop]
            if payload.get("kind") == "constant":
                disease_log_probs = np.full((features.shape[0], len(class_ids)), -1e9, dtype=np.float64)
                disease_log_probs[:, int(payload["constant_local_class"])] = 0.0
            else:
                scaler = payload.get("scaler")
                x_scaled = scaler.transform(features) if scaler is not None else features
                raw_log_probs = payload["classifier"].predict_log_proba(x_scaled)
                disease_log_probs = np.full((features.shape[0], len(class_ids)), -1e9, dtype=np.float64)
                seen_local_classes = payload.get(
                    "seen_local_classes",
                    list(range(raw_log_probs.shape[1])),
                )
                for column_index, local_class_index in enumerate(seen_local_classes):
                    disease_log_probs[:, int(local_class_index)] = raw_log_probs[:, column_index]

            crop_component = crop_scale * crop_log_prob_map.get(
                crop,
                np.full((features.shape[0],), -1e9, dtype=np.float64),
            )
            for local_index, class_id in enumerate(class_ids):
                scores[:, class_id] = crop_component + disease_log_probs[:, local_index]

        return scores

    def _build_prediction_bundle(
        self,
        final_probabilities: np.ndarray,
        artifacts: _BundleArtifacts,
    ) -> _HierarchicalPredictionBundle:
        id_to_label = self._load_id_to_label(artifacts)
        crop_to_ids = self._build_crop_to_ids(id_to_label)
        final_class_id = int(np.argmax(final_probabilities))
        final_label = id_to_label[final_class_id]
        final_crop, final_disease = final_label.split("::", 1)

        crop_marginals = {
            crop: float(np.sum(final_probabilities[class_ids]))
            for crop, class_ids in crop_to_ids.items()
        }
        crop_candidates = sorted(
            (
                PredictionCandidate(label=crop, confidence=probability)
                for crop, probability in crop_marginals.items()
            ),
            key=lambda candidate: candidate.confidence,
            reverse=True,
        )[:3]

        crop_class_ids = crop_to_ids[final_crop]
        crop_total_probability = max(float(np.sum(final_probabilities[crop_class_ids])), 1e-12)
        disease_candidates: list[PredictionCandidate] = []
        disease_supported_labels: list[str] = []
        for class_id in crop_class_ids:
            label = id_to_label[class_id]
            _, disease_name = label.split("::", 1)
            disease_supported_labels.append(disease_name)
            disease_candidates.append(
                PredictionCandidate(
                    label=disease_name,
                    confidence=float(final_probabilities[class_id] / crop_total_probability),
                )
            )
        disease_candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        disease_candidates = disease_candidates[:3]

        crop_prediction = ServicePrediction(
            label=final_crop,
            confidence=crop_marginals[final_crop],
            top_k=crop_candidates,
            model_ready=True,
            model_key=artifacts.model_key,
            artifact_count=artifacts.artifact_count,
            supported_labels=self._load_crop_order(artifacts),
            provider="phase6-hierarchical-bundle",
            runtime_stage="predicted",
            note=f"Crop prediction came from the hierarchical bundle final class `{final_label}`.",
        )
        disease_prediction = ServicePrediction(
            label=final_disease,
            confidence=disease_candidates[0].confidence if disease_candidates else 0.0,
            top_k=disease_candidates,
            model_ready=True,
            model_key=artifacts.model_key,
            artifact_count=artifacts.artifact_count,
            supported_labels=disease_supported_labels,
            provider="phase6-hierarchical-bundle",
            runtime_stage="predicted",
            note=f"Disease prediction came from the hierarchical bundle final class `{final_label}`.",
        )
        return _HierarchicalPredictionBundle(
            crop_prediction=crop_prediction,
            disease_prediction=disease_prediction,
            final_label=final_label,
            final_confidence=float(final_probabilities[final_class_id]),
            final_probabilities=final_probabilities,
        )

    def _build_constrained_disease_prediction(
        self,
        *,
        bundle: _HierarchicalPredictionBundle,
        crop_name: str,
    ) -> ServicePrediction | None:
        normalized_crop = str(crop_name or "").strip().lower()
        if not normalized_crop or bundle.final_probabilities is None:
            return None

        artifacts = self._load_artifacts()
        if artifacts is None:
            return None

        id_to_label = self._load_id_to_label(artifacts)
        crop_to_ids = self._build_crop_to_ids(id_to_label)
        crop_class_ids = crop_to_ids.get(normalized_crop)
        if not crop_class_ids:
            return None

        crop_total_probability = max(
            float(np.sum(bundle.final_probabilities[crop_class_ids])),
            1e-12,
        )
        disease_candidates: list[PredictionCandidate] = []
        for class_id in crop_class_ids:
            label = id_to_label[class_id]
            _, disease_name = label.split("::", 1)
            disease_candidates.append(
                PredictionCandidate(
                    label=disease_name,
                    confidence=float(bundle.final_probabilities[class_id] / crop_total_probability),
                )
            )

        disease_candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        if not disease_candidates:
            return None

        top_label = f"{normalized_crop}::{disease_candidates[0].label}"
        return ServicePrediction(
            label=disease_candidates[0].label,
            confidence=disease_candidates[0].confidence,
            top_k=disease_candidates[:3],
            model_ready=True,
            model_key=artifacts.model_key,
            artifact_count=artifacts.artifact_count,
            supported_labels=[candidate.label for candidate in disease_candidates],
            provider="phase6-hierarchical-bundle",
            runtime_stage="predicted",
            note=(
                "Disease prediction was constrained to the selected crop head and "
                f"came from the hierarchical bundle class `{top_label}`."
            ),
        )

    def _resolve_crop_names(
        self,
        router_payload: dict[str, Any],
        artifacts: _BundleArtifacts,
    ) -> list[str]:
        crop_names = router_payload.get("crop_names")
        if isinstance(crop_names, list) and crop_names:
            return [str(item) for item in crop_names]
        return self._load_crop_order(artifacts)

    @staticmethod
    def _build_crop_to_ids(id_to_label: dict[int, str]) -> dict[str, list[int]]:
        crop_to_ids: dict[str, list[int]] = {}
        for class_id, label in sorted(id_to_label.items()):
            crop_name = label.split("::", 1)[0]
            crop_to_ids.setdefault(crop_name, []).append(class_id)
        return crop_to_ids

    def _missing_artifact_bundle(self) -> _HierarchicalPredictionBundle:
        note = (
            "No hierarchical crop-disease bundle is registered yet. Add a bundle under "
            "`models/bundles/` to enable the shared backbone/router/head runtime."
        )
        crop_prediction = ServicePrediction(
            label="model_not_uploaded",
            confidence=0.0,
            model_ready=False,
            model_key="hierarchical_bundle",
            artifact_count=0,
            supported_labels=[],
            provider="phase6-hierarchical-bundle",
            runtime_stage="awaiting_artifacts",
            note=note,
        )
        disease_prediction = ServicePrediction(
            label="model_not_uploaded",
            confidence=0.0,
            model_ready=False,
            model_key="hierarchical_bundle",
            artifact_count=0,
            supported_labels=[],
            provider="phase6-hierarchical-bundle",
            runtime_stage="awaiting_artifacts",
            note=note,
        )
        return _HierarchicalPredictionBundle(
            crop_prediction=crop_prediction,
            disease_prediction=disease_prediction,
        )

    def _runtime_blocked_bundle(
        self,
        *,
        artifacts: _BundleArtifacts,
        runtime_stage: str,
        note: str,
    ) -> _HierarchicalPredictionBundle:
        crop_prediction = ServicePrediction(
            label="runtime_not_connected",
            confidence=0.0,
            model_ready=False,
            model_key=artifacts.model_key,
            artifact_count=artifacts.artifact_count,
            supported_labels=self._load_crop_order(artifacts),
            provider="phase6-hierarchical-bundle",
            runtime_stage=runtime_stage,
            note=note,
        )
        disease_prediction = ServicePrediction(
            label="runtime_not_connected",
            confidence=0.0,
            model_ready=False,
            model_key=artifacts.model_key,
            artifact_count=artifacts.artifact_count,
            supported_labels=[],
            provider="phase6-hierarchical-bundle",
            runtime_stage=runtime_stage,
            note=note,
        )
        return _HierarchicalPredictionBundle(
            crop_prediction=crop_prediction,
            disease_prediction=disease_prediction,
        )
