from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.models.runtime import ModelArtifactBundle, PredictionCandidate, ServicePrediction


class ImageClassifierRuntime(Protocol):
    def predict(
        self,
        image_path: str,
        bundle: ModelArtifactBundle,
        *,
        task_name: str,
    ) -> ServicePrediction: ...


@dataclass(slots=True)
class _TorchStack:
    torch: Any
    torch_nn: Any
    torchvision_models: Any


class TorchvisionImageClassifierRuntime:
    _default_architectures = (
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_v2_s",
        "resnet18",
        "resnet50",
    )
    _efficientnet_architectures = (
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_v2_s",
        "efficientnet_v2_m",
    )

    def __init__(self) -> None:
        self._model_cache: dict[tuple[str, str, int], Any] = {}

    def predict(
        self,
        image_path: str,
        bundle: ModelArtifactBundle,
        *,
        task_name: str,
    ) -> ServicePrediction:
        if not bundle.weight_files:
            return self._prediction(
                bundle=bundle,
                label="model_not_uploaded",
                confidence=0.0,
                top_k=[],
                model_ready=False,
                runtime_stage="awaiting_artifacts",
                note=f"No {task_name} model weights are available yet.",
            )

        stack = self._import_torch_stack()
        if stack is None:
            note = (
                f"The {task_name} runtime requires `torch` and `torchvision`, "
                "but they are not installed in the current Python environment."
            )
            if not bundle.supported_labels:
                note += " A crop label map is also still required before predictions can return names."
            return self._prediction(
                bundle=bundle,
                label="runtime_not_connected",
                confidence=0.0,
                top_k=[],
                model_ready=False,
                runtime_stage="missing_dependency",
                note=note,
            )

        checkpoint_path = Path(bundle.weight_files[0])
        try:
            checkpoint = stack.torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:
            return self._prediction(
                bundle=bundle,
                label="runtime_not_connected",
                confidence=0.0,
                top_k=[],
                model_ready=False,
                runtime_stage="checkpoint_load_failed",
                note=f"The checkpoint could not be loaded: {exc}",
            )

        state_dict = self._extract_state_dict(checkpoint)
        if not state_dict:
            return self._prediction(
                bundle=bundle,
                label="runtime_not_connected",
                confidence=0.0,
                top_k=[],
                model_ready=False,
                runtime_stage="unsupported_checkpoint",
                note="The checkpoint does not expose a usable model state dictionary.",
            )

        inferred_num_classes = self._infer_num_classes_from_state_dict(state_dict)
        labels = bundle.supported_labels

        if not labels:
            note = (
                "The model weights are registered, but a label map is still missing. "
                "Add `labels.txt` or `labels.json` under the model directory to enable named predictions."
            )
            if inferred_num_classes:
                note += f" The checkpoint appears to have {inferred_num_classes} output classes."
            return self._prediction(
                bundle=bundle,
                label="runtime_not_connected",
                confidence=0.0,
                top_k=[],
                model_ready=False,
                runtime_stage="awaiting_labels",
                note=note,
            )

        if inferred_num_classes and len(labels) != inferred_num_classes:
            return self._prediction(
                bundle=bundle,
                label="runtime_not_connected",
                confidence=0.0,
                top_k=[],
                model_ready=False,
                runtime_stage="label_mismatch",
                note=(
                    f"The label map has {len(labels)} entries, but the checkpoint head expects "
                    f"{inferred_num_classes} classes. Align the labels before enabling inference."
                ),
            )

        num_classes = inferred_num_classes or len(labels)
        candidate_architectures = self._resolve_candidate_architectures(bundle.manifest)
        load_result = self._load_runtime_model(
            stack=stack,
            state_dict=state_dict,
            candidate_architectures=candidate_architectures,
            num_classes=num_classes,
            checkpoint_path=checkpoint_path,
        )

        if load_result is None:
            return self._prediction(
                bundle=bundle,
                label="runtime_not_connected",
                confidence=0.0,
                top_k=[],
                model_ready=False,
                runtime_stage="architecture_mismatch",
                note=(
                    "The runtime could not match the checkpoint to a supported torchvision classifier architecture. "
                    "Confirm the training architecture in the manifest before enabling inference."
                ),
            )

        model, matched_architecture = load_result

        try:
            input_tensor = self._preprocess_image(
                image_path=image_path,
                torch_module=stack.torch,
                manifest=bundle.manifest,
            )
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            return self._prediction(
                bundle=bundle,
                label="runtime_not_connected",
                confidence=0.0,
                top_k=[],
                model_ready=False,
                runtime_stage="image_preprocess_failed",
                note=f"The uploaded image could not be prepared for inference: {exc}",
            )

        try:
            with stack.torch.inference_mode():
                logits = model(input_tensor)
                probabilities = stack.torch.softmax(logits, dim=1)[0]
        except Exception as exc:
            return self._prediction(
                bundle=bundle,
                label="runtime_not_connected",
                confidence=0.0,
                top_k=[],
                model_ready=False,
                runtime_stage="runtime_error",
                note=f"Inference failed while running the model: {exc}",
            )

        top_k_count = min(3, len(labels))
        top_values, top_indices = stack.torch.topk(probabilities, k=top_k_count)
        candidates = [
            PredictionCandidate(
                label=labels[int(index)],
                confidence=float(value),
            )
            for value, index in zip(top_values.tolist(), top_indices.tolist())
        ]

        if not candidates:
            return self._prediction(
                bundle=bundle,
                label="runtime_not_connected",
                confidence=0.0,
                top_k=[],
                model_ready=False,
                runtime_stage="empty_prediction",
                note="The model ran but did not produce a valid prediction vector.",
            )

        return self._prediction(
            bundle=bundle,
            label=candidates[0].label,
            confidence=candidates[0].confidence,
            top_k=candidates,
            model_ready=True,
            runtime_stage="predicted",
            provider=f"phase6-torchvision-{matched_architecture}",
            note=f"Prediction generated using `{matched_architecture}` on the registered checkpoint.",
        )

    def _import_torch_stack(self) -> _TorchStack | None:
        try:
            torch_module = import_module("torch")
            torch_nn = import_module("torch.nn")
            torchvision_models = import_module("torchvision.models")
        except ModuleNotFoundError:
            return None

        return _TorchStack(
            torch=torch_module,
            torch_nn=torch_nn,
            torchvision_models=torchvision_models,
        )

    def _extract_state_dict(self, checkpoint: Any) -> dict[str, Any]:
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return self._normalize_state_dict_keys(value)

            if self._looks_like_state_dict(checkpoint):
                return self._normalize_state_dict_keys(checkpoint)

            nested_model = checkpoint.get("model")
            if isinstance(nested_model, dict) and self._looks_like_state_dict(nested_model):
                return self._normalize_state_dict_keys(nested_model)

        return {}

    def _normalize_state_dict_keys(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        candidates = [state_dict]
        for prefix in ("module.", "model.", "backbone.", "net."):
            if any(str(key).startswith(prefix) for key in state_dict):
                stripped = {
                    str(key)[len(prefix) :] if str(key).startswith(prefix) else str(key): value
                    for key, value in state_dict.items()
                }
                candidates.append(stripped)

        scored = sorted(
            candidates,
            key=lambda candidate: self._state_dict_score(candidate),
            reverse=True,
        )
        return scored[0]

    def _state_dict_score(self, state_dict: dict[str, Any]) -> int:
        score = 0
        for key in state_dict:
            key_text = str(key)
            if key_text.startswith("features."):
                score += 2
            if key_text.startswith("classifier."):
                score += 3
            if key_text.startswith("fc."):
                score += 2
        return score

    def _looks_like_state_dict(self, payload: dict[str, Any]) -> bool:
        return any(
            str(key).endswith(".weight") or str(key).endswith(".bias")
            for key in payload
        )

    def _infer_num_classes_from_state_dict(self, state_dict: dict[str, Any]) -> int | None:
        preferred_suffixes = (
            "classifier.1.weight",
            "classifier.weight",
            "fc.weight",
            "head.weight",
        )

        for suffix in preferred_suffixes:
            for key, value in state_dict.items():
                if str(key).endswith(suffix) and getattr(value, "ndim", None) == 2:
                    return int(value.shape[0])

        fallback_keys = [
            value
            for key, value in state_dict.items()
            if str(key).endswith(".weight")
            and getattr(value, "ndim", None) == 2
            and any(token in str(key) for token in ("classifier", "fc", "head"))
        ]
        if fallback_keys:
            return int(fallback_keys[-1].shape[0])

        return None

    def _resolve_candidate_architectures(self, manifest: dict[str, Any]) -> tuple[str, ...]:
        runtime_config = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
        explicit_architecture = str(runtime_config.get("architecture", "")).strip()
        explicit_candidates = runtime_config.get("candidate_architectures", [])

        if isinstance(explicit_candidates, list):
            cleaned = tuple(
                str(item).strip()
                for item in explicit_candidates
                if str(item).strip()
            )
            if cleaned:
                return cleaned

        if explicit_architecture and explicit_architecture not in {"auto", "efficientnet_auto"}:
            return (explicit_architecture,)

        architecture_hint = (
            str(
                manifest.get("checkpoint_observations", {}).get("architecture_hint", "")
            ).lower()
            if isinstance(manifest, dict)
            else ""
        )
        if "efficientnet" in architecture_hint or explicit_architecture == "efficientnet_auto":
            return self._efficientnet_architectures

        return self._default_architectures

    def _load_runtime_model(
        self,
        *,
        stack: _TorchStack,
        state_dict: dict[str, Any],
        candidate_architectures: tuple[str, ...],
        num_classes: int,
        checkpoint_path: Path,
    ) -> tuple[Any, str] | None:
        for architecture in candidate_architectures:
            cache_key = (str(checkpoint_path), architecture, num_classes)
            cached = self._model_cache.get(cache_key)
            if cached is not None:
                return cached, architecture

            try:
                model = self._build_model(
                    stack=stack,
                    architecture=architecture,
                    num_classes=num_classes,
                )
                model.load_state_dict(state_dict, strict=True)
                model.eval()
            except Exception:
                continue

            self._model_cache[cache_key] = model
            return model, architecture

        return None

    def _build_model(
        self,
        *,
        stack: _TorchStack,
        architecture: str,
        num_classes: int,
    ) -> Any:
        if not hasattr(stack.torchvision_models, architecture):
            raise ValueError(f"Unsupported architecture `{architecture}`.")

        constructor = getattr(stack.torchvision_models, architecture)
        model = constructor(weights=None)
        self._replace_classifier_head(model, num_classes, stack.torch_nn)
        return model

    def _replace_classifier_head(self, model: Any, num_classes: int, torch_nn: Any) -> None:
        classifier = getattr(model, "classifier", None)

        if classifier is not None:
            if hasattr(classifier, "__len__") and len(classifier) > 0:
                last_index = len(classifier) - 1
                last_layer = classifier[last_index]
                in_features = getattr(last_layer, "in_features", None)
                if in_features is not None:
                    classifier[last_index] = torch_nn.Linear(in_features, num_classes)
                    return
            in_features = getattr(classifier, "in_features", None)
            if in_features is not None:
                model.classifier = torch_nn.Linear(in_features, num_classes)
                return

        fc_layer = getattr(model, "fc", None)
        if fc_layer is not None and getattr(fc_layer, "in_features", None) is not None:
            model.fc = torch_nn.Linear(fc_layer.in_features, num_classes)
            return

        head_layer = getattr(model, "head", None)
        if head_layer is not None and getattr(head_layer, "in_features", None) is not None:
            model.head = torch_nn.Linear(head_layer.in_features, num_classes)
            return

        raise ValueError("Could not identify a supported classifier head for this model.")

    def _preprocess_image(
        self,
        *,
        image_path: str,
        torch_module: Any,
        manifest: dict[str, Any],
    ) -> Any:
        runtime_config = manifest.get("runtime", {}) if isinstance(manifest, dict) else {}
        image_size = runtime_config.get("input_size") or [224, 224]
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            image_size = [224, 224]

        normalization = runtime_config.get("normalization", {})
        mean = normalization.get("mean") or [0.485, 0.456, 0.406]
        std = normalization.get("std") or [0.229, 0.224, 0.225]

        with Image.open(Path(image_path)) as raw_image:
            image = ImageOps.exif_transpose(raw_image).convert("RGB")
            image = image.resize((int(image_size[0]), int(image_size[1])))

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        tensor = torch_module.from_numpy(array)

        mean_tensor = torch_module.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
        std_tensor = torch_module.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean_tensor) / std_tensor

        return tensor.unsqueeze(0)

    def _prediction(
        self,
        *,
        bundle: ModelArtifactBundle,
        label: str,
        confidence: float,
        top_k: list[PredictionCandidate],
        model_ready: bool,
        runtime_stage: str,
        note: str,
        provider: str = "phase6-torchvision-runtime",
    ) -> ServicePrediction:
        return ServicePrediction(
            label=label,
            confidence=confidence,
            top_k=top_k,
            model_ready=model_ready,
            model_key=bundle.model_key,
            artifact_count=len(bundle.weight_files),
            supported_labels=bundle.supported_labels,
            provider=provider,
            runtime_stage=runtime_stage,
            note=note,
        )
