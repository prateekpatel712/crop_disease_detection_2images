from __future__ import annotations

from app.models.runtime import ModelArtifactBundle, ServicePrediction
from app.services.hierarchical_bundle_runtime import HierarchicalBundleRuntime
from app.services.image_classifier_runtime import (
    ImageClassifierRuntime,
    TorchvisionImageClassifierRuntime,
)
from app.services.model_registry import ModelRegistry


class DiseasePredictorService:
    def __init__(
        self,
        model_registry: ModelRegistry,
        runtime: ImageClassifierRuntime | None = None,
        hierarchical_runtime: HierarchicalBundleRuntime | None = None,
    ) -> None:
        self.model_registry = model_registry
        self.runtime = runtime or TorchvisionImageClassifierRuntime()
        self.hierarchical_runtime = hierarchical_runtime

    def model_artifacts_detected(self) -> bool:
        return bool(
            (self.hierarchical_runtime and self.hierarchical_runtime.artifacts_detected())
            or self.model_registry.available_disease_model_keys()
        )

    def predict(
        self,
        image_path: str,
        crop_name: str | None = None,
        model_key: str | None = None,
    ) -> ServicePrediction:
        if self.hierarchical_runtime and self.hierarchical_runtime.artifacts_detected():
            return self.hierarchical_runtime.predict_disease(
                image_path,
                crop_name=crop_name,
            )

        bundle = self.model_registry.disease_bundle(model_key)

        if not bundle.runtime_ready:
            return self._artifact_missing_prediction(bundle)

        return self.runtime.predict(
            image_path=image_path,
            bundle=bundle,
            task_name="disease classification",
        )

    def _artifact_missing_prediction(
        self,
        bundle: ModelArtifactBundle,
    ) -> ServicePrediction:
        return ServicePrediction(
            label="model_not_uploaded",
            confidence=0.0,
            model_ready=False,
            model_key=bundle.model_key,
            artifact_count=len(bundle.weight_files),
            supported_labels=bundle.supported_labels,
            provider="phase6-runtime-service",
            runtime_stage="awaiting_artifacts",
            note=(
                "No trained disease model is loaded for the selected route yet. "
                "Add artifacts under `models/disease/` to enable inference."
            ),
        )
