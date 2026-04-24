from __future__ import annotations

from app.models.runtime import ModelArtifactBundle, ServicePrediction
from app.services.hierarchical_bundle_runtime import HierarchicalBundleRuntime
from app.services.image_classifier_runtime import (
    ImageClassifierRuntime,
    TorchvisionImageClassifierRuntime,
)
from app.services.model_registry import ModelRegistry


class CropPredictorService:
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
            or self.model_registry.crop_bundle().runtime_ready
        )

    def predict(self, image_path: str) -> ServicePrediction:
        if self.hierarchical_runtime and self.hierarchical_runtime.artifacts_detected():
            return self.hierarchical_runtime.predict_crop(image_path)

        bundle = self.model_registry.crop_bundle()

        if not bundle.runtime_ready:
            return self._artifact_missing_prediction(bundle)

        return self.runtime.predict(
            image_path=image_path,
            bundle=bundle,
            task_name="crop classification",
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
                "No trained crop model weights are loaded yet. Add artifacts under "
                "`models/crop/` to enable inference."
            ),
        )
