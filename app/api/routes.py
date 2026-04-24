from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.security import require_api_key
from app.graph.state import build_initial_state
from app.graph.workflow import invoke_dual_image_pipeline
from app.schemas.api import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    PipelineResponse,
)
from app.services.audit_logger import PipelineAuditLogger
from app.services.crop_predictor import CropPredictorService
from app.services.disease_router import DiseaseModelRouter
from app.services.disease_predictor import DiseasePredictorService
from app.services.file_storage import UploadStorageService
from app.services.hierarchical_bundle_runtime import HierarchicalBundleRuntime
from app.services.model_registry import ModelRegistry
from app.services.rag_service import RAGService

settings = get_settings()
router = APIRouter(
    prefix=settings.api_prefix,
    tags=["crop-disease-detection"],
    dependencies=[Depends(require_api_key)],
)
audit_logger = PipelineAuditLogger(
    prediction_log_path=settings.prediction_audit_log_path,
    feedback_log_path=settings.feedback_log_path,
)


def _default_prediction(note: str) -> dict:
    return {
        "label": "not_available",
        "confidence": 0.0,
        "top_k": [],
        "model_ready": False,
        "model_key": "",
        "artifact_count": 0,
        "supported_labels": [],
        "provider": "placeholder",
        "runtime_stage": "placeholder",
        "note": note,
    }


def _build_response(final_state: dict) -> PipelineResponse:
    disease_payload = final_state.get(
        "disease_prediction",
        _default_prediction("Disease prediction did not execute."),
    )
    disease_name = disease_payload.get("label", "")
    if disease_name in {"", "unknown", "not_available", "model_not_uploaded", "runtime_not_connected"}:
        disease_name = ""

    return PipelineResponse(
        request_id=final_state["request_id"],
        image_path=final_state.get(
            "diseased_image_path",
            final_state["image_path"],
        ),
        disease_name=disease_name,
        final_answer=final_state.get(
            "final_answer",
            "Pipeline completed without a final answer.",
        ),
    )


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    model_registry = ModelRegistry(
        crop_model_dir=settings.crop_model_dir,
        disease_model_dir=settings.disease_model_dir,
        metadata_dir=settings.model_metadata_dir,
    )
    hierarchical_runtime = HierarchicalBundleRuntime(settings.model_bundles_dir)
    crop_service = CropPredictorService(
        model_registry,
        hierarchical_runtime=hierarchical_runtime,
    )
    disease_service = DiseasePredictorService(
        model_registry,
        hierarchical_runtime=hierarchical_runtime,
    )
    disease_router = DiseaseModelRouter(model_registry, settings.model_metadata_dir)
    rag_service = RAGService(
        raw_dir=settings.rag_raw_dir,
        processed_dir=settings.rag_processed_dir,
        vector_store_dir=settings.rag_vector_store_dir,
    )
    available_disease_models = model_registry.available_disease_model_keys()
    for model_key in hierarchical_runtime.available_model_keys():
        if model_key not in available_disease_models:
            available_disease_models.append(model_key)

    return HealthResponse(
        status="ok",
        version=settings.project_version,
        uploads_dir=str(settings.uploads_dir),
        crop_model_artifacts_detected=crop_service.model_artifacts_detected(),
        disease_model_artifacts_detected=disease_service.model_artifacts_detected(),
        rag_assets_detected=rag_service.knowledge_assets_detected(),
        available_disease_models=available_disease_models,
        routing_config_detected=disease_router.model_registry.routing_config_detected(),
    )


@router.post("/predict", response_model=PipelineResponse)
async def predict(
    crop_image: UploadFile = File(..., description="Healthy or reference crop leaf image."),
    diseased_image: UploadFile = File(..., description="Diseased leaf image used for diagnosis."),
) -> PipelineResponse:
    storage_service = UploadStorageService(settings.uploads_dir)
    crop_original_filename = crop_image.filename or "crop-image"
    crop_content_type = crop_image.content_type
    diseased_original_filename = diseased_image.filename or "diseased-image"
    diseased_content_type = diseased_image.content_type

    try:
        crop_image_path, crop_size_bytes = await storage_service.save(crop_image)
        diseased_image_path, diseased_size_bytes = await storage_service.save(diseased_image)
    except Exception as exc:  # pragma: no cover - defensive API guard
        raise HTTPException(
            status_code=500,
            detail="Unable to store the uploaded images.",
        ) from exc

    request_id = uuid4().hex
    crop_state = build_initial_state(
        request_id=request_id,
        image_path=str(crop_image_path),
        original_filename=crop_original_filename,
        content_type=crop_content_type,
        file_size_bytes=crop_size_bytes,
    )
    crop_state["crop_image_path"] = str(crop_image_path)
    crop_state["diseased_image_path"] = str(diseased_image_path)

    diseased_state = build_initial_state(
        request_id=request_id,
        image_path=str(diseased_image_path),
        original_filename=diseased_original_filename,
        content_type=diseased_content_type,
        file_size_bytes=diseased_size_bytes,
    )
    diseased_state["crop_image_path"] = str(crop_image_path)
    diseased_state["diseased_image_path"] = str(diseased_image_path)

    try:
        final_state = invoke_dual_image_pipeline(crop_state, diseased_state)
    except Exception as exc:  # pragma: no cover - defensive API guard
        raise HTTPException(
            status_code=500,
            detail="Pipeline execution failed.",
        ) from exc

    return _build_response(final_state)


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    try:
        audit_logger.log_feedback_event(payload.model_dump())
    except OSError as exc:  # pragma: no cover - defensive API guard
        raise HTTPException(
            status_code=500,
            detail="Unable to record feedback for this request.",
        ) from exc

    return FeedbackResponse(
        status="logged",
        request_id=payload.request_id,
        message="Feedback was recorded for this prediction request.",
    )
