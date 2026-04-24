from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import get_settings
from app.graph.state import PipelineState
from app.services.audit_logger import PipelineAuditLogger
from app.services.crop_predictor import CropPredictorService
from app.services.disease_router import DiseaseModelRouter
from app.services.disease_predictor import DiseasePredictorService
from app.services.hierarchical_bundle_runtime import HierarchicalBundleRuntime
from app.services.image_analysis import ImageAnalysisService
from app.services.image_validation import ImageValidationService
from app.services.model_registry import ModelRegistry
from app.services.prediction_verifier import PredictionVerificationService
from app.services.rag_service import RAGService
from app.services.response_builder import ResponseBuilder

logger = logging.getLogger(__name__)
settings = get_settings()
_prediction_placeholders = {
    "unknown",
    "model_not_uploaded",
    "runtime_not_connected",
    "not_available",
}

validation_service = ImageValidationService(settings)
image_analysis_service = ImageAnalysisService(settings)
model_registry = ModelRegistry(
    crop_model_dir=settings.crop_model_dir,
    disease_model_dir=settings.disease_model_dir,
    metadata_dir=settings.model_metadata_dir,
)
hierarchical_runtime = HierarchicalBundleRuntime(settings.model_bundles_dir)
crop_predictor_service = CropPredictorService(
    model_registry,
    hierarchical_runtime=hierarchical_runtime,
)
disease_router_service = DiseaseModelRouter(model_registry, settings.model_metadata_dir)
disease_predictor_service = DiseasePredictorService(
    model_registry,
    hierarchical_runtime=hierarchical_runtime,
)
rag_service = RAGService(
    raw_dir=settings.rag_raw_dir,
    processed_dir=settings.rag_processed_dir,
    vector_store_dir=settings.rag_vector_store_dir,
)
prediction_verifier = PredictionVerificationService()
audit_logger = PipelineAuditLogger(
    prediction_log_path=settings.prediction_audit_log_path,
    feedback_log_path=settings.feedback_log_path,
)
response_builder = ResponseBuilder()


def _merge_warnings(existing: list[str] | None, new_items: list[str]) -> list[str]:
    ordered = list(existing or [])
    for item in new_items:
        if item not in ordered:
            ordered.append(item)
    return ordered


def validate_image(state: PipelineState) -> PipelineState:
    outcome = validation_service.validate(
        filename=state.get("original_filename", ""),
        content_type=state.get("content_type", ""),
        size_bytes=state.get("file_size_bytes", 0),
    )
    return {
        "image_ok": outcome.image_ok,
        "warnings": _merge_warnings(state.get("warnings"), outcome.issues),
    }


def quality_check(state: PipelineState) -> PipelineState:
    analysis = image_analysis_service.analyze(state.get("image_path", ""))
    analysis_payload = analysis.to_dict()
    quality = analysis_payload["quality"]

    return {
        "analysis": analysis_payload,
        "image_ok": state.get("image_ok", False) and analysis.image_ok,
        "quality_score": quality["score"],
        "quality_issues": quality["issues"],
        "quality_gate_status": quality["status"],
        "warnings": _merge_warnings(state.get("warnings"), analysis_payload["warnings"]),
    }


def detect_leaf(state: PipelineState) -> PipelineState:
    leaf = state.get("analysis", {}).get("leaf", {})
    note = leaf.get("note", "")
    return {
        "is_leaf": leaf.get("detected"),
        "leaf_confidence": float(leaf.get("confidence", 0.0)),
        "warnings": _merge_warnings(state.get("warnings"), [note] if note else []),
    }


def detect_healthy_or_diseased(state: PipelineState) -> PipelineState:
    health = state.get("analysis", {}).get("health", {})
    note = health.get("note", "")
    return {
        "is_healthy": health.get("is_healthy"),
        "health_confidence": float(health.get("confidence", 0.0)),
        "warnings": _merge_warnings(state.get("warnings"), [note] if note else []),
    }


def classify_crop(state: PipelineState) -> PipelineState:
    prediction = crop_predictor_service.predict(state.get("image_path", ""))
    return {
        "crop_name": prediction.label,
        "crop_confidence": prediction.confidence,
        "crop_prediction": prediction.to_dict(),
    }


def crop_confidence_gate(state: PipelineState) -> PipelineState:
    prediction = state.get("crop_prediction", {})
    confidence = float(prediction.get("confidence", 0.0))

    if not prediction.get("model_ready", False):
        status = "model_missing"
    elif confidence >= settings.min_crop_confidence:
        status = "proceed"
    elif confidence >= settings.caution_crop_confidence:
        status = "caution"
    else:
        status = "stop"

    return {"crop_gate_status": status}


def route_disease_model(state: PipelineState) -> PipelineState:
    if hierarchical_runtime.artifacts_detected():
        route_payload = hierarchical_runtime.route_payload(state.get("crop_name"))
        note = route_payload.get("note", "")
        return {
            "disease_model_key": route_payload.get("resolved_model_key", ""),
            "disease_route": route_payload,
            "warnings": _merge_warnings(state.get("warnings"), [note] if note else []),
        }

    route = disease_router_service.resolve(state.get("crop_name"))
    route_payload = route.to_dict()
    note = route_payload.get("note", "")
    return {
        "disease_model_key": route.resolved_model_key,
        "disease_route": route_payload,
        "warnings": _merge_warnings(state.get("warnings"), [note] if note else []),
    }


def classify_disease(state: PipelineState) -> PipelineState:
    prediction = disease_predictor_service.predict(
        state.get("image_path", ""),
        state.get("crop_name"),
        state.get("disease_model_key"),
    )
    return {
        "disease_name": prediction.label,
        "disease_confidence": prediction.confidence,
        "disease_prediction": prediction.to_dict(),
    }


def disease_confidence_gate(state: PipelineState) -> PipelineState:
    prediction = state.get("disease_prediction", {})
    confidence = float(prediction.get("confidence", 0.0))

    if not prediction.get("model_ready", False):
        status = "model_missing"
    elif confidence >= settings.min_disease_confidence:
        status = "proceed"
    elif confidence >= settings.caution_disease_confidence:
        status = "caution"
    else:
        status = "stop"

    return {"disease_gate_status": status}


def build_rag_query(state: PipelineState) -> PipelineState:
    crop_name = state.get("crop_name", "")
    disease_name = state.get("disease_name", "")

    meaningful_parts = [
        part
        for part in (crop_name, disease_name)
        if part and part not in _prediction_placeholders
    ]

    if meaningful_parts:
        query = f"{' '.join(meaningful_parts)} symptoms treatment precautions prevention"
    else:
        query = ""

    return {"rag_query": query}


def retrieve_rag_docs(state: PipelineState) -> PipelineState:
    rag_result = rag_service.retrieve(
        query=state.get("rag_query", ""),
        crop_name=state.get("crop_name"),
        disease_name=state.get("disease_name"),
    )
    return {"rag": rag_result.to_dict()}


def rerank_rag_docs(state: PipelineState) -> PipelineState:
    rag_payload = dict(state.get("rag", {}))
    reranked = rag_service.rerank(
        rag_payload=rag_payload,
        query=state.get("rag_query", ""),
        crop_name=state.get("crop_name"),
        disease_name=state.get("disease_name"),
    )
    return {"rag": reranked}


def verify_prediction(state: PipelineState) -> PipelineState:
    verification = prediction_verifier.verify(state).to_dict()
    merged_warnings = _merge_warnings(
        state.get("warnings"),
        verification.get("issues", []),
    )

    return {
        "verification": verification,
        "verified": bool(verification.get("verified", False)),
        "warnings": merged_warnings,
    }


def generate_response(state: PipelineState) -> PipelineState:
    return {"final_answer": response_builder.build(state)}


def log_result(state: PipelineState) -> PipelineState:
    logger.info(
        "request_id=%s image_ok=%s crop=%s disease_route=%s disease=%s verified=%s verification_status=%s",
        state.get("request_id"),
        state.get("image_ok"),
        state.get("crop_name"),
        state.get("disease_model_key"),
        state.get("disease_name"),
        state.get("verified"),
        state.get("verification", {}).get("status", "pending"),
    )

    image_path = state.get("image_path")
    if image_path and not Path(image_path).exists():
        logger.warning("Uploaded image path is missing on disk: %s", image_path)

    try:
        audit_logger.log_prediction_event(state)
    except OSError:  # pragma: no cover - defensive logging guard
        logger.exception(
            "Prediction audit logging failed for request_id=%s",
            state.get("request_id"),
        )

    return {}
