from __future__ import annotations

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    request_id: str
    crop_image_path: str
    diseased_image_path: str
    image_path: str
    original_filename: str
    content_type: str
    file_size_bytes: int
    image_ok: bool
    analysis: dict[str, Any]
    quality_score: float | None
    quality_issues: list[str]
    quality_gate_status: str
    is_leaf: bool | None
    leaf_confidence: float
    is_healthy: bool | None
    health_confidence: float
    crop_name: str
    crop_confidence: float
    crop_prediction: dict[str, Any]
    crop_gate_status: str
    disease_model_key: str
    disease_route: dict[str, Any]
    disease_name: str
    disease_confidence: float
    disease_prediction: dict[str, Any]
    disease_gate_status: str
    rag_query: str
    rag: dict[str, Any]
    verification: dict[str, Any]
    verified: bool
    warnings: list[str]
    final_answer: str


def build_initial_state(
    *,
    request_id: str,
    image_path: str,
    original_filename: str,
    content_type: str | None,
    file_size_bytes: int,
) -> PipelineState:
    return PipelineState(
        request_id=request_id,
        crop_image_path="",
        diseased_image_path="",
        image_path=image_path,
        original_filename=original_filename,
        content_type=content_type or "",
        file_size_bytes=file_size_bytes,
        analysis={
            "image_ok": False,
            "metadata": {"width": 0, "height": 0, "format": "", "mode": ""},
            "quality": {
                "status": "unknown",
                "score": None,
                "issues": [],
                "brightness_mean": None,
                "contrast_std": None,
                "sharpness_score": None,
                "colorfulness": None,
            },
            "leaf": {
                "detected": None,
                "confidence": 0.0,
                "green_ratio": 0.0,
                "center_green_ratio": 0.0,
                "mean_saturation": 0.0,
                "provider": "phase2-heuristic-screening",
                "note": "",
            },
            "health": {
                "is_healthy": None,
                "confidence": 0.0,
                "lesion_ratio": 0.0,
                "provider": "phase2-heuristic-screening",
                "note": "",
            },
            "warnings": [],
        },
        warnings=[],
        quality_issues=[],
        disease_route={
            "requested_crop": "",
            "normalized_crop": "",
            "resolved_model_key": "",
            "route_found": False,
            "strategy": "no_route",
            "available_model_keys": [],
            "note": "",
        },
        rag={
            "query": "",
            "summary": "",
            "knowledge_base_ready": False,
            "retrieval_strategy": "none",
            "corpus_size": 0,
            "filtered_count": 0,
            "documents": [],
        },
        verification={
            "status": "pending",
            "verified": False,
            "score": 0.0,
            "summary": "",
            "recommended_action": "",
            "issues": [],
            "matched_evidence_count": 0,
            "checks": [],
        },
    )
