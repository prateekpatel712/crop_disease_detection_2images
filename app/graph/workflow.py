from __future__ import annotations

from functools import lru_cache
from typing import cast

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    build_rag_query,
    classify_crop,
    classify_disease,
    crop_confidence_gate,
    detect_healthy_or_diseased,
    detect_leaf,
    disease_confidence_gate,
    generate_response,
    log_result,
    quality_check,
    rerank_rag_docs,
    retrieve_rag_docs,
    route_disease_model,
    validate_image,
    verify_prediction,
)
from app.graph.state import PipelineState

_prediction_placeholders = {
    "",
    "unknown",
    "model_not_uploaded",
    "runtime_not_connected",
    "not_available",
}


def _has_concrete_prediction(prediction: dict | None) -> bool:
    payload = prediction or {}
    label = str(payload.get("label", "") or "").strip()
    return bool(label) and label not in _prediction_placeholders


def route_after_validation(state: PipelineState) -> str:
    return "quality_check" if state.get("image_ok", False) else "verify_prediction"


def route_after_quality_check(state: PipelineState) -> str:
    quality_status = state.get("quality_gate_status", "unknown")
    if not state.get("image_ok", False) or quality_status == "stop":
        return "verify_prediction"
    return "detect_leaf"


def route_after_leaf_detection(state: PipelineState) -> str:
    return (
        "detect_healthy_or_diseased"
        if state.get("is_leaf", False)
        else "verify_prediction"
    )


def route_after_crop_gate(state: PipelineState) -> str:
    crop_prediction = state.get("crop_prediction", {})
    if _has_concrete_prediction(crop_prediction) or (
        crop_prediction.get("model_key") == "hierarchical_bundle"
        and crop_prediction.get("model_ready", False)
    ):
        return "route_disease_model"
    return "verify_prediction"


def route_after_disease_gate(state: PipelineState) -> str:
    disease_prediction = state.get("disease_prediction", {})
    if _has_concrete_prediction(disease_prediction):
        return "build_rag_query"
    return "verify_prediction"


@lru_cache(maxsize=1)
def get_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("validate_image", validate_image)
    graph.add_node("quality_check", quality_check)
    graph.add_node("detect_leaf", detect_leaf)
    graph.add_node("detect_healthy_or_diseased", detect_healthy_or_diseased)
    graph.add_node("classify_crop", classify_crop)
    graph.add_node("crop_confidence_gate", crop_confidence_gate)
    graph.add_node("route_disease_model", route_disease_model)
    graph.add_node("classify_disease", classify_disease)
    graph.add_node("disease_confidence_gate", disease_confidence_gate)
    graph.add_node("build_rag_query", build_rag_query)
    graph.add_node("retrieve_rag_docs", retrieve_rag_docs)
    graph.add_node("rerank_rag_docs", rerank_rag_docs)
    graph.add_node("verify_prediction", verify_prediction)
    graph.add_node("generate_response", generate_response)
    graph.add_node("log_result", log_result)

    graph.add_edge(START, "validate_image")
    graph.add_conditional_edges(
        "validate_image",
        route_after_validation,
        {
            "quality_check": "quality_check",
            "verify_prediction": "verify_prediction",
        },
    )
    graph.add_conditional_edges(
        "quality_check",
        route_after_quality_check,
        {
            "detect_leaf": "detect_leaf",
            "verify_prediction": "verify_prediction",
        },
    )
    graph.add_conditional_edges(
        "detect_leaf",
        route_after_leaf_detection,
        {
            "detect_healthy_or_diseased": "detect_healthy_or_diseased",
            "verify_prediction": "verify_prediction",
        },
    )
    graph.add_edge("detect_healthy_or_diseased", "classify_crop")
    graph.add_edge("classify_crop", "crop_confidence_gate")
    graph.add_conditional_edges(
        "crop_confidence_gate",
        route_after_crop_gate,
        {
            "route_disease_model": "route_disease_model",
            "verify_prediction": "verify_prediction",
        },
    )
    graph.add_edge("route_disease_model", "classify_disease")
    graph.add_edge("classify_disease", "disease_confidence_gate")
    graph.add_conditional_edges(
        "disease_confidence_gate",
        route_after_disease_gate,
        {
            "build_rag_query": "build_rag_query",
            "verify_prediction": "verify_prediction",
        },
    )
    graph.add_edge("build_rag_query", "retrieve_rag_docs")
    graph.add_edge("retrieve_rag_docs", "rerank_rag_docs")
    graph.add_edge("rerank_rag_docs", "verify_prediction")
    graph.add_edge("verify_prediction", "generate_response")
    graph.add_edge("generate_response", "log_result")
    graph.add_edge("log_result", END)

    return graph.compile()


def invoke_pipeline(initial_state: PipelineState) -> PipelineState:
    compiled_graph = get_graph()
    return cast(PipelineState, compiled_graph.invoke(initial_state))


def _merge_state(state: PipelineState, updates: dict) -> PipelineState:
    merged = dict(state)
    merged.update(updates)
    return cast(PipelineState, merged)


def _run_node(state: PipelineState, node_fn) -> PipelineState:
    return _merge_state(state, node_fn(state))


def _merge_warnings(*warning_groups: list[str] | None) -> list[str]:
    merged: list[str] = []
    for group in warning_groups:
        for item in group or []:
            if item and item not in merged:
                merged.append(item)
    return merged


def _apply_crop_context(
    disease_state: PipelineState,
    crop_state: PipelineState,
) -> PipelineState:
    return _merge_state(
        disease_state,
        {
            "crop_image_path": crop_state.get(
                "crop_image_path",
                crop_state.get("image_path", ""),
            ),
            "diseased_image_path": disease_state.get(
                "diseased_image_path",
                disease_state.get("image_path", ""),
            ),
            "crop_name": crop_state.get("crop_name", ""),
            "crop_confidence": crop_state.get("crop_confidence", 0.0),
            "crop_prediction": crop_state.get("crop_prediction", {}),
            "crop_gate_status": crop_state.get("crop_gate_status", ""),
            "warnings": _merge_warnings(
                crop_state.get("warnings"),
                disease_state.get("warnings"),
            ),
        },
    )


def _finalize_blocked_state(
    base_state: PipelineState,
    *,
    status: str,
    summary: str,
    crop_state: PipelineState | None = None,
) -> PipelineState:
    state = dict(base_state)
    if crop_state is not None:
        state = dict(_apply_crop_context(cast(PipelineState, state), crop_state))

    state["verification"] = {
        "status": status,
        "verified": False,
        "score": 0.0,
        "summary": summary,
        "recommended_action": "Retry with clearer crop and diseased leaf images for a stronger diagnosis.",
        "issues": [summary],
        "matched_evidence_count": 0,
        "checks": [],
    }
    state["verified"] = False
    state["disease_prediction"] = {
        **state.get("disease_prediction", {}),
        "label": state.get("disease_prediction", {}).get("label", "not_available"),
        "note": summary,
    }
    state["warnings"] = _merge_warnings(
        state.get("warnings"),
        [summary],
    )

    finalized = cast(PipelineState, state)
    finalized = _run_node(finalized, generate_response)
    finalized = _run_node(finalized, log_result)
    return finalized


def _screen_leaf_image(
    state: PipelineState,
    *,
    blocked_input_summary: str,
    blocked_quality_summary: str,
    blocked_non_leaf_summary: str,
) -> tuple[PipelineState, str | None]:
    screened_state = _run_node(state, validate_image)
    validation_route = route_after_validation(screened_state)
    if validation_route == "verify_prediction":
        return (
            screened_state,
            "blocked_input",
        )

    screened_state = _run_node(screened_state, quality_check)
    quality_route = route_after_quality_check(screened_state)
    if quality_route == "verify_prediction":
        return (
            _merge_state(
                screened_state,
                {
                    "verification": {
                        "status": "blocked_quality",
                        "verified": False,
                        "score": 0.0,
                        "summary": blocked_quality_summary,
                        "recommended_action": "",
                        "issues": [blocked_quality_summary],
                        "matched_evidence_count": 0,
                        "checks": [],
                    }
                },
            ),
            "blocked_quality",
        )

    screened_state = _run_node(screened_state, detect_leaf)
    leaf_route = route_after_leaf_detection(screened_state)
    if leaf_route == "verify_prediction":
        return (
            _merge_state(
                screened_state,
                {
                    "verification": {
                        "status": "blocked_non_leaf",
                        "verified": False,
                        "score": 0.0,
                        "summary": blocked_non_leaf_summary,
                        "recommended_action": "",
                        "issues": [blocked_non_leaf_summary],
                        "matched_evidence_count": 0,
                        "checks": [],
                    }
                },
            ),
            "blocked_non_leaf",
        )

    screened_state = _run_node(screened_state, detect_healthy_or_diseased)
    return screened_state, None


def invoke_dual_image_pipeline(
    crop_state: PipelineState,
    diseased_state: PipelineState,
) -> PipelineState:
    crop_state, crop_block = _screen_leaf_image(
        crop_state,
        blocked_input_summary=(
            "The crop reference image is invalid. Upload a clear crop leaf image first."
        ),
        blocked_quality_summary=(
            "The crop reference image is too unclear for crop identification. Upload a sharper crop leaf image."
        ),
        blocked_non_leaf_summary=(
            "The crop reference image does not look like a leaf. Upload a clear crop leaf image first."
        ),
    )
    if crop_block == "blocked_input":
        return _finalize_blocked_state(
            diseased_state,
            status="blocked_input",
            summary="The crop reference image is invalid. Upload a clear crop leaf image first.",
            crop_state=crop_state,
        )
    if crop_block == "blocked_quality":
        return _finalize_blocked_state(
            diseased_state,
            status="blocked_quality",
            summary="The crop reference image is too unclear for crop identification. Upload a sharper crop leaf image.",
            crop_state=crop_state,
        )
    if crop_block == "blocked_non_leaf":
        return _finalize_blocked_state(
            diseased_state,
            status="blocked_non_leaf",
            summary="The crop reference image does not look like a leaf. Upload a clear crop leaf image first.",
            crop_state=crop_state,
        )

    crop_state = _run_node(crop_state, classify_crop)
    crop_state = _run_node(crop_state, crop_confidence_gate)

    if route_after_crop_gate(crop_state) != "route_disease_model":
        crop_name = str(crop_state.get("crop_name", "") or "").strip()
        if crop_name:
            summary = (
                f"The crop reference image was mapped to {crop_name}, but the confidence was too low to continue safely."
            )
        else:
            summary = (
                "The system could not confidently identify the crop from the reference image."
            )
        return _finalize_blocked_state(
            diseased_state,
            status="caution",
            summary=summary,
            crop_state=crop_state,
        )

    diseased_state, diseased_block = _screen_leaf_image(
        diseased_state,
        blocked_input_summary=(
            "The diseased image is invalid. Upload a clear diseased leaf image for diagnosis."
        ),
        blocked_quality_summary=(
            "The diseased image is too unclear for diagnosis. Upload a sharper diseased leaf image."
        ),
        blocked_non_leaf_summary=(
            "The diseased image does not look like a leaf. Upload a clear diseased leaf image for diagnosis."
        ),
    )
    if diseased_block == "blocked_input":
        return _finalize_blocked_state(
            diseased_state,
            status="blocked_input",
            summary="The diseased image is invalid. Upload a clear diseased leaf image for diagnosis.",
            crop_state=crop_state,
        )
    if diseased_block == "blocked_quality":
        return _finalize_blocked_state(
            diseased_state,
            status="blocked_quality",
            summary="The diseased image is too unclear for diagnosis. Upload a sharper diseased leaf image.",
            crop_state=crop_state,
        )
    if diseased_block == "blocked_non_leaf":
        return _finalize_blocked_state(
            diseased_state,
            status="blocked_non_leaf",
            summary="The diseased image does not look like a leaf. Upload a clear diseased leaf image for diagnosis.",
            crop_state=crop_state,
        )

    diseased_state = _apply_crop_context(diseased_state, crop_state)
    diseased_state = _run_node(diseased_state, route_disease_model)
    diseased_state = _run_node(diseased_state, classify_disease)
    diseased_state = _run_node(diseased_state, disease_confidence_gate)

    if route_after_disease_gate(diseased_state) == "build_rag_query":
        diseased_state = _run_node(diseased_state, build_rag_query)
        diseased_state = _run_node(diseased_state, retrieve_rag_docs)
        diseased_state = _run_node(diseased_state, rerank_rag_docs)

    diseased_state = _run_node(diseased_state, verify_prediction)
    diseased_state = _run_node(diseased_state, generate_response)
    diseased_state = _run_node(diseased_state, log_result)
    return diseased_state
