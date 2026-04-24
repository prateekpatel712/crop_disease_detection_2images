from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionCandidateModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    confidence: float


class PredictionModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    label: str
    confidence: float
    top_k: list[PredictionCandidateModel] = Field(default_factory=list)
    model_ready: bool = False
    model_key: str = ""
    artifact_count: int = 0
    supported_labels: list[str] = Field(default_factory=list)
    provider: str = "placeholder"
    runtime_stage: str = "placeholder"
    note: str = ""


class ImageMetadataModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    width: int = 0
    height: int = 0
    format: str = ""
    mode: str = ""


class ImageQualityModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str = "unknown"
    score: float | None = None
    issues: list[str] = Field(default_factory=list)
    brightness_mean: float | None = None
    contrast_std: float | None = None
    sharpness_score: float | None = None
    colorfulness: float | None = None


class LeafAssessmentModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    detected: bool | None = None
    confidence: float = 0.0
    green_ratio: float = 0.0
    center_green_ratio: float = 0.0
    mean_saturation: float = 0.0
    provider: str = "phase2-heuristic-screening"
    note: str = ""


class HealthAssessmentModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_healthy: bool | None = None
    confidence: float = 0.0
    lesion_ratio: float = 0.0
    provider: str = "phase2-heuristic-screening"
    note: str = ""


class ImageAnalysisModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_ok: bool = False
    metadata: ImageMetadataModel = Field(default_factory=ImageMetadataModel)
    quality: ImageQualityModel = Field(default_factory=ImageQualityModel)
    leaf: LeafAssessmentModel = Field(default_factory=LeafAssessmentModel)
    health: HealthAssessmentModel = Field(default_factory=HealthAssessmentModel)
    warnings: list[str] = Field(default_factory=list)


class DiseaseRouteModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requested_crop: str = ""
    normalized_crop: str = ""
    resolved_model_key: str = ""
    route_found: bool = False
    strategy: str = "no_route"
    available_model_keys: list[str] = Field(default_factory=list)
    note: str = ""


class RAGDocumentModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    content: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


class RAGContextModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = ""
    summary: str = ""
    knowledge_base_ready: bool = False
    retrieval_strategy: str = "none"
    corpus_size: int = 0
    filtered_count: int = 0
    chunk_count: int = 0
    embedding_backend: str = "none"
    generation_context: str = ""
    documents: list[RAGDocumentModel] = Field(default_factory=list)


class VerificationCheckModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    status: str
    details: str
    passed: bool | None = None


class VerificationModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str = "pending"
    verified: bool = False
    score: float = 0.0
    summary: str = ""
    recommended_action: str = ""
    issues: list[str] = Field(default_factory=list)
    matched_evidence_count: int = 0
    checks: list[VerificationCheckModel] = Field(default_factory=list)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    version: str
    uploads_dir: str
    crop_model_artifacts_detected: bool
    disease_model_artifacts_detected: bool
    rag_assets_detected: bool
    available_disease_models: list[str] = Field(default_factory=list)
    routing_config_detected: bool = False


class PipelineResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str
    image_path: str
    disease_name: str = ""
    final_answer: str


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str
    verdict: Literal["correct", "incorrect", "uncertain"]
    actual_crop_name: str = ""
    actual_disease_name: str = ""
    notes: str = ""
    reported_by: str = "user"


class FeedbackResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: str
    request_id: str
    message: str
