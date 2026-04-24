from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    width: int = 0
    height: int = 0
    format: str = ""
    mode: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "mode": self.mode,
        }


@dataclass(slots=True)
class ImageQualityAssessment:
    status: str = "unknown"
    score: float | None = None
    issues: list[str] = field(default_factory=list)
    brightness_mean: float | None = None
    contrast_std: float | None = None
    sharpness_score: float | None = None
    colorfulness: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "score": self.score,
            "issues": self.issues,
            "brightness_mean": self.brightness_mean,
            "contrast_std": self.contrast_std,
            "sharpness_score": self.sharpness_score,
            "colorfulness": self.colorfulness,
        }


@dataclass(slots=True)
class LeafAssessment:
    detected: bool | None = None
    confidence: float = 0.0
    green_ratio: float = 0.0
    center_green_ratio: float = 0.0
    mean_saturation: float = 0.0
    provider: str = "phase2-heuristic-screening"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "confidence": self.confidence,
            "green_ratio": self.green_ratio,
            "center_green_ratio": self.center_green_ratio,
            "mean_saturation": self.mean_saturation,
            "provider": self.provider,
            "note": self.note,
        }


@dataclass(slots=True)
class HealthAssessment:
    is_healthy: bool | None = None
    confidence: float = 0.0
    lesion_ratio: float = 0.0
    provider: str = "phase2-heuristic-screening"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_healthy": self.is_healthy,
            "confidence": self.confidence,
            "lesion_ratio": self.lesion_ratio,
            "provider": self.provider,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class PredictionCandidate:
    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class ServicePrediction:
    label: str
    confidence: float
    top_k: list[PredictionCandidate] = field(default_factory=list)
    model_ready: bool = False
    model_key: str = ""
    artifact_count: int = 0
    supported_labels: list[str] = field(default_factory=list)
    provider: str = "placeholder"
    runtime_stage: str = "placeholder"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "top_k": [candidate.to_dict() for candidate in self.top_k],
            "model_ready": self.model_ready,
            "model_key": self.model_key,
            "artifact_count": self.artifact_count,
            "supported_labels": self.supported_labels,
            "provider": self.provider,
            "runtime_stage": self.runtime_stage,
            "note": self.note,
        }


@dataclass(slots=True)
class RAGDocument:
    title: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
            "score": self.score,
        }


@dataclass(slots=True)
class RAGResult:
    query: str
    summary: str
    documents: list[RAGDocument] = field(default_factory=list)
    knowledge_base_ready: bool = False
    retrieval_strategy: str = "none"
    corpus_size: int = 0
    filtered_count: int = 0
    chunk_count: int = 0
    embedding_backend: str = "none"
    generation_context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "summary": self.summary,
            "knowledge_base_ready": self.knowledge_base_ready,
            "retrieval_strategy": self.retrieval_strategy,
            "corpus_size": self.corpus_size,
            "filtered_count": self.filtered_count,
            "chunk_count": self.chunk_count,
            "embedding_backend": self.embedding_backend,
            "generation_context": self.generation_context,
            "documents": [document.to_dict() for document in self.documents],
        }


@dataclass(slots=True)
class DiseaseKnowledgeRecord:
    record_id: str
    crop: str
    disease: str
    summary: str = ""
    symptoms: list[str] = field(default_factory=list)
    causes: list[str] = field(default_factory=list)
    treatment: list[str] = field(default_factory=list)
    precautions: list[str] = field(default_factory=list)
    prevention: list[str] = field(default_factory=list)
    severity: str = ""
    source: str = ""
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def title(self) -> str:
        if self.crop and self.disease:
            return f"{self.crop} - {self.disease}"
        return self.disease or self.crop or self.record_id

    def search_text(self) -> str:
        parts = [
            self.crop,
            self.disease,
            self.summary,
            " ".join(self.aliases),
            " ".join(self.tags),
            " ".join(self.symptoms),
            " ".join(self.causes),
            " ".join(self.treatment),
            " ".join(self.precautions),
            " ".join(self.prevention),
            self.severity,
        ]
        return " ".join(part for part in parts if part)

    def content(self) -> str:
        sections: list[str] = []

        if self.summary:
            sections.append(f"Summary: {self.summary}")
        if self.symptoms:
            sections.append(f"Symptoms: {', '.join(self.symptoms)}")
        if self.causes:
            sections.append(f"Causes: {', '.join(self.causes)}")
        if self.treatment:
            sections.append(f"Treatment: {', '.join(self.treatment)}")
        if self.precautions:
            sections.append(f"Precautions: {', '.join(self.precautions)}")
        if self.prevention:
            sections.append(f"Prevention: {', '.join(self.prevention)}")
        if self.severity:
            sections.append(f"Severity: {self.severity}")

        return " | ".join(sections)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "crop": self.crop,
            "disease": self.disease,
            "summary": self.summary,
            "aliases": self.aliases,
            "symptoms": self.symptoms,
            "causes": self.causes,
            "treatment": self.treatment,
            "precautions": self.precautions,
            "prevention": self.prevention,
            "severity": self.severity,
            "tags": self.tags,
            **self.metadata,
        }


@dataclass(slots=True)
class ModelArtifactBundle:
    model_key: str
    model_dir: str
    runtime_ready: bool = False
    weight_files: list[str] = field(default_factory=list)
    label_map_path: str | None = None
    manifest_path: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    supported_labels: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "model_dir": self.model_dir,
            "runtime_ready": self.runtime_ready,
            "weight_files": self.weight_files,
            "label_map_path": self.label_map_path,
            "manifest_path": self.manifest_path,
            "manifest": self.manifest,
            "supported_labels": self.supported_labels,
            "note": self.note,
        }


@dataclass(slots=True)
class DiseaseRouteSelection:
    requested_crop: str = ""
    normalized_crop: str = ""
    resolved_model_key: str = ""
    route_found: bool = False
    strategy: str = "no_route"
    available_model_keys: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_crop": self.requested_crop,
            "normalized_crop": self.normalized_crop,
            "resolved_model_key": self.resolved_model_key,
            "route_found": self.route_found,
            "strategy": self.strategy,
            "available_model_keys": self.available_model_keys,
            "note": self.note,
        }


@dataclass(slots=True)
class VerificationCheck:
    name: str
    status: str
    details: str
    passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "details": self.details,
            "passed": self.passed,
        }


@dataclass(slots=True)
class VerificationResult:
    status: str = "pending"
    verified: bool = False
    score: float = 0.0
    summary: str = ""
    recommended_action: str = ""
    issues: list[str] = field(default_factory=list)
    matched_evidence_count: int = 0
    checks: list[VerificationCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "verified": self.verified,
            "score": self.score,
            "summary": self.summary,
            "recommended_action": self.recommended_action,
            "issues": self.issues,
            "matched_evidence_count": self.matched_evidence_count,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(slots=True)
class ImageAnalysisResult:
    image_ok: bool
    metadata: ImageMetadata = field(default_factory=ImageMetadata)
    quality: ImageQualityAssessment = field(default_factory=ImageQualityAssessment)
    leaf: LeafAssessment = field(default_factory=LeafAssessment)
    health: HealthAssessment = field(default_factory=HealthAssessment)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_ok": self.image_ok,
            "metadata": self.metadata.to_dict(),
            "quality": self.quality.to_dict(),
            "leaf": self.leaf.to_dict(),
            "health": self.health.to_dict(),
            "warnings": self.warnings,
        }
