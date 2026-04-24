from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from app.models.runtime import DiseaseKnowledgeRecord


@dataclass(slots=True)
class RAGChunk:
    chunk_id: str
    record_id: str
    crop: str
    disease: str
    title: str
    source: str
    section: str
    section_label: str
    text: str
    aliases: list[str]
    tags: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "record_id": self.record_id,
            "crop": self.crop,
            "disease": self.disease,
            "title": self.title,
            "source": self.source,
            "section": self.section,
            "section_label": self.section_label,
            "text": self.text,
            "aliases": self.aliases,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RAGChunk:
        return cls(
            chunk_id=str(payload.get("chunk_id", "")),
            record_id=str(payload.get("record_id", "")),
            crop=str(payload.get("crop", "")),
            disease=str(payload.get("disease", "")),
            title=str(payload.get("title", "")),
            source=str(payload.get("source", "")),
            section=str(payload.get("section", "")),
            section_label=str(payload.get("section_label", "")),
            text=str(payload.get("text", "")),
            aliases=[str(item).strip() for item in payload.get("aliases", []) if str(item).strip()],
            tags=[str(item).strip() for item in payload.get("tags", []) if str(item).strip()],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ChunkIndexArtifacts:
    signature: tuple[tuple[str, int, int], ...]
    chunks: list[RAGChunk]
    vectorizer: TfidfVectorizer
    tfidf_matrix: Any
    dense_embeddings: np.ndarray
    svd: TruncatedSVD | None
    embedding_backend: str

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


class ChunkedRAGIndexer:
    _manifest_name = "manifest.json"
    _chunks_name = "chunks.jsonl"
    _vectorizer_name = "vectorizer.joblib"
    _tfidf_name = "tfidf_matrix.joblib"
    _svd_name = "svd.joblib"
    _dense_name = "dense_embeddings.npy"
    _max_features = 12000
    _section_batch_size = 3

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        *,
        records: list[DiseaseKnowledgeRecord],
        signature: tuple[tuple[str, int, int], ...],
    ) -> ChunkIndexArtifacts:
        chunks = self._build_chunks(records)
        texts = [chunk.text for chunk in chunks]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=self._max_features,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)

        svd: TruncatedSVD | None = None
        embedding_backend = "tfidf_dense_v1"
        if tfidf_matrix.shape[0] >= 3 and tfidf_matrix.shape[1] >= 3:
            component_count = min(128, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
            if component_count >= 2:
                svd = TruncatedSVD(n_components=component_count, random_state=42)
                dense_embeddings = svd.fit_transform(tfidf_matrix).astype(np.float32, copy=False)
                embedding_backend = "tfidf_svd_dense_v1"
            else:
                dense_embeddings = tfidf_matrix.toarray().astype(np.float32, copy=False)
        else:
            dense_embeddings = tfidf_matrix.toarray().astype(np.float32, copy=False)

        dense_embeddings = self._normalize_dense(dense_embeddings)

        self._write_chunks(chunks)
        joblib.dump(vectorizer, self.output_dir / self._vectorizer_name)
        joblib.dump(tfidf_matrix, self.output_dir / self._tfidf_name)
        if svd is not None:
            joblib.dump(svd, self.output_dir / self._svd_name)
        elif (self.output_dir / self._svd_name).exists():
            (self.output_dir / self._svd_name).unlink()
        np.save(self.output_dir / self._dense_name, dense_embeddings, allow_pickle=False)
        self._write_manifest(
            signature=signature,
            chunk_count=len(chunks),
            embedding_backend=embedding_backend,
        )

        return ChunkIndexArtifacts(
            signature=signature,
            chunks=chunks,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            dense_embeddings=dense_embeddings,
            svd=svd,
            embedding_backend=embedding_backend,
        )

    def load(self) -> ChunkIndexArtifacts | None:
        manifest_path = self.output_dir / self._manifest_name
        chunks_path = self.output_dir / self._chunks_name
        vectorizer_path = self.output_dir / self._vectorizer_name
        tfidf_path = self.output_dir / self._tfidf_name
        dense_path = self.output_dir / self._dense_name

        required_paths = (
            manifest_path,
            chunks_path,
            vectorizer_path,
            tfidf_path,
            dense_path,
        )
        if any(not path.exists() for path in required_paths):
            return None

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = [
            RAGChunk.from_dict(json.loads(line))
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        svd_path = self.output_dir / self._svd_name
        return ChunkIndexArtifacts(
            signature=tuple(tuple(item) for item in manifest.get("knowledge_signature", [])),
            chunks=chunks,
            vectorizer=joblib.load(vectorizer_path),
            tfidf_matrix=joblib.load(tfidf_path),
            dense_embeddings=np.load(dense_path, allow_pickle=False),
            svd=joblib.load(svd_path) if svd_path.exists() else None,
            embedding_backend=str(manifest.get("embedding_backend", "tfidf_dense_v1")),
        )

    def index_is_current(
        self,
        *,
        signature: tuple[tuple[str, int, int], ...],
    ) -> bool:
        manifest_path = self.output_dir / self._manifest_name
        if not manifest_path.exists():
            return False

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return False

        stored_signature = tuple(tuple(item) for item in manifest.get("knowledge_signature", []))
        return stored_signature == signature

    def _write_manifest(
        self,
        *,
        signature: tuple[tuple[str, int, int], ...],
        chunk_count: int,
        embedding_backend: str,
    ) -> None:
        manifest = {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "knowledge_signature": [list(item) for item in signature],
            "chunk_count": chunk_count,
            "embedding_backend": embedding_backend,
            "vectorizer_backend": "sklearn_tfidf",
            "distance_metric": "cosine",
        }
        (self.output_dir / self._manifest_name).write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _write_chunks(self, chunks: list[RAGChunk]) -> None:
        payload = "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks)
        if payload:
            payload += "\n"
        (self.output_dir / self._chunks_name).write_text(payload, encoding="utf-8")

    def _build_chunks(self, records: list[DiseaseKnowledgeRecord]) -> list[RAGChunk]:
        chunks: list[RAGChunk] = []
        for record in records:
            record_metadata = dict(record.metadata)
            base_chunk_metadata = {
                "record_id": record.record_id,
                "crop": record.crop,
                "disease": record.disease,
                "aliases": list(record.aliases),
                "tags": list(record.tags),
                "source": record.source,
                **record_metadata,
            }
            title = record.title()

            chunk_texts = [
                ("overview", "Overview", self._build_overview_text(record)),
                ("full_record", "Record context", self._build_full_record_text(record)),
            ]
            chunk_texts.extend(
                self._chunk_list_field("symptoms", "Symptoms", record.symptoms)
            )
            chunk_texts.extend(self._chunk_list_field("causes", "Causes", record.causes))
            chunk_texts.extend(
                self._chunk_list_field("treatment", "Treatment", record.treatment)
            )
            chunk_texts.extend(
                self._chunk_list_field("precautions", "Precautions", record.precautions)
            )
            chunk_texts.extend(
                self._chunk_list_field("prevention", "Prevention", record.prevention)
            )
            chunk_texts.extend(self._extra_metadata_chunks(record))

            seen_texts: set[str] = set()
            section_counts: dict[str, int] = {}
            for section, section_label, body in chunk_texts:
                text = body.strip()
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)
                section_counts[section] = section_counts.get(section, 0) + 1
                chunk_suffix = section_counts[section]
                chunk_id = self._stable_chunk_id(
                    record_id=record.record_id,
                    section=section,
                    text=text,
                    suffix=chunk_suffix,
                )
                metadata = {
                    **base_chunk_metadata,
                    "chunk_id": chunk_id,
                    "chunk_section": section,
                    "chunk_section_label": section_label,
                }
                chunks.append(
                    RAGChunk(
                        chunk_id=chunk_id,
                        record_id=record.record_id,
                        crop=record.crop,
                        disease=record.disease,
                        title=title,
                        source=record.source,
                        section=section,
                        section_label=section_label,
                        text=text,
                        aliases=list(record.aliases),
                        tags=list(record.tags),
                        metadata=metadata,
                    )
                )

        return chunks

    def _build_overview_text(self, record: DiseaseKnowledgeRecord) -> str:
        lines = [f"Crop: {record.crop}", f"Disease: {record.disease}"]
        if record.summary:
            lines.append(f"Summary: {record.summary}")
        if record.severity:
            lines.append(f"Severity: {record.severity}")
        if record.aliases:
            lines.append(f"Aliases: {', '.join(record.aliases)}")
        if record.tags:
            lines.append(f"Tags: {', '.join(record.tags)}")
        return "\n".join(lines)

    def _build_full_record_text(self, record: DiseaseKnowledgeRecord) -> str:
        sections: list[str] = [
            f"Crop: {record.crop}",
            f"Disease: {record.disease}",
        ]
        if record.summary:
            sections.append(f"Summary: {record.summary}")
        if record.symptoms:
            sections.append(f"Symptoms: {'; '.join(record.symptoms)}")
        if record.causes:
            sections.append(f"Causes: {'; '.join(record.causes)}")
        if record.treatment:
            sections.append(f"Treatment: {'; '.join(record.treatment)}")
        if record.precautions:
            sections.append(f"Precautions: {'; '.join(record.precautions)}")
        if record.prevention:
            sections.append(f"Prevention: {'; '.join(record.prevention)}")
        if record.severity:
            sections.append(f"Severity: {record.severity}")
        extra_pairs = []
        for key in ("scientific_name", "disease_type", "yield_impact", "notes"):
            value = record.metadata.get(key)
            if isinstance(value, str) and value.strip():
                label = key.replace("_", " ").title()
                extra_pairs.append(f"{label}: {value.strip()}")
        sections.extend(extra_pairs)
        return "\n".join(sections)

    def _extra_metadata_chunks(self, record: DiseaseKnowledgeRecord) -> list[tuple[str, str, str]]:
        chunks: list[tuple[str, str, str]] = []
        metadata = record.metadata

        for key in ("spread_conditions", "organic_treatment", "affected_plant_parts"):
            values = metadata.get(key)
            if isinstance(values, list):
                label = key.replace("_", " ").title()
                chunks.extend(self._chunk_list_field(key, label, [str(item) for item in values]))

        for key in ("notes", "yield_impact", "scientific_name", "disease_type"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                label = key.replace("_", " ").title()
                chunks.append(
                    (
                        key,
                        label,
                        "\n".join(
                            [
                                f"Crop: {record.crop}",
                                f"Disease: {record.disease}",
                                f"{label}: {value.strip()}",
                            ]
                        ),
                    )
                )

        severity_levels = metadata.get("severity_levels")
        if isinstance(severity_levels, dict) and severity_levels:
            lines = [f"{str(level).title()}: {str(text).strip()}" for level, text in severity_levels.items() if str(text).strip()]
            if lines:
                chunks.append(
                    (
                        "severity_levels",
                        "Severity levels",
                        "\n".join(
                            [
                                f"Crop: {record.crop}",
                                f"Disease: {record.disease}",
                                "Severity levels:",
                                *lines,
                            ]
                        ),
                    )
                )

        references = metadata.get("references")
        if isinstance(references, list) and references:
            lines = [str(item).strip() for item in references if str(item).strip()]
            if lines:
                chunks.append(
                    (
                        "references",
                        "References",
                        "\n".join(
                            [
                                f"Crop: {record.crop}",
                                f"Disease: {record.disease}",
                                "References:",
                                *lines,
                            ]
                        ),
                    )
                )

        return chunks

    def _chunk_list_field(
        self,
        section: str,
        label: str,
        values: list[str],
    ) -> list[tuple[str, str, str]]:
        cleaned = [str(item).strip() for item in values if str(item).strip()]
        if not cleaned:
            return []

        chunks: list[tuple[str, str, str]] = []
        index = 0
        while index < len(cleaned):
            batch = cleaned[index:index + self._section_batch_size]
            body = "\n".join([f"{label}:", *[f"- {item}" for item in batch]])
            chunks.append((section, label, body))
            index += self._section_batch_size
        return chunks

    @staticmethod
    def _stable_chunk_id(
        *,
        record_id: str,
        section: str,
        text: str,
        suffix: int,
    ) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        return f"{record_id}_{section}_{suffix}_{digest}"

    @staticmethod
    def _normalize_dense(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    @staticmethod
    def hybrid_scores(
        *,
        query: str,
        index: ChunkIndexArtifacts,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query_tfidf = index.vectorizer.transform([query])
        lexical_scores = linear_kernel(query_tfidf, index.tfidf_matrix).ravel().astype(np.float32, copy=False)

        if index.svd is not None:
            query_dense = index.svd.transform(query_tfidf).astype(np.float32, copy=False)
        else:
            query_dense = query_tfidf.toarray().astype(np.float32, copy=False)

        query_dense = ChunkedRAGIndexer._normalize_dense(query_dense)
        semantic_scores = (index.dense_embeddings @ query_dense[0]).astype(np.float32, copy=False)
        hybrid = (0.55 * lexical_scores) + (0.45 * semantic_scores)
        return lexical_scores, semantic_scores, hybrid
