from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.models.runtime import DiseaseKnowledgeRecord, RAGDocument, RAGResult
from app.services.rag_indexing import ChunkIndexArtifacts, ChunkedRAGIndexer, RAGChunk


class RAGService:
    _ignored_filenames = {"readme.md"}
    _ignored_markers = (".example.",)
    _structured_suffixes = {".json", ".jsonl"}
    _placeholder_tokens = {
        "",
        "unknown",
        "model_not_uploaded",
        "runtime_not_connected",
        "not_available",
    }

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        vector_store_dir: Path,
    ) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.vector_store_dir = vector_store_dir
        self._cached_signature: tuple[tuple[str, int, int], ...] | None = None
        self._cached_records: list[DiseaseKnowledgeRecord] = []
        self._cached_index_signature: tuple[tuple[str, int, int], ...] | None = None
        self._cached_chunk_index: ChunkIndexArtifacts | None = None
        self._indexer = ChunkedRAGIndexer(vector_store_dir)

    def knowledge_assets_detected(self) -> bool:
        directories = (self.raw_dir, self.processed_dir, self.vector_store_dir)
        return any(
            self._is_knowledge_file(file_path)
            for directory in directories
            for file_path in directory.iterdir()
        )

    def ensure_index_ready(self) -> bool:
        records = self._load_records()
        if not records:
            return False
        self._load_or_build_index(records)
        return True

    def retrieve(
        self,
        *,
        query: str,
        crop_name: str | None,
        disease_name: str | None,
    ) -> RAGResult:
        records = self._load_records()
        normalized_crop = self._normalize_filter_value(crop_name)
        normalized_disease = self._normalize_filter_value(disease_name)
        effective_query = query.strip()
        if not effective_query:
            fallback_terms = [
                part
                for part, normalized in (
                    (crop_name or "", normalized_crop),
                    (disease_name or "", normalized_disease),
                )
                if part and normalized
            ]
            effective_query = " ".join(fallback_terms).strip()

        if not records:
            return RAGResult(
                query=effective_query,
                summary=(
                    "No structured RAG records are loaded yet. Add JSON or JSONL "
                    "disease documents under rag_data/processed/ or rag_data/raw/."
                ),
                documents=[],
                knowledge_base_ready=False,
                retrieval_strategy="empty_corpus",
                corpus_size=0,
                filtered_count=0,
                chunk_count=0,
                embedding_backend="none",
                generation_context="",
            )

        if not effective_query and not normalized_crop and not normalized_disease:
            return RAGResult(
                query="",
                summary=(
                    "Structured RAG records are loaded, but retrieval was skipped "
                    "because no crop or disease target is available yet."
                ),
                documents=[],
                knowledge_base_ready=True,
                retrieval_strategy="no_target",
                corpus_size=len(records),
                filtered_count=0,
                chunk_count=0,
                embedding_backend="none",
                generation_context="",
            )

        candidates, strategy = self._select_candidates(
            records=records,
            normalized_crop=normalized_crop,
            normalized_disease=normalized_disease,
        )
        candidate_record_ids = {record.record_id for record in candidates}
        record_lookup = {record.record_id: record for record in records}

        index = self._load_or_build_index(records)
        lexical_scores, semantic_scores, hybrid_scores = ChunkedRAGIndexer.hybrid_scores(
            query=effective_query,
            index=index,
        )

        scored_documents: list[RAGDocument] = []
        candidate_chunk_count = 0
        for index_position, chunk in enumerate(index.chunks):
            if chunk.record_id not in candidate_record_ids:
                continue
            candidate_chunk_count += 1
            record = record_lookup.get(chunk.record_id)
            if record is None:
                continue
            scored_documents.append(
                self._chunk_to_document(
                    chunk=chunk,
                    record=record,
                    lexical_score=float(lexical_scores[index_position]),
                    semantic_score=float(semantic_scores[index_position]),
                    hybrid_score=float(hybrid_scores[index_position]),
                    normalized_crop=normalized_crop,
                    normalized_disease=normalized_disease,
                    query=effective_query,
                )
            )

        scored_documents.sort(key=lambda document: document.score or 0.0, reverse=True)
        top_documents = scored_documents[:8]
        knowledge_base_ready = bool(top_documents)

        if top_documents:
            top_title = top_documents[0].title
            summary = (
                f"Retrieved {len(top_documents)} chunk(s) from {candidate_chunk_count} "
                f"candidate chunk(s) across {len(records)} record(s) using `{strategy}`. "
                f"Top match: {top_title}."
            )
        else:
            summary = (
                f"Structured RAG records are loaded, but no useful match was found "
                f"for the current query using `{strategy}`."
            )

        return RAGResult(
            query=effective_query,
            summary=summary,
            documents=top_documents,
            knowledge_base_ready=knowledge_base_ready,
            retrieval_strategy=strategy,
            corpus_size=len(records),
            filtered_count=candidate_chunk_count,
            chunk_count=index.chunk_count,
            embedding_backend=index.embedding_backend,
            generation_context=self._build_generation_context(top_documents),
        )

    def rerank(
        self,
        *,
        rag_payload: dict[str, Any],
        query: str,
        crop_name: str | None,
        disease_name: str | None,
    ) -> dict[str, Any]:
        documents = list(rag_payload.get("documents", []))
        if not documents:
            rag_payload["generation_context"] = ""
            return rag_payload

        normalized_crop = self._normalize_filter_value(crop_name)
        normalized_disease = self._normalize_filter_value(disease_name)
        query_terms = self._tokenize(query)

        for document in documents:
            metadata = dict(document.get("metadata", {}))
            base_score = float(document.get("score") or metadata.get("retrieval_score", 0.0))
            rerank_score = base_score

            crop_value = self._normalize_key(str(metadata.get("crop", "")))
            disease_value = self._normalize_key(str(metadata.get("disease", "")))
            alias_values = {
                self._normalize_key(str(alias))
                for alias in metadata.get("aliases", [])
                if str(alias).strip()
            }

            if normalized_crop and crop_value == normalized_crop:
                rerank_score += 0.20

            if normalized_disease and (
                disease_value == normalized_disease or normalized_disease in alias_values
            ):
                rerank_score += 0.30

            title_terms = self._tokenize(
                " ".join(
                    [
                        document.get("title", ""),
                        str(metadata.get("disease", "")),
                        str(metadata.get("chunk_section_label", "")),
                        " ".join(str(alias) for alias in metadata.get("aliases", [])),
                    ]
                )
            )
            if query_terms:
                rerank_score += 0.15 * (
                    len(query_terms & title_terms) / max(len(query_terms), 1)
                )

            section_key = str(metadata.get("chunk_section", "")).strip().lower()
            if section_key in {"overview", "full_record", "symptoms", "treatment"}:
                rerank_score += 0.03

            document["score"] = round(rerank_score, 4)
            metadata["rerank_score"] = round(rerank_score, 4)
            document["metadata"] = metadata

        documents.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        rag_payload["documents"] = documents[:5]
        rag_payload["summary"] = (
            f"Retrieved and reranked {len(rag_payload['documents'])} chunk(s) "
            f"using `{rag_payload.get('retrieval_strategy', 'unknown')}`."
        )
        rag_payload["generation_context"] = self._build_generation_context_from_payload(
            rag_payload["documents"]
        )
        return rag_payload

    def _load_records(self) -> list[DiseaseKnowledgeRecord]:
        knowledge_files = self._structured_knowledge_files()
        signature = tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in knowledge_files
        )

        if signature == self._cached_signature:
            return list(self._cached_records)

        records: list[DiseaseKnowledgeRecord] = []
        for path in knowledge_files:
            records.extend(self._parse_knowledge_file(path))

        self._cached_signature = signature
        self._cached_records = records
        self._cached_index_signature = None
        self._cached_chunk_index = None
        return list(records)

    def _load_or_build_index(self, records: list[DiseaseKnowledgeRecord]) -> ChunkIndexArtifacts:
        signature = self._cached_signature or ()
        if (
            self._cached_chunk_index is not None
            and self._cached_index_signature == signature
        ):
            return self._cached_chunk_index

        if self._indexer.index_is_current(signature=signature):
            index = self._indexer.load()
            if index is None:
                index = self._indexer.build(records=records, signature=signature)
        else:
            index = self._indexer.build(records=records, signature=signature)

        self._cached_index_signature = signature
        self._cached_chunk_index = index
        return index

    def _structured_knowledge_files(self) -> list[Path]:
        candidates: list[Path] = []
        for directory in (self.processed_dir, self.raw_dir):
            for path in sorted(directory.iterdir(), key=lambda item: str(item)):
                if not self._is_knowledge_file(path):
                    continue
                if path.suffix.lower() not in self._structured_suffixes:
                    continue
                candidates.append(path)
        return candidates

    def _is_knowledge_file(self, path: Path) -> bool:
        if not path.is_file():
            return False

        filename = path.name.lower()
        if filename in self._ignored_filenames:
            return False

        return not any(marker in filename for marker in self._ignored_markers)

    def _parse_knowledge_file(self, path: Path) -> list[DiseaseKnowledgeRecord]:
        try:
            if path.suffix.lower() == ".jsonl":
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("documents"), list):
                    rows = payload["documents"]
                elif isinstance(payload, list):
                    rows = payload
                else:
                    rows = [payload]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return []

        records: list[DiseaseKnowledgeRecord] = []
        for index, row in enumerate(rows):
            record = self._row_to_record(row=row, path=path, index=index)
            if record is not None:
                records.append(record)
        return records

    def _row_to_record(
        self,
        *,
        row: Any,
        path: Path,
        index: int,
    ) -> DiseaseKnowledgeRecord | None:
        if not isinstance(row, dict):
            return None

        crop = self._as_text(row.get("crop"))
        disease = self._as_text(row.get("disease"))
        if not crop and not disease:
            return None

        record_id = self._as_text(row.get("id")) or f"{path.stem}:{index}"
        reserved_keys = {
            "id",
            "crop",
            "disease",
            "summary",
            "symptoms",
            "causes",
            "treatment",
            "precautions",
            "prevention",
            "severity",
            "source",
            "aliases",
            "tags",
        }

        extra_metadata = {
            key: value
            for key, value in row.items()
            if key not in reserved_keys
        }
        extra_metadata["source_file"] = str(path)

        return DiseaseKnowledgeRecord(
            record_id=record_id,
            crop=crop,
            disease=disease,
            summary=self._as_text(row.get("summary")),
            symptoms=self._as_list(row.get("symptoms")),
            causes=self._as_list(row.get("causes")),
            treatment=self._as_list(row.get("treatment")),
            precautions=self._as_list(row.get("precautions")),
            prevention=self._as_list(row.get("prevention")),
            severity=self._as_text(row.get("severity")),
            source=self._as_text(row.get("source")) or str(path),
            aliases=self._as_list(row.get("aliases")),
            tags=self._as_list(row.get("tags")),
            metadata=extra_metadata,
        )

    def _select_candidates(
        self,
        *,
        records: list[DiseaseKnowledgeRecord],
        normalized_crop: str,
        normalized_disease: str,
    ) -> tuple[list[DiseaseKnowledgeRecord], str]:
        if normalized_crop and normalized_disease:
            exact = [
                record
                for record in records
                if self._crop_matches(record, normalized_crop)
                and self._disease_matches(record, normalized_disease)
            ]
            if exact:
                return exact, "exact_crop_disease"

        if normalized_crop:
            crop_filtered = [
                record
                for record in records
                if self._crop_matches(record, normalized_crop)
            ]
            if crop_filtered:
                return crop_filtered, "crop_filter"

        if normalized_disease:
            disease_filtered = [
                record
                for record in records
                if self._disease_matches(record, normalized_disease)
            ]
            if disease_filtered:
                return disease_filtered, "disease_filter"

        return records, "global_fallback"

    def _chunk_to_document(
        self,
        *,
        chunk: RAGChunk,
        record: DiseaseKnowledgeRecord,
        lexical_score: float,
        semantic_score: float,
        hybrid_score: float,
        normalized_crop: str,
        normalized_disease: str,
        query: str,
    ) -> RAGDocument:
        retrieval_score = self._score_chunk(
            chunk=chunk,
            record=record,
            query=query,
            lexical_score=lexical_score,
            semantic_score=semantic_score,
            hybrid_score=hybrid_score,
            normalized_crop=normalized_crop,
            normalized_disease=normalized_disease,
        )

        metadata = record.to_metadata()
        metadata.update(
            {
                "chunk_id": chunk.chunk_id,
                "chunk_section": chunk.section,
                "chunk_section_label": chunk.section_label,
                "chunk_text": chunk.text,
                "lexical_score": round(lexical_score, 4),
                "semantic_score": round(semantic_score, 4),
                "retrieval_score": round(retrieval_score, 4),
            }
        )

        return RAGDocument(
            title=record.title(),
            content=chunk.text,
            source=record.source,
            metadata=metadata,
            score=round(retrieval_score, 4),
        )

    def _score_chunk(
        self,
        *,
        chunk: RAGChunk,
        record: DiseaseKnowledgeRecord,
        query: str,
        lexical_score: float,
        semantic_score: float,
        hybrid_score: float,
        normalized_crop: str,
        normalized_disease: str,
    ) -> float:
        query_terms = self._tokenize(query)
        title_terms = self._tokenize(
            " ".join(
                [
                    record.title(),
                    chunk.section_label,
                    " ".join(record.aliases),
                ]
            )
        )

        score = hybrid_score

        if normalized_crop and self._crop_matches(record, normalized_crop):
            score += 0.20

        if normalized_disease and self._disease_matches(record, normalized_disease):
            score += 0.30

        if query_terms:
            score += 0.10 * (len(query_terms & title_terms) / max(len(query_terms), 1))

        if chunk.section in {"overview", "full_record", "symptoms", "treatment"}:
            score += 0.03

        if record.summary:
            score += 0.03
        if record.treatment or record.precautions or record.prevention:
            score += 0.04

        return score

    def _build_generation_context(self, documents: list[RAGDocument]) -> str:
        return self._build_generation_context_from_payload(
            [document.to_dict() for document in documents]
        )

    def _build_generation_context_from_payload(self, documents: list[dict[str, Any]]) -> str:
        sections: list[str] = []
        for document in documents[:5]:
            metadata = document.get("metadata", {})
            section_label = (
                str(metadata.get("chunk_section_label", "")).strip()
                or str(metadata.get("chunk_section", "")).strip()
                or "Context"
            )
            sections.append(
                "\n".join(
                    [
                        f"Title: {document.get('title', '')}",
                        f"Section: {section_label}",
                        f"Content: {document.get('content', '')}",
                    ]
                )
            )
        return "\n\n---\n\n".join(part for part in sections if part.strip())

    def _crop_matches(self, record: DiseaseKnowledgeRecord, normalized_crop: str) -> bool:
        return self._normalize_key(record.crop) == normalized_crop

    def _disease_matches(
        self,
        record: DiseaseKnowledgeRecord,
        normalized_disease: str,
    ) -> bool:
        disease_key = self._normalize_key(record.disease)
        alias_keys = {
            self._normalize_key(alias)
            for alias in record.aliases
            if alias.strip()
        }
        return normalized_disease == disease_key or normalized_disease in alias_keys

    def _normalize_filter_value(self, value: str | None) -> str:
        normalized = self._normalize_key(value or "")
        if normalized in self._placeholder_tokens:
            return ""
        return normalized

    @staticmethod
    def _normalize_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 1
        }

    @staticmethod
    def _as_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @classmethod
    def _as_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [cls._as_text(item) for item in value if cls._as_text(item)]
        text_value = cls._as_text(value)
        return [text_value] if text_value else []
