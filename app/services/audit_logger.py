from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PipelineAuditLogger:
    def __init__(
        self,
        *,
        prediction_log_path: Path,
        feedback_log_path: Path,
    ) -> None:
        self.prediction_log_path = prediction_log_path
        self.feedback_log_path = feedback_log_path
        self.prediction_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_prediction_event(self, state: dict[str, Any]) -> None:
        payload = {
            "timestamp": self._timestamp(),
            "request_id": state.get("request_id", ""),
            "crop_image_path": state.get("crop_image_path", ""),
            "diseased_image_path": state.get("diseased_image_path", ""),
            "image_path": state.get("image_path", ""),
            "image_ok": state.get("image_ok", False),
            "quality_status": state.get("quality_gate_status", ""),
            "is_leaf": state.get("is_leaf"),
            "is_healthy": state.get("is_healthy"),
            "crop_prediction": state.get("crop_prediction", {}),
            "disease_route": state.get("disease_route", {}),
            "disease_prediction": state.get("disease_prediction", {}),
            "rag": {
                "query": state.get("rag", {}).get("query", ""),
                "knowledge_base_ready": state.get("rag", {}).get("knowledge_base_ready", False),
                "retrieval_strategy": state.get("rag", {}).get("retrieval_strategy", ""),
                "document_count": len(state.get("rag", {}).get("documents", [])),
            },
            "verification": state.get("verification", {}),
            "warnings": state.get("warnings", []),
        }
        self._append_jsonl(self.prediction_log_path, payload)

    def log_feedback_event(self, feedback_payload: dict[str, Any]) -> None:
        payload = {
            "timestamp": self._timestamp(),
            **feedback_payload,
        }
        self._append_jsonl(self.feedback_log_path, payload)

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
