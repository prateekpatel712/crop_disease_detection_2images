from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.services.audit_logger import PipelineAuditLogger


class AuditLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tempdir.name)
        self.logger = PipelineAuditLogger(
            prediction_log_path=self.base_path / "logs" / "prediction.jsonl",
            feedback_log_path=self.base_path / "logs" / "feedback.jsonl",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_prediction_event_is_written_as_jsonl(self) -> None:
        self.logger.log_prediction_event(
            {
                "request_id": "req-1",
                "image_path": "uploads/example.png",
                "image_ok": True,
                "quality_gate_status": "proceed",
                "is_leaf": True,
                "is_healthy": False,
                "crop_prediction": {"label": "Tomato"},
                "disease_route": {"resolved_model_key": "tomato"},
                "disease_prediction": {"label": "Early Blight"},
                "rag": {
                    "query": "Tomato Early Blight",
                    "knowledge_base_ready": True,
                    "retrieval_strategy": "exact_crop_disease",
                    "documents": [{"title": "doc"}],
                },
                "verification": {"status": "verified"},
                "warnings": ["example warning"],
            }
        )

        rows = self._read_jsonl(self.logger.prediction_log_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_id"], "req-1")
        self.assertEqual(rows[0]["rag"]["document_count"], 1)
        self.assertEqual(rows[0]["verification"]["status"], "verified")

    def test_feedback_event_is_written_as_jsonl(self) -> None:
        self.logger.log_feedback_event(
            {
                "request_id": "req-2",
                "verdict": "incorrect",
                "actual_crop_name": "Tomato",
            }
        )

        rows = self._read_jsonl(self.logger.feedback_log_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_id"], "req-2")
        self.assertEqual(rows[0]["verdict"], "incorrect")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


if __name__ == "__main__":
    unittest.main()
