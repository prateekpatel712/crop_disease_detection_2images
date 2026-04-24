from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.services.rag_service import RAGService


class RAGServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tempdir.name)
        self.raw_dir = self.base_path / "rag_data" / "raw"
        self.processed_dir = self.base_path / "rag_data" / "processed"
        self.vector_store_dir = self.base_path / "rag_data" / "vector_store"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.service = RAGService(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
            vector_store_dir=self.vector_store_dir,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_knowledge_assets_detected_ignores_example_files(self) -> None:
        (self.processed_dir / "disease_document.example.json").write_text(
            json.dumps({"documents": []}),
            encoding="utf-8",
        )

        self.assertFalse(self.service.knowledge_assets_detected())

    def test_retrieve_prefers_exact_crop_and_disease_match(self) -> None:
        self._write_documents(
            [
                {
                    "id": "tomato_early_blight",
                    "crop": "Tomato",
                    "disease": "Early Blight",
                    "summary": "Tomato early blight causes concentric lesions.",
                    "symptoms": ["concentric leaf spots"],
                    "treatment": ["remove infected leaves"],
                    "precautions": ["avoid overhead irrigation"],
                    "prevention": ["rotate crops"],
                    "source": "test-source",
                },
                {
                    "id": "potato_early_blight",
                    "crop": "Potato",
                    "disease": "Early Blight",
                    "summary": "Potato early blight affects older leaves first.",
                    "symptoms": ["dark circular lesions"],
                    "source": "test-source",
                },
            ]
        )

        result = self.service.retrieve(
            query="Tomato early blight symptoms treatment",
            crop_name="Tomato",
            disease_name="Early Blight",
        )

        self.assertTrue(result.knowledge_base_ready)
        self.assertEqual(result.retrieval_strategy, "exact_crop_disease")
        self.assertGreater(result.chunk_count, 0)
        self.assertNotEqual(result.embedding_backend, "none")
        self.assertTrue(result.generation_context)
        self.assertEqual(result.documents[0].metadata["crop"], "Tomato")
        self.assertEqual(result.documents[0].metadata["disease"], "Early Blight")

    def test_retrieve_skips_when_no_target_is_available(self) -> None:
        self._write_documents(
            [
                {
                    "id": "tomato_early_blight",
                    "crop": "Tomato",
                    "disease": "Early Blight",
                    "summary": "Tomato early blight causes concentric lesions.",
                    "source": "test-source",
                }
            ]
        )

        result = self.service.retrieve(
            query="",
            crop_name="runtime_not_connected",
            disease_name="model_not_uploaded",
        )

        self.assertTrue(result.knowledge_base_ready)
        self.assertEqual(result.retrieval_strategy, "no_target")
        self.assertEqual(result.documents, [])

    def test_rerank_promotes_exact_disease_match(self) -> None:
        self._write_documents(
            [
                {
                    "id": "tomato_early_blight",
                    "crop": "Tomato",
                    "disease": "Early Blight",
                    "aliases": ["Alternaria"],
                    "summary": "Tomato early blight causes concentric lesions.",
                    "source": "test-source",
                },
                {
                    "id": "tomato_leaf_mold",
                    "crop": "Tomato",
                    "disease": "Leaf Mold",
                    "summary": "Tomato leaf mold causes fuzzy growth under leaves.",
                    "source": "test-source",
                },
            ]
        )

        result = self.service.retrieve(
            query="Tomato early blight precautions",
            crop_name="Tomato",
            disease_name="Early Blight",
        )
        payload = result.to_dict()
        reranked = self.service.rerank(
            rag_payload=payload,
            query=result.query,
            crop_name="Tomato",
            disease_name="Early Blight",
        )

        self.assertEqual(reranked["documents"][0]["metadata"]["disease"], "Early Blight")
        self.assertEqual(reranked["retrieval_strategy"], "exact_crop_disease")
        self.assertTrue(reranked["generation_context"])

    def test_ensure_index_ready_builds_vector_store_artifacts(self) -> None:
        self._write_documents(
            [
                {
                    "id": "wheat_leaf_blight",
                    "crop": "Wheat",
                    "disease": "Leaf blight",
                    "summary": "Leaf blight causes blotches that move upward toward the flag leaf.",
                    "symptoms": ["dark brown blotches", "yellow halos", "premature drying"],
                    "treatment": ["protect the flag leaf with fungicide"],
                    "source": "test-source",
                }
            ]
        )

        ready = self.service.ensure_index_ready()

        self.assertTrue(ready)
        self.assertTrue((self.vector_store_dir / "manifest.json").exists())
        self.assertTrue((self.vector_store_dir / "chunks.jsonl").exists())
        self.assertTrue((self.vector_store_dir / "vectorizer.joblib").exists())
        self.assertTrue((self.vector_store_dir / "dense_embeddings.npy").exists())

    def test_global_fallback_returns_semantic_chunk_match(self) -> None:
        self._write_documents(
            [
                {
                    "id": "tomato_early_blight",
                    "crop": "Tomato",
                    "disease": "Early Blight",
                    "summary": "Tomato early blight causes bullseye lesions on older lower leaves.",
                    "symptoms": ["bullseye spots", "yellowing lower canopy", "fruit spots near calyx"],
                    "treatment": ["use chlorothalonil", "remove infected lower leaves"],
                    "precautions": ["avoid soil splash", "stake plants"],
                    "source": "test-source",
                },
                {
                    "id": "tomato_leaf_mold",
                    "crop": "Tomato",
                    "disease": "Leaf Mold",
                    "summary": "Leaf mold causes fuzzy olive growth under leaves in humid structures.",
                    "symptoms": ["yellow spots", "olive mold underneath"],
                    "source": "test-source",
                },
            ]
        )

        result = self.service.retrieve(
            query="tomato lower leaves with bullseye spotting",
            crop_name=None,
            disease_name=None,
        )

        self.assertTrue(result.knowledge_base_ready)
        self.assertEqual(result.retrieval_strategy, "global_fallback")
        self.assertEqual(result.documents[0].title, "Tomato - Early Blight")
        self.assertIn("bullseye", result.generation_context.lower())

    def _write_documents(self, documents: list[dict]) -> None:
        (self.processed_dir / "knowledge.json").write_text(
            json.dumps({"documents": documents}),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
