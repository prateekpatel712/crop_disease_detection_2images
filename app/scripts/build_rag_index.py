from __future__ import annotations

from app.core.config import get_settings
from app.services.rag_service import RAGService


def main() -> int:
    settings = get_settings()
    settings.ensure_runtime_dirs()

    service = RAGService(
        raw_dir=settings.rag_raw_dir,
        processed_dir=settings.rag_processed_dir,
        vector_store_dir=settings.rag_vector_store_dir,
    )
    ready = service.ensure_index_ready()
    return 0 if ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
