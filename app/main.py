from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_application() -> FastAPI:
    settings = get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.logs_dir)

    application = FastAPI(
        title=settings.project_name,
        version=settings.project_version,
        description=(
            "Phase 1 scaffold for crop classification, disease detection, "
            "and RAG orchestration with LangGraph."
        ),
    )
    application.include_router(router)

    @application.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {
            "name": settings.project_name,
            "version": settings.project_version,
            "health_endpoint": f"{settings.api_prefix}/health",
            "predict_endpoint": f"{settings.api_prefix}/predict",
        }

    return application


app = create_application()
