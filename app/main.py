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
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        openapi_url=settings.openapi_url,
    )
    application.include_router(router)

    @application.get("/", tags=["meta"])
    def root() -> dict[str, str | bool]:
        payload: dict[str, str | bool] = {
            "name": settings.project_name,
            "version": settings.project_version,
            "environment": settings.environment,
            "api_docs_enabled": settings.enable_api_docs,
        }
        if settings.enable_api_docs and settings.docs_url:
            payload["docs_url"] = settings.docs_url
        return payload

    return application


app = create_application()
