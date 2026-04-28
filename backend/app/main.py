from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api import dashboard, exports, mining, upload
from app.utils.file_utils import ensure_dir


def create_app() -> FastAPI:
    app = FastAPI(
        title="Palestine SME Social Media Intelligence Platform",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin, "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure storage folders exist
    storage = settings.storage_path()
    ensure_dir(storage / "raw")
    ensure_dir(storage / "cleaned")
    ensure_dir(storage / "outputs")
    ensure_dir(storage / "reports")

    @app.get("/health")
    def health():
        return {"status": "ok", "env": settings.app_env}

    app.include_router(upload.router, prefix="/api", tags=["upload"])
    app.include_router(mining.router, prefix="/api", tags=["pipeline"])
    app.include_router(dashboard.router, prefix="/api", tags=["dashboard"])
    app.include_router(exports.router, prefix="/api", tags=["exports"])

    return app


app = create_app()

