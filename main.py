"""
FastAPI application entry point.
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=== AI Photo Selection System starting ===")
    await init_db()
    logger.info("Database initialised")

    # Create data directories
    for rel_path in [
        settings.paths.watch_folder,
        settings.paths.upload_folder,
        settings.paths.models_folder,
    ]:
        settings.abs_path(rel_path).mkdir(parents=True, exist_ok=True)

    # Start watchdog on default batch folder
    # (watcher is started per-batch via API; just ensure dir exists)
    logger.info("Server ready — open http://localhost:%d in your browser", settings.server.port)
    yield

    # Shutdown
    from app.services.ingestion import stop_watching
    stop_watching()
    logger.info("Server shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Photo Selection System",
    description="ระบบคัดเลือกภาพงานราชกิจฯ แบบกึ่งอัตโนมัติ",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
from app.api import dashboard, review, upload  # noqa: E402

app.include_router(upload.router)
app.include_router(review.router)
app.include_router(dashboard.router)

# Static files (frontend)
static_dir = Path(__file__).parent / "app" / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        log_level="info",
    )
