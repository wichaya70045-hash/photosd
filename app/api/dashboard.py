"""
Dashboard API — history, stats, model versions, retraining.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks
from sqlalchemy import func, select

from app.core.database import AsyncSessionLocal
from app.models.db_models import Batch, ModelVersion, Photo, Selection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/stats")
async def get_stats():
    """Overall system statistics."""
    async with AsyncSessionLocal() as db:
        batch_count_result = await db.execute(select(func.count(Batch.id)))
        batch_count = batch_count_result.scalar() or 0

        photo_count_result = await db.execute(select(func.count(Photo.id)))
        photo_count = photo_count_result.scalar() or 0

        selection_count_result = await db.execute(select(func.count(Selection.id)))
        selection_count = selection_count_result.scalar() or 0

        active_model_result = await db.execute(
            select(ModelVersion).where(ModelVersion.is_active == True)
        )
        active_model = active_model_result.scalar_one_or_none()

    return {
        "total_batches": batch_count,
        "total_photos": photo_count,
        "total_selections": selection_count,
        "ml_mode": "active" if active_model else "cold_start",
        "active_model": {
            "id": active_model.id,
            "trained_at": active_model.trained_at.isoformat() if active_model else None,
            "val_score": active_model.validation_score if active_model else None,
            "sample_count": active_model.training_sample_count if active_model else None,
        } if active_model else None,
    }


@router.get("/model-versions")
async def list_model_versions():
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ModelVersion).order_by(ModelVersion.trained_at.desc()).limit(20)
        )
        versions = result.scalars().all()
    return [
        {
            "id": v.id,
            "trained_at": v.trained_at.isoformat() if v.trained_at else None,
            "sample_count": v.training_sample_count,
            "val_score": v.validation_score,
            "is_active": v.is_active,
        }
        for v in versions
    ]


@router.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Manually trigger model retraining."""
    from app.services.retraining import run_retraining
    background_tasks.add_task(run_retraining)
    return {"status": "retraining_started"}


@router.get("/config")
async def get_config():
    """Return current configuration."""
    from app.core.config import get_settings
    s = get_settings()
    return {
        "quality_filter": s.quality_filter.model_dump(),
        "clustering": s.clustering.model_dump(),
        "ml": s.ml.model_dump(),
        "shortlist": s.shortlist.model_dump(),
    }
