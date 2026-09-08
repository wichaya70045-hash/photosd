"""
Continuous Learning — Phase 3
Reads selection labels → extracts embeddings → retrains classifier → versions model.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.db_models import ModelVersion, Photo, Selection
from app.services.ml_scorer import (
    PhotoClassifier,
    bytes_to_embedding,
    get_classifier,
)

logger = logging.getLogger(__name__)
settings = get_settings()
ml_cfg = settings.ml


async def get_training_data() -> tuple[np.ndarray, np.ndarray] | None:
    """
    Collect (X, y) from selections table.
    X = embedding blobs; y = 0/1 labels.
    Returns None if insufficient data.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Selection, Photo)
            .join(Photo, Photo.id == Selection.photo_id)
            .where(Photo.embedding.isnot(None))
        )
        rows = result.all()

    if len(rows) < ml_cfg.cold_start_min_samples:
        logger.info(
            "Training data insufficient: %d samples < %d required",
            len(rows), ml_cfg.cold_start_min_samples
        )
        return None

    X, y = [], []
    for sel, photo in rows:
        try:
            emb = bytes_to_embedding(photo.embedding)
            X.append(emb)
            y.append(1 if sel.selected else 0)
        except Exception as exc:
            logger.warning("Could not decode embedding for photo %s: %s", photo.id, exc)

    if not X:
        return None

    return np.array(X), np.array(y)


async def run_retraining() -> bool:
    """
    Main retraining entry point. Called from background task.
    Returns True if a new model was saved and activated.
    """
    logger.info("Retraining job started at %s", datetime.now().isoformat())

    data = await get_training_data()
    if data is None:
        return False

    X, y = data

    clf = PhotoClassifier()
    val_score = clf.train(X, y)

    if val_score == 0.0:
        logger.warning("Retraining produced degenerate model; aborting")
        return False

    # Save versioned model
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = settings.abs_path(settings.paths.models_folder)
    model_path = models_dir / f"classifier_{ts}.pkl"
    clf.save(model_path)

    # Also overwrite the "active" symlink / copy
    active_path = models_dir / "active_classifier.pkl"
    clf.save(active_path)

    # Reload the global classifier
    global_clf = get_classifier()
    global_clf.load(active_path)

    # Persist to DB
    async with AsyncSessionLocal() as db:
        # Deactivate all previous
        result = await db.execute(select(ModelVersion).where(ModelVersion.is_active == True))
        for mv in result.scalars():
            mv.is_active = False

        new_version = ModelVersion(
            training_sample_count=len(X),
            validation_score=val_score,
            model_path=str(model_path),
            is_active=True,
        )
        db.add(new_version)
        await db.commit()

    logger.info(
        "Retraining complete — val_score=%.3f, samples=%d, path=%s",
        val_score, len(X), model_path
    )
    return True


async def maybe_trigger_retraining(completed_batch_count: int) -> None:
    """
    Trigger retraining every N completed batches.
    Called from the batch completion handler.
    """
    n = ml_cfg.retrain_every_n_batches
    if completed_batch_count > 0 and completed_batch_count % n == 0:
        logger.info("Triggering scheduled retraining (batch #%d)", completed_batch_count)
        asyncio.create_task(_retraining_task())


async def _retraining_task() -> None:
    try:
        await run_retraining()
    except Exception as exc:
        logger.error("Retraining task failed: %s", exc, exc_info=True)
