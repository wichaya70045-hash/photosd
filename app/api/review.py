"""
Review API — serve photos and receive selection feedback.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.db_models import Batch, Photo, Selection

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api", tags=["review"])


# ---------------------------------------------------------------------------
# Photo serving
# ---------------------------------------------------------------------------

@router.get("/photos/{photo_id}/image")
async def serve_photo(photo_id: str):
    """Serve the actual photo file."""
    async with AsyncSessionLocal() as db:
        photo = await db.get(Photo, photo_id)
    if not photo:
        raise HTTPException(404, "Photo not found")
    path = Path(photo.filepath)
    if not path.exists():
        raise HTTPException(404, "Photo file missing")
    return FileResponse(path, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Review data
# ---------------------------------------------------------------------------

@router.get("/batches/{batch_id}/review")
async def get_review(batch_id: str):
    """
    Return photos grouped by activity cluster, ranked by score.
    Only photos that passed the technical filter are included in ranking.
    All photos are returned for completeness (failed ones flagged).
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Photo)
            .where(Photo.batch_id == batch_id)
            .order_by(Photo.activity_cluster_id, Photo.rank_in_cluster)
        )
        photos = result.scalars().all()

    if not photos:
        raise HTTPException(404, "No photos found for this batch")

    # Group by cluster
    clusters: dict[str, list] = {}
    for p in photos:
        cid = p.activity_cluster_id or "ไม่ระบุ"
        clusters.setdefault(cid, [])

        score = p.ml_score if p.ml_score is not None else (
            p.blur_score / 10000.0 if p.blur_score else 0.0
        )

        clusters[cid].append({
            "id": p.id,
            "filename": p.original_filename,
            "rank": p.rank_in_cluster,
            "blur_score": round(p.blur_score or 0, 2),
            "exposure_score": round(p.exposure_score or 0, 4),
            "obstruction_flag": p.obstruction_flag,
            "passed_technical_filter": p.passed_technical_filter,
            "ml_score": round(p.ml_score, 4) if p.ml_score is not None else None,
            "score": round(score, 4),
            "image_url": f"/api/photos/{p.id}/image",
        })

    # Compute shortlist: top photos across all clusters
    shortlist_n = settings.shortlist.top_n_per_batch
    all_passed = [
        p for p in photos if p.passed_technical_filter
    ]
    all_passed.sort(
        key=lambda p: (p.ml_score or 0) if p.ml_score is not None else (p.blur_score or 0) / 10000.0,
        reverse=True,
    )

    # Diversity: take top-N ensuring all clusters represented
    seen_clusters: set[str] = set()
    shortlist: list[str] = []
    second_pass: list[Photo] = []
    for p in all_passed:
        cid = p.activity_cluster_id or "ไม่ระบุ"
        if cid not in seen_clusters:
            shortlist.append(p.id)
            seen_clusters.add(cid)
        else:
            second_pass.append(p)

    for p in second_pass:
        if len(shortlist) >= shortlist_n:
            break
        shortlist.append(p.id)

    return {
        "batch_id": batch_id,
        "total_photos": len(photos),
        "shortlist_ids": shortlist[:shortlist_n],
        "clusters": [
            {
                "id": cid,
                "name": f"กิจกรรม {cid}",
                "photo_count": len(clist),
                "photos": clist,
            }
            for cid, clist in sorted(clusters.items())
        ],
    }


# ---------------------------------------------------------------------------
# Selections (feedback)
# ---------------------------------------------------------------------------

class SelectionPayload(BaseModel):
    selections: dict[str, bool]  # {photo_id: selected}


@router.post("/batches/{batch_id}/selections")
async def save_selections(batch_id: str, payload: SelectionPayload):
    """Save user selections for a batch (training labels)."""
    async with AsyncSessionLocal() as db:
        batch = await db.get(Batch, batch_id)
        if not batch:
            raise HTTPException(404, "Batch not found")

        saved = 0
        for photo_id, selected in payload.selections.items():
            sel = Selection(
                photo_id=photo_id,
                batch_id=batch_id,
                selected=selected,
            )
            db.add(sel)
            saved += 1
        await db.commit()

    return {"saved": saved, "batch_id": batch_id}


@router.get("/batches/{batch_id}/selections")
async def get_selections(batch_id: str):
    """Return existing selections for a batch."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Selection).where(Selection.batch_id == batch_id)
        )
        sels = result.scalars().all()
    return {s.photo_id: s.selected for s in sels}
