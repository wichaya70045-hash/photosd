"""
Chunked upload endpoint — accepts batches of photos from WiFi hotspot devices.
Supports:
  - Single-shot upload (small files)
  - Chunked multipart upload (large batches)
  - Progress tracking per batch
  - Auto-retry on partial chunk failure
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.db_models import Batch, Photo, Selection
from app.services.ingestion import (
    cluster_batch,
    find_local_photos,
    process_photo,
    scan_and_ingest_folder,
)

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/api", tags=["upload"])

# In-memory chunk assembly state  {upload_id: {chunk_index: bytes}}
_chunk_store: dict[str, dict[int, bytes]] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class FolderScanRequest(BaseModel):
    folder_path: str = Field(..., description="Absolute or relative path to local photo folder")
    recursive: bool = Field(False, description="Whether to include subdirectories")
    auto_cluster: bool = Field(True, description="Automatically cluster when scan completes")


class QuickScanRequest(BaseModel):
    report_name: str = Field("ไม่ระบุชื่อ", description="Batch / Report name")
    folder_path: str = Field(..., description="Absolute or relative path to local photo folder")
    recursive: bool = Field(False, description="Whether to include subdirectories")


# ---------------------------------------------------------------------------
# Batch management
# ---------------------------------------------------------------------------

@router.post("/batches")
async def create_batch(report_name: str = "ไม่ระบุชื่อ"):
    """Create a new batch and return its ID."""
    batch_id = str(uuid.uuid4())
    async with AsyncSessionLocal() as db:
        batch = Batch(id=batch_id, report_name=report_name)
        db.add(batch)
        await db.commit()
    return {"batch_id": batch_id, "id": batch_id, "report_name": report_name}



@router.get("/batches")
async def list_batches():
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Batch).order_by(Batch.created_at.desc()).limit(50))
        batches = result.scalars().all()
    return [
        {
            "id": b.id,
            "report_name": b.report_name,
            "created_at": b.created_at.isoformat() if b.created_at else None,
            "completed": b.completed,
        }
        for b in batches
    ]


@router.delete("/batches/{batch_id}")
async def delete_batch(batch_id: str):
    """Delete a batch, all its photos, selections, and staging folder."""
    import shutil
    from sqlalchemy import delete

    async with AsyncSessionLocal() as db:
        batch = await db.get(Batch, batch_id)
        if not batch:
            raise HTTPException(404, "Batch not found")

        # Delete selections
        await db.execute(delete(Selection).where(Selection.batch_id == batch_id))
        # Delete photos
        await db.execute(delete(Photo).where(Photo.batch_id == batch_id))
        # Delete batch
        await db.delete(batch)
        await db.commit()

    # Clean up upload staging files if exist
    upload_dir = settings.abs_path(settings.paths.upload_folder) / batch_id
    if upload_dir.exists() and upload_dir.is_dir():
        shutil.rmtree(upload_dir, ignore_errors=True)

    logger.info("Batch %s deleted successfully", batch_id)
    return {"status": "deleted", "batch_id": batch_id, "message": f"ลบ Batch {batch_id[:8]} เรียบร้อย"}



@router.get("/batches/{batch_id}/progress")
async def batch_progress(batch_id: str):
    """Return upload/processing progress for a batch."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Photo).where(Photo.batch_id == batch_id))
        photos = result.scalars().all()

    total = len(photos)
    done = sum(1 for p in photos if p.processing_status == "done")
    errors = sum(1 for p in photos if p.processing_status == "error")
    passed = sum(1 for p in photos if p.passed_technical_filter)
    failed_tech = sum(1 for p in photos if p.passed_technical_filter is False)

    return {
        "total": total,
        "done": done,
        "errors": errors,
        "passed_technical_filter": passed,
        "failed_technical_filter": failed_tech,
        "pending": total - done - errors,
    }


# ---------------------------------------------------------------------------
# Single-shot upload (≤ ~10 photos per request)
# ---------------------------------------------------------------------------

@router.post("/upload/{batch_id}")
async def upload_photos(
    batch_id: str,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    """Upload multiple photos in one request."""
    async with AsyncSessionLocal() as db:
        batch = await db.get(Batch, batch_id)
        if not batch:
            raise HTTPException(404, "Batch not found")

    results = []
    upload_dir = settings.abs_path(settings.paths.upload_folder) / batch_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        try:
            content = await file.read()
            safe_name = Path(file.filename or "photo").name
            dest = upload_dir / safe_name
            # Handle duplicate filenames
            if dest.exists():
                dest = upload_dir / f"{dest.stem}_{uuid.uuid4().hex[:6]}{dest.suffix}"
            dest.write_bytes(content)

            background_tasks.add_task(
                process_photo, dest, batch_id, file.filename or safe_name
            )
            results.append({"filename": file.filename, "status": "queued"})
        except Exception as exc:
            logger.error("Upload failed for %s: %s", file.filename, exc)
            results.append({"filename": file.filename, "status": "error", "detail": str(exc)})

    return {"batch_id": batch_id, "results": results}


# ---------------------------------------------------------------------------
# Chunked upload (large batches, resumable)
# ---------------------------------------------------------------------------

@router.post("/upload/{batch_id}/chunk/init")
async def init_chunked_upload(batch_id: str, filename: str, total_chunks: int):
    """Initialise a chunked upload session. Returns upload_id."""
    upload_id = f"{batch_id}_{uuid.uuid4().hex}"
    _chunk_store[upload_id] = {}
    return {
        "upload_id": upload_id,
        "chunk_size": settings.upload.chunk_size_bytes,
        "total_chunks": total_chunks,
    }


@router.post("/upload/chunk/{upload_id}")
async def upload_chunk(
    upload_id: str,
    chunk_index: Annotated[int, Form()],
    file: UploadFile = File(...),
):
    """Receive one chunk of a file."""
    if upload_id not in _chunk_store:
        raise HTTPException(404, "upload_id not found — call /chunk/init first")
    data = await file.read()
    _chunk_store[upload_id][chunk_index] = data
    return {"upload_id": upload_id, "chunk_index": chunk_index, "received_bytes": len(data)}


@router.post("/upload/chunk/{upload_id}/complete")
async def complete_chunked_upload(
    upload_id: str,
    batch_id: str,
    filename: str,
    total_chunks: int,
    background_tasks: BackgroundTasks,
):
    """Assemble all chunks and enqueue processing."""
    if upload_id not in _chunk_store:
        raise HTTPException(404, "upload_id not found")

    chunks = _chunk_store[upload_id]
    if len(chunks) != total_chunks:
        missing = [i for i in range(total_chunks) if i not in chunks]
        raise HTTPException(400, f"Missing chunks: {missing}")

    upload_dir = settings.abs_path(settings.paths.upload_folder) / batch_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(filename).name
    dest = upload_dir / safe_name
    with open(dest, "wb") as f:
        for idx in range(total_chunks):
            f.write(chunks[idx])

    # Clean up memory
    del _chunk_store[upload_id]

    background_tasks.add_task(process_photo, dest, batch_id, filename)
    return {"status": "assembling", "filename": filename, "batch_id": batch_id}


# ---------------------------------------------------------------------------
# Complete batch (trigger clustering)
# ---------------------------------------------------------------------------

@router.post("/batches/{batch_id}/complete")
async def complete_batch(batch_id: str, background_tasks: BackgroundTasks):
    """
    Mark batch upload as complete.
    Triggers activity clustering and potentially retraining.
    """
    async with AsyncSessionLocal() as db:
        batch = await db.get(Batch, batch_id)
        if not batch:
            raise HTTPException(404, "Batch not found")
        batch.completed = True
        await db.commit()

    background_tasks.add_task(_finalize_batch, batch_id)
    return {"status": "finalizing", "batch_id": batch_id}


async def _finalize_batch(batch_id: str) -> None:
    """Cluster photos and optionally trigger retraining."""
    try:
        await cluster_batch(batch_id)

        # Count completed batches for retraining trigger
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(Batch).where(Batch.completed == True))
            count = len(result.scalars().all())

        from app.services.retraining import maybe_trigger_retraining
        await maybe_trigger_retraining(count)

    except Exception as exc:
        logger.error("Batch finalization failed for %s: %s", batch_id, exc, exc_info=True)


# ---------------------------------------------------------------------------
# Direct Local Folder Scan Endpoints
# ---------------------------------------------------------------------------

@router.get("/folder-presets")
async def get_folder_presets():
    """Return common user directory presets for easy picking."""
    import os
    home = Path.home()
    presets = [
        {"name": "โฟลเดอร์รูปภาพ (Pictures)", "path": str(home / "Pictures")},
        {"name": "ดาวน์โหลด (Downloads)", "path": str(home / "Downloads")},
        {"name": "หน้าเดสก์ท็อป (Desktop)", "path": str(home / "Desktop")},
        {"name": "Watch Inbox (ของระบบ)", "path": str(settings.abs_path(settings.paths.watch_folder))},
    ]
    # Filter only those that exist
    return [p for p in presets if Path(p["path"]).exists()]


@router.post("/batches/{batch_id}/scan-folder")
async def scan_batch_folder(
    batch_id: str,
    req: FolderScanRequest,
    background_tasks: BackgroundTasks,
):
    """
    Directly scan a local folder on this machine without HTTP upload.
    Finds and processes all photos in the directory asynchronously.
    """
    folder = Path(req.folder_path.strip().strip('"').strip("'"))
    if not folder.exists():
        raise HTTPException(404, f"ไม่พบโฟลเดอร์: {folder}")
    if not folder.is_dir():
        raise HTTPException(400, f"เส้นทางนี้ไม่ใช่โฟลเดอร์: {folder}")

    async with AsyncSessionLocal() as db:
        batch = await db.get(Batch, batch_id)
        if not batch:
            raise HTTPException(404, "Batch not found")

    photos = find_local_photos(folder, recursive=req.recursive)
    if not photos:
        return {
            "batch_id": batch_id,
            "status": "empty",
            "message": f"ไม่พบไฟล์ภาพในโฟลเดอร์ {folder}",
            "photos_found": 0,
        }

    # Queue background ingestion & auto-clustering
    background_tasks.add_task(
        scan_and_ingest_folder,
        folder,
        batch_id,
        req.recursive,
        req.auto_cluster,
    )

    return {
        "batch_id": batch_id,
        "status": "scanning",
        "folder_path": str(folder),
        "photos_found": len(photos),
        "message": f"พบ {len(photos)} ภาพ — กำลังเริ่มตรวจสอบไฟล์ในเครื่องโดยตรง",
    }


@router.post("/scan-local")
async def quick_scan_local_folder(
    req: QuickScanRequest,
    background_tasks: BackgroundTasks,
):
    """
    1-Click Scan: Create batch and start scanning local folder immediately.
    """
    folder = Path(req.folder_path.strip().strip('"').strip("'"))
    if not folder.exists():
        raise HTTPException(404, f"ไม่พบโฟลเดอร์: {folder}")
    if not folder.is_dir():
        raise HTTPException(400, f"เส้นทางนี้ไม่ใช่โฟลเดอร์: {folder}")

    photos = find_local_photos(folder, recursive=req.recursive)
    if not photos:
        raise HTTPException(400, f"ไม่พบไฟล์ภาพที่รองรับในโฟลเดอร์: {folder}")

    batch_id = str(uuid.uuid4())
    report_name = req.report_name.strip() or f"สแกน {folder.name}"

    async with AsyncSessionLocal() as db:
        batch = Batch(id=batch_id, report_name=report_name)
        db.add(batch)
        await db.commit()

    # Queue scanning
    background_tasks.add_task(
        scan_and_ingest_folder,
        folder,
        batch_id,
        req.recursive,
        True,
    )

    return {
        "batch_id": batch_id,
        "id": batch_id,
        "report_name": report_name,
        "status": "scanning",
        "folder_path": str(folder),
        "photos_found": len(photos),
        "message": f"สร้าง Batch สำเร็จและเริ่มตรวจสอบ {len(photos)} ภาพจากเครื่องโดยตรง",
    }

