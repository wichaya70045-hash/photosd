"""
Ingestion service:
  • Watched folder monitor (watchdog)
  • HEIC → JPEG conversion (pillow-heif)
  • SHA256 deduplication
  • Spawns the processing pipeline per photo
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal
from app.models.db_models import Batch, Photo

logger = logging.getLogger(__name__)
settings = get_settings()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tiff", ".tif"}

# Global observer instance
_observer: Optional[Observer] = None


# ---------------------------------------------------------------------------
# HEIC conversion
# ---------------------------------------------------------------------------

def convert_heic_to_jpeg(src: Path, dest_dir: Path) -> Path:
    """Convert HEIC/HEIF to JPEG. Returns path to converted file."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        from PIL import Image

        dest = dest_dir / (src.stem + ".jpg")
        img = Image.open(src)
        img = img.convert("RGB")
        img.save(dest, "JPEG", quality=95)
        logger.info("HEIC converted: %s → %s", src.name, dest.name)
        return dest
    except Exception as exc:
        logger.error("HEIC conversion failed for %s: %s", src, exc)
        raise


def ensure_jpeg(src: Path, staging_dir: Path, copy_if_local: bool = False) -> Path:
    """Return a JPEG-ready path (converts HEIC if needed, otherwise uses original or staging)."""
    ext = src.suffix.lower()
    if ext in {".heic", ".heif"}:
        return convert_heic_to_jpeg(src, staging_dir)
    if copy_if_local:
        dest = staging_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        return dest
    return src


# ---------------------------------------------------------------------------
# SHA256 hash
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Processing pipeline (async)
# ---------------------------------------------------------------------------

async def process_photo(
    filepath: Path,
    batch_id: str,
    original_filename: str,
) -> Optional[str]:
    """
    Full processing pipeline for one photo:
    1. Convert HEIC if needed
    2. Hash → dedup check
    3. Quality filter
    4. Save to DB
    5. Extract embedding (non-blocking)
    Returns photo_id or None on failure.
    """
    from app.services.quality_filter import run_quality_filter
    from app.services.ml_scorer import extract_embedding, embedding_to_bytes, score_photo

    # Staging area
    upload_dir = settings.abs_path(settings.paths.upload_folder)
    staging_dir = upload_dir / batch_id
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        ready_path = ensure_jpeg(filepath, staging_dir)
        file_hash = sha256_file(ready_path)
        photo_id = f"{batch_id}_{file_hash}"

        async with AsyncSessionLocal() as db:
            existing = await db.get(Photo, photo_id)
            if existing:
                logger.debug("Duplicate photo in batch %s skipped: %s", batch_id, file_hash[:8])
                return photo_id

            # Quality filter (CPU-bound → run in thread)
            qr = await asyncio.get_event_loop().run_in_executor(
                None, run_quality_filter, ready_path
            )
            if qr is None:
                logger.warning("Quality filter returned None for %s", ready_path)
                return None

            photo = Photo(
                id=photo_id,
                filepath=str(ready_path),
                original_filename=original_filename,
                batch_id=batch_id,
                blur_score=qr.blur_score,
                exposure_score=qr.exposure_score,
                obstruction_flag=qr.obstruction_flag,
                passed_technical_filter=qr.passed,
                processing_status="done",
            )
            db.add(photo)
            await db.commit()

        # Extract embedding in background (non-blocking)
        asyncio.create_task(_extract_and_save_embedding(photo_id, ready_path))

        return photo_id

    except Exception as exc:
        logger.error("process_photo failed for %s: %s", filepath, exc, exc_info=True)
        return None


async def _extract_and_save_embedding(photo_id: str, filepath: Path) -> None:
    """Background task: extract embedding and update DB."""
    from app.services.ml_scorer import extract_embedding, embedding_to_bytes, score_photo

    loop = asyncio.get_event_loop()
    emb = await loop.run_in_executor(None, extract_embedding, filepath)
    if emb is None:
        return

    emb_bytes = embedding_to_bytes(emb)
    ml_score = score_photo(emb)

    async with AsyncSessionLocal() as db:
        photo = await db.get(Photo, photo_id)
        if photo:
            photo.embedding = emb_bytes
            photo.ml_score = ml_score
            await db.commit()


# ---------------------------------------------------------------------------
# Clustering trigger
# ---------------------------------------------------------------------------

async def cluster_batch(batch_id: str) -> None:
    """Run activity clustering on all photos in a batch and update DB."""
    from app.services.clustering import assign_clusters

    async with AsyncSessionLocal() as db:
        from sqlalchemy import select
        result = await db.execute(
            select(Photo).where(Photo.batch_id == batch_id)
        )
        photos = result.scalars().all()

    if not photos:
        return

    photo_pairs = [(p.id, Path(p.filepath)) for p in photos]
    loop = asyncio.get_event_loop()
    clusters = await loop.run_in_executor(None, assign_clusters, photo_pairs)

    # Rank within each cluster by ml_score (or blur_score as fallback)
    cluster_groups: dict[str, list[Photo]] = {}
    for p in photos:
        cid = clusters.get(p.id, "?")
        cluster_groups.setdefault(cid, []).append(p)

    async with AsyncSessionLocal() as db:
        for cid, group in cluster_groups.items():
            # Sort by ml_score desc, then blur_score desc
            group.sort(
                key=lambda p: (p.ml_score or 0.0) if p.ml_score is not None
                              else (p.blur_score or 0.0) / 10000.0,
                reverse=True,
            )
            for rank, p in enumerate(group, start=1):
                photo = await db.get(Photo, p.id)
                if photo:
                    photo.activity_cluster_id = cid
                    photo.rank_in_cluster = rank
            await db.commit()

    logger.info("Batch %s clustered into %d activities", batch_id, len(cluster_groups))


# ---------------------------------------------------------------------------
# Watchdog handler
# ---------------------------------------------------------------------------

class PhotoHandler(FileSystemEventHandler):
    """Handles new files dropped into the watched folder."""

    def __init__(self, loop: asyncio.AbstractEventLoop, batch_id: str):
        self.loop = loop
        self.batch_id = batch_id

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            logger.info("Watchdog detected: %s", path.name)
            asyncio.run_coroutine_threadsafe(
                process_photo(path, self.batch_id, path.name),
                self.loop,
            )


def start_watching(batch_id: str) -> None:
    """Start watchdog on the configured folder."""
    global _observer
    watch_dir = settings.abs_path(settings.paths.watch_folder)
    watch_dir.mkdir(parents=True, exist_ok=True)

    if _observer and _observer.is_alive():
        _observer.stop()

    loop = asyncio.get_event_loop()
    handler = PhotoHandler(loop, batch_id)
    _observer = Observer()
    _observer.schedule(handler, str(watch_dir), recursive=False)
    _observer.start()
    logger.info("Watching folder: %s (batch=%s)", watch_dir, batch_id)


def stop_watching() -> None:
    global _observer
    if _observer:
        _observer.stop()
        _observer.join()
        _observer = None


# ---------------------------------------------------------------------------
# Direct Local Folder Ingestion
# ---------------------------------------------------------------------------

def find_local_photos(folder_path: Path, recursive: bool = False) -> list[Path]:
    """Find all supported photos in a local folder."""
    if not folder_path.exists() or not folder_path.is_dir():
        return []

    photos = []
    pattern = "**/*" if recursive else "*"
    for p in folder_path.glob(pattern):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            photos.append(p)
    return sorted(photos, key=lambda x: x.name)


async def scan_and_ingest_folder(
    folder_path: Path,
    batch_id: str,
    recursive: bool = False,
    auto_cluster_when_done: bool = True,
) -> int:
    """
    Scan a local folder on this machine and ingest all photos into the batch.
    Processes photos asynchronously and optionally triggers clustering when all are processed.
    """
    photos = find_local_photos(folder_path, recursive=recursive)
    logger.info("Found %d photos in local folder: %s (batch=%s)", len(photos), folder_path, batch_id)

    if not photos:
        return 0

    # Process all photos concurrently with a semaphore to avoid overwhelming CPU
    sem = asyncio.Semaphore(4)

    async def _worker(p: Path):
        async with sem:
            await process_photo(p, batch_id, p.name)

    tasks = [_worker(p) for p in photos]
    await asyncio.gather(*tasks, return_exceptions=True)

    if auto_cluster_when_done:
        logger.info("Auto-clustering batch %s after scanning folder", batch_id)
        await cluster_batch(batch_id)
        from app.services.retraining import maybe_trigger_retraining
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select
            res = await db.execute(select(Batch).where(Batch.completed == True))
            count = len(res.scalars().all())
        await maybe_trigger_retraining(count)

    return len(photos)

