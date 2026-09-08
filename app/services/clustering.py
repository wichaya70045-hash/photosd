"""
Activity clustering:
  Primary  — EXIF DateTimeOriginal time-gap clustering
  Fallback — perceptual hashing + agglomerative clustering (no EXIF)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import imagehash
from PIL import Image

from app.core.config import get_settings

try:
    import exifread
    EXIFREAD_AVAILABLE = True
except ImportError:
    EXIFREAD_AVAILABLE = False

logger = logging.getLogger(__name__)
settings = get_settings()
cfg = settings.clustering


# ---------------------------------------------------------------------------
# EXIF helpers
# ---------------------------------------------------------------------------

def _read_exif_datetime(filepath: Path) -> Optional[datetime]:
    """Extract DateTimeOriginal from EXIF. Returns None if unavailable."""
    if not EXIFREAD_AVAILABLE:
        return None
    try:
        with open(filepath, "rb") as f:
            tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal", details=False)
        tag = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
        if tag is None:
            return None
        dt_str = str(tag)  # "2024:01:15 10:30:00"
        return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except Exception as exc:
        logger.debug("EXIF read failed for %s: %s", filepath, exc)
        return None


# ---------------------------------------------------------------------------
# Primary: EXIF time-gap clustering
# ---------------------------------------------------------------------------

def cluster_by_exif_time(
    photos: list[tuple[str, Path]],  # (photo_id, filepath)
) -> dict[str, str]:
    """
    Assign cluster IDs based on EXIF timestamps.
    Returns {photo_id: cluster_label} e.g. {"img1": "A", "img2": "A", ...}
    Photos without EXIF are returned with cluster_id = None (handled by caller).
    """
    with_time: list[tuple[str, datetime]] = []
    no_time: list[str] = []

    for pid, fp in photos:
        dt = _read_exif_datetime(fp)
        if dt is not None:
            with_time.append((pid, dt))
        else:
            no_time.append(pid)

    # Sort by time
    with_time.sort(key=lambda x: x[1])

    gap = timedelta(minutes=cfg.time_gap_minutes)
    clusters: dict[str, str] = {}
    cluster_idx = 0
    prev_time: Optional[datetime] = None

    for pid, dt in with_time:
        if prev_time is None or (dt - prev_time) > gap:
            cluster_idx += 1
        label = _idx_to_label(cluster_idx)
        clusters[pid] = label
        prev_time = dt

    logger.info(
        "EXIF clustering: %d photos → %d clusters; %d photos lack EXIF",
        len(with_time), cluster_idx, len(no_time)
    )
    return clusters, no_time


# ---------------------------------------------------------------------------
# Fallback: perceptual hash agglomerative clustering
# ---------------------------------------------------------------------------

def cluster_by_perceptual_hash(
    photos: list[tuple[str, Path]],
    existing_cluster_count: int = 0,
) -> dict[str, str]:
    """
    Group photos without EXIF using perceptual hash similarity.
    Uses single-linkage agglomerative clustering with Hamming distance threshold.
    """
    if not photos:
        return {}

    hashes: list[tuple[str, imagehash.ImageHash]] = []
    for pid, fp in photos:
        try:
            h = imagehash.phash(Image.open(fp))
            hashes.append((pid, h))
        except Exception as exc:
            logger.warning("Hash failed for %s: %s", fp, exc)
            hashes.append((pid, None))

    threshold = cfg.perceptual_hash_distance
    cluster_map: dict[str, int] = {}
    cluster_idx = existing_cluster_count

    for i, (pid_i, hash_i) in enumerate(hashes):
        if hash_i is None:
            cluster_idx += 1
            cluster_map[pid_i] = cluster_idx
            continue
        # Find nearest existing cluster
        assigned = False
        for j in range(i):
            pid_j, hash_j = hashes[j]
            if hash_j is not None and abs(hash_i - hash_j) <= threshold:
                cluster_map[pid_i] = cluster_map[pid_j]
                assigned = True
                break
        if not assigned:
            cluster_idx += 1
            cluster_map[pid_i] = cluster_idx

    return {pid: _idx_to_label(cid) for pid, cid in cluster_map.items()}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assign_clusters(photos: list[tuple[str, Path]]) -> dict[str, str]:
    """
    Full clustering pipeline:
    1. Try EXIF time-gap clustering.
    2. Fallback perceptual hash for photos missing EXIF.
    Returns {photo_id: cluster_label}.
    """
    exif_clusters, no_exif_ids = cluster_by_exif_time(photos)

    # Build path lookup
    path_map = {pid: fp for pid, fp in photos}

    all_clusters = dict(exif_clusters)

    if no_exif_ids:
        logger.info("Running perceptual hash clustering for %d EXIF-less photos", len(no_exif_ids))
        fallback_photos = [(pid, path_map[pid]) for pid in no_exif_ids]
        existing_count = max(
            (int(_label_to_idx(v)) for v in exif_clusters.values()),
            default=0,
        )
        fallback_clusters = cluster_by_perceptual_hash(fallback_photos, existing_count)
        all_clusters.update(fallback_clusters)

    return all_clusters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _idx_to_label(idx: int) -> str:
    """Convert 1-based integer to label: 1→A, 2→B, …, 27→AA"""
    label = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        label = chr(65 + rem) + label
    return label or "A"


def _label_to_idx(label: str) -> int:
    idx = 0
    for ch in label:
        idx = idx * 26 + (ord(ch) - 64)
    return idx
