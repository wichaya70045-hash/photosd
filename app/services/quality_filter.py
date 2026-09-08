"""
Technical quality filter:
  - Blur detection (Laplacian variance)
  - Exposure detection (histogram analysis)
  - Obstruction detection (edge density in centre crop)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
qf = settings.quality_filter

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    import numpy as np
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class QualityResult:
    blur_score: float
    exposure_score: float
    obstruction_flag: bool
    passed: bool


def _load_image_gray_and_rgb(filepath: Path):
    """Load image and return grayscale & rgb numpy arrays."""
    if CV2_AVAILABLE:
        bgr = cv2.imread(str(filepath))
        if bgr is not None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            return gray, bgr
    if SCIPY_AVAILABLE:
        try:
            with Image.open(filepath) as pil_img:
                rgb = np.array(pil_img.convert("RGB"))
                gray = np.array(pil_img.convert("L"))
                return gray, rgb
        except Exception as exc:
            logger.warning("Could not open image %s: %s", filepath, exc)
    return None, None


def compute_blur_score(img_gray: np.ndarray) -> float:
    """
    Laplacian variance — higher = sharper.
    Below threshold → blurry.
    """
    if CV2_AVAILABLE:
        lap = cv2.Laplacian(img_gray, cv2.CV_64F)
        return float(lap.var())
    elif SCIPY_AVAILABLE:
        # Standard 3x3 discrete Laplacian kernel
        lap_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        lap = ndimage.convolve(img_gray.astype(np.float64), lap_kernel)
        return float(lap.var())
    return 500.0


def compute_exposure_score(img_rgb: np.ndarray, img_gray: np.ndarray) -> float:
    """
    Returns a penalty score [0, 1].
    High value means severely over- or under-exposed.
    Combines fraction of near-white pixels (over) and near-black pixels (under).
    """
    total = img_gray.size
    if total == 0:
        return 0.0
    over_frac = float(np.sum(img_gray >= 250) / total)
    under_frac = float(np.sum(img_gray <= 5) / total)
    return max(over_frac, under_frac)


def compute_obstruction_flag(img_rgb: np.ndarray, img_gray: np.ndarray) -> bool:
    """
    Detect obstruction (e.g. finger over lens) via low edge density in centre crop.
    A uniformly coloured region in the centre with very low edge density is suspicious.
    """
    h, w = img_gray.shape[:2]
    # Centre 40% crop
    cy, cx = h // 2, w // 2
    crop_h, crop_w = int(h * 0.4), int(w * 0.4)
    centre_gray = img_gray[cy - crop_h // 2 : cy + crop_h // 2, cx - crop_w // 2 : cx + crop_w // 2]
    if centre_gray.size == 0:
        return False

    if CV2_AVAILABLE:
        edges = cv2.Canny(centre_gray, 50, 150)
        edge_density = float(np.count_nonzero(edges) / edges.size)
    elif SCIPY_AVAILABLE:
        # Sobel edge magnitude
        sx = ndimage.sobel(centre_gray.astype(float), axis=0)
        sy = ndimage.sobel(centre_gray.astype(float), axis=1)
        mag = np.hypot(sx, sy)
        edge_density = float(np.count_nonzero(mag > 50) / mag.size)
    else:
        edge_density = 0.1

    return edge_density < qf.obstruction_edge_density



def run_quality_filter(filepath: Path) -> QualityResult | None:
    """Run the full technical quality pipeline on one image file."""
    gray, rgb = _load_image_gray_and_rgb(filepath)
    if gray is None or rgb is None:
        return None

    blur = compute_blur_score(gray)
    exposure = compute_exposure_score(rgb, gray)
    obstruction = compute_obstruction_flag(rgb, gray)

    passed = (
        blur >= qf.blur_threshold
        and exposure <= max(qf.exposure_over_threshold, qf.exposure_under_threshold)
        and not obstruction
    )

    return QualityResult(
        blur_score=blur,
        exposure_score=exposure,
        obstruction_flag=obstruction,
        passed=passed,
    )

