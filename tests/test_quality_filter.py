"""
Unit tests for the technical quality filter.
"""
import sys
import unittest
from pathlib import Path

# Ensure project root is on path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from app.services.quality_filter import (
    compute_blur_score,
    compute_exposure_score,
    compute_obstruction_flag,
)


def make_gray(val: int = 128, size=(200, 300)) -> np.ndarray:
    """Create a uniform grey image."""
    return np.full(size, val, dtype=np.uint8)


def make_rgb(r=128, g=128, b=128, size=(200, 300)) -> np.ndarray:
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def make_textured_gray(size=(200, 300)) -> np.ndarray:
    """Image with strong edges (checker pattern)."""
    img = np.zeros(size, dtype=np.uint8)
    for i in range(0, size[0], 20):
        for j in range(0, size[1], 20):
            if (i // 20 + j // 20) % 2 == 0:
                img[i:i+20, j:j+20] = 255
    return img


class TestQualityFilter(unittest.TestCase):
    def test_uniform_image_is_blurry(self):
        """A completely uniform image has zero Laplacian variance → blurry."""
        gray = make_gray(128)
        score = compute_blur_score(gray)
        self.assertLess(score, 10.0, f"Expected low blur score, got {score}")

    def test_textured_image_is_sharp(self):
        """A checker-board image has very high Laplacian variance → sharp."""
        gray = make_textured_gray()
        score = compute_blur_score(gray)
        self.assertGreater(score, 1000.0, f"Expected high blur score, got {score}")

    def test_score_is_nonnegative(self):
        gray = make_gray()
        self.assertGreaterEqual(compute_blur_score(gray), 0)

    def test_overexposed(self):
        """Nearly white image → high exposure score."""
        gray = make_gray(255)
        rgb = make_rgb(r=255, g=255, b=255)
        score = compute_exposure_score(rgb, gray)
        self.assertGreater(score, 0.5, f"Expected high exposure score, got {score}")

    def test_underexposed(self):
        """Nearly black image → high exposure score."""
        gray = make_gray(0)
        rgb = make_rgb(r=0, g=0, b=0)
        score = compute_exposure_score(rgb, gray)
        self.assertGreater(score, 0.5)

    def test_normal_exposure(self):
        """Mid-grey image → low exposure score."""
        gray = make_gray(128)
        rgb = make_rgb(r=128, g=128, b=128)
        score = compute_exposure_score(rgb, gray)
        self.assertLess(score, 0.05, f"Expected low exposure score, got {score}")

    def test_uniform_centre_flagged_obstruction(self):
        """Uniform image has very low edge density → obstruction flagged."""
        gray = make_gray(200)
        rgb = make_rgb(r=200, g=200, b=200)
        flagged = compute_obstruction_flag(rgb, gray)
        self.assertTrue(flagged)

    def test_textured_centre_not_flagged_obstruction(self):
        """Textured image has high edge density → not flagged."""
        gray = make_textured_gray()
        rgb = np.stack([gray, gray, gray], axis=-1)
        flagged = compute_obstruction_flag(rgb, gray)
        self.assertFalse(flagged)


if __name__ == "__main__":
    unittest.main()

