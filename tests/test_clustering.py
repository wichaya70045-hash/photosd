"""
Unit tests for activity clustering and pHash distance.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np
from app.services.clustering import cluster_by_perceptual_hash, assign_clusters


class TestClustering(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path("data/test_clustering_tmp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Create two different images
        self.img1_path = self.temp_dir / "img1.jpg"
        self.img2_path = self.temp_dir / "img2.jpg"
        self.img3_path = self.temp_dir / "img3.jpg"

        # img1 and img2 are burst shots of the same scene
        arr1 = np.full((100, 100, 3), 200, dtype=np.uint8)
        arr1[20:60, 20:60] = 50
        Image.fromarray(arr1).save(self.img1_path)

        arr2 = arr1.copy()
        arr2[21:61, 21:61] = 52
        Image.fromarray(arr2).save(self.img2_path)

        # img3 is completely dark (different scene)
        arr3 = np.zeros((100, 100, 3), dtype=np.uint8)
        Image.fromarray(arr3).save(self.img3_path)

    def tearDown(self):
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_phash_clustering_groups_similar_photos(self):
        photos = [
            ("p1", self.img1_path),
            ("p2", self.img2_path),
            ("p3", self.img3_path),
        ]
        clusters = cluster_by_perceptual_hash(photos)
        self.assertEqual(len(clusters), 3)
        # p1 and p2 should have the same cluster ID
        self.assertEqual(clusters["p1"], clusters["p2"])
        # p3 should have a different cluster ID
        self.assertNotEqual(clusters["p1"], clusters["p3"])

    def test_assign_clusters_fallback(self):
        photos = [
            ("p1", self.img1_path),
            ("p2", self.img2_path),
        ]
        clusters = assign_clusters(photos)
        self.assertIn("p1", clusters)
        self.assertIn("p2", clusters)


if __name__ == "__main__":
    unittest.main()
