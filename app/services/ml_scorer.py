"""
ML Scorer — Phase 2
  • MobileNetV3-Small (frozen backbone) extracts 576-dim feature vectors
  • LogisticRegression classifier trained on user selections
  • Cold-start: falls back to rule-based score if < N training samples
"""
from __future__ import annotations

import io
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
ml_cfg = settings.ml

# ---------------------------------------------------------------------------
# Lazy-loaded torch / torchvision to avoid heavy import at startup
# ---------------------------------------------------------------------------
_torch = None
_transforms = None
_model = None  # shared backbone instance


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_transforms():
    global _transforms
    if _transforms is None:
        from torchvision import transforms
        _transforms = transforms
    return _transforms


def _get_backbone():
    """Load and cache the pretrained backbone (frozen)."""
    global _model
    if _model is not None:
        return _model

    torch = _get_torch()
    import torchvision.models as tvm

    with torch.no_grad():
        if ml_cfg.backbone == "mobilenet_v3_small":
            base = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
            # Remove final classifier → keep adaptive avg pool output (576-d)
            base.classifier = torch.nn.Identity()
        else:
            base = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
            base.fc = torch.nn.Identity()

        base.eval()
        for p in base.parameters():
            p.requires_grad_(False)

    _model = base
    logger.info("Backbone %s loaded (frozen)", ml_cfg.backbone)
    return _model


def _fallback_embedding(img: Image.Image) -> np.ndarray:
    """Generate a 576-dim spatial feature vector when PyTorch is unavailable."""
    thumb = np.array(img.convert("L").resize((24, 24), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    vec = thumb.flatten()
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec


def extract_embedding(filepath: Path) -> Optional[np.ndarray]:
    """Extract feature vector from one image file. Returns None on error."""
    try:
        img = Image.open(filepath).convert("RGB")
    except Exception as exc:
        logger.error("Could not open image for embedding: %s: %s", filepath, exc)
        return None

    try:
        torch = _get_torch()
        transforms = _get_transforms()
        backbone = _get_backbone()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        tensor = preprocess(img).unsqueeze(0)  # [1, C, H, W]

        with torch.no_grad():
            feat = backbone(tensor)  # [1, D]

        return feat.squeeze(0).numpy()

    except Exception:
        # Fallback to pure numpy/PIL 576-dim feature vector
        return _fallback_embedding(img)



def embedding_to_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob))


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class PhotoClassifier:
    """Wrapper around scikit-learn classifier with save/load."""

    def __init__(self):
        self._clf = None

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train and return validation accuracy (stratified 80/20 split)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if len(np.unique(y)) < 2:
            logger.warning("Only one class in training data; skipping training")
            return 0.0

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )

        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(X_train, y_train)
        val_score = float(clf.score(X_val, y_val))

        self._clf = (scaler, clf)
        logger.info("Classifier trained — val accuracy: %.3f on %d samples", val_score, len(X))
        return val_score

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of class=1 for each sample."""
        if self._clf is None:
            raise RuntimeError("Classifier not trained")
        scaler, clf = self._clf
        X_scaled = scaler.transform(X)
        return clf.predict_proba(X_scaled)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._clf, f)
        logger.info("Classifier saved to %s", path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        with open(path, "rb") as f:
            self._clf = pickle.load(f)
        logger.info("Classifier loaded from %s", path)
        return True

    @property
    def is_ready(self) -> bool:
        return self._clf is not None


# ---------------------------------------------------------------------------
# Singleton classifier instance
# ---------------------------------------------------------------------------
_classifier: Optional[PhotoClassifier] = None


def get_classifier() -> PhotoClassifier:
    global _classifier
    if _classifier is None:
        _classifier = PhotoClassifier()
        # Try to load the active model
        active_model_path = settings.abs_path(settings.paths.models_folder) / "active_classifier.pkl"
        _classifier.load(active_model_path)
    return _classifier


def score_photo(embedding: np.ndarray) -> Optional[float]:
    """
    Score a photo embedding. Returns None in cold-start mode.
    """
    clf = get_classifier()
    if not clf.is_ready:
        return None
    try:
        score = float(clf.predict_proba(embedding.reshape(1, -1))[0])
        return score
    except Exception as exc:
        logger.error("Scoring failed: %s", exc)
        return None
