"""
Core configuration loader — reads config.yaml and provides a singleton Settings object.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


class PathSettings(BaseModel):
    watch_folder: str = "data/watch_inbox"
    upload_folder: str = "data/uploads"
    models_folder: str = "data/models"
    database: str = "data/photos.db"


class QualityFilterSettings(BaseModel):
    blur_threshold: float = 100.0
    exposure_over_threshold: float = 0.15
    exposure_under_threshold: float = 0.15
    obstruction_edge_density: float = 0.05


class ClusteringSettings(BaseModel):
    time_gap_minutes: int = 15
    perceptual_hash_distance: int = 10


class MLSettings(BaseModel):
    backbone: Literal["mobilenet_v3_small", "resnet18"] = "mobilenet_v3_small"
    embedding_dim: int = 576
    cold_start_min_samples: int = 50
    retrain_every_n_batches: int = 5
    classifier: Literal["logistic", "mlp"] = "logistic"


class UploadSettings(BaseModel):
    chunk_size_bytes: int = 2_097_152  # 2 MB
    max_batch_size: int = 900


class ShortlistSettings(BaseModel):
    top_n_per_batch: int = 20


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    server: ServerSettings = ServerSettings()
    paths: PathSettings = PathSettings()
    quality_filter: QualityFilterSettings = QualityFilterSettings()
    clustering: ClusteringSettings = ClusteringSettings()
    ml: MLSettings = MLSettings()
    upload: UploadSettings = UploadSettings()
    shortlist: ShortlistSettings = ShortlistSettings()

    # Resolved absolute base dir (project root)
    base_dir: Path = Path(__file__).resolve().parent.parent.parent

    def abs_path(self, relative: str) -> Path:
        return self.base_dir / relative

    @property
    def db_url(self) -> str:
        db = self.abs_path(self.paths.database)
        db.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{db}"

    @property
    def db_url_sync(self) -> str:
        db = self.abs_path(self.paths.database)
        db.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db}"


@lru_cache()
def get_settings() -> Settings:
    config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return Settings(**raw)
    return Settings()
