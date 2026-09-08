"""
ORM models — map exactly to the SQLite schema in the system prompt.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    report_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    completed: Mapped[bool] = mapped_column(Boolean, default=False)

    photos: Mapped[list[Photo]] = relationship("Photo", back_populates="batch")
    selections: Mapped[list[Selection]] = relationship("Selection", back_populates="batch")


class Photo(Base):
    __tablename__ = "photos"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # sha256 / perceptual hash
    filepath: Mapped[str] = mapped_column(Text, nullable=False)
    original_filename: Mapped[str | None] = mapped_column(Text, nullable=True)
    batch_id: Mapped[str] = mapped_column(String, ForeignKey("batches.id"), nullable=False)
    activity_cluster_id: Mapped[str | None] = mapped_column(String, nullable=True)

    captured_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Technical quality scores
    blur_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    exposure_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    obstruction_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    passed_technical_filter: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # ML scores
    ml_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rank_in_cluster: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Feature vector stored as BLOB (numpy array serialised with np.save)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=func.now())
    processing_status: Mapped[str] = mapped_column(String, default="pending")  # pending|processing|done|error

    batch: Mapped[Batch] = relationship("Batch", back_populates="photos")
    selections: Mapped[list[Selection]] = relationship("Selection", back_populates="photo")


class Selection(Base):
    __tablename__ = "selections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    photo_id: Mapped[str] = mapped_column(String, ForeignKey("photos.id"), nullable=False)
    batch_id: Mapped[str] = mapped_column(String, ForeignKey("batches.id"), nullable=False)
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False)  # 1=selected, 0=rejected
    selected_at: Mapped[datetime] = mapped_column(default=func.now())

    photo: Mapped[Photo] = relationship("Photo", back_populates="selections")
    batch: Mapped[Batch] = relationship("Batch", back_populates="selections")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trained_at: Mapped[datetime] = mapped_column(default=func.now())
    training_sample_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    validation_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
