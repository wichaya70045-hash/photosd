"""
SQLAlchemy async engine + session factory + base model.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.db_url,
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:  # type: ignore[override]
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create all tables on startup."""
    from app.models import db_models  # noqa: F401 — registers ORM models

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
