"""
app/db/session.py
=================
Konfigurasi Engine & Session Database SQLAlchemy 2.0.
Mendukung koneksi PostgreSQL (melalui DATABASE_URL) dengan fallback lokal SQLite.
"""

import logging
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.config import settings

logger = logging.getLogger("DATABASE")

DATABASE_URL = settings.database_url

# Pastikan folder penyimpanan SQLite dibuat jika menggunakan file lokal
if DATABASE_URL.startswith("sqlite"):
    db_path = DATABASE_URL.replace("sqlite:///", "")
    if db_path and not Path(db_path).is_absolute():
        full_p = Path(settings.base_dir) / db_path
        full_p.parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Dependency injection generator untuk FastAPI router."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Inisialisasi skema tabel jika belum ada di database."""
    try:
        from app.db import models  # noqa: F401
        Base.metadata.create_all(bind=engine)
        logger.info(f"Database tables initialized successfully on: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}", exc_info=True)
