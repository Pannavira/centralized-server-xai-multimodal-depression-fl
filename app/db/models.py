"""
app/db/models.py
================
Model Tabel Database Relasional (SQLAlchemy 2.0):
1. screenings       : Registrasi sesi dan status skrining
2. screening_results : Nilai prediksi PHQ-8, stratifikasi risiko, proxy probabilitas
3. xai_results      : Hasil atribusi Exact Shapley & serialized explanation JSON
4. model_versions   : Registry metadata model, tipe, checkpoint, checksum SHA-256
5. audit_logs       : Jejak audit riwayat event sistem & lifecycle
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    JSON,
    Text,
)
from sqlalchemy.orm import relationship

from app.db.session import Base


def utc_now():
    return datetime.now(timezone.utc)


class ScreeningModel(Base):
    __tablename__ = "screenings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    screening_id = Column(String(64), unique=True, index=True, nullable=False)
    participant_id = Column(String(64), index=True, nullable=False)
    status = Column(String(32), default="created", nullable=False)  # "created", "completed", "failed"
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    result = relationship("ScreeningResultModel", back_populates="screening", uselist=False, cascade="all, delete-orphan")
    xai_result = relationship("XAIResultModel", back_populates="screening", uselist=False, cascade="all, delete-orphan")
    audit_logs = relationship("AuditLogModel", back_populates="screening", cascade="all, delete-orphan")


class ScreeningResultModel(Base):
    __tablename__ = "screening_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    screening_id = Column(String(64), ForeignKey("screenings.screening_id", ondelete="CASCADE"), unique=True, index=True, nullable=False)
    phq8_score = Column(Float, nullable=False)
    risk_category = Column(String(32), nullable=False)  # "low", "borderline", "moderate", "high"
    binary_prediction = Column(String(32), nullable=False)  # "depression_risk", "non_depression"
    risk_probability_proxy = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    screening = relationship("ScreeningModel", back_populates="result")


class XAIResultModel(Base):
    __tablename__ = "xai_results"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    screening_id = Column(String(64), ForeignKey("screenings.screening_id", ondelete="CASCADE"), unique=True, index=True, nullable=False)
    text_shap = Column(Float, nullable=False)
    visual_shap = Column(Float, nullable=False)
    audio_shap = Column(Float, nullable=False)
    dominant_modality = Column(String(32), nullable=False)
    baseline_value = Column(Float, nullable=False)
    method = Column(String(64), default="exact_modality_shapley_coalition", nullable=False)
    explanation_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    screening = relationship("ScreeningModel", back_populates="xai_result")


class ModelVersionModel(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_name = Column(String(128), nullable=False)
    model_type = Column(String(64), index=True, nullable=False)  # "centralized", "fl_iid", "fl_non_iid"
    checkpoint = Column(String(256), nullable=False)
    checksum = Column(String(64), nullable=False)  # SHA-256
    xai_baseline = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)


class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    screening_id = Column(String(64), ForeignKey("screenings.screening_id", ondelete="CASCADE"), nullable=True, index=True)
    event = Column(String(64), nullable=False)  # "SCREENING_CREATED", "INFERENCE_COMPLETED", etc.
    model_version = Column(String(64), nullable=True)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False)

    screening = relationship("ScreeningModel", back_populates="audit_logs")
