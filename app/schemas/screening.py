"""
app/schemas/screening.py
========================
Skema Pydantic Kontrak API untuk Skrining Depresi Multimodal dan Lifecycle Persistence.
Mengikuti Milestone 2.5 Audit Konsistensi Ilmiah:
1. risk_probability_proxy (score-to-probability sigmoid proxy)
2. feature_xai dipisah & diset null hingga evaluasi continuous KernelSHAP diimplementasikan
3. Mendukung Lifecycle Skrining (POST /screenings, POST /screenings/{id}/predict, GET /screenings/{id}, GET /screenings)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------
# 1. LIFECYCLE REQUEST SCHEMAS
# ------------------------------------------------------------------------------
class ScreeningCreateRequest(BaseModel):
    participant_id: str = Field(
        ...,
        description="Identitas subjek atau pasien (misal: 'P600')",
        examples=["P600"],
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata sesi klinis (misal: catatan petugas, platform klien)",
    )


class ScreeningCreateResponse(BaseModel):
    screening_id: str = Field(..., description="ID sesi skrining unik terdaftar")
    participant_id: str = Field(..., description="ID subjek skrining")
    status: str = Field(default="created", description="Status sesi skrining")
    created_at: str = Field(..., description="Waktu pendaftaran sesi (ISO-8601)")


class ScreeningExecuteRequest(BaseModel):
    text_features: Union[List[float], List[List[float]]] = Field(
        ...,
        description="Fitur teks Sentence-BERT: 1D array (768,) atau representasi sequence (T, 768)",
    )
    audio_features: Union[List[float], List[List[float]]] = Field(
        ...,
        description="Fitur audio log-mel/akustik: 1D array (248,) atau representasi sequence (T, 80)",
    )
    visual_features: Union[List[float], List[List[float]], List[List[List[float]]]] = Field(
        ...,
        description="Fitur visual wajah/AU/gaze: 1D array (75,) atau representasi 3D (T, 72, 3)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata tambahan saat eksekusi inferensi",
    )


class ScreeningRequest(BaseModel):
    """Backward-compatible all-in-one request contract."""
    participant_id: str = Field(..., description="ID subjek skrining", examples=["P600"])
    screening_id: Optional[str] = Field(default=None, description="ID unik sesi skrining")
    text_features: Union[List[float], List[List[float]]] = Field(..., description="Fitur teks")
    audio_features: Union[List[float], List[List[float]]] = Field(..., description="Fitur audio")
    visual_features: Union[List[float], List[List[float]], List[List[List[float]]]] = Field(..., description="Fitur visual")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata sesi")


# ------------------------------------------------------------------------------
# 2. PREDICTION & XAI SCHEMAS
# ------------------------------------------------------------------------------
class PredictionResult(BaseModel):
    phq8_score: float = Field(..., description="Estimasi skor PHQ-8 model (rentang klinis 0.0 – 24.0)")
    risk_probability_proxy: float = Field(
        ...,
        description="Score-to-probability sigmoid proxy (pusat di PHQ-8=10.0, bukan formal calibrated probability)",
    )
    binary_prediction: str = Field(
        ...,
        description="Indikator biner ('depression_risk' jika PHQ-8 >= 10.0, 'non_depression' jika < 10.0)",
    )
    risk_category: str = Field(
        ...,
        description="Stratifikasi risiko klinis ('low', 'borderline', 'moderate', 'high')",
    )


class PatientShapValues(BaseModel):
    text: float = Field(..., description="Nilai kontribusi Shapley modalitas Teks (+/-)")
    visual: float = Field(..., description="Nilai kontribusi Shapley modalitas Visual (+/-)")
    audio: float = Field(..., description="Nilai kontribusi Shapley modalitas Audio (+/-)")


class PatientModalityAttribution(BaseModel):
    text_pct: float = Field(..., description="Persentase kontribusi relatif Teks (%)")
    visual_pct: float = Field(..., description="Persentase kontribusi relatif Visual (%)")
    audio_pct: float = Field(..., description="Persentase kontribusi relatif Audio (%)")
    dominant_modality: str = Field(..., description="Modalitas dengan kontribusi terbesar ('text', 'visual', 'audio')")
    shap_values: PatientShapValues = Field(..., description="Nilai eksak Shapley value per modalitas")


class GlobalReferenceAttribution(BaseModel):
    cohort: str = Field(default="E-DAIC Test Set (N=56)", description="Nama kohort rujukan global")
    text_pct: float = Field(default=61.8, description="Rata-rata kontribusi global Teks (%)")
    visual_pct: float = Field(default=33.8, description="Rata-rata kontribusi global Visual (%)")
    audio_pct: float = Field(default=4.4, description="Rata-rata kontribusi global Audio (%)")
    dominant_modality: str = Field(default="text", description="Modalitas dominan kohort global")


class QualityFlags(BaseModel):
    low_text_information: bool = Field(
        ...,
        description="Peringatan jika variansi/kuantitas teks sangat rendah (misal jawaban terlalu singkat)",
    )
    poor_audio_quality: bool = Field(
        ...,
        description="Peringatan jika energi atau variabilitas sinyal vokal mendekati nol",
    )
    poor_visual_quality: bool = Field(
        ...,
        description="Peringatan jika sinyal dinamika wajah statis / minim variasi",
    )
    details: Optional[str] = Field(default=None, description="Detail anomali sinyal input jika ada")


class ModelMetadata(BaseModel):
    model_id: str = Field(..., description="ID tipe model ('centralized', 'fl_iid', 'fl_non_iid')")
    name: str
    version: str
    architecture: str
    checkpoint: str
    checksum: str = Field(..., description="SHA-256 checksum checkpoint bobot model")
    xai_baseline: str = Field(..., description="File baseline latents yang cocok dengan model aktif")
    device: str


class XAIMetadata(BaseModel):
    method: str = Field(default="exact_modality_shapley_coalition", description="Metode komputasi XAI eksak")
    num_coalitions: int = Field(default=8, description="Jumlah evaluasi koalisi (2^3 = 8)")
    efficiency_axiom_verified: bool = Field(default=True, description="Verifikasi aksioma efisiensi Shapley")
    baseline_expected_value: float = Field(..., description="Nilai ekspektasi baseline kohort E[f(X)] model aktif")


# ------------------------------------------------------------------------------
# 3. RESPONSE CONTRACT
# ------------------------------------------------------------------------------
class ScreeningResponse(BaseModel):
    screening_id: str = Field(..., description="ID sesi skrining unik")
    participant_id: str = Field(..., description="ID subjek skrining")
    status: str = Field(default="completed", description="Status sesi skrining ('created', 'completed', 'failed')")
    prediction: Optional[PredictionResult] = Field(default=None, description="Hasil prediksi skor PHQ-8 dan stratifikasi")
    patient_modality_attribution: Optional[PatientModalityAttribution] = Field(
        default=None, description="Atribusi modalitas pasien eksak (bukan hardcoded)"
    )
    global_reference: GlobalReferenceAttribution = Field(
        default_factory=GlobalReferenceAttribution,
        description="Data rujukan kohort global sebagai perbandingan kontekstual",
    )
    feature_xai: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Continuous feature SHAP (diaudit: diset null hingga pipeline KernelSHAP selesai)",
    )
    quality_flags: Optional[QualityFlags] = Field(default=None, description="Indikator validitas dan kualitas sinyal multimodal")
    clinical_narrative: Optional[str] = Field(default=None, description="Laporan narasi sintetis untuk tenaga medis/psikolog")
    model: Optional[ModelMetadata] = Field(default=None, description="Metadata versi dan arsitektur model")
    xai: Optional[XAIMetadata] = Field(default=None, description="Metadata metode komputasi Explainable AI")
    disclaimer: Optional[str] = Field(
        default="Hasil ini merupakan instrumen triase skrining awal berbasis kecerdasan buatan terfederasi. Penjelasan ini BUKAN diagnosis klinis final dan WAJIB ditinjau bersama asesmen klinis profesional.",
        description="Pernyataan batasan etika dan disclaimer medis resmi",
    )
    created_at: Optional[str] = Field(default=None, description="Waktu sesi dibuat")
    completed_at: Optional[str] = Field(default=None, description="Waktu komputasi selesai")


# ------------------------------------------------------------------------------
# 4. HISTORY & SUMMARY SCHEMAS
# ------------------------------------------------------------------------------
class ScreeningSummaryItem(BaseModel):
    screening_id: str
    participant_id: str
    status: str
    phq8_score: Optional[float] = None
    risk_category: Optional[str] = None
    binary_prediction: Optional[str] = None
    dominant_modality: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class ScreeningListResponse(BaseModel):
    total: int
    screenings: List[ScreeningSummaryItem]
