"""
app/services/screening_service.py
=================================
Orkestrasi Service End-to-End untuk Skrining Multimodal Depresi Klinis,
Explainable AI (Exact Modality Shapley), dan Siklus Hidup Persistensi Database (Milestone 3).

Prinsip Privasi (Privacy-First Policy):
- Seluruh tensor fitur multimodal mentah (512x80, 512x72x3, 512x768) hanya diproses
  di memori kerja selama inferensi, lalu segera dibuang.
- Database HANYA menyimpan ringkasan prediksi, nilai XAI, metadata, dan jejak audit.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy.orm import Session

from app.config import settings
from app.core.inference import inference_pipeline
from app.core.preprocessing import preprocessor
from app.core.xai_explainer import xai_explainer
from app.db.models import (
    AuditLogModel,
    ModelVersionModel,
    ScreeningModel,
    ScreeningResultModel,
    XAIResultModel,
)
from app.schemas.screening import (
    GlobalReferenceAttribution,
    ModelMetadata,
    PatientModalityAttribution,
    PatientShapValues,
    PredictionResult,
    QualityFlags,
    ScreeningCreateRequest,
    ScreeningCreateResponse,
    ScreeningExecuteRequest,
    ScreeningListResponse,
    ScreeningRequest,
    ScreeningResponse,
    ScreeningSummaryItem,
)
from app.services.model_service import model_service

logger = logging.getLogger("SCREENING_SERVICE")

DISCLAIMER_TEXT = (
    "Hasil ini merupakan instrumen dukungan keputusan triase skrining awal berbasis kecerdasan "
    "buatan terfederasi dan BUKAN diagnosis klinis final. Penjelasan XAI dimaksudkan untuk transparansi "
    "penalaran model. Penilaian akhir wajib dilakukan oleh tenaga profesional kesehatan jiwa (psikolog/psikiater)."
)


class ScreeningService:
    """
    Service Layer untuk menangani siklus hidup skrining (Lifecycle & Persistence).
    """

    def create_screening_session(
        self,
        request: ScreeningCreateRequest,
        db: Session,
    ) -> ScreeningCreateResponse:
        """Langkah 1 Lifecycle: Mendaftarkan sesi skrining baru (status: created)."""
        screening_id = f"SCR-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:5].upper()}"

        record = ScreeningModel(
            screening_id=screening_id,
            participant_id=request.participant_id,
            status="created",
            metadata_json=request.metadata or {},
            created_at=datetime.now(timezone.utc),
        )
        db.add(record)

        audit = AuditLogModel(
            screening_id=screening_id,
            event="SCREENING_CREATED",
            model_version=settings.model_type,
            details=f"Sesi skrining dibuat untuk pasien {request.participant_id}",
            timestamp=datetime.now(timezone.utc),
        )
        db.add(audit)
        db.commit()
        db.refresh(record)

        return ScreeningCreateResponse(
            screening_id=record.screening_id,
            participant_id=record.participant_id,
            status=record.status,
            created_at=record.created_at.isoformat() + "Z",
        )

    def execute_screening_by_id(
        self,
        screening_id: str,
        features_req: ScreeningExecuteRequest,
        db: Session,
    ) -> ScreeningResponse:
        """Langkah 2 Lifecycle: Menjalankan inferensi dan XAI untuk sesi skrining yang ada."""
        screening = db.query(ScreeningModel).filter(ScreeningModel.screening_id == screening_id).first()
        if not screening:
            raise KeyError(f"Sesi skrining dengan ID '{screening_id}' tidak ditemukan.")

        participant_id = screening.participant_id
        response = self._run_inference_pipeline(
            screening_id=screening_id,
            participant_id=participant_id,
            audio_features=features_req.audio_features,
            visual_features=features_req.visual_features,
            text_features=features_req.text_features,
            db=db,
            screening_record=screening,
        )
        return response

    def get_screening_by_id(self, screening_id: str, db: Session) -> ScreeningResponse:
        """Langkah 3 Lifecycle: Mengambil detail rekaman skrining dan XAI yang tersimpan."""
        screening = db.query(ScreeningModel).filter(ScreeningModel.screening_id == screening_id).first()
        if not screening:
            raise KeyError(f"Sesi skrining '{screening_id}' tidak ditemukan.")

        if screening.status != "completed" or not screening.result or not screening.xai_result:
            return ScreeningResponse(
                screening_id=screening.screening_id,
                participant_id=screening.participant_id,
                status=screening.status,
                created_at=screening.created_at.isoformat() if screening.created_at else None,
                completed_at=screening.completed_at.isoformat() if screening.completed_at else None,
                prediction=None,
                patient_modality_attribution=None,
                quality_flags=None,
                clinical_narrative=None,
                model=None,
                xai=None,
            )

        # Rekonstruksi ScreeningResponse dari database
        pred_rec = screening.result
        xai_rec = screening.xai_result
        xai_json = xai_rec.explanation_json

        pred_res = PredictionResult(
            phq8_score=pred_rec.phq8_score,
            risk_probability_proxy=pred_rec.risk_probability_proxy,
            binary_prediction=pred_rec.binary_prediction,
            risk_category=pred_rec.risk_category,
        )

        modality_attr = PatientModalityAttribution(
            text_pct=xai_json.get("text_pct", 0.0),
            visual_pct=xai_json.get("visual_pct", 0.0),
            audio_pct=xai_json.get("audio_pct", 0.0),
            dominant_modality=xai_rec.dominant_modality,
            shap_values=PatientShapValues(
                text=xai_rec.text_shap,
                visual=xai_rec.visual_shap,
                audio=xai_rec.audio_shap,
            ),
        )

        quality_flags = QualityFlags(**xai_json.get("quality_flags", {
            "low_text_information": False,
            "poor_audio_quality": False,
            "poor_visual_quality": False,
        }))

        model_meta = ModelMetadata(**xai_json.get("model", {
            "model_id": settings.model_type,
            "name": "FED-MIND Multimodal",
            "version": settings.app_version,
            "architecture": "Centralized Multimodal Baseline",
            "checkpoint": "model.pt",
            "checksum": "persisted",
            "xai_baseline": "xai_baseline.pt",
            "device": "cpu",
        }))

        return ScreeningResponse(
            screening_id=screening.screening_id,
            participant_id=screening.participant_id,
            status=screening.status,
            prediction=pred_res,
            patient_modality_attribution=modality_attr,
            global_reference=GlobalReferenceAttribution(),
            feature_xai=None,
            quality_flags=quality_flags,
            clinical_narrative=xai_json.get("clinical_narrative", ""),
            model=model_meta,
            xai=xai_json.get("xai_meta", {
                "method": xai_rec.method,
                "num_coalitions": 8,
                "efficiency_axiom_verified": True,
                "baseline_expected_value": xai_rec.baseline_value,
            }),
            disclaimer=DISCLAIMER_TEXT,
            created_at=screening.created_at.isoformat() + "Z",
            completed_at=screening.completed_at.isoformat() + "Z" if screening.completed_at else None,
        )

    def list_screenings(
        self,
        participant_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        db: Optional[Session] = None,
    ) -> ScreeningListResponse:
        """Langkah 4 Lifecycle: Riwayat sesi skrining untuk antarmuka History mobile/dashboard."""
        query = db.query(ScreeningModel)
        if participant_id:
            query = query.filter(ScreeningModel.participant_id == participant_id)
        if status:
            query = query.filter(ScreeningModel.status == status)

        total = query.count()
        records = query.order_by(ScreeningModel.created_at.desc()).offset(offset).limit(limit).all()

        items = []
        for r in records:
            phq = r.result.phq8_score if r.result else None
            risk = r.result.risk_category if r.result else None
            bin_p = r.result.binary_prediction if r.result else None
            dom = r.xai_result.dominant_modality if r.xai_result else None

            items.append(
                ScreeningSummaryItem(
                    screening_id=r.screening_id,
                    participant_id=r.participant_id,
                    status=r.status,
                    phq8_score=phq,
                    risk_category=risk,
                    binary_prediction=bin_p,
                    dominant_modality=dom,
                    created_at=r.created_at.isoformat() + "Z",
                    completed_at=r.completed_at.isoformat() + "Z" if r.completed_at else None,
                )
            )

        return ScreeningListResponse(total=total, screenings=items)

    def process_screening(
        self,
        request: ScreeningRequest,
        db: Optional[Session] = None,
    ) -> ScreeningResponse:
        """Backward-compatible endpoint: Menangani pendaftaran otomatis, inferensi, dan persistensi."""
        screening_id = request.screening_id or f"SCR-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:5].upper()}"

        screening_rec = None
        if db:
            screening_rec = db.query(ScreeningModel).filter(ScreeningModel.screening_id == screening_id).first()
            if not screening_rec:
                screening_rec = ScreeningModel(
                    screening_id=screening_id,
                    participant_id=request.participant_id,
                    status="created",
                    metadata_json=request.metadata or {},
                    created_at=datetime.now(timezone.utc),
                )
                db.add(screening_rec)
                db.commit()
                db.refresh(screening_rec)

        response = self._run_inference_pipeline(
            screening_id=screening_id,
            participant_id=request.participant_id,
            audio_features=request.audio_features,
            visual_features=request.visual_features,
            text_features=request.text_features,
            db=db,
            screening_record=screening_rec,
        )
        return response

    def _run_inference_pipeline(
        self,
        screening_id: str,
        participant_id: str,
        audio_features: Any,
        visual_features: Any,
        text_features: Any,
        db: Optional[Session],
        screening_record: Optional[ScreeningModel],
    ) -> ScreeningResponse:
        """Pipeline internal: inferensi model, XAI exact, dan persistensi DB."""
        if not model_service.is_loaded():
            raise RuntimeError("Model FED-MIND belum dimuat ke memory server.")

        model = model_service.get_model()
        device = model_service.get_device()

        # 1. Preprocessing & Adaptasi Tensor
        tensors, q_flags_dict = preprocessor.prepare_multimodal_tensors(
            audio_input=audio_features,
            visual_input=visual_features,
            text_input=text_features,
            device=device,
        )
        quality_flags = QualityFlags(**q_flags_dict)

        # 2. Forward Pass Inferensi
        prediction_result, features = inference_pipeline.predict(
            model=model,
            audio_tensor=tensors["audio"],
            visual_tensor=tensors["visual"],
            text_tensor=tensors["text"],
        )

        # 3. Komputasi Exact Modality Shapley Values (8 Koalisi Game Theory)
        audio_lat = features["audio_latent"]
        visual_lat = features["visual_latent"]
        text_lat = features["text_latent"]

        shap_vals, attr_pcts, xai_meta = xai_explainer.compute_exact_modality_shapley(
            regressor=model.regressor,
            audio_latent=audio_lat,
            visual_latent=visual_lat,
            text_latent=text_lat,
        )

        p_text = attr_pcts["text_pct"]
        p_vis = attr_pcts["visual_pct"]
        p_aud = attr_pcts["audio_pct"]
        if p_text >= max(p_vis, p_aud):
            dom_mod = "text"
        elif p_vis >= p_aud:
            dom_mod = "visual"
        else:
            dom_mod = "audio"

        patient_attribution = PatientModalityAttribution(
            text_pct=p_text,
            visual_pct=p_vis,
            audio_pct=p_aud,
            dominant_modality=dom_mod,
            shap_values=PatientShapValues(**shap_vals),
        )

        # 4. Generator Narasi Sintetis Klinis
        model_info = model_service.get_model_info()
        narrative = xai_explainer.generate_clinical_narrative(
            participant_id=participant_id,
            prediction=prediction_result,
            attribution=patient_attribution,
            quality_flags=quality_flags,
            model_name=model_info.get("model_name", "FED-MIND Multimodal"),
        )

        # 5. Metadata Model Aktif
        model_meta = ModelMetadata(
            model_id=model_info.get("model_type", settings.model_type),
            name=model_info.get("model_name", "FED-MIND Multimodal"),
            version=settings.app_version,
            architecture=model_info.get("architecture", "Centralized Multimodal Baseline"),
            checkpoint=model_info.get("checkpoint_path", "model.pt"),
            checksum=model_info.get("checkpoint_checksum", "unknown"),
            xai_baseline=model_info.get("xai_baseline_path", "xai_baseline.pt"),
            device=model_info.get("device", "cuda"),
        )

        completed_time = datetime.now(timezone.utc)

        # 6. Persistensi ke Database (Privacy-First: Fitur tensor mentah TIDAK disimpan)
        if db and screening_record:
            # Hapus result lama jika ada (idempotent)
            db.query(ScreeningResultModel).filter(ScreeningResultModel.screening_id == screening_id).delete()
            db.query(XAIResultModel).filter(XAIResultModel.screening_id == screening_id).delete()
            db.flush()

            res_db = ScreeningResultModel(
                screening_id=screening_id,
                phq8_score=prediction_result.phq8_score,
                risk_category=prediction_result.risk_category,
                binary_prediction=prediction_result.binary_prediction,
                risk_probability_proxy=prediction_result.risk_probability_proxy,
                created_at=completed_time,
            )
            db.add(res_db)

            xai_json_payload = {
                "text_pct": p_text,
                "visual_pct": p_vis,
                "audio_pct": p_aud,
                "quality_flags": q_flags_dict,
                "clinical_narrative": narrative,
                "model": model_meta.model_dump(),
                "xai_meta": xai_meta.model_dump(),
            }

            xai_db = XAIResultModel(
                screening_id=screening_id,
                text_shap=shap_vals["text"],
                visual_shap=shap_vals["visual"],
                audio_shap=shap_vals["audio"],
                dominant_modality=dom_mod,
                baseline_value=xai_meta.baseline_expected_value,
                method=xai_meta.method,
                explanation_json=xai_json_payload,
                created_at=completed_time,
            )
            db.add(xai_db)

            # Update status screening & audit log
            screening_record.status = "completed"
            screening_record.completed_at = completed_time

            audit_log = AuditLogModel(
                screening_id=screening_id,
                event="INFERENCE_COMPLETED",
                model_version=f"{model_meta.model_id}:{model_meta.checksum[:8]}",
                details=f"PHQ-8={prediction_result.phq8_score:.2f}, Risk={prediction_result.risk_category}, Dominant={dom_mod}",
                timestamp=completed_time,
            )
            db.add(audit_log)
            db.commit()

        # 7. Pembersihan tensor fitur mentah dari memori kerja (Garbage Collect Explicit)
        del tensors
        del features

        return ScreeningResponse(
            screening_id=screening_id,
            participant_id=participant_id,
            status="completed",
            prediction=prediction_result,
            patient_modality_attribution=patient_attribution,
            global_reference=GlobalReferenceAttribution(),
            feature_xai=None,
            quality_flags=quality_flags,
            clinical_narrative=narrative,
            model=model_meta,
            xai=xai_meta,
            disclaimer=DISCLAIMER_TEXT,
            created_at=screening_record.created_at.isoformat() + "Z" if screening_record else completed_time.isoformat() + "Z",
            completed_at=completed_time.isoformat() + "Z",
        )


# Global singleton screening service
screening_service = ScreeningService()
