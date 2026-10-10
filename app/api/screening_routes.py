"""
app/api/screening_routes.py
===========================
Router API untuk Operasi Skrining Depresi Multimodal, Explainable AI,
dan Lifecycle Screening Persistensi (Milestone 3).

Endpoints:
1. POST /screenings                     : Mendaftarkan sesi skrining baru (status: created)
2. POST /screenings/{id}/predict        : Menjalankan inferensi & XAI untuk sesi yang terdaftar
3. GET  /screenings/{id}                : Mengambil detail rekaman skrining dan XAI yang tersimpan
4. GET  /screenings                     : Riwayat sesi skrining (filter participant_id, status, pagination)
5. POST /screenings/predict             : All-in-one backward-compatible predict endpoint
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.screening import (
    ScreeningCreateRequest,
    ScreeningCreateResponse,
    ScreeningExecuteRequest,
    ScreeningListResponse,
    ScreeningRequest,
    ScreeningResponse,
)
from app.services.screening_service import screening_service

logger = logging.getLogger("SCREENING_ROUTES")

router = APIRouter(prefix="/screenings", tags=["Screenings"])


@router.post(
    "",
    response_model=ScreeningCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Langkah 1 Lifecycle: Daftarkan Sesi Skrining Baru",
    description="Mendaftarkan sesi skrining baru di database dengan status 'created' sebelum data fitur dikirim.",
)
def create_screening_session(
    request: ScreeningCreateRequest,
    db: Session = Depends(get_db),
) -> ScreeningCreateResponse:
    try:
        return screening_service.create_screening_session(request, db)
    except Exception as e:
        logger.error(f"Error creating screening session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal mendaftarkan sesi skrining: {str(e)}",
        )


@router.post(
    "/{screening_id}/predict",
    response_model=ScreeningResponse,
    status_code=status.HTTP_200_OK,
    summary="Langkah 2 Lifecycle: Eksekusi Inferensi & XAI untuk Sesi Skrining",
    description="Menerima tensor fitur untuk sesi yang sudah terdaftar, menjalankan model di CUDA/CPU, menghitung Shapley XAI, dan menyimpan hasil.",
)
def execute_screening_by_id(
    screening_id: str,
    features_req: ScreeningExecuteRequest,
    db: Session = Depends(get_db),
) -> ScreeningResponse:
    try:
        return screening_service.execute_screening_by_id(screening_id, features_req, db)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing screening {screening_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses inferensi skrining: {str(e)}",
        )


@router.get(
    "/{screening_id}",
    response_model=ScreeningResponse,
    status_code=status.HTTP_200_OK,
    summary="Langkah 3 Lifecycle: Ambil Rekaman Skrining & XAI Tersimpan",
    description="Mengambil detail hasil skrining, estimasi skor PHQ-8, stratifikasi risiko, dan penjelasan XAI dari database.",
)
def get_screening_by_id(
    screening_id: str,
    db: Session = Depends(get_db),
) -> ScreeningResponse:
    try:
        return screening_service.get_screening_by_id(screening_id, db)
    except KeyError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving screening {screening_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal mengambil rekaman skrining: {str(e)}",
        )


@router.get(
    "",
    response_model=ScreeningListResponse,
    status_code=status.HTTP_200_OK,
    summary="Langkah 4 Lifecycle: Riwayat Skrining (History)",
    description="Menampilkan histori sesi skrining yang tersimpan untuk halaman Riwayat pada klien mobile / web.",
)
def list_screenings(
    participant_id: Optional[str] = Query(None, description="Filter berdasarkan ID subjek (misal: 'P600')"),
    screening_status: Optional[str] = Query(None, alias="status", description="Filter berdasarkan status ('created', 'completed')"),
    limit: int = Query(50, ge=1, le=200, description="Jumlah maksimal rekaman per halaman"),
    offset: int = Query(0, ge=0, description="Offset rekaman"),
    db: Session = Depends(get_db),
) -> ScreeningListResponse:
    try:
        return screening_service.list_screenings(
            participant_id=participant_id,
            status=screening_status,
            limit=limit,
            offset=offset,
            db=db,
        )
    except Exception as e:
        logger.error(f"Error listing screenings: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal mengambil riwayat skrining: {str(e)}",
        )


@router.post(
    "/predict",
    response_model=ScreeningResponse,
    status_code=status.HTTP_200_OK,
    summary="All-in-One Skrining Depresi (Backward-Compatible)",
    description="Endpoint all-in-one yang mendaftarkan sesi otomatis, mengeksekusi inferensi, dan menyimpan hasil.",
)
def predict_screening(
    request: ScreeningRequest,
    db: Session = Depends(get_db),
) -> ScreeningResponse:
    try:
        return screening_service.process_screening(request, db=db)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error processing screening: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memproses skrining klinis: {str(e)}",
        )
