"""
tests/test_screening.py
========================
Pengujian Otomatis Komprehensif untuk Pipeline Skrining Multimodal FED-MIND (Milestone 2.5 & 3):
1. Unit Tests: Normalisasi, Adaptasi Sequence, Stratifikasi Risiko, dan Score-to-Probability Proxy
2. Golden Regression Test: Kasus Acuan P600 (Skor 9.69, Borderline, Visual 56.8%)
3. Lifecycle Persistence Tests: POST /screenings, POST /{id}/predict, GET /{id}, GET /screenings
4. API Error Tests: HTTP 404 (Sesi tidak ada), HTTP 422 (Payload malformed)
"""

import json
from pathlib import Path
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.core.inference import classify_risk, calculate_risk_probability_proxy
from app.core.preprocessing import adapt_to_sequence_shape, preprocessor

client = TestClient(app)


# ------------------------------------------------------------------------------
# 1. UNIT TESTS: PREPROCESSING & LOGIKA KLINIS
# ------------------------------------------------------------------------------
def test_risk_stratification_protocol():
    """Memastikan ambang batas stratifikasi risiko sesuai protokol riset."""
    assert classify_risk(0.0) == "low"
    assert classify_risk(7.99) == "low"
    assert classify_risk(8.0) == "borderline"
    assert classify_risk(9.69) == "borderline"
    assert classify_risk(9.99) == "borderline"
    assert classify_risk(10.0) == "moderate"
    assert classify_risk(14.99) == "moderate"
    assert classify_risk(15.0) == "high"
    assert classify_risk(24.0) == "high"


def test_risk_probability_proxy():
    """Memastikan score-to-probability sigmoid proxy berpusat di batas 10.0."""
    p_10 = calculate_risk_probability_proxy(10.0)
    assert 0.49 <= p_10 <= 0.51

    p_low = calculate_risk_probability_proxy(5.0)
    assert p_low < 0.25

    p_high = calculate_risk_probability_proxy(18.0)
    assert p_high > 0.90


def test_adapt_to_sequence_shape():
    """Memastikan konversi fitur 1D ke tensor temporal sesuai bentuk model."""
    audio_1d = np.ones(248, dtype=np.float32)
    visual_1d = np.ones(75, dtype=np.float32)
    text_1d = np.ones(768, dtype=np.float32)

    audio_seq = adapt_to_sequence_shape(audio_1d, "audio", target_length=512)
    visual_seq = adapt_to_sequence_shape(visual_1d, "visual", target_length=512)
    text_seq = adapt_to_sequence_shape(text_1d, "text", target_length=512)

    assert audio_seq.shape == (512, 80)
    assert visual_seq.shape == (512, 72, 3)
    assert text_seq.shape == (512, 768)


def test_quality_flags_detection():
    """Memastikan deteksi sinyal abnormal atau minim informasi bekerja."""
    zero_signal = np.zeros(100, dtype=np.float32)
    normal_signal = np.random.randn(100).astype(np.float32)

    flags = preprocessor.check_quality_flags(zero_signal, normal_signal, normal_signal)
    assert flags["poor_audio_quality"] is True
    assert flags["low_text_information"] is False
    assert flags["poor_visual_quality"] is False


# ------------------------------------------------------------------------------
# 2. GOLDEN REGRESSION TEST: PARTICIPANT 600
# ------------------------------------------------------------------------------
def test_golden_regression_p600():
    """
    Golden Regression Test menggunakan fitur asli Participant P600.
    Memverifikasi determinisme output:
    - PHQ-8 Score: 9.69 (+/- 0.05)
    - Kategori Risiko: borderline
    - Prediksi Biner: non_depression
    - Modalitas Dominan: visual (~56.8%)
    - Text Attribution: ~39.1%
    - Audio Attribution: ~4.1%
    - SHAP Values: Visual > 0 (menaikkan), Text < 0 (menurunkan)
    - Feature XAI: None (Audit Milestone 2.5: dipisah dari Modality XAI)
    """
    fixture_path = Path("tests/fixtures/p600_features.json")
    assert fixture_path.exists(), "File fixture p600_features.json wajib ada."

    with open(fixture_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    response = client.post("/screenings/predict", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"

    data = response.json()

    # Validasi Identitas & Status
    assert data["participant_id"] == "P600"
    assert data["status"] == "completed"

    # Validasi Prediksi Skor & Proxy Probabilitas
    pred = data["prediction"]
    assert 9.60 <= pred["phq8_score"] <= 9.75, f"Expected ~9.69, got {pred['phq8_score']}"
    assert pred["risk_category"] == "borderline"
    assert pred["binary_prediction"] == "non_depression"
    assert 0.40 <= pred["risk_probability_proxy"] <= 0.55

    # Validasi Atribusi Modalitas Pasien (Exact Shapley)
    attr = data["patient_modality_attribution"]
    assert attr["dominant_modality"] == "visual"
    assert 54.0 <= attr["visual_pct"] <= 59.0, f"Expected Visual ~56.8%, got {attr['visual_pct']}"
    assert 37.0 <= attr["text_pct"] <= 42.0, f"Expected Text ~39.1%, got {attr['text_pct']}"
    assert 3.0 <= attr["audio_pct"] <= 6.0, f"Expected Audio ~4.1%, got {attr['audio_pct']}"

    # Validasi Arah SHAP Values
    shap_vals = attr["shap_values"]
    assert shap_vals["visual"] > 0, "Visual seharusnya menaikkan risiko pada P600"
    assert shap_vals["text"] < 0, "Teks afektif seharusnya menurunkan risiko pada P600"

    # Audit Milestone 2.5: feature_xai diset null
    assert data["feature_xai"] is None

    # Validasi Metadata Model & Checksum
    model = data["model"]
    assert model["model_id"] == "fl_iid"
    assert len(model["checksum"]) >= 16

    # Validasi XAI Metadata & Efficiency Axiom
    xai = data["xai"]
    assert xai["method"] == "exact_modality_shapley_coalition"
    assert xai["efficiency_axiom_verified"] is True
    assert 8.30 <= xai["baseline_expected_value"] <= 8.55


# ------------------------------------------------------------------------------
# 3. LIFECYCLE & PERSISTENCE TESTS (MILESTONE 3)
# ------------------------------------------------------------------------------
def test_screening_lifecycle_flow():
    """
    Menguji alur siklus hidup lengkap skrining & persistensi database:
    1. POST /screenings -> registrasi sesi (status: created)
    2. POST /screenings/{id}/predict -> eksekusi & persistensi (status: completed)
    3. GET /screenings/{id} -> verifikasi data tersimpan cocok
    4. GET /screenings -> verifikasi listing riwayat
    """
    # 1. Daftarkan sesi baru
    create_payload = {
        "participant_id": "P600_LIFECYCLE",
        "metadata": {"clinic": "Unit Riset FED-MIND", "examiner": "dr. Sp.KJ"}
    }
    create_resp = client.post("/screenings", json=create_payload)
    assert create_resp.status_code == 201
    created_data = create_resp.json()
    screening_id = created_data["screening_id"]
    assert created_data["status"] == "created"
    assert created_data["participant_id"] == "P600_LIFECYCLE"

    # 2. Eksekusi inferensi pada sesi terdaftar
    fixture_path = Path("tests/fixtures/p600_features.json")
    with open(fixture_path, "r", encoding="utf-8") as f:
        p600_data = json.load(f)

    exec_payload = {
        "audio_features": p600_data["audio_features"],
        "visual_features": p600_data["visual_features"],
        "text_features": p600_data["text_features"],
        "metadata": {"device": "mobile_app"}
    }
    exec_resp = client.post(f"/screenings/{screening_id}/predict", json=exec_payload)
    assert exec_resp.status_code == 200
    exec_data = exec_resp.json()
    assert exec_data["screening_id"] == screening_id
    assert exec_data["status"] == "completed"
    assert exec_data["prediction"]["risk_category"] == "borderline"

    # 3. Ambil data tersimpan dari database via GET
    get_resp = client.get(f"/screenings/{screening_id}")
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    assert get_data["screening_id"] == screening_id
    assert get_data["status"] == "completed"
    assert get_data["prediction"]["phq8_score"] == exec_data["prediction"]["phq8_score"]
    assert get_data["patient_modality_attribution"]["dominant_modality"] == "visual"

    # 4. Ambil daftar riwayat via GET /screenings
    list_resp = client.get("/screenings?participant_id=P600_LIFECYCLE")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    pids = [s["participant_id"] for s in list_data["screenings"]]
    assert "P600_LIFECYCLE" in pids


def test_screening_lifecycle_not_found():
    """Memastikan HTTP 404 dikembalikan saat ID skrining tidak ditemukan."""
    fake_id = "SCR-DOES-NOT-EXIST"
    resp = client.get(f"/screenings/{fake_id}")
    assert resp.status_code == 404

    resp_exec = client.post(
        f"/screenings/{fake_id}/predict",
        json={"audio_features": [0.0], "visual_features": [0.0], "text_features": [0.0]}
    )
    assert resp_exec.status_code == 404


def test_predict_malformed_input():
    """Memastikan API mengembalikan HTTP 422 saat field wajib tidak disertakan."""
    incomplete_payload = {
        "participant_id": "P_INVALID",
        "audio_features": [0.1, 0.2],
    }
    response = client.post("/screenings/predict", json=incomplete_payload)
    assert response.status_code == 422
