"""
app/core/inference.py
=====================
Pipeline Inferensi Model Baseline Multimodal & Stratifikasi Risiko Klinis.
Mencakup forward pass PyTorch (inference_mode), kalibrasi skor PHQ-8,
dan pengkategorian risiko (Low, Borderline, Moderate, High).
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn

from app.schemas.screening import PredictionResult


def classify_risk(score: float) -> str:
    """
    Stratifikasi risiko klinis berdasarkan batas skor protokol riset FED-MIND:
    - score < 8.0   : 'low' (Non-Risk)
    - 8.0 <= score < 10.0 : 'borderline' (Zona Borderline / Pengawasan Klinis)
    - 10.0 <= score < 15.0: 'moderate' (Risiko Sedang / Terindikasi Depresi)
    - score >= 15.0 : 'high' (Risiko Tinggi / Depresi Berat)
    """
    if score < 8.0:
        return "low"
    elif score < 10.0:
        return "borderline"
    elif score < 15.0:
        return "moderate"
    return "high"


def calculate_risk_probability_proxy(score: float, threshold: float = 10.0, scale: float = 0.35) -> float:
    """
    Kalkulasi score-to-probability sigmoid proxy yang terpusat pada ambang batas klinis 10.0.
    (Catatan audit Milestone 2.5: ini adalah transformasi proxy sigmoid, bukan formal calibrated
    probability via Platt scaling/ECE).
    """
    z = scale * (score - threshold)
    prob = 1.0 / (1.0 + np.exp(-z))
    return float(np.clip(prob, 0.001, 0.999))


class MultimodalInferencePipeline:
    """
    Pipeline eksekusi inferensi model baseline multimodal FED-MIND.
    """

    def __init__(self, clip_bounds: Tuple[float, float] = (0.0, 24.0)) -> None:
        self.clip_min, self.clip_max = clip_bounds

    def predict(
        self,
        model: nn.Module,
        audio_tensor: torch.Tensor,
        visual_tensor: torch.Tensor,
        text_tensor: torch.Tensor,
    ) -> Tuple[PredictionResult, Dict[str, torch.Tensor]]:
        """
        Menjalankan forward pass inferensi dan ekstraksi representasi laten unimodal:
        - Output: PredictionResult Pydantic dan Dict latent tensors (audio_latent, visual_latent, text_latent).
        """
        with torch.inference_mode():
            # 1. Ekstraksi representasi laten tiap modalitas (masing-masing 256 dimensi)
            features = model.extract_features(audio_tensor, visual_tensor, text_tensor)

            # 2. Forward pass ke regressor
            raw_pred = model(audio_tensor, visual_tensor, text_tensor).item()

            # 3. Batasi skor ke rentang PHQ-8 valid (0.0 – 24.0)
            bounded_score = float(np.clip(raw_pred, self.clip_min, self.clip_max))
            phq8_score = round(bounded_score, 2)

            # 4. Stratifikasi risiko & penentuan indikator biner
            risk_category = classify_risk(phq8_score)
            binary_prediction = "depression_risk" if phq8_score >= 10.0 else "non_depression"
            probability_proxy = round(calculate_risk_probability_proxy(phq8_score), 4)

            prediction_result = PredictionResult(
                phq8_score=phq8_score,
                risk_probability_proxy=probability_proxy,
                binary_prediction=binary_prediction,
                risk_category=risk_category,
            )

            return prediction_result, features


# Global singleton inference pipeline
inference_pipeline = MultimodalInferencePipeline()
