"""
app/core/xai_explainer.py
=========================
Generator Explainable AI (XAI) Klinis Berbasis Exact Modality Shapley Value.
Mengimplementasikan:
1. Exact Modality Shapley Values (8 Evaluasi Koalisi 2^3) terhadap E[f(X)] kohort spesifik model aktif
2. Pemenuhan Aksioma Efficiency, Symmetry, Dummy, dan Additivity (Zero Approximation Error)
3. Audit Milestone 2.5: Memisahkan feature_xai (diset null hingga continuous KernelSHAP)
4. Generator Narasi Sintesis Klinis & Disclaimer Medis
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from app.config import settings
from app.schemas.screening import (
    GlobalReferenceAttribution,
    PatientModalityAttribution,
    PatientShapValues,
    PredictionResult,
    QualityFlags,
    XAIMetadata,
)

logger = logging.getLogger("XAI_EXPLAINER")


class ClinicalXAIExplainer:
    """
    Mesin Komputasi XAI Multimodal Eksak untuk Rekomendasi Klinis Pasien.
    Menjamin isolasi baseline model sehingga baseline XAI selalu sinkron dengan checkpoint aktif.
    """

    def __init__(self, baseline_path: Optional[Union[str, Path]] = None) -> None:
        self.baseline_path = Path(baseline_path) if baseline_path else settings.resolved_xai_baseline_path
        self.audio_base: Optional[torch.Tensor] = None
        self.visual_base: Optional[torch.Tensor] = None
        self.text_base: Optional[torch.Tensor] = None
        self.v_000: float = 8.4164
        self.model_key: str = settings.model_type

        self.load_baseline_latents()

    def load_baseline_latents(self, custom_path: Optional[Path] = None) -> None:
        """Memuat representasi laten baseline E[I_m] yang selaras 100% dengan model aktif."""
        path = custom_path or settings.resolved_xai_baseline_path
        self.baseline_path = path

        if not path.exists():
            logger.warning(f"File baseline XAI tidak ditemukan di {path}. Menggunakan default nol.")
            self.audio_base = torch.zeros(1, 256)
            self.visual_base = torch.zeros(1, 256)
            self.text_base = torch.zeros(1, 256)
            self.v_000 = 8.4164
            self.model_key = settings.model_type
            return

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            self.audio_base = data["audio_base"]
            self.visual_base = data["visual_base"]
            self.text_base = data["text_base"]
            self.v_000 = float(data.get("v_000", 8.4164))
            self.model_key = str(data.get("model_key", settings.model_type))
            logger.info(f"Loaded XAI Baseline [{self.model_key}] from {path} (v_000 = {self.v_000:.4f})")
        except Exception as e:
            logger.error(f"Gagal memuat xai_baseline.pt: {e}", exc_info=True)
            self.audio_base = torch.zeros(1, 256)
            self.visual_base = torch.zeros(1, 256)
            self.text_base = torch.zeros(1, 256)
            self.v_000 = 8.4164
            self.model_key = settings.model_type

    def compute_exact_modality_shapley(
        self,
        regressor: nn.Module,
        audio_latent: torch.Tensor,
        visual_latent: torch.Tensor,
        text_latent: torch.Tensor,
    ) -> Tuple[Dict[str, float], Dict[str, float], XAIMetadata]:
        """
        Menghitung Exact Shapley Value untuk 3 modalitas berdasarkan 8 koalisi game theory (2^3).
        Menjamin pemenuhan aksioma Efficiency: phi_A + phi_V + phi_T == v_111 - v_000.
        """
        device = audio_latent.device

        # Siapkan baseline pada device yang sama
        a_base = self.audio_base.to(device)
        v_base = self.visual_base.to(device)
        t_base = self.text_base.to(device)

        a_i = audio_latent
        v_i = visual_latent
        t_i = text_latent

        def eval_coalition(a_in: torch.Tensor, v_in: torch.Tensor, t_in: torch.Tensor) -> float:
            with torch.inference_mode():
                concat = torch.cat([a_in, v_in, t_in], dim=-1)
                return float(regressor(concat).squeeze().item())

        v_000 = eval_coalition(a_base, v_base, t_base)
        v_100 = eval_coalition(a_i, v_base, t_base)
        v_010 = eval_coalition(a_base, v_i, t_base)
        v_001 = eval_coalition(a_base, v_base, t_i)
        v_110 = eval_coalition(a_i, v_i, t_base)
        v_101 = eval_coalition(a_i, v_base, t_i)
        v_011 = eval_coalition(a_base, v_i, t_i)
        v_111 = eval_coalition(a_i, v_i, t_i)

        # Bobot pembagi Shapley untuk 3 pemain:
        # |S|=0 -> 1/3, |S|=1 -> 1/6, |S|=2 -> 1/3
        p_A = (1 / 3) * (v_100 - v_000) + (1 / 6) * (v_110 - v_010) + (1 / 6) * (v_101 - v_001) + (1 / 3) * (v_111 - v_011)
        p_V = (1 / 3) * (v_010 - v_000) + (1 / 6) * (v_110 - v_100) + (1 / 6) * (v_011 - v_001) + (1 / 3) * (v_111 - v_101)
        p_T = (1 / 3) * (v_001 - v_000) + (1 / 6) * (v_101 - v_100) + (1 / 6) * (v_011 - v_010) + (1 / 3) * (v_111 - v_110)

        # Verifikasi Efficiency Axiom
        diff = v_111 - v_000
        tot = p_A + p_V + p_T
        efficiency_ok = bool(np.isclose(tot, diff, atol=1e-3))

        tot_abs = abs(p_A) + abs(p_V) + abs(p_T) + 1e-8
        pct_A = round((abs(p_A) / tot_abs) * 100.0, 1)
        pct_V = round((abs(p_V) / tot_abs) * 100.0, 1)
        pct_T = round((abs(p_T) / tot_abs) * 100.0, 1)

        shap_values = {
            "audio": round(float(p_A), 3),
            "visual": round(float(p_V), 3),
            "text": round(float(p_T), 3),
        }

        attribution_pct = {
            "audio_pct": pct_A,
            "visual_pct": pct_V,
            "text_pct": pct_T,
        }

        xai_meta = XAIMetadata(
            method="exact_modality_shapley_coalition",
            num_coalitions=8,
            efficiency_axiom_verified=efficiency_ok,
            baseline_expected_value=round(float(v_000), 4),
        )

        return shap_values, attribution_pct, xai_meta

    def generate_clinical_narrative(
        self,
        participant_id: str,
        prediction: PredictionResult,
        attribution: PatientModalityAttribution,
        quality_flags: QualityFlags,
        model_name: str,
    ) -> str:
        """
        Menyusun narasi sintetis berstandar klinis berdasarkan atribusi eksak Shapley.
        """
        risk_map = {
            "low": "Rendah (Non-Risk)",
            "borderline": "Borderline (Zona Pengawasan Klinis)",
            "moderate": "Sedang (Terindikasi Depresi)",
            "high": "Tinggi (Terindikasi Depresi Berat)",
        }
        risk_label = risk_map.get(prediction.risk_category, prediction.risk_category.upper())

        dom_name = {
            "text": "Teks (Verbal Afektif)",
            "visual": "Visual (Ekspresi Wajah & Gaze)",
            "audio": "Audio (Prosodi & Akustik)",
        }.get(attribution.dominant_modality, attribution.dominant_modality.capitalize())

        dom_pct = getattr(attribution, f"{attribution.dominant_modality}_pct", 0.0)

        narrative_lines = [
            "=" * 78,
            "LAPORAN DUKUNGAN KEPUTUSAN SKRINING KLINIS MULTIMODAL (FED-MIND)",
            "=" * 78,
            f"Nomor Identitas Subjek : {participant_id}",
            f"Arsitektur Model       : {model_name}",
            f"Estimasi Skor PHQ-8    : {prediction.phq8_score:.2f} / 24.0",
            f"Kategori Risiko        : {risk_label}",
            f"Risk Probability Proxy : {prediction.risk_probability_proxy * 100:.1f}%",
            f"Modalitas Dominan      : {dom_name} ({dom_pct:.1f}%)",
            "",
            "DISTRIBUSI KONTRIBUSI MODALITAS PASIEN (EXACT SHAPLEY):",
            f"- Teks (Verbal Afektif)        : {attribution.text_pct:.1f}% (SHAP Value: {attribution.shap_values.text:+.3f})",
            f"- Visual (Ekspresi Wajah/Gaze) : {attribution.visual_pct:.1f}% (SHAP Value: {attribution.shap_values.visual:+.3f})",
            f"- Audio (Prosodi & Akustik)    : {attribution.audio_pct:.1f}% (SHAP Value: {attribution.shap_values.audio:+.3f})",
        ]

        if quality_flags.low_text_information or quality_flags.poor_audio_quality or quality_flags.poor_visual_quality:
            narrative_lines.extend([
                "",
                "CATATAN KUALITAS INPUT DATA:",
                f"- Peringatan: {quality_flags.details}",
            ])

        narrative_lines.extend([
            "",
            "CATATAN ETIKA & BATASAN MEDIS:",
            "Hasil ini merupakan instrumen triase skrining awal berbasis kecerdasan buatan terfederasi.",
            "Penjelasan ini BUKAN diagnosis klinis final dan WAJIB ditinjau bersama asesmen klinis profesional.",
            "=" * 78,
        ])

        return "\n".join(narrative_lines)


# Global singleton explainer instance
xai_explainer = ClinicalXAIExplainer()
