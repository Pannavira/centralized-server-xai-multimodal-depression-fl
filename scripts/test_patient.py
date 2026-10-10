"""
scripts/test_patient.py
=======================
Utilitas CLI interaktif untuk menguji inferensi dan XAI pada berbagai ID subjek:
- Menguji kasus Acuan (P600 - Borderline, P605 - Low, P604 - Moderate, P716 - High, P602 - False Negative)
- Menguji sembarang ID subjek dari dataset riset atau file fixture
- Menampilkan visualisasi persentase modalitas dan narasi klinis

Contoh Penggunaan:
    python scripts/test_patient.py --pid 716
    python scripts/test_patient.py --pid 600
    python scripts/test_patient.py --list
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SERVER_URL = "http://127.0.0.1:8000/screenings/predict"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
RESEARCH_FEATURE_DIR = Path("C:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/modeling/01_Input_From_Partition")

CASE_PROFILES = {
    600: {"true_phq8": 5, "profile": "Borderline Zone (Visual Dominan 56.8%)"},
    602: {"true_phq8": 13, "profile": "False Negative Case (Depresi Gagal Terdeteksi)"},
    604: {"true_phq8": 12, "profile": "Moderate Risk True Positive"},
    605: {"true_phq8": 2, "profile": "Low Risk / Non-Depression (Teks Protektif)"},
    606: {"true_phq8": 5, "profile": "False Positive / Alarm Palsu (Teks Tinggi)"},
    618: {"true_phq8": 4, "profile": "False Positive / Visual Dominan"},
    624: {"true_phq8": 22, "profile": "Moderate-High Risk True Positive"},
    716: {"true_phq8": 15, "profile": "High Risk True Positive (Teks Dominan 71.3%)"},
}


def load_payload_for_pid(pid: int) -> Optional[Dict[str, Any]]:
    # 1. Cek dari fixtures lokal
    fix_path = FIXTURES_DIR / f"p{pid}_features.json"
    if fix_path.exists():
        with open(fix_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2. Cek dari folder dataset riset
    if RESEARCH_FEATURE_DIR.exists():
        import numpy as np

        p_audio = RESEARCH_FEATURE_DIR / f"final_audio_features/{pid}.npy"
        p_visual = RESEARCH_FEATURE_DIR / f"final_visual_features/{pid}.npy"
        p_text = RESEARCH_FEATURE_DIR / f"final_text_features/{pid}.npy"

        if p_audio.exists() and p_visual.exists() and p_text.exists():
            return {
                "participant_id": f"P{pid}",
                "screening_id": f"SCR-CLI-P{pid}",
                "audio_features": np.load(p_audio).astype(np.float32).tolist(),
                "visual_features": np.load(p_visual).astype(np.float32).tolist(),
                "text_features": np.load(p_text).astype(np.float32).tolist(),
                "metadata": {"source": "research_dataset", "pid": pid},
            }

    return None


def print_banner(title: str):
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Test FED-MIND Multimodal Screening by Participant ID")
    parser.add_argument("--pid", type=int, default=716, help="Participant ID numerik (misal: 600, 716, 605, 604, 602)")
    parser.add_argument("--list", action="store_true", help="Tampilkan daftar ID subjek yang direkomendasikan untuk uji")
    parser.add_argument("--server", type=str, default=SERVER_URL, help="URL server screening endpoint")
    args = parser.parse_args()

    if args.list:
        print_banner("DAFTAR REKOMENDASI KASUS SUBJEK UNTUK PENGUJIAN KLINIS")
        print(f"{'PID':<8} | {'True PHQ-8':<12} | {'Profil Kasus':<45}")
        print("-" * 75)
        for pid, meta in CASE_PROFILES.items():
            print(f"P{pid:<7} | {meta['true_phq8']:<12} | {meta['profile']:<45}")
        print("-" * 75)
        print("\nContoh perintah: python scripts/test_patient.py --pid 600")
        print("                 python scripts/test_patient.py --pid 716\n")
        return

    pid = args.pid
    payload = load_payload_for_pid(pid)
    if not payload:
        print(f"[ERROR] Fitur untuk subjek P{pid} tidak ditemukan di fixtures maupun dataset riset.")
        sys.exit(1)

    print(f"\n[INFO] Mengirim request skrining untuk Subjek P{pid} ke {args.server}...")
    try:
        resp = requests.post(args.server, json=payload, timeout=10)
    except Exception as e:
        print(f"[FAIL] Server tidak merespon di {args.server}. Pastikan server menyala (run_server.bat).")
        print(f"       Detail error: {e}")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"[FAIL] HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()

    pred = data["prediction"]
    attr = data["patient_modality_attribution"]
    shap_v = attr["shap_values"]

    print_banner(f"HASIL SKRINING KLINIS: SUBJEK P{pid}")
    print(f"  Estimasi Skor PHQ-8       : {pred['phq8_score']:.2f} / 24.0")
    print(f"  Kategori Risiko Klinis    : {pred['risk_category'].upper()}")
    print(f"  Keputusan Triase Biner    : {pred['binary_prediction'].upper()}")
    prob_val = pred.get("risk_probability_proxy", pred.get("probability", 0.0))
    print(f"  Risk Probability Proxy    : {prob_val * 100:.1f}%\n")

    print(f"  Modalitas Dominan Pasien  : {attr['dominant_modality'].upper()}")
    print("  Distribusi Atribusi Modalitas (Exact Shapley):")
    print(f"    - Teks   : {attr['text_pct']:>5.1f}% | SHAP: {shap_v['text']:+.3f}")
    print(f"    - Visual : {attr['visual_pct']:>5.1f}% | SHAP: {shap_v['visual']:+.3f}")
    print(f"    - Audio  : {attr['audio_pct']:>5.1f}% | SHAP: {shap_v['audio']:+.3f}\n")

    print("  Rujukan Kohort Global (E-DAIC N=56):")
    ref = data["global_reference"]
    print(f"    - Teks: {ref['text_pct']:.1f}% | Visual: {ref['visual_pct']:.1f}% | Audio: {ref['audio_pct']:.1f}%\n")

    print("\n" + "=" * 78)
    print(" LAPORAN NARASI SINTETIS DARI SERVER:")
    print("=" * 78)
    print(data["clinical_narrative"])
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
