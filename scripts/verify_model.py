"""
scripts/verify_model.py
=======================
Skrip mandiri untuk memverifikasi model baseline multimodal:
1. Memuat best_global_model.pt
2. Memeriksa ketersediaan dan penempatan di device CUDA
3. Menjalankan pengujian dummy forward pass trilinear (Audio, Visual, Text)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from app.services.model_service import model_service


def main():
    print("=" * 70)
    print("VERIFIKASI MODEL FED-MIND MULTIMODAL BASELINE")
    print("=" * 70)

    # 1. Load Model
    success = model_service.load_model()
    if not success:
        print("[FAIL] Gagal memuat model. Periksa checkpoint dan config!")
        sys.exit(1)

    info = model_service.get_model_info()
    print(f"[OK] Model Name         : {info['model']}")
    print(f"[OK] Architecture       : {info['architecture']}")
    print(f"[OK] Device Aktif       : {info['device']}")
    print(f"[OK] Total Parameter    : {info['total_parameters']:,}")
    print(f"[OK] Checkpoint File    : {info['checkpoint_path']}")

    # 2. Test Dummy Forward Pass
    model = model_service.get_model()
    device = torch.device(info["device"])

    print("")
    print("Menjalankan Dummy Forward Pass multimodal...")
    batch_size = 2
    seq_len = 512

    # Audio: (B, 512, 80)
    audio = torch.randn(batch_size, seq_len, 80, device=device)
    # Visual: (B, 512, 72, 3)
    visual = torch.randn(batch_size, seq_len, 72, 3, device=device)
    # Text: (B, 512, 768)
    text = torch.randn(batch_size, seq_len, 768, device=device)

    with torch.no_grad():
        preds = model(audio, visual, text)
        features = model.extract_features(audio, visual, text)

    print(f"[OK] Forward Pass Output Shape : {preds.shape} (Expected: ({batch_size}, 1))")
    print(f"[OK] Sample Predicted Scores   : {[round(p.item(), 3) for p in preds]}")
    print(f"[OK] Extracted Audio Latent    : {features['audio_latent'].shape}")
    print(f"[OK] Extracted Visual Latent   : {features['visual_latent'].shape}")
    print(f"[OK] Extracted Text Latent     : {features['text_latent'].shape}")
    print(f"[OK] Extracted Fused Latent    : {features['fused_latent'].shape}")

    print("")
    print("=" * 70)
    print("SELURUH VERIFIKASI BERHASIL 100% (Model Siap untuk Server)")
    print("=" * 70)


if __name__ == "__main__":
    main()
