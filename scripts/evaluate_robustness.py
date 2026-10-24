"""
scripts/evaluate_robustness.py
==============================
Skrip Evaluasi Robustness Suplementer (Milestone 5.4 Section 16).
Menguji ketahanan model final M5.1 (IID Control) vs M5.4 (Modality Skew)
terhadap 4 kondisi pengujian tetap (fixed test conditions):
A. Clean test data (56 subjek asli)
B. Fixed audio-noisy test data (Gaussian perturbation sigma=0.25)
C. Fixed visually degraded test data (20% temporal frame dropout)
D. Fixed combined degradation (Audio noise + Visual frame dropout)
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from app.core.model import MultimodalBaselineModel
from federated.data import build_global_test_dataloader
from federated.metrics import calculate_clinical_metrics
from federated.modality_degradation import (
    ModalityDegradationConfig,
    apply_audio_noise,
    apply_visual_frame_dropout,
    get_deterministic_sample_seed,
    get_visual_neutral_frame,
)
from federated.task import build_model, evaluate_model

M51_R50_PATH = Path("storage/federated/runs/FL-RUN-20261016-130403-45D587/rounds/round_050/model.pt")


class RobustnessTestDataset(torch.utils.data.Dataset):
    """Dataset wrapper yang menerapkan perlakuan degradasi tetap pada test set global."""

    def __init__(
        self,
        base_dataset: Any,
        condition: str = "clean",
        audio_sigma: float = 0.25,
        visual_dropout_rate: float = 0.20,
        seed: int = 42,
    ) -> None:
        self.base_dataset = base_dataset
        self.condition = condition.lower()
        self.audio_sigma = audio_sigma
        self.visual_dropout_rate = visual_dropout_rate
        self.seed = seed
        self.neutral_frame = get_visual_neutral_frame()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raw_item = self.base_dataset._get_raw_item(idx) if hasattr(self.base_dataset, "_get_raw_item") else self.base_dataset[idx]
        item = dict(raw_item)
        pid = str(item.get("participant_id", "unknown"))

        # Condition A: Clean (tidak ada perubahan)
        if self.condition == "clean":
            return item

        # Condition B: Audio Noisy
        elif self.condition == "audio_noisy":
            seed = get_deterministic_sample_seed(self.seed, "test", pid, "audio")
            a_noisy, _ = apply_audio_noise(item["audio"], sigma=self.audio_sigma, seed=seed)
            item["audio"] = a_noisy
            return item

        # Condition C: Visual Degraded
        elif self.condition == "visual_degraded":
            seed = get_deterministic_sample_seed(self.seed, "test", pid, "visual")
            v_deg, _ = apply_visual_frame_dropout(
                item["visual"],
                dropout_rate=self.visual_dropout_rate,
                neutral_frame=self.neutral_frame,
                seed=seed,
            )
            item["visual"] = v_deg
            return item

        # Condition D: Combined (Audio Noisy + Visual Degraded)
        elif self.condition == "combined":
            seed_a = get_deterministic_sample_seed(self.seed, "test", pid, "audio")
            seed_v = get_deterministic_sample_seed(self.seed, "test", pid, "visual")
            a_noisy, _ = apply_audio_noise(item["audio"], sigma=self.audio_sigma, seed=seed_a)
            v_deg, _ = apply_visual_frame_dropout(
                item["visual"],
                dropout_rate=self.visual_dropout_rate,
                neutral_frame=self.neutral_frame,
                seed=seed_v,
            )
            item["audio"] = a_noisy
            item["visual"] = v_deg
            return item

        else:
            raise ValueError(f"Kondisi robustness tidak dikenal: {self.condition}")


def run_robustness_evaluation(
    candidate_run_dir: Path,
    m51_model_path: Optional[Path] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Menjalankan evaluasi komparatif robustness antara model final M5.1 dan candidate model (M5.4/M5.5)."""
    cand_dir = Path(candidate_run_dir)
    cand_r50_path = cand_dir / "rounds" / "round_050" / "model.pt"
    m51_path = Path(m51_model_path or M51_R50_PATH)

    if not cand_r50_path.exists():
        raise FileNotFoundError(f"Model Round 50 tidak ditemukan di: {cand_r50_path}")
    if not m51_path.exists():
        raise FileNotFoundError(f"Model M5.1 Round 50 tidak ditemukan di: {m51_path}")

    cand_label = "Combined Non-IID" if "COMBINED" in cand_dir.name else "Modality Skew"

    # 1. Muat kedua model
    m51_model = build_model()
    m51_sd = torch.load(m51_path, map_location="cpu", weights_only=False)
    m51_model.load_state_dict(m51_sd, strict=True)
    m51_model.to(device).eval()

    cand_model = build_model()
    cand_sd = torch.load(cand_r50_path, map_location="cpu", weights_only=False)
    cand_model.load_state_dict(cand_sd, strict=True)
    cand_model.to(device).eval()

    # 2. Base test loader
    _, base_test_ds = build_global_test_dataloader()

    conditions = [
        ("clean", "Condition A: Clean Test Data"),
        ("audio_noisy", "Condition B: Audio-Noisy Test Data (σ = 0.25)"),
        ("visual_degraded", "Condition C: Visually Degraded (20% Dropout)"),
        ("combined", "Condition D: Combined (Audio Noise + Visual Dropout)"),
    ]

    results: Dict[str, Any] = {
        "m51_run_id": "FL-RUN-20261016-130403-45D587",
        "candidate_run_id": cand_dir.name,
        "m54_run_id": cand_dir.name,
        "candidate_label": cand_label,
        "conditions": {},
    }

    csv_rows = []

    for cond_key, cond_name in conditions:
        rob_ds = RobustnessTestDataset(base_test_ds, condition=cond_key)
        loader = torch.utils.data.DataLoader(rob_ds, batch_size=16, shuffle=False)

        # Evaluasi M5.1
        loss_m51, m51_metrics = evaluate_model(m51_model, loader, device=device)
        m51_metrics["test_loss"] = round(loss_m51, 4)

        # Evaluasi candidate (M5.4 / M5.5)
        loss_cand, cand_metrics = evaluate_model(cand_model, loader, device=device)
        cand_metrics["test_loss"] = round(loss_cand, 4)

        delta_mae = cand_metrics["mae"] - m51_metrics["mae"]
        delta_f1 = cand_metrics["macro_f1"] - m51_metrics["macro_f1"]
        delta_recall = cand_metrics["recall"] - m51_metrics["recall"]

        results["conditions"][cond_key] = {
            "condition_name": cond_name,
            "m51_iid": {
                "mae": round(m51_metrics["mae"], 4),
                "rmse": round(m51_metrics["rmse"], 4),
                "macro_f1": round(m51_metrics["macro_f1"], 4),
                "recall": round(m51_metrics["recall"], 4),
                "specificity": round(m51_metrics["specificity"], 4),
                "auc_roc": round(m51_metrics.get("auc_roc", 0.0), 4),
                "tp": int(m51_metrics["tp"]),
                "fp": int(m51_metrics["fp"]),
                "tn": int(m51_metrics["tn"]),
                "fn": int(m51_metrics["fn"]),
            },
            "candidate": {
                "mae": round(cand_metrics["mae"], 4),
                "rmse": round(cand_metrics["rmse"], 4),
                "macro_f1": round(cand_metrics["macro_f1"], 4),
                "recall": round(cand_metrics["recall"], 4),
                "specificity": round(cand_metrics["specificity"], 4),
                "auc_roc": round(cand_metrics.get("auc_roc", 0.0), 4),
                "tp": int(cand_metrics["tp"]),
                "fp": int(cand_metrics["fp"]),
                "tn": int(cand_metrics["tn"]),
                "fn": int(cand_metrics["fn"]),
            },
            "m54_modality_skew": {
                "mae": round(cand_metrics["mae"], 4),
                "rmse": round(cand_metrics["rmse"], 4),
                "macro_f1": round(cand_metrics["macro_f1"], 4),
                "recall": round(cand_metrics["recall"], 4),
                "specificity": round(cand_metrics["specificity"], 4),
                "auc_roc": round(cand_metrics.get("auc_roc", 0.0), 4),
                "tp": int(cand_metrics["tp"]),
                "fp": int(cand_metrics["fp"]),
                "tn": int(cand_metrics["tn"]),
                "fn": int(cand_metrics["fn"]),
            },
            "deltas": {
                "delta_mae": round(delta_mae, 4),
                "delta_macro_f1": round(delta_f1, 4),
                "delta_recall": round(delta_recall, 4),
            },
        }

        csv_rows.append({
            "condition": cond_key,
            "condition_name": cond_name,
            "m51_mae": round(m51_metrics["mae"], 4),
            "cand_mae": round(cand_metrics["mae"], 4),
            "delta_mae": round(delta_mae, 4),
            "m51_rmse": round(m51_metrics["rmse"], 4),
            "cand_rmse": round(cand_metrics["rmse"], 4),
            "m51_macro_f1": round(m51_metrics["macro_f1"], 4),
            "cand_macro_f1": round(cand_metrics["macro_f1"], 4),
            "delta_macro_f1": round(delta_f1, 4),
            "m51_recall": round(m51_metrics["recall"], 4),
            "cand_recall": round(cand_metrics["recall"], 4),
            "delta_recall": round(delta_recall, 4),
        })

    # Simpan hasil
    out_json = cand_dir / "robustness_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    df_csv = pd.DataFrame(csv_rows)
    df_csv.to_csv(cand_dir / "robustness_evaluation.csv", index=False)

    # Buat plot robustness_comparison.png
    plot_path = cand_dir / "plots" / "robustness_comparison.png"
    plot_robustness_chart(results, plot_path)

    print(f"Robustness evaluation tersimpan di: {out_json}")
    print(f"Robustness plot tersimpan di: {plot_path}")
    return results


def plot_robustness_chart(results: Dict[str, Any], output_path: Path) -> None:
    """Membuat visualisasi komparatif performa model M5.1 vs M5.4 across conditions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conditions = ["Clean", "Audio Noisy", "Visual Degraded", "Combined"]
    cond_keys = ["clean", "audio_noisy", "visual_degraded", "combined"]

    m51_mae = [results["conditions"][k]["m51_iid"]["mae"] for k in cond_keys]
    m54_mae = [results["conditions"][k]["m54_modality_skew"]["mae"] for k in cond_keys]

    m51_f1 = [results["conditions"][k]["m51_iid"]["macro_f1"] for k in cond_keys]
    m54_f1 = [results["conditions"][k]["m54_modality_skew"]["macro_f1"] for k in cond_keys]

    cand_name = results.get("candidate_label", "M5.5 Combined Non-IID")
    m51_rec = [results["conditions"][k]["m51_iid"]["recall"] for k in cond_keys]
    m54_rec = [results["conditions"][k]["m54_modality_skew"]["recall"] for k in cond_keys]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    x = np.arange(len(conditions))
    width = 0.35

    # Panel 1: MAE
    ax = axes[0]
    ax.bar(x - width/2, m51_mae, width, label="M5.1 IID Control", color="#2b5c8f")
    ax.bar(x + width/2, m54_mae, width, label=cand_name, color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10, fontweight="bold")
    ax.set_ylabel("PHQ-8 MAE (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title("A. Continuous Estimation Error (MAE)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Panel 2: Macro F1
    ax = axes[1]
    ax.bar(x - width/2, m51_f1, width, label="M5.1 IID Control", color="#2b5c8f")
    ax.bar(x + width/2, m54_f1, width, label=cand_name, color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10, fontweight="bold")
    ax.set_ylabel("Macro F1-Score (Higher is Better)", fontsize=11, fontweight="bold")
    ax.set_title("B. Screening F1-Score", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Panel 3: Recall
    ax = axes[2]
    ax.bar(x - width/2, m51_rec, width, label="M5.1 IID Control", color="#2b5c8f")
    ax.bar(x + width/2, m54_rec, width, label=cand_name, color="#e67e22")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10, fontweight="bold")
    ax.set_ylabel("Sensitivity / Recall (Higher is Better)", fontsize=11, fontweight="bold")
    ax.set_title("C. Clinical Screening Recall", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    plt.suptitle(f"FED-MIND: Supplementary Robustness Stress-Testing ({cand_name} vs IID Control)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        # Cari run modality skew terbaru
        runs_dir = Path("storage/federated/runs")
        cands = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("FL-MODALITY-SKEW-")])
        if not cands:
            raise FileNotFoundError("Tidak ada direktori FL-MODALITY-SKEW ditemukan")
        run_dir = cands[-1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_robustness_evaluation(run_dir, device=device)
