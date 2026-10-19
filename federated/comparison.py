"""
federated/comparison.py
=======================
Modul Komparasi Ilmiah dan Generator Plot Kontrol M5.1 (FL-IID) vs M5.2 (Label Skew).
Menghasilkan:
- iid_comparison.json
- Plot komparatif:
  1. global_loss_iid_vs_label_skew.png
  2. mae_rmse_iid_vs_label_skew.png
  3. classification_metrics_iid_vs_label_skew.png
  4. recall_iid_vs_label_skew.png
  5. communication_iid_vs_label_skew.png
  6. client_drift_label_skew.png
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_M51_CONTROL_RUN_ID = "FL-RUN-20261016-130403-45D587"


def generate_iid_vs_label_skew_comparison(
    label_skew_run_dir: Union[str, Path],
    iid_control_run_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Membandingkan performa model akhir (Ronde 50) dan trajektori pelatihan
    antara M5.1 (FL-IID Control) dan M5.2 (Non-IID Label Skew).
    """
    ls_dir = Path(label_skew_run_dir)
    runs_base = ls_dir.parent
    iid_dir = Path(iid_control_run_dir or (runs_base / DEFAULT_M51_CONTROL_RUN_ID))

    if not iid_dir.exists():
        raise FileNotFoundError(f"Direktori run kontrol M5.1 tidak ditemukan: {iid_dir}")

    out_dir = Path(output_dir or ls_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Muat metrics.csv
    df_iid = pd.read_csv(iid_dir / "metrics.csv")
    df_ls = pd.read_csv(ls_dir / "metrics.csv")

    # 2. Muat communication.csv
    comm_iid = pd.read_csv(iid_dir / "communication.csv")
    comm_ls = pd.read_csv(ls_dir / "communication.csv")

    # 3. Baris Ronde 50 Akhir
    r50_iid = df_iid.iloc[-1].to_dict()
    r50_ls = df_ls.iloc[-1].to_dict()

    # Perhitungan FNR dan FPR jika belum tercatat di data lama
    def _calc_rates(row: Dict[str, Any]) -> Tuple[float, float]:
        tp = float(row.get("tp", 0))
        fn = float(row.get("fn", 0))
        tn = float(row.get("tn", 0))
        fp = float(row.get("fp", 0))
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return float(fnr), float(fpr)

    fnr_iid, fpr_iid = _calc_rates(r50_iid)
    fnr_ls, fpr_ls = _calc_rates(r50_ls)

    metric_keys = [
        ("mae", "Mean Absolute Error", False),
        ("rmse", "Root Mean Squared Error", False),
        ("mse", "Mean Squared Error", False),
        ("macro_f1", "Macro F1-Score", True),
        ("recall", "Recall / Sensitivity", True),
        ("specificity", "Specificity", True),
        ("auc_roc", "ROC-AUC", True),
        ("ccc", "Lin's CCC", True),
        ("pearson_r", "Pearson Correlation", True),
    ]

    metrics_comparison = {}
    for k, label, higher_is_better in metric_keys:
        val_iid = float(r50_iid.get(k, 0.0))
        val_ls = float(r50_ls.get(k, 0.0))
        delta = val_ls - val_iid
        rel_delta = (delta / val_iid * 100.0) if val_iid != 0 else 0.0

        metrics_comparison[k] = {
            "name": label,
            "iid_control": round(val_iid, 4),
            "label_skew": round(val_ls, 4),
            "absolute_delta": round(delta, 4),
            "relative_delta_pct": round(rel_delta, 2),
            "higher_is_better": higher_is_better,
        }

    # Metrik kesalahan klinis
    clinical_error_comparison = {
        "true_positives": {
            "iid_control": int(r50_iid.get("tp", 0)),
            "label_skew": int(r50_ls.get("tp", 0)),
            "delta": int(r50_ls.get("tp", 0)) - int(r50_iid.get("tp", 0)),
        },
        "false_negatives": {
            "iid_control": int(r50_iid.get("fn", 0)),
            "label_skew": int(r50_ls.get("fn", 0)),
            "delta": int(r50_ls.get("fn", 0)) - int(r50_iid.get("fn", 0)),
        },
        "true_negatives": {
            "iid_control": int(r50_iid.get("tn", 0)),
            "label_skew": int(r50_ls.get("tn", 0)),
            "delta": int(r50_ls.get("tn", 0)) - int(r50_iid.get("tn", 0)),
        },
        "false_positives": {
            "iid_control": int(r50_iid.get("fp", 0)),
            "label_skew": int(r50_ls.get("fp", 0)),
            "delta": int(r50_ls.get("fp", 0)) - int(r50_iid.get("fp", 0)),
        },
        "false_negative_rate": {
            "iid_control": round(fnr_iid, 4),
            "label_skew": round(fnr_ls, 4),
            "delta": round(fnr_ls - fnr_iid, 4),
        },
        "false_positive_rate": {
            "iid_control": round(fpr_iid, 4),
            "label_skew": round(fpr_ls, 4),
            "delta": round(fpr_ls - fpr_iid, 4),
        },
    }

    # 4. Late-round stability (Rounds 46-50)
    sub_iid_late = df_iid[(df_iid["round"] >= 46) & (df_iid["round"] <= 50)]
    sub_ls_late = df_ls[(df_ls["round"] >= 46) & (df_ls["round"] <= 50)]

    late_stability = {
        "mae": {
            "iid_mean": round(float(sub_iid_late["mae"].mean()), 4),
            "iid_std": round(float(sub_iid_late["mae"].std()), 4),
            "label_skew_mean": round(float(sub_ls_late["mae"].mean()), 4),
            "label_skew_std": round(float(sub_ls_late["mae"].std()), 4),
        },
        "macro_f1": {
            "iid_mean": round(float(sub_iid_late["macro_f1"].mean()), 4),
            "iid_std": round(float(sub_iid_late["macro_f1"].std()), 4),
            "label_skew_mean": round(float(sub_ls_late["macro_f1"].mean()), 4),
            "label_skew_std": round(float(sub_ls_late["macro_f1"].std()), 4),
        },
        "recall": {
            "iid_mean": round(float(sub_iid_late["recall"].mean()), 4),
            "iid_std": round(float(sub_iid_late["recall"].std()), 4),
            "label_skew_mean": round(float(sub_ls_late["recall"].mean()), 4),
            "label_skew_std": round(float(sub_ls_late["recall"].std()), 4),
        },
    }

    # 5. Komunikasi
    c_r50_iid = comm_iid.iloc[-1].to_dict()
    c_r50_ls = comm_ls.iloc[-1].to_dict()
    comm_comparison = {
        "cumulative_parameter_bytes_iid": int(c_r50_iid.get("cumulative_parameter_bytes", 0)),
        "cumulative_parameter_bytes_label_skew": int(c_r50_ls.get("cumulative_parameter_bytes", 0)),
        "cumulative_mb_iid": round(float(c_r50_iid.get("cumulative_mb", 0.0)), 4),
        "cumulative_mb_label_skew": round(float(c_r50_ls.get("cumulative_mb", 0.0)), 4),
        "cumulative_gb_iid": round(float(c_r50_iid.get("cumulative_mb", 0.0)) / 1024.0, 4),
        "cumulative_gb_label_skew": round(float(c_r50_ls.get("cumulative_mb", 0.0)) / 1024.0, 4),
        "delta_bytes": int(c_r50_ls.get("cumulative_parameter_bytes", 0)) - int(c_r50_iid.get("cumulative_parameter_bytes", 0)),
    }

    # 6. Analisis Client Drift
    client_drift_summary = {}
    cl_csv_path = ls_dir / "client_metrics.csv"
    if cl_csv_path.exists():
        cl_df = pd.read_csv(cl_csv_path)
        if "client_update_l2_norm" in cl_df.columns:
            round_drift = cl_df.groupby("round")["client_update_l2_norm"].agg(["mean", "max", "min"])
            client_drift_summary = {
                "overall_mean_drift": round(float(cl_df["client_update_l2_norm"].mean()), 4),
                "overall_max_drift": round(float(cl_df["client_update_l2_norm"].max()), 4),
                "round_01_mean_drift": round(float(cl_df[cl_df["round"] == 1]["client_update_l2_norm"].mean()), 4),
                "round_50_mean_drift": round(float(cl_df[cl_df["round"] == 50]["client_update_l2_norm"].mean()), 4) if (cl_df["round"] == 50).any() else None,
            }

    # 7. Susun berkas iid_comparison.json
    comparison_report = {
        "title": "FED-MIND: Controlled Label Skew (M5.2) vs IID Control Baseline (M5.1)",
        "control_run_id": iid_dir.name,
        "label_skew_run_id": ls_dir.name,
        "comparison_scope": "50 Communication Rounds (FedAvg, Seed 42, 3 Clients)",
        "metrics_round_50": metrics_comparison,
        "clinical_error_metrics": clinical_error_comparison,
        "late_round_stability_46_50": late_stability,
        "communication_accounting": comm_comparison,
        "client_drift_summary": client_drift_summary,
    }

    with open(out_dir / "iid_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    # 8. Hasilkan Plot Visualisasi
    _generate_comparison_plots(df_iid, df_ls, comm_iid, comm_ls, plots_dir, cl_csv_path)

    return comparison_report


def _generate_comparison_plots(
    df_iid: pd.DataFrame,
    df_ls: pd.DataFrame,
    comm_iid: pd.DataFrame,
    comm_ls: pd.DataFrame,
    plots_dir: Path,
    client_metrics_path: Optional[Path] = None,
) -> None:
    """Membuat seluruh grafik visual komparatif beresolusi tinggi (300 DPI)."""
    rounds = df_ls["round"]
    rounds_iid = df_iid["round"]

    # 1. Global Loss
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(rounds_iid, df_iid["test_loss"], label="M5.1 IID Control", color="#2b5c8f", linewidth=2)
    ax.plot(rounds, df_ls["test_loss"], label="M5.2 Non-IID Label Skew", color="#d95f02", linewidth=2, linestyle="--")
    ax.set_title("FED-MIND: Global Hold-Out Test Loss Convergence", fontsize=12, fontweight="bold")
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Weighted MSE Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / "global_loss_iid_vs_label_skew.png")
    plt.close(fig)

    # 2. MAE & RMSE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    ax1.plot(rounds_iid, df_iid["mae"], label="IID Control", color="#2b5c8f", linewidth=2)
    ax1.plot(rounds, df_ls["mae"], label="Label Skew", color="#d95f02", linewidth=2, linestyle="--")
    ax1.set_title("Mean Absolute Error (MAE)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("PHQ-8 Points")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(rounds_iid, df_iid["rmse"], label="IID Control", color="#2b5c8f", linewidth=2)
    ax2.plot(rounds, df_ls["rmse"], label="Label Skew", color="#d95f02", linewidth=2, linestyle="--")
    ax2.set_title("Root Mean Squared Error (RMSE)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("PHQ-8 Points")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "mae_rmse_iid_vs_label_skew.png")
    plt.close(fig)

    # 3. Classification Metrics (Macro F1 & AUC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    ax1.plot(rounds_iid, df_iid["macro_f1"], label="IID Control", color="#2b5c8f", linewidth=2)
    ax1.plot(rounds, df_ls["macro_f1"], label="Label Skew", color="#d95f02", linewidth=2, linestyle="--")
    ax1.set_title("Macro F1-Score (Cutoff >= 10.0)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Macro F1")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    auc_iid = df_iid["auc_roc"].fillna(0)
    auc_ls = df_ls["auc_roc"].fillna(0)
    ax2.plot(rounds_iid, auc_iid, label="IID Control", color="#2b5c8f", linewidth=2)
    ax2.plot(rounds, auc_ls, label="Label Skew", color="#d95f02", linewidth=2, linestyle="--")
    ax2.set_title("ROC-AUC Trajectory", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("AUC")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "classification_metrics_iid_vs_label_skew.png")
    plt.close(fig)

    # 4. Recall (Sensitivity) & FNR
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    ax1.plot(rounds_iid, df_iid["recall"], label="IID Control", color="#2b5c8f", linewidth=2)
    ax1.plot(rounds, df_ls["recall"], label="Label Skew", color="#d95f02", linewidth=2, linestyle="--")
    ax1.set_title("Recall / Sensitivity (Critical for Screening)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Recall")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Hitung FNR trajektori
    fnr_iid_traj = []
    for _, row in df_iid.iterrows():
        tp = row.get("tp", 0)
        fn = row.get("fn", 0)
        fnr_iid_traj.append(fn / (tp + fn) if (tp + fn) > 0 else 0.0)
    fnr_ls_traj = []
    for _, row in df_ls.iterrows():
        tp = row.get("tp", 0)
        fn = row.get("fn", 0)
        fnr_ls_traj.append(fn / (tp + fn) if (tp + fn) > 0 else 0.0)

    ax2.plot(rounds_iid, fnr_iid_traj, label="IID Control FNR", color="#2b5c8f", linewidth=2)
    ax2.plot(rounds, fnr_ls_traj, label="Label Skew FNR", color="#e41a1c", linewidth=2, linestyle="--")
    ax2.set_title("False Negative Rate (Missed Depression)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("FNR")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "recall_iid_vs_label_skew.png")
    plt.close(fig)

    # 5. Communication Overhead
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(comm_iid["round"], comm_iid["cumulative_mb"], label="IID Control Payload", color="#2b5c8f", linewidth=2)
    ax.plot(comm_ls["round"], comm_ls["cumulative_mb"], label="Label Skew Payload", color="#d95f02", linewidth=2, linestyle=":")
    ax.set_title("Cumulative Network Parameter Payload (MB)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Total Logical Parameter Transfer (MB)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "communication_iid_vs_label_skew.png")
    plt.close(fig)

    # 6. Client Drift (L2 Parameter Norm)
    if client_metrics_path and Path(client_metrics_path).exists():
        cl_df = pd.read_csv(client_metrics_path)
        if "client_update_l2_norm" in cl_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
            drift_grouped = cl_df.groupby(["round", "client_id"])["client_update_l2_norm"].mean().unstack()
            for col in drift_grouped.columns:
                ax.plot(drift_grouped.index, drift_grouped[col], label=f"Client {col}", alpha=0.7)
            mean_drift = cl_df.groupby("round")["client_update_l2_norm"].mean()
            ax.plot(mean_drift.index, mean_drift, label="Mean Client Drift", color="black", linewidth=2.5)
            ax.set_title("FED-MIND: Client Parameter Drift (L2 Norm to Global Model)", fontsize=12, fontweight="bold")
            ax.set_xlabel("Round")
            ax.set_ylabel("|| W_local - W_global ||_2")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots_dir / "client_drift_label_skew.png")
            plt.close(fig)


def generate_iid_vs_quantity_skew_comparison(
    quantity_skew_run_dir: Union[str, Path],
    iid_control_run_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Membandingkan performa model akhir (Ronde 50) dan trajektori pelatihan
    antara M5.1 (FL-IID Control) dan M5.3 (Non-IID Quantity Skew).
    """
    import shutil
    qs_dir = Path(quantity_skew_run_dir)
    runs_base = qs_dir.parent
    iid_dir = Path(iid_control_run_dir or (runs_base / DEFAULT_M51_CONTROL_RUN_ID))

    if not iid_dir.exists():
        raise FileNotFoundError(f"Direktori run kontrol M5.1 tidak ditemukan: {iid_dir}")

    out_dir = Path(output_dir or qs_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Muat metrics.csv
    df_iid = pd.read_csv(iid_dir / "metrics.csv")
    df_qs = pd.read_csv(qs_dir / "metrics.csv")

    # 2. Muat communication.csv
    comm_iid = pd.read_csv(iid_dir / "communication.csv")
    comm_qs = pd.read_csv(qs_dir / "communication.csv")

    # 3. Baris Ronde 50 Akhir
    r50_iid = df_iid.iloc[-1].to_dict()
    r50_qs = df_qs.iloc[-1].to_dict()

    def _calc_rates(row: Dict[str, Any]) -> Tuple[float, float]:
        tp = float(row.get("tp", 0))
        fn = float(row.get("fn", 0))
        tn = float(row.get("tn", 0))
        fp = float(row.get("fp", 0))
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return float(fnr), float(fpr)

    fnr_iid, fpr_iid = _calc_rates(r50_iid)
    fnr_qs, fpr_qs = _calc_rates(r50_qs)

    metric_keys = [
        ("mae", "Mean Absolute Error", False),
        ("rmse", "Root Mean Squared Error", False),
        ("mse", "Mean Squared Error", False),
        ("macro_f1", "Macro F1-Score", True),
        ("recall", "Recall / Sensitivity", True),
        ("specificity", "Specificity", True),
        ("auc_roc", "ROC-AUC", True),
        ("ccc", "Lin's CCC", True),
        ("pearson_r", "Pearson Correlation", True),
    ]

    metrics_comparison = {}
    for k, label, higher_is_better in metric_keys:
        val_iid = float(r50_iid.get(k, 0.0))
        val_qs = float(r50_qs.get(k, 0.0))
        delta = val_qs - val_iid
        rel_delta = (delta / val_iid * 100.0) if val_iid != 0 else 0.0

        metrics_comparison[k] = {
            "name": label,
            "iid_control": round(val_iid, 4),
            "quantity_skew": round(val_qs, 4),
            "absolute_delta": round(delta, 4),
            "relative_delta_pct": round(rel_delta, 2),
            "higher_is_better": higher_is_better,
        }

    # Metrik kesalahan klinis
    clinical_error_comparison = {
        "true_positives": {
            "iid_control": int(r50_iid.get("tp", 0)),
            "quantity_skew": int(r50_qs.get("tp", 0)),
            "delta": int(r50_qs.get("tp", 0)) - int(r50_iid.get("tp", 0)),
        },
        "false_negatives": {
            "iid_control": int(r50_iid.get("fn", 0)),
            "quantity_skew": int(r50_qs.get("fn", 0)),
            "delta": int(r50_qs.get("fn", 0)) - int(r50_iid.get("fn", 0)),
        },
        "true_negatives": {
            "iid_control": int(r50_iid.get("tn", 0)),
            "quantity_skew": int(r50_qs.get("tn", 0)),
            "delta": int(r50_qs.get("tn", 0)) - int(r50_iid.get("tn", 0)),
        },
        "false_positives": {
            "iid_control": int(r50_iid.get("fp", 0)),
            "quantity_skew": int(r50_qs.get("fp", 0)),
            "delta": int(r50_qs.get("fp", 0)) - int(r50_iid.get("fp", 0)),
        },
        "false_negative_rate": {
            "iid_control": round(fnr_iid, 4),
            "quantity_skew": round(fnr_qs, 4),
            "delta": round(fnr_qs - fnr_iid, 4),
        },
        "false_positive_rate": {
            "iid_control": round(fpr_iid, 4),
            "quantity_skew": round(fpr_qs, 4),
            "delta": round(fpr_qs - fpr_iid, 4),
        },
    }

    # 4. Late-round stability (Rounds 46-50)
    sub_iid_late = df_iid[(df_iid["round"] >= 46) & (df_iid["round"] <= 50)]
    sub_qs_late = df_qs[(df_qs["round"] >= 46) & (df_qs["round"] <= 50)]

    late_stability = {
        "mae": {
            "iid_mean": round(float(sub_iid_late["mae"].mean()), 4),
            "iid_std": round(float(sub_iid_late["mae"].std()), 4),
            "quantity_skew_mean": round(float(sub_qs_late["mae"].mean()), 4),
            "quantity_skew_std": round(float(sub_qs_late["mae"].std()), 4),
        },
        "rmse": {
            "iid_mean": round(float(sub_iid_late["rmse"].mean()), 4),
            "iid_std": round(float(sub_iid_late["rmse"].std()), 4),
            "quantity_skew_mean": round(float(sub_qs_late["rmse"].mean()), 4),
            "quantity_skew_std": round(float(sub_qs_late["rmse"].std()), 4),
        },
        "macro_f1": {
            "iid_mean": round(float(sub_iid_late["macro_f1"].mean()), 4),
            "iid_std": round(float(sub_iid_late["macro_f1"].std()), 4),
            "quantity_skew_mean": round(float(sub_qs_late["macro_f1"].mean()), 4),
            "quantity_skew_std": round(float(sub_qs_late["macro_f1"].std()), 4),
        },
        "recall": {
            "iid_mean": round(float(sub_iid_late["recall"].mean()), 4),
            "iid_std": round(float(sub_iid_late["recall"].std()), 4),
            "quantity_skew_mean": round(float(sub_qs_late["recall"].mean()), 4),
            "quantity_skew_std": round(float(sub_qs_late["recall"].std()), 4),
        },
    }

    # 5. Komunikasi
    c_r50_iid = comm_iid.iloc[-1].to_dict()
    c_r50_qs = comm_qs.iloc[-1].to_dict()
    comm_comparison = {
        "cumulative_parameter_bytes_iid": int(c_r50_iid.get("cumulative_parameter_bytes", 0)),
        "cumulative_parameter_bytes_quantity_skew": int(c_r50_qs.get("cumulative_parameter_bytes", 0)),
        "cumulative_mb_iid": round(float(c_r50_iid.get("cumulative_mb", 0.0)), 4),
        "cumulative_mb_quantity_skew": round(float(c_r50_qs.get("cumulative_mb", 0.0)), 4),
        "cumulative_gb_iid": round(float(c_r50_iid.get("cumulative_mb", 0.0)) / 1024.0, 4),
        "cumulative_gb_quantity_skew": round(float(c_r50_qs.get("cumulative_mb", 0.0)) / 1024.0, 4),
        "delta_bytes": int(c_r50_qs.get("cumulative_parameter_bytes", 0)) - int(c_r50_iid.get("cumulative_parameter_bytes", 0)),
    }

    # 6. Analisis Client Dominance & Training Dynamics
    client_dominance_summary: Dict[str, Any] = {}
    cl_csv_path = qs_dir / "client_metrics.csv"
    if cl_csv_path.exists():
        cl_df = pd.read_csv(cl_csv_path)
        # Pisahkan per klien berdasarkan num_examples: 98 -> C1, 49 -> C2, 16 -> C3
        c_stats = {}
        for n_ex, c_label in [(98, "client_1"), (49, "client_2"), (16, "client_3")]:
            sub_c = cl_df[cl_df["num_examples"] == n_ex]
            if len(sub_c) > 0:
                mean_norm = float(sub_c["client_update_l2_norm"].mean())
                weighted_norm = float(mean_norm * (n_ex / 163.0))
                opt_steps = int(sub_c["optimizer_steps"].iloc[0]) if "optimizer_steps" in sub_c.columns else (
                    14 if n_ex == 98 else (8 if n_ex == 49 else 2)
                )
                c_stats[c_label] = {
                    "num_examples": n_ex,
                    "fedavg_weight_pct": round((n_ex / 163.0) * 100, 2),
                    "optimizer_steps_per_round": opt_steps,
                    "mean_update_l2_norm": round(mean_norm, 4),
                    "mean_weighted_update_l2_norm": round(weighted_norm, 4),
                    "mean_train_loss": round(float(sub_c["train_loss"].mean()), 4),
                    "mean_train_mae": round(float(sub_c["train_mae"].mean()), 4),
                    "mean_val_loss": round(float(sub_c["val_loss"].mean()), 4),
                    "mean_val_mae": round(float(sub_c["val_mae"].mean()), 4),
                }

        client_dominance_summary = {
            "per_client_dynamics": c_stats,
            "largest_client_weight": "60.12% (98/163)",
            "smallest_client_weight": "9.82% (16/163)",
            "aggregation_formula": "w_global = (98/163)*w1 + (49/163)*w2 + (16/163)*w3",
        }

    # 7. Susun berkas iid_comparison.json
    comparison_report = {
        "title": "FED-MIND: Controlled Quantity Skew (M5.3) vs IID Control Baseline (M5.1)",
        "control_run_id": iid_dir.name,
        "quantity_skew_run_id": qs_dir.name,
        "comparison_scope": "50 Communication Rounds (FedAvg, Seed 42, 3 Clients, 60:30:10 Allocation)",
        "quantity_allocation": {
            "client_1": {"samples": 98, "weight": "98/163 (60.12%)"},
            "client_2": {"samples": 49, "weight": "49/163 (30.06%)"},
            "client_3": {"samples": 16, "weight": "16/163 (9.82%)"},
            "total": 163,
        },
        "metrics_round_50": metrics_comparison,
        "clinical_error_metrics": clinical_error_comparison,
        "late_round_stability_46_50": late_stability,
        "communication_accounting": comm_comparison,
        "client_dominance_summary": client_dominance_summary,
    }

    with open(out_dir / "iid_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    # 8. Pastikan partisi plots disalin ke direktori plots
    part_plots_dir = Path("storage/federated/partitions/quantity_skew_seed42/plots")
    if part_plots_dir.exists():
        for p_f in part_plots_dir.glob("*.png"):
            dest_f = plots_dir / p_f.name
            if not dest_f.exists():
                shutil.copy2(p_f, dest_f)

    # 9. Hasilkan Plot Visualisasi Quantity Skew
    _generate_quantity_comparison_plots(df_iid, df_qs, comm_iid, comm_qs, plots_dir, cl_csv_path)

    return comparison_report


def _generate_quantity_comparison_plots(
    df_iid: pd.DataFrame,
    df_qs: pd.DataFrame,
    comm_iid: pd.DataFrame,
    comm_qs: pd.DataFrame,
    plots_dir: Path,
    client_metrics_path: Optional[Path] = None,
) -> None:
    """Membuat seluruh grafik visual komparatif beresolusi tinggi (300 DPI) untuk Quantity Skew."""
    rounds = df_qs["round"]
    rounds_iid = df_iid["round"]

    # 1. Global Loss: global_loss_iid_vs_quantity.png
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(rounds_iid, df_iid["test_loss"], label="M5.1 IID Control (54/55/54)", color="#2b5c8f", linewidth=2)
    ax.plot(rounds, df_qs["test_loss"], label="M5.3 Quantity Skew (98/49/16)", color="#d95f02", linewidth=2, linestyle="--")
    ax.set_title("FED-MIND: Global Hold-Out Test Loss (IID vs. Quantity Skew)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Weighted MSE Loss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / "global_loss_iid_vs_quantity.png")
    plt.close(fig)

    # 2. MAE & RMSE: mae_rmse_iid_vs_quantity.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    ax1.plot(rounds_iid, df_iid["mae"], label="IID Control", color="#2b5c8f", linewidth=2)
    ax1.plot(rounds, df_qs["mae"], label="Quantity Skew", color="#d95f02", linewidth=2, linestyle="--")
    ax1.set_title("Mean Absolute Error (MAE)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("PHQ-8 Points")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(rounds_iid, df_iid["rmse"], label="IID Control", color="#2b5c8f", linewidth=2)
    ax2.plot(rounds, df_qs["rmse"], label="Quantity Skew", color="#d95f02", linewidth=2, linestyle="--")
    ax2.set_title("Root Mean Squared Error (RMSE)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("PHQ-8 Points")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "mae_rmse_iid_vs_quantity.png")
    plt.close(fig)

    # 3. Macro F1 & Recall: macro_f1_recall_iid_vs_quantity.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    ax1.plot(rounds_iid, df_iid["macro_f1"], label="IID Control", color="#2b5c8f", linewidth=2)
    ax1.plot(rounds, df_qs["macro_f1"], label="Quantity Skew", color="#d95f02", linewidth=2, linestyle="--")
    ax1.set_title("Macro F1-Score (Cutoff >= 10.0)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Macro F1")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(rounds_iid, df_iid["recall"], label="IID Control Recall", color="#2b5c8f", linewidth=2)
    ax2.plot(rounds, df_qs["recall"], label="Quantity Skew Recall", color="#d95f02", linewidth=2, linestyle="--")
    ax2.set_title("Recall / Sensitivity Trajectory", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Recall")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "macro_f1_recall_iid_vs_quantity.png")
    plt.close(fig)

    # 4. Communication: communication_iid_vs_quantity.png
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.plot(comm_iid["round"], comm_iid["cumulative_mb"], label="IID Control (54/55/54)", color="#2b5c8f", linewidth=2)
    ax.plot(comm_qs["round"], comm_qs["cumulative_mb"], label="Quantity Skew (98/49/16)", color="#d95f02", linewidth=2, linestyle=":")
    ax.set_title("Cumulative Network Parameter Payload (MB)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Total Logical Parameter Transfer (MB)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "communication_iid_vs_quantity.png")
    plt.close(fig)

    # 5. Client Update Norms & 6. Weighted Contributions
    if client_metrics_path and Path(client_metrics_path).exists():
        cl_df = pd.read_csv(client_metrics_path)
        if "client_update_l2_norm" in cl_df.columns:
            # Client mapping: 98 -> C1, 49 -> C2, 16 -> C3
            def _map_c(r):
                n = r["num_examples"]
                if n == 98: return "Client 1 (N=98, 60.1%)"
                if n == 49: return "Client 2 (N=49, 30.1%)"
                if n == 16: return "Client 3 (N=16, 9.8%)"
                return str(r["client_id"])
            cl_df["client_label"] = cl_df.apply(_map_c, axis=1)

            # 5. client_update_norms.png
            fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
            norm_grouped = cl_df.groupby(["round", "client_label"])["client_update_l2_norm"].mean().unstack()
            c_colors = {"Client 1 (N=98, 60.1%)": "#1d4ed8", "Client 2 (N=49, 30.1%)": "#059669", "Client 3 (N=16, 9.8%)": "#dc2626"}
            for col in norm_grouped.columns:
                ax.plot(norm_grouped.index, norm_grouped[col], label=col, color=c_colors.get(col, None), linewidth=1.8, alpha=0.85)
            mean_norm = cl_df.groupby("round")["client_update_l2_norm"].mean()
            ax.plot(mean_norm.index, mean_norm, label="Mean Client Update Norm", color="black", linewidth=2.2, linestyle="--")
            ax.set_title("FED-MIND: Local Client Update L2 Norms ||Δw_i||", fontsize=12, fontweight="bold")
            ax.set_xlabel("Round")
            ax.set_ylabel("L2 Norm || W_local - W_global ||_2")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(plots_dir / "client_update_norms.png")
            plt.close(fig)

            # 6. client_weighted_contributions.png
            fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
            # Weighted norm = (n_i / 163) * client_update_l2_norm
            cl_df["weighted_norm"] = cl_df["client_update_l2_norm"] * (cl_df["num_examples"] / 163.0)
            wt_grouped = cl_df.groupby(["round", "client_label"])["weighted_norm"].mean().unstack()
            for col in wt_grouped.columns:
                ax.plot(wt_grouped.index, wt_grouped[col], label=f"Weighted: {col}", color=c_colors.get(col, None), linewidth=2.0)
            ax.set_title("FED-MIND: FedAvg Weighted Contribution ( (n_i / N) * ||Δw_i|| )", fontsize=12, fontweight="bold")
            ax.set_xlabel("Round")
            ax.set_ylabel("Weighted Contribution Magnitude")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(plots_dir / "client_weighted_contributions.png")
            plt.close(fig)

