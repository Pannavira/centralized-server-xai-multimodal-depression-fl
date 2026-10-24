"""
scripts/analyze_m5_6_results.py
===============================
Consolidated Statistical Analysis and Visualization Engine for FED-MIND Milestone 5.6.

Implements:
1. Canonical evaluation extraction from all valid runs in run_registry.json.
2. Per-seed metrics extraction -> seed_metrics.csv.
3. Scenario summary statistics (Mean, Sample SD ddof=1, Median, Min, Max, 95% Student-t CI) -> summary_statistics.csv & json.
4. Paired comparisons vs IID (same seed pairing): delta_metric(S, X) = metric(S, X) - metric(S, IID) -> paired_deltas.csv & paired_summary.csv.
5. Clinical screening error analysis (TP, TN, FP, FN, Recall, Specificity, FNR, FPR) -> clinical_error_summary.csv.
6. Client drift analysis -> client_drift_summary.csv.
7. Communication overhead payload analysis -> communication_summary.csv.
8. 13 Publication-Quality Plots saved to storage/federated/multiseed/m5_6/plots/.
"""

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
logger = logging.getLogger("m5_6_analysis")

REPO_ROOT = Path("C:/Wira/Kuliah/Penelitian/PROTOTYPE APLIKASI FL/centralized-server-xai-multimodal-depression-fl")
OUTPUT_DIR = REPO_ROOT / "storage" / "federated" / "multiseed" / "m5_6"
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS_ORDER = [
    ("IID_3Clients", "FL-IID"),
    ("label_skew", "Non-IID Label Skew"),
    ("quantity_skew", "Non-IID Quantity Skew"),
    ("modality_skew", "Non-IID Modality Skew"),
    ("combined_skew", "Non-IID Combined Skew"),
]

METRICS_TO_ANALYZE = [
    ("mae", "PHQ-8 MAE", True),             # lower is better
    ("rmse", "PHQ-8 RMSE", True),           # lower is better
    ("mse", "PHQ-8 MSE", True),             # lower is better
    ("macro_f1", "Macro F1", False),        # higher is better
    ("binary_f1", "Binary F1", False),      # higher is better
    ("recall", "Sensitivity / Recall", False),# higher is better
    ("specificity", "Specificity", False),  # higher is better
    ("auc_roc", "ROC-AUC", False),          # higher is better
    ("ccc", "Lin's CCC", False),            # higher is better
    ("pearson_r", "Pearson r", False),      # higher is better
]


def compute_student_t_ci(values: List[float], confidence: float = 0.95) -> Tuple[Optional[float], Optional[float]]:
    """Menghitung 95% Confidence Interval untuk rata-rata sampel menggunakan distribusi Student-t."""
    n = len(values)
    if n < 2:
        return None, None
    mean = float(np.mean(values))
    std_err = float(stats.sem(values, ddof=1))
    if std_err == 0.0 or math.isnan(std_err):
        return mean, mean
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    margin = t_crit * std_err
    return round(mean - margin, 4), round(mean + margin, 4)


def extract_seed_metrics(registry: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Mengekstraksi metrik ronde 50 (dan R0) untuk setiap pasangan scenario-seed yang valid."""
    rows = []
    
    for key, item in sorted(registry.items()):
        status = item.get("completion_status")
        if status not in ("REUSED_VERIFIED", "COMPLETED"):
            continue
            
        run_dir = Path(item["run_directory"])
        r50_meta_f = run_dir / "rounds" / "round_050" / "metadata.json"
        if not r50_meta_f.exists():
            continue
            
        with open(r50_meta_f, "r", encoding="utf-8") as f:
            r50_meta = json.load(f)
            
        eval_metrics = r50_meta.get("global_evaluation_metrics", {})
        comm_meta = r50_meta.get("communication_estimate", {})
        
        row = {
            "matrix_key": key,
            "scenario": item.get("scenario"),
            "display_name": item.get("display_name"),
            "seed": int(item.get("seed")),
            "run_id": item.get("run_id"),
            "status": status,
            "source": item.get("source"),
            "initial_checkpoint_sha256": item.get("initialization_sha256"),
            "final_checkpoint_sha256": item.get("final_checkpoint_sha256"),
            
            # Regression metrics
            "mae": float(eval_metrics.get("mae", np.nan)),
            "rmse": float(eval_metrics.get("rmse", np.nan)),
            "mse": float(eval_metrics.get("mse", np.nan)),
            "ccc": float(eval_metrics.get("ccc", np.nan)),
            "pearson_r": float(eval_metrics.get("pearson_r", np.nan)),
            
            # Classification metrics
            "macro_f1": float(eval_metrics.get("macro_f1", np.nan)),
            "binary_f1": float(eval_metrics.get("binary_f1", np.nan)),
            "recall": float(eval_metrics.get("recall", np.nan)),
            "specificity": float(eval_metrics.get("specificity", np.nan)),
            "auc_roc": float(eval_metrics.get("auc_roc", np.nan)),
            "accuracy": float(eval_metrics.get("accuracy", np.nan)),
            
            # Clinical confusion matrix counts
            "tp": int(eval_metrics.get("tp", 0)),
            "tn": int(eval_metrics.get("tn", 0)),
            "fp": int(eval_metrics.get("fp", 0)),
            "fn": int(eval_metrics.get("fn", 0)),
            "fnr": float(eval_metrics.get("false_negative_rate", int(eval_metrics.get("fn", 0)) / (int(eval_metrics.get("fn", 0)) + int(eval_metrics.get("tp", 0))) if (int(eval_metrics.get("fn", 0)) + int(eval_metrics.get("tp", 0))) > 0 else np.nan)),
            "fpr": float(eval_metrics.get("false_positive_rate", int(eval_metrics.get("fp", 0)) / (int(eval_metrics.get("fp", 0)) + int(eval_metrics.get("tn", 0))) if (int(eval_metrics.get("fp", 0)) + int(eval_metrics.get("tn", 0))) > 0 else np.nan)),
            
            # Communication
            "cumulative_parameter_bytes": int(comm_meta.get("cumulative_parameter_bytes", 9398401200)),
        }
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "seed_metrics.csv", index=False)
    logger.info(f"Saved seed_metrics.csv with {len(df)} valid runs.")
    return df


def generate_summary_statistics(seed_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Menghitung Mean, Sample SD (ddof=1), Median, Min, Max, dan 95% CI untuk setiap skenario."""
    summary_rows = []
    summary_dict = {}
    
    for scen_id, disp_name in SCENARIOS_ORDER:
        scen_df = seed_df[seed_df["scenario"] == scen_id]
        n_valid = len(scen_df)
        if n_valid == 0:
            continue
            
        summary_dict[scen_id] = {
            "display_name": disp_name,
            "n_valid_runs": n_valid,
            "seeds": scen_df["seed"].tolist(),
            "metrics": {},
        }
        
        for m_key, m_label, _ in METRICS_TO_ANALYZE:
            vals = scen_df[m_key].dropna().tolist()
            if not vals:
                continue
                
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            med_val = float(np.median(vals))
            min_val = float(np.min(vals))
            max_val = float(np.max(vals))
            ci_low, ci_high = compute_student_t_ci(vals, 0.95)
            
            row = {
                "scenario": scen_id,
                "display_name": disp_name,
                "metric": m_key,
                "metric_label": m_label,
                "n_runs": len(vals),
                "mean": round(mean_val, 4),
                "sample_std": round(std_val, 4),
                "median": round(med_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
            }
            summary_rows.append(row)
            
            summary_dict[scen_id]["metrics"][m_key] = {
                "mean": round(mean_val, 4),
                "sample_std": round(std_val, 4),
                "median": round(med_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
            }
            
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False)
    with open(OUTPUT_DIR / "summary_statistics.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
        
    logger.info("Saved summary_statistics.csv and summary_statistics.json.")
    return sum_df, summary_dict


def compute_paired_comparisons_vs_iid(seed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Menghitung delta berpasangan terhadap IID untuk seed yang sama: delta = metric(S, X) - metric(S, IID)."""
    iid_df = seed_df[seed_df["scenario"] == "IID_3Clients"].set_index("seed")
    if iid_df.empty:
        logger.warning("No IID baseline runs found for paired comparisons!")
        return pd.DataFrame(), pd.DataFrame()
        
    paired_rows = []
    
    for scen_id, disp_name in SCENARIOS_ORDER:
        if scen_id == "IID_3Clients":
            continue
            
        scen_df = seed_df[seed_df["scenario"] == scen_id]
        for _, row in scen_df.iterrows():
            seed = int(row["seed"])
            if seed not in iid_df.index:
                logger.warning(f"Seed {seed} missing in IID, skipping paired delta for {scen_id}")
                continue
                
            iid_row = iid_df.loc[seed]
            p_row = {
                "scenario": scen_id,
                "display_name": disp_name,
                "seed": seed,
                "run_id": row["run_id"],
                "iid_run_id": iid_row["run_id"],
            }
            
            for m_key, _, _ in METRICS_TO_ANALYZE:
                val_scen = row.get(m_key)
                val_iid = iid_row.get(m_key)
                if pd.notna(val_scen) and pd.notna(val_iid):
                    delta = float(val_scen) - float(val_iid)
                    p_row[f"delta_{m_key}"] = round(delta, 4)
                else:
                    p_row[f"delta_{m_key}"] = np.nan
                    
            paired_rows.append(p_row)
            
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(OUTPUT_DIR / "paired_deltas.csv", index=False)
    
    # Hitung ringkasan paired deltas
    summary_paired = []
    for scen_id, disp_name in SCENARIOS_ORDER:
        if scen_id == "IID_3Clients":
            continue
            
        sub_df = paired_df[paired_df["scenario"] == scen_id]
        if sub_df.empty:
            continue
            
        for m_key, m_label, lower_is_better in METRICS_TO_ANALYZE:
            col = f"delta_{m_key}"
            deltas = sub_df[col].dropna().tolist()
            if not deltas:
                continue
                
            mean_d = float(np.mean(deltas))
            std_d = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
            med_d = float(np.median(deltas))
            ci_low, ci_high = compute_student_t_ci(deltas, 0.95)
            
            n_pos = sum(1 for d in deltas if d > 1e-6)
            n_neg = sum(1 for d in deltas if d < -1e-6)
            n_zero = len(deltas) - n_pos - n_neg
            
            summary_paired.append({
                "scenario": scen_id,
                "display_name": disp_name,
                "metric": m_key,
                "metric_label": m_label,
                "lower_is_better": lower_is_better,
                "n_paired_runs": len(deltas),
                "mean_paired_delta": round(mean_d, 4),
                "sample_std_paired_delta": round(std_d, 4),
                "median_paired_delta": round(med_d, 4),
                "ci_95_lower": ci_low,
                "ci_95_upper": ci_high,
                "n_positive_deltas": n_pos,
                "n_negative_deltas": n_neg,
                "n_zero_deltas": n_zero,
            })
            
    summary_paired_df = pd.DataFrame(summary_paired)
    summary_paired_df.to_csv(OUTPUT_DIR / "paired_summary.csv", index=False)
    logger.info("Saved paired_deltas.csv and paired_summary.csv.")
    return paired_df, summary_paired_df


def generate_clinical_error_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Membuat ringkasan kesalahan klinis (FN, FP, FNR, FPR) per skenario lintas seed."""
    rows = []
    
    for scen_id, disp_name in SCENARIOS_ORDER:
        sub = seed_df[seed_df["scenario"] == scen_id]
        if sub.empty:
            continue
            
        fn_vals = sub["fn"].tolist()
        fp_vals = sub["fp"].tolist()
        fnr_vals = sub["fnr"].tolist()
        fpr_vals = sub["fpr"].tolist()
        rec_vals = sub["recall"].tolist()
        spec_vals = sub["specificity"].tolist()
        
        rows.append({
            "scenario": scen_id,
            "display_name": disp_name,
            "n_runs": len(sub),
            "mean_fn": round(float(np.mean(fn_vals)), 2),
            "std_fn": round(float(np.std(fn_vals, ddof=1)), 2) if len(fn_vals) > 1 else 0.0,
            "min_fn": int(np.min(fn_vals)),
            "max_fn": int(np.max(fn_vals)),
            "mean_fp": round(float(np.mean(fp_vals)), 2),
            "std_fp": round(float(np.std(fp_vals, ddof=1)), 2) if len(fp_vals) > 1 else 0.0,
            "min_fp": int(np.min(fp_vals)),
            "max_fp": int(np.max(fp_vals)),
            "mean_recall": round(float(np.mean(rec_vals)), 4),
            "mean_specificity": round(float(np.mean(spec_vals)), 4),
            "mean_fnr": round(float(np.mean(fnr_vals)), 4),
            "mean_fpr": round(float(np.mean(fpr_vals)), 4),
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "clinical_error_summary.csv", index=False)
    logger.info("Saved clinical_error_summary.csv.")
    return df


def generate_communication_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan beban komunikasi model parameter dalam unit byte, MiB, dan GiB."""
    model_params = 7832001
    model_bytes = model_params * 4
    bytes_per_round = 2 * 3 * model_bytes  # 3 clients downlink + uplink
    full_run_bytes = 50 * bytes_per_round   # 9,398,401,200 bytes
    
    rows = []
    for scen_id, disp_name in SCENARIOS_ORDER:
        sub = seed_df[seed_df["scenario"] == scen_id]
        if sub.empty:
            continue
        rows.append({
            "scenario": scen_id,
            "display_name": disp_name,
            "valid_completed_runs": len(sub),
            "model_parameters": model_params,
            "fp32_model_bytes": model_bytes,
            "fp32_model_mib": round(model_bytes / (1024**2), 4),
            "per_round_bidirectional_bytes": bytes_per_round,
            "per_round_bidirectional_mib": round(bytes_per_round / (1024**2), 4),
            "total_50_rounds_bytes_per_run": full_run_bytes,
            "total_50_rounds_gib_per_run": round(full_run_bytes / (1024**3), 4),
            "total_scenario_bytes": len(sub) * full_run_bytes,
            "total_scenario_gib": round((len(sub) * full_run_bytes) / (1024**3), 4),
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "communication_summary.csv", index=False)
    logger.info("Saved communication_summary.csv.")
    return df


def generate_client_drift_summary(registry: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Mengekstraksi client drift dan norma update L2 rata-rata lintas ronde dan seed."""
    drift_records = []
    
    for key, item in registry.items():
        if item.get("completion_status") not in ("REUSED_VERIFIED", "COMPLETED"):
            continue
        run_dir = Path(item["run_directory"])
        client_metrics_f = run_dir / "client_metrics.csv"
        if not client_metrics_f.exists():
            continue
            
        try:
            cm_df = pd.read_csv(client_metrics_f)
            if "client_update_l2_norm" in cm_df.columns:
                mean_l2 = float(cm_df["client_update_l2_norm"].mean())
                weighted_l2 = float(cm_df["weighted_update_l2_norm"].mean()) if "weighted_update_l2_norm" in cm_df.columns else mean_l2
                drift_records.append({
                    "scenario": item.get("scenario"),
                    "display_name": item.get("display_name"),
                    "seed": item.get("seed"),
                    "run_id": item.get("run_id"),
                    "mean_raw_client_l2_norm": round(mean_l2, 4),
                    "mean_weighted_l2_norm": round(weighted_l2, 4),
                })
        except Exception as e:
            logger.debug(f"Could not parse client_metrics for {key}: {e}")
            
    df = pd.DataFrame(drift_records)
    if not df.empty:
        summary_drift = df.groupby(["scenario", "display_name"]).agg(
            n_runs=("seed", "count"),
            avg_raw_l2_norm=("mean_raw_client_l2_norm", "mean"),
            std_raw_l2_norm=("mean_raw_client_l2_norm", lambda x: np.std(x, ddof=1) if len(x)>1 else 0.0),
            avg_weighted_l2_norm=("mean_weighted_l2_norm", "mean"),
        ).reset_index()
        summary_drift.to_csv(OUTPUT_DIR / "client_drift_summary.csv", index=False)
        logger.info("Saved client_drift_summary.csv.")
    return df


def generate_13_visualizations(seed_df: pd.DataFrame, paired_df: pd.DataFrame, registry: Dict[str, Dict[str, Any]]) -> None:
    """Membuat 13 grafik visualisasi ilmiah siap publikasi."""
    logger.info("Generating 13 publication-quality visualizations...")
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77"]
    
    unique_seeds = sorted([int(s) for s in seed_df["seed"].unique()])
    n_seeds = len(unique_seeds)
    if n_seeds < 5:
        status_tag = f"[PRELIMINARY — n={n_seeds} seeds: {unique_seeds}]"
    else:
        status_tag = f"[Final 5-Seed Benchmark: Seeds {unique_seeds}]"
        
    logger.info(f"Visualizations status tag: {status_tag}")
    
    # 1. mae_by_scenario_mean_ci.png
    fig, ax = plt.subplots(figsize=(8, 5))
    scens = [s[0] for s in SCENARIOS_ORDER]
    names = [s[1] for s in SCENARIOS_ORDER]
    means = [seed_df[seed_df["scenario"] == s]["mae"].mean() for s in scens]
    stds = [seed_df[seed_df["scenario"] == s]["mae"].std(ddof=1) for s in scens]
    yerr = [s if not math.isnan(s) else 0 for s in stds]
    ax.bar(names, means, yerr=yerr, capsize=5, color=colors, alpha=0.85, edgecolor="black", label="Mean ± Sample SD (ddof=1)")
    ax.set_ylabel("PHQ-8 MAE (Lower is Better)")
    ax.set_title(f"PHQ-8 Test Set MAE Across Scenarios\n{status_tag}")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mae_by_scenario_mean_ci.png", dpi=300)
    plt.close()

    # 2. macro_f1_by_scenario_mean_ci.png
    fig, ax = plt.subplots(figsize=(8, 5))
    means_f1 = [seed_df[seed_df["scenario"] == s]["macro_f1"].mean() for s in scens]
    stds_f1 = [seed_df[seed_df["scenario"] == s]["macro_f1"].std(ddof=1) for s in scens]
    ax.bar(names, means_f1, yerr=stds_f1, capsize=5, color=colors, alpha=0.85, edgecolor="black", label="Mean ± Sample SD (ddof=1)")
    ax.set_ylabel("Macro F1 (Higher is Better)")
    ax.set_title(f"Screening Macro F1 Across Scenarios\n{status_tag}")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "macro_f1_by_scenario_mean_ci.png", dpi=300)
    plt.close()

    # 3. recall_by_scenario_mean_ci.png
    fig, ax = plt.subplots(figsize=(8, 5))
    means_rec = [seed_df[seed_df["scenario"] == s]["recall"].mean() for s in scens]
    stds_rec = [seed_df[seed_df["scenario"] == s]["recall"].std(ddof=1) for s in scens]
    ax.bar(names, means_rec, yerr=stds_rec, capsize=5, color=colors, alpha=0.85, edgecolor="black", label="Mean ± Sample SD (ddof=1)")
    ax.set_ylabel("Sensitivity / Recall (Higher is Better)")
    ax.set_title(f"Sensitivity / Recall Across Scenarios\n{status_tag}")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "recall_by_scenario_mean_ci.png", dpi=300)
    plt.close()

    # 4. auc_by_scenario_mean_ci.png
    fig, ax = plt.subplots(figsize=(8, 5))
    means_auc = [seed_df[seed_df["scenario"] == s]["auc_roc"].mean() for s in scens]
    stds_auc = [seed_df[seed_df["scenario"] == s]["auc_roc"].std(ddof=1) for s in scens]
    ax.bar(names, means_auc, yerr=stds_auc, capsize=5, color=colors, alpha=0.85, edgecolor="black", label="Mean ± Sample SD (ddof=1)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"ROC-AUC Across Scenarios\n{status_tag}")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "auc_by_scenario_mean_ci.png", dpi=300)
    plt.close()

    # 5. iid_paired_mae_deltas.png
    if not paired_df.empty and "delta_mae" in paired_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        non_iid_scens = [s[0] for s in SCENARIOS_ORDER if s[0] != "IID_3Clients"]
        non_iid_names = [s[1] for s in SCENARIOS_ORDER if s[0] != "IID_3Clients"]
        for idx, s in enumerate(non_iid_scens):
            s_deltas = paired_df[paired_df["scenario"] == s]["delta_mae"]
            ax.scatter([non_iid_names[idx]] * len(s_deltas), s_deltas, color=colors[idx+1], s=80, alpha=0.8, edgecolor="black", label=f"{non_iid_names[idx]} Seeds" if idx==0 else "")
            ax.plot([idx-0.2, idx+0.2], [s_deltas.mean(), s_deltas.mean()], color="black", lw=3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_ylabel("Δ MAE vs IID (Negative = Better than IID)")
        ax.set_title(f"Paired ΔMAE vs IID (Same-Seed Pairings)\n{status_tag}")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "iid_paired_mae_deltas.png", dpi=300)
        plt.close()

    # 6. iid_paired_f1_deltas.png
    if not paired_df.empty and "delta_macro_f1" in paired_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, s in enumerate(non_iid_scens):
            s_deltas = paired_df[paired_df["scenario"] == s]["delta_macro_f1"]
            ax.scatter([non_iid_names[idx]] * len(s_deltas), s_deltas, color=colors[idx+1], s=80, alpha=0.8, edgecolor="black")
            ax.plot([idx-0.2, idx+0.2], [s_deltas.mean(), s_deltas.mean()], color="black", lw=3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_ylabel("Δ Macro F1 vs IID (Positive = Better than IID)")
        ax.set_title(f"Paired ΔMacro F1 vs IID (Same-Seed Pairings)\n{status_tag}")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "iid_paired_f1_deltas.png", dpi=300)
        plt.close()

    # 7. iid_paired_recall_deltas.png
    if not paired_df.empty and "delta_recall" in paired_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, s in enumerate(non_iid_scens):
            s_deltas = paired_df[paired_df["scenario"] == s]["delta_recall"]
            ax.scatter([non_iid_names[idx]] * len(s_deltas), s_deltas, color=colors[idx+1], s=80, alpha=0.8, edgecolor="black")
            ax.plot([idx-0.2, idx+0.2], [s_deltas.mean(), s_deltas.mean()], color="black", lw=3)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_ylabel("Δ Recall vs IID (Positive = Better than IID)")
        ax.set_title(f"Paired ΔRecall vs IID (Same-Seed Pairings)\n{status_tag}")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "iid_paired_recall_deltas.png", dpi=300)
        plt.close()

    # Convergence curves (8, 9, 10)
    # Kumpulkan trajectory dari metrics.csv di setiap folder run
    conv_data: Dict[str, List[pd.DataFrame]] = {s[0]: [] for s in SCENARIOS_ORDER}
    for item in registry.values():
        if item.get("completion_status") not in ("REUSED_VERIFIED", "COMPLETED"):
            continue
        scen = item.get("scenario")
        m_csv = Path(item["run_directory"]) / "metrics.csv"
        if m_csv.exists() and scen in conv_data:
            try:
                m_df = pd.read_csv(m_csv)
                if len(m_df) >= 50:
                    conv_data[scen].append(m_df.head(51))
            except Exception:
                pass

    # 8. convergence_mae_mean_sd.png
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (scen, name) in enumerate(SCENARIOS_ORDER):
        dfs = conv_data.get(scen, [])
        if dfs:
            rounds = dfs[0]["round"].values
            mae_matrix = np.array([df["mae"].values for df in dfs])
            mean_mae = np.mean(mae_matrix, axis=0)
            std_mae = np.std(mae_matrix, axis=0, ddof=1) if len(dfs) > 1 else np.zeros_like(mean_mae)
            ax.plot(rounds, mean_mae, label=name, color=colors[idx], lw=2)
            ax.fill_between(rounds, mean_mae - std_mae, mean_mae + std_mae, color=colors[idx], alpha=0.15)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test MAE")
    ax.set_title(f"Convergence Trajectory: Global Test MAE (Mean ± Sample SD)\n{status_tag}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "convergence_mae_mean_sd.png", dpi=300)
    plt.close()

    # 9. convergence_macro_f1_mean_sd.png
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (scen, name) in enumerate(SCENARIOS_ORDER):
        dfs = conv_data.get(scen, [])
        if dfs:
            rounds = dfs[0]["round"].values
            f1_matrix = np.array([df["macro_f1"].values for df in dfs])
            mean_f1 = np.mean(f1_matrix, axis=0)
            std_f1 = np.std(f1_matrix, axis=0, ddof=1) if len(dfs) > 1 else np.zeros_like(mean_f1)
            ax.plot(rounds, mean_f1, label=name, color=colors[idx], lw=2)
            ax.fill_between(rounds, mean_f1 - std_f1, mean_f1 + std_f1, color=colors[idx], alpha=0.15)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Macro F1")
    ax.set_title(f"Convergence Trajectory: Global Test Macro F1 (Mean ± Sample SD)\n{status_tag}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "convergence_macro_f1_mean_sd.png", dpi=300)
    plt.close()

    # 10. convergence_recall_mean_sd.png
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (scen, name) in enumerate(SCENARIOS_ORDER):
        dfs = conv_data.get(scen, [])
        if dfs:
            rounds = dfs[0]["round"].values
            rec_matrix = np.array([df["recall"].values for df in dfs])
            mean_rec = np.mean(rec_matrix, axis=0)
            std_rec = np.std(rec_matrix, axis=0, ddof=1) if len(dfs) > 1 else np.zeros_like(mean_rec)
            ax.plot(rounds, mean_rec, label=name, color=colors[idx], lw=2)
            ax.fill_between(rounds, mean_rec - std_rec, mean_rec + std_rec, color=colors[idx], alpha=0.15)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Recall")
    ax.set_title(f"Convergence Trajectory: Global Test Recall (Mean ± Sample SD)\n{status_tag}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "convergence_recall_mean_sd.png", dpi=300)
    plt.close()

    # 11. clinical_fn_fp_by_scenario.png
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(scens))
    width = 0.35
    mean_fns = [seed_df[seed_df["scenario"] == s]["fn"].mean() for s in scens]
    mean_fps = [seed_df[seed_df["scenario"] == s]["fp"].mean() for s in scens]
    ax.bar(x - width/2, mean_fns, width, label="False Negatives (Missed Screening-Positive Cases)", color="#d95f02")
    ax.bar(x + width/2, mean_fps, width, label="False Positives (Over-screened Cases)", color="#2b5c8f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Count of Test Participants (N=56)")
    ax.set_title(f"Clinical Screening Errors Across Scenarios (Mean Counts)\n{status_tag}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "clinical_fn_fp_by_scenario.png", dpi=300)
    plt.close()

    # 12. client_drift_comparison.png
    fig, ax = plt.subplots(figsize=(8, 5))
    drift_f = OUTPUT_DIR / "client_drift_summary.csv"
    if drift_f.exists():
        drift_df = pd.read_csv(drift_f)
        ax.bar(drift_df["display_name"], drift_df["avg_raw_l2_norm"], color=colors[:len(drift_df)], edgecolor="black")
        ax.set_ylabel("Mean Raw Client Update L2 Norm (||Δw||)")
        ax.set_title(f"Client Update Magnitude (Drift Measure) Across Scenarios\n{status_tag}")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "client_drift_comparison.png", dpi=300)
        plt.close()

    # 13. communication_comparison.png
    fig, ax = plt.subplots(figsize=(8, 5))
    comm_f = OUTPUT_DIR / "communication_summary.csv"
    if comm_f.exists():
        comm_df = pd.read_csv(comm_f)
        ax.bar(comm_df["display_name"], comm_df["total_50_rounds_gib_per_run"], color="#4daf4a", edgecolor="black")
        ax.set_ylabel("Total Logical Parameter Payload per Run (GiB)")
        ax.set_title(f"Communication Overhead per 50-Round Run (FedAvg Full Participation)\n{status_tag}")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "communication_comparison.png", dpi=300)
        plt.close()

    logger.info("Successfully produced all 13 plots.")


def main():
    reg_f = OUTPUT_DIR / "run_registry.json"
    if not reg_f.exists():
        logger.error(f"Run registry not found: {reg_f}")
        return
        
    with open(reg_f, "r", encoding="utf-8") as f:
        registry = json.load(f)
        
    seed_df = extract_seed_metrics(registry)
    if seed_df.empty:
        logger.error("No valid runs found in registry!")
        return
        
    generate_summary_statistics(seed_df)
    paired_df, _ = compute_paired_comparisons_vs_iid(seed_df)
    generate_clinical_error_summary(seed_df)
    generate_communication_summary(seed_df)
    generate_client_drift_summary(registry)
    generate_13_visualizations(seed_df, paired_df, registry)
    logger.info("Statistical analysis and visualizations completed successfully.")


if __name__ == "__main__":
    main()
