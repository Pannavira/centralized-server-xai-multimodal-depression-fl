"""
federated/partitions/combined_skew.py
======================================
Generator Partisi Controlled Non-IID Combined Skew untuk Milestone 5.5 FED-MIND.
Menggabungkan tiga bentuk heterogenitas data Federated Learning:
1. Quantity Skew: Pembagian ukuran dataset asimetris 98 / 49 / 16 (~60% / 30% / 10%).
2. Label Skew: Distribusi prevalensi risiko depresi heterogen (18.37% / 38.78% / 81.25%).
3. Modality Skew: Kualitas modalitas audio dan visual heterogen per-klien:
   - Client 1: Clean (Audio, Visual, Text)
   - Client 2: Audio Gaussian noise (sigma=0.25)
   - Client 3: Visual temporal frame dropout (20% dropout dengan neutral mean-imputation)
   - Text: Selalu 100% bersih di semua klien
   - Shared Validation & Global Test: Selalu 100% bersih

Invarian Ilmiah:
1. Total partisipan training: TEPAT 163 subjek (100% pool M5.1 dipertahankan, S_IID == S_COMBINED).
2. Nol duplikasi subjek antar klien (disjoint partitions).
3. Zero leakage terhadap Global Hold-Out Test (56) dan Shared Validation Cohort (56).
4. Pelestarian total label global: Tepat 50 positif (PHQ-8 >= 10) dan 113 negatif (PHQ-8 < 10).
5. Deterministik murni dengan seed 42.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from federated.data import DEFAULT_PARTITION_DIR


def get_file_sha256(filepath: Union[str, Path]) -> str:
    """Menghitung checksum SHA-256 berkas secara streaming."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_binary_entropy(prob_pos: float) -> float:
    """Menghitung entropi biner Shannon (basis 2)."""
    if prob_pos <= 0.0 or prob_pos >= 1.0:
        return 0.0
    prob_neg = 1.0 - prob_pos
    return float(-(prob_pos * np.log2(prob_pos) + prob_neg * np.log2(prob_neg)))


def compute_pairwise_js_divergence(prevalences: List[float]) -> float:
    """Menghitung rata-rata Jensen-Shannon divergence berpasangan antar klien."""
    n = len(prevalences)
    if n < 2:
        return 0.0
    js_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            p = np.array([prevalences[i], 1.0 - prevalences[i]])
            q = np.array([prevalences[j], 1.0 - prevalences[j]])
            js_vals.append(float(jensenshannon(p, q, base=2)))
    return float(np.mean(js_vals))


def get_phq8_severity_bin_proportions(df: pd.DataFrame) -> Dict[str, float]:
    """Menghitung proporsi subjek dalam rentang keparahan klinis PHQ-8."""
    total = len(df)
    if total == 0:
        return {
            "none_minimal_0_4": 0.0,
            "mild_5_9": 0.0,
            "moderate_10_14": 0.0,
            "moderately_severe_15_19": 0.0,
            "severe_20_24": 0.0,
        }
    s = df["phq8_score"]
    return {
        "none_minimal_0_4": float((s < 5).sum() / total),
        "mild_5_9": float(((s >= 5) & (s < 10)).sum() / total),
        "moderate_10_14": float(((s >= 10) & (s < 15)).sum() / total),
        "moderately_severe_15_19": float(((s >= 15) & (s < 20)).sum() / total),
        "severe_20_24": float((s >= 20).sum() / total),
    }


def generate_controlled_combined_skew_partitions(
    source_iid_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    global_test_path: Optional[Union[str, Path]] = None,
    shared_val_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
    target_quantities: Tuple[int, int, int] = (98, 49, 16),
    target_positives: Tuple[int, int, int] = (18, 19, 13),
) -> Dict[str, Any]:
    """
    Menghasilkan partisi controlled_combined_skew_v1 secara deterministik dari pool data M5.1.
    """
    iid_base = Path(source_iid_dir or (DEFAULT_PARTITION_DIR / "IID_3Clients"))
    out_base = Path(output_dir or Path("storage/federated/partitions/combined_skew_seed42"))
    plots_dir = out_base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    test_file = Path(global_test_path or (DEFAULT_PARTITION_DIR / "global_test_index.csv"))
    val_file = Path(shared_val_path or (iid_base / "client_1_val.csv"))

    if not iid_base.exists():
        raise FileNotFoundError(f"Direktori partisi IID tidak ditemukan: {iid_base}")
    if not test_file.exists():
        raise FileNotFoundError(f"Berkas global test tidak ditemukan: {test_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Berkas shared validation tidak ditemukan: {val_file}")

    # 1. Muat dan kumpulkan pool 163 subjek resmi dari M5.1
    iid_dfs = []
    source_manifest = {}
    for c in [1, 2, 3]:
        train_path = iid_base / f"client_{c}_train.csv"
        df_c = pd.read_csv(train_path)
        source_manifest[f"client_{c}_train"] = {
            "samples": len(df_c),
            "sha256": get_file_sha256(train_path),
        }
        iid_dfs.append(df_c)

    all_train_df = pd.concat(iid_dfs, ignore_index=True)
    total_samples = len(all_train_df)
    assert total_samples == 163, f"Jumlah subjek training bukan 163, melainkan: {total_samples}"

    # Pastikan rel_paths diformat seragam
    for col in ["audio_feature_path", "visual_feature_path", "text_feature_path"]:
        if col in all_train_df.columns:
            all_train_df[col] = all_train_df[col].astype(str).str.replace("\\", "/", regex=False)

    # 2. Pisahkan strata positif dan negatif secara deterministik
    pos_df = all_train_df[all_train_df["phq8_score"] >= 10.0].sort_values("participant_id").reset_index(drop=True)
    neg_df = all_train_df[all_train_df["phq8_score"] < 10.0].sort_values("participant_id").reset_index(drop=True)

    total_pos = len(pos_df)
    total_neg = len(neg_df)
    assert total_pos == 50, f"Jumlah positif global bukan 50, melainkan {total_pos}"
    assert total_neg == 113, f"Jumlah negatif global bukan 113, melainkan {total_neg}"

    q1, q2, q3 = target_quantities
    p1, p2, p3 = target_positives
    n1, n2, n3 = q1 - p1, q2 - p2, q3 - p3

    assert p1 + p2 + p3 == total_pos, f"Total positif target ({p1+p2+p3}) != total positif global ({total_pos})"
    assert n1 + n2 + n3 == total_neg, f"Total negatif target ({n1+n2+n3}) != total negatif global ({total_neg})"
    assert q1 + q2 + q3 == total_samples, f"Total kuantitas target ({q1+q2+q3}) != total sampel global ({total_samples})"

    # 3. Permutasi deterministik menggunakan seed 42
    rng = np.random.RandomState(seed)
    pos_perm = rng.permutation(total_pos)
    neg_perm = rng.permutation(total_neg)

    c1_pos = pos_df.iloc[pos_perm[:p1]]
    c2_pos = pos_df.iloc[pos_perm[p1 : p1 + p2]]
    c3_pos = pos_df.iloc[pos_perm[p1 + p2 :]]

    c1_neg = neg_df.iloc[neg_perm[:n1]]
    c2_neg = neg_df.iloc[neg_perm[n1 : n1 + n2]]
    c3_neg = neg_df.iloc[neg_perm[n1 + n2 :]]

    client_dfs = [
        pd.concat([c1_pos, c1_neg]).sort_values("participant_id").reset_index(drop=True),
        pd.concat([c2_pos, c2_neg]).sort_values("participant_id").reset_index(drop=True),
        pd.concat([c3_pos, c3_neg]).sort_values("participant_id").reset_index(drop=True),
    ]

    # 4. Validasi Invarian Ilmiah Ketat
    # a. Kuantitas per klien
    assert len(client_dfs[0]) == q1
    assert len(client_dfs[1]) == q2
    assert len(client_dfs[2]) == q3

    # b. Positif & Negatif per klien
    for idx, (df_c, p_target, n_target) in enumerate(zip(client_dfs, [p1, p2, p3], [n1, n2, n3])):
        act_p = int((df_c["phq8_score"] >= 10.0).sum())
        act_n = int((df_c["phq8_score"] < 10.0).sum())
        assert act_p == p_target, f"Client {idx+1} positif mismatch: {act_p} != {p_target}"
        assert act_n == n_target, f"Client {idx+1} negatif mismatch: {act_n} != {n_target}"

    # c. Pool Invariance (S_IID == S_COMBINED)
    all_combined_ids = set()
    for df_c in client_dfs:
        all_combined_ids.update(df_c["participant_id"].astype(str))
    iid_ids = set(all_train_df["participant_id"].astype(str))
    assert all_combined_ids == iid_ids, "S_COMBINED tidak persis sama dengan S_IID!"
    assert len(all_combined_ids) == 163, "Jumlah partisipan unik bukan 163!"

    # d. Zero Leakage terhadap Global Test dan Validation Cohort
    test_df = pd.read_csv(test_file)
    test_ids = set(test_df["participant_id"].astype(str))
    leakage_test = all_combined_ids.intersection(test_ids)
    assert len(leakage_test) == 0, f"DATA LEAKAGE DETECTED DENGAN GLOBAL TEST: {leakage_test}"

    val_df = pd.read_csv(val_file)
    val_ids = set(val_df["participant_id"].astype(str))
    leakage_val = all_combined_ids.intersection(val_ids)
    assert len(leakage_val) == 0, f"DATA LEAKAGE DETECTED DENGAN VALIDATION SET: {leakage_val}"

    # 5. Tulis file partisi CSV ke direktori artefak
    file_manifest: Dict[str, Any] = {}
    client_train_paths: List[Path] = []
    for c_idx, df_c in enumerate(client_dfs, 1):
        target_csv = out_base / f"client_{c_idx}_train.csv"
        # Simpan kolom standar
        cols_to_save = ["participant_id", "phq8_score", "depression_label", "audio_feature_path", "visual_feature_path", "text_feature_path"]
        available_cols = [c for c in cols_to_save if c in df_c.columns]
        df_c[available_cols].to_csv(target_csv, index=False)
        client_train_paths.append(target_csv)
        file_manifest[f"client_{c_idx}_train.csv"] = {
            "samples": len(df_c),
            "positives": int((df_c["phq8_score"] >= 10.0).sum()),
            "negatives": int((df_c["phq8_score"] < 10.0).sum()),
            "sha256": get_file_sha256(target_csv),
        }

    # Salin shared validation cohort untuk keselarasan antarmuka DataLoader
    for c_idx in [1, 2, 3]:
        val_dst = out_base / f"client_{c_idx}_val.csv"
        shutil.copy2(val_file, val_dst)
        file_manifest[f"client_{c_idx}_val.csv"] = {
            "samples": len(val_df),
            "sha256": get_file_sha256(val_dst),
            "note": "Identical shared validation cohort (zero leakage)",
        }

    # 6. Komputasi Distribusi dan Metrik Heterogenitas
    prevalences = [float((df_c["phq8_score"] >= 10.0).sum() / len(df_c)) for df_c in client_dfs]
    global_prevalence = float(total_pos / total_samples)
    js_divergence = compute_pairwise_js_divergence(prevalences)
    max_min_quantity_ratio = float(max(target_quantities) / min(target_quantities))

    modality_treatments = {
        "client_1": {
            "treatment": "clean_full_modalities",
            "audio": "clean",
            "visual": "clean",
            "text": "clean",
            "note": "Baseline reference sensors, zero corruption",
        },
        "client_2": {
            "treatment": "audio_noisy",
            "audio": "gaussian_noise_sigma_0.25",
            "visual": "clean",
            "text": "clean",
            "note": "Gaussian perturbation sigma=0.25 on normalized audio features",
        },
        "client_3": {
            "treatment": "visual_degraded",
            "audio": "clean",
            "visual": "temporal_frame_dropout_0.20_neutral_mean_imputation",
            "text": "clean",
            "note": "20% temporal frame dropout with neutral mean frame imputation",
        },
    }

    clients_data = {}
    csv_rows = []
    for c_idx, (df_c, p_cnt, n_cnt, q_cnt) in enumerate(zip(client_dfs, [p1, p2, p3], [n1, n2, n3], target_quantities), 1):
        prev = float(p_cnt / q_cnt)
        phq_s = df_c["phq8_score"]
        c_key = f"client_{c_idx}"
        treatment_info = modality_treatments[c_key]
        c_info = {
            "client_id": c_idx,
            "file": f"client_{c_idx}_train.csv",
            "samples": q_cnt,
            "proportion_of_total": float(q_cnt / total_samples),
            "fedavg_weight": float(q_cnt / total_samples),
            "fedavg_weight_fraction": f"{q_cnt}/{total_samples}",
            "positives": p_cnt,
            "negatives": n_cnt,
            "positive_prevalence": prev,
            "prevalence_deviation_from_global": float(abs(prev - global_prevalence)),
            "binary_entropy": compute_binary_entropy(prev),
            "phq8_mean": float(phq_s.mean()),
            "phq8_std": float(phq_s.std()),
            "phq8_median": float(phq_s.median()),
            "phq8_min": float(phq_s.min()),
            "phq8_max": float(phq_s.max()),
            "phq8_severity_bins": get_phq8_severity_bin_proportions(df_c),
            "modality_treatment": treatment_info,
            "sha256": file_manifest[f"client_{c_idx}_train.csv"]["sha256"],
        }
        clients_data[c_key] = c_info
        csv_rows.append({
            "client": f"Client {c_idx}",
            "samples": q_cnt,
            "proportion": f"{q_cnt / total_samples:.2%}",
            "fedavg_weight": f"{q_cnt}/163",
            "positives": p_cnt,
            "negatives": n_cnt,
            "positive_prevalence": f"{prev:.2%}",
            "phq8_mean": f"{phq_s.mean():.2f}",
            "phq8_median": f"{phq_s.median():.1f}",
            "phq8_std": f"{phq_s.std():.2f}",
            "binary_entropy": f"{compute_binary_entropy(prev):.4f}",
            "audio_treatment": treatment_info["audio"],
            "visual_treatment": treatment_info["visual"],
            "text_treatment": treatment_info["text"],
        })

    dist_report = {
        "scenario": "controlled_non_iid_combined_skew",
        "protocol_version": "controlled_combined_skew_v1",
        "seed": seed,
        "total_training_participants": total_samples,
        "unique_training_participants": len(all_combined_ids),
        "global_positives": total_pos,
        "global_negatives": total_neg,
        "global_positive_prevalence": global_prevalence,
        "global_phq8_mean": float(all_train_df["phq8_score"].mean()),
        "global_phq8_std": float(all_train_df["phq8_score"].std()),
        "global_phq8_median": float(all_train_df["phq8_score"].median()),
        "global_phq8_severity_bins": get_phq8_severity_bin_proportions(all_train_df),
        "max_min_quantity_ratio": max_min_quantity_ratio,
        "mean_pairwise_js_divergence": js_divergence,
        "prevalence_spread_max_min": float(max(prevalences) - min(prevalences)),
        "clients": clients_data,
        "shared_validation": {
            "samples": len(val_df),
            "sha256": get_file_sha256(val_file),
            "modality": "clean",
        },
        "global_test": {
            "samples": len(test_df),
            "sha256": get_file_sha256(test_file),
            "modality": "clean",
        },
    }

    # Simpan JSON dan CSV distribusi
    with open(out_base / "combined_skew_distribution.json", "w", encoding="utf-8") as f:
        json.dump(dist_report, f, indent=2)
    with open(out_base / "label_quantity_distribution.json", "w", encoding="utf-8") as f:
        json.dump(dist_report, f, indent=2)

    dist_df = pd.DataFrame(csv_rows)
    dist_df.to_csv(out_base / "combined_skew_distribution.csv", index=False)
    dist_df.to_csv(out_base / "label_quantity_distribution.csv", index=False)

    # Simpan partition manifest
    manifest = {
        "protocol": "controlled_combined_skew_v1",
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "target_quantities": list(target_quantities),
        "target_positives": list(target_positives),
        "source_iid_files": source_manifest,
        "generated_files": file_manifest,
        "zero_leakage_assertion": {
            "test_leakage_count": 0,
            "validation_leakage_count": 0,
            "all_training_pool_preserved": True,
            "pool_cardinality": total_samples,
        },
    }
    with open(out_base / "partition_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 7. Generate Plots
    _generate_combined_plots(dist_report, plots_dir)

    return dist_report


def _generate_combined_plots(dist: Dict[str, Any], plots_dir: Path) -> None:
    """Menghasilkan plot visualisasi distribusi gabungan: kuantitas, label, dan perlakuan modalitas."""
    clients = ["Client 1\n(Clean)", "Client 2\n(Audio Noisy)", "Client 3\n(Visual Degraded)"]
    quantities = [dist["clients"][f"client_{i}"]["samples"] for i in [1, 2, 3]]
    positives = [dist["clients"][f"client_{i}"]["positives"] for i in [1, 2, 3]]
    negatives = [dist["clients"][f"client_{i}"]["negatives"] for i in [1, 2, 3]]
    prevalences = [dist["clients"][f"client_{i}"]["positive_prevalence"] * 100 for i in [1, 2, 3]]

    # 1. Quantity Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(clients, quantities, color=["#2b5c8f", "#d95f02", "#7570b3"], width=0.5)
    ax.set_ylabel("Jumlah Partisipan", fontsize=12)
    ax.set_title("FED-MIND Milestone 5.5: Combined Non-IID Quantity Distribution", fontsize=13, fontweight="bold")
    for b, q in zip(bars, quantities):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{q} ({q/163:.1%})", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plots_dir / "quantity_distribution.png", dpi=200)
    plt.close(fig)

    # 2. Label Distribution Plot (Stacked Bar & Prevalence Line)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(clients))
    w = 0.45
    b_neg = ax1.bar(x, negatives, w, label="PHQ-8 < 10 (Negatif)", color="#4575b4")
    b_pos = ax1.bar(x, positives, w, bottom=negatives, label="PHQ-8 ≥ 10 (Positif)", color="#d73027")
    ax1.set_ylabel("Jumlah Sampel", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(clients, fontsize=11)
    ax1.set_title("FED-MIND Milestone 5.5: Combined Label & Severity Distribution", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.set_ylim(0, 115)

    # Anotasi bar
    for i in range(3):
        ax1.text(x[i], negatives[i] / 2, f"{negatives[i]}", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        ax1.text(x[i], negatives[i] + positives[i] / 2, f"{positives[i]}", ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    # Axis kedua untuk prevalensi
    ax2 = ax1.twinx()
    ax2.plot(x, prevalences, color="#000000", marker="o", linewidth=2.5, markersize=8, label="Prevalensi Positif (%)")
    ax2.axhline(30.67, color="gray", linestyle=":", label="Global Prevalensi (30.7%)")
    ax2.set_ylabel("Prevalensi Positif (%)", fontsize=12)
    ax2.set_ylim(0, 100)
    for i, p in enumerate(prevalences):
        ax2.annotate(f"{p:.1f}%", (x[i], p), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold", fontsize=11)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(plots_dir / "label_distribution.png", dpi=200)
    plt.close(fig)

    # 3. Modality Treatment Assignment Plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    modalities = ["Audio", "Visual", "Text"]
    treatment_matrix = np.array([
        [0, 0, 0],  # Client 1: Clean, Clean, Clean
        [1, 0, 0],  # Client 2: Noisy, Clean, Clean
        [0, 2, 0],  # Client 3: Clean, Degraded, Clean
    ])
    # 0 = Clean (Green), 1 = Audio Perturbation (Orange), 2 = Visual Dropout (Purple)
    cmap = matplotlib.colors.ListedColormap(["#2ca02c", "#ff7f0e", "#9467bd"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    cax = ax.imshow(treatment_matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(modalities)))
    ax.set_yticks(np.arange(len(clients)))
    ax.set_xticklabels(modalities, fontsize=12, fontweight="bold")
    ax.set_yticklabels(clients, fontsize=11, fontweight="bold")
    ax.set_title("FED-MIND Milestone 5.5: Modality Quality Treatment Matrix", fontsize=13, fontweight="bold")

    # Anotasi teks di dalam cell
    labels = [
        ["Clean (100%)", "Clean (100%)", "Clean (100%)"],
        ["Gaussian Noise (σ=0.25)", "Clean (100%)", "Clean (100%)"],
        ["Clean (100%)", "20% Frame Dropout\n(Mean Imputation)", "Clean (100%)"],
    ]
    for r in range(3):
        for c in range(3):
            ax.text(c, r, labels[r][c], ha="center", va="center", color="white", fontweight="bold", fontsize=10)

    fig.tight_layout()
    fig.savefig(plots_dir / "modality_assignment.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    rep = generate_controlled_combined_skew_partitions()
    print("Combined Skew Partition Generated Successfully!")
    print(json.dumps(rep, indent=2))
