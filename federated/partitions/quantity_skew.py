"""
federated/partitions/quantity_skew.py
======================================
Generator Partisi Controlled Non-IID Quantity Skew untuk Milestone 5.3 FED-MIND.

Invarian Ilmiah:
1. Total partisipan training: TEPAT 163 subjek (100% pool M5.1 dipertahankan, S_IID == S_QUANTITY).
2. Rasio target: 60% / 30% / 10% -> Alokasi integer terdekat:
   - Client 1: 98 partisipan (~60.12%)
   - Client 2: 49 partisipan (~30.06%)
   - Client 3: 16 partisipan (~9.82%)
   - Total: 163 partisipan
3. Nol duplikasi subjek antar klien (disjoint partitions).
4. Zero leakage terhadap Global Hold-Out Test (56) dan Shared Validation Cohort (56).
5. Pelestarian distribusi label depresi: Positif (PHQ-8 >= 10) dan Negatif (PHQ-8 < 10)
   didistribusikan secara berimbang sesuai prevalensi global (~30.67%):
   - Client 1: 30 pos, 68 neg (30.61%)
   - Client 2: 15 pos, 34 neg (30.61%)
   - Client 3:  5 pos, 11 neg (31.25%)
   - Total:    50 pos, 113 neg (30.67%)
6. Deterministik murni dengan seed 42.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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


def generate_controlled_quantity_skew_partitions(
    source_iid_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    global_test_path: Optional[Union[str, Path]] = None,
    shared_val_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
    target_quantities: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Any]:
    """
    Menghasilkan partisi controlled_quantity_skew secara deterministik dari pool data M5.1.
    """
    if target_quantities is None:
        target_quantities = (98, 49, 16)
    q1, q2, q3 = target_quantities
    total_target = q1 + q2 + q3
    assert total_target == 163, f"Total target harus 163, diberikan {total_target}"

    iid_base = Path(source_iid_dir or (DEFAULT_PARTITION_DIR / "IID_3Clients"))
    out_base = Path(output_dir or Path("storage/federated/partitions/quantity_skew_seed42"))
    out_base.mkdir(parents=True, exist_ok=True)

    test_file = Path(global_test_path or (DEFAULT_PARTITION_DIR / "global_test_index.csv"))
    val_file = Path(shared_val_path or (iid_base / "client_1_val.csv"))

    if not iid_base.exists():
        raise FileNotFoundError(f"Direktori partisi IID tidak ditemukan: {iid_base}")
    if not test_file.exists():
        raise FileNotFoundError(f"Berkas global test tidak ditemukan: {test_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Berkas shared validation tidak ditemukan: {val_file}")

    # 1. Muat data training IID M5.1
    iid_dfs = []
    source_iid_manifest: Dict[str, Dict[str, Any]] = {}
    for c in [1, 2, 3]:
        train_path = iid_base / f"client_{c}_train.csv"
        df_c = pd.read_csv(train_path)
        df_c["orig_iid_client"] = c
        iid_dfs.append(df_c)
        source_iid_manifest[f"client_{c}"] = {
            "path": str(train_path),
            "sha256": get_file_sha256(train_path),
            "samples": len(df_c),
            "positives": int((df_c["phq8_score"] >= 10.0).sum()),
            "negatives": int((df_c["phq8_score"] < 10.0).sum()),
        }

    all_train = pd.concat(iid_dfs, ignore_index=True)
    all_train_pids = set(all_train["participant_id"].astype(str))
    total_samples = len(all_train)
    assert total_samples == 163, f"Total sampel IID harus 163, ditemukan {total_samples}"

    test_df = pd.read_csv(test_file)
    test_pids = set(test_df["participant_id"].astype(str))
    test_sha = get_file_sha256(test_file)

    val_df = pd.read_csv(val_file)
    val_pids = set(val_df["participant_id"].astype(str))
    val_sha = get_file_sha256(val_file)

    # 2. Pisahkan strata positif dan negatif
    all_train["pid_num"] = all_train["participant_id"].astype(int)
    all_train_sorted = all_train.sort_values("pid_num").reset_index(drop=True)

    pos_df = all_train_sorted[all_train_sorted["phq8_score"] >= 10.0].copy().reset_index(drop=True)
    neg_df = all_train_sorted[all_train_sorted["phq8_score"] < 10.0].copy().reset_index(drop=True)

    assert len(pos_df) == 50, f"Expected 50 positives, found {len(pos_df)}"
    assert len(neg_df) == 113, f"Expected 113 negatives, found {len(neg_df)}"

    # Alokasi integer optimal label positif dan negatif untuk kapasitas (98, 49, 16):
    # C1 (98): 30 pos, 68 neg -> 30/98 = 30.61%
    # C2 (49): 15 pos, 34 neg -> 15/49 = 30.61%
    # C3 (16):  5 pos, 11 neg ->  5/16 = 31.25%
    p1, p2, p3 = 30, 15, 5
    n1, n2, n3 = 68, 34, 11
    assert p1 + p2 + p3 == 50
    assert n1 + n2 + n3 == 113
    assert p1 + n1 == q1
    assert p2 + n2 == q2
    assert p3 + n3 == q3

    # 3. Permutasi deterministik terkontrol seed 42
    rng = np.random.RandomState(seed)
    pos_perm = rng.permutation(len(pos_df))
    neg_perm = rng.permutation(len(neg_df))

    pos_shuffled = pos_df.iloc[pos_perm].reset_index(drop=True)
    neg_shuffled = neg_df.iloc[neg_perm].reset_index(drop=True)

    c1_pos = pos_shuffled.iloc[:p1]
    c2_pos = pos_shuffled.iloc[p1 : p1 + p2]
    c3_pos = pos_shuffled.iloc[p1 + p2 : p1 + p2 + p3]

    c1_neg = neg_shuffled.iloc[:n1]
    c2_neg = neg_shuffled.iloc[n1 : n1 + n2]
    c3_neg = neg_shuffled.iloc[n1 + n2 : n1 + n2 + n3]

    canonical_cols = [
        "participant_id",
        "split",
        "depression_label",
        "text_feature_path",
        "audio_feature_path",
        "visual_feature_path",
        "phq8_score",
    ]

    client_dfs: Dict[str, pd.DataFrame] = {}
    for c_id, pos_slice, neg_slice in [
        ("client_1", c1_pos, c1_neg),
        ("client_2", c2_pos, c2_neg),
        ("client_3", c3_pos, c3_neg),
    ]:
        combined = pd.concat([pos_slice, neg_slice], ignore_index=True)
        # Urutkan secara kanonikal berdasarkan participant_id numerik
        combined = combined.sort_values("pid_num").reset_index(drop=True)
        # Pastikan kolom sesuai struktur resmi M5.1
        clean_df = combined[canonical_cols].copy()
        clean_df["participant_id"] = clean_df["participant_id"].astype(int)
        clean_df["depression_label"] = clean_df["depression_label"].astype(int)
        clean_df["phq8_score"] = clean_df["phq8_score"].astype(float)
        client_dfs[c_id] = clean_df

    # 4. Validasi Invarian Ilmiah Mutlak
    all_qs_pids = set()
    for c_id, df_c in client_dfs.items():
        c_pids = set(df_c["participant_id"].astype(str))
        # Pastikan tidak ada duplikasi internal
        assert len(c_pids) == len(df_c), f"Duplikasi terdeteksi pada {c_id}"
        # Pastikan disjoint dengan klien lain
        overlap_with_prev = all_qs_pids.intersection(c_pids)
        assert len(overlap_with_prev) == 0, f"Overlap terdeteksi antar klien: {overlap_with_prev}"
        all_qs_pids.update(c_pids)

    # Invarian 1: S_IID == S_QUANTITY
    assert all_qs_pids == all_train_pids, "INVARIANT VIOLATION: S_QUANTITY != S_IID"
    assert len(all_qs_pids) == 163, f"Total subjek training harus 163, ditemukan {len(all_qs_pids)}"

    # Invarian 2: Zero Leakage
    leak_test = all_qs_pids.intersection(test_pids)
    assert len(leak_test) == 0, f"DATA LEAKAGE TEST: {leak_test}"

    leak_val = all_qs_pids.intersection(val_pids)
    assert len(leak_val) == 0, f"DATA LEAKAGE VAL: {leak_val}"

    # Invarian 3: Jumlah sampel per klien
    assert len(client_dfs["client_1"]) == q1, f"C1 size {len(client_dfs['client_1'])} != {q1}"
    assert len(client_dfs["client_2"]) == q2, f"C2 size {len(client_dfs['client_2'])} != {q2}"
    assert len(client_dfs["client_3"]) == q3, f"C3 size {len(client_dfs['client_3'])} != {q3}"

    # 5. Tulis berkas CSV kanonikal
    client_manifest: Dict[str, Dict[str, Any]] = {}
    distribution_rows = []

    global_pos = 50
    global_neg = 113
    global_prev = global_pos / 163.0

    for c_id in ["client_1", "client_2", "client_3"]:
        csv_path = out_base / f"{c_id}_train.csv"
        df_c = client_dfs[c_id]
        df_c.to_csv(csv_path, index=False)

        sha = get_file_sha256(csv_path)
        n_pos = int((df_c["phq8_score"] >= 10.0).sum())
        n_neg = int((df_c["phq8_score"] < 10.0).sum())
        n_total = len(df_c)
        prev = float(n_pos / n_total)
        entropy = compute_binary_entropy(prev)
        dev = float(abs(prev - global_prev))

        sev_bins = get_phq8_severity_bin_proportions(df_c)

        client_manifest[c_id] = {
            "file": str(csv_path.name),
            "full_path": str(csv_path.resolve()),
            "sha256": sha,
            "samples": n_total,
            "proportion_of_total": float(n_total / 163.0),
            "fedavg_weight": float(n_total / 163.0),
            "fedavg_weight_fraction": f"{n_total}/163",
            "positives": n_pos,
            "negatives": n_neg,
            "positive_prevalence": prev,
            "prevalence_deviation_from_global": dev,
            "binary_entropy": entropy,
            "phq8_mean": float(df_c["phq8_score"].mean()),
            "phq8_std": float(df_c["phq8_score"].std()),
            "phq8_median": float(df_c["phq8_score"].median()),
            "phq8_min": float(df_c["phq8_score"].min()),
            "phq8_max": float(df_c["phq8_score"].max()),
            "phq8_severity_bins": sev_bins,
            "target_allocation": f"{n_pos} pos, {n_neg} neg",
        }

        distribution_rows.append({
            "client_id": c_id,
            "total_samples": n_total,
            "proportion_pct": round((n_total / 163.0) * 100, 2),
            "fedavg_weight": f"{n_total}/163",
            "positives": n_pos,
            "negatives": n_neg,
            "positive_prevalence_pct": round(prev * 100, 2),
            "prevalence_deviation_pct": round(dev * 100, 4),
            "phq8_mean": round(float(df_c["phq8_score"].mean()), 2),
            "phq8_std": round(float(df_c["phq8_score"].std()), 2),
            "phq8_median": round(float(df_c["phq8_score"].median()), 1),
            "sha256": sha,
        })

    # Simpan juga symlink/copy untuk client val ke shared validation cohort
    for c in [1, 2, 3]:
        c_val_path = out_base / f"client_{c}_val.csv"
        if not c_val_path.exists():
            val_df.to_csv(c_val_path, index=False)

    # 6. Hitung Jensen-Shannon divergence berpasangan
    prevalences = [client_manifest[c]["positive_prevalence"] for c in ["client_1", "client_2", "client_3"]]
    mean_js_div = compute_pairwise_js_divergence(prevalences)

    global_sev = get_phq8_severity_bin_proportions(all_train)

    # 7. Simpan manifest JSON dan CSV
    dist_df = pd.DataFrame(distribution_rows)
    dist_csv_path = out_base / "quantity_distribution.csv"
    dist_df.to_csv(dist_csv_path, index=False)

    dist_json_path = out_base / "quantity_distribution.json"
    dist_summary = {
        "scenario": "controlled_non_iid_quantity_skew",
        "protocol": "60_30_10",
        "seed": seed,
        "total_training_participants": 163,
        "global_positives": global_pos,
        "global_negatives": global_neg,
        "global_positive_prevalence": global_prev,
        "global_phq8_mean": float(all_train["phq8_score"].mean()),
        "global_phq8_std": float(all_train["phq8_score"].std()),
        "global_phq8_median": float(all_train["phq8_score"].median()),
        "global_phq8_severity_bins": global_sev,
        "max_min_quantity_ratio": float(q1 / q3),
        "mean_pairwise_js_divergence": mean_js_div,
        "clients": client_manifest,
    }
    with open(dist_json_path, "w", encoding="utf-8") as f:
        json.dump(dist_summary, f, indent=2)

    partition_manifest = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": "non_iid_quantity_skew",
        "scientific_status": "controlled_experiment_quantity_skew",
        "generator": "federated.partitions.quantity_skew",
        "seed": seed,
        "source_iid_control_run": "FL-RUN-20261016-130403-45D587",
        "source_iid_manifest": source_iid_manifest,
        "invariants": {
            "S_IID_equals_S_QUANTITY": True,
            "total_participants": 163,
            "client_quantities": [q1, q2, q3],
            "client_proportions": [float(q1 / 163), float(q2 / 163), float(q3 / 163)],
            "fedavg_weights": [f"{q1}/163", f"{q2}/163", f"{q3}/163"],
            "disjoint_clients": True,
            "zero_leakage_global_test": True,
            "zero_leakage_shared_validation": True,
            "global_positive_count": global_pos,
            "global_negative_count": global_neg,
        },
        "global_test": {
            "file": str(test_file.name),
            "sha256": test_sha,
            "samples": len(test_df),
        },
        "shared_validation": {
            "file": str(val_file.name),
            "sha256": val_sha,
            "samples": len(val_df),
        },
        "client_partitions": client_manifest,
        "divergence_metrics": {
            "mean_pairwise_js_divergence": mean_js_div,
            "max_prevalence_deviation": max(c["prevalence_deviation_from_global"] for c in client_manifest.values()),
        },
    }

    manifest_path = out_base / "partition_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(partition_manifest, f, indent=2)

    # 8. Buat visualisasi
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_plots_dir = out_base / "plots"
    out_plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = create_quantity_skew_plots(
        dist_df=dist_df,
        all_train=all_train,
        client_dfs=client_dfs,
        plots_dirs=[plots_dir, out_plots_dir],
    )

    partition_manifest["plots"] = {k: str(v) for k, v in plot_paths.items()}

    # Update manifest with plot info
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(partition_manifest, f, indent=2)

    return partition_manifest


def create_quantity_skew_plots(
    dist_df: pd.DataFrame,
    all_train: pd.DataFrame,
    client_dfs: Dict[str, pd.DataFrame],
    plots_dirs: List[Path],
) -> Dict[str, str]:
    """
    Membuat dua visualisasi diagnostik resmi:
    1. quantity_distribution_iid_vs_skew.png: Membandingkan alokasi kuantitas IID (54/55/54) vs Skew (98/49/16).
    2. label_balance_quantity_skew.png: Menampilkan keseimbangan prevalensi label dan skor PHQ-8.
    """
    plot_files: Dict[str, str] = {}

    # Plot 1: IID vs Quantity Skew Distribution
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    clients = ["Client 1", "Client 2", "Client 3"]
    iid_counts = [54, 55, 54]
    skew_counts = [int(dist_df.loc[dist_df["client_id"] == f"client_{i+1}", "total_samples"].values[0]) for i in range(3)]
    skew_weights = ["98/163 (60.12%)", "49/163 (30.06%)", "16/163 (9.82%)"]

    x = np.arange(len(clients))
    width = 0.35

    rects1 = ax.bar(x - width/2, iid_counts, width, label="M5.1 FL-IID Control (~33% each)", color="#3b82f6", alpha=0.85, edgecolor="#1d4ed8")
    rects2 = ax.bar(x + width/2, skew_counts, width, label="M5.3 Quantity Skew (60:30:10)", color="#f59e0b", alpha=0.85, edgecolor="#b45309")

    ax.set_ylabel("Number of Training Participants", fontsize=11, fontweight="bold")
    ax.set_title("FED-MIND M5.3: Training Sample Allocation (IID vs. Quantity Skew)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(clients, fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Bar labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{int(h)}\n(33.1%)",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    for idx, rect in enumerate(rects2):
        h = rect.get_height()
        ax.annotate(f"{int(h)}\n{skew_weights[idx]}",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color="#b45309")

    ax.set_ylim(0, 115)
    plt.tight_layout()

    for p_dir in plots_dirs:
        p1 = p_dir / "quantity_distribution_iid_vs_skew.png"
        fig.savefig(p1)
        plot_files["quantity_distribution_iid_vs_skew"] = str(p1)
    plt.close(fig)

    # Plot 2: Label Balance and PHQ-8 Diagnostics under Quantity Skew
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Left: Positive vs Negative proportion per client
    cohorts = ["Global (N=163)", "Client 1 (N=98)", "Client 2 (N=49)", "Client 3 (N=16)"]
    pos_rates = [
        (50 / 163) * 100,
        (30 / 98) * 100,
        (15 / 49) * 100,
        (5 / 16) * 100,
    ]
    neg_rates = [100 - p for p in pos_rates]

    y_pos = np.arange(len(cohorts))
    h_bar = 0.45

    ax1.barh(y_pos, pos_rates, height=h_bar, label="Positive (PHQ-8 >= 10)", color="#ef4444", alpha=0.85)
    ax1.barh(y_pos, neg_rates, height=h_bar, left=pos_rates, label="Negative (PHQ-8 < 10)", color="#10b981", alpha=0.85)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(cohorts, fontsize=10, fontweight="bold")
    ax1.set_xlabel("Prevalence Percentage (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Depression Label Prevalence Balance\n(Verified ~30.6% across all clients)", fontsize=12, fontweight="bold")
    ax1.axvline(30.67, color="#b91c1c", linestyle="--", linewidth=1.5, label="Global Pos Rate (30.67%)")
    ax1.legend(loc="lower right", fontsize=9, frameon=True)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    for i, val in enumerate(pos_rates):
        ax1.text(val / 2, i, f"{val:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9)
        ax1.text(val + (neg_rates[i] / 2), i, f"{neg_rates[i]:.1f}%", va="center", ha="center", color="white", fontweight="bold", fontsize=9)

    # Right: PHQ-8 Score Boxplots
    phq8_data = [
        all_train["phq8_score"].values,
        client_dfs["client_1"]["phq8_score"].values,
        client_dfs["client_2"]["phq8_score"].values,
        client_dfs["client_3"]["phq8_score"].values,
    ]
    bp = ax2.boxplot(phq8_data, tick_labels=["Global", "Client 1", "Client 2", "Client 3"], patch_artist=True)
    colors = ["#94a3b8", "#38bdf8", "#fbbf24", "#f87171"]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax2.axhline(10.0, color="#dc2626", linestyle=":", linewidth=1.5, label="Depression Threshold (PHQ-8=10)")
    ax2.set_ylabel("PHQ-8 Clinical Score", fontsize=11, fontweight="bold")
    ax2.set_title("Continuous PHQ-8 Score Distribution\n(Mean, Median, & Quartiles)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9, frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    for p_dir in plots_dirs:
        p2 = p_dir / "label_balance_quantity_skew.png"
        fig.savefig(p2)
        plot_files["label_balance_quantity_skew"] = str(p2)
    plt.close(fig)

    return plot_files


if __name__ == "__main__":
    print("Generating FED-MIND Controlled Non-IID Quantity Skew Partitions (Milestone 5.3)...")
    manifest = generate_controlled_quantity_skew_partitions()
    print("Partition generation successfully completed!")
    print(f"Manifest written to: storage/federated/partitions/quantity_skew_seed42/partition_manifest.json")
    for cid, cinfo in manifest["client_partitions"].items():
        print(f"  {cid}: {cinfo['samples']} samples ({cinfo['fedavg_weight_fraction']}), "
              f"pos={cinfo['positives']} ({cinfo['positive_prevalence']*100:.2f}%), "
              f"PHQ-8 mean={cinfo['phq8_mean']:.2f}, sha256={cinfo['sha256'][:16]}...")
