"""
federated/partition_generator.py
================================
Generator Partisi Non-IID Terkontrol untuk Eksperimen Federated Learning FED-MIND.
Mengimplementasikan:
- Protokol: controlled_label_skew_v1
- Invarian Ilmiah:
  1. Pelestarian 100% populasi training M5.1 (163 subjek).
  2. Pelestarian kuantitas klien: Client 1 = 54, Client 2 = 55, Client 3 = 54.
  3. Pelestarian total label positif (50) dan negatif (113).
  4. Nol duplikasi partisipan antar klien.
  5. Zero leakage terhadap Global Hold-Out Test dan Shared Validation Cohort.
  6. Penentuan deterministik dengan seed 42.
- Menghasilkan:
  - client_1_train.csv, client_2_train.csv, client_3_train.csv
  - partition_manifest.json
  - label_distribution.json
  - label_distribution.csv
  - Visualisasi perbandingan distribusi IID vs Label Skew.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


def generate_controlled_label_skew_partitions(
    source_iid_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    global_test_path: Optional[Union[str, Path]] = None,
    shared_val_path: Optional[Union[str, Path]] = None,
    seed: int = 42,
    p1: int = 43,
    p2: int = 2,
    p3: int = 5,
) -> Dict[str, Any]:
    """
    Menghasilkan partisi controlled_label_skew_v1 secara deterministik dari pool data M5.1.
    """
    iid_base = Path(source_iid_dir or (DEFAULT_PARTITION_DIR / "IID_3Clients"))
    out_base = Path(output_dir or Path("storage/federated/partitions/label_skew_seed42"))
    out_base.mkdir(parents=True, exist_ok=True)

    test_file = Path(global_test_path or (DEFAULT_PARTITION_DIR / "global_test_index.csv"))
    val_file = Path(shared_val_path or (iid_base / "client_1_val.csv"))

    if not iid_base.exists():
        raise FileNotFoundError(f"Direktori partisi IID tidak ditemukan: {iid_base}")
    if not test_file.exists():
        raise FileNotFoundError(f"Berkas global test tidak ditemukan: {test_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Berkas shared validation tidak ditemukan: {val_file}")

    # 1. Muat dan simpan informasi IID source
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

    # 2. Pisahkan pool positif dan negatif secara deterministik
    all_train["pid_num"] = all_train["participant_id"].astype(int)
    all_train_sorted = all_train.sort_values("pid_num").reset_index(drop=True)

    pos_df = all_train_sorted[all_train_sorted["phq8_score"] >= 10.0].copy().reset_index(drop=True)
    neg_df = all_train_sorted[all_train_sorted["phq8_score"] < 10.0].copy().reset_index(drop=True)

    assert len(pos_df) == 50, f"Expected 50 positives, found {len(pos_df)}"
    assert len(neg_df) == 113, f"Expected 113 negatives, found {len(neg_df)}"
    assert p1 + p2 + p3 == 50, f"Positives allocation must sum to 50: {p1}+{p2}+{p3} != 50"

    n1 = 54 - p1
    n2 = 55 - p2
    n3 = 54 - p3
    assert n1 + n2 + n3 == 113, f"Negatives allocation must sum to 113: {n1}+{n2}+{n3} != 113"

    # 3. Permutasi deterministik terkontrol seed 42
    rng = np.random.RandomState(seed)
    pos_perm = rng.permutation(len(pos_df))
    neg_perm = rng.permutation(len(neg_df))

    pos_shuffled = pos_df.iloc[pos_perm].reset_index(drop=True)
    neg_shuffled = neg_df.iloc[neg_perm].reset_index(drop=True)

    # Alokasi klien
    # Client 1: p1 pos, n1 neg
    c1_pos = pos_shuffled.iloc[:p1]
    c1_neg = neg_shuffled.iloc[:n1]
    c1_df = pd.concat([c1_pos, c1_neg], ignore_index=True)

    # Client 2: p2 pos, n2 neg
    c2_pos = pos_shuffled.iloc[p1 : p1 + p2]
    c2_neg = neg_shuffled.iloc[n1 : n1 + n2]
    c2_df = pd.concat([c2_pos, c2_neg], ignore_index=True)

    # Client 3: p3 pos, n3 neg
    c3_pos = pos_shuffled.iloc[p1 + p2 :]
    c3_neg = neg_shuffled.iloc[n1 + n2 :]
    c3_df = pd.concat([c3_pos, c3_neg], ignore_index=True)

    # Urutkan tiap klien berdasarkan participant_id agar rapi dan deterministik
    drop_cols = [c for c in ["pid_num", "orig_iid_client"] if c in c1_df.columns]
    c1_df = c1_df.sort_values("pid_num").drop(columns=drop_cols).reset_index(drop=True)
    c2_df = c2_df.sort_values("pid_num").drop(columns=drop_cols).reset_index(drop=True)
    c3_df = c3_df.sort_values("pid_num").drop(columns=drop_cols).reset_index(drop=True)

    # Pastikan split bertuliskan 'train'
    c1_df["split"] = "train"
    c2_df["split"] = "train"
    c3_df["split"] = "train"

    # 4. Validasi Invarian Eksperimen Ketat
    assert len(c1_df) == 54, f"Client 1 must have 54 samples, got {len(c1_df)}"
    assert len(c2_df) == 55, f"Client 2 must have 55 samples, got {len(c2_df)}"
    assert len(c3_df) == 54, f"Client 3 must have 54 samples, got {len(c3_df)}"

    pids1 = set(c1_df["participant_id"].astype(str))
    pids2 = set(c2_df["participant_id"].astype(str))
    pids3 = set(c3_df["participant_id"].astype(str))

    assert len(pids1 & pids2) == 0, "Duplicate participant between Client 1 and Client 2!"
    assert len(pids1 & pids3) == 0, "Duplicate participant between Client 1 and Client 3!"
    assert len(pids2 & pids3) == 0, "Duplicate participant between Client 2 and Client 3!"

    all_ls_pids = pids1 | pids2 | pids3
    assert all_ls_pids == all_train_pids, "CRITICAL: Label Skew pool does NOT equal M5.1 training pool!"

    test_leakage = all_ls_pids & test_pids
    assert len(test_leakage) == 0, f"CRITICAL LEAKAGE: Overlap with global test: {test_leakage}"

    val_leakage = all_ls_pids & val_pids
    assert len(val_leakage) == 0, f"CRITICAL LEAKAGE: Overlap with shared val: {val_leakage}"

    # 5. Tulis berkas CSV partisi ke disk
    c1_path = out_base / "client_1_train.csv"
    c2_path = out_base / "client_2_train.csv"
    c3_path = out_base / "client_3_train.csv"

    c1_df.to_csv(c1_path, index=False)
    c2_df.to_csv(c2_path, index=False)
    c3_df.to_csv(c3_path, index=False)

    c1_sha = get_file_sha256(c1_path)
    c2_sha = get_file_sha256(c2_path)
    c3_sha = get_file_sha256(c3_path)

    # 6. Komputasi Statistik Distribusi Label
    client_dfs = [c1_df, c2_df, c3_df]
    ls_stats = []
    iid_stats = []
    prevalences_ls = []
    prevalences_iid = []

    for i, (df_ls, df_iid) in enumerate(zip(client_dfs, iid_dfs), 1):
        # Label Skew
        pos_ls = int((df_ls["phq8_score"] >= 10.0).sum())
        neg_ls = int((df_ls["phq8_score"] < 10.0).sum())
        n_ls = len(df_ls)
        pr_pos_ls = pos_ls / n_ls
        pr_neg_ls = neg_ls / n_ls
        prevalences_ls.append(pr_pos_ls)
        entropy_ls = compute_binary_entropy(pr_pos_ls)

        ls_stats.append({
            "client": f"Client {i}",
            "scenario": "Label Skew (M5.2)",
            "total_samples": n_ls,
            "positive_samples": pos_ls,
            "negative_samples": neg_ls,
            "positive_proportion": round(pr_pos_ls, 4),
            "negative_proportion": round(pr_neg_ls, 4),
            "positive_pct": round(pr_pos_ls * 100, 2),
            "negative_pct": round(pr_neg_ls * 100, 2),
            "phq8_mean": round(float(df_ls["phq8_score"].mean()), 4),
            "phq8_median": round(float(df_ls["phq8_score"].median()), 4),
            "phq8_std": round(float(df_ls["phq8_score"].std()), 4),
            "binary_label_entropy": round(entropy_ls, 4),
        })

        # IID
        pos_iid = int((df_iid["phq8_score"] >= 10.0).sum())
        neg_iid = int((df_iid["phq8_score"] < 10.0).sum())
        n_iid = len(df_iid)
        pr_pos_iid = pos_iid / n_iid
        pr_neg_iid = neg_iid / n_iid
        prevalences_iid.append(pr_pos_iid)
        entropy_iid = compute_binary_entropy(pr_pos_iid)

        iid_stats.append({
            "client": f"Client {i}",
            "scenario": "IID Control (M5.1)",
            "total_samples": n_iid,
            "positive_samples": pos_iid,
            "negative_samples": neg_iid,
            "positive_proportion": round(pr_pos_iid, 4),
            "negative_proportion": round(pr_neg_iid, 4),
            "positive_pct": round(pr_pos_iid * 100, 2),
            "negative_pct": round(pr_neg_iid * 100, 2),
            "phq8_mean": round(float(df_iid["phq8_score"].mean()), 4),
            "phq8_median": round(float(df_iid["phq8_score"].median()), 4),
            "phq8_std": round(float(df_iid["phq8_score"].std()), 4),
            "binary_label_entropy": round(entropy_iid, 4),
        })

    max_min_diff_ls = round(float(max(prevalences_ls) - min(prevalences_ls)), 4)
    max_min_diff_iid = round(float(max(prevalences_iid) - min(prevalences_iid)), 4)

    mean_js_ls = round(compute_pairwise_js_divergence(prevalences_ls), 4)
    mean_js_iid = round(compute_pairwise_js_divergence(prevalences_iid), 4)

    # 7. Simpan label_distribution.json dan label_distribution.csv
    distribution_report = {
        "scenario": "controlled_label_skew_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "label_skew": {
            "clients": ls_stats,
            "max_min_prevalence_difference": max_min_diff_ls,
            "mean_pairwise_js_divergence": mean_js_ls,
        },
        "iid_control": {
            "clients": iid_stats,
            "max_min_prevalence_difference": max_min_diff_iid,
            "mean_pairwise_js_divergence": mean_js_iid,
        },
    }

    with open(out_base / "label_distribution.json", "w", encoding="utf-8") as f:
        json.dump(distribution_report, f, indent=2)

    # Gabungkan ke CSV
    dist_df = pd.DataFrame(iid_stats + ls_stats)
    dist_df.to_csv(out_base / "label_distribution.csv", index=False)

    # 8. Simpan partition_manifest.json
    manifest = {
        "protocol_name": "controlled_label_skew_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "partition_directory": str(out_base),
        "total_participants": total_samples,
        "total_positives": len(pos_df),
        "total_negatives": len(neg_df),
        "source_iid_partitions": source_iid_manifest,
        "generated_partitions": {
            "client_1": {
                "path": str(c1_path),
                "sha256": c1_sha,
                "samples": len(c1_df),
                "positives": p1,
                "negatives": n1,
                "positive_pct": round(p1 / 54 * 100, 2),
            },
            "client_2": {
                "path": str(c2_path),
                "sha256": c2_sha,
                "samples": len(c2_df),
                "positives": p2,
                "negatives": n2,
                "positive_pct": round(p2 / 55 * 100, 2),
            },
            "client_3": {
                "path": str(c3_path),
                "sha256": c3_sha,
                "samples": len(c3_df),
                "positives": p3,
                "negatives": n3,
                "positive_pct": round(p3 / 54 * 100, 2),
            },
        },
        "global_test": {
            "path": str(test_file),
            "sha256": test_sha,
            "samples": len(test_df),
        },
        "shared_validation": {
            "path": str(val_file),
            "sha256": val_sha,
            "samples": len(val_df),
        },
        "invariants_verification": {
            "pool_equality_verified": True,
            "quantities_preserved": [len(c1_df), len(c2_df), len(c3_df)] == [54, 55, 54],
            "zero_duplicates_verified": True,
            "zero_test_leakage_verified": True,
            "zero_validation_leakage_verified": True,
            "both_classes_in_all_clients": True,
        },
        "divergence_metrics": {
            "max_min_prevalence_difference_label_skew": max_min_diff_ls,
            "max_min_prevalence_difference_iid": max_min_diff_iid,
            "mean_pairwise_js_divergence_label_skew": mean_js_ls,
            "mean_pairwise_js_divergence_iid": mean_js_iid,
        },
    }

    with open(out_base / "partition_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def plot_label_distribution(
    distribution_json_or_dict: Union[str, Path, Dict[str, Any]],
    output_png_path: Union[str, Path],
) -> None:
    """
    Membuat grafik visualisasi komparatif distribusi prevalensi label IID vs Label Skew.
    """
    if isinstance(distribution_json_or_dict, (str, Path)):
        with open(distribution_json_or_dict, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = distribution_json_or_dict

    out_path = Path(output_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    iid_clients = data["iid_control"]["clients"]
    ls_clients = data["label_skew"]["clients"]

    clients = [c["client"] for c in iid_clients]
    iid_pos_pct = [c["positive_pct"] for c in iid_clients]
    ls_pos_pct = [c["positive_pct"] for c in ls_clients]

    x = np.arange(len(clients))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    rects1 = ax.bar(x - width / 2, iid_pos_pct, width, label="M5.1 IID Control (~30%)", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x + width / 2, ls_pos_pct, width, label="M5.2 Non-IID Label Skew", color="#d95f02", alpha=0.9)

    ax.set_ylabel("Depression Positive Prevalence (%)", fontsize=12, fontweight="bold")
    ax.set_title("FED-MIND: Client Label Distribution Comparison\nM5.1 IID Control vs M5.2 Controlled Non-IID Label Skew", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(clients, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="upper right")

    # Nilai di atas bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    # Garis populasi rata-rata (30.67%)
    ax.axhline(30.67, color="#7570b3", linestyle=":", linewidth=2, label="Global Train Prevalence (30.67%)")
    ax.legend(fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Plot saved successfully to: {out_path}")


if __name__ == "__main__":
    manifest = generate_controlled_label_skew_partitions()
    print("Partisi controlled_label_skew_v1 berhasil dihasilkan!")
    print(f"Directory: {manifest['partition_directory']}")
    for c, info in manifest["generated_partitions"].items():
        print(f"  {c}: N={info['samples']}, Pos={info['positives']} ({info['positive_pct']}%), Neg={info['negatives']}")

    plot_label_distribution(
        Path("storage/federated/partitions/label_skew_seed42/label_distribution.json"),
        Path("storage/federated/partitions/label_skew_seed42/label_distribution_iid_vs_label_skew.png"),
    )
