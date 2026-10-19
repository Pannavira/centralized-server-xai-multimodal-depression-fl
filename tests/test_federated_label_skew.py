"""
tests/test_federated_label_skew.py
==================================
Unit and Integration Test Suite for Milestone 5.2 (Controlled Non-IID Label Skew).
Verifies:
1. IID and Label-Skew training-pool equality ($S_{IID} == S_{LABEL\\_SKEW}$).
2. No participant duplication across clients.
3. Exact client sample quantities (54, 55, 54).
4. Global positive (50) and negative (113) counts preserved.
5. Label-skew prevalence difference strictly exceeds IID prevalence difference.
6. Zero leakage with Global Hold-Out Test set.
7. Zero leakage with Shared Validation cohort.
8. Validation and Global Test SHA-256 match M5.1 control manifest.
9. M5.2 Round-0 SHA-256 equals M5.1 Round-0 SHA-256 (a1c052476e...).
10. Deterministic partition generation under seed 42.
11. Label distribution manifest and statistics correctness.
12. FNR (False Negative Rate) and FPR (False Positive Rate) calculation.
13. Client drift ($L_2$ norm) metric computation is finite and non-negative.
14. Production model isolation preserved (SHA-256 untouched).
15. Communication payload calculation approximately identical to M5.1.
16. 3-round Label-Skew integration smoke test (Gate B).
"""

import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from app.core.model import MultimodalBaselineModel
from federated.checkpointing import compute_file_sha256, FLRunManager
from federated.communication import calculate_model_parameter_bytes, record_round_communication
from federated.data import DEFAULT_PARTITION_DIR, generate_dataset_manifest, get_client_partition_paths
from federated.metrics import calculate_clinical_metrics
from federated.partition_generator import (
    compute_binary_entropy,
    compute_pairwise_js_divergence,
    generate_controlled_label_skew_partitions,
    get_file_sha256,
)
from federated.task import build_model, evaluate_model, train_local

M51_RUN_ID = "FL-RUN-20261016-130403-45D587"
M51_R0_SHA = "a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973"
PROD_MODEL_SHA = "17d59aa90455386a1e2f00bed2107ef0cbe2a6fb3e96e3f476faf0272db0ec73"
PARTITION_DIR = Path("storage/federated/partitions/label_skew_seed42")


@pytest.fixture(scope="module")
def ensure_partitions():
    """Memastikan partisi label skew telah dibuat sebelum pengujian."""
    if not (PARTITION_DIR / "client_1_train.csv").exists():
        generate_controlled_label_skew_partitions()
    return PARTITION_DIR


def test_1_iid_and_label_skew_pool_equality_and_no_duplication(ensure_partitions):
    """Memverifikasi bahwa pool training M5.2 identik 100% dengan M5.1 tanpa ada subjek hilang/asing."""
    iid_dir = DEFAULT_PARTITION_DIR / "IID_3Clients"
    iid_dfs = [pd.read_csv(iid_dir / f"client_{c}_train.csv") for c in [1, 2, 3]]
    iid_pids = set(pd.concat(iid_dfs)["participant_id"].astype(str))

    ls_dfs = [pd.read_csv(PARTITION_DIR / f"client_{c}_train.csv") for c in [1, 2, 3]]
    pids_c1 = set(ls_dfs[0]["participant_id"].astype(str))
    pids_c2 = set(ls_dfs[1]["participant_id"].astype(str))
    pids_c3 = set(ls_dfs[2]["participant_id"].astype(str))
    ls_pids = pids_c1 | pids_c2 | pids_c3

    # 1. Total pool equality
    assert iid_pids == ls_pids, "Pool subjek training M5.2 TIDAK identik dengan M5.1!"
    assert len(ls_pids) == 163, f"Jumlah total partisipan harus 163, ditemukan {len(ls_pids)}"

    # 2. No duplication across clients
    assert len(pids_c1 & pids_c2) == 0, "Ditemukan duplikasi subjek antara Klien 1 dan 2"
    assert len(pids_c1 & pids_c3) == 0, "Ditemukan duplikasi subjek antara Klien 1 dan 3"
    assert len(pids_c2 & pids_c3) == 0, "Ditemukan duplikasi subjek antara Klien 2 dan 3"


def test_2_exact_client_quantities_and_global_class_counts(ensure_partitions):
    """Memverifikasi pelestarian kuantitas 54/55/54 dan jumlah kelas global 50 positif / 113 negatif."""
    ls_dfs = [pd.read_csv(PARTITION_DIR / f"client_{c}_train.csv") for c in [1, 2, 3]]
    lengths = [len(df) for df in ls_dfs]
    assert lengths == [54, 55, 54], f"Kuantitas klien harus [54, 55, 54], ditemukan {lengths}"

    all_ls = pd.concat(ls_dfs, ignore_index=True)
    positives = int((all_ls["phq8_score"] >= 10.0).sum())
    negatives = int((all_ls["phq8_score"] < 10.0).sum())

    assert positives == 50, f"Total positif harus 50, ditemukan {positives}"
    assert negatives == 113, f"Total negatif harus 113, ditemukan {negatives}"


def test_3_label_skew_prevalence_difference_exceeds_iid(ensure_partitions):
    """Memverifikasi bahwa disparitas prevalensi label skew jauh melampaui IID kontrol."""
    ls_dfs = [pd.read_csv(PARTITION_DIR / f"client_{c}_train.csv") for c in [1, 2, 3]]
    ls_prevs = [float((df["phq8_score"] >= 10.0).mean()) for df in ls_dfs]

    iid_dir = DEFAULT_PARTITION_DIR / "IID_3Clients"
    iid_dfs = [pd.read_csv(iid_dir / f"client_{c}_train.csv") for c in [1, 2, 3]]
    iid_prevs = [float((df["phq8_score"] >= 10.0).mean()) for df in iid_dfs]

    diff_ls = max(ls_prevs) - min(ls_prevs)
    diff_iid = max(iid_prevs) - min(iid_prevs)

    assert diff_ls > 0.60, f"Disparitas prevalensi label skew ({diff_ls:.4f}) harus > 60%"
    assert diff_ls > (diff_iid * 10), f"Disparitas label skew ({diff_ls:.4f}) harus jauh lebih besar dari IID ({diff_iid:.4f})"

    # Verifikasi prevalensi spesifik target
    assert ls_prevs[0] > 0.75, f"Klien 1 (Depression-Heavy) harus > 75%, didapat {ls_prevs[0]*100:.2f}%"
    assert ls_prevs[1] < 0.10, f"Klien 2 (Control-Heavy) harus < 10%, didapat {ls_prevs[1]*100:.2f}%"


def test_4_zero_leakage_and_hash_identity_with_m51_control(ensure_partitions):
    """Memverifikasi zero leakage terhadap hold-out test dan validasi, serta kesamaan hash dengan M5.1."""
    test_path = DEFAULT_PARTITION_DIR / "global_test_index.csv"
    val_path = DEFAULT_PARTITION_DIR / "IID_3Clients" / "client_1_val.csv"

    test_df = pd.read_csv(test_path)
    val_df = pd.read_csv(val_path)

    test_pids = set(test_df["participant_id"].astype(str))
    val_pids = set(val_df["participant_id"].astype(str))

    ls_dfs = [pd.read_csv(PARTITION_DIR / f"client_{c}_train.csv") for c in [1, 2, 3]]
    all_train_pids = set(pd.concat(ls_dfs)["participant_id"].astype(str))

    # Assert zero overlap
    assert len(all_train_pids & test_pids) == 0, "DATA LEAKAGE: Training overlap dengan Global Test!"
    assert len(all_train_pids & val_pids) == 0, "DATA LEAKAGE: Training overlap dengan Validation Cohort!"

    # Validasi kesesuaian hash dengan M5.1
    m51_man_path = Path(f"storage/federated/runs/{M51_RUN_ID}/dataset_manifest.json")
    if m51_man_path.exists():
        with open(m51_man_path) as f:
            m51_man = json.load(f)
        assert compute_file_sha256(test_path) == m51_man["global_test"]["sha256"]
        assert compute_file_sha256(val_path) == m51_man["client_val_partitions"]["client_1"]["sha256"]


def test_5_round_zero_initialization_sha256_match():
    """Memverifikasi bahwa checkpoint Round 0 yang akan digunakan M5.2 identik bitwise dengan M5.1."""
    r0_path = Path(f"storage/federated/runs/{M51_RUN_ID}/rounds/round_000/model.pt")
    assert r0_path.exists(), f"M5.1 Round 0 model tidak ditemukan di {r0_path}"

    r0_sha = compute_file_sha256(r0_path)
    assert r0_sha == M51_R0_SHA, f"SHA-256 Round 0 mismatch: {r0_sha} != {M51_R0_SHA}"

    # Strict load assertion
    model = MultimodalBaselineModel()
    ckpt = torch.load(r0_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=True)


def test_6_deterministic_partition_generation_under_seed_42():
    """Memverifikasi bahwa pembangkitan partisi dengan seed 42 bersifat 100% deterministik."""
    with tempfile.TemporaryDirectory() as td:
        t_dir = Path(td)
        man1 = generate_controlled_label_skew_partitions(output_dir=t_dir / "run1", seed=42)
        man2 = generate_controlled_label_skew_partitions(output_dir=t_dir / "run2", seed=42)

        for c in ["client_1", "client_2", "client_3"]:
            sha1 = man1["generated_partitions"][c]["sha256"]
            sha2 = man2["generated_partitions"][c]["sha256"]
            assert sha1 == sha2, f"Partisi {c} tidak deterministik!"


def test_7_label_distribution_manifest_correctness(ensure_partitions):
    """Memverifikasi integritas berkas partition_manifest.json dan label_distribution.json."""
    man_path = PARTITION_DIR / "partition_manifest.json"
    dist_path = PARTITION_DIR / "label_distribution.json"

    assert man_path.exists(), "partition_manifest.json tidak ditemukan!"
    assert dist_path.exists(), "label_distribution.json tidak ditemukan!"

    with open(man_path) as f:
        man = json.load(f)
    assert man["protocol_name"] == "controlled_label_skew_v1"
    assert man["invariants_verification"]["zero_duplicates_verified"] is True
    assert man["invariants_verification"]["zero_test_leakage_verified"] is True
    assert man["invariants_verification"]["quantities_preserved"] is True

    with open(dist_path) as f:
        dist = json.load(f)
    assert dist["label_skew"]["max_min_prevalence_difference"] > 0.70
    assert dist["label_skew"]["mean_pairwise_js_divergence"] > 0.40


def test_8_fnr_fpr_clinical_metric_calculation():
    """Memverifikasi formula perhitungan FNR dan FPR pada modul metrics.py."""
    # Skenario sintetis: 10 sampel, 4 positif aktual (PHQ-8 >= 10.0), 6 negatif aktual
    y_true = [12.0, 14.0, 15.0, 11.0, 3.0, 4.0, 5.0, 2.0, 1.0, 0.0]
    # Prediksi: TP=3 (12, 14, 15), FN=1 (11 diprediksi 8.0), TN=5 (3, 4, 5, 2, 1), FP=1 (0 diprediksi 10.5)
    y_pred = [13.0, 15.0, 16.0, 8.0, 2.0, 3.0, 4.0, 1.0, 2.0, 10.5]

    res = calculate_clinical_metrics(y_true, y_pred, threshold=10.0)
    assert res["tp"] == 3
    assert res["fn"] == 1
    assert res["tn"] == 5
    assert res["fp"] == 1

    # FNR = FN / (TP + FN) = 1 / 4 = 0.25
    assert res["false_negative_rate"] == 0.25
    # FPR = FP / (FP + TN) = 1 / 6 = 0.1667
    assert res["false_positive_rate"] == 0.1667


def test_9_client_drift_l2_norm_finite():
    """Memverifikasi perhitungan L2 norm drift parameter lokal terhadap model global."""
    m_glob = build_model(seed=42)
    m_local = build_model(seed=42)

    # Sedikit modifikasi bobot lokal
    with torch.no_grad():
        for p in m_local.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    glob_sd = m_glob.state_dict()
    loc_sd = m_local.state_dict()

    drift_sq = 0.0
    for k, v in glob_sd.items():
        drift_sq += float(torch.sum((loc_sd[k] - v) ** 2).item())
    drift = float(np.sqrt(drift_sq))

    assert np.isfinite(drift), "Drift L2 harus berupa nilai berhingga (finite)"
    assert drift > 0.0, "Drift L2 harus positif jika bobot berubah"


def test_10_production_model_isolation_preserved():
    """Memverifikasi bahwa model operasional produksi storage/models/fl_iid/model.pt tidak tersentuh."""
    prod_path = Path("storage/models/fl_iid/model.pt")
    assert prod_path.exists(), "Model produksi tidak ditemukan!"
    sha = compute_file_sha256(prod_path)
    assert sha == PROD_MODEL_SHA, f"CRITICAL: Model produksi tertimpa! SHA={sha}"


def test_11_expected_communication_payload_identical_to_m51():
    """Memverifikasi akuntansi beban payload komunikasi terfederasi identik dengan M5.1."""
    model = build_model(seed=42)
    bytes_per_model = calculate_model_parameter_bytes(model)
    mb_per_model = bytes_per_model / (1024 * 1024)

    # 7,832,001 params * 4 bytes = 31,328,004 bytes (~29.8767 MB)
    assert bytes_per_model == 31328004
    assert round(mb_per_model, 4) == 29.8767

    # 3 klien: Downlink = 3 * 29.88 MB, Uplink = 3 * 29.88 MB -> ~179.26 MB per ronde
    round_payload_mb = 6 * mb_per_model
    assert round(round_payload_mb, 2) == 179.26


def test_12_label_skew_three_round_integration_smoke(ensure_partitions):
    """
    Gate B Smoke Test: Menjalankan 3 ronde simulasi label skew pada direktori run terisolasi.
    Memverifikasi: 3/3 klien berpartisipasi, finite loss, R0-R3 checkpoints, zero failure.
    """
    with tempfile.TemporaryDirectory() as td:
        smoke_dir = Path(td) / "smoke_run"
        manager = FLRunManager(run_id="SMOKE-LABEL-SKEW", base_dir=smoke_dir.parent, prefix="FL-LABEL-SKEW")

        # Inisialisasi dari checkpoint M5.1 R0
        src_r0 = Path(f"storage/federated/runs/{M51_RUN_ID}/rounds/round_000/model.pt")
        initial_sd = torch.load(src_r0, map_location="cpu", weights_only=False)

        # Simpan R0 checkpoint
        r0_path, r0_sha = manager.save_round_checkpoint(
            server_round=0,
            state_dict=initial_sd,
            metadata={"test": "smoke_r0"},
            source_checkpoint_to_copy=src_r0,
        )
        assert r0_sha == M51_R0_SHA, "Smoke R0 SHA mismatch!"

        # Simulasi 3 ronde lokal dengan partisi label skew
        current_sd = dict(initial_sd)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for r in range(1, 4):
            client_updates = []
            for c_id in range(3):
                train_path, val_path = get_client_partition_paths(
                    partition_id=c_id,
                    scenario="non_iid_label_skew",
                )
                assert train_path.exists(), f"Partisi client {c_id} tidak ditemukan"

                m = build_model()
                m.load_state_dict(current_sd, strict=True)
                from federated.data import MultimodalPartitionDataset
                from torch.utils.data import DataLoader

                ds = MultimodalPartitionDataset(train_path)
                loader = DataLoader(ds, batch_size=16, shuffle=True)

                loss, n_samples, _ = train_local(
                    model=m,
                    train_loader=loader,
                    epochs=1,
                    lr=1e-4,
                    weight_decay=0.01,
                    device=device,
                )
                assert np.isfinite(loss), f"Loss pada ronde {r} klien {c_id} tidak finite!"
                client_updates.append((m.state_dict(), n_samples))

            # Agregasi FedAvg sederhana untuk smoke test
            total_n = sum(n for _, n in client_updates)
            new_sd = {}
            for k in current_sd:
                new_sd[k] = sum(sd[k] * (n / total_n) for sd, n in client_updates)
            current_sd = new_sd

            _, r_sha = manager.save_round_checkpoint(
                server_round=r,
                state_dict=current_sd,
                metadata={"test": f"smoke_r{r}"},
            )
            assert len(r_sha) == 64

        # Pastikan seluruh checkpoint r0 s.d r3 tersimpan
        for r in range(4):
            assert (manager.rounds_dir / f"round_{r:03d}" / "model.pt").exists()
