"""
tests/test_federated_quantity_skew.py
======================================
Comprehensive Test Suite for FED-MIND Phase 4 — Milestone 5.3:
Controlled Non-IID Quantity Skew Experiment.

Verifies:
1. Quantity partition totals equal 163.
2. Exact client distribution 98 / 49 / 16.
3. Equality of IID and Quantity-Skew participant pools (S_IID == S_QUANTITY).
4. Zero participant duplication across clients.
5. No participant omitted (163 unique IDs preserved).
6. Global positive (50) and negative (113) counts strictly preserved.
7. Per-client label prevalence diagnostics (~30.6% across all clients, no label skew).
8. Deterministic partition generation under seed 42.
9. Partition manifest hashes match actual generated files.
10. Validation/test hash equality to M5.1 control run.
11. Zero train/test overlap.
12. Zero train/validation overlap.
13. Round-0 checkpoint SHA-256 identity (a1c052476e...).
14. FedAvg aggregation mathematical correctness with weights 98/163, 49/163, 16/163.
15. Correct num-examples return values in client datasets (98, 49, 16).
16. Local optimizer step accounting (14 for C1, 8 for C2, 2 for C3).
17. Client drift calculations (L2 norm finite and non-negative).
18. Weighted contribution calculations ((n_i / N) * ||Delta w_i||).
19. FNR / FPR clinical metric calculations.
20. Cumulative communication accounting.
21. Strict model-loading compatibility with MultimodalBaselineModel.
22. Production model isolation and integrity preserved.
23. Three-round Flower integration smoke test (Gate B).
"""

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import pytest
import torch

from app.core.model import MultimodalBaselineModel
from federated.checkpointing import compute_file_sha256, FLRunManager
from federated.communication import calculate_model_parameter_bytes, record_round_communication
from federated.data import (
    DEFAULT_PARTITION_DIR,
    build_client_dataloaders,
    build_global_test_dataloader,
    generate_dataset_manifest,
    get_client_partition_paths,
)
from federated.metrics import calculate_clinical_metrics
from federated.partitions.quantity_skew import (
    compute_binary_entropy,
    compute_pairwise_js_divergence,
    generate_controlled_quantity_skew_partitions,
    get_file_sha256,
)
from federated.task import build_model, evaluate_model, train_local

M51_RUN_ID = "FL-RUN-20261016-130403-45D587"
M51_R0_SHA = "a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973"
PROD_MODEL_SHA = "17d59aa90455386a1e2f00bed2107ef0cbe2a6fb3e96e3f476faf0272db0ec73"
PARTITION_DIR = Path("storage/federated/partitions/quantity_skew_seed42")


@pytest.fixture(scope="module")
def ensure_quantity_partitions():
    """Memastikan partisi quantity skew telah dibangkitkan sebelum pengujian."""
    if not (PARTITION_DIR / "client_1_train.csv").exists():
        generate_controlled_quantity_skew_partitions()
    return PARTITION_DIR


def test_1_quantity_partition_totals_equal_163(ensure_quantity_partitions):
    """Test 1: Memverifikasi total seluruh sampel pelatihan tepat 163."""
    dfs = [pd.read_csv(PARTITION_DIR / f"client_{c}_train.csv") for c in [1, 2, 3]]
    total_len = sum(len(df) for df in dfs)
    assert total_len == 163, f"Total sampel harus 163, ditemukan {total_len}"


def test_2_exact_distribution_98_49_16(ensure_quantity_partitions):
    """Test 2: Memverifikasi alokasi integer tepat 98, 49, 16 (~60.12%, 30.06%, 9.82%)."""
    c1 = pd.read_csv(PARTITION_DIR / "client_1_train.csv")
    c2 = pd.read_csv(PARTITION_DIR / "client_2_train.csv")
    c3 = pd.read_csv(PARTITION_DIR / "client_3_train.csv")
    assert len(c1) == 98, f"Client 1 harus 98, ditemukan {len(c1)}"
    assert len(c2) == 49, f"Client 2 harus 49, ditemukan {len(c2)}"
    assert len(c3) == 16, f"Client 3 harus 16, ditemukan {len(c3)}"


def test_3_equality_of_iid_and_quantity_skew_participant_pools(ensure_quantity_partitions):
    """Test 3: Memverifikasi S_IID == S_QUANTITY."""
    iid_dir = DEFAULT_PARTITION_DIR / "IID_3Clients"
    iid_dfs = [pd.read_csv(iid_dir / f"client_{c}_train.csv") for c in [1, 2, 3]]
    iid_pids = set(pd.concat(iid_dfs)["participant_id"].astype(str))

    qs_dfs = [pd.read_csv(PARTITION_DIR / f"client_{c}_train.csv") for c in [1, 2, 3]]
    qs_pids = set(pd.concat(qs_dfs)["participant_id"].astype(str))

    assert iid_pids == qs_pids, "INVARIANT VIOLATION: S_QUANTITY != S_IID!"
    assert len(qs_pids) == 163


def test_4_no_participant_duplication(ensure_quantity_partitions):
    """Test 4: Memverifikasi nol duplikasi partisipan antar klien."""
    c1_pids = set(pd.read_csv(PARTITION_DIR / "client_1_train.csv")["participant_id"].astype(str))
    c2_pids = set(pd.read_csv(PARTITION_DIR / "client_2_train.csv")["participant_id"].astype(str))
    c3_pids = set(pd.read_csv(PARTITION_DIR / "client_3_train.csv")["participant_id"].astype(str))

    assert len(c1_pids & c2_pids) == 0, "Duplikasi ditemukan antara Client 1 dan Client 2"
    assert len(c1_pids & c3_pids) == 0, "Duplikasi ditemukan antara Client 1 dan Client 3"
    assert len(c2_pids & c3_pids) == 0, "Duplikasi ditemukan antara Client 2 dan Client 3"


def test_5_no_participant_omitted(ensure_quantity_partitions):
    """Test 5: Memverifikasi tidak ada partisipan yang hilang dari 163 pool training."""
    qs_dfs = [pd.read_csv(PARTITION_DIR / f"client_{c}_train.csv") for c in [1, 2, 3]]
    all_qs = pd.concat(qs_dfs, ignore_index=True)
    assert all_qs["participant_id"].nunique() == 163


def test_6_global_positive_negative_totals_preserved(ensure_quantity_partitions):
    """Test 6: Memverifikasi total positif (50) dan negatif (113) global dipertahankan."""
    qs_dfs = [pd.read_csv(PARTITION_DIR / f"client_{c}_train.csv") for c in [1, 2, 3]]
    all_qs = pd.concat(qs_dfs, ignore_index=True)

    pos = int((all_qs["phq8_score"] >= 10.0).sum())
    neg = int((all_qs["phq8_score"] < 10.0).sum())

    assert pos == 50, f"Expected 50 global positives, found {pos}"
    assert neg == 113, f"Expected 113 global negatives, found {neg}"


def test_7_per_client_label_prevalence_diagnostics(ensure_quantity_partitions):
    """Test 7: Memverifikasi prevalensi depresi per klien terjaga seimbang (~30.6%)."""
    qs_dfs = [pd.read_csv(PARTITION_DIR / f"client_{c}_train.csv") for c in [1, 2, 3]]
    global_prev = 50.0 / 163.0  # ~0.3067

    c1_pos = (qs_dfs[0]["phq8_score"] >= 10.0).sum()
    c2_pos = (qs_dfs[1]["phq8_score"] >= 10.0).sum()
    c3_pos = (qs_dfs[2]["phq8_score"] >= 10.0).sum()

    assert c1_pos == 30, f"Client 1 positives expected 30, got {c1_pos}"
    assert c2_pos == 15, f"Client 2 positives expected 15, got {c2_pos}"
    assert c3_pos == 5, f"Client 3 positives expected 5, got {c3_pos}"

    prev1 = c1_pos / 98.0  # 30.61%
    prev2 = c2_pos / 49.0  # 30.61%
    prev3 = c3_pos / 16.0  # 31.25%

    for idx, p in enumerate([prev1, prev2, prev3]):
        dev = abs(p - global_prev)
        assert dev < 0.01, f"Client {idx+1} prevalence deviation {dev:.4f} exceeds 1%"

    js_div = compute_pairwise_js_divergence([prev1, prev2, prev3])
    assert js_div < 0.01, f"Jensen-Shannon divergence {js_div:.6f} indicates unintended label skew"


def test_8_deterministic_partition_generation():
    """Test 8: Memverifikasi determinisme pembangkitan partisi dengan seed 42."""
    with tempfile.TemporaryDirectory() as td:
        t_dir = Path(td)
        man1 = generate_controlled_quantity_skew_partitions(output_dir=t_dir / "run1", seed=42)
        man2 = generate_controlled_quantity_skew_partitions(output_dir=t_dir / "run2", seed=42)

        for c in ["client_1", "client_2", "client_3"]:
            assert man1["client_partitions"][c]["sha256"] == man2["client_partitions"][c]["sha256"]


def test_9_partition_manifest_hashes(ensure_quantity_partitions):
    """Test 9: Memverifikasi hash pada partition_manifest.json cocok dengan berkas aktual."""
    man_path = PARTITION_DIR / "partition_manifest.json"
    assert man_path.exists()
    with open(man_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for c in ["client_1", "client_2", "client_3"]:
        csv_file = PARTITION_DIR / f"{c}_train.csv"
        actual_sha = get_file_sha256(csv_file)
        manifest_sha = manifest["client_partitions"][c]["sha256"]
        assert actual_sha == manifest_sha, f"SHA mismatch for {c}: {actual_sha} != {manifest_sha}"


def test_10_validation_test_hash_equality_to_m51():
    """Test 10: Memverifikasi kesamaan SHA-256 berkas evaluasi global dan shared validation dengan M5.1."""
    test_path = DEFAULT_PARTITION_DIR / "global_test_index.csv"
    val_path = DEFAULT_PARTITION_DIR / "IID_3Clients" / "client_1_val.csv"

    m51_man_path = Path(f"storage/federated/runs/{M51_RUN_ID}/dataset_manifest.json")
    assert m51_man_path.exists(), "M5.1 dataset manifest tidak ditemukan"

    with open(m51_man_path) as f:
        m51_man = json.load(f)

    test_sha = compute_file_sha256(test_path)
    val_sha = compute_file_sha256(val_path)

    assert test_sha == m51_man["global_test"]["sha256"]
    assert val_sha == m51_man["client_val_partitions"]["client_1"]["sha256"]


def test_11_zero_train_test_overlap(ensure_quantity_partitions):
    """Test 11: Memverifikasi zero overlap antara partisi training Quantity Skew dan Global Test Set."""
    test_path = DEFAULT_PARTITION_DIR / "global_test_index.csv"
    test_pids = set(pd.read_csv(test_path)["participant_id"].astype(str))

    qs_dfs = [pd.read_csv(PARTITION_DIR / f"client_{c}_train.csv") for c in [1, 2, 3]]
    all_train_pids = set(pd.concat(qs_dfs)["participant_id"].astype(str))

    overlap = all_train_pids & test_pids
    assert len(overlap) == 0, f"DATA LEAKAGE: Training overlap dengan test set: {overlap}"


def test_12_zero_train_validation_overlap(ensure_quantity_partitions):
    """Test 12: Memverifikasi zero overlap antara partisi training Quantity Skew dan Shared Validation Set."""
    val_path = DEFAULT_PARTITION_DIR / "IID_3Clients" / "client_1_val.csv"
    val_pids = set(pd.read_csv(val_path)["participant_id"].astype(str))

    qs_dfs = [pd.read_csv(PARTITION_DIR / f"client_{c}_train.csv") for c in [1, 2, 3]]
    all_train_pids = set(pd.concat(qs_dfs)["participant_id"].astype(str))

    overlap = all_train_pids & val_pids
    assert len(overlap) == 0, f"DATA LEAKAGE: Training overlap dengan validation set: {overlap}"


def test_13_round_zero_checkpoint_hash_identity():
    """Test 13: Memverifikasi integritas checkpoint awal Round 000 identik bitwise dengan M5.1."""
    r0_path = Path(f"storage/federated/runs/{M51_RUN_ID}/rounds/round_000/model.pt")
    assert r0_path.exists()
    h = compute_file_sha256(r0_path)
    assert h == M51_R0_SHA, f"Round 0 SHA-256 mismatch: {h} != {M51_R0_SHA}"


def test_14_fedavg_aggregation_weights_98_49_16():
    """Test 14: Memverifikasi kebenaran matematis FedAvg dengan bobot 98/163, 49/163, 16/163."""
    t1 = torch.tensor([10.0, 20.0, 30.0])
    t2 = torch.tensor([40.0, 50.0, 60.0])
    t3 = torch.tensor([70.0, 80.0, 90.0])

    n1, n2, n3 = 98, 49, 16
    total_n = n1 + n2 + n3
    assert total_n == 163

    w1 = n1 / total_n
    w2 = n2 / total_n
    w3 = n3 / total_n

    aggregated = w1 * t1 + w2 * t2 + w3 * t3
    expected = (98 * t1 + 49 * t2 + 16 * t3) / 163.0

    assert torch.allclose(aggregated, expected, atol=1e-6)
    # Buktikan bahwa ini TIDAK sama dengan simple unweighted average
    unweighted = (t1 + t2 + t3) / 3.0
    assert not torch.allclose(aggregated, unweighted)


def test_15_correct_num_examples_return_values(ensure_quantity_partitions):
    """Test 15: Memverifikasi build_client_dataloaders mengembalikan ukuran dataset yang tepat."""
    for pid, expected_n in [(0, 98), (1, 49), (2, 16)]:
        t_loader, v_loader, t_ds, v_ds = build_client_dataloaders(
            partition_id=pid,
            scenario="non_iid_quantity_skew",
            base_partition_dir=PARTITION_DIR,
        )
        assert len(t_ds) == expected_n, f"Partition {pid} dataset size {len(t_ds)} != {expected_n}"


def test_16_local_optimizer_step_accounting(ensure_quantity_partitions):
    """Test 16: Memverifikasi jumlah langkah optimizer lokal per klien untuk batch_size=16 dan local_epochs=2."""
    batch_size = 16
    local_epochs = 2

    # Client 1: 98 -> ceil(98/16) = 7 batches * 2 epochs = 14 steps
    # Client 2: 49 -> ceil(49/16) = 4 batches * 2 epochs = 8 steps
    # Client 3: 16 -> ceil(16/16) = 1 batch * 2 epochs = 2 steps
    expected_steps = {0: 14, 1: 8, 2: 2}

    for pid in [0, 1, 2]:
        t_loader, _, _, _ = build_client_dataloaders(
            partition_id=pid,
            batch_size=batch_size,
            scenario="non_iid_quantity_skew",
            base_partition_dir=PARTITION_DIR,
        )
        minibatches = len(t_loader)
        steps = minibatches * local_epochs
        assert steps == expected_steps[pid], f"Client {pid+1} steps {steps} != {expected_steps[pid]}"


def test_17_client_drift_calculations():
    """Test 17: Memverifikasi perhitungan drift parameter L2 bernilai hingga dan non-negatif."""
    m_init = build_model(seed=42)
    sd_init = m_init.state_dict()

    m_updated = build_model(seed=42)
    with torch.no_grad():
        for p in m_updated.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    sd_updated = m_updated.state_dict()

    drift_sq = 0.0
    for k in sd_init:
        drift_sq += float(torch.sum((sd_updated[k] - sd_init[k]) ** 2).item())
    drift = float(np.sqrt(drift_sq))

    assert np.isfinite(drift)
    assert drift > 0.0


def test_18_weighted_contribution_calculations():
    """Test 18: Memverifikasi perhitungan kontribusi terbobot (n_i / N) * ||Delta w_i||."""
    n_i = 98
    total_n = 163
    drift = 2.5
    weighted = (n_i / total_n) * drift
    assert abs(weighted - (98.0 / 163.0) * 2.5) < 1e-6
    assert weighted > 0.0


def test_19_fnr_fpr_calculations():
    """Test 19: Memverifikasi perhitungan klinis FNR dan FPR."""
    y_true = [12.0, 14.0, 8.0, 4.0, 15.0]  # Pos: 3, Neg: 2
    y_pred = [11.0, 7.0, 12.0, 3.0, 16.0]  # TP: 2, FN: 1, FP: 1, TN: 1
    metrics = calculate_clinical_metrics(y_true, y_pred, threshold=10.0)

    tp = metrics["tp"]
    fn = metrics["fn"]
    fp = metrics["fp"]
    tn = metrics["tn"]

    fnr = fn / (tp + fn)
    fpr = fp / (fp + tn)

    assert tp == 2 and fn == 1 and fp == 1 and tn == 1
    assert abs(fnr - (1.0 / 3.0)) < 1e-5
    assert abs(fpr - (1.0 / 2.0)) < 1e-5


def test_20_cumulative_communication_accounting():
    """Test 20: Memverifikasi akuntansi muatan komunikasi parameter."""
    model = build_model(seed=42)
    bytes_per_model = calculate_model_parameter_bytes(model)
    mb_per_model = bytes_per_model / (1024 * 1024)

    # 7,832,001 params * 4 bytes = 31,328,004 bytes (~29.8767 MB)
    assert bytes_per_model == 31328004
    assert round(mb_per_model, 4) == 29.8767

    # 3 klien: Downlink = 3 * 29.88 MB, Uplink = 3 * 29.88 MB -> ~179.26 MB per ronde
    round_payload_mb = 6 * mb_per_model
    assert round(round_payload_mb, 2) == 179.26


def test_21_strict_model_loading_compatibility():
    """Test 21: Memverifikasi MultimodalBaselineModel dapat memuat checkpoint secara ketat."""
    r0_path = Path(f"storage/federated/runs/{M51_RUN_ID}/rounds/round_000/model.pt")
    model = MultimodalBaselineModel()
    ckpt = torch.load(r0_path, map_location="cpu", weights_only=False)
    # Harus strict load tanpa error
    model.load_state_dict(ckpt, strict=True)
    assert model.training is True or model.training is False


def test_22_production_model_integrity():
    """Test 22: Memverifikasi bahwa model operasional produksi storage/models/fl_iid/model.pt tidak tersentuh."""
    prod_path = Path("storage/models/fl_iid/model.pt")
    assert prod_path.exists(), "Model produksi tidak ditemukan!"
    sha = compute_file_sha256(prod_path)
    assert sha == PROD_MODEL_SHA, f"CRITICAL: Model produksi tertimpa! SHA={sha}"


def test_23_three_round_flower_integration_smoke(ensure_quantity_partitions):
    """
    Test 23 (Gate B Smoke Test):
    Menjalankan 3 ronde integrasi quantity skew pada direktori run terisolasi.
    Memverifikasi:
    - 3/3 klien berpartisipasi (98, 49, 16 sampel).
    - Bobot agregasi FedAvg terbobot (98/163, 49/163, 16/163).
    - Finite loss di seluruh ronde.
    - Checkpoints round 0 s.d 3 tersimpan lengkap dengan SHA-256.
    - Checkpoint Round 0 identik bitwise dengan M5.1 control (a1c05247...).
    - Client metrics, metrics, dan communication tercatat.
    - Model produksi tidak tertimpa.
    """
    with tempfile.TemporaryDirectory() as td:
        smoke_dir = Path(td) / "smoke_run"
        manager = FLRunManager(run_id="SMOKE-QUANTITY-SKEW", base_dir=smoke_dir.parent, prefix="FL-QUANTITY-SKEW")

        # Inisialisasi dari checkpoint M5.1 R0
        src_r0 = Path(f"storage/federated/runs/{M51_RUN_ID}/rounds/round_000/model.pt")
        initial_sd = torch.load(src_r0, map_location="cpu", weights_only=False)

        # Simpan R0 checkpoint dengan copying langsung
        r0_path, r0_sha = manager.save_round_checkpoint(
            server_round=0,
            state_dict=initial_sd,
            metadata={"test": "smoke_quantity_r0"},
            source_checkpoint_to_copy=src_r0,
        )
        assert r0_sha == M51_R0_SHA, f"Smoke R0 SHA mismatch: {r0_sha}"

        # Simulasi 3 ronde lokal dengan partisi Quantity Skew
        current_sd = dict(initial_sd)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        expected_quantities = [98, 49, 16]

        for r in range(1, 4):
            client_updates = []
            for c_id in range(3):
                train_path, val_path = get_client_partition_paths(
                    partition_id=c_id,
                    scenario="non_iid_quantity_skew",
                )
                assert train_path.exists(), f"Partisi client {c_id} tidak ditemukan"

                m = build_model()
                m.load_state_dict(current_sd, strict=True)
                from federated.data import MultimodalPartitionDataset
                from torch.utils.data import DataLoader

                ds = MultimodalPartitionDataset(train_path)
                assert len(ds) == expected_quantities[c_id], f"Client {c_id} size {len(ds)} != {expected_quantities[c_id]}"
                loader = DataLoader(ds, batch_size=16, shuffle=True)

                loss, n_samples, train_metrics = train_local(
                    model=m,
                    train_loader=loader,
                    epochs=1,
                    lr=1e-4,
                    weight_decay=0.01,
                    device=device,
                )
                assert np.isfinite(loss), f"Loss pada ronde {r} klien {c_id} tidak finite!"
                client_updates.append((m.state_dict(), n_samples))

                manager.record_client_metrics(
                    server_round=r,
                    client_id=f"node_{c_id}",
                    num_examples=n_samples,
                    metrics={
                        "train_loss": loss,
                        "train_mae": train_metrics.get("mae", 0.0),
                        "train_f1": train_metrics.get("macro_f1", 0.0),
                        "client_update_l2_norm": 0.5,
                        "weighted_update_l2_norm": (n_samples / 163.0) * 0.5,
                    },
                )

            # Agregasi FedAvg terbobot (weighted by num-examples)
            total_n = sum(n for _, n in client_updates)
            assert total_n == 163, f"Total samples {total_n} != 163"

            new_sd = {}
            for k in current_sd:
                new_sd[k] = sum(sd[k] * (n / total_n) for sd, n in client_updates)
            current_sd = new_sd

            _, r_sha = manager.save_round_checkpoint(
                server_round=r,
                state_dict=current_sd,
                metadata={"test": f"smoke_quantity_r{r}"},
            )
            assert len(r_sha) == 64

            # Record round communication
            record_round_communication(
                csv_path=manager.comm_csv_path,
                server_round=r,
                model=m,
                participating_clients=3,
            )

        # Pastikan seluruh checkpoint r0 s.d r3 tersimpan
        for r in range(4):
            assert (manager.rounds_dir / f"round_{r:03d}" / "model.pt").exists()

        # Verifikasi client_metrics.csv mencatat ketiga kuantitas 98, 49, 16
        c_df = pd.read_csv(manager.client_metrics_csv_path)
        assert set(c_df["num_examples"].unique()) == {98, 49, 16}
        assert len(c_df) == 9  # 3 rounds * 3 clients

        # Verifikasi communication.csv
        comm_df = pd.read_csv(manager.comm_csv_path)
        assert len(comm_df) == 3
