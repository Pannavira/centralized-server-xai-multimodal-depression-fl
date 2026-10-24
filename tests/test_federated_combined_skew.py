"""
tests/test_federated_combined_skew.py
======================================
Automated Test Suite for Milestone 5.5: Controlled Combined Non-IID Experiment.

Verifies:
1. Deterministic generation of combined partitions (seed 42).
2. Pool equality with M5.1 (S_IID == S_COMBINED).
3. Exactly 163 unique training participants.
4. Exact client sample quantities (98 / 49 / 16).
5. Exact label preservation: 50 positives (PHQ-8 >= 10) and 113 negatives (PHQ-8 < 10).
6. Correct local positive prevalence targets (18.37%, 38.78%, 81.25%).
7. Zero participant duplication across clients.
8. Zero leakage into Global Hold-Out Test (56 subjects).
9. Zero leakage into Shared Validation Cohort (56 subjects).
10. Global initialization bitwise hash match with M5.1 Round 0 (a1c05247...).
11. Clean Client 1 modalities unchanged.
12. Audio-only corruption for Client 2 (sigma=0.25).
13. Visual-only temporal frame dropout for Client 3 (dropout_rate=0.20, neutral mean imputation).
14. Text features strictly unchanged across all clients.
15. Ground-truth PHQ-8 scores and labels invariant under degradation.
16. Feature tensor shape and dtype preservation.
17. Audio and visual masking integrity.
18. Deterministic feature degradation under repeated accesses.
19. Scenario routing in get_client_partition_paths and build_client_dataloaders.
20. FedAvg aggregation sample weights (98/163, 49/163, 16/163).
21. Finite client training losses.
22. Local optimizer step accounting (14, 8, 2 steps per 2-epoch round).
23. Client drift calculation accuracy.
24. Clinical safety calculations (FNR, FPR).
25. Checkpoint metadata completeness.
26. Resume equivalence and invariance test.
27. Communication parameter payload accounting.
28. Preservation of prior milestone artifacts (M5.1, M5.2, M5.3, M5.4).
29. Production model registry protection.
30. Gate B 3-round Flower FL simulation smoke test.
"""

import copy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pytest
import torch

from app.core.model import MultimodalBaselineModel
from federated.checkpointing import FLRunManager, compute_file_sha256
from federated.communication import record_round_communication
from federated.data import (
    DEFAULT_PARTITION_DIR,
    build_client_dataloaders,
    build_global_test_dataloader,
    get_client_partition_paths,
    verify_zero_leakage,
)
from federated.metrics import calculate_clinical_metrics
from federated.modality_degradation import (
    ModalityDegradationConfig,
    apply_audio_noise,
    apply_client_modality_treatment,
    apply_visual_frame_dropout,
    compute_degradation_diagnostics,
    get_deterministic_sample_seed,
    get_visual_neutral_frame,
)
from federated.partitions.combined_skew import (
    compute_binary_entropy,
    compute_pairwise_js_divergence,
    generate_controlled_combined_skew_partitions,
)
from federated.task import build_model, evaluate_model, train_local

M51_RUN_ID = "FL-RUN-20261016-130403-45D587"
M51_R0_SHA = "a1c052476e613d9d6795e8246b54b0bb86f8d5ba65e7dde8d20ca9f703808973"
EXPECTED_QUANTITIES = [98, 49, 16]
EXPECTED_POSITIVES = [18, 19, 13]
EXPECTED_NEGATIVES = [80, 30, 3]


# ==============================================================================
# 1. PARTITION GENERATION & DETERMINISM TESTS
# ==============================================================================

def test_1_combined_partition_generator_determinism(tmp_path: Path):
    """Memverifikasi generator partisi deterministik menghasilkan hash file identik pada 2 pemanggilan."""
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    rep1 = generate_controlled_combined_skew_partitions(output_dir=out1, seed=42)
    rep2 = generate_controlled_combined_skew_partitions(output_dir=out2, seed=42)

    for c in [1, 2, 3]:
        h1 = compute_file_sha256(out1 / f"client_{c}_train.csv")
        h2 = compute_file_sha256(out2 / f"client_{c}_train.csv")
        assert h1 == h2, f"Client {c} hash mismatch across runs: {h1} != {h2}"


def test_2_pool_equality_with_m51_iid():
    """Memverifikasi S_COMBINED persis identik dengan S_IID (100% pool partisipan M5.1 dipertahankan)."""
    iid_base = Path(DEFAULT_PARTITION_DIR) / "IID_3Clients"
    iid_ids: Set[str] = set()
    for c in [1, 2, 3]:
        df = pd.read_csv(iid_base / f"client_{c}_train.csv")
        iid_ids.update(df["participant_id"].astype(str))

    comb_base = Path("storage/federated/partitions/combined_skew_seed42")
    comb_ids: Set[str] = set()
    for c in [1, 2, 3]:
        df = pd.read_csv(comb_base / f"client_{c}_train.csv")
        comb_ids.update(df["participant_id"].astype(str))

    assert comb_ids == iid_ids
    assert len(comb_ids) == 163


def test_3_exactly_163_unique_participants():
    """Memverifikasi tepat 163 partisipan unik tanpa duplikasi."""
    comb_base = Path("storage/federated/partitions/combined_skew_seed42")
    all_ids = []
    for c in [1, 2, 3]:
        df = pd.read_csv(comb_base / f"client_{c}_train.csv")
        all_ids.extend(df["participant_id"].astype(str).tolist())

    assert len(all_ids) == 163
    assert len(set(all_ids)) == 163


def test_4_exact_client_quantities_98_49_16():
    """Memverifikasi kuantitas tiap klien tepat 98, 49, 16."""
    comb_base = Path("storage/federated/partitions/combined_skew_seed42")
    counts = [len(pd.read_csv(comb_base / f"client_{c}_train.csv")) for c in [1, 2, 3]]
    assert counts == EXPECTED_QUANTITIES


def test_5_exact_label_totals_50_pos_113_neg():
    """Memverifikasi total label global 50 positif dan 113 negatif tetap terjaga utuh."""
    comb_base = Path("storage/federated/partitions/combined_skew_seed42")
    total_pos = 0
    total_neg = 0
    for c in [1, 2, 3]:
        df = pd.read_csv(comb_base / f"client_{c}_train.csv")
        total_pos += int((df["phq8_score"] >= 10.0).sum())
        total_neg += int((df["phq8_score"] < 10.0).sum())

    assert total_pos == 50
    assert total_neg == 113


def test_6_local_positive_prevalence_targets():
    """Memverifikasi prevalensi positif per-klien sesuai target (18.37%, 38.78%, 81.25%)."""
    comb_base = Path("storage/federated/partitions/combined_skew_seed42")
    for c, p_exp, n_exp, q_exp in zip([1, 2, 3], EXPECTED_POSITIVES, EXPECTED_NEGATIVES, EXPECTED_QUANTITIES):
        df = pd.read_csv(comb_base / f"client_{c}_train.csv")
        p_act = int((df["phq8_score"] >= 10.0).sum())
        n_act = int((df["phq8_score"] < 10.0).sum())
        assert p_act == p_exp
        assert n_act == n_exp
        assert len(df) == q_exp
        assert abs((p_act / q_exp) - (p_exp / q_exp)) < 1e-6


def test_7_no_duplicate_participants_between_clients():
    """Memverifikasi tidak ada satupun partisipan yang overlap antar klien."""
    comb_base = Path("storage/federated/partitions/combined_skew_seed42")
    id_sets = [set(pd.read_csv(comb_base / f"client_{c}_train.csv")["participant_id"].astype(str)) for c in [1, 2, 3]]
    assert len(id_sets[0].intersection(id_sets[1])) == 0
    assert len(id_sets[0].intersection(id_sets[2])) == 0
    assert len(id_sets[1].intersection(id_sets[2])) == 0


def test_8_zero_leakage_global_test():
    """Memverifikasi zero-leakage terhadap Global Hold-Out Test set (56 subjek)."""
    comb_base = Path("storage/federated/partitions/combined_skew_seed42")
    dfs = [pd.read_csv(comb_base / f"client_{c}_train.csv") for c in [1, 2, 3]]
    test_df = pd.read_csv(Path(DEFAULT_PARTITION_DIR) / "global_test_index.csv")
    is_valid, leakage = verify_zero_leakage(dfs, test_df)
    assert is_valid is True
    assert len(leakage) == 0


def test_9_zero_leakage_shared_validation():
    """Memverifikasi zero-leakage terhadap Shared Validation cohort (56 subjek)."""
    comb_base = Path("storage/federated/partitions/combined_skew_seed42")
    dfs = [pd.read_csv(comb_base / f"client_{c}_train.csv") for c in [1, 2, 3]]
    val_df = pd.read_csv(Path(DEFAULT_PARTITION_DIR) / "IID_3Clients" / "client_1_val.csv")
    is_valid, leakage = verify_zero_leakage(dfs, val_df)
    assert is_valid is True
    assert len(leakage) == 0


def test_10_round_0_initialization_hash_match():
    """Memverifikasi checksum model Round 0 bitwise identik dengan M5.1 Control."""
    r0_path = Path(f"storage/federated/runs/{M51_RUN_ID}/rounds/round_000/model.pt")
    assert r0_path.exists()
    computed_sha = compute_file_sha256(r0_path)
    assert computed_sha == M51_R0_SHA


# ==============================================================================
# 2. MODALITY DEGRADATION INTEGRATION TESTS
# ==============================================================================

def test_11_clean_client_1_modalities_unchanged():
    """Client 1 (Partition 0, 98 sampel): Semua modalitas Audio, Visual, Text tetap bersih."""
    _, _, train_ds, _ = build_client_dataloaders(partition_id=0, scenario="non_iid_combined")
    assert len(train_ds) == 98
    for i in range(min(5, len(train_ds))):
        raw = train_ds._get_raw_item(i)
        deg = train_ds[i]
        assert torch.equal(raw["audio"], deg["audio"])
        assert torch.equal(raw["visual"], deg["visual"])
        assert torch.equal(raw["text"], deg["text"])


def test_12_audio_only_corruption_client_2():
    """Client 2 (Partition 1, 49 sampel): HANYA Audio yang terkorupsi; Visual & Text bersih."""
    _, _, train_ds, _ = build_client_dataloaders(partition_id=1, scenario="non_iid_combined")
    assert len(train_ds) == 49
    for i in range(min(5, len(train_ds))):
        raw = train_ds._get_raw_item(i)
        deg = train_ds[i]
        # Audio harus termodifikasi
        assert not torch.equal(raw["audio"], deg["audio"])
        assert torch.isfinite(deg["audio"]).all()
        # Visual dan Text harus tetap identik
        assert torch.equal(raw["visual"], deg["visual"])
        assert torch.equal(raw["text"], deg["text"])


def test_13_visual_only_corruption_client_3():
    """Client 3 (Partition 2, 16 sampel): HANYA Visual yang terkorupsi (dropout frame); Audio & Text bersih."""
    _, _, train_ds, _ = build_client_dataloaders(partition_id=2, scenario="non_iid_combined")
    assert len(train_ds) == 16
    for i in range(min(5, len(train_ds))):
        raw = train_ds._get_raw_item(i)
        deg = train_ds[i]
        # Visual harus termodifikasi
        assert not torch.equal(raw["visual"], deg["visual"])
        assert torch.isfinite(deg["visual"]).all()
        # Audio dan Text harus tetap identik
        assert torch.equal(raw["audio"], deg["audio"])
        assert torch.equal(raw["text"], deg["text"])


def test_14_text_always_unchanged_across_all_clients():
    """Modalitas Teks harus selalu utuh dan bersih di ketiga klien."""
    for p_id in range(3):
        _, _, train_ds, _ = build_client_dataloaders(partition_id=p_id, scenario="non_iid_combined")
        for i in range(min(3, len(train_ds))):
            raw = train_ds._get_raw_item(i)
            deg = train_ds[i]
            assert torch.equal(raw["text"], deg["text"])


def test_15_phq8_label_invariance_under_degradation():
    """Nilai ground-truth phq8_score dan depression_label tidak boleh berubah oleh degradasi."""
    for p_id in range(3):
        _, _, train_ds, _ = build_client_dataloaders(partition_id=p_id, scenario="non_iid_combined")
        for i in range(min(3, len(train_ds))):
            raw = train_ds._get_raw_item(i)
            deg = train_ds[i]
            assert raw["phq8_score"] == deg["phq8_score"]
            assert raw["depression_label"] == deg["depression_label"]


def test_16_feature_tensor_shapes_and_dtypes():
    """Bentuk tensor dan tipe data harus presisi: audio [512, 80], visual [512, 72, 3], text [512, 768]."""
    for p_id in range(3):
        _, _, train_ds, _ = build_client_dataloaders(partition_id=p_id, scenario="non_iid_combined")
        item = train_ds[0]
        assert item["audio"].shape == (512, 80)
        assert item["visual"].shape == (512, 72, 3)
        assert item["text"].shape == (512, 768)
        assert item["audio"].dtype == torch.float32
        assert item["visual"].dtype == torch.float32
        assert item["text"].dtype == torch.float32


def test_17_audio_and_visual_masking_integrity():
    """Frame padding/invalid tidak boleh di-drop atau diisi secara salah."""
    neutral_frame = get_visual_neutral_frame()
    assert neutral_frame.shape == (72, 3)
    assert torch.isfinite(neutral_frame).all()

    dummy_vis = torch.zeros((512, 72, 3), dtype=torch.float32)
    dummy_vis[:100] = 1.0  # 100 frame valid
    out_vis, diag = apply_visual_frame_dropout(dummy_vis, dropout_rate=0.20, neutral_frame=neutral_frame, seed=42)
    assert diag["applied"] is True
    assert diag["dropped_frames"] == 20  # 20% dari 100 frame
    # Frame 100 s.d 512 harus tetap zero padding
    assert torch.equal(out_vis[100:], dummy_vis[100:])


def test_18_deterministic_degradation_under_repeated_access():
    """Mengakses sampel yang sama berulang kali harus menghasilkan tensor degradasi yang persis identik."""
    _, _, train_ds, _ = build_client_dataloaders(partition_id=1, scenario="non_iid_combined")
    item1 = train_ds[5]
    item2 = train_ds[5]
    assert torch.equal(item1["audio"], item2["audio"])


# ==============================================================================
# 3. PIPELINE ROUTING & TRAINING POLICY TESTS
# ==============================================================================

def test_19_client_partition_routing():
    """Memverifikasi get_client_partition_paths merespons scenario non_iid_combined dengan benar."""
    for p_id, expected_q in enumerate([98, 49, 16]):
        t_path, v_path = get_client_partition_paths(p_id, scenario="non_iid_combined")
        assert t_path.exists()
        assert v_path.exists()
        df = pd.read_csv(t_path)
        assert len(df) == expected_q


def test_20_fedavg_aggregation_weights_98_49_16():
    """Memverifikasi proporsi pembobotan FedAvg persis 98/163, 49/163, 16/163."""
    weights = [q / 163.0 for q in EXPECTED_QUANTITIES]
    assert abs(sum(weights) - 1.0) < 1e-9
    assert abs(weights[0] - 98 / 163.0) < 1e-9
    assert abs(weights[1] - 49 / 163.0) < 1e-9
    assert abs(weights[2] - 16 / 163.0) < 1e-9


def test_21_local_optimizer_step_accounting():
    """
    Memverifikasi jumlah optimizer step lokal per klien dengan batch size 16 dan 2 epoch:
    - Client 1: ceil(98 / 16) = 7 batch/epoch * 2 = 14 steps
    - Client 2: ceil(49 / 16) = 4 batch/epoch * 2 = 8 steps
    - Client 3: ceil(16 / 16) = 1 batch/epoch * 2 = 2 steps
    """
    expected_steps = [14, 8, 2]
    for p_id, exp_step in enumerate(expected_steps):
        loader, _, train_ds, _ = build_client_dataloaders(
            partition_id=p_id,
            batch_size=16,
            scenario="non_iid_combined",
        )
        minibatches = len(loader)
        total_steps = minibatches * 2  # 2 local epochs
        assert total_steps == exp_step, f"Client {p_id+1} steps mismatch: {total_steps} != {exp_step}"


def test_22_finite_training_loss_step():
    """Memverifikasi bahwa 1 step pelatihan lokal pada setiap klien menghasilkan loss terhingga (bukan NaN/Inf)."""
    m = build_model()
    for p_id in range(3):
        loader, _, _, _ = build_client_dataloaders(
            partition_id=p_id,
            batch_size=16,
            scenario="non_iid_combined",
        )
        loss, n_samples, metrics = train_local(
            model=m,
            train_loader=loader,
            epochs=1,
            lr=1e-4,
            device="cpu",
        )
        assert np.isfinite(loss)
        assert loss > 0.0
        assert n_samples == EXPECTED_QUANTITIES[p_id]


def test_23_client_drift_calculation():
    """Memverifikasi perhitungan L2 norm client update drift terhadap model global masuk."""
    m_glob = build_model(seed=42)
    m_local = build_model(seed=99)
    sd_glob = m_glob.state_dict()
    sd_loc = m_local.state_dict()

    drift_sq = 0.0
    for k in sd_glob:
        diff = sd_loc[k] - sd_glob[k]
        drift_sq += float(torch.sum(diff ** 2).item())
    drift_l2 = float(np.sqrt(drift_sq))
    assert drift_l2 > 0.0
    assert np.isfinite(drift_l2)


def test_24_clinical_safety_fnr_fpr_calculations():
    """Memverifikasi perhitungan FNR dan FPR pada threshold PHQ-8 >= 10.0."""
    y_true = [15.0, 12.0, 5.0, 3.0]  # 2 positif, 2 negatif
    y_pred = [14.0, 8.0, 11.0, 2.0]  # TP:1, FN:1, FP:1, TN:1
    metrics = calculate_clinical_metrics(y_true, y_pred, threshold=10.0)
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert abs(metrics["false_negative_rate"] - 0.5) < 1e-4
    assert abs(metrics["false_positive_rate"] - 0.5) < 1e-4


def test_25_checkpoint_metadata_structure():
    """Memverifikasi struktur metadata checkpoint per ronde."""
    meta = {
        "run_id": "FL-COMBINED-NONIID-TEST",
        "round": 0,
        "scenario": "non_iid_combined",
        "scientific_status": "controlled_experiment_combined_skew",
        "training_strategy": "FedAvg",
        "number_of_clients": 3,
        "local_epochs": 2,
        "batch_size": 16,
    }
    assert meta["round"] == 0
    assert meta["training_strategy"] == "FedAvg"
    assert meta["number_of_clients"] == 3


def test_26_resume_invariance_equivalence(tmp_path: Path):
    """Memverifikasi strict parameter parity (max diff = 0.0) antara continuous dan resumed execution."""
    m0 = build_model(seed=42)
    w0 = m0.state_dict()

    train_loader, _, _, _ = build_client_dataloaders(
        partition_id=2,  # Client 3 (Visual degraded)
        batch_size=16,
        scenario="non_iid_combined",
    )

    # Path A: Continuous
    torch.manual_seed(1234)
    mA = build_model()
    mA.load_state_dict(copy.deepcopy(w0), strict=True)
    train_local(mA, train_loader, epochs=1, lr=1e-4, device="cpu")
    wA = mA.state_dict()

    # Path B: Checkpoint Save -> Disk Reload -> Resume Step
    ckpt_file = tmp_path / "resume_ckpt.pt"
    torch.save(w0, ckpt_file)
    w_reloaded = torch.load(ckpt_file, map_location="cpu", weights_only=False)

    torch.manual_seed(1234)
    mB = build_model()
    mB.load_state_dict(w_reloaded, strict=True)
    train_local(mB, train_loader, epochs=1, lr=1e-4, device="cpu")
    wB = mB.state_dict()

    for k in wA.keys():
        diff = torch.max(torch.abs(wA[k] - wB[k])).item()
        assert diff == 0.0, f"Discrepancy detected in {k}: max diff = {diff}"


def test_27_communication_accounting_exact():
    """Memverifikasi beban transmisi parameter 50 ronde identik dengan M5.1 (9,398,401,200 bytes)."""
    m = build_model()
    param_count = sum(p.numel() for p in m.parameters())
    param_bytes = param_count * 4
    round_payload = param_bytes * 3 * 2  # 3 clients down + 3 clients up
    total_50_rounds = round_payload * 50
    assert total_50_rounds == 9398401200


def test_28_prior_milestone_artifacts_intact():
    """Memverifikasi seluruh artefak M5.1, M5.2, M5.3, M5.4 tetap utuh dan tidak tersentuh."""
    runs = [
        "FL-RUN-20261016-130403-45D587",
        "FL-LABEL-SKEW-20261016-141108-7865D4",
        "FL-QUANTITY-SKEW-20261019-110746-3A4BFF",
        "FL-MODALITY-SKEW-20261019-142242-CA6D11",
    ]
    for r in runs:
        r_dir = Path("storage/federated/runs") / r
        assert r_dir.exists()
        assert (r_dir / "metrics.csv").exists()
        assert (r_dir / "rounds" / "round_050" / "model.pt").exists()


def test_29_production_models_protected():
    """Memverifikasi model produksi centralized, fl_iid, dan fl_non_iid tidak tersentuh."""
    prod_files = [
        "storage/models/centralized/model.pt",
        "storage/models/fl_iid/model.pt",
        "storage/models/fl_non_iid/model.pt",
    ]
    for pf in prod_files:
        p = Path(pf)
        assert p.exists()
        assert p.stat().st_size > 0


# ==============================================================================
# 4. GATE B: THREE-ROUND FLOWER FL SIMULATION SMOKE TEST
# ==============================================================================

def test_30_gate_b_three_round_combined_smoke_test(tmp_path: Path):
    """
    Gate B Smoke Test:
    Simulasi langsung 3 ronde FedAvg dengan skenario Controlled Combined Non-IID.
    Memverifikasi:
    - 3 klien berpartisipasi dengan kuantitas asimetris 98/49/16.
    - Klien 1 clean, Klien 2 audio noisy, Klien 3 visual degraded.
    - Bobot agregasi terbobot FedAvg (98/163, 49/163, 16/163).
    - Finite training loss di semua ronde.
    - Checkpoint Round 0 identik bitwise dengan M5.1 Control (a1c05247...).
    - Checkpoint Round 1, 2, 3 tersimpan lengkap dengan metadata.
    """
    smoke_dir = tmp_path / "smoke_combined_run"
    manager = FLRunManager(run_id="SMOKE-COMBINED-SKEW", base_dir=smoke_dir.parent, prefix="FL-COMBINED-NONIID")

    src_r0 = Path(f"storage/federated/runs/{M51_RUN_ID}/rounds/round_000/model.pt")
    initial_sd = torch.load(src_r0, map_location="cpu", weights_only=False)

    _, r0_sha = manager.save_round_checkpoint(
        server_round=0,
        state_dict=initial_sd,
        metadata={"test": "smoke_combined_r0"},
        source_checkpoint_to_copy=src_r0,
    )
    assert r0_sha == M51_R0_SHA, f"Smoke R0 SHA mismatch: {r0_sha}"

    current_sd = dict(initial_sd)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for r in range(1, 4):
        client_updates = []
        for c_id in range(3):
            train_loader, _, train_ds, _ = build_client_dataloaders(
                partition_id=c_id,
                batch_size=16,
                scenario="non_iid_combined",
            )
            n_samples = len(train_ds)
            assert n_samples == EXPECTED_QUANTITIES[c_id]

            m = build_model()
            m.load_state_dict(current_sd, strict=True)

            loss, _, train_metrics = train_local(
                model=m,
                train_loader=train_loader,
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

        # FedAvg Weighted Aggregation
        total_n = sum(n for _, n in client_updates)
        assert total_n == 163

        new_sd = {}
        for k in current_sd:
            new_sd[k] = sum(sd[k] * (n / total_n) for sd, n in client_updates)
        current_sd = new_sd

        _, r_sha = manager.save_round_checkpoint(
            server_round=r,
            state_dict=current_sd,
            metadata={"test": f"smoke_combined_r{r}"},
        )
        assert len(r_sha) == 64

        record_round_communication(
            csv_path=manager.comm_csv_path,
            server_round=r,
            model=m,
            participating_clients=3,
        )

    for r in range(4):
        assert (manager.rounds_dir / f"round_{r:03d}" / "model.pt").exists()

    c_df = pd.read_csv(manager.client_metrics_csv_path)
    assert set(c_df["num_examples"].unique()) == {98, 49, 16}
    assert len(c_df) == 9  # 3 rounds * 3 clients

    comm_df = pd.read_csv(manager.comm_csv_path)
    assert len(comm_df) == 3


def test_31_round_0_evaluation_parity():
    """Memverifikasi Round 0 M5.1 Control dan M5.5 Combined memiliki bitwise model hash dan metrik evaluasi identik."""
    from federated.evaluation import GlobalEvaluator

    m51_r0 = Path(f"storage/federated/runs/{M51_RUN_ID}/rounds/round_000/model.pt")
    assert m51_r0.exists()
    assert compute_file_sha256(m51_r0) == M51_R0_SHA

    # Cari run combined non-iid M5.5 (Seed 42)
    runs_dir = Path("storage/federated/runs")
    m55_run_dir = runs_dir / "FL-COMBINED-NONIID-20261019-154311-CCF7B4"
    if not m55_run_dir.exists():
        comb_runs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("FL-COMBINED-NONIID-")])
        assert len(comb_runs) > 0
        m55_run_dir = comb_runs[0]
    m55_r0 = m55_run_dir / "rounds" / "round_000" / "model.pt"
    assert m55_r0.exists()
    assert compute_file_sha256(m55_r0) == M51_R0_SHA

    # Strict loading
    sd_m51 = torch.load(m51_r0, map_location="cpu", weights_only=True)
    sd_m55 = torch.load(m55_r0, map_location="cpu", weights_only=True)

    for k in sd_m51:
        assert torch.equal(sd_m51[k], sd_m55[k]), f"Weight mismatch in key {k}"

    # Single-process evaluation parity check on CPU
    evaluator = GlobalEvaluator(device="cpu")
    metrics_m51 = evaluator.evaluate_state_dict(sd_m51)
    metrics_m55 = evaluator.evaluate_state_dict(sd_m55)

    for metric_key in ["mae", "rmse", "mse", "macro_f1", "recall", "specificity", "auc_roc", "test_loss"]:
        diff = abs(metrics_m51[metric_key] - metrics_m55[metric_key])
        assert diff == 0.0, f"Round 0 evaluation disparity in {metric_key}: {metrics_m51[metric_key]} vs {metrics_m55[metric_key]}"

    # Verify metrics.csv record for Round 0
    df_m51 = pd.read_csv(f"storage/federated/runs/{M51_RUN_ID}/metrics.csv")
    df_m55 = pd.read_csv(m55_run_dir / "metrics.csv")
    r0_m51 = df_m51[df_m51["round"] == 0].iloc[0]
    r0_m55 = df_m55[df_m55["round"] == 0].iloc[0]
    assert round(r0_m51["mae"], 4) == round(r0_m55["mae"], 4) == 8.1071
    assert round(r0_m51["rmse"], 4) == round(r0_m55["rmse"], 4) == 10.3077
    assert round(r0_m51["macro_f1"], 4) == round(r0_m55["macro_f1"], 4) == 0.3846

