"""
tests/test_federated_checkpoint.py
==================================
Pengujian Unit untuk Checkpointing, SHA-256 Hashing, dan Communication Tracking (Milestone 4).
"""

import json
from pathlib import Path
import pytest
import torch

from app.core.model import MultimodalBaselineModel
from federated.checkpointing import FLRunManager, compute_file_sha256
from federated.communication import calculate_model_parameter_bytes, record_round_communication
from federated.task import build_model


def test_checkpointing_lifecycle(tmp_path: Path):
    """
    Memverifikasi pembuatan run_id, penyimpanan checkpoint Round 000 & 001,
    hashing SHA-256, dan integritas file metadata.json.
    """
    run_manager = FLRunManager(run_id="TEST-RUN-001", base_dir=tmp_path)
    assert run_manager.run_dir.exists()

    model = build_model(seed=42)
    state_dict = model.state_dict()

    # 1. Simpan Round 000
    meta_r0 = {"round": 0, "status": "initialized", "notes": "initial global model"}
    ckpt_path_r0, sha256_r0 = run_manager.save_round_checkpoint(
        server_round=0,
        state_dict=state_dict,
        metadata=meta_r0,
    )
    assert ckpt_path_r0.exists()
    assert len(sha256_r0) == 64  # SHA-256 hex string

    meta_file_r0 = run_manager.rounds_dir / "round_000" / "metadata.json"
    assert meta_file_r0.exists()
    with open(meta_file_r0, "r", encoding="utf-8") as f:
        loaded_meta_r0 = json.load(f)
    assert loaded_meta_r0["round"] == 0
    assert loaded_meta_r0["checkpoint_sha256"] == sha256_r0

    # 2. Simpan Round 001
    meta_r1 = {"round": 1, "status": "fedavg_completed", "train_loss": 0.45}
    ckpt_path_r1, sha256_r1 = run_manager.save_round_checkpoint(
        server_round=1,
        state_dict=state_dict,
        metadata=meta_r1,
    )
    assert ckpt_path_r1.exists()
    assert len(sha256_r1) == 64

    # 3. Verifikasi metrics.csv logging
    run_manager.record_metrics(server_round=0, metrics={"mae": 2.5, "rmse": 3.1})
    run_manager.record_metrics(server_round=1, metrics={"mae": 2.1, "rmse": 2.8})
    assert run_manager.metrics_csv_path.exists()

    with open(run_manager.metrics_csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 3  # Header + 2 rows


def test_communication_accounting_deterministic():
    """
    Memverifikasi determinisme perhitungan ukuran parameter model dan overhead komunikasi.
    """
    model = build_model()
    param_bytes = calculate_model_parameter_bytes(model)

    # 7.832.001 parameter float32 (4 bytes per parameter) = 31.328.004 bytes (~29.87 MB)
    expected_param_count = sum(p.numel() for p in model.parameters())
    assert expected_param_count == 7832001
    assert param_bytes == 7832001 * 4

    csv_temp = Path("temp_comm_test.csv")
    if csv_temp.exists():
        csv_temp.unlink()

    try:
        rec = record_round_communication(
            csv_path=csv_temp,
            server_round=1,
            model=model,
            participating_clients=3,
        )

        assert rec["round"] == 1
        assert rec["participating_clients"] == 3
        assert rec["model_parameter_bytes"] == param_bytes
        assert rec["estimated_downlink_parameter_bytes"] == param_bytes * 3
        assert rec["estimated_uplink_parameter_bytes"] == param_bytes * 3
        assert rec["estimated_total_parameter_bytes"] == param_bytes * 6
        assert csv_temp.exists()
    finally:
        if csv_temp.exists():
            csv_temp.unlink()


def test_existing_production_checkpoint_not_overwritten():
    """
    Memastikan bahwa eksekusi FL smoke test tidak menimpa storage/models/fl_iid/model.pt.
    """
    repo_root = Path(__file__).resolve().parent.parent
    prod_ckpt = repo_root / "storage" / "models" / "fl_iid" / "model.pt"

    if prod_ckpt.exists():
        before_hash = compute_file_sha256(prod_ckpt)

        # Inisialisasi run manager terpisah
        run_manager = FLRunManager(run_id="PROD-SAFETY-CHECK")
        model = build_model(seed=999)
        run_manager.save_round_checkpoint(0, model.state_dict(), {"test": True})

        after_hash = compute_file_sha256(prod_ckpt)
        assert before_hash == after_hash, "CRITICAL ERROR: storage/models/fl_iid/model.pt tertimpa!"


def test_resume_lifecycle_and_metric_deduplication(tmp_path: Path):
    """
    Memverifikasi mekanisme resume eksperimen:
    - Mendeteksi ronde terakhir yang tersimpan (get_latest_completed_round).
    - Memuat bobot checkpoint untuk resume.
    - Melanjutkan pencatatan metrik tanpa menduplikasi baris ronde sebelumnya.
    """
    run_manager = FLRunManager(run_id="TEST-RESUME-001", base_dir=tmp_path)
    model = build_model(seed=42)
    state_dict = model.state_dict()

    # 1. Simpan ronde 0 s/d 3
    for r in range(4):
        run_manager.save_round_checkpoint(r, state_dict, {"round": r})
        run_manager.record_metrics(r, {"mae": 5.0 - r * 0.5, "rmse": 7.0 - r * 0.5})

    latest = run_manager.get_latest_completed_round()
    assert latest == 3

    # 2. Muat checkpoint ronde 3
    loaded_sd, loaded_meta = run_manager.load_round_checkpoint(3)
    assert loaded_meta["round"] == 3
    assert set(loaded_sd.keys()) == set(state_dict.keys())

    # 3. Simulasikan resume: coba catat ulang ronde 3, lalu catat ronde 4
    run_manager.record_metrics(3, {"mae": 99.0, "rmse": 99.0})  # Harus diabaikan karena sudah ada
    run_manager.record_metrics(4, {"mae": 3.0, "rmse": 4.5})

    with open(run_manager.metrics_csv_path, "r", encoding="utf-8") as f:
        rows = [line.strip().split(",") for line in f.readlines() if line.strip()]

    # Header + 5 data rows (ronde 0, 1, 2, 3, 4) = 6 rows
    assert len(rows) == 6
    recorded_rounds = [int(r[0]) for r in rows[1:]]
    assert recorded_rounds == [0, 1, 2, 3, 4]


def test_dataset_manifest_and_environment_serialization(tmp_path: Path):
    """
    Memverifikasi pembuatan dan persistensi dataset_manifest.json dan environment.json.
    """
    from federated.data import generate_dataset_manifest
    from federated.server_app import get_environment_info

    run_manager = FLRunManager(run_id="TEST-METADATA-001", base_dir=tmp_path)

    # 1. Manifest Dataset
    ds_manifest = generate_dataset_manifest()
    assert ds_manifest["zero_leakage_verified"] is True
    assert ds_manifest["total_train_participants"] == 163
    assert ds_manifest["total_test_participants"] == 56

    manifest_file = run_manager.save_dataset_manifest(ds_manifest)
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        loaded_ds = json.load(f)
    assert loaded_ds["total_train_participants"] == 163

    # 2. Metadata Environment
    env_info = get_environment_info()
    assert "pytorch_version" in env_info
    assert "flower_version" in env_info
    assert "git_commit" in env_info
    assert env_info["model_parameter_count"] == 7832001

    env_file = run_manager.save_environment(env_info)
    assert env_file.exists()


def test_cumulative_communication_tracking(tmp_path: Path):
    """
    Memverifikasi bahwa overhead komunikasi parameter terakumulasi secara benar antar ronde.
    """
    csv_file = tmp_path / "comm_cum_test.csv"
    model = build_model()
    param_bytes = calculate_model_parameter_bytes(model)

    # Ronde 0: 0 clients (initial global setup)
    rec0 = record_round_communication(csv_file, server_round=0, model=model, participating_clients=0)
    assert rec0["cumulative_parameter_bytes"] == 0

    # Ronde 1: 3 clients -> downlink (3 * P) + uplink (3 * P) = 6 * P
    rec1 = record_round_communication(csv_file, server_round=1, model=model, participating_clients=3)
    assert rec1["estimated_total_parameter_bytes"] == param_bytes * 6
    assert rec1["cumulative_parameter_bytes"] == param_bytes * 6

    # Ronde 2: 3 clients -> akumulasi menjadi 12 * P
    rec2 = record_round_communication(csv_file, server_round=2, model=model, participating_clients=3)
    assert rec2["estimated_total_parameter_bytes"] == param_bytes * 6
    assert rec2["cumulative_parameter_bytes"] == param_bytes * 12


def test_reference_model_loading_and_comparison():
    """
    Memverifikasi bahwa checkpoint referensi historis dapat dimuat secara kompatibel
    oleh MultimodalBaselineModel dan fungsi komparasi menghasilkan metrik lengkap.
    """
    from federated.evaluation import compare_with_historical_reference

    model = build_model(seed=42)
    state_dict = model.state_dict()

    comp_result = compare_with_historical_reference(
        reproduction_state_dict=state_dict,
        device="cpu",
    )

    if comp_result.get("status") != "REFERENCE_NOT_FOUND":
        assert "reference_sha256" in comp_result
        assert "prediction_correlation" in comp_result
        assert "metrics_comparison" in comp_result
        comp_metrics = comp_result["metrics_comparison"]
        assert "mae" in comp_metrics
        assert "rmse" in comp_metrics
        assert "macro_f1" in comp_metrics
        assert comp_metrics["mae"]["absolute_delta"] is not None
