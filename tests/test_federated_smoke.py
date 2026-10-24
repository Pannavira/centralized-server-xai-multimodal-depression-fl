"""
tests/test_federated_smoke.py
=============================
Integration & Smoke Test untuk Siklus 1 Ronde Federated Learning (Milestone 4).

Memverifikasi siklus end-to-end:
1. Inisialisasi model global awal (Round 0).
2. 3 klien (Partisi 0, 1, 2) melatih model secara lokal pada partisinya masing-masing.
3. Klien mengembalikan bobot terupdate dan jumlah sampel (num-examples).
4. Agregasi FedAvg terbobot menghasilkan model global Round 1.
5. Evaluasi terpusat pada global hold-out test set dieksekusi untuk Round 0 dan Round 1.
6. Checkpoint, SHA-256, metrics.csv, dan communication.csv tersimpan dengan benar.
7. Memvalidasi bahwa artefak simulasi Flower yang telah dieksekusi lengkap dan valid.
"""

from pathlib import Path
import pytest
import torch

from app.core.model import MultimodalBaselineModel
from federated.checkpointing import FLRunManager, compute_file_sha256
from federated.communication import record_round_communication
from federated.data import build_client_dataloaders
from federated.evaluation import GlobalEvaluator
from federated.task import build_model, train_local


def test_one_round_fl_integration_lifecycle(tmp_path: Path):
    """
    Simulasi langsung siklus 1 ronde FL dengan 3 client lokal dan FedAvg terbobot.
    """
    device = torch.device("cpu")
    seed = 42

    # 1. Inisialisasi Run Manager & Model Global R0
    run_manager = FLRunManager(run_id="TEST-SMOKE-R1", base_dir=tmp_path)
    global_model = build_model(seed=seed)
    r0_state_dict = {k: v.clone() for k, v in global_model.state_dict().items()}

    # Evaluasi R0 pada global holdout
    evaluator = GlobalEvaluator()
    r0_metrics = evaluator.evaluate_state_dict(r0_state_dict, device=device)
    assert "mae" in r0_metrics
    assert "rmse" in r0_metrics
    assert r0_metrics["mae"] > 0.0

    # Simpan checkpoint Round 000
    r0_ckpt_path, r0_sha256 = run_manager.save_round_checkpoint(
        server_round=0,
        state_dict=r0_state_dict,
        metadata={"round": 0, "status": "initialized", "metrics": r0_metrics},
    )
    assert r0_ckpt_path.exists()

    # 2. Pelatihan Lokal 3 Klien
    client_updates = []
    client_weights = []

    for partition_id in [0, 1, 2]:
        client_loader, _, train_ds, _ = build_client_dataloaders(
            partition_id=partition_id,
            batch_size=16,
        )
        num_samples = len(train_ds)
        assert num_samples > 0

        # Setiap klien memuat copy dari global model R0
        client_model = build_model()
        client_model.load_state_dict(r0_state_dict, strict=True)

        # Latih 1 batch saja untuk efisiensi smoke test
        avg_loss, trained_samples, train_metrics = train_local(
            model=client_model,
            train_loader=client_loader,
            epochs=1,
            learning_rate=1e-4,
            device=device,
        )

        assert avg_loss > 0.0
        assert not torch.isnan(torch.tensor(avg_loss))

        client_updates.append(client_model.state_dict())
        client_weights.append(num_samples)

    total_samples = sum(client_weights)
    assert total_samples == 163  # 54 + 55 + 54

    # 3. Agregasi FedAvg Terbobot (Manual Exact Verification)
    aggregated_state_dict = {}
    for key in r0_state_dict.keys():
        weighted_sum = sum(
            update[key].float() * (weight / total_samples)
            for update, weight in zip(client_updates, client_weights)
        )
        aggregated_state_dict[key] = weighted_sum.to(r0_state_dict[key].dtype)

    # 4. Evaluasi Global Hold-out Model Round 1
    r1_metrics = evaluator.evaluate_state_dict(aggregated_state_dict, device=device)
    assert "mae" in r1_metrics
    assert "rmse" in r1_metrics

    # 5. Simpan Checkpoint Round 001
    r1_ckpt_path, r1_sha256 = run_manager.save_round_checkpoint(
        server_round=1,
        state_dict=aggregated_state_dict,
        metadata={"round": 1, "status": "fedavg_completed", "metrics": r1_metrics},
    )
    assert r1_ckpt_path.exists()

    # 6. Verifikasi Checkpoint Hashes Berbeda (Bobot benar-benar terupdate)
    assert r0_sha256 != r1_sha256

    # 7. Verifikasi Communication Overhead Tracking
    comm_rec = record_round_communication(
        csv_path=run_manager.comm_csv_path,
        server_round=1,
        model=global_model,
        participating_clients=3,
    )
    assert comm_rec["participating_clients"] == 3
    assert comm_rec["model_parameter_bytes"] == 31328004
    assert run_manager.comm_csv_path.exists()


def test_flower_simulation_artifacts_integrity():
    """
    Memverifikasi integritas artefak yang dihasilkan oleh Flower simulation run.
    """
    repo_root = Path(__file__).resolve().parent.parent
    runs_dir = repo_root / "storage" / "federated" / "runs"
    assert runs_dir.exists(), "Direktori storage/federated/runs harus ada."

    runs = list(runs_dir.glob("FL-RUN-*"))
    completed_runs = [r for r in runs if (r / "manifest.json").exists()]
    assert len(completed_runs) > 0, "Harus ada minimal satu run FL yang telah selesai."

    latest_run = sorted(completed_runs)[-1]
    manifest_path = latest_run / "manifest.json"
    config_path = latest_run / "config.json"
    metrics_path = latest_run / "metrics.csv"
    comm_path = latest_run / "communication.csv"
    r0_model = latest_run / "rounds" / "round_000" / "model.pt"
    r1_model = latest_run / "rounds" / "round_001" / "model.pt"

    assert manifest_path.exists(), "manifest.json harus ada"
    assert config_path.exists(), "config.json harus ada"
    assert metrics_path.exists(), "metrics.csv harus ada"
    assert comm_path.exists(), "communication.csv harus ada"
    assert r0_model.exists(), "round_000/model.pt harus ada"
    assert r1_model.exists(), "round_001/model.pt harus ada"

    # Verifikasi SHA256 model R0 dan R1
    r0_sha = compute_file_sha256(r0_model)
    r1_sha = compute_file_sha256(r1_model)
    assert len(r0_sha) == 64
    assert len(r1_sha) == 64
    assert r0_sha != r1_sha
