"""
tests/test_federated_client.py
==============================
Pengujian Round-trip Parameter ArrayRecord dan Pelatihan Lokal Klien (Milestone 4).
"""

import math
import copy
import pytest
import torch
from flwr.app.message.arrayrecord import ArrayRecord

from app.core.model import MultimodalBaselineModel
from federated.data import build_client_dataloaders
from federated.task import build_model, train_local


def test_model_parameter_arrayrecord_roundtrip():
    """
    Memverifikasi konversi dua arah (round-trip) PyTorch state_dict <-> ArrayRecord Flower 1.40.
    Semua key dan dimensi tensor harus identik presisi tanpa degradasi.
    """
    model = build_model(seed=42)
    original_sd = model.state_dict()

    # 1. Konversi state_dict -> ArrayRecord
    array_record = ArrayRecord.from_torch_state_dict(original_sd)
    assert array_record.count_bytes() > 0

    # 2. Konversi balik ArrayRecord -> state_dict
    restored_sd = array_record.to_torch_state_dict()

    assert set(restored_sd.keys()) == set(original_sd.keys())

    # 3. Verifikasi kesamaan numerik setiap tensor
    for k in original_sd:
        diff = (original_sd[k] - restored_sd[k]).abs().max().item()
        assert diff == 0.0, f"Perbedaan nilai terdeteksi pada parameter '{k}': diff={diff}"


def test_local_training_step_changes_weights_and_loss_finite():
    """
    Memverifikasi bahwa eksekusi pelatihan lokal:
    - Menghasilkan loss yang berhingga (finite)
    - Benar-benar memperbarui bobot model (parameter berubah setelah optimizer.step)
    - Mengembalikan jumlah sampel yang sesuai
    """
    model = build_model(seed=42)
    initial_weights = copy.deepcopy(model.regressor[0].weight.data)

    train_loader, _, train_ds, _ = build_client_dataloaders(partition_id=0, batch_size=8)
    assert len(train_ds) > 0

    avg_loss, num_samples, metrics = train_local(
        model=model,
        train_loader=train_loader,
        epochs=1,
        lr=1e-3,  # LR cukup besar untuk memastikan perubahan terukur
        device="cpu",
    )

    # 1. Verifikasi finiteness loss
    assert math.isfinite(avg_loss)
    assert avg_loss > 0.0

    # 2. Verifikasi bobot model berubah secara nyata
    updated_weights = model.regressor[0].weight.data
    weight_diff = (updated_weights - initial_weights).abs().sum().item()
    assert weight_diff > 1e-5, f"Bobot model tidak berubah setelah pelatihan: diff={weight_diff}"

    # 3. Verifikasi jumlah sampel
    assert num_samples == len(train_ds)
    assert "mae" in metrics
    assert "macro_f1" in metrics


def test_client_metric_record_privacy_and_aggregation_safety():
    """
    Memverifikasi bahwa MetricRecord klien Flower:
    - Hanya memuat metrik agregat skalar aman (num-examples, train_loss, train_mae, train_f1, val_*)
    - TIDAK memuat partition_id (M5.1-A cleanup)
    - TIDAK memuat data privasi (raw audio, video, teks, fitur multimodal, ID partisipan, individual gradients)
    """
    from flwr.app.message.arrayrecord import ArrayRecord
    from flwr.app.message.context import Context
    from flwr.app.message.message import Message
    from flwr.app.message.recorddict import RecordDict
    from flwr.supercore.task_identity import TaskIdentity
    from federated.client_app import train

    TaskIdentity._run_id = 1
    TaskIdentity._node_id = 0
    TaskIdentity._task_id = 1

    model = build_model(seed=42)
    arrays = ArrayRecord.from_torch_state_dict(model.state_dict())
    msg_content = RecordDict({"arrays": arrays})
    msg = Message(content=msg_content, dst_node_id=1, message_type="train")

    # Simulasikan node context
    context = Context(
        run_id=101,
        node_id=1,
        node_config={"partition-id": 0},
        state=RecordDict(),
        run_config={"local-epochs": 1, "batch-size": 16, "learning-rate": 1e-4},
    )

    reply_msg = train(message=msg, context=context)
    assert "metrics" in reply_msg.content
    metrics_rec = reply_msg.content["metrics"]

    # 1. Pastikan partition_id TIDAK ADA di MetricRecord
    assert "partition_id" not in metrics_rec, "CRITICAL ERROR: partition_id tidak boleh ada di MetricRecord!"

    # 2. Pastikan metrik wajib ada
    assert "num-examples" in metrics_rec
    assert "train_loss" in metrics_rec
    assert "train_mae" in metrics_rec
    assert "train_f1" in metrics_rec
    assert "val_loss" in metrics_rec

    # 3. Pastikan semua metrik adalah skalar numerik yang aman untuk agregasi
    for k, v in metrics_rec.items():
        assert isinstance(v, (int, float)), f"Nilai metrik '{k}' harus berupa skalar numerik: {type(v)}"
        assert not math.isnan(float(v))
        assert math.isfinite(float(v))

    # 4. Pastikan tidak ada kunci yang membocorkan data klinis
    prohibited_keys = {"audio", "visual", "text", "participant_id", "patient_id", "gradient", "labels"}
    for k in metrics_rec.keys():
        assert k.lower() not in prohibited_keys
