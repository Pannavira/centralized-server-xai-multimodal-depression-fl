"""
federated/client_app.py
=======================
Implementasi ClientApp modern Flower 1.40 Message API untuk FED-MIND FL.

Fitur:
- Menerima ArrayRecord dari ServerApp dan mengonversinya ke PyTorch state_dict secara langsung.
- Mengidentifikasi partisi dataset lokal dari context.node_config["partition-id"].
- Memuat HANYA partisi data lokal milik klien (Zero Leakage).
- Melakukan pelatihan lokal dengan ClassWeightedMSELoss dan AdamW + CosineAnnealingLR.
- Mengembalikan bobot model terupdate (ArrayRecord) dan MetricRecord yang aman (num-examples, train_loss).
- Menjamin privasi: Tidak ada raw audio/video, transkrip, feature tensor, atau ID partisipan yang dikirimkan.
"""

import logging
import math
from typing import Any, Dict
import numpy as np
import torch
from flwr.app.message.arrayrecord import ArrayRecord
from flwr.app.message.context import Context
from flwr.app.message.message import Message
from flwr.app.message.metricrecord import MetricRecord
from flwr.app.message.recorddict import RecordDict
from flwr.client import ClientApp

from federated.data import build_client_dataloaders
from federated.task import build_model, evaluate_model, train_local

logger = logging.getLogger("fedmind.client")
app = ClientApp()


@app.train()
def train(message: Message, context: Context) -> Message:
    """
    Handler pelatihan lokal klien berbasis Flower 1.40 Message API.

    Kontrak Privasi Klien -> Server:
    TERKIRIM (Transmitted):
      - Bobot parameter model PyTorch (ArrayRecord)
      - Bobot agregasi jumlah sampel (num-examples)
      - Metrik skalar pelatihan & validasi agregat aman (MetricRecord)
    TIDAK TERKIRIM (Not Transmitted):
      - Raw audio, raw video, transkrip teks
      - Feature tensor multimodal pasien
      - ID partisipan / identitas klinis
      - Prediksi per-pasien individual
      - Individual gradient traces
      - Partition identifiers / metadata partisi lokal
    """
    # 1. Ekstraksi bobot model global dari ArrayRecord
    content = message.content
    if "arrays" not in content:
        raise KeyError("Pesan dari server tidak memuat 'arrays' (ArrayRecord).")

    arrays: ArrayRecord = content["arrays"]
    global_state_dict = arrays.to_torch_state_dict()

    # 2. Instansiasi MultimodalBaselineModel dan load parameter secara ketat
    model = build_model()
    model.load_state_dict(global_state_dict, strict=True)

    # 3. Identifikasi partisi lokal klien (0, 1, 2)
    node_cfg = context.node_config
    partition_id = int(node_cfg.get("partition-id", 0))

    # 4. Baca hyperparameter dari run_config atau content["config"]
    run_cfg = context.run_config
    local_epochs = int(run_cfg.get("local-epochs", 2))
    batch_size = int(run_cfg.get("batch-size", 16))
    lr = float(run_cfg.get("learning-rate", 1e-4))
    weight_decay = float(run_cfg.get("weight-decay", 0.01))
    scenario = str(run_cfg.get("scenario", "IID_3Clients"))
    partition_dir = str(run_cfg.get("partition-dir", "")).strip() or None

    # 5. Bangun DataLoader hanya untuk partisi lokal klien ini
    deg_cfg = None
    is_modality = scenario in (
        "non_iid_modality_skew", "modality_skew", "controlled_modality_skew_v1",
        "non_iid_combined", "combined_skew", "controlled_combined_skew_v1",
    )
    if is_modality:
        from federated.modality_degradation import ModalityDegradationConfig
        audio_sigma = float(run_cfg.get("audio-noise-sigma", 0.25))
        visual_dropout = float(run_cfg.get("visual-frame-dropout-rate", 0.20))
        seed_cfg = int(run_cfg.get("seed", 42))
        deg_cfg = ModalityDegradationConfig(
            scenario=scenario,
            audio_noise_sigma=audio_sigma,
            visual_frame_dropout_rate=visual_dropout,
            seed=seed_cfg,
        )

    train_loader, val_loader, train_ds, val_ds = build_client_dataloaders(
        partition_id=partition_id,
        batch_size=batch_size,
        scenario=scenario,
        base_partition_dir=partition_dir,
        degradation_config=deg_cfg,
    )
    num_samples = len(train_ds)

    logger.info(
        f"[Client Node {context.node_id} / Partition {partition_id}] "
        f"Memulai pelatihan lokal: {num_samples} sampel, {local_epochs} epoch, LR={lr}, Skenario={scenario}"
    )

    # 6. Jalankan pelatihan lokal
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avg_loss, trained_samples, train_metrics = train_local(
        model=model,
        train_loader=train_loader,
        epochs=local_epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    # 7. Evaluasi validasi lokal klien
    val_loss = 0.0
    val_metrics = {}
    if len(val_ds) > 0:
        val_loss, val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
        )

    # 8. Hitung client drift (L2 norm perubahan parameter lokal thd bobot global masuk)
    local_sd = model.state_dict()
    drift_sq = 0.0
    for k, v_glob in global_state_dict.items():
        if k in local_sd:
            drift_sq += float(torch.sum((local_sd[k].cpu() - v_glob.cpu()) ** 2).item())
    client_drift_l2 = float(np.sqrt(drift_sq))

    logger.info(
        f"[Client Node {context.node_id} / Partition {partition_id}] "
        f"Pelatihan selesai: train_loss={avg_loss:.4f}, MAE={train_metrics.get('mae', 0.0):.4f}, "
        f"val_loss={val_loss:.4f}, drift_L2={client_drift_l2:.4f}"
    )

    # 9. Konversi bobot model lokal kembali ke ArrayRecord
    model.cpu()
    updated_arrays = ArrayRecord.from_torch_state_dict(model.state_dict())

    # 10. Buat MetricRecord aman (hanya metrik agregat skalar, tanpa data privasi, TANPA partition_id)
    minibatches_per_epoch = len(train_loader)
    optimizer_steps = minibatches_per_epoch * local_epochs
    weighted_l2 = float((num_samples / 163.0) * client_drift_l2)

    metric_record = MetricRecord({
        "num-examples": int(num_samples),
        "local_epochs": int(local_epochs),
        "minibatches_per_epoch": int(minibatches_per_epoch),
        "optimizer_steps": int(optimizer_steps),
        "train_loss": float(avg_loss),
        "train_mae": float(train_metrics.get("mae", 0.0)),
        "train_f1": float(train_metrics.get("macro_f1", 0.0)),
        "val_loss": float(val_loss),
        "val_mae": float(val_metrics.get("mae", 0.0)),
        "val_f1": float(val_metrics.get("macro_f1", 0.0)),
        "val_samples": int(len(val_ds)),
        "client_update_l2_norm": float(client_drift_l2),
        "weighted_update_l2_norm": float(weighted_l2),
    })

    reply_content = RecordDict({
        "arrays": updated_arrays,
        "metrics": metric_record,
    })

    return Message(content=reply_content, reply_to=message)
