"""
federated/task.py
=================
Authoritative Model Factory, Loss Functions, dan Prosedur Pelatihan/Evaluasi Lokal.

Prinsip Utama:
- Menggunakan 100% arsitektur MultimodalBaselineModel resmi dari app.core.model.
- Mengimplementasikan ClassWeightedMSELoss persis dengan metodologi riset FED-MIND.
- Memvalidasi finiteness loss (mencegah NaN/Inf).
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from app.core.model import MultimodalBaselineModel
from federated.metrics import calculate_clinical_metrics


class ClassWeightedMSELoss(nn.Module):
    """
    Class-Weighted Mean Squared Error Loss dengan bobot frekuensi kelas dan penalti False Negative.
    Sesuai dengan protokol penelitian FED-MIND (Model_Architecture/loss.py).
    """

    def __init__(
        self,
        binarize_threshold: float = 10.0,
        pos_weight_multiplier: float = 2.5,
        fn_penalty: float = 2.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.binarize_threshold = float(binarize_threshold)
        self.pos_weight_multiplier = float(pos_weight_multiplier)
        self.fn_penalty = float(fn_penalty)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        y_bin = (target_flat >= self.binarize_threshold).float()

        n_pos = torch.sum(y_bin)
        n_neg = torch.sum(1.0 - y_bin)

        device = pred.device
        w_pos = (1.0 / (n_pos + self.eps)) * self.pos_weight_multiplier if n_pos > 0 else torch.tensor(1.0, device=device)
        w_neg = (1.0 / (n_neg + self.eps)) if n_neg > 0 else torch.tensor(1.0, device=device)

        weights = torch.where(y_bin == 1.0, w_pos, w_neg)

        # Penalti False Negative: sampel depresi (y_bin=1) yang diprediksi di bawah threshold
        if self.fn_penalty > 1.0:
            is_fn = (y_bin == 1.0) & (pred_flat < self.binarize_threshold)
            weights = torch.where(is_fn, weights * self.fn_penalty, weights)

        # Normalisasi bobot agar rata-rata = 1.0
        weights = weights / (weights.mean() + self.eps)

        sq_err = (pred_flat - target_flat) ** 2
        weighted_loss = torch.mean(weights * sq_err)
        return weighted_loss


def build_model(seed: Optional[int] = None) -> MultimodalBaselineModel:
    """
    Authoritative factory untuk instansiasi MultimodalBaselineModel.
    Dapat diinisialisasi secara deterministik jika seed diberikan.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    model = MultimodalBaselineModel()
    return model


def train_local(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    device: str = "cpu",
    threshold: float = 10.0,
    pos_weight_multiplier: float = 2.5,
    fn_penalty: float = 2.0,
    learning_rate: Optional[float] = None,
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Menjalankan pelatihan lokal selama `epochs` pada satu klien.
    Memvalidasi finiteness loss (mencegah NaN / Inf).
    """
    if learning_rate is not None:
        lr = learning_rate

    model.to(device)
    model.train()

    criterion = ClassWeightedMSELoss(
        binarize_threshold=threshold,
        pos_weight_multiplier=pos_weight_multiplier,
        fn_penalty=fn_penalty,
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-7)

    total_loss_accum = 0.0
    total_batches = 0
    all_preds: List[float] = []
    all_targets: List[float] = []
    total_samples = 0

    for epoch in range(epochs):
        for batch in train_loader:
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            text = batch["text"].to(device)
            phq8_true = batch["phq8_score"].to(device)
            batch_sz = phq8_true.size(0)

            optimizer.zero_grad()
            pred_phq8 = model(audio, visual, text)

            loss = criterion(pred_phq8, phq8_true)

            loss_val = float(loss.item())
            if not math.isfinite(loss_val):
                raise ValueError(f"CRITICAL ERROR: Loss tidak berhingga (NaN / Inf terdeteksi): {loss_val}")

            loss.backward()

            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            total_loss_accum += loss_val
            total_batches += 1
            total_samples += batch_sz

            all_preds.extend(pred_phq8.detach().cpu().flatten().tolist())
            all_targets.extend(phq8_true.detach().cpu().flatten().tolist())

        scheduler.step()

    avg_loss = total_loss_accum / max(total_batches, 1)
    metrics = calculate_clinical_metrics(all_targets, all_preds, threshold=threshold)

    return avg_loss, len(train_loader.dataset), metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    threshold: float = 10.0,
    pos_weight_multiplier: float = 2.5,
    fn_penalty: float = 2.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Mengevaluasi model pada dataset evaluasi terpusat atau validasi lokal.
    """
    model.to(device)
    model.eval()

    criterion = ClassWeightedMSELoss(
        binarize_threshold=threshold,
        pos_weight_multiplier=pos_weight_multiplier,
        fn_penalty=fn_penalty,
    )

    total_loss = 0.0
    total_batches = 0
    all_preds: List[float] = []
    all_targets: List[float] = []

    with torch.no_grad():
        for batch in dataloader:
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            text = batch["text"].to(device)
            phq8_true = batch["phq8_score"].to(device)

            pred_phq8 = model(audio, visual, text)
            loss = criterion(pred_phq8, phq8_true)

            total_loss += float(loss.item())
            total_batches += 1

            all_preds.extend(pred_phq8.cpu().flatten().tolist())
            all_targets.extend(phq8_true.cpu().flatten().tolist())

    avg_loss = total_loss / max(total_batches, 1)
    metrics = calculate_clinical_metrics(all_targets, all_preds, threshold=threshold)

    return avg_loss, metrics
