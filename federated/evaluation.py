"""
federated/evaluation.py
=======================
Evaluasi Terpusat Server (Centralized Global Hold-Out Evaluation).

Mengevaluasi performa model global secara independen pada dataset uji terpusat
(global_test_index.csv) yang terisolasi total dari partisi training klien lokal.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from flwr.app.message.arrayrecord import ArrayRecord

from app.core.model import MultimodalBaselineModel
from federated.data import build_global_test_dataloader
from federated.metrics import calculate_clinical_metrics
from federated.task import build_model, evaluate_model


class GlobalEvaluator:
    """Evaluator terpusat server untuk menguji model global pada setiap ronde."""

    def __init__(
        self,
        test_index_path: Optional[Union[Path, str]] = None,
        feature_base_dir: Optional[Union[Path, str]] = None,
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.test_loader, self.test_ds = build_global_test_dataloader(
            test_index_path=test_index_path,
            feature_base_dir=feature_base_dir,
            batch_size=batch_size,
        )

    def evaluate_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        threshold: float = 10.0,
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Mengevaluasi state_dict model global PyTorch."""
        eval_device = device if device is not None else self.device
        model = build_model()
        model.load_state_dict(state_dict, strict=True)
        avg_loss, metrics = evaluate_model(
            model=model,
            dataloader=self.test_loader,
            device=eval_device,
            threshold=threshold,
        )
        metrics["test_loss"] = round(avg_loss, 4)
        return metrics

    def evaluate_array_record(
        self,
        array_record: ArrayRecord,
        threshold: float = 10.0,
        device: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Mengevaluasi ArrayRecord dari Flower Message API."""
        state_dict = array_record.to_torch_state_dict()
        return self.evaluate_state_dict(state_dict=state_dict, threshold=threshold, device=device)


def evaluate_global_model(
    model_or_state: Union[nn.Module, Dict[str, torch.Tensor], ArrayRecord],
    test_index_path: Optional[Union[Path, str]] = None,
    feature_base_dir: Optional[Union[Path, str]] = None,
    device: str = "cpu",
    threshold: float = 10.0,
) -> Dict[str, Any]:
    """Fungsi pembantu mandiri untuk evaluasi global test set."""
    evaluator = GlobalEvaluator(
        test_index_path=test_index_path,
        feature_base_dir=feature_base_dir,
        device=device,
    )
    if isinstance(model_or_state, ArrayRecord):
        return evaluator.evaluate_array_record(model_or_state, threshold=threshold)
    elif isinstance(model_or_state, dict):
        return evaluator.evaluate_state_dict(model_or_state, threshold=threshold)
    elif isinstance(model_or_state, nn.Module):
        avg_loss, metrics = evaluate_model(
            model=model_or_state,
            dataloader=evaluator.test_loader,
            device=device,
            threshold=threshold,
        )
        metrics["test_loss"] = round(avg_loss, 4)
        return metrics
    else:
        raise TypeError(f"Tipe model tidak didukung: {type(model_or_state)}")


def compare_with_historical_reference(
    reproduction_state_dict: Dict[str, torch.Tensor],
    reference_checkpoint_path: Optional[Union[Path, str]] = None,
    evaluator: Optional[GlobalEvaluator] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Membandingkan model hasil reproduksi dengan checkpoint referensi historis.
    Menghitung metrik perbandingan, absolute delta, korelasi prediksi kohort,
    dan kompatibilitas arsitektur.
    """
    import hashlib
    from federated.metrics import calculate_clinical_metrics

    ref_path = Path(
        reference_checkpoint_path
        or "C:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl/federated_learning_output/experiments_v2/fedavg_iid_3clients_pretrained/best_global_model.pt"
    )

    if not ref_path.exists():
        # Fallback ke copy storage/models/fl_iid/model.pt
        ref_path = Path("storage/models/fl_iid/model.pt")

    if not ref_path.exists():
        return {"status": "REFERENCE_NOT_FOUND", "message": f"Checkpoint referensi tidak ditemukan di: {ref_path}"}

    # 1. Hitung SHA-256 file referensi
    sha_ref = hashlib.sha256()
    with open(ref_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha_ref.update(chunk)
    ref_sha256 = sha_ref.hexdigest()

    # 2. Muat checkpoint referensi
    ref_ckpt = torch.load(ref_path, map_location="cpu", weights_only=False)
    ref_sd = ref_ckpt.get("state_dict", ref_ckpt.get("model_state_dict", ref_ckpt))

    # 3. Evaluasi kedua model pada global test set
    if evaluator is None:
        evaluator = GlobalEvaluator(device=device)

    # Prediksi model reproduksi
    m_repro = build_model()
    m_repro.load_state_dict(reproduction_state_dict, strict=True)
    m_repro.to(device)
    m_repro.eval()

    # Prediksi model referensi
    m_ref = build_model()
    m_ref.load_state_dict(ref_sd, strict=True)
    m_ref.to(device)
    m_ref.eval()

    all_targets = []
    preds_repro = []
    preds_ref = []

    with torch.no_grad():
        for batch in evaluator.test_loader:
            audio = batch["audio"].to(device)
            visual = batch["visual"].to(device)
            text = batch["text"].to(device)
            target = batch["phq8_score"].to(device)

            p_repro = m_repro(audio, visual, text).cpu().flatten().tolist()
            p_ref = m_ref(audio, visual, text).cpu().flatten().tolist()
            t = target.cpu().flatten().tolist()

            preds_repro.extend(p_repro)
            preds_ref.extend(p_ref)
            all_targets.extend(t)

    metrics_repro = calculate_clinical_metrics(all_targets, preds_repro, threshold=10.0)
    metrics_ref = calculate_clinical_metrics(all_targets, preds_ref, threshold=10.0)

    # 4. Hitung korelasi prediksi kohort & mean absolute difference
    arr_repro = np.array(preds_repro)
    arr_ref = np.array(preds_ref)

    if np.std(arr_repro) > 1e-8 and np.std(arr_ref) > 1e-8:
        pred_corr = float(np.corrcoef(arr_repro, arr_ref)[0, 1])
    else:
        pred_corr = 0.0

    mean_abs_diff = float(np.mean(np.abs(arr_repro - arr_ref)))

    # 5. Tabel Metrik Komparasi & Absolute Delta
    comparison_table = {}
    metric_keys = ["mae", "rmse", "mse", "macro_f1", "recall", "specificity", "auc_roc", "ccc", "pearson_r"]
    for k in metric_keys:
        v_ref = metrics_ref.get(k)
        v_rep = metrics_repro.get(k)
        delta = None
        if v_ref is not None and v_rep is not None:
            delta = round(abs(float(v_rep) - float(v_ref)), 4)
        comparison_table[k] = {
            "historical": round(float(v_ref), 4) if v_ref is not None else None,
            "reproduction": round(float(v_rep), 4) if v_rep is not None else None,
            "absolute_delta": delta,
        }

    return {
        "reference_checkpoint_path": str(ref_path),
        "reference_sha256": ref_sha256,
        "cohort_size": len(all_targets),
        "prediction_correlation": round(pred_corr, 4),
        "mean_absolute_prediction_diff": round(mean_abs_diff, 4),
        "metrics_comparison": comparison_table,
    }
