"""
federated/metrics.py
====================
Fungsi komputasi metrik evaluasi klinis standar FED-MIND (PHQ-8 regresi dan triase biner).
Mencakup:
- Metrik Regresi: MAE, RMSE, MSE, CCC (Lin's Concordance Correlation Coefficient), Pearson r.
- Metrik Klasifikasi Biner (Cut-off PHQ-8 >= 10.0): Accuracy, Precision, Recall / Sensitivity, Specificity, F1-Score (Macro), F2-Score, ROC-AUC.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Menghitung Lin's Concordance Correlation Coefficient (CCC).
    Formula: 2 * s_xy / (s_x^2 + s_y^2 + (m_x - m_y)^2)
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_true) < 2:
        return 0.0

    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))

    var_true = float(np.var(y_true, ddof=0))
    var_pred = float(np.var(y_pred, ddof=0))

    covariance = float(np.mean((y_true - mean_true) * (y_pred - mean_pred)))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2 + eps

    if denom <= 0.0 or np.isnan(denom):
        return 0.0

    ccc = (2.0 * covariance) / denom
    return float(np.clip(ccc, -1.0, 1.0))


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Menghitung Pearson Correlation Coefficient r."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    corr_matrix = np.corrcoef(y_true, y_pred)
    val = corr_matrix[0, 1]
    return float(val) if not np.isnan(val) else 0.0


def calculate_clinical_metrics(
    y_true_cont: Union[np.ndarray, list],
    y_pred_cont: Union[np.ndarray, list],
    threshold: float = 10.0,
) -> Dict[str, Any]:
    """
    Menghitung metrik komprehensif klinis untuk evaluasi model terpusat server.
    """
    y_true_cont = np.asarray(y_true_cont, dtype=np.float64).flatten()
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float64).flatten()

    if len(y_true_cont) == 0:
        return {}

    # 1. Metrik Regresi Kontinu
    mse = float(mean_squared_error(y_true_cont, y_pred_cont))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_cont, y_pred_cont))
    ccc = compute_ccc(y_true_cont, y_pred_cont)
    pearson_r = compute_pearson(y_true_cont, y_pred_cont)

    # 2. Binerisasi untuk Triase Klinis
    y_true_bin = (y_true_cont >= threshold).astype(int)
    y_pred_bin = (y_pred_cont >= threshold).astype(int)

    acc = float(accuracy_score(y_true_bin, y_pred_bin))
    prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    recall = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    macro_f1 = float(f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0))
    binary_f1 = float(f1_score(y_true_bin, y_pred_bin, average="binary", zero_division=0))
    f2 = float(fbeta_score(y_true_bin, y_pred_bin, beta=2.0, zero_division=0))

    # Confusion Matrix: tn, fp, fn, tp
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC (Hanya dihitung jika terdapat minimal 2 kelas pada ground truth)
    auc_roc: Optional[float] = None
    if len(np.unique(y_true_bin)) > 1:
        try:
            auc_roc = float(roc_auc_score(y_true_bin, y_pred_cont))
        except Exception:
            auc_roc = None

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mse": round(mse, 4),
        "ccc": round(ccc, 4),
        "pearson_r": round(pearson_r, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(recall, 4),
        "macro_f1": round(macro_f1, 4),
        "binary_f1": round(binary_f1, 4),
        "f2_score": round(f2, 4),
        "specificity": round(specificity, 4),
        "auc_roc": round(auc_roc, 4) if auc_roc is not None else None,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "false_negative_rate": round(float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0, 4),
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0, 4),
        "total_samples": int(len(y_true_cont)),
    }

