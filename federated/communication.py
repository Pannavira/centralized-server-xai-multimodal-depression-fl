"""
federated/communication.py
==========================
Pelacakan beban komunikasi parameter model per ronde federated learning.
Mencatat estimasi payload parameter model secara deterministik (downlink & uplink).

Catatan Metodologis:
Estimasi ini merepresentasikan muatan parameter PyTorch murni (element_size * numel)
dan tidak mengklaim sebagai total lalu lintas jaringan TCP/TLS/gRPC riil (overhead framing).
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch.nn as nn


def calculate_model_parameter_bytes(model: nn.Module) -> int:
    """Menghitung total bytes parameter bobot model PyTorch."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def record_round_communication(
    csv_path: Path,
    server_round: int,
    model: nn.Module,
    participating_clients: int,
) -> Dict[str, Any]:
    """
    Mencatat estimasi komunikasi parameter ke file CSV dan mengembalikan dict metrik.
    Mendukung pelacakan kumulatif per ronde.
    """
    param_bytes = calculate_model_parameter_bytes(model)
    downlink_bytes = param_bytes * participating_clients
    uplink_bytes = param_bytes * participating_clients
    total_bytes = downlink_bytes + uplink_bytes

    # Hitung kumulatif payload dari histori sebelumnya jika ada
    prev_cumulative_bytes = 0
    if csv_path.exists():
        try:
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "cumulative_parameter_bytes" in row and row["cumulative_parameter_bytes"]:
                        prev_cumulative_bytes = int(row["cumulative_parameter_bytes"])
                    elif "estimated_total_parameter_bytes" in row and row["estimated_total_parameter_bytes"]:
                        prev_cumulative_bytes += int(row["estimated_total_parameter_bytes"])
        except Exception:
            prev_cumulative_bytes = 0

    cumulative_bytes = prev_cumulative_bytes + total_bytes

    record = {
        "round": server_round,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "participating_clients": participating_clients,
        "model_parameter_bytes": param_bytes,
        "model_parameter_mb": round(param_bytes / (1024 * 1024), 4),
        "estimated_downlink_parameter_bytes": downlink_bytes,
        "estimated_downlink_mb": round(downlink_bytes / (1024 * 1024), 4),
        "estimated_uplink_parameter_bytes": uplink_bytes,
        "estimated_uplink_mb": round(uplink_bytes / (1024 * 1024), 4),
        "estimated_total_parameter_bytes": total_bytes,
        "estimated_total_mb": round(total_bytes / (1024 * 1024), 4),
        "cumulative_parameter_bytes": cumulative_bytes,
        "cumulative_mb": round(cumulative_bytes / (1024 * 1024), 4),
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    fieldnames = list(record.keys())
    existing_rounds = set()
    if file_exists:
        try:
            with open(csv_path, mode="r", encoding="utf-8") as rf:
                rdr = csv.DictReader(rf)
                for row in rdr:
                    if "round" in row:
                        existing_rounds.add(int(row["round"]))
        except Exception:
            pass

    if server_round not in existing_rounds:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    return record
