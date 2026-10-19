"""
federated/checkpointing.py
==========================
Manajemen persistensi checkpoint, artefak metrik, dan manifest eksperimen Federated Learning.

Struktur Direktori:
storage/federated/runs/<run_id>/
├── config.json
├── manifest.json
├── metrics.csv
├── communication.csv
└── rounds/
    ├── round_000/
    │   ├── model.pt
    │   └── metadata.json
    └── round_001/
        ├── model.pt
        └── metadata.json
"""

import csv
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch


def generate_run_id(prefix: str = "FL-RUN") -> str:
    """Menghasilkan ID eksekusi FL unik berbasis waktu dan UUID."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:6].upper()
    return f"{prefix}-{ts}-{short_uuid}"


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Menghitung hash SHA-256 berkas biner."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class FLRunManager:
    """Mengelola artefak, penyimpanan checkpoint, dan logging metrik untuk satu sesi FL."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        base_dir: Optional[Path] = None,
        prefix: str = "FL-RUN",
    ):
        self.run_id = run_id or generate_run_id(prefix=prefix)
        if base_dir is None:
            workspace_root = Path("C:/Wira/Kuliah/Penelitian/PROTOTYPE APLIKASI FL/centralized-server-xai-multimodal-depression-fl")
            if workspace_root.exists():
                base_dir = workspace_root / "storage" / "federated" / "runs"
            else:
                repo_root = Path(__file__).resolve().parent.parent
                base_dir = repo_root / "storage" / "federated" / "runs"
        self.run_dir = base_dir / self.run_id
        self.rounds_dir = self.run_dir / "rounds"
        self.plots_dir = self.run_dir / "plots"
        self.metrics_csv_path = self.run_dir / "metrics.csv"
        self.client_metrics_csv_path = self.run_dir / "client_metrics.csv"
        self.comm_csv_path = self.run_dir / "communication.csv"

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.rounds_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config_dict: Dict[str, Any]) -> Path:
        """Menyimpan konfigurasi run ke config.json."""
        config_path = self.run_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, default=str)
        return config_path

    def save_environment(self, env_dict: Dict[str, Any]) -> Path:
        """Menyimpan metadata lingkungan komputasi ke environment.json."""
        env_path = self.run_dir / "environment.json"
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(env_dict, f, indent=2, default=str)
        return env_path

    def save_dataset_manifest(self, dataset_manifest_dict: Dict[str, Any]) -> Path:
        """Menyimpan manifest integritas dataset ke dataset_manifest.json."""
        manifest_path = self.run_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(dataset_manifest_dict, f, indent=2, default=str)
        return manifest_path

    def save_reference_comparison(self, comp_dict: Dict[str, Any]) -> Path:
        """Menyimpan perbandingan dengan model referensi ke reference_comparison.json."""
        comp_path = self.run_dir / "reference_comparison.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(comp_dict, f, indent=2, default=str)
        return comp_path

    def save_manifest(self, manifest_dict: Dict[str, Any]) -> Path:
        """Menyimpan ringkasan manifest run ke manifest.json."""
        manifest_path = self.run_dir / "manifest.json"
        manifest_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, indent=2, default=str)
        return manifest_path

    def save_round_checkpoint(
        self,
        server_round: int,
        state_dict: Dict[str, Any],
        metadata: Dict[str, Any],
        source_checkpoint_to_copy: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, str]:
        """
        Menyimpan checkpoint model untuk ronde tertentu dan metadata JSON.
        Jika source_checkpoint_to_copy diberikan dan ada, berkas disalin langsung
        untuk memastikan checksum SHA-256 bitwise 100% identik dengan sumbernya.
        Mengembalikan (checkpoint_path, sha256_hash).
        """
        round_dir = self.rounds_dir / f"round_{server_round:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        model_path = round_dir / "model.pt"
        if source_checkpoint_to_copy and Path(source_checkpoint_to_copy).exists():
            import shutil
            shutil.copy2(source_checkpoint_to_copy, model_path)
        else:
            torch.save(state_dict, model_path)

        sha256 = compute_file_sha256(model_path)

        metadata_to_save = dict(metadata)
        metadata_to_save.update({
            "run_id": self.run_id,
            "round": server_round,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_sha256": sha256,
        })

        meta_path = round_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_to_save, f, indent=2, default=str)

        return model_path, sha256


    def load_round_checkpoint(self, server_round: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Memuat checkpoint state_dict dan metadata dari ronde tertentu untuk resume."""
        round_dir = self.rounds_dir / f"round_{server_round:03d}"
        model_path = round_dir / "model.pt"
        meta_path = round_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint model round {server_round} tidak ditemukan di: {model_path}")

        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        return state_dict, metadata

    def get_latest_completed_round(self) -> Optional[int]:
        """Mencari nomor ronde terakhir yang telah tersimpan lengkap dengan model.pt."""
        if not self.rounds_dir.exists():
            return None
        existing_rounds = []
        for rd in self.rounds_dir.glob("round_*"):
            if (rd / "model.pt").exists():
                try:
                    r_num = int(rd.name.split("_")[-1])
                    existing_rounds.append(r_num)
                except ValueError:
                    continue
        return max(existing_rounds) if existing_rounds else None

    def record_metrics(self, server_round: int, metrics: Dict[str, Any]) -> None:
        """Mencatat metrik evaluasi ronde ke metrics.csv tanpa duplikasi ronde."""
        record = {
            "round": server_round,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        record.update(metrics)

        file_exists = self.metrics_csv_path.exists()
        fieldnames = list(record.keys())

        # Cegah duplikasi baris jika resume
        existing_rounds = set()
        if file_exists:
            try:
                with open(self.metrics_csv_path, mode="r", encoding="utf-8") as rf:
                    rdr = csv.DictReader(rf)
                    for row in rdr:
                        if "round" in row:
                            existing_rounds.add(int(row["round"]))
            except Exception:
                pass

        if server_round not in existing_rounds:
            with open(self.metrics_csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)

    def record_client_metrics(
        self,
        server_round: int,
        client_id: str,
        num_examples: int,
        metrics: Dict[str, Any],
    ) -> None:
        """Mencatat metrik pelatihan dan validasi klien per ronde ke client_metrics.csv."""
        record = {
            "round": server_round,
            "client_id": str(client_id),
            "num_examples": int(num_examples),
            "local_epochs": int(metrics.get("local_epochs", 2)),
            "minibatches_per_epoch": int(metrics.get("minibatches_per_epoch", 0)),
            "optimizer_steps": int(metrics.get("optimizer_steps", 0)),
            "train_loss": float(metrics.get("train_loss", 0.0)),
            "train_mae": float(metrics.get("train_mae", 0.0)),
            "train_f1": float(metrics.get("train_f1", 0.0)),
            "val_loss": float(metrics.get("val_loss", 0.0)),
            "val_mae": float(metrics.get("val_mae", 0.0)),
            "val_f1": float(metrics.get("val_f1", 0.0)),
            "client_update_l2_norm": float(metrics.get("client_drift_l2", metrics.get("client_update_l2_norm", 0.0))),
            "weighted_update_l2_norm": float(metrics.get("weighted_update_l2_norm", (num_examples / 163.0) * float(metrics.get("client_drift_l2", metrics.get("client_update_l2_norm", 0.0))))),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        file_exists = self.client_metrics_csv_path.exists()
        fieldnames = list(record.keys())

        with open(self.client_metrics_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

    def generate_plots(self) -> Dict[str, Path]:
        """
        Membuat kurva konvergensi visual eksperimen:
        1. global_loss.png
        2. mae_rmse.png
        3. classification_metrics.png
        4. communication_cumulative.png
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        plots = {}
        if not self.metrics_csv_path.exists():
            return plots

        try:
            df_m = pd.read_csv(self.metrics_csv_path)
            if len(df_m) == 0:
                return plots

            rounds = df_m["round"]

            # 1. Global Loss
            if "test_loss" in df_m or "loss" in df_m:
                loss_col = "test_loss" if "test_loss" in df_m else "loss"
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(rounds, df_m[loss_col], marker="o", color="#1f77b4", label="Global Test Loss")
                ax.set_title("FED-MIND FL-IID: Global Loss per Round")
                ax.set_xlabel("Communication Round")
                ax.set_ylabel("Loss (ClassWeightedMSE)")
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend()
                loss_plot = self.plots_dir / "global_loss.png"
                fig.tight_layout()
                fig.savefig(loss_plot, dpi=150)
                plt.close(fig)
                plots["global_loss"] = loss_plot

            # 2. MAE & RMSE
            if "mae" in df_m and "rmse" in df_m:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(rounds, df_m["mae"], marker="s", color="#ff7f0e", label="MAE")
                ax.plot(rounds, df_m["rmse"], marker="^", color="#2ca02c", label="RMSE")
                ax.set_title("FED-MIND FL-IID: Regression Errors (MAE & RMSE)")
                ax.set_xlabel("Communication Round")
                ax.set_ylabel("Error Points")
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend()
                mae_plot = self.plots_dir / "mae_rmse.png"
                fig.tight_layout()
                fig.savefig(mae_plot, dpi=150)
                plt.close(fig)
                plots["mae_rmse"] = mae_plot

            # 3. Classification Metrics (Macro F1, Recall, Specificity)
            fig, ax = plt.subplots(figsize=(8, 5))
            plotted_any = False
            if "macro_f1" in df_m:
                ax.plot(rounds, df_m["macro_f1"], marker="o", color="#d62728", label="Macro F1")
                plotted_any = True
            if "recall" in df_m:
                ax.plot(rounds, df_m["recall"], marker="x", color="#9467bd", label="Recall / Sens.")
                plotted_any = True
            if "specificity" in df_m:
                ax.plot(rounds, df_m["specificity"], marker="d", color="#8c564b", label="Specificity")
                plotted_any = True

            if plotted_any:
                ax.set_title("FED-MIND FL-IID: Clinical Detection Metrics")
                ax.set_xlabel("Communication Round")
                ax.set_ylabel("Score (0.0 - 1.0)")
                ax.set_ylim(-0.05, 1.05)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend()
                cls_plot = self.plots_dir / "classification_metrics.png"
                fig.tight_layout()
                fig.savefig(cls_plot, dpi=150)
                plt.close(fig)
                plots["classification_metrics"] = cls_plot

            # 4. Cumulative Communication
            if self.comm_csv_path.exists():
                df_c = pd.read_csv(self.comm_csv_path)
                if "cumulative_mb" in df_c and len(df_c) > 0:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(df_c["round"], df_c["cumulative_mb"], marker="o", color="#17becf", label="Cumulative Payload (MB)")
                    ax.set_title("FED-MIND FL-IID: Cumulative Parameter Communication")
                    ax.set_xlabel("Communication Round")
                    ax.set_ylabel("Payload Volume (MB)")
                    ax.grid(True, linestyle="--", alpha=0.6)
                    ax.legend()
                    comm_plot = self.plots_dir / "communication_cumulative.png"
                    fig.tight_layout()
                    fig.savefig(comm_plot, dpi=150)
                    plt.close(fig)
                    plots["communication_cumulative"] = comm_plot

        except Exception as e:
            pass

        return plots
