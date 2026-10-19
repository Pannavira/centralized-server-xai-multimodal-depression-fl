"""
federated/data.py
=================
Adapter Dataset dan DataLoader Klien & Server Global Test untuk Federated Learning.

Membaca partisi resmi penelitian:
- IID Partisi: clinical-xai-multimodal-depression-fl/federated_learning_output/IID_3Clients/
- Fitur Multimodal: clinical-xai-multimodal-depression-fl/modeling/01_Input_From_Partition/
- Global Hold-Out Test: clinical-xai-multimodal-depression-fl/federated_learning_output/global_test_index.csv

Prinsip Privasi & Zero-Leakage:
- Dataset klien hanya memuat partisi lokal miliknya.
- Verifikasi otomatis membuktikan tidak ada tumpang tindih antara data training klien dan test set global.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from app.core.preprocessing import adapt_to_sequence_shape

# Default Path ke Repositori Riset Acuan
DEFAULT_RESEARCH_DIR = Path("C:/Wira/Kuliah/Penelitian/clinical-xai-multimodal-depression-fl")
DEFAULT_PARTITION_DIR = DEFAULT_RESEARCH_DIR / "federated_learning_output"
DEFAULT_FEATURE_DIR = DEFAULT_RESEARCH_DIR / "modeling" / "01_Input_From_Partition"


def get_default_norm_stats_path() -> Path:
    """Mengembalikan path file normalisasi Z-score."""
    repo_root = Path(__file__).resolve().parent.parent
    local_path = repo_root / "storage" / "models" / "train_norm_stats.pt"
    if local_path.exists():
        return local_path
    return DEFAULT_RESEARCH_DIR / "storage" / "models" / "train_norm_stats.pt"


class MultimodalPartitionDataset(Dataset):
    """
    Dataset PyTorch untuk memuat fitur multimodal (Audio, Visual, Text) dan skor PHQ-8.
    """

    TARGET_SHAPES = {
        "audio": (512, 80),
        "visual": (512, 72, 3),
        "text": (512, 768),
    }

    def __init__(
        self,
        index_df_or_path: Union[pd.DataFrame, Path, str],
        feature_base_dir: Optional[Union[Path, str]] = None,
        norm_stats_path: Optional[Union[Path, str]] = None,
    ) -> None:
        if isinstance(index_df_or_path, (str, Path)):
            self.df = pd.read_csv(index_df_or_path)
        else:
            self.df = index_df_or_path.copy().reset_index(drop=True)

        self.feature_base_dir = Path(feature_base_dir or DEFAULT_FEATURE_DIR)

        # Muat normalisasi Z-score jika tersedia
        self.norm_stats: Dict[str, Dict[str, np.ndarray]] = {}
        target_norm_path = Path(norm_stats_path) if norm_stats_path else get_default_norm_stats_path()
        if target_norm_path.exists():
            try:
                stats = torch.load(target_norm_path, map_location="cpu")
                if isinstance(stats, dict):
                    self.norm_stats = stats
            except Exception:
                self.norm_stats = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_raw_array(self, rel_path_str: str) -> Optional[np.ndarray]:
        """Memuat array numpy dari path relatif."""
        if not rel_path_str or pd.isna(rel_path_str):
            return None
        # Handle format path Windows/Linux
        clean_path = str(rel_path_str).replace("\\", "/")
        full_path = self.feature_base_dir / clean_path
        if not full_path.exists():
            # Coba alternatif jika path di index hanya nama file
            filename = Path(clean_path).name
            for sub in ["final_audio_features", "final_visual_features", "final_text_features"]:
                cand = self.feature_base_dir / sub / filename
                if cand.exists():
                    full_path = cand
                    break

        if full_path.exists():
            try:
                return np.load(full_path)
            except Exception:
                return None
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        pid = str(row["participant_id"])
        phq8 = float(row.get("phq8_score", row.get("depression_label", 0.0)))
        dep_bin = int(row.get("depression_label", 1 if phq8 >= 10.0 else 0))

        item: Dict[str, Any] = {
            "participant_id": pid,
            "phq8_score": torch.tensor(phq8, dtype=torch.float32),
            "depression_label": torch.tensor(dep_bin, dtype=torch.long),
        }

        # 1. Audio
        audio_raw = self._load_raw_array(row.get("audio_feature_path", f"final_audio_features/{pid}.npy"))
        if audio_raw is not None:
            if "audio" in self.norm_stats:
                mean = self.norm_stats["audio"].get("mean")
                std = self.norm_stats["audio"].get("std")
                if mean is not None and std is not None:
                    audio_raw = (audio_raw - mean) / (std + 1e-8)
            audio_adapted = adapt_to_sequence_shape(audio_raw, modality="audio", target_length=512)
        else:
            audio_adapted = np.zeros(self.TARGET_SHAPES["audio"], dtype=np.float32)

        # 2. Visual
        visual_raw = self._load_raw_array(row.get("visual_feature_path", f"final_visual_features/{pid}.npy"))
        if visual_raw is not None:
            if "visual" in self.norm_stats:
                mean = self.norm_stats["visual"].get("mean")
                std = self.norm_stats["visual"].get("std")
                if mean is not None and std is not None:
                    visual_raw = (visual_raw - mean) / (std + 1e-8)
            visual_adapted = adapt_to_sequence_shape(visual_raw, modality="visual", target_length=512)
        else:
            visual_adapted = np.zeros(self.TARGET_SHAPES["visual"], dtype=np.float32)

        # 3. Text
        text_raw = self._load_raw_array(row.get("text_feature_path", f"final_text_features/{pid}.npy"))
        if text_raw is not None:
            if "text" in self.norm_stats:
                mean = self.norm_stats["text"].get("mean")
                std = self.norm_stats["text"].get("std")
                if mean is not None and std is not None:
                    text_raw = (text_raw - mean) / (std + 1e-8)
            text_adapted = adapt_to_sequence_shape(text_raw, modality="text", target_length=512)
        else:
            text_adapted = np.zeros(self.TARGET_SHAPES["text"], dtype=np.float32)

        item["audio"] = torch.from_numpy(audio_adapted.astype(np.float32))
        item["visual"] = torch.from_numpy(visual_adapted.astype(np.float32))
        item["text"] = torch.from_numpy(text_adapted.astype(np.float32))

        return item


def verify_zero_leakage(
    client_training_dfs: List[pd.DataFrame],
    global_test_df: pd.DataFrame,
) -> Tuple[bool, Set[str]]:
    """
    Memverifikasi secara otomatis bahwa tidak ada satupun subjek pada training partition klien
    yang bocor ke dataset evaluasi global test terpusat server.
    """
    test_ids = set(global_test_df["participant_id"].astype(str))
    all_client_train_ids: Set[str] = set()

    for df in client_training_dfs:
        all_client_train_ids.update(df["participant_id"].astype(str))

    leakage = all_client_train_ids.intersection(test_ids)
    if leakage:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! {len(leakage)} participant IDs ditemukan pada training partition dan test set: {leakage}"
        )

    return True, leakage


def get_client_partition_paths(
    partition_id: int,
    scenario: str = "IID_3Clients",
    base_partition_dir: Optional[Union[Path, str]] = None,
) -> Tuple[Path, Path]:
    """
    Mendapatkan path file index train dan val untuk client partition_id (0-indexed: 0, 1, 2).
    """
    if base_partition_dir is not None:
        base_dir = Path(base_partition_dir)
        if (base_dir / scenario).exists():
            base_dir = base_dir / scenario
    elif scenario in ("non_iid_quantity_skew", "quantity_skew", "controlled_quantity_skew_v1"):
        local_qs_dir = Path("storage/federated/partitions/quantity_skew_seed42")
        if local_qs_dir.exists():
            base_dir = local_qs_dir
        else:
            base_dir = Path(DEFAULT_PARTITION_DIR) / scenario
    elif scenario in ("non_iid_label_skew", "label_skew", "controlled_label_skew_v1"):
        local_ls_dir = Path("storage/federated/partitions/label_skew_seed42")
        if local_ls_dir.exists():
            base_dir = local_ls_dir
        else:
            base_dir = Path(DEFAULT_PARTITION_DIR) / scenario
    else:
        base_dir = Path(DEFAULT_PARTITION_DIR) / scenario

    client_num = partition_id + 1  # 0 -> client_1, 1 -> client_2, 2 -> client_3

    train_path = base_dir / f"client_{client_num}_train.csv"
    val_path = base_dir / f"client_{client_num}_val.csv"

    if not val_path.exists():
        # Fallback to shared validation cohort
        shared_val = Path(DEFAULT_PARTITION_DIR) / "IID_3Clients" / f"client_{client_num}_val.csv"
        if shared_val.exists():
            val_path = shared_val

    if not train_path.exists():
        raise FileNotFoundError(f"File partisi training klien tidak ditemukan: {train_path}")

    return train_path, val_path



def build_client_dataloaders(
    partition_id: int,
    batch_size: int = 16,
    scenario: str = "IID_3Clients",
    base_partition_dir: Optional[Union[Path, str]] = None,
    feature_base_dir: Optional[Union[Path, str]] = None,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, MultimodalPartitionDataset, MultimodalPartitionDataset]:
    """
    Membangun DataLoader training dan validation untuk klien tertentu.
    """
    train_path, val_path = get_client_partition_paths(partition_id, scenario, base_partition_dir)

    train_ds = MultimodalPartitionDataset(train_path, feature_base_dir=feature_base_dir)
    val_ds = MultimodalPartitionDataset(val_path, feature_base_dir=feature_base_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, train_ds, val_ds


def build_global_test_dataloader(
    test_index_path: Optional[Union[Path, str]] = None,
    feature_base_dir: Optional[Union[Path, str]] = None,
    batch_size: int = 16,
) -> Tuple[DataLoader, MultimodalPartitionDataset]:
    """
    Membangun DataLoader untuk dataset evaluasi global hold-out test terpusat server.
    """
    if test_index_path is None:
        test_index_path = DEFAULT_PARTITION_DIR / "global_test_index.csv"

    test_path = Path(test_index_path)
    if not test_path.exists():
        raise FileNotFoundError(f"File index global test tidak ditemukan: {test_path}")

    test_ds = MultimodalPartitionDataset(test_path, feature_base_dir=feature_base_dir)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return test_loader, test_ds


def generate_dataset_manifest(
    scenario: str = "IID_3Clients",
    base_partition_dir: Optional[Union[Path, str]] = None,
    test_index_path: Optional[Union[Path, str]] = None,
    num_clients: int = 3,
) -> Dict[str, Any]:
    """
    Membuat manifest komprehensif seluruh file partisi dan dataset evaluasi.
    Menghitung hash SHA-256, jumlah sampel, serta memvalidasi zero-leakage
    antara seluruh partisi training/validasi klien dan global test set.
    """
    import hashlib

    def _file_hash(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    if base_partition_dir is not None:
        p_dir = Path(base_partition_dir)
        if (p_dir / "client_1_train.csv").exists():
            base_dir = p_dir
        elif (p_dir / scenario / "client_1_train.csv").exists():
            base_dir = p_dir / scenario
        else:
            base_dir = p_dir
    elif scenario in ("non_iid_quantity_skew", "quantity_skew", "controlled_quantity_skew_v1"):
        local_qs_dir = Path("storage/federated/partitions/quantity_skew_seed42")
        if local_qs_dir.exists():
            base_dir = local_qs_dir
        else:
            base_dir = Path(DEFAULT_PARTITION_DIR) / scenario
    elif scenario in ("non_iid_label_skew", "label_skew", "controlled_label_skew_v1"):
        local_ls_dir = Path("storage/federated/partitions/label_skew_seed42")
        if local_ls_dir.exists():
            base_dir = local_ls_dir
        else:
            base_dir = Path(DEFAULT_PARTITION_DIR) / scenario
    else:
        base_dir = Path(DEFAULT_PARTITION_DIR) / scenario


    test_file = Path(test_index_path or (DEFAULT_PARTITION_DIR / "global_test_index.csv"))
    if not test_file.exists():
        raise FileNotFoundError(f"Global test file tidak ditemukan: {test_file}")

    test_df = pd.read_csv(test_file)
    test_ids = set(test_df["participant_id"].astype(str))

    client_train_manifests = {}
    client_val_manifests = {}
    all_train_ids: Set[str] = set()
    all_val_ids: Set[str] = set()

    for i in range(num_clients):
        c_num = i + 1
        t_path = base_dir / f"client_{c_num}_train.csv"
        v_path = base_dir / f"client_{c_num}_val.csv"

        if not t_path.exists():
            raise FileNotFoundError(f"File training client {c_num} tidak ditemukan: {t_path}")
        if not v_path.exists():
            raise FileNotFoundError(f"File validation client {c_num} tidak ditemukan: {v_path}")

        t_df = pd.read_csv(t_path)
        v_df = pd.read_csv(v_path)

        t_ids = set(t_df["participant_id"].astype(str))
        v_ids = set(v_df["participant_id"].astype(str))

        all_train_ids.update(t_ids)
        all_val_ids.update(v_ids)

        client_train_manifests[f"client_{c_num}"] = {
            "file": str(t_path),
            "sha256": _file_hash(t_path),
            "samples": len(t_df),
            "participants": len(t_ids),
        }
        client_val_manifests[f"client_{c_num}"] = {
            "file": str(v_path),
            "sha256": _file_hash(v_path),
            "samples": len(v_df),
            "participants": len(v_ids),
        }

    # Validasi Zero-Leakage
    train_leakage = all_train_ids.intersection(test_ids)
    if train_leakage:
        raise ValueError(f"CRITICAL LEAKAGE: Training and test overlap: {train_leakage}")

    val_leakage = all_val_ids.intersection(test_ids)
    if val_leakage:
        raise ValueError(f"CRITICAL LEAKAGE: Validation and test overlap: {val_leakage}")

    return {
        "scenario": scenario,
        "partition_directory": str(base_dir),
        "num_clients": num_clients,
        "total_train_participants": len(all_train_ids),
        "total_val_participants": len(all_val_ids),
        "total_test_participants": len(test_ids),
        "zero_leakage_verified": True,
        "train_test_overlap_count": 0,
        "val_test_overlap_count": 0,
        "global_test": {
            "file": str(test_file),
            "sha256": _file_hash(test_file),
            "samples": len(test_df),
            "participants": len(test_ids),
        },
        "client_train_partitions": client_train_manifests,
        "client_val_partitions": client_val_manifests,
    }
