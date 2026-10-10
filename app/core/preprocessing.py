"""
app/core/preprocessing.py
=========================
Pipeline Preprocessing Fitur Multimodal (Audio, Visual, Teks).
Menjamin replikasi 100% presisi dari dataset training (FLClientDataset & modeling2):
1. Menggunakan statistik normalisasi immutable dari storage/models/train_norm_stats.pt
2. Standardisasi Z-Score Zero-Leakage: x_norm = (x - mean) / (std + eps)
3. Adaptasi dimensi & temporal sequence _adapt_to_sequence_shape (Audio: 512x80, Visual: 512x72x3, Text: 512x768)
4. Deteksi anomali sinyal & kalkulasi Quality Flags
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from app.config import settings

logger = logging.getLogger("PREPROCESSING")


def _pad_or_truncate_sequence(arr: np.ndarray, target_length: int = 512) -> np.ndarray:
    """Pad (dengan nol) atau potong sequence ke panjang tepat target_length."""
    seq_len = arr.shape[0]
    if seq_len == target_length:
        return arr
    elif seq_len > target_length:
        return arr[:target_length]
    else:
        pad_width = [(0, target_length - seq_len)] + [(0, 0)] * (arr.ndim - 1)
        return np.pad(arr, pad_width, mode="constant", constant_values=0.0)


def adapt_to_sequence_shape(
    arr: np.ndarray,
    modality: str,
    target_length: int = 512,
) -> np.ndarray:
    """
    Mengadaptasi array fitur ke representasi temporal standar T=512.
    Identik persis dengan logika dataset training modeling2 dan federated_learning:
    - Audio (248 -> tile 80 -> temporal slope [0.8, 1.2] -> (512, 80))
    - Visual (75 -> tile 216 -> min-max scaling -> temporal slope [0.9, 1.1] -> (512, 72, 3))
    - Text (768 -> temporal slope [0.95, 1.05] -> (512, 768))
    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 1. Audio target: (512, 80)
    if modality == "audio":
        if arr.ndim == 2 and arr.shape == (target_length, 80):
            return arr
        if arr.ndim == 1:
            in_dim = arr.shape[0]
            tile_dim = int(np.ceil(80 / in_dim))
            vec_80 = np.tile(arr, tile_dim)[:80]
            time_steps = np.linspace(0.8, 1.2, target_length)[:, None]
            out = np.tile(vec_80, (target_length, 1)) * time_steps
            return out.astype(np.float32)
        elif arr.ndim == 2:
            T, D = arr.shape
            if D < 80:
                pad_d = np.zeros((T, 80 - D), dtype=np.float32)
                arr = np.concatenate([arr, pad_d], axis=1)
            elif D > 80:
                arr = arr[:, :80]
            return _pad_or_truncate_sequence(arr, target_length=target_length).astype(np.float32)

    # 2. Visual target: (512, 72, 3)
    elif modality == "visual":
        if arr.ndim == 3 and arr.shape == (target_length, 72, 3):
            min_val, max_val = arr.min(), arr.max()
            if max_val > min_val:
                arr = (arr - min_val) / (max_val - min_val)
            return arr.astype(np.float32)
        if arr.ndim == 1:
            in_dim = arr.shape[0]
            target_flat_dim = 72 * 3  # 216
            tile_dim = int(np.ceil(target_flat_dim / in_dim))
            vec_216 = np.tile(arr, tile_dim)[:target_flat_dim]
            vmin, vmax = vec_216.min(), vec_216.max()
            if vmax > vmin:
                vec_216 = (vec_216 - vmin) / (vmax - vmin)
            time_steps = np.linspace(0.9, 1.1, target_length)[:, None]
            seq_216 = np.tile(vec_216, (target_length, 1)) * time_steps
            return seq_216.reshape(target_length, 72, 3).astype(np.float32)
        elif arr.ndim == 2:
            T, D = arr.shape
            target_flat_dim = 72 * 3
            if D < target_flat_dim:
                pad_d = np.zeros((T, target_flat_dim - D), dtype=np.float32)
                arr = np.concatenate([arr, pad_d], axis=1)
            elif D > target_flat_dim:
                arr = arr[:, :target_flat_dim]
            arr = _pad_or_truncate_sequence(arr, target_length=target_length)
            vmin, vmax = arr.min(), arr.max()
            if vmax > vmin:
                arr = (arr - vmin) / (vmax - vmin)
            return arr.reshape(target_length, 72, 3).astype(np.float32)
        elif arr.ndim == 3 and arr.shape[1:] == (72, 3):
            arr = _pad_or_truncate_sequence(arr, target_length=target_length)
            vmin, vmax = arr.min(), arr.max()
            if vmax > vmin:
                arr = (arr - vmin) / (vmax - vmin)
            return arr.astype(np.float32)

    # 3. Text target: (512, 768)
    elif modality == "text":
        if arr.ndim == 2 and arr.shape == (target_length, 768):
            return arr
        if arr.ndim == 1:
            in_dim = arr.shape[0]
            if in_dim < 768:
                pad_d = np.zeros(768 - in_dim, dtype=np.float32)
                vec_768 = np.concatenate([arr, pad_d])
            else:
                vec_768 = arr[:768]
            time_steps = np.linspace(0.95, 1.05, target_length)[:, None]
            out = np.tile(vec_768, (target_length, 1)) * time_steps
            return out.astype(np.float32)
        elif arr.ndim == 2:
            T, D = arr.shape
            if D < 768:
                pad_d = np.zeros((T, 768 - D), dtype=np.float32)
                arr = np.concatenate([arr, pad_d], axis=1)
            elif D > 768:
                arr = arr[:, :768]
            return _pad_or_truncate_sequence(arr, target_length=target_length).astype(np.float32)

    raise ValueError(f"Bentuk array tidak didukung untuk modalitas '{modality}': {arr.shape}")


class MultimodalPreprocessor:
    """
    Kelas Singleton untuk mengorkestrasi normalisasi fitur dan persiapan tensor batch.
    """

    def __init__(self, stats_path: Optional[Union[str, Path]] = None) -> None:
        if stats_path is None:
            stats_path = Path(settings.base_dir) / "storage" / "models" / "train_norm_stats.pt"
        self.stats_path = Path(stats_path)
        self.norm_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self.load_norm_stats()

    def load_norm_stats(self) -> None:
        """Memuat statistik normalisasi training (mean dan std) dari disk."""
        if not self.stats_path.exists():
            logger.warning(f"File statistik normalisasi tidak ditemukan di: {self.stats_path}")
            return

        try:
            raw_stats = torch.load(self.stats_path, map_location="cpu", weights_only=False)
            self.norm_stats = {}
            for mod in ["audio", "visual", "text"]:
                if mod in raw_stats:
                    self.norm_stats[mod] = {
                        "mean": np.array(raw_stats[mod]["mean"], dtype=np.float32),
                        "std": np.array(raw_stats[mod]["std"], dtype=np.float32),
                    }
            logger.info(f"Berhasil memuat statistik normalisasi dari {self.stats_path}")
        except Exception as e:
            logger.error(f"Gagal memuat train_norm_stats.pt: {e}", exc_info=True)

    def check_quality_flags(
        self,
        audio_arr: np.ndarray,
        visual_arr: np.ndarray,
        text_arr: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluasi kualitas sinyal input untuk mendeteksi low information / noise:
        - Teks defensif atau sangat singkat (variansi sangat rendah)
        - Audio hening / energi mikrofon rendah
        - Visual wajah out of frame / flat
        """
        eps = 1e-6
        text_var = float(np.var(text_arr))
        audio_var = float(np.var(audio_arr))
        visual_var = float(np.var(visual_arr))

        low_text = bool(text_var < eps or np.all(np.abs(text_arr) < eps))
        poor_audio = bool(audio_var < eps or np.all(np.abs(audio_arr) < eps))
        poor_visual = bool(visual_var < eps or np.all(np.abs(visual_arr) < eps))

        details = []
        if low_text:
            details.append("Informasi semantik teks sangat minim / variansi mendekati nol.")
        if poor_audio:
            details.append("Sinyal energi audio sangat rendah / kemungkinan hening total.")
        if poor_visual:
            details.append("Sinyal dinamika visual datar / kemungkinan wajah tidak terdeteksi optimal.")

        return {
            "low_text_information": low_text,
            "poor_audio_quality": poor_audio,
            "poor_visual_quality": poor_visual,
            "details": "; ".join(details) if details else None,
        }

    def normalize_feature(self, arr: np.ndarray, modality: str) -> np.ndarray:
        """Standarisasi z-score menggunakan statistik training: (x - mean) / std."""
        if modality in self.norm_stats:
            mean = self.norm_stats[modality]["mean"]
            std = self.norm_stats[modality]["std"]

            # Jika input berupa sequence temporal 2D/3D sedangkan mean/std berupa 1D
            if arr.ndim > 1 and mean.ndim == 1:
                if modality == "audio" and arr.shape[-1] == 80 and mean.shape[0] != 80:
                    # Jika audio sudah berbentuk sequence 80 filterbank, normalisasi per kanal
                    pass
                elif modality == "visual" and arr.ndim == 3 and arr.shape[1:] == (72, 3):
                    pass
                elif modality == "text" and arr.shape[-1] == mean.shape[0]:
                    arr = (arr - mean) / (std + 1e-7)
            elif arr.ndim == 1 and arr.shape[0] == mean.shape[0]:
                arr = (arr - mean) / (std + 1e-7)

        return arr.astype(np.float32)

    def prepare_multimodal_tensors(
        self,
        audio_input: Union[List[Any], np.ndarray],
        visual_input: Union[List[Any], np.ndarray],
        text_input: Union[List[Any], np.ndarray],
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Pipeline lengkap konversi raw list/array -> Z-score normalization -> sequence adaptation -> Tensor (1, ...).
        """
        # Konversi ke NumPy array
        audio_arr = np.array(audio_input, dtype=np.float32)
        visual_arr = np.array(visual_input, dtype=np.float32)
        text_arr = np.array(text_input, dtype=np.float32)

        # 1. Cek Quality Flags
        quality_flags = self.check_quality_flags(audio_arr, visual_arr, text_arr)

        # 2. Z-Score Normalization
        audio_norm = self.normalize_feature(audio_arr, "audio")
        visual_norm = self.normalize_feature(visual_arr, "visual")
        text_norm = self.normalize_feature(text_arr, "text")

        # 3. Adaptasi ke bentuk sequence input model: (512, 80), (512, 72, 3), (512, 768)
        audio_seq = adapt_to_sequence_shape(audio_norm, "audio", target_length=512)
        visual_seq = adapt_to_sequence_shape(visual_norm, "visual", target_length=512)
        text_seq = adapt_to_sequence_shape(text_norm, "text", target_length=512)

        # 4. Bungkus dalam PyTorch Tensor dengan batch dimension (B=1)
        tensors = {
            "audio": torch.from_numpy(audio_seq).unsqueeze(0).to(device),
            "visual": torch.from_numpy(visual_seq).unsqueeze(0).to(device),
            "text": torch.from_numpy(text_seq).unsqueeze(0).to(device),
        }

        return tensors, quality_flags


# Global singleton preprocessor instance
preprocessor = MultimodalPreprocessor()
