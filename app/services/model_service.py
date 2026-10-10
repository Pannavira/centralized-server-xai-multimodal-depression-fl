import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml
import torch
import torch.nn as nn

from app.config import settings
from app.core.model import MultimodalBaselineModel, build_model_from_config

logger = logging.getLogger("FED_MIND_SERVER")


class ModelService:
    """
    Singleton service untuk mengelola siklus hidup model multimodal FED-MIND di memory server.
    Mendukung isolasi model_type (centralized / fl_iid / fl_non_iid) dan tracking checksum SHA-256.
    """

    def __init__(self) -> None:
        self._model: Optional[MultimodalBaselineModel] = None
        self._device: torch.device = torch.device("cpu")
        self._is_loaded: bool = False
        self._total_params: int = 0
        self._checkpoint_file: str = ""
        self._checkpoint_checksum: str = ""
        self._model_type: str = settings.model_type

    def load_model(self, force_reload: bool = False) -> bool:
        if self._is_loaded and not force_reload:
            return True

        ckpt_path = settings.resolved_checkpoint_path
        cfg_path = settings.resolved_config_path

        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found at: {ckpt_path}")
            return False

        if not cfg_path.exists():
            logger.error(f"Model config not found at: {cfg_path}")
            return False

        # 1. Resolve Device
        if settings.device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._model_type = settings.model_type
        logger.info(f"Loading FED-MIND Multimodal Model [{self._model_type}] on device: {self._device}...")

        # 2. Build Model Architecture
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._model = build_model_from_config(cfg)

        # 3. Load State Dict
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        self._model.load_state_dict(state_dict, strict=True)
        self._model.to(self._device)
        self._model.eval()

        self._total_params = sum(p.numel() for p in self._model.parameters())
        self._checkpoint_file = str(ckpt_path.name)
        
        # Checksum calculation (SHA-256)
        try:
            self._checkpoint_checksum = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        except Exception:
            self._checkpoint_checksum = "unknown"

        self._is_loaded = True

        logger.info(
            f"Model [{self._model_type}] loaded! Params: {self._total_params:,} | SHA256: {self._checkpoint_checksum[:12]}... | Device: {self._device}"
        )
        return True

    def ensure_loaded(self) -> None:
        """Memastikan model dimuat (lazy-load jika lifespan belum dieksekusi)."""
        if not self._is_loaded:
            self.load_model()

    def get_model(self) -> Optional[MultimodalBaselineModel]:
        self.ensure_loaded()
        return self._model

    def is_loaded(self) -> bool:
        self.ensure_loaded()
        return self._is_loaded

    def get_device(self) -> torch.device:
        self.ensure_loaded()
        return self._device

    def get_device_name(self) -> str:
        self.ensure_loaded()
        return self._device.type

    def get_model_type(self) -> str:
        return self._model_type

    def get_checkpoint_checksum(self) -> str:
        return self._checkpoint_checksum

    def get_model_info(self) -> Dict[str, Any]:
        self.ensure_loaded()
        name_map = {
            "centralized": "Centralized Multimodal Baseline",
            "fl_iid": "FedAvg FL-IID (3 Clients Pretrained)",
            "fl_non_iid": "FedAvg FL-Non-IID Combined Skew",
        }
        return {
            "model": "FED-MIND Multimodal",
            "model_type": self._model_type,
            "model_name": name_map.get(self._model_type, f"FED-MIND ({self._model_type})"),
            "architecture": "Centralized Multimodal Baseline",
            "modalities": ["text", "audio", "visual"],
            "device": self.get_device_name(),
            "status": "ready" if self._is_loaded else "not_loaded",
            "total_parameters": self._total_params,
            "checkpoint_path": self._checkpoint_file,
            "checkpoint_checksum": self._checkpoint_checksum,
            "xai_baseline_path": str(settings.resolved_xai_baseline_path.name),
        }


# Global Singleton
model_service = ModelService()
