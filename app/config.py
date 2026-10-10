import os
from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "FED-MIND Centralized Server"
    app_version: str = "1.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    device: str = "cuda"

    model_type: str = "fl_iid"  # "centralized" | "fl_iid" | "fl_non_iid"
    model_checkpoint_path: str = "storage/models/fl_iid/model.pt"
    model_config_path: str = "configs/federated.yaml"
    database_url: str = "sqlite:///storage/fedmind.db"
    
    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def resolved_checkpoint_path(self) -> Path:
        p = Path(self.model_checkpoint_path)
        if not p.is_absolute():
            p = BASE_DIR / p
        if not p.exists():
            # Fallback to model_type directory
            fallback = BASE_DIR / "storage" / "models" / self.model_type / "model.pt"
            if fallback.exists():
                return fallback
        return p

    @property
    def resolved_xai_baseline_path(self) -> Path:
        specific = BASE_DIR / "storage" / "models" / self.model_type / "xai_baseline.pt"
        if specific.exists():
            return specific
        return BASE_DIR / "storage" / "models" / "xai_baseline_latents.pt"

    @property
    def resolved_config_path(self) -> Path:
        p = Path(self.model_config_path)
        return p if p.is_absolute() else (BASE_DIR / p)

settings = Settings()
