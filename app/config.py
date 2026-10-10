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

    model_checkpoint_path: str = "storage/models/best_global_model.pt"
    model_config_path: str = "configs/federated.yaml"
    
    @property
    def resolved_checkpoint_path(self) -> Path:
        p = Path(self.model_checkpoint_path)
        return p if p.is_absolute() else (BASE_DIR / p)

    @property
    def resolved_config_path(self) -> Path:
        p = Path(self.model_config_path)
        return p if p.is_absolute() else (BASE_DIR / p)

settings = Settings()
