from typing import List, Optional
from pydantic import BaseModel, Field

class ModelInfoResponse(BaseModel):
    model: str = Field(..., description="Nama model")
    architecture: str = Field(..., description="Arsitektur baseline")
    modalities: List[str] = Field(..., description="Daftar modalitas input")
    device: str = Field(..., description="Device komputasi aktif (cuda/cpu)")
    status: str = Field(..., description="Status kesiapan model")
    total_parameters: Optional[int] = Field(None, description="Jumlah total parameter model")
    checkpoint_path: Optional[str] = Field(None, description="Lokasi file checkpoint")
