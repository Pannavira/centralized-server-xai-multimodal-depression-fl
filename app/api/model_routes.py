from fastapi import APIRouter, HTTPException
from app.services.model_service import model_service
from app.schemas.model import ModelInfoResponse

router = APIRouter(prefix="/model", tags=["Model"])


@router.get("/info", response_model=ModelInfoResponse)
def get_model_info():
    """
    Mengembalikan metadata arsitektur model baseline multimodal dan status kesiapan.
    """
    info = model_service.get_model_info()
    return info
