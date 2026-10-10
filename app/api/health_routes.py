from fastapi import APIRouter
from app.config import settings
from app.services.model_service import model_service

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    """
    Endpoint pemantauan kesehatan server dan status model.
    """
    return {
        "status": "ok",
        "service": settings.app_name,
        "model_loaded": model_service.is_loaded(),
        "device": model_service.get_device_name(),
    }
