from fastapi import APIRouter

router = APIRouter(prefix="/screenings", tags=["Screenings"])


@router.get("")
def list_screenings():
    """Placeholder endpoint histori skrining."""
    return {"message": "Screening endpoint placeholder for Phase 1 Step 2"}
