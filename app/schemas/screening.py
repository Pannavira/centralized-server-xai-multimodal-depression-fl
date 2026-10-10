"""
app/schemas/screening.py
========================
Skema request dan response skrining depresi klinis (Placeholder).
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class ScreeningPrediction(BaseModel):
    predicted_phq8_score: float
    risk_category: str
    binary_indicator: str

class ScreeningResponse(BaseModel):
    participant_id: str
    prediction: ScreeningPrediction
    clinical_disclaimer: str
