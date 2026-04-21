"""Vercel function for /api/predict endpoint."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai_pipeline import DATASET_PATH, ModelBundle, build_optimized_model, predict_disease


app = FastAPI(title="Heart Disease Predict API", version="1.0.0")


class PredictRequest(BaseModel):
    """Expected numeric feature payload for inference."""

    age: float = Field(..., ge=1, le=120)
    sex: float = Field(..., ge=0, le=1)
    cp: float = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=50, le=300)
    chol: float = Field(..., ge=50, le=1000)
    fbs: float = Field(..., ge=0, le=1)
    restecg: float = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=30, le=250)
    exang: float = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: float = Field(..., ge=0, le=2)
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=0, le=3)


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    """Train and cache model artifacts for repeated requests."""
    return build_optimized_model(str(DATASET_PATH))


@app.post("/")
def predict(payload: PredictRequest) -> Dict[str, float | int | str]:
    """Return calibrated prediction for the submitted payload."""
    try:
        bundle = get_bundle()
        label, probability = predict_disease(payload.model_dump(), bundle)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "prediction": label,
        "probability": probability,
        "threshold": bundle.decision_threshold,
        "selected_model": bundle.selected_model_name,
    }
