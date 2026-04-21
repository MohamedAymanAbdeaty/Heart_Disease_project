"""Vercel entrypoint exposing heart disease predictions as an HTTP API."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict

# Ensure the project root is on sys.path so ai_pipeline can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ai_pipeline import DATASET_PATH, ModelBundle, build_optimized_model, predict_disease


app = FastAPI(
    title="Heart Disease Predictor API",
    version="1.0.0",
    description="Vercel-ready API for heart disease risk inference.",
)


ROOT_DIR = Path(__file__).resolve().parent.parent


class PredictRequest(BaseModel):
    """Expected numeric feature payload for model inference."""

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
    """Train and cache the selected model bundle for reuse across requests."""
    return build_optimized_model(str(DATASET_PATH))


@app.get("/")
def root() -> HTMLResponse:
    """Serve the static frontend HTML from the Python function."""
    html_path = ROOT_DIR / "index.html"

    try:
        html = html_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return HTMLResponse(content=html)


@app.get("/status")
def status() -> Dict[str, str]:
    """Machine-readable status endpoint for quick API checks."""
    return {
        "status": "ok",
        "message": "Heart Disease Predictor API is running.",
        "predict_endpoint": "POST /predict",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    """Health endpoint that also confirms model loading succeeds."""
    try:
        bundle = get_bundle()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "healthy",
        "selected_model": bundle.selected_model_name,
    }


@app.post("/predict")
@app.post("/api/predict")
def predict(payload: PredictRequest) -> Dict[str, float | int | str]:
    """Return calibrated prediction for the submitted feature payload."""
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
