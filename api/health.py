"""Vercel function for /api/health endpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from functools import lru_cache
from typing import Dict

# Ensure the project root is on sys.path so ai_pipeline can be imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from mangum import Mangum

from ai_pipeline import DATASET_PATH, ModelBundle, build_optimized_model


app = FastAPI(title="Heart Disease Health API", version="1.0.0")


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    """Train and cache model artifacts for repeated requests."""
    return build_optimized_model(str(DATASET_PATH))


@app.get("/")
def health() -> Dict[str, str]:
    """Return health information and selected model metadata."""
    try:
        bundle = get_bundle()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "healthy",
        "selected_model": bundle.selected_model_name,
    }


# Vercel ASGI entry point
handler = Mangum(app, lifespan="off")
