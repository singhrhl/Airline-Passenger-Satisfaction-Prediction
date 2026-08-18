"""
backend/app/main.py
"""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import model_loader
from .schemas import (
    PassengerInput, PredictionResponse,
    ModelMetricsResponse, EdaSummaryResponse, HealthResponse,
)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

app = FastAPI(title="Airline Passenger Satisfaction API")

# allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend origin before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# load eda_summary.json once at startup, same pattern as metrics.json in model_loader
with open(MODELS_DIR / "eda_summary.json") as f:
    eda_summary = json.load(f)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_loaded": model_loader.pipeline is not None}


@app.get("/eda-summary", response_model=EdaSummaryResponse)
def get_eda_summary():
    return eda_summary


@app.get("/model-metrics", response_model=ModelMetricsResponse)
def get_model_metrics():
    return model_loader.get_model_metrics()


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerInput):
    try:
        # by_alias=True → produces the ORIGINAL column names ("Gender", "Age", ...)
        # that model_loader.predict_passenger() and the pipeline expect
        passenger_dict = passenger.model_dump(by_alias=True)
        result = model_loader.predict_passenger(passenger_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")