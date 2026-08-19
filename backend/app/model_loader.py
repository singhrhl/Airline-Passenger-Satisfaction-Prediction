"""
backend/app/model_loader.py
Loads the pipeline + metrics once at import time, exposes predict_passenger().
"""
import json
import joblib
import shap
import pandas as pd
from pathlib import Path
import os

MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parent.parent.parent / "models"))
# ---------------------------------------------------------------------------
# 1. Load artifacts ONCE at module import
# ---------------------------------------------------------------------------
pipeline = joblib.load(MODELS_DIR / "pipeline.joblib")

with open(MODELS_DIR / "metrics.json") as f:
    metrics = json.load(f)

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

explainer = shap.TreeExplainer(classifier)

FEATURE_NAMES = preprocessor.get_feature_names_out()


# ---------------------------------------------------------------------------
# 2. Single-passenger prediction + local SHAP drivers
# ---------------------------------------------------------------------------
def predict_passenger(passenger_dict: dict, top_n: int = 5) -> dict:
    """
    passenger_dict: dict using the ORIGINAL column names (aliases), e.g.
        {"Gender": "Male", "Age": 35, ...}
    Returns a dict matching schemas.PredictionResponse's fields.
    """
    row_df = pd.DataFrame([passenger_dict])

    proba = pipeline.predict_proba(row_df)[:, 1][0]
    predicted_label = "dissatisfied" if proba >= 0.5 else "satisfied"

    row_transformed = preprocessor.transform(row_df)

    shap_vals = explainer.shap_values(row_transformed)[0]  # first (only) row

    # pair feature names with their SHAP value, sort by ABSOLUTE value descending
    paired = list(zip(FEATURE_NAMES, shap_vals))
    paired.sort(key=lambda pair: abs(pair[1]), reverse=True)
    top_paired = paired[:top_n]

    top_drivers = [
        {
            "feature": name,
            "shap_value": float(val),
            "direction": "increases_dissatisfaction" if val > 0 else "decreases_dissatisfaction",
        }
        for name, val in top_paired
    ]

    return {
        "dissatisfaction_probability": float(proba),
        "predicted_label": predicted_label,
        "top_dissatisfaction_drivers": top_drivers,
    }


# ---------------------------------------------------------------------------
# 3. Metrics passthrough for GET /model-metrics
# ---------------------------------------------------------------------------
def get_model_metrics() -> dict:
    return metrics