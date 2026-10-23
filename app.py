
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "intrusion_model"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"
FEATURES_PATH = MODEL_DIR / "features.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

app = FastAPI(
    title="AI Intrusion Detection API",
    description="Cloud API for network-flow intrusion detection.",
    version="2.0.0",
)

model = None
feature_names = []
metadata = {}


# --------------------------------------------------
# Load the trained model
# --------------------------------------------------

try:
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("features.json must contain a non-empty list.")

    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Duplicate feature names found.")

    model = joblib.load(MODEL_PATH)

    if len(feature_names) != getattr(model, "n_features_in_", -1):
        raise ValueError("Feature count does not match the trained model.")

    if list(model.classes_) != [0, 1]:
        raise ValueError("Expected model classes [0, 1].")

    print(f"Model loaded successfully: {len(feature_names)} features")

except Exception as exc:
    # Keep the service available so health/status endpoints can explain
    # the loading failure. Predictions remain disabled.
    model = None
    print(f"MODEL LOAD ERROR: {type(exc).__name__}: {exc}")


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Map of the 77 feature names to numeric values.",
    )


# --------------------------------------------------
# Health endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Intrusion Detection API is running",
        "app": "xyzapi",
        "status": "healthy",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/status")
def model_status():
    return {
        "model_loaded": model is not None,
        "feature_count": len(feature_names),
        "model_type": metadata.get("model_type", "RandomForestClassifier"),
        "message": (
            "Model ready for predictions."
            if model is not None
            else "Model unavailable. Check deployment logs and model files."
        ),
    }


# --------------------------------------------------
# Intrusion prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The trained model is not available.",
        )

    supplied = request.features

    expected = set(feature_names)
    received = set(supplied)

    missing = sorted(expected - received)
    unexpected = sorted(received - expected)

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature names do not match the trained model.",
                "missing_features": missing,
                "unexpected_features": unexpected,
            },
        )

    try:
        # Build input in the exact order used during training.
        values = np.asarray(
            [supplied[name] for name in feature_names],
            dtype=np.float64,
        )

        if not np.isfinite(values).all():
            raise ValueError("All feature values must be finite numbers.")

        input_row = values.reshape(1, -1)

        prediction = int(model.predict(input_row)[0])
        probabilities = model.predict_proba(input_row)[0]
        attack_probability = float(probabilities[1])

        return {
            "prediction": prediction,
            "classification": "Attack" if prediction == 1 else "Benign",
            "confidence": round(
                float(probabilities[prediction]), 6
            ),
            "attack_probability": round(attack_probability, 6),
            "benign_probability": round(
                float(probabilities[0]), 6
            ),
            "model": "RandomForestClassifier",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature values: {exc}",
        ) from exc
