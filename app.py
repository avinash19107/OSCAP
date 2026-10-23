
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

# --------------------------------------------------
# Lightweight feature-importance explanation
# --------------------------------------------------

@app.post("/explain")
def explain(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The trained model is not available."
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
                "message": "Feature names do not match the model.",
                "missing_features": missing,
                "unexpected_features": unexpected,
            }
        )

    try:
        values = np.asarray(
            [supplied[name] for name in feature_names],
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise ValueError("Feature values must be finite.")

        input_row = values.reshape(1, -1)

        prediction = int(model.predict(input_row)[0])
        probabilities = model.predict_proba(input_row)[0]

        importances = model.feature_importances_

        ranked_features = sorted(
            [
                {
                    "feature": name,
                    "input_value": float(supplied[name]),
                    "global_importance": float(importance),
                }
                for name, importance in zip(
                    feature_names, importances
                )
            ],
            key=lambda item: item["global_importance"],
            reverse=True
        )

        return {
            "classification": (
                "Attack" if prediction == 1 else "Benign"
            ),
            "attack_probability": round(
                float(probabilities[1]), 6
            ),
            "top_features": ranked_features[:10],
            "explanation_type": "Global Random Forest feature importance",
            "note": (
                "These features are globally important to the model; "
                "they are not per-sample SHAP contributions."
            )
        }

    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature values: {exc}"
        ) from exc
        
# --------------------------------------------------
# Module 4: Concept Drift Detection
# --------------------------------------------------

from statistics import median

# --------------------------------------------------
# Module 4: Concept Drift Detection
# --------------------------------------------------

DRIFT_BASELINE_PATH = MODEL_DIR / "drift_baseline.json"

try:
    with open(DRIFT_BASELINE_PATH, "r", encoding="utf-8") as f:
        drift_baseline = json.load(f)
    print("Drift baseline loaded successfully.")
except Exception as exc:
    drift_baseline = {}
    print(f"DRIFT BASELINE ERROR: {type(exc).__name__}: {exc}")


class DriftRequest(BaseModel):
    samples: list[dict[str, float]] = Field(
        ...,
        min_length=2,
        description="At least two network-flow samples."
    )


@app.get("/drift/status")
def drift_status():
    return {
        "baseline_loaded": bool(drift_baseline),
        "features": list(drift_baseline.keys()),
        "message": (
            "Drift baseline ready."
            if drift_baseline
            else "Baseline unavailable. Check drift_baseline.json."
        )
    }


@app.post("/drift")
def detect_drift(request: DriftRequest):
    if not drift_baseline:
        raise HTTPException(
            status_code=503,
            detail="Drift baseline is not available."
        )

    if len(request.samples) < 2:
        raise HTTPException(
            status_code=422,
            detail="Provide at least two samples."
        )

    baseline_features = list(drift_baseline.keys())
    expected = set(baseline_features)

    current = {name: [] for name in baseline_features}

    for index, sample in enumerate(request.samples):
        received = set(sample.keys())
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)

        if missing or unexpected:
            raise HTTPException(
                status_code=422,
                detail={
                    "sample_index": index,
                    "missing_features": missing,
                    "unexpected_features": unexpected
                }
            )

        for name in baseline_features:
            try:
                value = float(sample[name])
            except (TypeError, ValueError, OverflowError):
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid value for {name} in sample {index}."
                )

            if not np.isfinite(value):
                raise HTTPException(
                    status_code=422,
                    detail=f"Non-finite value for {name} in sample {index}."
                )

            current[name].append(value)

    results = []
    shifted_count = 0

    for name in baseline_features:
        reference = drift_baseline[name]
        values = sorted(current[name])

        current_median = float(np.median(values))
        current_q25 = float(np.percentile(values, 25))
        current_q75 = float(np.percentile(values, 75))

        baseline_median = float(reference["median"])
        baseline_q25 = float(reference["q25"])
        baseline_q75 = float(reference["q75"])

        baseline_iqr = baseline_q75 - baseline_q25
        current_iqr = current_q75 - current_q25

        # Normalize median shift by baseline IQR.
        # A small floor avoids division by zero.
        scale = max(abs(baseline_iqr), 1e-9)
        median_shift = abs(current_median - baseline_median) / scale

        # Flag a feature when its median shifts by more than
        # one baseline IQR. This is a heuristic, not a statistical test.
        shifted = median_shift > 1.0

        if shifted:
            shifted_count += 1

        results.append({
            "feature": name,
            "baseline_median": baseline_median,
            "current_median": current_median,
            "baseline_iqr": baseline_iqr,
            "current_iqr": current_iqr,
            "normalized_median_shift": round(median_shift, 4),
            "shift_detected": shifted
        })

    drift_detected = shifted_count > 0

    return {
        "samples_analyzed": len(request.samples),
        "drift_detected": drift_detected,
        "shifted_feature_count": shifted_count,
        "total_features_checked": len(baseline_features),
        "method": "Median shift normalized by benign baseline IQR",
        "threshold": 1.0,
        "warning": (
            "Distribution drift does not prove an attack. "
            "Investigate alongside model predictions and traffic context."
        ),
        "feature_results": results
    }

