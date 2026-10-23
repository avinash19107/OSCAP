
from fastapi import FastAPI
from datetime import datetime, timezone

app = FastAPI(
    title="AI Intrusion Detection API",
    description="Cloud API for network traffic analysis and intrusion detection.",
    version="1.0.0",
)


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
        "model_loaded": False,
        "message": "No trained model has been uploaded yet.",
    }
