import json
import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.inference import predict
from api.schemas import CustomerData, ModelInfo, PredictionResponse
from src.config import METRICS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Predictor API",
    description="API untuk memprediksi risiko pelanggan berhenti berlangganan (churn).",
    version="1.0.0",
)

# Origin frontend diizinkan lewat CORS, dipisah koma via env var ALLOWED_ORIGINS
# (mis. "https://fe-customer-churn-mlops.vercel.app,http://localhost:4200").
# Default localhost:4200 supaya dev lokal (ng serve) tetap jalan tanpa config.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:4200").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Metrics belum tersedia. Jalankan `dvc repro` dari root proyek terlebih dahulu.",
        )
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(data: CustomerData):
    try:
        return predict(data.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
