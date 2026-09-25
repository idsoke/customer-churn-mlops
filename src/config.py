import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = BASE_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "models")

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
METRICS_PATH = BASE_DIR / "metrics.json"
PARAMS_PATH = BASE_DIR / "params.yaml"

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

MODEL_NAME = os.getenv("MODEL_NAME", "churn_model.pkl")
MODEL_PATH = MODEL_DIR / MODEL_NAME

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
