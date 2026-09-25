"""
Test end-to-end untuk seluruh pipeline DVC (lihat dvc.yaml):

    data mentah -> preprocess.main() -> train.main() -> model + metrics

Setiap proses dijalankan dengan implementasi aslinya (bukan mock), tapi
semua path (raw data, processed data, model, metrics, params) di-monkeypatch
ke folder temporary supaya test ini tidak menimpa output pipeline asli di
data/processed/, models/, atau metrics.json.
"""

import json
import shutil

import joblib
import pandas as pd
import pytest

import preprocess
import train
from config import DATA_DIR as REAL_RAW_DATA_DIR

RAW_FILE = "E Commerce Dataset.xlsx"


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shutil.copy(REAL_RAW_DATA_DIR / RAW_FILE, raw_dir / RAW_FILE)

    processed_dir = tmp_path / "processed"
    model_path = tmp_path / "models" / "churn_model.pkl"
    metrics_path = tmp_path / "metrics.json"

    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        """
preprocess:
  raw_file: "E Commerce Dataset.xlsx"
  sheet_name: "E Comm"
  test_size: 0.2
  random_state: 42

train:
  n_estimators: 50
  max_depth: 5
  random_state: 42
"""
    )

    # Arahkan kedua module ke lokasi temporary, bukan path project asli.
    monkeypatch.setattr(preprocess, "DATA_DIR", raw_dir)
    monkeypatch.setattr(preprocess, "PROCESSED_DATA_DIR", processed_dir)
    monkeypatch.setattr(preprocess, "PARAMS_PATH", params_path)

    monkeypatch.setattr(train, "TRAIN_DATA_PATH", processed_dir / "train.csv")
    monkeypatch.setattr(train, "TEST_DATA_PATH", processed_dir / "test.csv")
    monkeypatch.setattr(train, "MODEL_PATH", model_path)
    monkeypatch.setattr(train, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(train, "PARAMS_PATH", params_path)

    return {
        "processed_dir": processed_dir,
        "model_path": model_path,
        "metrics_path": metrics_path,
    }


def test_preprocess_stage_produces_clean_train_test_csv(pipeline_env):
    preprocess.main()

    train_csv = pipeline_env["processed_dir"] / "train.csv"
    test_csv = pipeline_env["processed_dir"] / "test.csv"
    assert train_csv.exists()
    assert test_csv.exists()

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    assert "Churn" in train_df.columns
    assert "Churn" in test_df.columns
    assert not train_df.isna().any().any()
    assert not test_df.isna().any().any()
    # test_size=0.2 di params.yaml -> proporsi test set sekitar 20% dari total.
    total = len(train_df) + len(test_df)
    assert abs(len(test_df) / total - 0.2) < 0.02


def test_train_stage_produces_model_and_metrics(pipeline_env):
    preprocess.main()
    train.main()

    assert pipeline_env["model_path"].exists()
    assert pipeline_env["metrics_path"].exists()

    metrics = json.loads(pipeline_env["metrics_path"].read_text())
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0

    model = joblib.load(pipeline_env["model_path"])
    test_df = pd.read_csv(pipeline_env["processed_dir"] / "test.csv")
    X_test = test_df.drop(columns=["Churn"])

    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_full_pipeline_end_to_end(pipeline_env):
    """Menjalankan preprocess -> train berurutan, meniru `dvc repro`."""
    preprocess.main()
    train.main()

    assert (pipeline_env["processed_dir"] / "train.csv").exists()
    assert (pipeline_env["processed_dir"] / "test.csv").exists()
    assert pipeline_env["model_path"].exists()
    assert pipeline_env["metrics_path"].exists()

    metrics = json.loads(pipeline_env["metrics_path"].read_text())
    # Model harus jauh lebih baik dari tebakan acak pada dataset ini.
    assert metrics["roc_auc"] > 0.8
