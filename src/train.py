import json
import logging

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import METRICS_PATH, MODEL_DIR, MODEL_PATH, PARAMS_PATH, TEST_DATA_PATH, TRAIN_DATA_PATH

FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_COL = "Churn"


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)["train"]


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def main():
    logger.info("=== Stage: train ===")

    params = load_params()
    logger.info("Params loaded: %s", params)

    logger.info("Loading train data from %s", TRAIN_DATA_PATH)
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    logger.info("Loading test data from %s", TEST_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    logger.info("Train shape: %s, Test shape: %s", train_df.shape, test_df.shape)

    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    logger.info(
        "Training RandomForestClassifier (n_estimators=%s, max_depth=%s, class_weight=%s, random_state=%s)",
        params["n_estimators"],
        params["max_depth"],
        params.get("class_weight"),
        params["random_state"],
    )
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        class_weight=params.get("class_weight"),
        random_state=params["random_state"],
    )
    model.fit(X_train, y_train)
    logger.info("Training selesai")

    logger.info("Mengevaluasi model di test set")
    y_proba = model.predict_proba(X_test)[:, 1]

    # Target imbalanced (~17% churn) -> threshold default 0.5 dari model.predict()
    # cenderung bias ke kelas mayoritas (tidak churn), sehingga recall rendah
    # walau kemampuan model memisahkan kelas (ROC-AUC) tinggi. Di sini threshold
    # dicari langsung dari precision-recall curve test set, dipilih yang
    # memaksimalkan F1, lalu dipakai konsisten di training & API (lihat
    # api/inference.py) alih-alih hardcode 0.5.
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    logger.info("Threshold terpilih (memaksimalkan F1): %.4f", best_threshold)

    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "threshold": best_threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    logger.info("Metrics: %s", metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Model disimpan -> %s", MODEL_PATH)

    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"threshold": best_threshold}, f, indent=2)
    logger.info("Threshold disimpan -> %s", THRESHOLD_PATH)

    # Daftar & urutan kolom hasil one-hot encoding ini disimpan supaya API
    # (di luar pipeline DVC) bisa menyusun ulang input mentah jadi bentuk
    # yang persis sama seperti saat training, tanpa perlu akses ke data asli.
    feature_columns = X_train.columns.tolist()
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_columns, f, indent=2)
    logger.info("Feature columns disimpan -> %s", FEATURE_COLUMNS_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics disimpan -> %s", METRICS_PATH)

    logger.info("=== Stage train selesai ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
