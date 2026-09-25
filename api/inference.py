import json

import joblib
import pandas as pd

from src.config import MODEL_DIR, MODEL_PATH

FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"
DEFAULT_THRESHOLD = 0.5

_model = None
_feature_columns: list[str] | None = None
_threshold: float | None = None


def load_model():
    """Load model, daftar kolom fitur, & threshold sekali saja, lalu simpan di cache modul."""
    global _model, _feature_columns, _threshold
    if _model is None:
        if not MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
            raise FileNotFoundError(
                "Model belum tersedia. Jalankan `dvc repro` dari root proyek terlebih dahulu."
            )
        _model = joblib.load(MODEL_PATH)
        with open(FEATURE_COLUMNS_PATH) as f:
            _feature_columns = json.load(f)

        # Threshold hasil tuning (memaksimalkan F1 di precision-recall curve,
        # lihat src/train.py) dipakai di sini supaya konsisten dengan metrics.json.
        # Fallback ke 0.5 kalau file belum ada (mis. model lama sebelum tuning ini ada).
        if THRESHOLD_PATH.exists():
            with open(THRESHOLD_PATH) as f:
                _threshold = json.load(f)["threshold"]
        else:
            _threshold = DEFAULT_THRESHOLD
    return _model, _feature_columns, _threshold


def encode_input(data: dict) -> pd.DataFrame:
    """Ubah data mentah 1 pelanggan menjadi baris fitur sesuai format training.

    Kolom numerik disalin langsung berdasarkan nama. Kolom kategorikal
    dicocokkan ke kolom one-hot hasil training (mis. "PreferredLoginDevice"
    + "Mobile Phone" -> kolom "PreferredLoginDevice_Mobile Phone" = 1).
    Kategori yang jadi baseline saat training (dibuang oleh drop_first)
    otomatis direpresentasikan sebagai semua dummy 0 -- sama seperti saat
    training, tanpa perlu logika khusus di sini.
    """
    _, feature_columns, _ = load_model()
    row = dict.fromkeys(feature_columns, 0)

    for key, value in data.items():
        dummy_col = f"{key}_{value}"
        if dummy_col in row:
            row[dummy_col] = 1
        elif key in row:
            row[key] = value

    return pd.DataFrame([row], columns=feature_columns)


def predict(data: dict) -> dict:
    model, _, threshold = load_model()
    X = encode_input(data)
    probability = float(model.predict_proba(X)[0, 1])
    is_high_risk = probability >= threshold

    return {
        "churn": is_high_risk,
        "churn_probability": round(probability, 4),
        "risk_level": "Tinggi" if is_high_risk else "Aman",
    }
