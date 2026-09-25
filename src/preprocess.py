"""
Stage 1 dari pipeline DVC (lihat dvc.yaml): preprocessing data mentah.

Alur:
    data/raw/E Commerce Dataset.xlsx
        -> drop kolom ID
        -> imputasi missing value (numerik, dengan median)
        -> normalisasi kategori duplikat (mis. "CC" & "Credit Card")
        -> one-hot encoding kolom kategorikal
        -> split train/test (stratified by target)
        -> data/processed/train.csv, data/processed/test.csv

Parameter (test_size, random_state, nama file & sheet) dibaca dari params.yaml
supaya bisa diubah tanpa menyentuh kode, dan supaya DVC bisa mendeteksi kapan
stage ini perlu dijalankan ulang (lihat bagian `params:` di dvc.yaml).
"""

import logging

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from config import DATA_DIR, PARAMS_PATH, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Kolom target yang akan diprediksi (1 = pelanggan churn, 0 = tidak churn).
TARGET_COL = "Churn"

# Kolom identifier, tidak punya nilai prediktif -> dibuang sebelum training.
ID_COLS = ["CustomerID"]

# Kolom kategorikal (tipe teks) di dataset "E Comm" yang perlu di-encode
# menjadi angka sebelum bisa dipakai model scikit-learn.
CATEGORICAL_COLS = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

# Dataset mentah punya kategori yang maknanya sama tapi ditulis berbeda
# (temuan EDA di notebooks/exploratory_analysis.ipynb). Tanpa digabung,
# one-hot encoding akan memecahnya jadi kolom terpisah (mis.
# "PreferredPaymentMode_CC" dan "PreferredPaymentMode_Credit Card"),
# melemahkan sinyal fitur tersebut ke model.
CATEGORY_NORMALIZATION_MAP = {
    "PreferredPaymentMode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
}


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan kategori duplikat sebelum one-hot encoding."""
    for col, mapping in CATEGORY_NORMALIZATION_MAP.items():
        df[col] = df[col].replace(mapping)
    return df


def load_params() -> dict:
    """Baca sub-bagian `preprocess:` dari params.yaml di root project."""
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)["preprocess"]


def load_raw_data(raw_file: str, sheet_name: str) -> pd.DataFrame:
    """Load dataset mentah dari file Excel (data/raw/<raw_file>, sheet tertentu)."""
    return pd.read_excel(DATA_DIR / raw_file, sheet_name=sheet_name)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Bersihkan & ubah data mentah menjadi bentuk siap-training.

    Langkah:
    1. Drop kolom ID (CustomerID) -- bukan fitur, hanya identifier baris.
    2. Isi missing value kolom numerik dengan median kolom tersebut.
       Beberapa kolom seperti Tenure, WarehouseToHome, HourSpendOnApp, dll
       punya ratusan baris kosong pada dataset asli; median dipilih karena
       tahan terhadap outlier dibanding mean.
    3. Normalisasi kategori duplikat (lihat CATEGORY_NORMALIZATION_MAP)
       sebelum encoding, supaya "CC" dan "Credit Card" dihitung sebagai
       kategori yang sama.
    4. One-hot encode kolom kategorikal (mis. Gender, MaritalStatus) menjadi
       kolom biner 0/1, karena model tree-based scikit-learn butuh input
       numerik. `drop_first=True` membuang satu kategori per kolom untuk
       menghindari kolom yang saling redundan (dummy variable trap).
    """
    df = df.drop(columns=ID_COLS)

    # Semua kolom selain target dan kategorikal dianggap numerik.
    numeric_cols = df.columns.drop([TARGET_COL, *CATEGORICAL_COLS])
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    df = normalize_categories(df)

    return pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)


def main():
    logger.info("=== Stage: preprocess ===")

    params = load_params()
    logger.info("Params loaded: %s", params)

    logger.info(
        "Loading raw data from %s (sheet=%s)",
        DATA_DIR / params["raw_file"],
        params["sheet_name"],
    )
    df = load_raw_data(params["raw_file"], params["sheet_name"])
    logger.info("Raw data shape: %s", df.shape)

    logger.info("Dropping ID column(s): %s", ID_COLS)
    logger.info("Imputing missing numeric values with median")
    logger.info("Normalizing duplicate categories: %s", CATEGORY_NORMALIZATION_MAP)
    logger.info("One-hot encoding categorical columns: %s", CATEGORICAL_COLS)
    df = clean(df)
    logger.info("Cleaned data shape: %s", df.shape)

    # stratify=df[TARGET_COL] menjaga proporsi churn/tidak-churn tetap sama
    # antara train dan test, penting karena target ini imbalanced (~17% churn).
    logger.info(
        "Splitting train/test (test_size=%s, random_state=%s, stratify=%s)",
        params["test_size"],
        params["random_state"],
        TARGET_COL,
    )
    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df[TARGET_COL],
    )

    # Output ini yang dideklarasikan sebagai `outs:` stage preprocess di
    # dvc.yaml -- DVC akan meng-cache isinya dan membuat data/processed/.gitignore
    # otomatis, jadi file CSV-nya sendiri tidak ikut ke Git.
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved train set -> %s (shape=%s)", train_path, train_df.shape)
    logger.info("Saved test set  -> %s (shape=%s)", test_path, test_df.shape)

    logger.info("=== Stage preprocess selesai ===")


if __name__ == "__main__":
    main()
