import numpy as np
import pandas as pd
import pytest

from preprocess import CATEGORICAL_COLS, ID_COLS, TARGET_COL, clean, normalize_categories


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4, 5],
            "Churn": [0, 1, 0, 1, 0],
            "Tenure": [1.0, np.nan, 3.0, 4.0, 2.0],
            "CityTier": [1, 2, 3, 1, 2],
            "PreferredLoginDevice": ["Mobile Phone", "Phone", "Computer", "Mobile Phone", "Phone"],
            "PreferredPaymentMode": ["Debit Card", "Credit Card", "Debit Card", "UPI", "CC"],
            "Gender": ["Male", "Female", "Male", "Female", "Male"],
            "PreferedOrderCat": ["Laptop & Accessory", "Mobile", "Fashion", "Grocery", "Mobile Phone"],
            "MaritalStatus": ["Single", "Married", "Divorced", "Single", "Married"],
        }
    )


def test_clean_drops_id_columns(raw_df):
    result = clean(raw_df)
    for col in ID_COLS:
        assert col not in result.columns


def test_clean_imputes_missing_numeric_values_with_median(raw_df):
    result = clean(raw_df)
    assert not result["Tenure"].isna().any()
    assert result.loc[1, "Tenure"] == raw_df["Tenure"].median()


def test_clean_one_hot_encodes_categorical_columns(raw_df):
    result = clean(raw_df)
    for col in CATEGORICAL_COLS:
        assert col not in result.columns
    assert result.select_dtypes(include="object").empty


def test_clean_keeps_target_column_intact(raw_df):
    result = clean(raw_df)
    assert TARGET_COL in result.columns
    assert not result[TARGET_COL].isna().any()
    assert result[TARGET_COL].tolist() == raw_df[TARGET_COL].tolist()


def test_clean_output_has_no_missing_values_at_all(raw_df):
    result = clean(raw_df)
    assert not result.isna().any().any()


def test_normalize_categories_merges_cc_into_credit_card(raw_df):
    result = normalize_categories(raw_df.copy())
    assert "CC" not in result["PreferredPaymentMode"].values
    # Baris index 1 ("Credit Card" asli) dan index 4 ("CC" asli) harus
    # jadi nilai yang identik setelah normalisasi.
    assert result.loc[1, "PreferredPaymentMode"] == "Credit Card"
    assert result.loc[4, "PreferredPaymentMode"] == "Credit Card"


def test_normalize_categories_merges_mobile_into_mobile_phone(raw_df):
    result = normalize_categories(raw_df.copy())
    assert "Mobile" not in result["PreferedOrderCat"].values
    # Baris index 1 ("Mobile" asli) dan index 4 ("Mobile Phone" asli) harus
    # jadi nilai yang identik setelah normalisasi.
    assert result.loc[1, "PreferedOrderCat"] == "Mobile Phone"
    assert result.loc[4, "PreferedOrderCat"] == "Mobile Phone"


def test_clean_produces_same_encoding_for_merged_categories(raw_df):
    # Setelah clean(), baris dengan kategori yang sudah digabung ("Credit
    # Card" vs "CC", "Mobile Phone" vs "Mobile") harus punya nilai one-hot
    # yang identik untuk seluruh kolom PreferredPaymentMode_* dan
    # PreferedOrderCat_*, karena secara semantik itu kategori yang sama.
    result = clean(raw_df)

    payment_cols = [c for c in result.columns if c.startswith("PreferredPaymentMode_")]
    order_cols = [c for c in result.columns if c.startswith("PreferedOrderCat_")]

    assert result.loc[1, payment_cols].tolist() == result.loc[4, payment_cols].tolist()
    assert result.loc[1, order_cols].tolist() == result.loc[4, order_cols].tolist()
