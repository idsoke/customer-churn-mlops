import pandas as pd

from train import TARGET_COL, split_xy


def test_split_xy_separates_target_from_features():
    df = pd.DataFrame({"feature_a": [1, 2, 3], "feature_b": [4, 5, 6], TARGET_COL: [0, 1, 0]})

    X, y = split_xy(df)

    assert TARGET_COL not in X.columns
    assert list(X.columns) == ["feature_a", "feature_b"]
    assert y.tolist() == [0, 1, 0]
