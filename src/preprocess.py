"""
src/preprocess.py
Tiền xử lý dữ liệu tiểu đường:
  - Thay thế giá trị 0 không hợp lệ bằng median
  - Tách features / target
  - Chia train / test
  - Chuẩn hóa (StandardScaler)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Các cột không được phép có giá trị 0 (về mặt sinh học)
ZERO_INVALID_COLS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

TARGET_COL = "Outcome"


def load_raw(path: str) -> pd.DataFrame:
    """Đọc file CSV thô."""
    df = pd.read_csv(path)
    print(f"[INFO] Dataset shape: {df.shape}")
    return df


def fix_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Thay thế 0 không hợp lệ bằng median của cột đó."""
    df = df.copy()
    for col in ZERO_INVALID_COLS:
        if col not in df.columns:
            continue
        n_zeros = (df[col] == 0).sum()
        if n_zeros:
            median_val = df.loc[df[col] != 0, col].median()
            df.loc[df[col] == 0, col] = median_val
            print(f"  [FIX] {col}: thay {n_zeros} giá trị 0 → median={median_val:.2f}")
    return df


def split_features_target(df: pd.DataFrame):
    """Tách X (features) và y (target)."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def preprocess(
    path: str,
    test_size: float = 0.20,
    random_state: int = 42,
    scale: bool = True,
):
    """
    Pipeline tiền xử lý hoàn chỉnh.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    feature_names                     : list[str]
    scaler                            : StandardScaler hoặc None
    """
    df = load_raw(path)
    df = fix_invalid_zeros(df)
    X, y = split_features_target(df)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        print("[INFO] Đã chuẩn hóa dữ liệu (StandardScaler)")
    else:
        X_train = X_train.values
        X_test = X_test.values

    y_train = y_train.values
    y_test = y_test.values

    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")
    preprocess(data_path)
