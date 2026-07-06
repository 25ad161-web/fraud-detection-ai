"""
backend/utils/preprocessing.py
-------------------------------
Single source of truth for feature engineering / preprocessing logic.

This module is imported by BOTH:
  1. notebooks/01_eda_and_training.py  (at training time)
  2. backend/utils/model_loader.py      (at inference time)

Keeping this logic in one shared place avoids "training/serving skew" -
a very common real-world bug where the API preprocesses incoming data
slightly differently than how the model was trained, silently degrading
accuracy in production.
"""

import numpy as np
import pandas as pd

# Column names expected in the raw Kaggle-style dataset.
RAW_FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV from disk into a DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Strategy:
    - Numeric columns: impute with the column MEDIAN (robust to outliers,
      which is important here since 'Amount' is heavily right-skewed).
    - Drop any row that is missing the target label 'Class' (can't train
      or evaluate without ground truth).
    """
    df = df.copy()

    if TARGET_COLUMN in df.columns:
        before = len(df)
        df = df.dropna(subset=[TARGET_COLUMN])
        dropped = before - len(df)
        if dropped > 0:
            print(f"[preprocessing] Dropped {dropped} rows with missing target label.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            print(f"[preprocessing] Imputed {n_missing} missing values in '{col}' with median={median_val:.4f}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a small number of derived features that help tree-based and
    linear models pick up fraud patterns more easily.

    - hour_of_day: cyclical hour extracted from 'Time' (seconds since first txn)
    - amount_log: log1p transform of Amount to tame the heavy right skew
    """
    df = df.copy()
    if "Time" in df.columns:
        # Time is "seconds since first transaction" in the original dataset.
        # Convert to an hour-of-day proxy (assumes data spans <= a few days).
        df["hour_of_day"] = (df["Time"] // 3600) % 24

    if "Amount" in df.columns:
        df["amount_log"] = np.log1p(df["Amount"])

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return the final list of feature column names used for modeling
    (everything except the target, in a fixed, predictable order).
    """
    exclude = {TARGET_COLUMN}
    return [c for c in df.columns if c not in exclude]


def prepare_features_and_target(df: pd.DataFrame):
    """
    Full feature-prep pipeline (minus scaling, which depends on a
    fitted scaler and is handled separately so the SAME fitted scaler
    can be reused at inference time).

    Returns: (X DataFrame, y Series or None if no target column present)
    """
    df = handle_missing_values(df)
    df = engineer_features(df)

    y = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    return X, y, feature_cols


def build_single_transaction_dataframe(payload: dict) -> pd.DataFrame:
    """
    Convert a single transaction JSON payload (from the API) into a
    one-row DataFrame with the same raw columns as the training data,
    filling any V1-V28 fields not supplied by the user with 0.0
    (a neutral value, since these are PCA components centered at 0).

    This is what allows the frontend form to only ask for a few
    human-meaningful fields (Amount, Time) while still letting power
    users / automated tests supply full V1-V28 vectors if they have them.
    """
    row = {col: 0.0 for col in RAW_FEATURE_COLUMNS}
    row["Time"] = payload.get("time", 0.0)
    row["Amount"] = payload.get("amount", 0.0)

    for i in range(1, 29):
        key = f"V{i}"
        if key in payload:
            row[key] = payload[key]
        elif key.lower() in payload:
            row[key] = payload[key.lower()]

    return pd.DataFrame([row])
