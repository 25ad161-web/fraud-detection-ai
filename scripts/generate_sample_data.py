"""
generate_sample_data.py
------------------------
Generates a SYNTHETIC dataset that mimics the schema and statistical
properties of the Kaggle "Credit Card Fraud Detection" dataset
(https://www.kaggle.com/datasets/mlg-ulb/ulb-machine-learning-group/creditcardfraud).

WHY THIS EXISTS:
The real Kaggle dataset requires a Kaggle account + API token to download.
To make this project runnable out-of-the-box for grading / demo purposes,
this script creates a same-schema synthetic dataset:
    Time, V1-V28 (PCA-like features), Amount, Class
with a realistic class imbalance (~0.17% fraud), similar to the real data.

USAGE:
    python scripts/generate_sample_data.py

If you have the real Kaggle file, just drop `creditcard.csv` into data/raw/
and this script will NOT overwrite it.
"""

import os
import numpy as np
import pandas as pd

# Reproducibility
RNG = np.random.default_rng(42)

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_PATH = os.path.join(RAW_DIR, "creditcard.csv")

N_NORMAL = 50000      # number of legitimate transactions to simulate
N_FRAUD = 90          # ~0.18% fraud rate, similar to the real dataset
N_FEATURES = 28       # V1...V28, mimicking PCA components


def generate_normal_transactions(n):
    """Simulate legitimate transactions: V1-V28 ~ tight Gaussian noise,
    moderate transaction amounts, spread across a realistic time window."""
    v_features = RNG.normal(loc=0.0, scale=1.0, size=(n, N_FEATURES))
    amount = np.abs(RNG.gamma(shape=2.0, scale=40.0, size=n))  # right-skewed amounts
    time = RNG.uniform(0, 172800, size=n)  # 2 days, in seconds, like the real dataset
    return v_features, amount, time


def generate_fraud_transactions(n):
    """Simulate fraudulent transactions: shifted/more volatile V-features
    and a different amount distribution (frauds are often small 'test' amounts
    or unusually large one-off amounts)."""
    v_features = RNG.normal(loc=2.5, scale=3.0, size=(n, N_FEATURES))
    # Bimodal amounts: many tiny test transactions, a few large ones
    small = np.abs(RNG.normal(loc=5, scale=3, size=n // 2))
    large = np.abs(RNG.normal(loc=500, scale=150, size=n - n // 2))
    amount = np.concatenate([small, large])
    RNG.shuffle(amount)
    time = RNG.uniform(0, 172800, size=n)
    return v_features, amount, time


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        print(f"[INFO] '{OUTPUT_PATH}' already exists. Skipping generation "
              f"to avoid overwriting a real dataset. Delete it manually to regenerate.")
        return

    normal_v, normal_amt, normal_time = generate_normal_transactions(N_NORMAL)
    fraud_v, fraud_amt, fraud_time = generate_fraud_transactions(N_FRAUD)

    v_cols = [f"V{i}" for i in range(1, N_FEATURES + 1)]

    df_normal = pd.DataFrame(normal_v, columns=v_cols)
    df_normal["Time"] = normal_time
    df_normal["Amount"] = normal_amt
    df_normal["Class"] = 0

    df_fraud = pd.DataFrame(fraud_v, columns=v_cols)
    df_fraud["Time"] = fraud_time
    df_fraud["Amount"] = fraud_amt
    df_fraud["Class"] = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle rows

    # Reorder columns to match the real Kaggle schema: Time, V1..V28, Amount, Class
    ordered_cols = ["Time"] + v_cols + ["Amount", "Class"]
    df = df[ordered_cols]

    # Inject a small number of missing values intentionally so the
    # preprocessing pipeline's missing-value handling has something to do
    # (the real Kaggle dataset has none, but production data often does).
    missing_mask = RNG.random(size=df.shape[0]) < 0.001
    df.loc[missing_mask, "Amount"] = np.nan

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Synthetic dataset written to {OUTPUT_PATH}")
    print(f"     Shape: {df.shape}, Fraud rate: {df['Class'].mean() * 100:.4f}%")


if __name__ == "__main__":
    main()
