"""
notebooks/01_eda_and_training.py
----------------------------------
End-to-end EDA + preprocessing + model training + evaluation script for the
AI-Enabled Fraud Detection project.

This is written as a plain .py script (not a .ipynb) so it can be version
controlled cleanly and run from the command line, but it is structured in
clearly separated, numbered sections - exactly like notebook cells - so you
can also paste sections into Jupyter / VS Code interactive cells if you prefer.

WHAT THIS SCRIPT DOES (in order):
  1. Load the raw Kaggle Credit Card Fraud dataset
  2. Exploratory Data Analysis (EDA) with saved plots
  3. Preprocessing: missing values, feature engineering, train/test split, scaling
  4. Handle class imbalance using SMOTE (on the TRAINING set only)
  5. Train 5 models: Isolation Forest, Local Outlier Factor, Random Forest,
     XGBoost, Logistic Regression
  6. Evaluate every model: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
  7. Pick the best model (by F1-score on the fraud class) and save it + the
     fitted scaler + a metrics report, so the Flask backend can load them later.

RUN:
    python notebooks/01_eda_and_training.py
"""

import os
import sys
import json
import warnings
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # render plots to files, no GUI needed (works on servers too)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
import joblib

warnings.filterwarnings("ignore")

# Make the backend.utils package importable from this script's location
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.utils.preprocessing import (
    load_raw_dataset, handle_missing_values, engineer_features,
    get_feature_columns, TARGET_COLUMN
)

# ----------------------------------------------------------------------------
# Optional dependencies: SMOTE (imblearn), XGBoost, SHAP.
# These are standard, well-documented libraries that WILL be available once
# you `pip install -r requirements.txt`. We guard the imports so that this
# script still runs (with reduced functionality + a clear warning) in
# environments where they happen not to be installed yet.
# ----------------------------------------------------------------------------
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("[WARN] imbalanced-learn not installed. Run: pip install imbalanced-learn")
    print("       Falling back to manual random oversampling of the minority class.")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] xgboost not installed. Run: pip install xgboost")
    print("       XGBoost will be skipped in model comparison.")


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "creditcard.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CHARTS_DIR = os.path.join(PROJECT_ROOT, "backend", "static", "charts")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

RANDOM_STATE = 42


def section(title):
    """Pretty-print a section header so console output is easy to scan."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================================
# 1. LOAD DATA
# ============================================================================
section("1. LOADING RAW DATASET")

if not os.path.exists(RAW_DATA_PATH):
    print(f"[ERROR] Could not find dataset at {RAW_DATA_PATH}")
    print("        Place creditcard.csv in data/raw/, or run:")
    print("        python scripts/generate_sample_data.py")
    sys.exit(1)

df_raw = load_raw_dataset(RAW_DATA_PATH)
print(f"Loaded dataset with shape: {df_raw.shape}")
print(df_raw.head())


# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
section("2. EXPLORATORY DATA ANALYSIS")

print("\n--- Basic Info ---")
print(df_raw.info())

print("\n--- Missing Values per Column ---")
missing = df_raw.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

print("\n--- Class Distribution ---")
class_counts = df_raw[TARGET_COLUMN].value_counts()
fraud_pct = (class_counts.get(1, 0) / len(df_raw)) * 100
print(class_counts)
print(f"Fraud rate: {fraud_pct:.4f}%")

# --- Plot 1: Class imbalance bar chart ---
plt.figure(figsize=(6, 4))
sns.countplot(x=TARGET_COLUMN, data=df_raw, hue=TARGET_COLUMN, palette=["#2563eb", "#dc2626"], legend=False)
plt.title("Class Distribution: Legitimate (0) vs Fraud (1)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.yscale("log")  # log scale needed - fraud is a tiny sliver otherwise invisible
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "class_distribution.png"), dpi=120)
plt.close()
print(f"[Saved] class_distribution.png")

# --- Plot 2: Transaction amount distribution by class ---
plt.figure(figsize=(8, 4))
sns.histplot(df_raw[df_raw[TARGET_COLUMN] == 0]["Amount"], bins=50, color="#2563eb",
             label="Legitimate", stat="density", alpha=0.6)
sns.histplot(df_raw[df_raw[TARGET_COLUMN] == 1]["Amount"], bins=50, color="#dc2626",
             label="Fraud", stat="density", alpha=0.6)
plt.xlim(0, 500)  # zoom into the bulk of the distribution
plt.title("Transaction Amount Distribution by Class")
plt.xlabel("Amount")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "amount_distribution.png"), dpi=120)
plt.close()
print(f"[Saved] amount_distribution.png")

# --- Plot 3: Correlation heatmap of top features with target ---
corr_with_target = df_raw.corr(numeric_only=True)[TARGET_COLUMN].drop(TARGET_COLUMN)
top_features = corr_with_target.abs().sort_values(ascending=False).head(10).index.tolist()

plt.figure(figsize=(8, 6))
sns.heatmap(df_raw[top_features + [TARGET_COLUMN]].corr(numeric_only=True),
            annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap: Top 10 Features Most Correlated with Fraud")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "correlation_heatmap.png"), dpi=120)
plt.close()
print(f"[Saved] correlation_heatmap.png")

# --- Plot 4: Transactions over time, by class ---
plt.figure(figsize=(10, 4))
plt.scatter(df_raw[df_raw[TARGET_COLUMN] == 0]["Time"],
            df_raw[df_raw[TARGET_COLUMN] == 0]["Amount"],
            s=2, alpha=0.2, color="#2563eb", label="Legitimate")
plt.scatter(df_raw[df_raw[TARGET_COLUMN] == 1]["Time"],
            df_raw[df_raw[TARGET_COLUMN] == 1]["Amount"],
            s=15, alpha=0.8, color="#dc2626", label="Fraud")
plt.title("Transaction Amount Over Time, by Class")
plt.xlabel("Time (seconds since first transaction)")
plt.ylabel("Amount")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "time_vs_amount.png"), dpi=120)
plt.close()
print(f"[Saved] time_vs_amount.png")


# ============================================================================
# 3. PREPROCESSING: missing values, feature engineering, split, scaling
# ============================================================================
section("3. PREPROCESSING")

df_clean = handle_missing_values(df_raw)
df_clean = engineer_features(df_clean)

feature_cols = get_feature_columns(df_clean)
X = df_clean[feature_cols]
y = df_clean[TARGET_COLUMN]

print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

# Stratified split keeps the same fraud ratio in both train and test sets -
# critical for imbalanced data, otherwise the test set could end up with
# zero (or wildly differing) fraud examples by chance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train fraud rate: {y_train.mean()*100:.4f}%, Test fraud rate: {y_test.mean()*100:.4f}%")

# Feature scaling: fit the scaler ONLY on training data, then apply to both.
# Fitting on the full dataset (including test) would leak test-set
# distribution information into training - a common and serious bug.
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=feature_cols, index=X_test.index
)

# Save the scaler now - the Flask backend needs this EXACT fitted scaler
# to transform incoming transactions the same way at inference time.
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"[Saved] {scaler_path}")

# Save the feature column order too - inference must build feature vectors
# in this exact order, or predictions will be silently wrong.
feature_cols_path = os.path.join(MODELS_DIR, "feature_columns.json")
with open(feature_cols_path, "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"[Saved] {feature_cols_path}")

# Persist processed splits to disk (useful for reproducibility / debugging)
X_train_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
X_test_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)
print("[Saved] processed train/test splits to data/processed/")


# ============================================================================
# 4. HANDLE CLASS IMBALANCE WITH SMOTE (training set only!)
# ============================================================================
section("4. HANDLING CLASS IMBALANCE (SMOTE)")

# IMPORTANT: SMOTE is applied ONLY to the training set, never to the test
# set. The test set must reflect the real-world imbalanced distribution,
# otherwise evaluation metrics would be meaningless (we'd be "testing" on
# synthetic data that doesn't represent real incoming transactions).
if SMOTE_AVAILABLE:
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print("Applied SMOTE (Synthetic Minority Over-sampling Technique).")
else:
    # Fallback: simple random oversampling with replacement of the minority
    # class, so the script still produces a usable result without imblearn.
    fraud_idx = y_train[y_train == 1].index
    legit_idx = y_train[y_train == 0].index
    fraud_upsampled_idx = np.random.choice(fraud_idx, size=len(legit_idx), replace=True)
    resampled_idx = np.concatenate([legit_idx, fraud_upsampled_idx])
    X_train_res = X_train_scaled.loc[resampled_idx].reset_index(drop=True)
    y_train_res = y_train.loc[resampled_idx].reset_index(drop=True)
    print("Applied fallback random oversampling (install imbalanced-learn for true SMOTE).")

print(f"Before resampling: {y_train.value_counts().to_dict()}")
print(f"After resampling:  {pd.Series(y_train_res).value_counts().to_dict()}")


# ============================================================================
# 5. TRAIN MODELS
# ============================================================================
section("5. TRAINING MODELS")

results = {}        # model_name -> metrics dict
trained_models = {} # model_name -> fitted model object

# ---- 5a. Logistic Regression (supervised, trained on SMOTE-balanced data) ----
print("\n[Training] Logistic Regression...")
t0 = time.time()
log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_res, y_train_res)
trained_models["Logistic Regression"] = log_reg
print(f"  done in {time.time()-t0:.1f}s")

# ---- 5b. Random Forest (supervised, trained on SMOTE-balanced data) ----
print("\n[Training] Random Forest...")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train_res, y_train_res)
trained_models["Random Forest"] = rf
print(f"  done in {time.time()-t0:.1f}s")

# ---- 5c. XGBoost (supervised, trained on SMOTE-balanced data) ----
if XGBOOST_AVAILABLE:
    print("\n[Training] XGBoost...")
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb.fit(X_train_res, y_train_res)
    trained_models["XGBoost"] = xgb
    print(f"  done in {time.time()-t0:.1f}s")
else:
    print("\n[Skipped] XGBoost (not installed)")

# ---- 5d. Isolation Forest (UNSUPERVISED anomaly detector) ----
# Trained on the ORIGINAL imbalanced training data (not SMOTE-resampled),
# since it learns what "normal" looks like and flags deviations from it -
# feeding it synthetic fraud examples would distort that "normal" baseline.
print("\n[Training] Isolation Forest...")
t0 = time.time()
iso_forest = IsolationForest(
    contamination=y_train.mean(),  # expected proportion of anomalies
    random_state=RANDOM_STATE, n_jobs=-1
)
iso_forest.fit(X_train_scaled)
trained_models["Isolation Forest"] = iso_forest
print(f"  done in {time.time()-t0:.1f}s")

# ---- 5e. Local Outlier Factor (UNSUPERVISED, novelty detection mode) ----
# novelty=True lets us call .predict() on new/unseen data after fitting -
# by default LOF only supports fit_predict on the same data it was fit on.
print("\n[Training] Local Outlier Factor...")
t0 = time.time()
lof = LocalOutlierFactor(
    n_neighbors=20, contamination=y_train.mean(), novelty=True, n_jobs=-1
)
lof.fit(X_train_scaled)
trained_models["Local Outlier Factor"] = lof
print(f"  done in {time.time()-t0:.1f}s")


# ============================================================================
# 6. EVALUATE MODELS
# ============================================================================
section("6. EVALUATING MODELS")

def evaluate_supervised(name, model, X_test, y_test):
    """Evaluate a standard supervised classifier (predict_proba available)."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics, y_pred, y_proba


def evaluate_unsupervised(name, model, X_test, y_test):
    """
    Evaluate an unsupervised anomaly detector (Isolation Forest / LOF).
    These models output -1 for anomaly, 1 for normal - we map that to our
    1=fraud, 0=legit convention. They also expose decision_function /
    score_samples instead of predict_proba, which we min-max normalize
    into a pseudo-probability for ROC-AUC and for displaying a "risk score".
    """
    raw_pred = model.predict(X_test)              # -1 = anomaly, 1 = normal
    y_pred = np.where(raw_pred == -1, 1, 0)        # convert to 1 = fraud, 0 = legit

    raw_scores = model.decision_function(X_test)   # higher = more "normal"
    # Flip and normalize so higher = more "fraud-like", matching predict_proba style
    risk_scores = -raw_scores
    risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-9)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, risk_scores),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics, y_pred, risk_scores


supervised_names = {"Logistic Regression", "Random Forest", "XGBoost"}

for name, model in trained_models.items():
    print(f"\n--- {name} ---")
    if name in supervised_names:
        metrics, y_pred, y_proba = evaluate_supervised(name, model, X_test_scaled, y_test)
    else:
        metrics, y_pred, y_proba = evaluate_unsupervised(name, model, X_test_scaled, y_test)

    results[name] = metrics
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    # Save a confusion matrix heatmap per model
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(np.array(metrics["confusion_matrix"]), annot=True, fmt="d",
                cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(CHARTS_DIR, f"confusion_matrix_{safe_name}.png"), dpi=120)
    plt.close()


# --- Combined model comparison bar chart ---
metrics_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
print("\n--- Model Comparison Table ---")
print(metrics_df.round(4))

metrics_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"))

plt.figure(figsize=(11, 5))
metrics_df.plot(kind="bar", ax=plt.gca(), colormap="viridis")
plt.title("Model Comparison Across Metrics")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "model_comparison.png"), dpi=120)
plt.close()
print("[Saved] model_comparison.png")

# --- Combined ROC curves (supervised models only - they have clean probas) ---
plt.figure(figsize=(7, 6))
for name in supervised_names:
    if name not in trained_models:
        continue
    model = trained_models[name]
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: Supervised Models")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "roc_curves.png"), dpi=120)
plt.close()
print("[Saved] roc_curves.png")


# ============================================================================
# 7. SELECT BEST MODEL AND SAVE
# ============================================================================
section("7. SELECTING AND SAVING THE BEST MODEL")

# We select by F1-score on the fraud class, since for highly imbalanced
# fraud detection, accuracy is misleading (predicting "never fraud" gives
# ~99.8% accuracy but is useless) - F1 balances precision and recall, which
# is exactly the precision/recall tradeoff that matters operationally:
# missing fraud (false negatives) vs annoying customers with false alarms
# (false positives).
best_model_name = max(results, key=lambda name: results[name]["f1_score"])
best_model = trained_models[best_model_name]
best_metrics = results[best_model_name]

print(f"Best model selected: {best_model_name}")
print(f"  F1-score : {best_metrics['f1_score']:.4f}")
print(f"  Precision: {best_metrics['precision']:.4f}")
print(f"  Recall   : {best_metrics['recall']:.4f}")
print(f"  ROC-AUC  : {best_metrics['roc_auc']:.4f}")

best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
joblib.dump(best_model, best_model_path)
print(f"[Saved] {best_model_path}")

is_supervised = best_model_name in supervised_names
metadata = {
    "model_name": best_model_name,
    "is_supervised": is_supervised,
    "metrics": best_metrics,
    "feature_columns": feature_cols,
    "trained_at": pd.Timestamp.now().isoformat(),
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
}
metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"[Saved] {metadata_path}")

# Also save ALL trained models (not just the best) - useful for the
# /api/compare endpoint in the backend and for SHAP explainability later.
all_models_path = os.path.join(MODELS_DIR, "all_models.pkl")
joblib.dump(trained_models, all_models_path)
print(f"[Saved] {all_models_path}")

# Full classification report for the best model, for the README / writeup
if is_supervised:
    y_pred_best = best_model.predict(X_test_scaled)
else:
    raw_pred = best_model.predict(X_test_scaled)
    y_pred_best = np.where(raw_pred == -1, 1, 0)

print("\n--- Final Classification Report (Best Model) ---")
print(classification_report(y_test, y_pred_best, target_names=["Legit", "Fraud"]))

section("TRAINING COMPLETE")
print(f"Best model: {best_model_name}")
print(f"All artifacts saved to: {MODELS_DIR}")
print(f"All charts saved to: {CHARTS_DIR}")
