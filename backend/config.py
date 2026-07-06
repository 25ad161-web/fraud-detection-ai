"""
backend/config.py
-------------------
Central configuration for the Flask app: file paths, server settings,
and tunable thresholds. Keeping these in one place (instead of scattered
magic numbers/strings across route files) makes the project easier to
deploy to different environments and easier for a reviewer to audit.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
ALL_MODELS_PATH = os.path.join(MODELS_DIR, "all_models.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.json")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

# Fraud history is stored as a simple JSON file for this project's scope.
# In a real production system this would be a database (Postgres, etc.),
# but a JSON file keeps the project dependency-free and easy to inspect
# for a student / portfolio project while still being a real persistence layer.
HISTORY_PATH = os.path.join(DATA_DIR, "processed", "prediction_history.json")
MAX_HISTORY_RECORDS = 500  # cap so the file doesn't grow unbounded in a demo

# Risk score thresholds used to label predictions for the UI
RISK_THRESHOLD_HIGH = 0.7
RISK_THRESHOLD_MEDIUM = 0.4

HOST = "0.0.0.0"
PORT = 5000
DEBUG = True
