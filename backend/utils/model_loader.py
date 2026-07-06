"""
backend/utils/model_loader.py
-------------------------------
Loads the trained model, scaler, feature column order, and metadata ONCE
at app startup, and exposes a single predict_transaction() function used
by the API routes.

Centralizing this here (rather than loading files inside each route)
avoids re-reading pickle files from disk on every request, and ensures
the SAME preprocessing path used in training (backend/utils/preprocessing.py)
is reused at inference time - preventing training/serving skew.
"""

import os
import json
import numpy as np
import joblib

from backend import config
from backend.utils.preprocessing import engineer_features, build_single_transaction_dataframe
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ModelNotTrainedError(Exception):
    """Raised when prediction is attempted before model artifacts exist on disk."""
    pass


class FraudModel:
    """
    Wraps the trained model + scaler + feature columns + metadata,
    and exposes a clean .predict_one(payload) API for the Flask routes.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.metadata = None
        self.is_supervised = True
        self._loaded = False

    def load(self):
        """Load all model artifacts from disk. Raises ModelNotTrainedError if missing."""
        missing = [p for p in [
            config.BEST_MODEL_PATH, config.SCALER_PATH,
            config.FEATURE_COLUMNS_PATH, config.METADATA_PATH
        ] if not os.path.exists(p)]

        if missing:
            raise ModelNotTrainedError(
                "Model artifacts not found: "
                f"{missing}. Run 'python notebooks/01_eda_and_training.py' first."
            )

        self.model = joblib.load(config.BEST_MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)

        with open(config.FEATURE_COLUMNS_PATH) as f:
            self.feature_columns = json.load(f)

        with open(config.METADATA_PATH) as f:
            self.metadata = json.load(f)

        self.is_supervised = self.metadata.get("is_supervised", True)
        self._loaded = True
        logger.info(f"Loaded model '{self.metadata.get('model_name')}' "
                    f"(supervised={self.is_supervised}) with {len(self.feature_columns)} features.")

    def ensure_loaded(self):
        if not self._loaded:
            self.load()

    def _to_feature_vector(self, payload: dict):
        """Convert a raw API payload into a correctly-ordered, scaled feature vector."""
        df = build_single_transaction_dataframe(payload)
        df = engineer_features(df)

        # Reindex to the EXACT column order used at training time. Any
        # engineered column the model expects but isn't present gets
        # filled with 0 (a neutral default) instead of raising an error,
        # so the API stays robust to minor payload variation.
        df = df.reindex(columns=self.feature_columns, fill_value=0.0)

        # Keep it as a DataFrame (not a bare numpy array) through scaling,
        # since the scaler and downstream model were both fit on DataFrames
        # with named columns - passing a plain array triggers a harmless but
        # noisy sklearn UserWarning ("X does not have valid feature names").
        import pandas as pd
        scaled = self.scaler.transform(df)
        scaled_df = pd.DataFrame(scaled, columns=self.feature_columns)
        return scaled_df

    def predict_one(self, payload: dict) -> dict:
        """
        Run a single transaction through the model and return a structured
        result dict: prediction label, risk score (0-1), and model name used.
        """
        self.ensure_loaded()
        X = self._to_feature_vector(payload)

        if self.is_supervised:
            pred = int(self.model.predict(X)[0])
            proba = float(self.model.predict_proba(X)[0, 1])
        else:
            # Unsupervised anomaly detectors: -1 = anomaly/fraud, 1 = normal.
            raw_pred = self.model.predict(X)[0]
            pred = 1 if raw_pred == -1 else 0
            raw_score = self.model.decision_function(X)[0]  # higher = more normal
            # Squash into a pseudo-probability in [0, 1] via a logistic-style
            # transform, since these models don't natively output probabilities.
            proba = float(1 / (1 + np.exp(raw_score)))

        risk_label = (
            "High" if proba >= config.RISK_THRESHOLD_HIGH else
            "Medium" if proba >= config.RISK_THRESHOLD_MEDIUM else
            "Low"
        )

        return {
            "is_fraud": bool(pred),
            "risk_score": round(proba, 4),
            "risk_label": risk_label,
            "model_used": self.metadata.get("model_name"),
        }


# Module-level singleton, imported by route files. Loaded lazily on first
# request (or explicitly at app startup in app.py) - not eagerly at import
# time, so importing this module never fails even if the model hasn't been
# trained yet (the error is raised clearly when prediction is attempted).
fraud_model = FraudModel()
