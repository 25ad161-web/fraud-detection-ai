"""
backend/utils/explainability.py
----------------------------------
Model explainability using SHAP (SHapley Additive exPlanations).

For a given transaction, this answers: "WHY did the model flag this as
fraud (or not)?" by showing which features pushed the prediction towards
fraud vs towards legitimate, and by how much.

SHAP is optional at runtime: if it isn't installed, explain_prediction()
returns a clear, honest message instead of crashing the API. This keeps
the rest of the project (predictions, dashboard, history) fully usable
even in environments where SHAP's native dependencies are awkward to
install (e.g. some restricted/offline machines).
"""

import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap is not installed. Run: pip install shap. "
                    "Explainability endpoint will return a fallback response.")


_explainer_cache = {}


def _get_explainer(model, model_name: str, is_supervised: bool):
    """
    Build (and cache) the right kind of SHAP explainer for the model type.
    TreeExplainer is fast and exact for tree-based models (Random Forest,
    XGBoost). For everything else we fall back to KernelExplainer, which
    is model-agnostic but slower - acceptable here since we only explain
    one transaction at a time, on demand, not in bulk.
    """
    if model_name in _explainer_cache:
        return _explainer_cache[model_name]

    tree_based = model_name in {"Random Forest", "XGBoost"}

    if tree_based:
        explainer = shap.TreeExplainer(model)
    else:
        # KernelExplainer needs a background dataset; a small all-zero
        # baseline is a reasonable, cheap default since our features are
        # standard-scaled (so 0 represents the dataset mean for every column).
        explainer = None  # built lazily per-call with proper background data

    _explainer_cache[model_name] = explainer
    return explainer


def explain_prediction(model, model_name: str, is_supervised: bool,
                        feature_vector: np.ndarray, feature_names: list) -> dict:
    """
    Return a SHAP-based explanation for a single (already scaled) feature
    vector, as a list of {feature, shap_value, abs_value} sorted by impact.

    If SHAP isn't installed, returns a graceful fallback dict instead of
    raising, so the /api/explain endpoint always returns valid JSON.
    """
    if not SHAP_AVAILABLE:
        return {
            "available": False,
            "message": "SHAP is not installed on this server. "
                        "Run 'pip install shap' and restart the backend to enable explainability.",
            "top_features": [],
        }

    try:
        tree_based = model_name in {"Random Forest", "XGBoost"}

        if tree_based:
            explainer = _get_explainer(model, model_name, is_supervised)
            shap_values = explainer.shap_values(feature_vector)
            # For binary classifiers, shap_values may be a list [class0, class1]
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]
        else:
            # Model-agnostic fallback for Logistic Regression / anomaly detectors.
            background = np.zeros((1, feature_vector.shape[1]))
            predict_fn = (
                (lambda x: model.predict_proba(x)[:, 1])
                if is_supervised else
                (lambda x: 1 / (1 + np.exp(model.decision_function(x))))
            )
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(feature_vector, nsamples=100)
            values = np.array(shap_values).flatten()

        contributions = [
            {"feature": name, "shap_value": round(float(val), 5), "abs_value": round(abs(float(val)), 5)}
            for name, val in zip(feature_names, values)
        ]
        contributions.sort(key=lambda x: x["abs_value"], reverse=True)

        return {
            "available": True,
            "message": "Top features influencing this prediction (positive = pushes toward fraud).",
            "top_features": contributions[:10],
        }

    except Exception as e:
        # Explainability is a "nice to have" - never let it break the
        # core prediction flow. Log the real error, return a safe fallback.
        logger.error(f"SHAP explanation failed: {e}")
        return {
            "available": False,
            "message": "Explanation could not be generated for this prediction.",
            "top_features": [],
        }
