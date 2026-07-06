"""
backend/routes/predict_routes.py
-----------------------------------
REST API endpoints for fraud prediction.

  POST /api/predict        -> predict fraud for a single transaction
  POST /api/predict/batch  -> predict fraud for a list of transactions
  POST /api/explain        -> SHAP explanation for a single transaction
"""

from flask import Blueprint, request, jsonify
import numpy as np

from backend.utils.model_loader import fraud_model, ModelNotTrainedError
from backend.utils.history_store import add_record
from backend.utils.explainability import explain_prediction
from backend.utils.logger import get_logger

logger = get_logger(__name__)

predict_bp = Blueprint("predict", __name__, url_prefix="/api")


def _validate_transaction_payload(data: dict):
    """
    Basic input validation for a transaction payload. Returns an error
    message string if invalid, or None if the payload is acceptable.
    Only 'amount' is strictly required - V1-V28 default to 0 if omitted,
    since most demo users won't have real PCA-anonymized feature values.
    """
    if data is None:
        return "Request body must be JSON."
    if "amount" not in data:
        return "Field 'amount' is required."
    try:
        amount = float(data["amount"])
        if amount < 0:
            return "Field 'amount' must be non-negative."
    except (TypeError, ValueError):
        return "Field 'amount' must be a number."
    return None


@predict_bp.route("/predict", methods=["POST"])
def predict_single():
    """
    Predict fraud for a single transaction.

    Expected JSON body (minimum):
        { "amount": 250.0, "time": 43200 }
    Optionally also accepts V1..V28 numeric fields for full fidelity.
    """
    try:
        data = request.get_json(silent=True)
        error = _validate_transaction_payload(data)
        if error:
            logger.warning(f"Bad request to /predict: {error}")
            return jsonify({"error": error}), 400

        result = fraud_model.predict_one(data)
        add_record(data, result)

        logger.info(f"Prediction: amount={data.get('amount')} -> "
                    f"fraud={result['is_fraud']} risk={result['risk_score']}")
        return jsonify(result), 200

    except ModelNotTrainedError as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Unexpected error in /predict")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@predict_bp.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict fraud for multiple transactions at once.

    Expected JSON body:
        { "transactions": [ {"amount": 100, "time": 1000}, {...}, ... ] }
    """
    try:
        data = request.get_json(silent=True)
        if not data or "transactions" not in data or not isinstance(data["transactions"], list):
            return jsonify({"error": "Request must include a 'transactions' list."}), 400

        transactions = data["transactions"]
        if len(transactions) == 0:
            return jsonify({"error": "'transactions' list cannot be empty."}), 400
        if len(transactions) > 500:
            return jsonify({"error": "Batch size limited to 500 transactions per request."}), 400

        results = []
        for i, txn in enumerate(transactions):
            error = _validate_transaction_payload(txn)
            if error:
                results.append({"index": i, "error": error})
                continue
            prediction = fraud_model.predict_one(txn)
            add_record(txn, prediction)
            results.append({"index": i, **prediction})

        logger.info(f"Batch prediction completed for {len(transactions)} transactions.")
        return jsonify({"results": results}), 200

    except ModelNotTrainedError as e:
        logger.error(str(e))
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Unexpected error in /predict/batch")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@predict_bp.route("/explain", methods=["POST"])
def explain():
    """
    Return a SHAP-based explanation for why a transaction was (or wasn't)
    flagged as fraud. Same payload shape as /predict.
    """
    try:
        data = request.get_json(silent=True)
        error = _validate_transaction_payload(data)
        if error:
            return jsonify({"error": error}), 400

        fraud_model.ensure_loaded()
        feature_vector = fraud_model._to_feature_vector(data)

        explanation = explain_prediction(
            model=fraud_model.model,
            model_name=fraud_model.metadata.get("model_name"),
            is_supervised=fraud_model.is_supervised,
            feature_vector=feature_vector,
            feature_names=fraud_model.feature_columns,
        )
        return jsonify(explanation), 200

    except ModelNotTrainedError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        logger.exception("Unexpected error in /explain")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
