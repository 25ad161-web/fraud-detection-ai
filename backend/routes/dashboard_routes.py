"""
backend/routes/dashboard_routes.py
-------------------------------------
REST API endpoints powering the frontend dashboard.

  GET /api/stats    -> aggregate fraud stats for dashboard cards/charts
  GET /api/history   -> recent prediction history (fraud history table)
  GET /api/model-info -> which model is deployed + its evaluation metrics
  GET /api/charts     -> list of available EDA/evaluation chart filenames
"""

import os
import json

from flask import Blueprint, jsonify, request

from backend import config
from backend.utils.history_store import get_history, get_stats
from backend.utils.logger import get_logger

logger = get_logger(__name__)

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/api")


@dashboard_bp.route("/stats", methods=["GET"])
def stats():
    """Aggregate fraud detection stats, computed from prediction history."""
    try:
        return jsonify(get_stats()), 200
    except Exception as e:
        logger.exception("Error computing stats")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@dashboard_bp.route("/history", methods=["GET"])
def history():
    """Recent prediction history. Optional ?limit=N query param (default 50)."""
    try:
        limit = request.args.get("limit", default=50, type=int)
        limit = max(1, min(limit, config.MAX_HISTORY_RECORDS))
        return jsonify({"history": get_history(limit=limit)}), 200
    except Exception as e:
        logger.exception("Error fetching history")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@dashboard_bp.route("/model-info", methods=["GET"])
def model_info():
    """Return metadata about the currently deployed model and its training metrics."""
    try:
        if not os.path.exists(config.METADATA_PATH):
            return jsonify({"error": "Model metadata not found. Train the model first."}), 503

        with open(config.METADATA_PATH) as f:
            metadata = json.load(f)

        comparison = None
        comparison_csv = os.path.join(config.MODELS_DIR, "model_comparison.csv")
        if os.path.exists(comparison_csv):
            import pandas as pd  # local import - only needed for this optional field
            comparison = pd.read_csv(comparison_csv, index_col=0).round(4).to_dict(orient="index")

        return jsonify({"deployed_model": metadata, "all_models_comparison": comparison}), 200
    except Exception as e:
        logger.exception("Error fetching model info")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@dashboard_bp.route("/charts", methods=["GET"])
def charts():
    """List available chart image filenames (served statically from /static/charts/)."""
    try:
        if not os.path.exists(config.CHARTS_DIR):
            return jsonify({"charts": []}), 200
        files = sorted(f for f in os.listdir(config.CHARTS_DIR) if f.endswith(".png"))
        return jsonify({"charts": files}), 200
    except Exception as e:
        logger.exception("Error listing charts")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
