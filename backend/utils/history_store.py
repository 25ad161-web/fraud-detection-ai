"""
backend/utils/history_store.py
---------------------------------
Lightweight JSON-file-backed storage for prediction history, used to power
the "Fraud History" table and dashboard stats in the frontend.

A real production system would use a database; a JSON file is used here
deliberately to keep the project's external dependencies minimal and the
data easy to inspect directly (open the file, read it) for a student /
portfolio project, while still being genuine persistence across server
restarts (unlike an in-memory list).
"""

import os
import json
import threading
from datetime import datetime

from backend import config
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()  # guards concurrent read-modify-write of the JSON file


def _ensure_file():
    os.makedirs(os.path.dirname(config.HISTORY_PATH), exist_ok=True)
    if not os.path.exists(config.HISTORY_PATH):
        with open(config.HISTORY_PATH, "w") as f:
            json.dump([], f)


def add_record(transaction: dict, prediction: dict) -> dict:
    """Append a new prediction record to history, trimming to MAX_HISTORY_RECORDS."""
    _ensure_file()
    record = {
        "timestamp": datetime.now().isoformat(),
        "amount": transaction.get("amount"),
        "time": transaction.get("time"),
        "is_fraud": prediction["is_fraud"],
        "risk_score": prediction["risk_score"],
        "risk_label": prediction["risk_label"],
        "model_used": prediction["model_used"],
    }

    with _lock:
        try:
            with open(config.HISTORY_PATH, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            history = []

        history.insert(0, record)  # newest first
        history = history[: config.MAX_HISTORY_RECORDS]

        with open(config.HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)

    return record


def get_history(limit: int = 50) -> list:
    """Return the most recent `limit` prediction records, newest first."""
    _ensure_file()
    with _lock:
        try:
            with open(config.HISTORY_PATH, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            history = []
    return history[:limit]


def get_stats() -> dict:
    """Aggregate stats over stored history, used by the dashboard."""
    history = get_history(limit=config.MAX_HISTORY_RECORDS)
    total = len(history)
    fraud_count = sum(1 for r in history if r["is_fraud"])

    return {
        "total_predictions": total,
        "fraud_detected": fraud_count,
        "legit_count": total - fraud_count,
        "fraud_rate_pct": round((fraud_count / total) * 100, 2) if total else 0.0,
    }
