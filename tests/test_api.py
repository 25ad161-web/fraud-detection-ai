"""
tests/test_api.py
--------------------
Automated tests for the Flask REST API.

RUN:
    pytest tests/test_api.py -v

These use Flask's test client (no need to have the server actually running
on a port) so they're fast and CI-friendly. They assume the model has
already been trained (python notebooks/01_eda_and_training.py has been run
at least once) since /api/predict depends on the saved model artifacts.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert "model_ready" in data


def test_predict_valid_transaction(client):
    res = client.post("/api/predict", json={"amount": 250.0, "time": 40000})
    assert res.status_code == 200
    data = res.get_json()
    assert "is_fraud" in data
    assert "risk_score" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_label"] in {"Low", "Medium", "High"}
    assert "model_used" in data


def test_predict_missing_amount(client):
    res = client.post("/api/predict", json={"time": 1000})
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_predict_negative_amount(client):
    res = client.post("/api/predict", json={"amount": -50})
    assert res.status_code == 400


def test_predict_non_numeric_amount(client):
    res = client.post("/api/predict", json={"amount": "not-a-number"})
    assert res.status_code == 400


def test_predict_empty_body(client):
    res = client.post("/api/predict", json={})
    assert res.status_code == 400


def test_predict_batch_valid(client):
    payload = {"transactions": [{"amount": 10}, {"amount": 9000}, {"amount": 75.5}]}
    res = client.post("/api/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["results"]) == 3
    for r in data["results"]:
        assert "is_fraud" in r or "error" in r


def test_predict_batch_empty_list(client):
    res = client.post("/api/predict/batch", json={"transactions": []})
    assert res.status_code == 400


def test_predict_batch_missing_field(client):
    res = client.post("/api/predict/batch", json={"not_transactions": []})
    assert res.status_code == 400


def test_explain_returns_valid_shape(client):
    res = client.post("/api/explain", json={"amount": 500})
    assert res.status_code == 200
    data = res.get_json()
    assert "available" in data
    assert "top_features" in data
    assert isinstance(data["top_features"], list)


def test_stats_endpoint(client):
    client.post("/api/predict", json={"amount": 100})  # ensure at least 1 record exists
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.get_json()
    assert "total_predictions" in data
    assert "fraud_detected" in data
    assert "fraud_rate_pct" in data


def test_history_endpoint(client):
    client.post("/api/predict", json={"amount": 60})
    res = client.get("/api/history?limit=5")
    assert res.status_code == 200
    data = res.get_json()
    assert "history" in data
    assert len(data["history"]) <= 5


def test_history_limit_is_clamped(client):
    res = client.get("/api/history?limit=99999")
    assert res.status_code == 200  # should clamp internally, not error


def test_model_info_endpoint(client):
    res = client.get("/api/model-info")
    assert res.status_code == 200
    data = res.get_json()
    assert "deployed_model" in data
    assert "model_name" in data["deployed_model"]


def test_charts_endpoint(client):
    res = client.get("/api/charts")
    assert res.status_code == 200
    data = res.get_json()
    assert "charts" in data
    assert isinstance(data["charts"], list)


def test_404_on_unknown_route(client):
    res = client.get("/api/does-not-exist")
    assert res.status_code == 404


def test_index_page_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Fraud Detection Console" in res.data
