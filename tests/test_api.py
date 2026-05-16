# tests/test_api.py
# Integration tests for the FastAPI serving layer
# These run against the live API — start it first:
#   uvicorn drift.serving.api:app --port 8000

import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"

SAMPLE_USER = {
    "recency_days": 12.0,
    "total_sessions": 8,
    "sessions_per_week": 1.5,
    "total_revenue": 120.0,
    "avg_session_revenue": 15.0,
    "avg_events_per_session": 6.0,
    "avg_duration_min": 8.0,
    "total_views": 60,
    "total_cart_adds": 8,
    "total_purchases": 2,
    "cart_abandonment_rate": 0.65,
    "purchase_conversion_rate": 0.03,
    "weekend_activity_ratio": 0.4,
    "night_owl_score": 0.15,
    "payday_activity_ratio": 0.2,
    "activity_trend": -0.05,
    "category_diversity": 1.2,
    "brand_loyalty_score": 0.4,
    "avg_price_point": 75.0,
    "price_sensitivity": 30.0,
}

HIGH_RISK_USER = {**SAMPLE_USER, "recency_days": 28.0, "total_purchases": 0,
                  "cart_abandonment_rate": 0.95, "activity_trend": -0.8}


def api_available():
    try:
        return requests.get(f"{BASE_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False


# Skip all API tests if the server isn't running
pytestmark = pytest.mark.skipif(
    not api_available(),
    reason="Drift API not running — start with: uvicorn drift.serving.api:app --port 8000"
)


# ------------------------------------------------------------------ #
# Health check
# ------------------------------------------------------------------ #
def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["churn_model_loaded"] is True


# ------------------------------------------------------------------ #
# Model info
# ------------------------------------------------------------------ #
def test_model_info():
    r = requests.get(f"{BASE_URL}/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "model_type" in data
    assert "n_features" in data
    assert data["n_features"] > 0


# ------------------------------------------------------------------ #
# Single prediction
# ------------------------------------------------------------------ #
def test_predict_returns_200():
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE_USER)
    assert r.status_code == 200


def test_predict_response_schema():
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE_USER)
    data = r.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "risk_level" in data
    assert "archetype" in data
    assert "top_risk_factors" in data
    assert "recommendation" in data


def test_predict_probability_range():
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE_USER)
    prob = r.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_risk_level_valid():
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE_USER)
    risk = r.json()["risk_level"]
    assert risk in ["low", "medium", "high", "critical"]


def test_predict_high_risk_user():
    """A user with high recency, no purchases, and high abandonment should be critical."""
    r = requests.post(f"{BASE_URL}/predict", json=HIGH_RISK_USER)
    assert r.status_code == 200
    data = r.json()
    assert data["churn_probability"] > 0.5
    assert data["churn_prediction"] is True


def test_predict_archetype_non_empty():
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE_USER)
    archetype = r.json()["archetype"]
    assert isinstance(archetype, str)
    assert len(archetype) > 0


def test_predict_risk_factors_list():
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE_USER)
    factors = r.json()["top_risk_factors"]
    assert isinstance(factors, list)
    assert len(factors) >= 1
    for f in factors:
        assert "factor" in f
        assert "detail" in f
        assert "impact" in f


# ------------------------------------------------------------------ #
# Batch prediction
# ------------------------------------------------------------------ #
def test_batch_predict():
    payload = {"users": [SAMPLE_USER, HIGH_RISK_USER, SAMPLE_USER]}
    r = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert "summary" in data
    assert len(data["predictions"]) == 3


def test_batch_summary_fields():
    payload = {"users": [SAMPLE_USER, HIGH_RISK_USER]}
    r = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    summary = r.json()["summary"]
    assert "total_users" in summary
    assert "predicted_churners" in summary
    assert "avg_churn_probability" in summary
    assert "high_risk_users" in summary
    assert "archetype_distribution" in summary
    assert summary["total_users"] == 2


# ------------------------------------------------------------------ #
# Input validation
# ------------------------------------------------------------------ #
def test_predict_missing_field_returns_422():
    bad_payload = {"recency_days": 5.0}  # missing required fields
    r = requests.post(f"{BASE_URL}/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_invalid_type_returns_422():
    bad_payload = {**SAMPLE_USER, "total_sessions": "not_a_number"}
    r = requests.post(f"{BASE_URL}/predict", json=bad_payload)
    assert r.status_code == 422
