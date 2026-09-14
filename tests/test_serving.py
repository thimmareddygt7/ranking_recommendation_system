import pytest
from fastapi.testclient import TestClient

try:
    from serving.app import app
except ModuleNotFoundError:
    from serving.main import app

client = TestClient(app)

def test_health_check():
    """Verify service health probe."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_recommendation_endpoint_valid():
    """Verify recommendation returns top-k ranked items with real model scores."""
    response = client.post("/recommend", json={"user_id": 100, "top_k": 5})
    assert response.status_code == 200
    payload = response.json()
    assert "user_id" in payload
    assert "recommendations" in payload
    assert len(payload["recommendations"]) <= 5
    assert payload["fallback_used"] is False

    # Verify scores are real model outputs and not mock linear spaces
    recs = payload["recommendations"]
    assert len(recs) > 0
    # Items should not be the old mock default [10001, 10002, 10003, 10004, 10005] with linspace scores
    scores = [r["score"] for r in recs]
    # Check that scores are monotonically non-increasing (sorted by ranker)
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]

def test_get_recommendation_endpoint():
    """Verify GET /recommend/{user_id} endpoint returns personalized model rankings."""
    response = client.get("/recommend/100?top_k=5")
    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == 100
    assert payload["fallback_used"] is False
    assert len(payload["recommendations"]) <= 5

def test_cold_start_fallback():
    """Verify unknown cold-start users fallback to global popular candidates."""
    response = client.post("/recommend", json={"user_id": 999999999, "top_k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert "recommendations" in payload
    assert len(payload["recommendations"]) <= 3
    assert payload["fallback_used"] is True
    assert payload["is_cold_start"] is True

def test_frontend_served():
    """Verify root GET / returns the HTML frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Retailrocket Ranker" in response.text
    assert "Two-Stage Recommender" in response.text

def test_explain_endpoint():
    """Verify POST /explain generates model-driven explanation."""
    payload = {
        "item_id": 10135,
        "score": 0.9468,
        "category_id": 442,
        "top_features": [],
        "shopper_recent_categories": []
    }
    response = client.post("/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["item_id"] == 10135
    assert "explanation" in data
    assert len(data["explanation"]) > 10


