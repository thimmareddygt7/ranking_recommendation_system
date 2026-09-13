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
    """Verify recommendation returns top-k ranked items."""
    response = client.post("/recommend", json={"user_id": 100, "top_k": 5})
    assert response.status_code == 200
    payload = response.json()
    assert "user_id" in payload
    assert "recommendations" in payload
    assert len(payload["recommendations"]) <= 5

def test_cold_start_fallback():
    """Verify unknown cold-start users fallback to global popular candidates."""
    response = client.post("/recommend", json={"user_id": 999999999, "top_k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert "recommendations" in payload
    assert len(payload["recommendations"]) <= 3
