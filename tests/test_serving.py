import sys
import os

# Ensure the root repository directory is in the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import serving.main as serving_module
from serving.main import app, load_artifacts

# Trigger lifecycle startup event manually for testing
load_artifacts()
client = TestClient(app)

def test_warm_user_recommendation():
    # Dynamically select an existing warm user from the loaded store
    warm_user_id = int(list(serving_module.candidate_store.keys())[0])

    response = client.get(f"/recommend/{warm_user_id}?k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == warm_user_id
    assert data["is_cold_start"] is False
    assert len(data["recommendations"]) == 5
    assert data["latency_ms"] < 100.0  # Latency guardrail check
    print(f"Warm user {warm_user_id} passed: Latency = {data['latency_ms']} ms")

def test_cold_user_fallback():
    # Synthetic user guaranteed not to exist
    cold_user_id = -99999
    response = client.get(f"/recommend/{cold_user_id}?k=5")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == cold_user_id
    assert data["is_cold_start"] is True
    assert len(data["recommendations"]) == 5
    print(f"Cold user {cold_user_id} fallback passed: Latency = {data['latency_ms']} ms")

if __name__ == "__main__":
    test_warm_user_recommendation()
    test_cold_user_fallback()
    print("Serving layer validation passed successfully!")
