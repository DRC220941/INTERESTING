from fastapi.testclient import TestClient
from src.main import app, working_mem
import pytest

client = TestClient(app)

# ===== FIXTURE POUR RÉINITIALISER LA MÉMOIRE =====
@pytest.fixture(autouse=True)
def reset_memory():
    """Réinitialise la mémoire avant chaque test"""
    working_mem.reset()
    yield

# ===== TESTS =====
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_empty():
    response = client.post("/predict", json={"values": []})
    assert response.status_code == 200
    assert "Données insuffisantes" in response.json()["warnings"]

def test_predict_with_data():
    response = client.post("/predict", json={"values": [1.5, 2.0, 1.8, 3.0, 2.5]})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3
    assert response.json()["top_features"]["count"] == 5

def test_reset():
    working_mem.add_multiple([1.5, 2.0])
    assert len(working_mem.get_all()) == 2
    response = client.post("/reset")
    assert response.status_code == 200
    assert len(working_mem.get_all()) == 0

def test_history():
    working_mem.add_multiple([1.0, 2.0, 3.0])
    response = client.get("/history")
    assert response.status_code == 200
    assert response.json()["count"] == 3
