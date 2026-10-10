import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "FED-MIND Multimodal"
    assert data["architecture"] == "Centralized Multimodal Baseline"
    assert data["modalities"] == ["text", "audio", "visual"]
    assert data["status"] == "ready"
    assert data["device"] in ["cuda", "cpu"]
    assert data["total_parameters"] == 7832001
