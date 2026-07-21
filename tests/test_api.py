from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qa_service.api import create_app
from qa_service.training import train_and_save


@pytest.fixture()
def client(tmp_path: Path):
    artifact_path = tmp_path / "qa_model.pt"
    project_root = Path(__file__).resolve().parents[1]
    train_and_save(project_root / "100_Unique_QA_Dataset.csv", artifact_path, epochs=100)
    with TestClient(create_app(artifact_path)) as client:
        yield client


def test_health_and_prediction(client: TestClient) -> None:
    health = client.get("/health")
    prediction = client.post("/predict", json={"question": "What is the capital of France?"})

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert prediction.status_code == 200
    assert prediction.json()["answer"] == "Paris"
    assert prediction.json()["supported"] is True


def test_rejects_invalid_request(client: TestClient) -> None:
    response = client.post("/predict", json={"question": ""})
    assert response.status_code == 422


def test_rejects_unfamiliar_question(client: TestClient) -> None:
    response = client.post("/predict", json={"question": "flibbertigibbet quasar zephyr"})
    assert response.status_code == 200
    assert response.json()["answer"] is None
    assert response.json()["supported"] is False
