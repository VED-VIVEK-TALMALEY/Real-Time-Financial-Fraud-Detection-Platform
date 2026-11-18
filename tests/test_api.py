from fastapi.testclient import TestClient
from src.api.main import app


def test_health_and_model_info():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/model-info").status_code == 200
