from fastapi.testclient import TestClient
import pytest

from infra.http.main import app

client = TestClient(app)

@pytest.mark.slow_integration_test
def test_read_main():
    response = client.post("/", json={"text": "Hello World", "aspect_labels": ["world", "peace"]})
    assert response.status_code == 200
