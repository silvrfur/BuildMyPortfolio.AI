# tests/conversation/test_routes.py
from fastapi.testclient import TestClient

def test_chat_endpoint(client: TestClient):
    response = client.post(
        "/api/chat",
        json={
            "user_id": "u1",
            "message": "Should I sell now?"
        }
    )

    assert response.status_code == 200
    assert "response" in response.json()

def test_chat_validation_error(client: TestClient):
    response = client.post(
        "/api/chat",
        json={"user_id": "u1"}
    )

    assert response.status_code == 422
