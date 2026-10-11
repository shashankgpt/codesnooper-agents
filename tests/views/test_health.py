from fastapi.testclient import TestClient


def test_hello_endpoint(client: TestClient) -> None:
    """Verify /api/v1/hello returns the expected greeting and metadata."""
    response = client.get("/api/v1/hello")
    assert response.status_code == 200

    data = response.json()
    assert "Hello from CodeSnooper Agents!" in data["message"]
    assert "environment" in data
    assert "version" in data
    assert data["version"] == "0.1.0"


def test_health_endpoint(client: TestClient) -> None:
    """Verify /api/v1/health returns status, Firebase, and Gemini connectivity booleans."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "firebase_connected" in data
    assert isinstance(data["firebase_connected"], bool)
    assert "gemini_connected" in data
    assert isinstance(data["gemini_connected"], bool)
