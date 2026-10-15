import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app


def test_health_endpoints_exempt_without_token(client: TestClient) -> None:
    """Verify health endpoints do not require token."""
    health_resp = client.get("/api/v1/health")
    assert health_resp.status_code == 200

    hello_resp = client.get("/api/v1/hello")
    assert hello_resp.status_code == 200


def test_docs_endpoints_exempt_without_token(client: TestClient) -> None:
    """Verify docs and openapi endpoints do not require token."""
    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200

    openapi_resp = client.get("/openapi.json")
    assert openapi_resp.status_code == 200


def test_cors_options_preflight_bypasses_token(client: TestClient) -> None:
    """Verify OPTIONS pre-flight requests do not require token."""
    resp = client.options(
        "/api/v1/resume/parse",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 200


def test_protected_endpoint_without_token_fails(client: TestClient) -> None:
    """Verify protected endpoints reject requests missing an auth token with 401."""
    resp = client.post("/api/v1/resume/parse", json={"url": "https://example.com/resume.pdf"})
    assert resp.status_code == 401
    data = resp.json()
    assert "Invalid or missing API gateway token" in data["detail"]
    assert resp.headers.get("www-authenticate") == "Bearer"


def test_protected_endpoint_with_invalid_token_fails(client: TestClient) -> None:
    """Verify requests with incorrect token are rejected with 401."""
    # Bearer header
    resp = client.post(
        "/api/v1/resume/parse",
        headers={"Authorization": "Bearer invalid_secret_token"},
        json={"url": "https://example.com/resume.pdf"},
    )
    assert resp.status_code == 401

    # X-API-Key header
    resp_key = client.post(
        "/api/v1/resume/parse",
        headers={"X-API-Key": "invalid_secret_token"},
        json={"url": "https://example.com/resume.pdf"},
    )
    assert resp_key.status_code == 401


def test_protected_endpoint_with_valid_token_in_bearer_passes(client: TestClient) -> None:
    """Verify request with valid Bearer token passes auth middleware."""
    settings = get_settings()
    token = settings.api_gateway_token
    assert token, "api_gateway_token should be configured in settings"

    # Send request with valid token but empty body to check route reached.
    # Expect 422 Unprocessable Entity rather than 401 Unauthorized.
    resp = client.post(
        "/api/v1/resume/parse",
        headers={"Authorization": f"Bearer {token}"},
        json={},
    )
    assert resp.status_code == 422  # Passed auth middleware and reached FastAPI validation


def test_protected_endpoint_with_valid_token_in_x_api_key_passes(client: TestClient) -> None:
    """Verify request with valid X-API-Key header passes auth middleware."""
    settings = get_settings()
    token = settings.api_gateway_token
    assert token, "api_gateway_token should be configured in settings"

    resp = client.post(
        "/api/v1/resume/parse",
        headers={"X-API-Key": token},
        json={},
    )
    assert resp.status_code == 422


def test_protected_endpoint_with_valid_token_in_x_gateway_token_passes(client: TestClient) -> None:
    """Verify request with valid X-Gateway-Token header passes auth middleware."""
    settings = get_settings()
    token = settings.api_gateway_token
    assert token, "api_gateway_token should be configured in settings"

    resp = client.post(
        "/api/v1/resume/parse",
        headers={"X-Gateway-Token": token},
        json={},
    )
    assert resp.status_code == 422


def test_missing_server_token_returns_500(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify 500 status if API gateway token is unconfigured on server."""
    from app.config import Settings

    empty_settings = Settings(api_gateway_token="")
    monkeypatch.setattr("app.core.middleware.get_settings", lambda: empty_settings)

    app = create_app()
    with TestClient(app) as test_client:
        resp = test_client.post("/api/v1/resume/parse", json={})
        assert resp.status_code == 500
        assert "not configured on server" in resp.json()["detail"]


def test_openapi_swagger_security_schemes(client: TestClient) -> None:
    """Verify OpenAPI schema registers securitySchemes and secures non-health endpoints."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()

    # Security schemes present for Swagger UI Authorize dialog
    schemes = schema.get("components", {}).get("securitySchemes", {})
    assert "HTTPBearer" in schemes
    assert "APIKeyHeader" in schemes

    # Health endpoint does not require security
    health_op = schema["paths"]["/api/v1/health"]["get"]
    assert "security" not in health_op or health_op["security"] is None

    # Protected endpoint requires security
    resume_op = schema["paths"]["/api/v1/resume/parse"]["post"]
    assert "security" in resume_op
    assert {"HTTPBearer": []} in resume_op["security"]
    assert {"APIKeyHeader": []} in resume_op["security"]


def test_swagger_ui_request_duration_configured(client: TestClient) -> None:
    """Verify Swagger UI (/docs) is configured with displayRequestDuration: true."""
    resp = client.get("/docs")
    assert resp.status_code == 200
    # FastAPI injects swagger_ui_parameters into swaggerOptions in the docs HTML
    assert '"displayRequestDuration": true' in resp.text


def test_timing_middleware_headers(client: TestClient) -> None:
    """Verify timing middleware attaches X-Process-Time and X-Response-Time headers."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert "X-Process-Time" in resp.headers
    assert "X-Response-Time" in resp.headers
    assert resp.headers["X-Process-Time"].endswith("s")
    assert resp.headers["X-Response-Time"].endswith("ms")


def test_openapi_resume_schema_has_latency_field(client: TestClient) -> None:
    """Verify ResumeParseResponse in OpenAPI schema documents latency_ms."""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    response_schema = schema["components"]["schemas"]["ResumeParseResponse"]
    assert "latency_ms" in response_schema["properties"]


