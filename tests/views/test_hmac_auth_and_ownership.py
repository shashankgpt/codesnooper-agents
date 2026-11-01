"""
Tests for Phase 2 (HMAC-SHA256 request signing) and Phase 1 uid-ownership
enforcement on resume job endpoints.

Covers:
  - HMAC validation (valid sig passes, wrong sig / expired ts / missing headers fail)
  - Backward-compat: legacy bearer token still works when M2M_SIGNING_SECRET absent
  - uid ownership: GET/DELETE /resume/jobs/{job_id} return 404 when uid mismatches
  - Multi-secret rotation: both old and new secrets accepted simultaneously
"""

import hashlib
import hmac
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app


# ─── Helpers ────────────────────────────────────────────────────────────────


def _sign(secret: str, timestamp: int, method: str, path: str, body: bytes = b"") -> str:
    signed_payload = f"{timestamp}.{method.upper()}.{path}.".encode() + body
    return hmac.new(secret.encode(), signed_payload, hashlib.sha256).hexdigest()


def _hmac_headers(
    secret: str,
    method: str,
    path: str,
    body: bytes = b"",
    user_id: str = "uid-alice",
    ts_offset: int = 0,
) -> dict[str, str]:
    ts = int(time.time()) + ts_offset
    return {
        "X-Timestamp": str(ts),
        "X-Signature": _sign(secret, ts, method, path, body),
        "X-User-Id": user_id,
        "Content-Type": "application/json",
    }


# ─── Fixtures ────────────────────────────────────────────────────────────────


SIGNING_SECRET = "super-secret-hmac-key-at-least-32-chars-long"
LEGACY_TOKEN = "legacy-bearer-token-for-tests"


@pytest.fixture()
def hmac_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """App configured for HMAC-only auth (M2M_SIGNING_SECRET set)."""
    settings = Settings(m2m_signing_secret=SIGNING_SECRET)
    monkeypatch.setattr("app.core.middleware.get_settings", lambda: settings)
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def legacy_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """App configured for legacy bearer token only."""
    settings = Settings(api_gateway_token=LEGACY_TOKEN)
    monkeypatch.setattr("app.core.middleware.get_settings", lambda: settings)
    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def rotation_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """App with both old and new HMAC secrets (zero-downtime rotation)."""
    old_secret = "old-hmac-signing-secret-at-least-32-chars"
    new_secret = SIGNING_SECRET
    settings = Settings(m2m_signing_secrets=f"{old_secret},{new_secret}")
    monkeypatch.setattr("app.core.middleware.get_settings", lambda: settings)
    app = create_app()
    with TestClient(app) as client:
        yield client


# ─── HMAC Auth Tests ─────────────────────────────────────────────────────────


class TestHmacAuth:
    """HMAC-SHA256 request signing validation."""

    PATH = "/api/v1/resume/parse"
    BODY = b'{"uid":"uid-alice","url":"https://example.com/resume.pdf","model":"auto"}'

    def test_valid_hmac_signature_passes_middleware(self, hmac_client: TestClient) -> None:
        """A correctly signed request reaches the route (422 = schema validation, not 401)."""
        headers = _hmac_headers(SIGNING_SECRET, "POST", self.PATH, self.BODY)
        resp = hmac_client.post(self.PATH, content=self.BODY, headers=headers)
        # 422 means auth passed; FastAPI rejected empty/invalid body schema
        assert resp.status_code != 401, f"Expected to pass auth, got 401: {resp.json()}"

    def test_wrong_signature_returns_401(self, hmac_client: TestClient) -> None:
        ts = int(time.time())
        headers = {
            "X-Timestamp": str(ts),
            "X-Signature": "deadbeef" * 8,  # garbage sig
            "X-User-Id": "uid-alice",
            "Content-Type": "application/json",
        }
        resp = hmac_client.post(self.PATH, content=self.BODY, headers=headers)
        assert resp.status_code == 401
        assert "signature" in resp.json()["detail"].lower()

    def test_missing_signature_header_returns_401(self, hmac_client: TestClient) -> None:
        resp = hmac_client.post(
            self.PATH,
            content=self.BODY,
            headers={"X-Timestamp": str(int(time.time())), "Content-Type": "application/json"},
        )
        assert resp.status_code == 401

    def test_missing_timestamp_header_returns_401(self, hmac_client: TestClient) -> None:
        ts = int(time.time())
        sig = _sign(SIGNING_SECRET, ts, "POST", self.PATH, self.BODY)
        resp = hmac_client.post(
            self.PATH,
            content=self.BODY,
            headers={"X-Signature": sig, "Content-Type": "application/json"},
        )
        assert resp.status_code == 401

    def test_expired_timestamp_returns_401(self, hmac_client: TestClient) -> None:
        """Timestamps older than 5 minutes must be rejected (replay protection)."""
        old_ts = int(time.time()) - 310  # 5 min + 10 s ago
        headers = {
            "X-Timestamp": str(old_ts),
            "X-Signature": _sign(SIGNING_SECRET, old_ts, "POST", self.PATH, self.BODY),
            "X-User-Id": "uid-alice",
            "Content-Type": "application/json",
        }
        resp = hmac_client.post(self.PATH, content=self.BODY, headers=headers)
        assert resp.status_code == 401
        assert "expired" in resp.json()["detail"].lower()

    def test_future_timestamp_within_window_passes(self, hmac_client: TestClient) -> None:
        """Clocks slightly ahead (up to 5 min) are still accepted."""
        headers = _hmac_headers(SIGNING_SECRET, "POST", self.PATH, self.BODY, ts_offset=60)
        resp = hmac_client.post(self.PATH, content=self.BODY, headers=headers)
        assert resp.status_code != 401

    def test_tampered_body_returns_401(self, hmac_client: TestClient) -> None:
        """Signing original body, then sending a different body must fail."""
        original_body = self.BODY
        tampered_body = b'{"uid":"uid-alice","url":"https://evil.com/steal","model":"auto"}'
        headers = _hmac_headers(SIGNING_SECRET, "POST", self.PATH, original_body)
        resp = hmac_client.post(self.PATH, content=tampered_body, headers=headers)
        assert resp.status_code == 401

    def test_invalid_timestamp_format_returns_401(self, hmac_client: TestClient) -> None:
        resp = hmac_client.post(
            self.PATH,
            content=self.BODY,
            headers={
                "X-Timestamp": "not-a-number",
                "X-Signature": "abc123",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 401
        assert "timestamp" in resp.json()["detail"].lower()


# ─── Backward-compat: legacy bearer token ────────────────────────────────────


class TestLegacyBearerTokenFallback:
    """Legacy API_GATEWAY_TOKEN still accepted when M2M_SIGNING_SECRET is absent."""

    def test_valid_legacy_token_passes(self, legacy_client: TestClient) -> None:
        resp = legacy_client.post(
            "/api/v1/resume/parse",
            headers={"Authorization": f"Bearer {LEGACY_TOKEN}"},
            json={},
        )
        assert resp.status_code == 422  # schema error, not auth error

    def test_missing_token_fails_without_hmac_secret(self, legacy_client: TestClient) -> None:
        resp = legacy_client.post("/api/v1/resume/parse", json={})
        assert resp.status_code == 401


# ─── Multi-secret rotation ────────────────────────────────────────────────────


class TestMultiSecretRotation:
    """Both old and new HMAC secrets must be accepted during zero-downtime rotation."""

    PATH = "/api/v1/resume/parse"
    BODY = b'{"uid":"uid-alice","url":"https://example.com/cv.pdf","model":"auto"}'

    def test_old_secret_still_accepted_during_rotation(self, rotation_client: TestClient) -> None:
        old_secret = "old-hmac-signing-secret-at-least-32-chars"
        headers = _hmac_headers(old_secret, "POST", self.PATH, self.BODY)
        resp = rotation_client.post(self.PATH, content=self.BODY, headers=headers)
        assert resp.status_code != 401

    def test_new_secret_accepted_during_rotation(self, rotation_client: TestClient) -> None:
        headers = _hmac_headers(SIGNING_SECRET, "POST", self.PATH, self.BODY)
        resp = rotation_client.post(self.PATH, content=self.BODY, headers=headers)
        assert resp.status_code != 401

    def test_unknown_secret_rejected_during_rotation(self, rotation_client: TestClient) -> None:
        headers = _hmac_headers("completely-wrong-secret-at-least-32-chars!!", "POST", self.PATH, self.BODY)
        resp = rotation_client.post(self.PATH, content=self.BODY, headers=headers)
        assert resp.status_code == 401


# ─── uid ownership on GET / DELETE ───────────────────────────────────────────


class TestUidOwnership:
    """
    GET /resume/jobs/{job_id} and DELETE /resume/jobs/{job_id} must enforce that
    the requesting user (X-User-Id) matches the uid stored in the Firestore job doc.
    """

    GET_PATH = "/api/v1/resume/jobs/job-abc"
    DELETE_PATH = "/api/v1/resume/jobs/job-abc"

    def _auth_headers(self, user_id: str) -> dict[str, str]:
        return _hmac_headers(SIGNING_SECRET, "GET", self.GET_PATH, user_id=user_id)

    def _delete_headers(self, user_id: str) -> dict[str, str]:
        return _hmac_headers(SIGNING_SECRET, "DELETE", self.DELETE_PATH, user_id=user_id)

    @pytest.fixture()
    def hmac_app(self, monkeypatch: pytest.MonkeyPatch) -> TestClient:
        settings = Settings(m2m_signing_secret=SIGNING_SECRET)
        monkeypatch.setattr("app.core.middleware.get_settings", lambda: settings)
        app = create_app()
        with TestClient(app) as client:
            yield client

    def test_get_job_returns_404_when_uid_mismatches(
        self, hmac_app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job owned by alice must not be visible to bob."""
        job_doc = {"uid": "uid-alice", "status": "queued", "result": None}
        mock_get = AsyncMock(return_value=job_doc)
        monkeypatch.setattr("app.views.resume.firebase_service.async_get", mock_get)

        headers = _hmac_headers(SIGNING_SECRET, "GET", self.GET_PATH, user_id="uid-bob")
        resp = hmac_app.get(self.GET_PATH, headers=headers)
        assert resp.status_code == 404

    def test_get_job_succeeds_when_uid_matches(
        self, hmac_app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job_doc = {"uid": "uid-alice", "status": "queued", "result": None}
        mock_get = AsyncMock(return_value=job_doc)
        monkeypatch.setattr("app.views.resume.firebase_service.async_get", mock_get)

        headers = _hmac_headers(SIGNING_SECRET, "GET", self.GET_PATH, user_id="uid-alice")
        resp = hmac_app.get(self.GET_PATH, headers=headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

    def test_get_job_returns_404_when_doc_missing(
        self, hmac_app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_get = AsyncMock(return_value=None)
        monkeypatch.setattr("app.views.resume.firebase_service.async_get", mock_get)

        headers = _hmac_headers(SIGNING_SECRET, "GET", self.GET_PATH, user_id="uid-alice")
        resp = hmac_app.get(self.GET_PATH, headers=headers)
        assert resp.status_code == 404

    def test_delete_job_returns_404_when_uid_mismatches(
        self, hmac_app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job_doc = {"uid": "uid-alice", "status": "completed", "result": None}
        mock_get = AsyncMock(return_value=job_doc)
        mock_delete = AsyncMock()
        monkeypatch.setattr("app.views.resume.firebase_service.async_get", mock_get)
        monkeypatch.setattr("app.views.resume.firebase_service.async_delete", mock_delete)

        headers = _hmac_headers(SIGNING_SECRET, "DELETE", self.DELETE_PATH, user_id="uid-bob")
        resp = hmac_app.delete(self.DELETE_PATH, headers=headers)
        assert resp.status_code == 404
        mock_delete.assert_not_called()

    def test_delete_job_succeeds_and_removes_doc_when_uid_matches(
        self, hmac_app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job_doc = {"uid": "uid-alice", "status": "completed", "result": None}
        mock_get = AsyncMock(return_value=job_doc)
        mock_delete = AsyncMock()
        monkeypatch.setattr("app.views.resume.firebase_service.async_get", mock_get)
        monkeypatch.setattr("app.views.resume.firebase_service.async_delete", mock_delete)

        headers = _hmac_headers(SIGNING_SECRET, "DELETE", self.DELETE_PATH, user_id="uid-alice")
        resp = hmac_app.delete(self.DELETE_PATH, headers=headers)
        assert resp.status_code == 204
        mock_delete.assert_awaited_once_with("resume_parse_jobs/job-abc")

    def test_get_job_missing_user_id_header_returns_400(
        self, hmac_app: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """X-User-Id header is required on job endpoints; absence must return 400."""
        job_doc = {"uid": "uid-alice", "status": "queued", "result": None}
        mock_get = AsyncMock(return_value=job_doc)
        monkeypatch.setattr("app.views.resume.firebase_service.async_get", mock_get)

        ts = int(time.time())
        sig = _sign(SIGNING_SECRET, ts, "GET", self.GET_PATH)
        # Omit X-User-Id deliberately
        headers = {"X-Timestamp": str(ts), "X-Signature": sig}
        resp = hmac_app.get(self.GET_PATH, headers=headers)
        assert resp.status_code == 400


# ─── Docs disabled in production ─────────────────────────────────────────────


class TestDocsDisabledInProduction:
    """Swagger / ReDoc must not be exposed when app_debug=False."""

    def _make_settings(self, debug: bool) -> Settings:
        return Settings(api_gateway_token=LEGACY_TOKEN, app_debug=debug)

    def test_docs_return_404_when_debug_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._make_settings(False)
        # Patch in both places: app factory AND middleware (both call get_settings())
        monkeypatch.setattr("app.main.get_settings", lambda: settings)
        monkeypatch.setattr("app.core.middleware.get_settings", lambda: settings)
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/docs").status_code == 404
            assert client.get("/redoc").status_code == 404
            assert client.get("/openapi.json").status_code == 404

    def test_docs_return_200_when_debug_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = self._make_settings(True)
        monkeypatch.setattr("app.main.get_settings", lambda: settings)
        monkeypatch.setattr("app.core.middleware.get_settings", lambda: settings)
        app = create_app()
        with TestClient(app) as client:
            assert client.get("/docs").status_code == 200
            assert client.get("/openapi.json").status_code == 200

