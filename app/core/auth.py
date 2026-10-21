"""Verified user identity for private resume resources."""

import asyncio
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth

firebase_bearer = HTTPBearer(auto_error=False, scheme_name="FirebaseIDToken")


def _verify_id_token(token: str) -> dict[str, Any]:
    return auth.verify_id_token(token, check_revoked=True)


async def get_authenticated_uid(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(firebase_bearer)],
) -> str:
    """Verify a Firebase ID token independently of the shared gateway token."""
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="A valid Firebase ID token is required",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise unauthorized
    try:
        claims = await asyncio.to_thread(_verify_id_token, credentials.credentials)
    except auth.CertificateFetchError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User authentication is temporarily unavailable",
        ) from exc
    except (
        auth.InvalidIdTokenError,
        auth.UserNotFoundError,
        auth.UserDisabledError,
        ValueError,
    ) as exc:
        raise unauthorized from exc
    uid = claims.get("uid")
    if not isinstance(uid, str) or not uid.strip() or uid != uid.strip() or "/" in uid:
        raise unauthorized
    return uid


AuthenticatedUid = Annotated[str, Depends(get_authenticated_uid)]
