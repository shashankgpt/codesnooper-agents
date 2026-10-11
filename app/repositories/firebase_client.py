import asyncio
import json
import logging
import os
from functools import partial
from typing import Any

import firebase_admin
from firebase_admin import credentials, db, firestore

from app.config import get_settings

logger = logging.getLogger(__name__)


class FirebaseService:
    """Firebase Firestore and RTDB service with non-blocking async execution wrappers."""

    def __init__(self) -> None:
        self._initialized = False
        self._firestore_db: Any = None

    def initialize(self) -> None:
        """Initialize Firebase Admin app.

        Supports credentials from:
        - FIREBASE_PRIVATE_KEY, FIREBASE_CLIENT_EMAIL, and FIREBASE_PROJECT_ID
        - GOOGLE_APPLICATION_CREDENTIALS_JSON
        - GOOGLE_APPLICATION_CREDENTIALS file path
        - Default Google Cloud environment credentials
        """
        if self._initialized:
            return

        settings = get_settings()

        try:
            # Check if Firebase default app is already initialized
            try:
                firebase_admin.get_app()
                self._initialized = True
                self._init_clients()
                return
            except ValueError:
                pass

            cred: credentials.Certificate | None = None
            if settings.firebase_private_key and settings.firebase_client_email:
                raw_key = settings.firebase_private_key
                # Strip wrapping quotes if present
                if (raw_key.startswith('"') and raw_key.endswith('"')) or (
                    raw_key.startswith("'") and raw_key.endswith("'")
                ):
                    raw_key = raw_key[1:-1]
                formatted_key = raw_key.replace("\\n", "\n")
                cred_dict: dict[str, Any] = {
                    "type": "service_account",
                    "project_id": settings.firebase_project_id or "codesnooper-staging",
                    "private_key": formatted_key,
                    "client_email": settings.firebase_client_email,
                    "token_uri": "https://oauth2.googleapis.com/token",
                }
                cred = credentials.Certificate(cred_dict)
            elif settings.google_application_credentials_json:
                cred_dict = json.loads(settings.google_application_credentials_json)
                cred = credentials.Certificate(cred_dict)
            elif settings.google_application_credentials and os.path.exists(
                settings.google_application_credentials
            ):
                cred = credentials.Certificate(settings.google_application_credentials)

            options: dict[str, Any] = {}
            if settings.firebase_database_url:
                options["databaseURL"] = settings.firebase_database_url
            if settings.firebase_project_id:
                options["projectId"] = settings.firebase_project_id

            if cred:
                firebase_admin.initialize_app(cred, options=options)
            else:
                firebase_admin.initialize_app(options=options)

            self._initialized = True
            self._init_clients()
            logger.info("Firebase Admin app successfully initialized.")
        except Exception as exc:
            logger.error("Failed to initialize Firebase Admin app: %s", exc)

    def _init_clients(self) -> None:
        try:
            self._firestore_db = firestore.client()
        except Exception as exc:
            logger.warning("Firestore client initialization skipped or failed: %s", exc)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def firestore(self) -> Any:
        return self._firestore_db

    def check_connection(self) -> bool:
        """Synchronous check for Firebase connectivity (Firestore / Realtime Database)."""
        if not self._initialized:
            return False

        # First check Firestore connectivity if client is available
        if self._firestore_db is not None:
            try:
                _ = next(self._firestore_db.collections(), None)
                return True
            except Exception as exc:
                logger.warning("Firestore connection check failed: %s", exc)

        # Fallback to Realtime Database if configured
        settings = get_settings()
        if settings.firebase_database_url:
            try:
                ref = db.reference()
                _ = ref.get(shallow=True)
                return True
            except Exception as exc:
                logger.warning("Firebase RTDB connection check failed: %s", exc)

        return False

    async def async_get(self, path: str) -> Any:
        """Fetch data from Firebase Realtime Database asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._sync_get, path))

    async def async_set(self, path: str, data: Any) -> None:
        """Write data to Firebase Realtime Database asynchronously."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(self._sync_set, path, data))

    async def async_update(self, path: str, data: dict[str, Any]) -> None:
        """Update data in Firebase Realtime Database asynchronously."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(self._sync_update, path, data))

    async def async_delete(self, path: str) -> None:
        """Delete data from Firebase Realtime Database asynchronously."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(self._sync_delete, path))

    def _sync_get(self, path: str) -> Any:
        data: Any = db.reference(path).get()
        return data

    def _sync_set(self, path: str, data: Any) -> None:
        db.reference(path).set(data)

    def _sync_update(self, path: str, data: dict[str, Any]) -> None:
        db.reference(path).update(data)

    def _sync_delete(self, path: str) -> None:
        db.reference(path).delete()


firebase_service = FirebaseService()
