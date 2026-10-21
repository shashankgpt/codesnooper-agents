"""Read private resume uploads directly from Vercel Blob storage."""

from pathlib import PurePosixPath

from app.config import Settings


async def fetch_resume_blob(pathname: str, uid: str, settings: Settings) -> tuple[bytes, str]:
    """Download an uploaded resume from the configured private Blob store."""
    expected_prefix = f"career-profiles/{uid}/"
    if not pathname.startswith(expected_prefix) or ".." in PurePosixPath(pathname).parts:
        raise ValueError("The uploaded resume Blob does not belong to this user")

    extension = PurePosixPath(pathname).suffix.lower().lstrip(".")
    if extension not in {"pdf", "doc", "docx"}:
        raise ValueError("Uploaded resume must be a PDF, DOC, or DOCX file")

    import os

    token = (
        settings.career_profile_blob_read_write_token
        or os.environ.get("CAREER_PROFILE_BLOB_READ_WRITE_TOKEN")
        or os.environ.get("BLOB_READ_WRITE_TOKEN")
        or os.environ.get("VERCEL_BLOB_READ_WRITE_TOKEN")
        or ""
    ).strip()
    store_id = (
        settings.career_profile_blob_store_id
        or os.environ.get("CAREER_PROFILE_BLOB_STORE_ID")
        or os.environ.get("BLOB_STORE_ID")
        or ""
    ).strip()

    if not store_id and token.startswith("vercel_blob_rw_"):
        try:
            from vercel.blob._internal.credentials import store_from_read_write_token

            store_id = store_from_read_write_token(token)
        except Exception:
            pass

    if not store_id or not token:
        raise RuntimeError(
            "Vercel Blob access is not configured on the resume worker; set the Blob store ID "
            "and read-write token in the worker environment"
        )

    from vercel import blob
    from vercel.api import session

    async def credentials_factory() -> blob.BlobCredentials:
        return blob.BlobCredentials(
            token=token,
            store_id=store_id,
            kind=blob.CredentialKind.READ_WRITE,
        )

    blob_options = blob.BlobServiceOptions(credentials_factory=credentials_factory)
    async with session(service_options=[blob_options]):
        result = await blob.get_blob(pathname, access="private")

    return result.body, extension
