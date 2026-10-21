import pytest

from app.agents.vercel_blob_source import fetch_resume_blob
from app.config import Settings


@pytest.mark.parametrize(
    "pathname, uid",
    [
        ("career-profiles/victim/resume.pdf", "owner"),
        ("career-profiles/owner-other/resume.pdf", "owner"),
        ("career-profiles/owner/../victim/resume.pdf", "owner"),
        ("career-profiles/owner/resume.pdf", ""),
        ("career-profiles/owner/subpath/resume.pdf", "owner/subpath"),
    ],
)
async def test_blob_ownership_is_checked_before_credentials_or_download(
    pathname: str, uid: str
) -> None:
    with pytest.raises(ValueError, match="does not belong to this user"):
        await fetch_resume_blob(pathname, uid, Settings())
