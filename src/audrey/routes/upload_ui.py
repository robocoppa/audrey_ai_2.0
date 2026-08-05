"""Serve the upload UI at GET /upload.

Single static HTML shipped from `audrey/static/upload.html`. Kept out of
the FastAPI OpenAPI docs (`include_in_schema=False`) since it's a human
endpoint, not part of the API surface.

Users reach this from a link in an OWUI banner. **Open WebUI has no
sidebar-links feature** — phase 40 planned around one that does not exist, and
this docstring named it too. A banner is the supported way to put a link in
front of an OWUI user; see `docs/campaign-2/phase-40-uploads-in-chat.md` step 1.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_HTML_PATH = Path(__file__).resolve().parent.parent / "static" / "upload.html"
_HTML_CACHE: str | None = None


@router.api_route("/upload", methods=["GET", "HEAD"], response_class=HTMLResponse, include_in_schema=False)
async def upload_page() -> HTMLResponse:
    # Cache on first read. The file ships in the container image and never
    # changes at runtime, so re-reading per request is wasted I/O.
    global _HTML_CACHE
    if _HTML_CACHE is None:
        try:
            _HTML_CACHE = _HTML_PATH.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail="Upload UI is missing from the image.") from e
    return HTMLResponse(_HTML_CACHE)


__all__ = ["router"]
