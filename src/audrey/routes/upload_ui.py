"""Serve the upload UI at GET /upload.

Single static HTML shipped from `audrey/static/upload.html`. Kept out of
the FastAPI OpenAPI docs (`include_in_schema=False`) since it's a human
endpoint, not part of the API surface.

Users land here from OWUI's sidebar link (see `docs/phase-13-deploy.md`).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_HTML_PATH = Path(__file__).resolve().parent.parent / "static" / "upload.html"


@router.api_route("/upload", methods=["GET", "HEAD"], response_class=HTMLResponse, include_in_schema=False)
async def upload_page() -> HTMLResponse:
    try:
        html = _HTML_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="Upload UI is missing from the image.") from e
    return HTMLResponse(html)


__all__ = ["router"]
