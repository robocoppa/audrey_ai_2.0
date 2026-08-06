"""Serve the upload UI at GET /upload.

Single static HTML shipped from `audrey/static/upload.html`. Kept out of
the FastAPI OpenAPI docs (`include_in_schema=False`) since it's a human
endpoint, not part of the API surface.

Users reach this from a link in an OWUI banner. **Open WebUI has no
sidebar-links feature** — phase 40 planned around one that does not exist, and
this docstring named it too. A banner is the supported way to put a link in
front of an OWUI user; see `docs/campaign-2/phase-40-uploads-in-chat.md` step 1.

The one thing the page cannot know for itself is the way back. The banner opens
it with `target="_blank"`, so there is no history to go back through, and phase
40 recorded the consequence: a banner is a strip above the chat pane, and once
it is dismissed there is nothing to navigate back to. So the Home link is
rendered here, from `OWUI_PUBLIC_URL`, rather than hardcoded in the page.
"""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_HTML_PATH = Path(__file__).resolve().parent.parent / "static" / "upload.html"

#: Replaced at serve time by the Home link, or by nothing. A placeholder rather
#: than a JS hook so a deployment with no `OWUI_PUBLIC_URL` ships a page with no
#: dead control in it at all, instead of one that hides a button after paint.
_HOME_SLOT = "<!--HOME-->"

_RENDERED: str | None = None


def _home_link(url: str) -> str:
    """The Home anchor, or empty when there is nowhere to send anyone.

    Only `http(s)` is accepted. The value is deployment-supplied rather than
    user-supplied, so this is not a sanitizer standing between an attacker and
    a victim — it is a guard against a typo or a `javascript:` paste becoming a
    live link on a page that holds a bearer token.
    """
    url = (url or "").strip()
    if not url or not url.lower().startswith(("http://", "https://")):
        return ""
    return f'<a class="home" href="{html.escape(url, quote=True)}">← Home</a>'


@router.api_route("/upload", methods=["GET", "HEAD"], response_class=HTMLResponse, include_in_schema=False)
async def upload_page(request: Request) -> HTMLResponse:
    # Rendered once. The file ships in the container image and the URL comes
    # from the environment, so neither changes while the process is alive —
    # re-reading and re-substituting per request is wasted work.
    global _RENDERED
    if _RENDERED is None:
        try:
            raw = _HTML_PATH.read_text(encoding="utf-8")
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail="Upload UI is missing from the image.") from e
        cfg = getattr(request.app.state, "cfg", None)
        public = getattr(getattr(cfg, "env", None), "owui_public_url", "") if cfg else ""
        _RENDERED = raw.replace(_HOME_SLOT, _home_link(public))
    return HTMLResponse(_RENDERED)


__all__ = ["router"]
