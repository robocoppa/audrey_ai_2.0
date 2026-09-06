"""Serve the feature-gated native Audrey single-page application."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, Response

router = APIRouter(tags=["ui"])

_STATIC_ROOT = Path(__file__).resolve().parent.parent / "static" / "app"
_HTML_CACHE = "no-store"
_ASSET_CACHE = "public, max-age=31536000, immutable"
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "connect-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self'; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'; "
        "form-action 'self'"
    ),
    "Referrer-Policy": "same-origin",
    "X-Content-Type-Options": "nosniff",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def _enabled(request: Request) -> bool:
    cfg = getattr(request.app.state, "cfg", None)
    return bool(getattr(getattr(cfg, "env", None), "native_ui_enabled", False))


def _resolve_asset(root: Path, requested: str) -> Path | None:
    """Resolve one browser path inside ``root`` without traversal."""

    root = root.resolve()
    candidate = (root / requested).resolve()
    if not candidate.is_relative_to(root):
        return None
    return candidate


def _response(path: Path, *, html: bool) -> FileResponse:
    headers = {
        **_SECURITY_HEADERS,
        "Cache-Control": _HTML_CACHE if html else _ASSET_CACHE,
    }
    return FileResponse(path, media_type="text/html" if html else None, headers=headers)


def _index(root: Path) -> FileResponse:
    index = root / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=503, detail="Native Audrey UI is not built.")
    return _response(index, html=True)


@router.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
async def native_ui_root(request: Request) -> FileResponse:
    if not _enabled(request):
        raise HTTPException(status_code=404, detail="Not found.")
    return _index(_STATIC_ROOT.resolve())


@router.api_route(
    "/assets/{asset_path:path}",
    methods=["GET", "HEAD"],
    include_in_schema=False,
)
async def native_ui_root_asset(asset_path: str, request: Request) -> FileResponse:
    if not _enabled(request):
        raise HTTPException(status_code=404, detail="Not found.")

    asset = _resolve_asset(_STATIC_ROOT, f"assets/{asset_path}")
    if asset is None or not asset.is_file():
        raise HTTPException(status_code=404, detail="Asset not found.")
    return _response(asset, html=False)


@router.api_route("/app", methods=["GET", "HEAD"], include_in_schema=False)
async def native_ui_redirect(request: Request) -> Response:
    if not _enabled(request):
        raise HTTPException(status_code=404, detail="Not found.")
    return RedirectResponse(url="/", status_code=307, headers=_SECURITY_HEADERS)


@router.api_route("/app/{asset_path:path}", methods=["GET", "HEAD"], include_in_schema=False)
async def native_ui(asset_path: str, request: Request) -> Response:
    if not _enabled(request):
        raise HTTPException(status_code=404, detail="Not found.")

    root = _STATIC_ROOT.resolve()
    if not asset_path:
        return RedirectResponse(url="/", status_code=307, headers=_SECURITY_HEADERS)

    asset = _resolve_asset(root, asset_path)
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found.")
    if asset.is_file():
        return _response(asset, html=False)

    # Client-side routes have no suffix. A missing filename such as a stale
    # hashed script must remain a 404 rather than receiving HTML as JavaScript.
    if not Path(asset_path).suffix:
        return _index(root)
    raise HTTPException(status_code=404, detail="Asset not found.")


__all__ = ["router"]
