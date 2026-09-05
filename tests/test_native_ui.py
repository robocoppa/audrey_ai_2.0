"""Feature flag, asset routing, and browser policy for the native UI."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.routes import native_ui


def _client(tmp_path, monkeypatch, *, enabled: bool = True) -> TestClient:
    root = tmp_path / "app"
    root.mkdir()
    (root / "index.html").write_text("<html><body>Audrey native</body></html>")
    assets = root / "assets"
    assets.mkdir()
    (assets / "index-deadbeef.js").write_text("console.log('audrey')")
    monkeypatch.setattr(native_ui, "_STATIC_ROOT", root)

    app = FastAPI()
    app.state.cfg = SimpleNamespace(
        env=SimpleNamespace(native_ui_enabled=enabled),
    )
    app.include_router(native_ui.router)
    return TestClient(app)


def test_native_ui_is_hidden_when_feature_flag_is_disabled(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, enabled=False) as client:
        assert client.get("/app/", follow_redirects=False).status_code == 404
        assert client.get("/app/assets/index-deadbeef.js").status_code == 404


def test_native_ui_redirect_and_html_policy(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        redirect = client.get("/app", follow_redirects=False)
        assert redirect.status_code == 307
        assert redirect.headers["location"] == "/app/"

        response = client.get("/app/")
        assert response.status_code == 200
        assert response.text == "<html><body>Audrey native</body></html>"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "same-origin"
        assert response.headers["permissions-policy"] == (
            "camera=(), microphone=(), geolocation=()"
        )
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]


def test_native_ui_serves_hashed_assets_and_spa_routes(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        asset = client.get("/app/assets/index-deadbeef.js")
        assert asset.status_code == 200
        assert asset.text == "console.log('audrey')"
        assert asset.headers["content-type"].startswith("text/javascript")
        assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"

        route = client.get("/app/conversations/con_example")
        assert route.status_code == 200
        assert route.headers["cache-control"] == "no-store"
        assert "Audrey native" in route.text

        assert client.get("/app/assets/missing.js").status_code == 404


def test_native_ui_head_and_missing_build(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.head("/app/")
        assert response.status_code == 200
        assert response.content == b""

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(native_ui, "_STATIC_ROOT", empty)
    app = FastAPI()
    app.state.cfg = SimpleNamespace(env=SimpleNamespace(native_ui_enabled=True))
    app.include_router(native_ui.router)
    with TestClient(app) as client:
        response = client.get("/app/")
    assert response.status_code == 503
    assert response.json() == {"detail": "Native Audrey UI is not built."}


def test_native_ui_asset_resolution_cannot_escape_root(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    assert native_ui._resolve_asset(root, "assets/app.js") == root / "assets/app.js"
    assert native_ui._resolve_asset(root, "../secret") is None
    assert native_ui._resolve_asset(root, "/etc/passwd") is None
