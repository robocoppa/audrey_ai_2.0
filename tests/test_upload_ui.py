"""The upload page's server-rendered part (2026-08-06).

Everything else in `upload.html` is static and runs in a browser nothing here
can drive. The one thing the server decides is the Home link, because the page
cannot know it: the OWUI banner opens the page with `target="_blank"`, so there
is no history to go back through, and `OWUI_URL` — the only OWUI address the
process already had — resolves inside the docker network and nowhere else.

The failure this guards against is a link that renders and does not work, which
is worse than no link: it is the control someone reaches for when they are
already lost.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.routes import upload_ui


@pytest.fixture(autouse=True)
def _fresh_render():
    """The rendered page is cached for the life of the process."""
    upload_ui._RENDERED = None
    yield
    upload_ui._RENDERED = None


def _client(public_url: str | None) -> TestClient:
    app = FastAPI()
    app.include_router(upload_ui.router)
    if public_url is not None:
        app.state.cfg = SimpleNamespace(env=SimpleNamespace(owui_public_url=public_url))
    return TestClient(app)


class TestTheHomeLink:
    def test_a_configured_url_becomes_a_link(self):
        body = _client("https://chat.example.com").get("/upload").text
        assert 'href="https://chat.example.com"' in body
        assert "Home" in body

    def test_an_unset_url_renders_no_link_at_all(self):
        body = _client("").get("/upload").text
        # Not a hidden element. A Home button that goes nowhere is worse than
        # none — it is the control someone reaches for when they are lost.
        assert 'class="home"' not in body
        assert upload_ui._HOME_SLOT not in body

    def test_a_missing_cfg_is_survivable(self):
        # The page is a human endpoint and must not 500 because a deployment
        # wired its config differently.
        r = _client(None).get("/upload")
        assert r.status_code == 200
        assert 'class="home"' not in r.text

    def test_the_internal_docker_url_is_not_what_gets_rendered(self):
        # The mistake this whole setting exists to prevent. `owui_url` resolves
        # on ollama-net and nowhere a browser lives, so rendering it produces a
        # link that fails for every user while looking perfectly correct in
        # config.
        app = FastAPI()
        app.include_router(upload_ui.router)
        app.state.cfg = SimpleNamespace(env=SimpleNamespace(
            owui_url="http://open-webui:8080", owui_public_url="",
        ))
        body = TestClient(app).get("/upload").text
        assert "open-webui:8080" not in body

    @pytest.mark.parametrize("bad", [
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "  ",
        "chat.example.com",          # no scheme: would resolve as a relative path
    ])
    def test_only_http_urls_are_accepted(self, bad: str):
        # Deployment-supplied, not user-supplied, so this is a guard against a
        # typo or a paste — not a sanitizer holding back an attacker. It still
        # matters: this page holds a bearer token in localStorage.
        assert upload_ui._home_link(bad) == ""

    def test_the_url_is_escaped_into_the_attribute(self):
        link = upload_ui._home_link('https://x.example.com/"><script>')
        assert '"><script>' not in link
        assert "&quot;&gt;&lt;script&gt;" in link


def test_the_placeholder_never_survives_into_the_page():
    for url in ("https://chat.example.com", ""):
        upload_ui._RENDERED = None
        assert upload_ui._HOME_SLOT not in _client(url).get("/upload").text
