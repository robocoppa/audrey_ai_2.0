"""Hermetic tests for the web_fetch page-opener (tools-server/fetch.py).

web_fetch fetches a MODEL-CHOSEN URL from inside ollama-net, so its whole reason
to be careful is SSRF: it must not become a lever for reaching internal services.
These tests exercise the guards without touching the network — httpx.MockTransport
serves canned responses, and the DNS guard is checked directly (raw-IP inputs to
getaddrinfo don't hit the network) or monkeypatched away where a test is about a
different concern.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

import fetch  # noqa: E402
from fetch import FetchError, _is_unsafe_address, _validate_url, fetch_readable  # noqa: E402

_HTML = (
    "<html><head><title>T</title><style>.x{}</style></head><body>"
    "<nav>skip me</nav><article><h1>Attention Is All You Need</h1>"
    "<p>The Transformer relies entirely on self-attention.</p></article>"
    "<script>tracker()</script></body></html>"
)


# ─── The DNS guard, directly (raw IPs don't hit the network) ──────────

@pytest.mark.parametrize("host", [
    "127.0.0.1", "10.0.0.5", "192.168.1.11", "169.254.1.1", "::1", "0.0.0.0",  # noqa: S104 — SSRF-guard inputs, not a bind address
])
def test_is_unsafe_address_blocks_internal(host):
    assert _is_unsafe_address(host) is True


@pytest.mark.parametrize("host", ["8.8.8.8", "1.1.1.1"])
def test_is_unsafe_address_allows_public(host):
    assert _is_unsafe_address(host) is False


def test_unresolvable_host_is_unsafe():
    # A name that won't resolve → reject rather than risk a search-domain hit.
    assert _is_unsafe_address("no-such-host.invalid") is True


# ─── URL validation ──────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "gopher://example.com/",
    "ftp://example.com/x",
])
def test_validate_url_rejects_nonhttp_schemes(url):
    with pytest.raises(FetchError, match="scheme"):
        _validate_url(url)


def test_validate_url_rejects_internal_host():
    with pytest.raises(FetchError, match="private/internal"):
        _validate_url("http://127.0.0.1:6333/collections")


# ─── The redirect-revalidation bypass (the reason this tool was hard) ─

async def test_redirect_to_internal_host_is_blocked(monkeypatch):
    # evil.example 302s to an internal service. Automatic redirect-following
    # would sail past the initial-host check; manual per-hop revalidation must
    # catch the internal Location. We fake DNS so evil.example looks public and
    # the redirect target looks internal.
    def fake_unsafe(host: str) -> bool:
        return host != "evil.example"

    monkeypatch.setattr(fetch, "_is_unsafe_address", fake_unsafe)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://qdrant:6333/collections"})

    with pytest.raises(FetchError, match="private/internal"):
        await fetch_readable(
            "http://evil.example/start", max_chars=6000,
            transport=httpx.MockTransport(handler),
        )


async def test_too_many_redirects(monkeypatch):
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)

    def handler(request: httpx.Request) -> httpx.Response:
        # Always redirect onward to another public-looking host.
        return httpx.Response(302, headers={"location": "http://loop.example/next"})

    with pytest.raises(FetchError, match="too many redirects"):
        await fetch_readable(
            "http://loop.example/start", max_chars=6000,
            transport=httpx.MockTransport(handler),
        )


# ─── Response gating ──────────────────────────────────────────────────

async def test_non_html_content_type_rejected(monkeypatch):
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF-")

    with pytest.raises(FetchError, match="not readable text"):
        await fetch_readable(
            "http://example.com/doc.pdf", max_chars=6000,
            transport=httpx.MockTransport(handler),
        )


async def test_http_error_status_rejected(monkeypatch):
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, headers={"content-type": "text/html"}, content=b"nope")

    with pytest.raises(FetchError, match="HTTP 404"):
        await fetch_readable(
            "http://example.com/missing", max_chars=6000,
            transport=httpx.MockTransport(handler),
        )


async def test_byte_cap_enforced(monkeypatch):
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)
    monkeypatch.setattr(fetch, "_BYTE_CAP", 1024)  # tiny cap for the test
    big = b"<html><body>" + b"x" * 4096 + b"</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=big)

    with pytest.raises(FetchError, match="cap"):
        await fetch_readable(
            "http://example.com/big", max_chars=6000,
            transport=httpx.MockTransport(handler),
        )


# ─── The happy path + truncation ──────────────────────────────────────

async def test_successful_fetch_extracts_readable_text(monkeypatch):
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"},
                              content=_HTML.encode())

    final_url, text = await fetch_readable(
        "http://example.com/paper", max_chars=6000,
        transport=httpx.MockTransport(handler),
    )
    assert final_url == "http://example.com/paper"
    assert "self-attention" in text
    # <script>/<style> contents must never leak into extracted text — trafilatura
    # drops them reliably even on a minimal document. (Nav-stripping needs a
    # realistic page to kick in, so we don't assert it on this toy fixture — that
    # would be testing trafilatura's heuristics, not our code.)
    assert "tracker" not in text
    assert ".x{}" not in text


async def test_max_chars_truncates(monkeypatch):
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=_HTML.encode())

    _, text = await fetch_readable(
        "http://example.com/paper", max_chars=20,
        transport=httpx.MockTransport(handler),
    )
    assert text.endswith("…[truncated]")


async def test_empty_extraction_reports_no_text(monkeypatch):
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)

    def handler(request: httpx.Request) -> httpx.Response:
        # Valid HTML with nothing trafilatura will treat as article content.
        return httpx.Response(200, headers={"content-type": "text/html"},
                              content=b"<html><body></body></html>")

    with pytest.raises(FetchError, match="no readable text"):
        await fetch_readable(
            "http://example.com/empty", max_chars=6000,
            transport=httpx.MockTransport(handler),
        )
