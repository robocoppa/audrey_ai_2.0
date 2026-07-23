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


class _AsyncBytes(httpx.AsyncByteStream):
    """One-shot async byte stream. A MockTransport response built with `content=`
    pre-sets `_content` and marks the stream consumed, which `aiter_raw()` rejects;
    a real streaming response (what production sees) exposes a raw stream. fetch.py
    reads `aiter_raw()` to bound decompression, so response-body tests must serve
    genuine streams rather than `content=`.
    """

    def __init__(self, data: bytes):
        self._data = data

    async def __aiter__(self):
        yield self._data


def _resp(status_code: int, *, headers: dict | None = None, content: bytes = b"") -> httpx.Response:
    return httpx.Response(status_code, headers=headers or {}, stream=_AsyncBytes(content))


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
        return _resp(200, headers={"content-type": "text/html"}, content=big)

    with pytest.raises(FetchError, match="cap"):
        await fetch_readable(
            "http://example.com/big", max_chars=6000,
            transport=httpx.MockTransport(handler),
        )


# ─── The happy path + truncation ──────────────────────────────────────

async def test_successful_fetch_extracts_readable_text(monkeypatch):
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)

    def handler(request: httpx.Request) -> httpx.Response:
        return _resp(200, headers={"content-type": "text/html; charset=utf-8"},
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
        return _resp(200, headers={"content-type": "text/html"}, content=_HTML.encode())

    _, text = await fetch_readable(
        "http://example.com/paper", max_chars=20,
        transport=httpx.MockTransport(handler),
    )
    assert text.endswith("…[truncated]")


async def test_empty_extraction_reports_no_text(monkeypatch):
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)

    def handler(request: httpx.Request) -> httpx.Response:
        # Valid HTML with nothing trafilatura will treat as article content.
        return _resp(200, headers={"content-type": "text/html"},
                     content=b"<html><body></body></html>")

    with pytest.raises(FetchError, match="no readable text"):
        await fetch_readable(
            "http://example.com/empty", max_chars=6000,
            transport=httpx.MockTransport(handler),
        )


# ─── The validation seam: malformed URLs report, they don't crash ─────
# These pass Pydantic (any 1–2000-char string) but are malformed for urlparse or
# httpx. httpx.InvalidURL is NOT an httpx.HTTPError, and urlparse/.port raise bare
# ValueErrors — without explicit handling each surfaces as an uncaught 500 that
# bypasses the model-safe FetchError contract. Every one must become a FetchError.

@pytest.mark.parametrize("url", [
    "http://8.8.8.8\t/",          # control char → httpx.InvalidURL (raised before any I/O)
    "http://8.8.8.8\n/",
    "http://8.8.8.8/\x00",
    "http://8.8.8.8#\r\nSet-Cookie: x",
    "http://[::1",                # bad IPv6 literal → urlparse ValueError
    "http://8.8.8.8:99999999/",   # port out of range → .port ValueError
])
async def test_malformed_url_reports_not_crashes(monkeypatch, url):
    # Stub the DNS guard so the control-char cases get past validation and reach
    # httpx (which rejects the URL before opening a socket — still hermetic). The
    # urlparse/port cases fail earlier, inside _validate_url.
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)
    with pytest.raises(FetchError):
        await fetch_readable(url, max_chars=6000)


def test_validate_url_rejects_out_of_range_port():
    with pytest.raises(FetchError, match="malformed"):
        _validate_url("http://8.8.8.8:99999999/")


def test_validate_url_rejects_bad_ipv6_literal():
    with pytest.raises(FetchError, match="malformed"):
        _validate_url("http://[::1")


# ─── trafilatura is XXE-safe: the HTML parser ignores DOCTYPE entities ─
# _extract_text feeds a hostile HTML string to trafilatura, which parses via
# libxml2's HTML parser — custom DOCTYPE entities are never resolved, so no
# local-file read and no entity-expansion blowup are reachable from a fetched page.

_ARTICLE_PAD = (
    "<p>The quick brown fox jumps over the lazy dog and this paragraph is long "
    "enough that trafilatura keeps it as the main article content of the page.</p>"
)


def test_extract_ignores_external_file_entity(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("XXE-CANARY-MUST-NOT-LEAK")
    payload = (
        '<?xml version="1.0"?>'
        f'<!DOCTYPE html [ <!ENTITY xxe SYSTEM "file://{secret}"> ]>'
        "<html><body><article><h1>Real Article Title</h1>"
        f"{_ARTICLE_PAD}<p>value: &xxe; and here is more body text to keep the "
        f"extractor happy indeed.</p>{_ARTICLE_PAD}</article></body></html>"
    )
    text = fetch._extract_text(payload)
    assert "CANARY" not in text          # entity was never expanded → no file read
    assert "lazy dog" in text            # the page WAS extracted, so the entity was in-scope


def test_extract_does_not_expand_entity_bomb():
    # Billion-laughs: nested entities must not expand (no CPU/memory blowup). The HTML
    # parser leaves them undefined, so the output stays small.
    bomb = (
        '<?xml version="1.0"?><!DOCTYPE lolz ['
        '<!ENTITY lol "lol">'
        '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
        '<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
        '<!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">'
        ']>'
        "<html><body><article><p>boom &lol4; plus filler words to reach the "
        "extraction threshold here now.</p></article></body></html>"
    )
    text = fetch._extract_text(bomb)
    assert len(text) < 10_000


# ─── The endpoint boundary: bad input is 422, never 500 ───────────────
# The Pydantic model (url 1–2000 chars, max_chars 500–20000) is enforced only at the
# FastAPI edge and had no test. TestClient exercises the real /web_fetch route.

@pytest.fixture(scope="module")
def client():
    import app  # tools-server app; imported lazily so a heavy import can't break collection
    from starlette.testclient import TestClient
    return TestClient(app.app)


@pytest.mark.parametrize("payload", [
    {},                                            # url missing
    {"url": ""},                                   # too short
    {"url": "x" * 2001},                           # too long
    {"url": "http://a.com", "max_chars": 499},     # below floor
    {"url": "http://a.com", "max_chars": 0},
    {"url": "http://a.com", "max_chars": -5},
    {"url": "http://a.com", "max_chars": 20001},   # above ceiling
    {"url": 123},                                  # wrong type
])
def test_web_fetch_endpoint_rejects_bad_input_422(client, payload):
    # Pydantic rejects these before the handler runs — no network touched.
    assert client.post("/web_fetch", json=payload).status_code == 422


def test_web_fetch_endpoint_malformed_url_is_422_not_500(client):
    # Passes Pydantic (valid string) but is malformed for urlparse → must map to the
    # model-safe 422 via FetchError, not an uncaught 500. urlparse raises before any DNS.
    assert client.post("/web_fetch", json={"url": "http://[::1"}).status_code == 422


# ─── Decompression is bounded: a compression bomb can't balloon memory ─
# The cap must count DECODED bytes AND bound the transient. httpx's own decoders
# expand each chunk fully before any check (a few-KB body → hundreds of MB), so
# fetch.py reads raw bytes and decompresses with a per-codec output limit. Bombs
# must raise, and peak memory must stay near the cap — not the decompressed size.

_MEM_CEILING = 24 * 1024 * 1024   # generous headroom over the 2 MB cap; the pre-fix
                                  # path peaked at 200-400 MB, so this is a wide margin


def _gzip(data: bytes) -> bytes:
    import gzip
    return gzip.compress(data)


def _handler_for(content: bytes, encoding: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return _resp(
            200, headers={"content-type": "text/html", "content-encoding": encoding},
            content=content,
        )
    return handler


async def test_gzip_bomb_rejected_with_bounded_memory(monkeypatch):
    import tracemalloc
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)
    bomb = _gzip(b"A" * (80 * 1024 * 1024))        # 80 MB → ~80 KB on the wire
    tracemalloc.start()
    try:
        with pytest.raises(FetchError, match="cap"):
            await fetch_readable(
                "http://e.example/b", max_chars=6000,
                transport=httpx.MockTransport(_handler_for(bomb, "gzip")),
            )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < _MEM_CEILING, f"peak {peak / 1e6:.0f} MB — decompression not bounded"


async def test_zstd_bomb_rejected_with_bounded_memory(monkeypatch):
    import tracemalloc
    zstd = pytest.importorskip("zstandard")
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)
    bomb = zstd.ZstdCompressor(level=19).compress(b"B" * (150 * 1024 * 1024))
    tracemalloc.start()
    try:
        with pytest.raises(FetchError, match="cap"):
            await fetch_readable(
                "http://e.example/z", max_chars=6000,
                transport=httpx.MockTransport(_handler_for(bomb, "zstd")),
            )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < _MEM_CEILING, f"peak {peak / 1e6:.0f} MB — decompression not bounded"


async def test_gzip_encoded_page_is_decoded(monkeypatch):
    # The happy path: a legitimately gzip-encoded page must still decode and extract.
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)
    _, text = await fetch_readable(
        "http://e.example/g", max_chars=6000,
        transport=httpx.MockTransport(_handler_for(_gzip(_HTML.encode()), "gzip")),
    )
    assert "self-attention" in text


async def test_raw_deflate_encoded_page_is_decoded(monkeypatch):
    # Raw deflate (no zlib header) exercises the wbits=-15 fallback in the decoder.
    import zlib
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    body = co.compress(_HTML.encode()) + co.flush()
    _, text = await fetch_readable(
        "http://e.example/d", max_chars=6000,
        transport=httpx.MockTransport(_handler_for(body, "deflate")),
    )
    assert "self-attention" in text


async def test_unsupported_content_encoding_rejected(monkeypatch):
    # brotli isn't handled — an attacker-forced `br` body must be rejected, never
    # handed to trafilatura as raw bytes.
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)
    with pytest.raises(FetchError, match="content-encoding"):
        await fetch_readable(
            "http://e.example/br", max_chars=6000,
            transport=httpx.MockTransport(_handler_for(b"\x1b\x0e\x00\xf8nope", "br")),
        )


async def test_multiple_content_encodings_rejected(monkeypatch):
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)
    with pytest.raises(FetchError, match="multiple content-encodings"):
        await fetch_readable(
            "http://e.example/m", max_chars=6000,
            transport=httpx.MockTransport(_handler_for(b"x", "gzip, br")),
        )


# ─── Overall deadline + concurrency cap: a whole call is bounded ───────
# `_FETCH_TIMEOUT_S` is per-operation and resets on each redirect hop, so it can't
# bound a whole chain; `fetch_readable` wraps the call in an overall wall-clock
# deadline (under the 30s dispatch ceiling) and a global concurrency semaphore
# acquired inside that deadline.

async def test_overall_deadline_stops_a_slow_fetch(monkeypatch):
    # A page slower than the deadline must raise near the deadline — not run to the
    # 15s per-op timeout, not hang. The deadline, not the per-op timeout, is the wall.
    import asyncio
    import time
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)
    monkeypatch.setattr(fetch, "_OVERALL_DEADLINE_S", 0.2)

    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return _resp(200, headers={"content-type": "text/html"}, content=_HTML.encode())

    t = time.monotonic()
    with pytest.raises(FetchError, match="deadline"):
        await fetch_readable(
            "http://e.example/slow", max_chars=6000, transport=httpx.MockTransport(slow),
        )
    assert time.monotonic() - t < 3, "deadline did not fire near _OVERALL_DEADLINE_S"


async def test_concurrency_cap_bounds_in_flight(monkeypatch):
    # web_fetch is model-steerable; a burst must not run unbounded fetches at once.
    # With the semaphore at 2, exactly 2 handlers are ever in flight together — the
    # cap holds, and concurrency actually happens (peak == 2, not serialized to 1).
    import asyncio
    monkeypatch.setattr(fetch, "_is_unsafe_address", lambda _h: False)
    monkeypatch.setattr(fetch, "_FETCH_SEMAPHORE", asyncio.Semaphore(2))

    in_flight = 0
    peak = 0

    async def tracked(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return _resp(200, headers={"content-type": "text/html"}, content=_HTML.encode())

    transport = httpx.MockTransport(tracked)
    await asyncio.gather(*(
        fetch_readable("http://e.example/c", max_chars=6000, transport=transport)
        for _ in range(6)
    ))
    assert peak == 2, f"peak in-flight {peak}, expected the cap of 2"
