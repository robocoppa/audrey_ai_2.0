"""web_fetch — open one URL and return its main readable text.

Why this exists: `web_search` returns title + url + snippet (~150 chars). Models
trained in search-then-read harnesses repeatedly invent a page-opener
(`web_fetch` / `read_url`) to follow a promising result — and without one, a
researcher can cite a URL but never read past its snippet, where the exact
dates, version numbers, specs, and direct quotes actually live. This closes that
gap, and turns the recurring hallucinated `web_fetch ❌` into a real tool.

SAFETY — this fetches a MODEL-CHOSEN URL from inside `ollama-net`, i.e. it is a
model-steerable HTTP client on the LAN. The guards below are defense-in-depth
(chat-completions already requires auth):

  - **scheme allowlist** (http/https only) — no `file://`, `gopher://`, etc.
  - **host must not resolve to a private/loopback/link-local/reserved address**
    (`_is_unsafe_address`, ported from `audrey/kb/embed.py`) — blocks
    `http://qdrant:6333`, `http://127.0.0.1:9099`, cloud metadata, and friends.
  - **redirects are followed MANUALLY, re-validating every hop.** Automatic
    redirect-following is the classic bypass: `http://evil.example` returns
    `302 → http://qdrant:6333`, and the initial-host check never sees the
    internal target. So `follow_redirects=False` and we re-run `_validate_url`
    on each Location.
  - **byte cap on the raw download AND the decompressed body** — decoded output
    is bounded per-codec (`_decompress_bounded`), so a gzip/zstd bomb can't OOM
    the container before the cap trips; unknown encodings (brotli) are rejected.
  - **content-type gate** — only HTML/plain text is extracted; a binary blob is
    rejected before trafilatura ever sees it.
  - **timeout well under the tool-dispatch ceiling** so a slow page reports
    "fetch timed out" instead of surfacing as a bare dispatch timeout (the same
    legibility lesson as the KB timeout ladder).

KNOWN GAP (identical to the image path): DNS rebinding. We validate the host's
resolved IPs, but httpx re-resolves at connect time and could land elsewhere.
The real fix (connect to the validated IP, pass the hostname via SNI) is out of
scope; the bar is an attacker who controls DNS for a domain a research query is
induced to fetch.

We use ONLY `trafilatura.extract(html_string)` — never `trafilatura.fetch_url`,
which would do its own unguarded HTTP request and bypass everything above.
"""

from __future__ import annotations

import asyncio
import io
import ipaddress
import logging
import socket
import urllib.parse
import zlib

import httpx
import trafilatura

try:
    import zstandard
except ImportError:  # pragma: no cover - declared dep; guarded so an absent build fails safe
    zstandard = None

log = logging.getLogger("custom-tools")

_ALLOWED_SCHEMES = frozenset({"http", "https"})
# Content-type prefixes we can extract text from. Matched against the part
# before any ";charset=…". Empty content-type is allowed through (some servers
# omit it) — trafilatura then returns nothing for a non-HTML body and we raise.
_ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
_BYTE_CAP = 2 * 1024 * 1024        # 2 MB of HTML — generous for an article, bounds memory
_MAX_REDIRECTS = 4
_FETCH_TIMEOUT_S = 15.0            # < the 30s dispatch ceiling, so a slow page reports itself
# A plain desktop UA. The point is to read public pages the model already found;
# an obvious bot UA gets blocked by many sites, defeating the purpose.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class FetchError(Exception):
    """A web_fetch failed for a reportable, model-safe reason.

    The message is handed straight to the model (via the endpoint's 422 detail),
    so it must describe what to do next — pick another URL, or fall back to the
    search snippet — not leak internals.
    """


def _is_unsafe_address(host: str) -> bool:
    """True if any DNS-resolved IP for `host` is private / loopback / etc.

    Ported verbatim from `audrey/kb/embed.py`. Resolves via `getaddrinfo`
    (IPv4 + IPv6); any single resolved address in a non-public range fails the
    host, as does an unresolvable host (reject rather than risk a weird hit via
    DNS search domains).
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return True
    for *_, sockaddr in infos:
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return True
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            return True
    return False


def _validate_url(url: str) -> None:
    """Raise FetchError if `url` is unsafe to fetch. Called on every redirect hop."""
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname
        _ = parsed.port  # property access validates the port range (raises ValueError)
    except ValueError as e:
        # Malformed URL — a bad IPv6 literal (`http://[::1`) makes urlparse raise, an
        # out-of-range port makes `.port` raise. These reach us because the Pydantic
        # layer only length-checks the string; without this they surface as an uncaught
        # 500 instead of a model-safe reason to pick another URL.
        raise FetchError("the URL is malformed and cannot be opened") from e
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise FetchError(
            f"unsupported URL scheme {parsed.scheme or '(none)'!r}; only http/https can be opened"
        )
    if not hostname:
        raise FetchError("URL has no host")
    if _is_unsafe_address(hostname):
        raise FetchError(
            f"host {hostname!r} resolves to a private/internal address and cannot be opened"
        )


def _decompress_bounded(raw: bytes, encoding: str) -> bytes:
    """Decompress `raw` per its Content-Encoding with output bounded to the byte cap.

    httpx's transparent decoders call `decompressor.decompress(data)` with no length
    limit, so a few-KB gzip/zstd body can expand to hundreds of MB in a single step
    before any cap check runs. Here every codec is capped at `_BYTE_CAP + 1` output
    bytes — a decompression bomb raises instead of ballooning memory. Encodings we
    can't safely bound (brotli, or anything unknown) are rejected outright.
    """
    if encoding in ("", "identity"):
        return raw
    limit = _BYTE_CAP + 1  # +1 lets us tell "exactly at cap" (ok) from "over cap"
    over_cap = f"page body exceeds the {_BYTE_CAP // (1024 * 1024)} MB cap"
    if encoding in ("gzip", "x-gzip", "deflate"):
        # gzip/zlib headers auto-detect at wbits=47; raw deflate needs -15. The raw
        # buffer is already capped, so trying both formats is cheap.
        for wbits in (47, -15):
            try:
                dobj = zlib.decompressobj(wbits)
                out = dobj.decompress(raw, limit)
            except zlib.error:
                continue
            if dobj.unconsumed_tail:            # output hit the limit -> over cap
                raise FetchError(over_cap)
            out += dobj.flush()
            break
        else:
            raise FetchError("the page body could not be decompressed")
    elif encoding == "zstd" and zstandard is not None:
        try:
            reader = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw))
            out = reader.read(limit)            # decompresses at most `limit` bytes
        except zstandard.ZstdError as e:
            raise FetchError("the page body could not be decompressed") from e
    else:
        raise FetchError(f"page uses unsupported content-encoding {encoding!r}")
    if len(out) > _BYTE_CAP:
        raise FetchError(over_cap)
    return out


async def _read_capped(resp: httpx.Response) -> bytes:
    """Stream the raw body under a byte cap, then decompress with bounded output.

    We iterate `aiter_raw()` (undecoded wire bytes) rather than `aiter_bytes()` so
    decompression runs through `_decompress_bounded`, which bounds decoded output.
    This caps both the raw download and the decompressed result, so no compression
    ratio can OOM the container before the cap trips.
    """
    encodings = [e.strip().lower() for e in resp.headers.get("content-encoding", "").split(",")]
    encodings = [e for e in encodings if e and e != "identity"]
    if len(encodings) > 1:
        raise FetchError("page uses multiple content-encodings, which are not supported")
    encoding = encodings[0] if encodings else ""

    chunks: list[bytes] = []
    total = 0
    async for chunk in resp.aiter_raw():
        total += len(chunk)
        if total > _BYTE_CAP:
            raise FetchError(f"page body exceeds the {_BYTE_CAP // (1024 * 1024)} MB cap")
        chunks.append(chunk)
    return _decompress_bounded(b"".join(chunks), encoding)


def _extract_text(html: str) -> str:
    """Pure, sync trafilatura extraction. Run via a thread — lxml parsing blocks."""
    return (trafilatura.extract(html, include_comments=False, include_tables=True) or "").strip()


async def fetch_readable(
    url: str, *, max_chars: int, transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[str, str]:
    """Fetch `url` and return (final_url_after_redirects, extracted_text).

    Raises FetchError for every reportable failure. `transport` is a test seam
    (an httpx.MockTransport); production passes None.
    """
    _validate_url(url)
    current = url
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=_FETCH_TIMEOUT_S,
        headers={"User-Agent": _USER_AGENT},
        transport=transport,
    ) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            _validate_url(current)  # re-validate BEFORE following any hop
            try:
                async with client.stream("GET", current) as resp:
                    if resp.is_redirect:
                        loc = resp.headers.get("location")
                        if not loc:
                            raise FetchError("server sent a redirect with no Location header")
                        current = str(httpx.URL(current).join(loc))
                        continue
                    if resp.status_code >= 400:
                        raise FetchError(f"the page returned HTTP {resp.status_code}")
                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if ctype and ctype not in _ALLOWED_CONTENT_TYPES:
                        raise FetchError(
                            f"page is {ctype!r}, not readable text; only HTML/plain-text can be opened"
                        )
                    raw = await _read_capped(resp)
            except httpx.TimeoutException as e:
                raise FetchError(f"fetch timed out after {int(_FETCH_TIMEOUT_S)}s") from e
            except (httpx.HTTPError, httpx.InvalidURL, ValueError) as e:
                # httpx.InvalidURL is NOT a subclass of httpx.HTTPError, and httpx can
                # raise a bare ValueError on some malformed inputs — a control char in
                # the URL or a redirect Location triggers this. Both must map to a clean
                # reportable failure, never an uncaught 500 that bypasses FetchError.
                raise FetchError(f"could not reach the page ({type(e).__name__})") from e

            html = raw.decode("utf-8", errors="replace")
            text = await asyncio.to_thread(_extract_text, html)
            if not text:
                raise FetchError("no readable text could be extracted from the page")
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "\n…[truncated]"
            log.info("web_fetch: %s -> %d chars", current, len(text))
            return current, text

    raise FetchError(f"too many redirects (more than {_MAX_REDIRECTS})")
