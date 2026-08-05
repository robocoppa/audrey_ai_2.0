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
  - **the host is resolved ONCE and the socket is PINNED to a vetted IP**
    (`_resolve_safe`): every address `getaddrinfo` returns must be public — a
    single private/loopback/link-local/reserved IP (or a resolution failure) fails
    the whole host — and the request then connects to that vetted IP directly, with
    the real hostname carried via the `Host` header and the TLS `sni_hostname`
    extension (SNI + cert verification still use the real name). Because httpx never
    re-resolves, there is no DNS-rebinding TOCTOU: the socket can only ever reach an
    address we validated. Blocks `http://qdrant:6333`, `http://127.0.0.1:9099`,
    cloud metadata, and friends.
  - **redirects are followed MANUALLY, re-resolving+re-pinning every hop.**
    Automatic redirect-following is the classic bypass: `http://evil.example`
    returns `302 → http://qdrant:6333`, and the initial-host check never sees the
    internal target. So `follow_redirects=False` and we re-run `_validate_url` +
    `_resolve_safe` on each Location before connecting.
  - **byte cap on the raw download AND the decompressed body** — decoded output
    is bounded per-codec (`_decompress_bounded`), so a gzip/zstd bomb can't OOM
    the container before the cap trips; unknown encodings (brotli) are rejected.
  - **content-type gate** — only HTML/plain text is extracted; a binary blob is
    rejected before trafilatura ever sees it.
  - **a per-operation timeout AND an overall wall-clock deadline, both under the
    tool-dispatch ceiling** so a slow or redirect-chained page reports itself
    instead of surfacing as a bare dispatch timeout (the same legibility lesson as
    the KB timeout ladder). The per-op timeout resets on each redirect hop, so it
    can't bound a whole chain; the overall deadline hard-stops the entire call.
  - **a concurrency cap** — web_fetch is model-steerable, so a burst mustn't open
    unbounded sockets or decompress unbounded bodies at once; excess callers wait
    out the deadline and fail as a clean timeout instead of hanging.

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
from settings import settings

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
_FETCH_TIMEOUT_S = 15.0            # per-op; resets each redirect hop — see _OVERALL_DEADLINE_S
# Overall wall-clock deadline for a whole fetch (redirects + reads + extraction).
# `_FETCH_TIMEOUT_S` is per-operation and resets on each hop, so a chain of
# slow-but-under-15s hops could stack past Audrey's 30s dispatch ceiling and orphan
# the coroutine. This hard-stops the whole call under that ceiling. Config-surfaced.
_OVERALL_DEADLINE_S = settings.web_fetch_overall_deadline_s
# Cap on concurrent in-flight fetches. Acquired INSIDE the deadline (see
# fetch_readable), so a saturated pool makes callers wait it out and fail as a clean
# timeout rather than hanging or opening unbounded sockets. Config-surfaced.
_MAX_CONCURRENT_FETCHES = settings.web_fetch_max_concurrent
_FETCH_SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)
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


def _ip_is_unsafe(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True if `ip` is in a non-public range we must never connect to."""
    return (ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified)


def _resolve_safe(host: str) -> list[str]:
    """Resolve `host` and return its vetted public IPs, or raise FetchError.

    The classification is ported from `audrey/kb/embed.py`, but this does more than
    the old bool guard: it resolves ONCE and hands back the IPs so the fetch can PIN
    the socket to one of them. That closes the DNS-rebinding TOCTOU — a validate-now,
    httpx-re-resolves-at-connect gap where the second lookup could land internal.
    Every address `getaddrinfo` returns must be public (a host that resolves partly
    internal is hostile); an unresolvable host is rejected rather than risking a hit
    via DNS search domains. `SOCK_STREAM` dedupes the per-socktype duplicates.
    """
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise FetchError(f"host {host!r} could not be resolved") from e
    ips: list[str] = []
    for *_, sockaddr in infos:
        addr = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError as e:
            raise FetchError(f"host {host!r} resolved to an unparseable address") from e
        if _ip_is_unsafe(ip):
            raise FetchError(
                f"host {host!r} resolves to a private/internal address and cannot be opened"
            )
        if addr not in ips:
            ips.append(addr)
    if not ips:
        raise FetchError(f"host {host!r} could not be resolved")
    return ips


def _validate_url(url: str) -> str:
    """Validate `url`'s scheme + shape and return its hostname. Raise FetchError.

    Does NOT resolve — resolution and address vetting happen in `_resolve_safe` at
    fetch time so the vetted IPs can be pinned into the connection. Called on every
    redirect hop.
    """
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
    return hostname


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

    Wraps `_fetch_readable` in two bounds: an overall wall-clock deadline
    (`_OVERALL_DEADLINE_S`, under Audrey's 30s dispatch ceiling) and a global
    concurrency semaphore. The semaphore is acquired INSIDE the deadline, so when
    the pool is saturated a caller waits it out and fails as a clean FetchError
    timeout rather than hanging past the ceiling and orphaning the coroutine.
    Raises FetchError for every reportable failure. `transport` is a test seam.
    """
    try:
        async with asyncio.timeout(_OVERALL_DEADLINE_S), _FETCH_SEMAPHORE:
            return await _fetch_readable(url, max_chars=max_chars, transport=transport)
    except TimeoutError as e:
        raise FetchError(f"fetch exceeded the {int(_OVERALL_DEADLINE_S)}s deadline") from e


async def _send_pinned(
    client: httpx.AsyncClient, target: httpx.URL, *,
    ips: list[str], hostname: str, host_header: str,
) -> httpx.Response:
    """Send to the first vetted IP that accepts a connection.

    The socket is pinned to an address `_resolve_safe` already vetted, while
    the real hostname rides in the `Host` header (vhost routing) and the
    `sni_hostname` extension (TLS SNI + certificate verification). httpx never
    re-resolves, so the connection cannot be rebound to an internal address
    after the check — that is the DNS-rebinding TOCTOU this pinning closes,
    and trying several addresses must not reopen it. **Every candidate here
    comes from the same vetted list**; this loop widens which vetted address
    is used, never what counts as vetted.

    Why more than one: pinning replaced anyio's own multi-address behaviour,
    which tries each result in turn. Taking `ips[0]` alone means a host whose
    first A record is down is unreachable to us and reachable to every other
    client — a dual-stack or anycast host with one bad endpoint fails
    permanently, and the error says "connection failed" with no hint that a
    working address was sitting in the list.

    **Only connection failures advance to the next address.** Once a server
    responds, the exchange belongs to that address: a 500, a redirect, or a
    read error mid-body are all answers, and retrying them elsewhere would
    turn one request into several and could replay a non-idempotent redirect
    target. `ConnectError` and `ConnectTimeout` are the only things that mean
    "this address never spoke to us".
    """
    last: Exception | None = None
    for ip in ips:
        req = client.build_request(
            "GET", target.copy_with(host=ip),
            extensions={"sni_hostname": hostname},
        )
        req.headers["Host"] = host_header
        try:
            return await client.send(req, stream=True, follow_redirects=False)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            last = e
            log.debug("web_fetch: %s did not accept a connection (%s)", ip, e)
            continue
    raise FetchError(
        f"could not connect to {hostname} on any of its "
        f"{len(ips)} address(es)"
    ) from last


async def _fetch_readable(
    url: str, *, max_chars: int, transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[str, str]:
    """Validate, follow redirects manually re-validating each hop, and extract text.

    Raises FetchError for every reportable failure. Runs under the deadline +
    concurrency guard in `fetch_readable`; don't call directly outside a bound.
    """
    current = url
    async with httpx.AsyncClient(
        follow_redirects=False,
        timeout=_FETCH_TIMEOUT_S,
        headers={"User-Agent": _USER_AGENT},
        transport=transport,
    ) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            hostname = _validate_url(current)   # scheme/shape; returns the hostname
            ips = _resolve_safe(hostname)       # resolve + vet BEFORE connecting to any hop
            try:
                # Pin the socket to a vetted IP: connect to the IP, but keep the real
                # hostname in the Host header (vhost routing) and the sni_hostname
                # extension (TLS SNI + cert verification). httpx never re-resolves, so
                # the connection can't be rebound to an internal address after the vet.
                target = httpx.URL(current)
                host_header = client.build_request("GET", target).headers["host"]
                resp = await _send_pinned(
                    client, target, ips=ips, hostname=hostname, host_header=host_header,
                )
                try:
                    if resp.is_redirect:
                        loc = resp.headers.get("location")
                        if not loc:
                            raise FetchError("server sent a redirect with no Location header")
                        current = str(target.join(loc))
                        continue
                    if resp.status_code >= 400:
                        raise FetchError(f"the page returned HTTP {resp.status_code}")
                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    if ctype and ctype not in _ALLOWED_CONTENT_TYPES:
                        raise FetchError(
                            f"page is {ctype!r}, not readable text; only HTML/plain-text can be opened"
                        )
                    raw = await _read_capped(resp)
                finally:
                    await resp.aclose()
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
