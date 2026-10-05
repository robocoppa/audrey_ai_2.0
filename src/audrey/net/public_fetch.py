"""Fetch public assets through sockets pinned to vetted DNS answers.

No ambient proxies, cookies, caller headers, credentials, or automatic redirects.
Each redirect repeats URL and address validation before opening its socket.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass

import httpx
from fastapi import HTTPException

MAX_URL_CHARS = 4096
MAX_REDIRECTS = 3
FETCH_DEADLINE_SECONDS = 30
_PROTOCOL_NETWORKS = (ipaddress.ip_network("192.0.0.0/24"), ipaddress.ip_network("192.88.99.0/24"))
_TRANSITION_NETWORKS = (ipaddress.ip_network("64:ff9b::/96"), ipaddress.ip_network("64:ff9b:1::/48"))


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": code, "message": message})


def parse_public_url(value: str) -> httpx.URL:
    """Validate syntax without DNS or I/O; also used by the request schema."""
    if len(value) > MAX_URL_CHARS or any(ord(c) <= 32 or ord(c) == 127 for c in value) or "\\" in value:
        raise ValueError("Remote input URL is malformed or exceeds 4096 characters.")
    try:
        url = httpx.URL(value)
    except (httpx.InvalidURL, ValueError) as exc:
        raise ValueError("Remote input URL is malformed.") from exc
    if url.scheme not in {"http", "https"} or not url.host or url.userinfo:
        raise ValueError("Remote inputs require an HTTP(S) URL without credentials.")
    if url.port is not None and url.port != {"http": 80, "https": 443}[url.scheme]:
        raise ValueError("Remote inputs allow only standard HTTP(S) ports.")
    if "%" in url.host:
        raise ValueError("Remote input hosts cannot contain zone identifiers or escapes.")
    return url.copy_with(fragment=None)


def _public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    if not ip.is_global or ip.is_reserved or ip.is_multicast or any(ip in n for n in _PROTOCOL_NETWORKS):
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.is_site_local:
            return False
        if ip.ipv4_mapped is not None:
            return _public_ip(str(ip.ipv4_mapped))
        if ip.sixtofour is not None or ip.teredo is not None or any(ip in n for n in _TRANSITION_NETWORKS):
            return False
    return True


@dataclass(frozen=True)
class ResolvedURL:
    url: httpx.URL
    addresses: tuple[str, ...]


async def resolve_public_url(value: str) -> ResolvedURL:
    try:
        url = parse_public_url(value)
    except ValueError as exc:
        raise _error(422, "responses_remote_input_blocked", str(exc)) from exc
    try:
        ipaddress.ip_address(url.host)
    except ValueError:
        # Absolute DNS names avoid container search-domain expansion.
        if "." not in url.host.rstrip("."):
            raise _error(422, "responses_remote_input_blocked", "Remote input host is not public.") from None
        host = url.host.rstrip(".") + "."
    else:
        host = url.host
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, url.port or (443 if url.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except OSError as exc:
        raise _error(502, "responses_remote_fetch_failed", "Remote input host could not be resolved.") from exc
    addresses = tuple(dict.fromkeys(info[4][0] for info in infos))
    if not addresses or not all(_public_ip(ip) for ip in addresses):
        raise _error(422, "responses_remote_input_blocked", "Remote inputs must resolve exclusively to public addresses.")
    return ResolvedURL(url, addresses)


@dataclass(frozen=True)
class PublicAsset:
    data: bytes
    media_type: str
    source_url: str


async def _send(transport: httpx.AsyncBaseTransport, resolved: ResolvedURL) -> httpx.Response:
    for address in resolved.addresses:
        request = httpx.Request(
            "GET", resolved.url.copy_with(host=address),
            headers={
                "Host": resolved.url.netloc.decode("ascii"),
                "Accept-Encoding": "identity", "User-Agent": "Audrey-file-input/1",
            },
            extensions={
                "sni_hostname": resolved.url.raw_host.decode("ascii"),
                "timeout": {"connect": 5, "read": 10, "write": 10, "pool": 10},
            },
        )
        try:
            # The transport has no cookie jar or automatic redirects. Bypassing
            # AsyncClient also avoids its INFO log of signed query parameters.
            return await transport.handle_async_request(request)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            continue
    raise _error(502, "responses_remote_fetch_failed", "Remote input connection failed.")


async def fetch_public_asset(
    resolved: ResolvedURL, *, max_bytes: int,
    transport: httpx.AsyncBaseTransport | None = None,
) -> PublicAsset:
    if max_bytes <= 0:
        raise _error(413, "responses_remote_input_too_large", "Remote input byte budget is exhausted.")
    try:
        outbound = transport if transport is not None else httpx.AsyncHTTPTransport(
            trust_env=False, verify=True,
            # IP origins can serve several names; each hop needs its own TLS SNI check.
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        async with asyncio.timeout(FETCH_DEADLINE_SECONDS), outbound:
            current = resolved
            for hop in range(MAX_REDIRECTS + 1):
                response = await _send(outbound, current)
                try:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location or hop == MAX_REDIRECTS:
                            raise _error(502, "responses_remote_fetch_failed", "Remote input redirect limit or missing location.")
                        target = str(current.url.join(location))
                        next_url = parse_public_url(target)
                        if current.url.scheme == "https" and next_url.scheme != "https":
                            raise _error(422, "responses_remote_input_blocked", "HTTPS input redirects cannot downgrade to HTTP.")
                        current = await resolve_public_url(target)
                        continue
                    if response.status_code != 200:
                        raise _error(502, "responses_remote_fetch_failed", f"Remote input returned HTTP {response.status_code}.")
                    if response.headers.get("content-encoding", "identity").strip().lower() not in {"", "identity"}:
                        raise _error(422, "responses_remote_input_blocked", "Compressed HTTP input bodies are not supported.")
                    declared = response.headers.get("content-length")
                    if declared is not None:
                        try:
                            length = int(declared)
                        except ValueError as exc:
                            raise _error(502, "responses_remote_fetch_failed", "Remote input has an invalid Content-Length.") from exc
                        if length < 0:
                            raise _error(502, "responses_remote_fetch_failed", "Remote input has an invalid Content-Length.")
                        if length > max_bytes:
                            raise _error(413, "responses_remote_input_too_large", "Remote input exceeds the byte limit.")
                    chunks = []
                    size = 0
                    async for chunk in response.aiter_raw():
                        size += len(chunk)
                        if size > max_bytes:
                            raise _error(413, "responses_remote_input_too_large", "Remote input exceeds the byte limit.")
                        chunks.append(chunk)
                    if not size:
                        raise _error(422, "responses_remote_input_invalid", "Remote input is empty.")
                    return PublicAsset(
                        b"".join(chunks), response.headers.get("content-type", "").split(";", 1)[0].strip().lower(),
                        # Signed query parameters must not enter prompts or archives.
                        str(current.url.copy_with(query=None, fragment=None)),
                    )
                finally:
                    await response.aclose()
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise _error(504, "responses_remote_input_timeout", "Remote input fetch timed out.") from exc
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
        raise _error(502, "responses_remote_fetch_failed", "Remote input could not be fetched safely.") from exc
    raise _error(502, "responses_remote_fetch_failed", "Remote input redirect limit exceeded.")
