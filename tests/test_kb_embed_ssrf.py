"""Tests for `_validate_image_url` + `_is_unsafe_address` (Phase 27 SSRF guards).

Promotes the 13 ad-hoc cases from the Phase 27 verification log
into committed tests. Stubs `socket.getaddrinfo` so tests don't hit
real DNS — tests are deterministic and run offline.

The guard rejects: non-https schemes, missing hosts, hosts that resolve
to private/loopback/link-local/multicast/reserved/unspecified IPs, and
hosts that fail to resolve at all (fail-closed). Accepts plain public
IPs.
"""

import socket

import pytest

from audrey.kb import embed


def _stub_getaddrinfo(returned_ip: str):
    """Build a fake getaddrinfo that always returns one address.

    Matches the shape `getaddrinfo` returns: a list of 5-tuples whose
    last element is `(addr, port, ...)`. Only `addr` matters here.
    """
    def fake(host: str, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (returned_ip, 0))]
    return fake


# ─── _is_unsafe_address: reject ────────────────────────────────────────

@pytest.mark.parametrize("ip,why", [
    ("127.0.0.1",      "loopback v4"),
    ("::1",            "loopback v6"),
    ("10.0.0.1",       "RFC1918 10/8"),
    ("192.168.1.50",   "RFC1918 192.168/16"),
    ("172.20.0.5",     "Docker default subnet (RFC1918 172.16/12)"),
    ("169.254.169.254","link-local AWS metadata"),
    ("224.0.0.1",      "multicast"),
    ("0.0.0.0",        "unspecified"),  # noqa: S104 — test data, not a bind address
    ("240.0.0.1",      "reserved"),
])
def test_unsafe_address_rejects_private_ips(monkeypatch, ip, why):
    monkeypatch.setattr(embed.socket, "getaddrinfo", _stub_getaddrinfo(ip))
    assert embed._is_unsafe_address("anyhost.example") is True, why


def test_unsafe_address_rejects_unresolvable_host(monkeypatch):
    # Fail-closed: getaddrinfo raising should mean "unsafe", not "safe".
    # An attacker-controlled host might not resolve right now but could
    # later (DNS rebinding) — and a typo for an internal name shouldn't
    # silently pass either.
    def boom(*args, **kwargs):
        raise socket.gaierror("name does not resolve")
    monkeypatch.setattr(embed.socket, "getaddrinfo", boom)
    assert embed._is_unsafe_address("definitely-not-a-host.example") is True


def test_unsafe_address_accepts_public_ipv4(monkeypatch):
    # Public IP — Google's DNS, used here only as a known-public example.
    monkeypatch.setattr(embed.socket, "getaddrinfo", _stub_getaddrinfo("8.8.8.8"))
    assert embed._is_unsafe_address("dns.google") is False


def test_unsafe_address_accepts_public_ipv6(monkeypatch):
    # Cloudflare's public v6 resolver — the only thing that matters is
    # that it's outside any private/reserved range.
    monkeypatch.setattr(embed.socket, "getaddrinfo", _stub_getaddrinfo("2606:4700:4700::1111"))
    assert embed._is_unsafe_address("one.one.one.one") is False


# ─── _validate_image_url: scheme checks ────────────────────────────────

@pytest.mark.parametrize("url", [
    "http://example.com/cat.jpg",
    "ftp://example.com/cat.jpg",
    "file:///etc/passwd",
    "data:image/png;base64,AAAA",
    "javascript:alert(1)",
])
def test_validate_url_rejects_non_https_scheme(monkeypatch, url):
    # Must reject before DNS lookup. Stub getaddrinfo to return a public
    # IP so the host check would pass — only the scheme guard should
    # raise here.
    monkeypatch.setattr(embed.socket, "getaddrinfo", _stub_getaddrinfo("8.8.8.8"))
    with pytest.raises(ValueError, match="scheme"):
        embed._validate_image_url(url)


def test_validate_url_rejects_missing_host(monkeypatch):
    # Edge: `https:///path` parses cleanly but has no hostname.
    monkeypatch.setattr(embed.socket, "getaddrinfo", _stub_getaddrinfo("8.8.8.8"))
    with pytest.raises(ValueError, match="no host"):
        embed._validate_image_url("https:///path/to/image.png")


def test_validate_url_rejects_private_host(monkeypatch):
    # Docker-DNS hostname like `qdrant` would resolve to a 172.x address
    # on the host running audrey. Must be rejected.
    monkeypatch.setattr(embed.socket, "getaddrinfo", _stub_getaddrinfo("172.20.0.5"))
    with pytest.raises(ValueError, match=r"private|loopback|link-local"):
        embed._validate_image_url("https://qdrant:6333/collections")


def test_validate_url_rejects_loopback_host(monkeypatch):
    monkeypatch.setattr(embed.socket, "getaddrinfo", _stub_getaddrinfo("127.0.0.1"))
    with pytest.raises(ValueError):
        embed._validate_image_url("https://localhost:8000/metrics")


def test_validate_url_accepts_public_https_url(monkeypatch):
    # The happy path. No exception means accepted.
    monkeypatch.setattr(embed.socket, "getaddrinfo", _stub_getaddrinfo("199.232.46.193"))
    embed._validate_image_url("https://upload.wikimedia.org/wiki/cat.png")


def test_validate_url_includes_offending_host_in_message(monkeypatch):
    # Error message must name the host so debug logs surface what was rejected.
    monkeypatch.setattr(embed.socket, "getaddrinfo", _stub_getaddrinfo("10.0.0.5"))
    with pytest.raises(ValueError, match="badhost"):
        embed._validate_image_url("https://badhost.example/x.png")
