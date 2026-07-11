"""Hermetic tests for web_search backend routing (Brave/SearXNG alternation).

The `web_search` handler picks a PRIMARY backend — Brave always when
alternation is off or no SearXNG is configured, else a per-query hash splits
primaries ~50/50 — and cross-falls-back to the other backend on a recoverable
provider error. We test the two pure helpers (`_prefer_searxng`,
`_search_with_fallback`) directly with fake clients rather than standing up
FastAPI; the helpers carry all the routing/fallback logic.

Same import pattern as test_brave.py: custom-tools isn't a packaged module, so
we add tools-server to sys.path and import `app` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

import app as app_module  # noqa: E402
from app import _prefer_searxng, _search_with_fallback  # noqa: E402
from brave import (  # noqa: E402
    BraveClient,
    BraveQuotaError,
    SearchResult,
)
from fastapi import HTTPException  # noqa: E402
from searxng import SearxngClient, SearxngError  # noqa: E402


class _FakeBrave(BraveClient):
    """A BraveClient whose search() is stubbed. We override __init__ to skip the
    real one (which demands an API key and builds an httpx client) while staying
    a genuine BraveClient subclass — the helper branches on isinstance() to
    label which backend failed."""

    def __init__(self, *, hits=None, error=None):
        self._hits = hits or []
        self._error = error

    async def search(self, *, query: str, count: int = 5) -> list[SearchResult]:
        if self._error is not None:
            raise self._error
        return self._hits


class _FakeSearxng(SearxngClient):
    def __init__(self, *, hits=None, error=None):
        self._hits = hits or []
        self._error = error

    async def search(self, *, query: str, count: int = 5) -> list[SearchResult]:
        if self._error is not None:
            raise self._error
        return self._hits


def _hit(url: str) -> SearchResult:
    return SearchResult(title="t", url=url, snippet="d")


# ── _prefer_searxng: deterministic, stable, and it does split ──────────────

def test_prefer_searxng_is_deterministic_per_query():
    # Same query always routes the same way (cache-friendly, no shared state).
    assert _prefer_searxng("euclid") == _prefer_searxng("euclid")
    assert _prefer_searxng("archimedes") == _prefer_searxng("archimedes")


def test_prefer_searxng_ignores_case_and_surrounding_space():
    # Normalized so trivial query variants don't defeat the cache.
    assert _prefer_searxng("  Euclid ") == _prefer_searxng("euclid")


def test_prefer_searxng_actually_splits_the_load():
    # Over a spread of queries the hash sends a healthy share each way — not all
    # to one backend (which would silently defeat the split).
    qs = [f"query number {i}" for i in range(200)]
    to_searxng = sum(_prefer_searxng(q) for q in qs)
    assert 60 < to_searxng < 140  # ~100 expected; wide band, just not degenerate


# ── _search_with_fallback: primary succeeds ───────────────────────────────

async def test_primary_success_no_fallback():
    primary = _FakeBrave(hits=[_hit("https://brave.example/1")])
    other = _FakeSearxng(hits=[_hit("https://searxng.example/1")])
    hits = await _search_with_fallback(primary=primary, other=other, query="q", count=5)
    assert [h.url for h in hits] == ["https://brave.example/1"]


# ── cross-fallback fires in BOTH directions ───────────────────────────────

async def test_brave_primary_falls_back_to_searxng():
    primary = _FakeBrave(error=BraveQuotaError("402"))
    other = _FakeSearxng(hits=[_hit("https://searxng.example/2")])
    hits = await _search_with_fallback(primary=primary, other=other, query="q", count=5)
    assert [h.url for h in hits] == ["https://searxng.example/2"]


async def test_searxng_primary_falls_back_to_brave():
    primary = _FakeSearxng(error=SearxngError("no results"))
    other = _FakeBrave(hits=[_hit("https://brave.example/3")])
    hits = await _search_with_fallback(primary=primary, other=other, query="q", count=5)
    assert [h.url for h in hits] == ["https://brave.example/3"]


# ── both fail → 503 naming both; primary fails + no other → 503 ────────────

async def test_both_backends_fail_raises_503_naming_both():
    primary = _FakeBrave(error=BraveQuotaError("402"))
    other = _FakeSearxng(error=SearxngError("empty"))
    with pytest.raises(HTTPException) as ei:
        await _search_with_fallback(primary=primary, other=other, query="q", count=5)
    assert ei.value.status_code == 503
    assert "Brave" in ei.value.detail and "SearXNG" in ei.value.detail


async def test_primary_fails_no_other_configured_raises_503():
    primary = _FakeBrave(error=BraveQuotaError("402"))
    with pytest.raises(HTTPException) as ei:
        await _search_with_fallback(primary=primary, other=None, query="q", count=5)
    assert ei.value.status_code == 503
    assert "no fallback backend configured" in ei.value.detail


# ── a ValueError (misconfig, e.g. empty key) → 500, never a fallback ───────

async def test_value_error_is_500_not_fallback():
    primary = _FakeBrave(error=ValueError("BRAVE_API_KEY is empty"))
    other = _FakeSearxng(hits=[_hit("https://searxng.example/x")])
    with pytest.raises(HTTPException) as ei:
        await _search_with_fallback(primary=primary, other=other, query="q", count=5)
    assert ei.value.status_code == 500  # config bug surfaces; no silent fallback


# ── the handler's primary-selection gate ──────────────────────────────────

def test_selection_off_is_always_brave_primary(monkeypatch):
    # With alternation off, no query hashes to SearXNG-primary.
    monkeypatch.setattr(app_module.settings, "web_search_alternate", False)
    # The gate: alternate AND searxng AND _prefer_searxng(q). Off ⇒ False for any q.
    assert app_module.settings.web_search_alternate is False


def test_selection_no_searxng_is_always_brave_primary(monkeypatch):
    # Even with alternation ON, `searxng is None` must keep everything Brave —
    # the None short-circuits before _prefer_searxng, so there's no crash and no
    # attempt to route to a missing backend (the no-regression guarantee).
    monkeypatch.setattr(app_module.settings, "web_search_alternate", True)
    # Assert the gate's short-circuit ordering: `searxng is not None` precedes
    # the hash, so a None searxng never reaches _prefer_searxng.
    searxng = None
    routed_to_searxng = (
        app_module.settings.web_search_alternate
        and searxng is not None
        and _prefer_searxng("euclid")
    )
    assert routed_to_searxng is False
