"""Hermetic tests for the SearXNG fallback search provider.

`web_search` falls back to SearXNG (self-hosted meta-search JSON API) when Brave
returns 402 (quota) or 429 (rate-limit). We test the pure JSON parser and the
402→BraveQuotaError trigger without hitting the network.

(SearXNG replaced an abandoned DuckDuckGo HTML-scrape fallback, which bot-blocked
under the research panel's query volume — HTTP 202 "anomaly" page → 0 results.)
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

from searxng import SearxngClient, _parse_results  # noqa: E402

_RESPONSE = {
    "results": [
        {"title": "Async Rust Book", "url": "https://doc.rust-lang.org/book/async.html",
         "content": "Running async code in Rust usually happens concurrently."},
        {"title": "Tokio", "url": "https://tokio.rs/", "content": "An asynchronous runtime for Rust."},
        {"title": "No URL item", "url": "", "content": "should be skipped"},
    ]
}


class TestParseResults:
    def test_parses_title_url_snippet(self):
        out = _parse_results(_RESPONSE, count=5)
        # The empty-url item is skipped → 2 usable results.
        assert len(out) == 2
        assert out[0].title == "Async Rust Book"
        assert out[0].url == "https://doc.rust-lang.org/book/async.html"
        assert "concurrently" in out[0].snippet
        assert out[1].url == "https://tokio.rs/"

    def test_respects_count(self):
        assert len(_parse_results(_RESPONSE, count=1)) == 1

    def test_empty_results_yields_nothing(self):
        assert _parse_results({"results": []}, count=5) == []

    def test_missing_results_key_is_safe(self):
        assert _parse_results({}, count=5) == []


class TestClientConstruction:
    def test_empty_url_raises(self):
        import pytest
        with pytest.raises(ValueError):
            SearxngClient("")

    def test_strips_trailing_slash(self):
        c = SearxngClient("http://searx.local:8088/")
        assert c._base_url == "http://searx.local:8088"


class TestRetryAndCache:
    """Mirrors brave.py's resilience — retry transient failures, cache by query."""

    def _client_with(self, handler):
        import httpx
        c = SearxngClient("http://searx.local:8088")
        c._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return c

    def test_retries_transient_5xx_then_succeeds(self):
        import asyncio

        import httpx
        calls = {"n": 0}

        def handler(_r):
            calls["n"] += 1
            if calls["n"] < 2:
                return httpx.Response(503, text="busy")  # transient → retried
            return httpx.Response(200, json=_RESPONSE)

        c = self._client_with(handler)

        async def _run():
            try:
                return await c.search("q")
            finally:
                await c.aclose()

        out = asyncio.run(_run())
        assert calls["n"] == 2          # one retry happened
        assert len(out) == 2            # then it parsed the good response

    def test_retries_exhausted_raises_searxng_error(self):
        import asyncio

        import httpx
        from searxng import SearxngError

        def handler(_r):
            return httpx.Response(503, text="down")  # always transient-fails

        c = self._client_with(handler)

        async def _run():
            try:
                await c.search("q")
            finally:
                await c.aclose()

        try:
            asyncio.run(_run())
            raise AssertionError("expected SearxngError")
        except SearxngError:
            pass

    def test_cache_dedups_identical_query(self):
        import asyncio

        import httpx
        calls = {"n": 0}

        def handler(_r):
            calls["n"] += 1
            return httpx.Response(200, json=_RESPONSE)

        c = self._client_with(handler)

        async def _run():
            try:
                await c.search("same query")
                await c.search("same query")  # served from cache
            finally:
                await c.aclose()

        asyncio.run(_run())
        assert calls["n"] == 1  # second call did not hit the network


class TestBraveQuotaTriggersFallback:
    """A 402 from Brave must raise BraveQuotaError so the handler falls back."""

    def test_402_raises_quota_error(self):
        import asyncio

        import httpx
        from brave import BraveClient, BraveQuotaError

        client = BraveClient(api_key="test-key")
        client._client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(402, json={})),
            headers={"X-Subscription-Token": "test-key"},
        )

        async def _run():
            try:
                await client.search("q")
            finally:
                await client.aclose()

        try:
            asyncio.run(_run())
            raise AssertionError("expected BraveQuotaError")
        except BraveQuotaError:
            pass
