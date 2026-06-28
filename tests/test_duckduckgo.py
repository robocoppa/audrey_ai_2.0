"""Hermetic tests for the DuckDuckGo fallback search provider.

`web_search` falls back to DuckDuckGo (keyless HTML scrape) when Brave returns
402 (quota) or 429 (rate-limit). The fragile part is parsing DDG's HTML results
page with regexes + unwrapping its /l/?uddg= redirect links — both pure, so we
test them against a representative page fragment without hitting the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

from duckduckgo import _parse_results, _unwrap_url  # noqa: E402

# A trimmed but realistic DDG /html/ results fragment (two results).
_PAGE = """
<div class="result results_links results_links_deep web-result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdoc.rust-lang.org%2Fbook%2Fasync.html&amp;rut=abc">
    Async <b>Rust</b> Book
  </a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">
    Running <b>async</b> code in Rust usually happens concurrently.
  </a>
</div>
<div class="result results_links results_links_deep web-result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ftokio.rs%2F&amp;rut=def">
    Tokio
  </a>
  <a class="result__snippet">An asynchronous runtime for Rust.</a>
</div>
"""


class TestUnwrapUrl:
    def test_unwraps_uddg_redirect(self):
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Ftokio.rs%2F&rut=def"
        assert _unwrap_url(href) == "https://tokio.rs/"

    def test_passes_through_plain_url(self):
        assert _unwrap_url("https://example.com/page") == "https://example.com/page"

    def test_adds_scheme_to_protocol_relative(self):
        # A protocol-relative non-redirect URL gets https:.
        assert _unwrap_url("//example.com/x").startswith("https://")


class TestParseResults:
    def test_parses_titles_urls_snippets(self):
        out = _parse_results(_PAGE, count=5)
        assert len(out) == 2
        assert out[0].title == "Async Rust Book"           # tags stripped
        assert out[0].url == "https://doc.rust-lang.org/book/async.html"  # unwrapped
        assert "concurrently" in out[0].snippet
        assert out[1].title == "Tokio"
        assert out[1].url == "https://tokio.rs/"

    def test_respects_count(self):
        assert len(_parse_results(_PAGE, count=1)) == 1

    def test_empty_page_yields_no_results(self):
        assert _parse_results("<html><body>no results</body></html>", count=5) == []

    def test_missing_snippet_is_empty_not_error(self):
        page = ('<a class="result__a" href="https://e.com">Title</a>')
        out = _parse_results(page, count=5)
        assert len(out) == 1
        assert out[0].snippet == ""


class TestBraveQuotaError:
    """A 402 from Brave must raise BraveQuotaError (→ DDG fallback), not the
    generic upstream error (→ 503 with no fallback)."""

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
