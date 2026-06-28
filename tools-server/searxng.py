"""SearXNG meta-search client — the self-hosted, keyless `web_search` fallback.

Brave is the primary provider; when its key is quota-exhausted (402) or
rate-limited (429), `web_search` falls back here. SearXNG is a self-hosted
meta-search engine (aggregates Google/Bing/DDG/etc.) exposing a clean JSON API
(`/search?q=…&format=json`) — no API key, no per-query cost, and no bot-blocking
(it runs on your own infra), which is why it replaced the abandoned
DuckDuckGo HTML scrape (that returned a 202 "anomaly" page under panel load).

Configured via `SEARXNG_URL` (e.g. http://192.168.1.11:8088). Returns the same
`SearchResult` shape as Brave so the handler stays provider-agnostic.

Note: the SearXNG instance must have the JSON format enabled in its
`settings.yml` (`search.formats: [html, json]`) — it's off by default.
"""

from __future__ import annotations

import httpx
from brave import SearchResult


class SearxngError(Exception):
    """Raised when the SearXNG fallback itself fails (so the handler can 503)."""


class SearxngClient:
    """Async SearXNG JSON-API client (the web_search fallback)."""

    def __init__(self, base_url: str, *, timeout_seconds: float = 10.0) -> None:
        if not base_url:
            raise ValueError("SEARXNG_URL is empty; set it to enable the fallback.")
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"Accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        """Run a SearXNG search. Returns up to `count` results. Best-effort."""
        count = max(1, min(count, 20))
        try:
            r = await self._client.get(
                f"{self._base_url}/search",
                params={"q": query.strip(), "format": "json", "safesearch": 1},
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            raise SearxngError(f"SearXNG fallback failed: {e}") from e
        except ValueError as e:  # JSON decode
            raise SearxngError(f"SearXNG returned non-JSON: {e}") from e
        return _parse_results(data, count)


def _parse_results(data: dict, count: int) -> list[SearchResult]:
    """Parse a SearXNG JSON response into SearchResults (pure; testable).

    SearXNG result items carry `title`, `url`, and `content` (the snippet).
    """
    results = data.get("results", []) or []
    out: list[SearchResult] = []
    for item in results[:count]:
        url = item.get("url", "") or ""
        if not url:
            continue
        out.append(
            SearchResult(
                title=item.get("title", "") or "",
                url=url,
                snippet=item.get("content", "") or "",
            )
        )
    return out
