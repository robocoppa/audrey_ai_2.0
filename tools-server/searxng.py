"""SearXNG meta-search client — the self-hosted, keyless `web_search` fallback.

Brave is the primary provider; when its key is quota-exhausted (402) or
rate-limited (429), `web_search` falls back here. SearXNG is a self-hosted
meta-search engine (aggregates Google/Bing/DDG/etc.) exposing a clean JSON API
(`/search?q=…&format=json`) — no API key, no per-query cost, and no bot-blocking
(it runs on your own infra), which is why it replaced the abandoned
DuckDuckGo HTML scrape (that returned a 202 "anomaly" page under panel load).

Configured via `SEARXNG_URL` (e.g. http://192.168.1.11:8088). Returns the same
`SearchResult` shape as Brave so the handler stays provider-agnostic.

**Resilience (mirrors `brave.py`):** while Brave is 402'd, SearXNG is the ONLY
live provider, and the research panel fires a burst of searches (3 workers ×
several ReAct rounds each) at the single instance. Without retry/cache, a
transient timeout or 429 under that burst just errors → the worker loses that
evidence → a thin ledger → an over-timid degraded answer. So this client:
- **retries** transient failures (timeout / 429 / 5xx) with bounded backoff —
  NOT a clean 0-results response (that's a real answer, not a failure), and NOT a
  JSON-decode error (retrying won't help).
- **caches** by (query, count) so identical queries across workers in the same
  run don't each re-hit SearXNG. Short TTL — web results for current topics go
  stale, but the win is intra-run dedup, not long-term caching.

Note: the SearXNG instance must have the JSON format enabled in its
`settings.yml` (`search.formats: [html, json]`) — it's off by default.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict

import httpx
from brave import SearchResult
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class SearxngError(Exception):
    """Raised when the SearXNG fallback itself fails (so the handler can 503)."""


class SearxngClient:
    """Async SearXNG JSON-API client (the web_search fallback) with retry+cache."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        cache_ttl_seconds: int = 900,
        cache_max_entries: int = 512,
    ) -> None:
        if not base_url:
            raise ValueError("SEARXNG_URL is empty; set it to enable the fallback.")
        self._base_url = base_url.rstrip("/")
        self._cache_ttl = cache_ttl_seconds
        self._cache_max = cache_max_entries
        self._cache: OrderedDict[tuple[str, int], tuple[float, list[SearchResult]]] = OrderedDict()
        self._cache_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"Accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        """Run a SearXNG search. Returns up to `count` results. Best-effort."""
        key = (query.strip(), max(1, min(count, 20)))
        cached = await self._cache_get(key)
        if cached is not None:
            return cached
        results = await self._fetch(query=key[0], count=key[1])
        await self._cache_put(key, results)
        return results

    async def _fetch(self, query: str, count: int) -> list[SearchResult]:
        async def _do() -> dict:
            r = await self._client.get(
                f"{self._base_url}/search",
                params={"q": query, "format": "json", "safesearch": 1},
            )
            # 429/5xx are transient under burst load — let the retry catch them.
            r.raise_for_status()
            return r.json()

        # Retry transient transport/HTTP errors (timeout, 429, 5xx). A JSON-decode
        # failure (ValueError) is NOT retried — a malformed body won't fix itself.
        # `reraise=True` → the last attempt's exception escapes for the except below.
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(
                    (httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError)
                ),
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                reraise=True,
            ):
                with attempt:
                    data = await _do()
        except httpx.HTTPError as e:
            raise SearxngError(f"SearXNG fallback failed: {e}") from e
        except ValueError as e:  # JSON decode
            raise SearxngError(f"SearXNG returned non-JSON: {e}") from e
        return _parse_results(data, count)

    async def _cache_get(self, key: tuple[str, int]) -> list[SearchResult] | None:
        async with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expires_at, results = entry
            if time.monotonic() >= expires_at:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return results

    async def _cache_put(self, key: tuple[str, int], results: list[SearchResult]) -> None:
        async with self._cache_lock:
            self._cache[key] = (time.monotonic() + self._cache_ttl, results)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)


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
