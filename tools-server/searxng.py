"""SearXNG JSON search client used by the sidecar's `web_search` tool.

SearXNG is a self-hosted search aggregator. With both providers configured,
`web_search` can select Brave or SearXNG by a stable query hash and try the
other provider after an error or an empty result. SearXNG can also serve alone.
Both clients return the same `SearchResult` shape.

Configured via `SEARXNG_URL`. The instance must enable JSON responses in its
`settings.yml` (`search.formats: [html, json]`); that format is off by default.

The client retries transport and HTTP failures with bounded backoff. It also
retries an empty result once: an upstream engine may be temporarily throttled,
so an empty response is not proof that no relevant source exists. Malformed
JSON is an error and is not retried.

Its short-lived cache is keyed by query because result count only slices the
response locally. A cached result can therefore serve different counts without
another upstream request. Empty results are never cached.
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

# One soft retry when a 200 comes back with zero results — see module docstring.
# Longer than the transport backoff: give the throttled upstream engines a beat
# to recover rather than re-hitting instantly.
_EMPTY_RETRY_WAIT_SECONDS = 1.5

# How many parsed results to keep per cache entry. `count` is never sent upstream
# (see `_fetch` — params are q/format/safesearch only), so parsing more costs
# nothing on the wire; it just lets ONE cached entry serve callers asking for
# different `count` values.
_MAX_KEEP = 20


class SearxngError(Exception):
    """Raised when SearXNG fails, allowing the handler to try another provider."""


class SearxngClient:
    """Async SearXNG JSON client with bounded retries and a result cache."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 10.0,
        cache_ttl_seconds: int = 900,
        cache_max_entries: int = 512,
    ) -> None:
        if not base_url:
            raise ValueError("SEARXNG_URL is empty; set it to enable the SearXNG provider.")
        self._base_url = base_url.rstrip("/")
        self._cache_ttl = cache_ttl_seconds
        self._cache_max = cache_max_entries
        self._cache: OrderedDict[str, tuple[float, list[SearchResult]]] = OrderedDict()
        self._cache_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"Accept": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        """Run a SearXNG search. Returns up to `count` results. Best-effort."""
        # Key on the QUERY ALONE, not (query, count). `count` is never sent
        # upstream here — it only slices the response client-side — so including
        # it in the key made two panel workers asking the SAME question with
        # different `count` values miss each other and fire two identical HTTP
        # requests at a rate-limited instance. Keep the full parsed list, slice
        # per caller. **brave.py must NOT copy this**: Brave does send `count`
        # upstream (`params={"q":…, "count": count}`), so its (query, count) key
        # is correct there and dropping it would serve short cached results to a
        # caller that asked for more.
        key = query.strip()
        want = max(1, min(count, _MAX_KEEP))
        cached = await self._cache_get(key)
        if cached is not None:
            return cached[:want]
        results = await self._fetch(query=key, count=_MAX_KEEP)
        # Never cache an empty result: a 200+0-results is usually a transient
        # upstream throttle, and caching it would starve every identical query
        # in this run for the whole TTL. Let the next caller re-fetch (and get
        # its own empty-retry shot).
        if results:
            await self._cache_put(key, results)
        return results[:want]

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
        async def _fetch_once() -> list[SearchResult]:
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
                raise SearxngError(f"SearXNG search failed: {e}") from e
            except ValueError as e:  # JSON decode
                raise SearxngError(f"SearXNG returned non-JSON: {e}") from e
            return _parse_results(data, count)

        results = await _fetch_once()
        # A clean 200 with zero results is usually a transient upstream throttle,
        # not "nothing exists" — give it exactly one more shot after a short wait.
        # (The transport retry above can't catch this: an empty 200 isn't an
        # error.) One retry only; re-hammering a throttled instance worsens it.
        if not results:
            await asyncio.sleep(_EMPTY_RETRY_WAIT_SECONDS)
            results = await _fetch_once()
        return results

    async def _cache_get(self, key: str) -> list[SearchResult] | None:
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

    async def _cache_put(self, key: str, results: list[SearchResult]) -> None:
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
