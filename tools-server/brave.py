"""Brave Search API client.

Shared between the custom-tools `web_search` endpoint and Audrey's
pre-generation search path. Stateless — constructed once at app startup.

Docs: https://api.search.brave.com/app/documentation/web-search/
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


@dataclass(slots=True, frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class BraveRateLimitError(Exception):
    """Raised when Brave returns 429 after retries are exhausted."""


class BraveQuotaError(Exception):
    """Raised when Brave returns 402 Payment Required (the monthly quota is
    exhausted). Distinct from a transient 429: retrying won't help, so the
    handler falls straight back to the keyless DuckDuckGo provider."""


class BraveUpstreamError(Exception):
    """Raised when a non-429 Brave HTTP error survives the retry budget.

    Without this the raw `httpx.HTTPStatusError` escaped `search()` and the
    `web_search` handler — which only catches `BraveRateLimitError`/`ValueError`
    — let it fall through to FastAPI as a generic 500. Normalizing it here lets
    the handler map it to a controlled 503 like the rate-limit path.
    """


class BraveClient:
    """Async Brave Search client with a small in-memory TTL cache.

    Cache keys: (query, count). One entry per unique (query, count) pair.
    Default TTL = 24h, matching our config to stretch the free tier.
    """

    def __init__(
        self,
        api_key: str,
        *,
        cache_ttl_seconds: int = 86_400,
        cache_max_entries: int = 512,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not api_key:
            raise ValueError("BRAVE_API_KEY is empty; set it in the environment.")
        self._api_key = api_key
        self._cache_ttl = cache_ttl_seconds
        self._cache_max = cache_max_entries
        self._cache: OrderedDict[tuple[str, int], tuple[float, list[SearchResult]]] = OrderedDict()
        self._cache_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        """Run a web search. Returns up to `count` results."""
        key = (query.strip(), max(1, min(count, 20)))
        cached = await self._cache_get(key)
        if cached is not None:
            return cached

        results = await self._fetch(query=key[0], count=key[1])
        await self._cache_put(key, results)
        return results

    async def _fetch(self, query: str, count: int) -> list[SearchResult]:
        async def _do() -> httpx.Response:
            r = await self._client.get(
                BRAVE_ENDPOINT,
                params={"q": query, "count": count, "safesearch": "moderate"},
            )
            if r.status_code == 402:
                # Quota exhausted — not transient, don't retry; signal fallback.
                raise BraveQuotaError("Brave returned 402 Payment Required (quota exhausted)")
            if r.status_code == 429:
                raise BraveRateLimitError("Brave returned 429 rate-limit")
            r.raise_for_status()
            return r

        # `reraise=True` means the *last* attempt's exception escapes (not a
        # RetryError). A 429 stays a `BraveRateLimitError` (handler → 503); a
        # non-429 `HTTPStatusError` would otherwise escape raw and become a
        # generic 500, so normalize it to `BraveUpstreamError` (handler → 503).
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type((BraveRateLimitError, httpx.HTTPStatusError)),
                stop=stop_after_attempt(4),
                wait=wait_exponential(multiplier=1, min=1, max=15),
                reraise=True,
            ):
                with attempt:
                    resp = await _do()
        except httpx.HTTPStatusError as e:
            raise BraveUpstreamError(f"Brave upstream error after retries: {e}") from e

        data: dict[str, Any] = resp.json()
        web_results = data.get("web", {}).get("results", []) or []
        return [
            SearchResult(
                title=item.get("title", "") or "",
                url=item.get("url", "") or "",
                snippet=item.get("description", "") or "",
            )
            for item in web_results[:count]
        ]

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
