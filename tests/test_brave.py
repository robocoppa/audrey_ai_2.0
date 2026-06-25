"""Hermetic tests for the Brave Search client's error normalization.

The custom-tools `web_search` handler catches `BraveRateLimitError` and
`ValueError`. Before the fix, a non-429 `httpx.HTTPStatusError` that survived
the retry budget escaped `search()` raw and fell through to FastAPI as a
generic 500. `search()` now normalizes that case to `BraveUpstreamError`, which
the handler maps to a controlled 503 (same client-facing shape as the 429 path).

We don't hit the network: `BraveClient` builds its own `httpx.AsyncClient`, so
we swap that client for one backed by `httpx.MockTransport`. The retry's
exponential backoff (min 1s × 4 attempts) is neutralized by stubbing
`asyncio.sleep` so the suite stays sub-second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

# Add tools-server to sys.path so we can import `brave` directly. The
# custom-tools service isn't packaged for installation; it runs as a script in
# its own container. (Same pattern as test_chat_archive.py.)
_TOOLS_SERVER = Path(__file__).resolve().parent.parent / "tools-server"
if str(_TOOLS_SERVER) not in sys.path:
    sys.path.insert(0, str(_TOOLS_SERVER))

from brave import BraveClient, BraveRateLimitError, BraveUpstreamError  # noqa: E402


def _client_returning(status_code: int) -> BraveClient:
    """A BraveClient whose underlying httpx client always returns `status_code`."""
    client = BraveClient(api_key="test-key")

    def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "boom"})

    # Swap the internally-constructed client for a mock-transport one. Keep the
    # subscription header so the request shape is unchanged.
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler),
        headers={"X-Subscription-Token": "test-key"},
    )
    return client


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Neutralize tenacity's exponential backoff so retries don't sleep."""
    async def _instant_sleep(_seconds):
        return None

    monkeypatch.setattr("asyncio.sleep", _instant_sleep)


@pytest.mark.asyncio
async def test_exhausted_non_429_normalizes_to_upstream_error():
    # A persistent 500 is retryable (HTTPStatusError is in the retry set) but
    # survives all attempts. It must surface as BraveUpstreamError — NOT a raw
    # httpx.HTTPStatusError, which the web_search handler wouldn't catch.
    client = _client_returning(500)
    try:
        with pytest.raises(BraveUpstreamError):
            await client.search(query="anything", count=5)
        # And specifically not the raw upstream exception type.
        # (pytest.raises above already asserts the type; this documents intent.)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_exhausted_429_stays_rate_limit_error():
    # The 429 path is unchanged: it stays a BraveRateLimitError (handler → 503).
    client = _client_returning(429)
    try:
        with pytest.raises(BraveRateLimitError):
            await client.search(query="anything", count=5)
    finally:
        await client.aclose()
