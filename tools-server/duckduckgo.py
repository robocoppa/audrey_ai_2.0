"""DuckDuckGo HTML search — the free, keyless fallback for `web_search`.

Brave is the primary provider (better results, structured JSON), but its key
has a hard quota; when Brave returns 402 (Payment Required) or is rate-limited,
`web_search` falls back here so research grounding stays alive instead of
returning nothing.

DuckDuckGo has no official JSON search API. The `html.duckduckgo.com/html/`
endpoint returns a simple results page we parse with focused regexes — no
HTML-parser dependency (keeps tools-server on httpx + tenacity only). This is a
best-effort fallback: lower quality and more fragile than Brave (the page markup
can change), so it is deliberately NOT the primary path. Returns the same
`SearchResult` shape as Brave so the handler is provider-agnostic.
"""

from __future__ import annotations

import html
import re
import urllib.parse

import httpx
from brave import SearchResult

DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"

# Each result on the HTML page is an <a class="result__a" href="...">title</a>
# followed by an <a class="result__snippet">snippet</a>. DDG wraps the real URL
# in a redirect (/l/?uddg=<encoded>), which we unwrap.
_RESULT_A = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL,
)
_SNIPPET = re.compile(
    r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")


class DuckDuckGoError(Exception):
    """Raised when the DDG fallback itself fails (so the handler can 503)."""


def _clean(text: str) -> str:
    """Strip HTML tags + unescape entities from a fragment."""
    return html.unescape(_TAG.sub("", text)).strip()


def _unwrap_url(href: str) -> str:
    """DDG result links are /l/?uddg=<encoded-real-url> redirects — unwrap them."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if parsed.path.endswith("/l/") or "uddg=" in (parsed.query or ""):
        qs = urllib.parse.parse_qs(parsed.query)
        target = qs.get("uddg", [""])[0]
        if target:
            return urllib.parse.unquote(target)
    return href


class DuckDuckGoClient:
    """Keyless DDG HTML-scrape search client (the web_search fallback)."""

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={
                # DDG's HTML endpoint expects a browser-ish UA; a bare client
                # gets an empty page.
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                ),
                "Accept": "text/html",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        """Run a DDG HTML search. Returns up to `count` results. Best-effort."""
        count = max(1, min(count, 20))
        try:
            r = await self._client.post(DDG_HTML_ENDPOINT, data={"q": query.strip()})
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise DuckDuckGoError(f"DuckDuckGo fallback failed: {e}") from e
        return _parse_results(r.text, count)


def _parse_results(page: str, count: int) -> list[SearchResult]:
    """Parse the DDG HTML results page into SearchResults (pure; testable)."""
    titles = list(_RESULT_A.finditer(page))
    snippets = list(_SNIPPET.finditer(page))
    out: list[SearchResult] = []
    for i, m in enumerate(titles[:count]):
        snippet = _clean(snippets[i].group("snippet")) if i < len(snippets) else ""
        out.append(
            SearchResult(
                title=_clean(m.group("title")),
                url=_unwrap_url(m.group("href")),
                snippet=snippet,
            )
        )
    return out
