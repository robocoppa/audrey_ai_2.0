"""custom-tools FastAPI server.

Endpoints, OpenAPI auto-discovered by the Audrey orchestrator:
  POST /web_search             — Brave Search API
  POST /web_fetch              — open one URL, return its readable text (SSRF-guarded)
  POST /kb_search              — text query proxied to Audrey /v1/kb/query
  POST /kb_image_search        — image query proxied to Audrey /v1/kb/query/image
  POST /memory_store           — save (key, value, user:<id>-tagged) to Qdrant
  POST /memory_recall          — fetch by exact key
  POST /memory_search          — semantic search over a user's memories
  POST /chat_history_search    — semantic search over a user's prior conversations
  POST /list_my_files          — a user's own uploaded files, via Audrey
                                 /v1/files/list (service token)

Internal-only (hidden from /openapi.json so the model can't call them):
  POST /chat_history/archive   — write a turn to the chat archive
  POST /chat_history/prune     — apply retention policy
  GET  /chat_history/stats     — row counts for admin/debug

Memory and chat archive are both Qdrant-backed and embedded with
nomic-embed-text. On first startup, a legacy `memory.db` SQLite file
(from an earlier backend) is migrated automatically and renamed to
`memory.db.migrated`.

Each tool endpoint has a clear operation_id so the orchestrator's
OpenAPI → Ollama-tool converter produces sensible tool names.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
from brave import (
    BraveClient,
    BraveQuotaError,
    BraveRateLimitError,
    BraveUpstreamError,
    SearchResult,
)
from chat_archive import ChatArchiveStore
from db import MemoryEntry, MemoryStore
from fastapi import FastAPI, HTTPException, status
from fetch import FetchError, fetch_readable
from pydantic import BaseModel, Field
from searxng import SearxngClient, SearxngError
from settings import settings

log = logging.getLogger("custom-tools")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ─── Lifespan (startup / shutdown) ────────────────────────────────────

def _service_headers(token: str) -> dict[str, str]:
    return {"X-Audrey-Service-Token": token} if token else {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    brave = BraveClient(
        api_key=settings.brave_api_key,
        cache_ttl_seconds=settings.brave_cache_ttl_hours * 3600,
    )
    # Keyless fallback used when Brave is quota-exhausted (402) or rate-limited.
    # Optional: only built when SEARXNG_URL is set (else no fallback → 503).
    searxng = SearxngClient(settings.searxng_url) if settings.searxng_url else None
    memory = MemoryStore(
        qdrant_url=settings.qdrant_url,
        ollama_url=settings.ollama_url,
        collection=settings.memory_collection,
        embed_model=settings.memory_embed_model,
        embed_dim=settings.memory_embed_dim,
        similarity_threshold=settings.memory_similarity_threshold,
        embed_timeout_s=settings.ollama_embed_timeout_s,
        legacy_sqlite_path=settings.memory_db_path,
    )
    await memory.init()
    audrey = httpx.AsyncClient(
        base_url=settings.audrey_url,
        timeout=settings.audrey_kb_timeout_seconds,
        headers=_service_headers(settings.kb_service_token),
    )
    chat_archive = ChatArchiveStore(
        sqlite_path=settings.chat_archive_db_path,
        qdrant_url=settings.qdrant_url,
        ollama_url=settings.ollama_url,
        collection=settings.chat_archive_collection,
        embed_model=settings.memory_embed_model,
        embed_dim=settings.memory_embed_dim,
        embed_timeout_s=settings.ollama_embed_timeout_s,
        chunk_max_chars=settings.chat_archive_chunk_max_chars,
        chunk_overlap_chars=settings.chat_archive_chunk_overlap_chars,
        search_threshold=settings.chat_archive_search_threshold,
        retention_days=settings.chat_archive_retention_days,
        max_bytes=settings.chat_archive_max_bytes,
    )
    await chat_archive.init()

    app.state.brave = brave
    app.state.searxng = searxng
    app.state.memory = memory
    app.state.audrey = audrey
    app.state.chat_archive = chat_archive
    log.info(
        "custom-tools ready. brave=%s searxng=%s audrey=%s qdrant=%s memory=%s archive=%s",
        "configured" if settings.brave_api_key else "UNSET",
        settings.searxng_url or "UNSET",
        settings.audrey_url,
        settings.qdrant_url,
        settings.memory_collection,
        settings.chat_archive_collection,
    )
    try:
        yield
    finally:
        await brave.aclose()
        if searxng is not None:
            await searxng.aclose()
        await audrey.aclose()
        await memory.aclose()
        await chat_archive.aclose()


app = FastAPI(
    title="Audrey custom-tools",
    version="0.1.0",
    description=(
        "Audrey's v1 tool surface. Every route is auto-discovered by the "
        "orchestrator and exposed to models as a callable tool."
    ),
    lifespan=lifespan,
)


# ─── Schemas ──────────────────────────────────────────────────────────

class WebSearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=500, description="Search query text.")]
    count: Annotated[int, Field(ge=1, le=10, description="Max results to return.")] = 5


class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearchResponse(BaseModel):
    query: str
    results: list[WebSearchResult]


class WebFetchRequest(BaseModel):
    url: Annotated[str, Field(
        min_length=1, max_length=2000,
        description="Full http(s) URL of a page to open and read, e.g. one returned by web_search.",
    )]
    max_chars: Annotated[int, Field(
        ge=500, le=20000,
        description="Maximum characters of extracted text to return.",
    )] = 6000


class WebFetchResponse(BaseModel):
    url: str            # final URL after any redirects
    text: str


class KBSearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=1000)]
    top_k: Annotated[int, Field(ge=1, le=20)] = 5
    user: str | None = Field(
        default=None,
        description=(
            "Optional user id. If the user has uploaded files via Audrey's "
            "/v1/files endpoint, their private collection is searched alongside "
            "the global KB and results are merged by score."
        ),
        max_length=200,
    )
    filename: str | None = Field(
        default=None,
        description=(
            "Optional. Restrict the search to one of the user's own uploaded "
            "files, by its exact filename as returned by list_my_files. Use "
            "this only when the user pointed at a particular file — 'in the "
            "standup video, what did they say about X'. Omit it to search "
            "everything; a wrong filename here answers confidently from the "
            "wrong source."
        ),
        max_length=500,
    )
    artifact: str | None = Field(
        default=None,
        description=(
            "Optional, and only meaningful for a video. 'transcript' searches "
            "what was said, 'visual' searches what was on screen (slides, "
            "signs, on-screen text), 'summary' searches the one-paragraph "
            "summary. Omit to search all three."
        ),
        pattern="^(transcript|visual|summary)$",
    )


class KBImageSearchRequest(BaseModel):
    query: str | None = Field(
        default=None,
        description=(
            "Text description of the image you want to find (e.g. 'someone in "
            "guard position', 'sedimentary rock with visible layers'). Encoded "
            "via CLIP's text tower and matched against image embeddings."
        ),
        max_length=2000,
    )
    image_url: str | None = Field(
        default=None,
        description="HTTP(S) URL of a reference image to find visually-similar matches.",
    )
    image_b64: str | None = Field(
        default=None,
        description="Base64-encoded reference image bytes.",
    )
    top_k: Annotated[int, Field(ge=1, le=20)] = 5
    user: str | None = Field(
        default=None,
        description="Optional user id; merges user's private image collection with the global one.",
        max_length=200,
    )


class KBSearchResponse(BaseModel):
    query: str | None = None
    results: list[dict[str, Any]]
    # Carries Audrey's explanation when a request could not be served as
    # asked — today, a `filename` that matched no file. The model has to see
    # it: empty results alone read as "that file says nothing about this",
    # and it would report exactly that about a file the user does not have.
    notice: str = ""


class MemoryStoreRequest(BaseModel):
    key: Annotated[str, Field(min_length=1, max_length=200)]
    value: Annotated[str, Field(min_length=1, max_length=20_000)]
    tags: Annotated[str, Field(max_length=500)] = ""


class MemoryRecallRequest(BaseModel):
    user: Annotated[str, Field(min_length=1, max_length=200, description="User scope. Filled in automatically by Audrey — you don't need to supply it. Memories are per-user; the scope can't be relaxed without leaking across accounts.")]
    key: Annotated[str, Field(min_length=1, max_length=200)]


class MemorySearchRequest(BaseModel):
    user: Annotated[str, Field(min_length=1, max_length=200, description="User scope. Filled in automatically by Audrey — you don't need to supply it. Memories are per-user.")]
    query: Annotated[str, Field(min_length=1, max_length=1000, description="Text to keyword-match against memory keys, values, and tags.")]
    top_k: Annotated[int, Field(ge=1, le=20)] = 5


class MemoryEntryResponse(BaseModel):
    key: str
    value: str
    tags: str
    created_at: str
    updated_at: str

    @classmethod
    def from_entry(cls, e: MemoryEntry) -> MemoryEntryResponse:
        return cls(
            key=e.key, value=e.value, tags=e.tags,
            created_at=e.created_at, updated_at=e.updated_at,
        )


class MemorySearchResponse(BaseModel):
    user: str
    query: str
    results: list[MemoryEntryResponse]


# ─── Health ───────────────────────────────────────────────────────────

@app.get("/health", operation_id="health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ─── Tools ────────────────────────────────────────────────────────────


def _prefer_searxng(query: str) -> bool:
    """Deterministically route a query to SearXNG-primary (True) or
    Brave-primary (False) by hashing the query. Deterministic per query so the
    same query always hits the same backend — keeps Brave's cache TTL effective
    and needs no shared mutable counter. blake2b (not Python's salted hash())
    so the split is stable across processes/restarts."""
    digest = hashlib.blake2b(query.strip().lower().encode(), digest_size=8).digest()
    return digest[0] % 2 == 1


# Provider errors that mean "this backend is unavailable right now" and so
# warrant a cross-fallback (vs. a ValueError = config bug that must surface).
_RECOVERABLE = (BraveQuotaError, BraveRateLimitError, BraveUpstreamError, SearxngError)


def _qlog(query: str, limit: int = 80) -> str:
    """Collapse+truncate a query for logging. Model-generated queries run long;
    80 chars is enough to tell two near-identical panel-worker searches apart,
    which is the whole reason these lines carry the query at all."""
    q = " ".join(query.split())
    return q if len(q) <= limit else q[: limit - 1] + "…"


async def _try_other(
    other: BraveClient | SearxngClient,
    other_name: str,
    query: str,
    count: int,
) -> list[SearchResult] | None:
    """Attempt the fallback backend. Return its hits (possibly empty), or None if
    it too hit a recoverable failure (a ValueError here is also treated as a
    recoverable dead-end for the fallback, not re-raised as a 500 — the primary
    already ran; we just couldn't get a second opinion). Shared by the
    error-path and empty-path fallbacks so their handling stays identical."""
    try:
        hits = await other.search(query=query, count=count)
        log.warning(
            "web_search: %s returned %d results for %r (first url=%r)",
            other_name, len(hits), _qlog(query), (hits[0].url if hits else ""),
        )
        return hits
    except (*_RECOVERABLE, ValueError) as fe:
        log.warning("web_search: %s fallback failed: %s", other_name, fe)
        return None


async def _search_with_fallback(
    *,
    primary: BraveClient | SearxngClient,
    other: BraveClient | SearxngClient | None,
    query: str,
    count: int,
) -> list[SearchResult]:
    """Run `primary.search`; cross-fall-back to `other` (the opposite backend) if
    the primary is unavailable OR returns EMPTY. Works either direction — Brave
    primary → SearXNG other, or SearXNG primary → Brave other — because both
    clients share `search(query, count) -> list[SearchResult]`.

    Two fallback triggers:
      * a recoverable provider ERROR (Brave 402/429/upstream, any SearxngError);
      * an EMPTY primary result. An empty is NOT an exception — a SearXNG `200`
        with zero results is usually all its upstream engines throttling at once,
        not "nothing exists" (see searxng.py). Before this, an empty primary
        returned `[]` and silently skipped a healthy `other`, so ~half of queries
        (whatever hashes to SearXNG-primary) got nothing whenever SearXNG's
        general engines throttled, even with Brave quota available. Cost: an empty
        primary now spends one `other` call, eroding the `_prefer_searxng` quota
        split — accepted, since an empty result is useless and grounding matters
        more than conserving the second provider's quota on empties.

    Outcomes: primary hits → return them. Primary empty/error + `other` has hits
    → return other's. Both empty → return empty (an honest "no results", NOT a
    503). Primary error + no `other` (or `other` also errors) → 503. A
    `ValueError` from the PRIMARY is a config bug (e.g. empty API key) → 500."""
    prim_name = "Brave" if isinstance(primary, BraveClient) else "SearXNG"
    other_name = "SearXNG" if prim_name == "Brave" else "Brave"
    try:
        hits = await primary.search(query=query, count=count)
    except ValueError as e:  # misconfiguration (e.g. empty API key) — not transient
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e),
        ) from e
    except _RECOVERABLE as e:
        if other is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"{prim_name} unavailable ({e}) and no fallback backend configured.",
            ) from e
        log.warning("web_search: %s unavailable (%s); falling back to %s for %r",
                    prim_name, e, other_name, _qlog(query))
        result = await _try_other(other, other_name, query, count)
        if result is None:  # both backends down → 503 naming both
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Search unavailable ({prim_name}: {e}; {other_name}: failed)",
            ) from e
        return result

    # Primary succeeded. Non-empty → done. Empty → try `other` for a better
    # answer; if there's no `other` or it also comes back empty/failed, the
    # honest result is the primary's empty (a 200 with [], never a 503).
    if hits or other is None:
        return hits
    log.warning("web_search: %s returned 0 results for %r; trying %s",
                prim_name, _qlog(query), other_name)
    result = await _try_other(other, other_name, query, count)
    return result or hits


@app.post(
    "/web_search",
    operation_id="web_search",
    response_model=WebSearchResponse,
    tags=["tools"],
    summary="Search the web via Brave Search API",
    description=(
        "Query the public web for current information. Use this for questions "
        "about news, recent events, today's date-sensitive facts, or anything "
        "not in the model's training data."
    ),
)
async def web_search(req: WebSearchRequest) -> WebSearchResponse:
    brave: BraveClient = app.state.brave
    searxng: SearxngClient | None = app.state.searxng

    # One line per REQUEST, logged before any routing. The warnings further down
    # only fire when a backend errors or comes back empty — a SearXNG-primary hit
    # logs nothing — so this is the ONLY complete record of what was asked. It's
    # what makes "how many of a deep panel's searches were near-duplicates of each
    # other?" a measurable question rather than an argument: one deep case can fan
    # out to 3 workers x 4 searches, and the cache below only dedups exact matches.
    log.info("web_search: q=%r count=%d", _qlog(req.query), req.count)

    # Route to a PRIMARY backend, then cross-fall-back to the other on failure.
    # With alternation off (or no SearXNG configured), primary is always Brave —
    # the pre-2026-07-09 Brave-primary/SearXNG-fallback behavior. With it on,
    # a per-query hash splits primaries ~50/50 between the two: this halves Brave
    # quota and decorrelates the backends (a bad window on one no longer sinks
    # every worker on a request), while cross-fallback preserves resilience.
    if settings.web_search_alternate and searxng is not None and _prefer_searxng(req.query):
        hits = await _search_with_fallback(
            primary=searxng, other=brave, query=req.query, count=req.count,
        )
    else:
        hits = await _search_with_fallback(
            primary=brave, other=searxng, query=req.query, count=req.count,
        )
    return WebSearchResponse(
        query=req.query,
        results=[WebSearchResult(title=h.title, url=h.url, snippet=h.snippet) for h in hits],
    )


@app.post(
    "/web_fetch",
    operation_id="web_fetch",
    response_model=WebFetchResponse,
    tags=["tools"],
    summary="Open a web page and return its readable text",
    description=(
        "Open a single http(s) URL — typically one a web_search result returned — "
        "and return its main readable text with navigation and boilerplate stripped. "
        "Use this to read past a search snippet when you need an exact date, version "
        "number, spec, or a direct quote the snippet doesn't contain. One URL per "
        "call. Returns an error (with the reason) for non-HTML pages, unreachable "
        "or internal hosts, or pages with no extractable text — on error, fall back "
        "to the search snippet and its URL rather than dropping the source."
    ),
)
async def web_fetch(req: WebFetchRequest) -> WebFetchResponse:
    try:
        final_url, text = await fetch_readable(req.url, max_chars=req.max_chars)
    except FetchError as e:
        # 422 so the reason reaches the model via dispatch's http_4xx detail — it
        # can then pick another URL or use the snippet. FetchError messages are
        # written to be model-safe (no internals). A genuine bug still 500s.
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
    return WebFetchResponse(url=final_url, text=text)


@app.post(
    "/kb_search",
    operation_id="kb_search",
    response_model=KBSearchResponse,
    tags=["tools"],
    summary="Search the local knowledge base (text)",
    description=(
        "Search Audrey's knowledge base for matching documents and image "
        "captions. Use this when the user asks about domain-specific "
        "material (e.g. geology references) or their own ingested docs, "
        "including the transcripts and on-screen text of videos they have "
        "uploaded. To search inside one particular file, pass its exact "
        "filename from list_my_files as `filename`."
    ),
)
async def kb_search(req: KBSearchRequest) -> KBSearchResponse:
    client: httpx.AsyncClient = app.state.audrey
    payload: dict[str, Any] = {"query": req.query, "top_k": req.top_k}
    if req.user:
        payload["user"] = req.user
    if req.filename:
        payload["filename"] = req.filename
    if req.artifact:
        payload["artifact"] = req.artifact
    try:
        r = await client.post("/v1/kb/query", json=payload)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Audrey KB unreachable: {e}",
        ) from e
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    body = r.json()
    return KBSearchResponse(
        query=req.query,
        results=body.get("results", []),
        notice=body.get("notice", "") or "",
    )


@app.post(
    "/kb_image_search",
    operation_id="kb_image_search",
    response_model=KBSearchResponse,
    tags=["tools"],
    summary="Search the local knowledge base for images",
    description=(
        "Find images in the KB by either a text description (e.g. 'someone "
        "in guard position') OR a reference image (URL / base64). Provide "
        "exactly one of: query, image_url, image_b64. Use this for image "
        "lookup; use kb_search for text/document lookup."
    ),
)
async def kb_image_search(req: KBImageSearchRequest) -> KBSearchResponse:
    if not req.query and not req.image_url and not req.image_b64:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="One of query, image_url, or image_b64 is required.",
        )
    client: httpx.AsyncClient = app.state.audrey
    payload: dict[str, Any] = {"top_k": req.top_k}
    if req.query:
        payload["query"] = req.query
    if req.image_url:
        payload["image_url"] = req.image_url
    if req.image_b64:
        payload["image_b64"] = req.image_b64
    if req.user:
        payload["user"] = req.user
    try:
        r = await client.post("/v1/kb/query/image", json=payload)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Audrey KB unreachable: {e}",
        ) from e
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    body = r.json()
    return KBSearchResponse(query=req.query, results=body.get("results", []))


@app.post(
    "/memory_store",
    operation_id="memory_store",
    response_model=MemoryEntryResponse,
    tags=["tools"],
    summary="Save a persistent memory for a specific user",
    description=(
        "Persist a key-value note to long-term memory, scoped to one user. "
        "The user scope is filled in automatically by Audrey — you don't "
        "need to supply it. Add comma-separated topic tags to improve "
        "recall (e.g. `topic:hardware,topic:preferences`). Overwrites any "
        "existing value for the same (user, key) pair."
    ),
)
async def memory_store(req: MemoryStoreRequest) -> MemoryEntryResponse:
    memory: MemoryStore = app.state.memory
    try:
        entry = await memory.store(key=req.key, value=req.value, tags=req.tags)
    except ValueError as e:
        # Missing `user:<id>` tag — memories without a user tag can't be
        # recalled and leak across scopes, so we refuse to write them.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    return MemoryEntryResponse.from_entry(entry)


@app.post(
    "/memory_recall",
    operation_id="memory_recall",
    response_model=MemoryEntryResponse,
    tags=["tools"],
    summary="Recall a persistent memory by key",
    description="Fetch a previously-stored memory by its exact key. Returns 404 if the key is unknown.",
)
async def memory_recall(req: MemoryRecallRequest) -> MemoryEntryResponse:
    memory: MemoryStore = app.state.memory
    entry = await memory.recall(req.key, user=req.user)
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No memory for key: {req.key!r}")
    return MemoryEntryResponse.from_entry(entry)


@app.post(
    "/memory_search",
    operation_id="memory_search",
    response_model=MemorySearchResponse,
    tags=["tools"],
    summary="Semantic search over a user's memories",
    description=(
        "Find a user's memories by embedding the query and cosine-matching "
        "against each stored memory's embedding (via nomic-embed-text). "
        "Scoped by the `user:<id>` tag. Results below the similarity "
        "threshold are dropped. Used by the orchestrator for auto-recall "
        "at the top of every request, but also callable as a tool."
    ),
)
async def memory_search(req: MemorySearchRequest) -> MemorySearchResponse:
    memory: MemoryStore = app.state.memory
    hits = await memory.search(user=req.user, query=req.query, top_k=req.top_k)
    return MemorySearchResponse(
        user=req.user,
        query=req.query,
        results=[MemoryEntryResponse.from_entry(h) for h in hits],
    )


# ─── Chat archive ─────────────────────────────────────────────────────

class ChatHistorySearchRequest(BaseModel):
    user: Annotated[str, Field(min_length=1, max_length=200, description="User scope. Filled in automatically by Audrey — you don't need to supply it. Chat history is per-user.")]
    query: Annotated[str, Field(min_length=1, max_length=1000, description="Natural-language search over the user's previous conversations.")]
    limit: Annotated[int, Field(
        ge=1, le=20,
        description=(
            "Max results to return. Must be 1-20. Default 5 is right for "
            "most lookups; use a larger limit only when you need broad recall."
        ),
    )] = 5
    date_from: str | None = Field(default=None, description="ISO timestamp — only return hits at or after this time.")
    date_to: str | None = Field(default=None, description="ISO timestamp — only return hits at or before this time.")


class ChatHistorySearchHit(BaseModel):
    conversation_id: str
    chunk_id: str
    created_at: str
    snippet: str
    score: float


class ChatHistorySearchResponse(BaseModel):
    query: str
    results: list[ChatHistorySearchHit]


@app.post(
    "/chat_history_search",
    operation_id="chat_history_search",
    response_model=ChatHistorySearchResponse,
    tags=["tools"],
    summary="Search this user's prior conversations with you",
    description=(
        "Search this user's prior conversations with you. Use only when the "
        "user references something previously discussed, or when answering "
        "requires a specific prior decision. Do not call to personalize "
        "ordinary answers or to repeat back recent context. Returns short "
        "snippets and conversation ids; never returns another user's data."
    ),
)
async def chat_history_search(req: ChatHistorySearchRequest) -> ChatHistorySearchResponse:
    archive: ChatArchiveStore = app.state.chat_archive
    hits = await archive.search(
        user=req.user, query=req.query, limit=req.limit,
        date_from=req.date_from, date_to=req.date_to,
    )
    return ChatHistorySearchResponse(
        query=req.query,
        results=[
            ChatHistorySearchHit(
                conversation_id=h.conversation_id,
                chunk_id=h.chunk_id,
                created_at=h.created_at,
                snippet=h.snippet,
                score=h.score,
            )
            for h in hits
        ],
    )


class ListMyFilesRequest(BaseModel):
    user: Annotated[str, Field(min_length=1, max_length=200, description="User scope. Filled in automatically by Audrey — you don't need to supply it. File lists are per-user.")]


class MyFileRow(BaseModel):
    # Mirrors `ModelFileRow` in `audrey/routes/files.py`. Only `description`
    # and the *request* schema reach the model (`ToolSpec.to_ollama_tool`), so
    # nothing documented on these fields is visible to it — anything the model
    # must know to read the result belongs in the description below.
    filename: str
    kind: str
    status: str
    uploaded_at: str
    duration_s: float = 0.0
    summary: str = ""
    failure_reason: str = ""
    waiting_for_s: float = 0.0


class ListMyFilesResponse(BaseModel):
    files: list[MyFileRow]


@app.post(
    "/list_my_files",
    operation_id="list_my_files",
    response_model=ListMyFilesResponse,
    tags=["tools"],
    summary="List the files this user has uploaded",
    description=(
        "List the files this user has uploaded to Audrey — videos, documents "
        "and images — returning each one's exact filename, kind, upload time, "
        "processing status and, for a processed video, its duration and a "
        "one-paragraph summary. Call this when the user asks what they have "
        "uploaded, refers to 'my video' or 'that recording' without naming "
        "it, or when you need a file's exact filename. A status of 'pending' "
        "or 'processing' means Audrey is still working on that file and its "
        "contents are not searchable yet — say so rather than reporting it as "
        "empty, and use waiting_for_s (seconds so far, and only meaningful "
        "for those two statuses) if asked how long it has been going. A "
        "'failed' file carries the reason. Returns only this user's own files."
    ),
)
async def list_my_files(req: ListMyFilesRequest) -> ListMyFilesResponse:
    client: httpx.AsyncClient = app.state.audrey
    try:
        r = await client.post("/v1/files/list", json={"user": req.user})
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Audrey files unreachable: {e}",
        ) from e
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    body = r.json()
    return ListMyFilesResponse(files=body.get("files", []))


# ─── Chat archive: internal write/admin (not in /openapi.json) ────────
# `include_in_schema=False` keeps these out of OpenAPI tool discovery.
# The model never sees them, only Audrey's archive client and ops.

class ArchiveTurnRequest(BaseModel):
    user: Annotated[str, Field(min_length=1, max_length=200)]
    conversation_id: Annotated[str, Field(min_length=1, max_length=200)]
    user_content: str
    assistant_content: str
    partial: bool = False
    virtual_model: str = ""
    concrete_model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@app.post("/chat_history/archive", include_in_schema=False, tags=["internal"])
async def chat_history_archive(req: ArchiveTurnRequest) -> dict[str, Any]:
    archive: ChatArchiveStore = app.state.chat_archive
    return await archive.archive_turn(
        user=req.user,
        conversation_id=req.conversation_id,
        user_content=req.user_content,
        assistant_content=req.assistant_content,
        partial=req.partial,
        virtual_model=req.virtual_model,
        concrete_model=req.concrete_model,
        prompt_tokens=req.prompt_tokens,
        completion_tokens=req.completion_tokens,
    )


@app.post("/chat_history/prune", include_in_schema=False, tags=["internal"])
async def chat_history_prune() -> dict[str, int]:
    archive: ChatArchiveStore = app.state.chat_archive
    return await archive.prune()


@app.get("/chat_history/stats", include_in_schema=False, tags=["internal"])
async def chat_history_stats() -> dict[str, int]:
    archive: ChatArchiveStore = app.state.chat_archive
    return await archive.stats()
