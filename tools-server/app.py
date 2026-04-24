"""custom-tools FastAPI server.

Six endpoints, OpenAPI auto-discovered by the Audrey orchestrator:
  POST /web_search         — Brave Search API
  POST /kb_search          — text query proxied to Audrey /v1/kb/query
  POST /kb_image_search    — image query proxied to Audrey /v1/kb/query/image
  POST /memory_store       — save (key, value, user:<id>-tagged) to Qdrant
  POST /memory_recall      — fetch by exact key
  POST /memory_search      — semantic search over a user's memories

Memory is Qdrant-backed (Phase 12): each entry is a point in the
`kb_memory` collection, embedded with nomic-embed-text. On first startup,
a legacy `memory.db` SQLite file (Phase 11) is migrated automatically and
renamed to `memory.db.migrated`.

Each endpoint has a clear operation_id so the orchestrator's OpenAPI →
Ollama-tool converter produces sensible tool names.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from brave import BraveClient, BraveRateLimitError, SearchResult
from db import MemoryEntry, MemoryStore
from settings import settings

log = logging.getLogger("custom-tools")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ─── Lifespan (startup / shutdown) ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    brave = BraveClient(
        api_key=settings.brave_api_key,
        cache_ttl_seconds=settings.brave_cache_ttl_hours * 3600,
    )
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
    )

    app.state.brave = brave
    app.state.memory = memory
    app.state.audrey = audrey
    log.info(
        "custom-tools ready. brave=%s audrey=%s qdrant=%s collection=%s threshold=%.2f",
        "configured" if settings.brave_api_key else "UNSET",
        settings.audrey_url,
        settings.qdrant_url,
        settings.memory_collection,
        settings.memory_similarity_threshold,
    )
    try:
        yield
    finally:
        await brave.aclose()
        await audrey.aclose()
        await memory.aclose()


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


class KBSearchRequest(BaseModel):
    query: Annotated[str, Field(min_length=1, max_length=1000)]
    top_k: Annotated[int, Field(ge=1, le=20)] = 5


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


class KBSearchResponse(BaseModel):
    query: str | None = None
    results: list[dict[str, Any]]


class MemoryStoreRequest(BaseModel):
    key: Annotated[str, Field(min_length=1, max_length=200)]
    value: Annotated[str, Field(min_length=1, max_length=20_000)]
    tags: Annotated[str, Field(max_length=500)] = ""


class MemoryRecallRequest(BaseModel):
    key: Annotated[str, Field(min_length=1, max_length=200)]


class MemorySearchRequest(BaseModel):
    user: Annotated[str, Field(min_length=1, max_length=200, description="User id to scope the search to. Required — memories are per-user.")]
    query: Annotated[str, Field(min_length=1, max_length=1000, description="Text to keyword-match against memory keys, values, and tags.")]
    top_k: Annotated[int, Field(ge=1, le=20)] = 5


class MemoryEntryResponse(BaseModel):
    key: str
    value: str
    tags: str
    created_at: str
    updated_at: str

    @classmethod
    def from_entry(cls, e: MemoryEntry) -> "MemoryEntryResponse":
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
    try:
        hits: list[SearchResult] = await brave.search(query=req.query, count=req.count)
    except BraveRateLimitError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Brave Search rate-limited: {e}",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    return WebSearchResponse(
        query=req.query,
        results=[WebSearchResult(title=h.title, url=h.url, snippet=h.snippet) for h in hits],
    )


@app.post(
    "/kb_search",
    operation_id="kb_search",
    response_model=KBSearchResponse,
    tags=["tools"],
    summary="Search the local knowledge base (text)",
    description=(
        "Search Audrey's knowledge base for matching documents and image "
        "captions. Use this when the user asks about domain-specific "
        "material (e.g. geology references) or their own ingested docs."
    ),
)
async def kb_search(req: KBSearchRequest) -> KBSearchResponse:
    client: httpx.AsyncClient = app.state.audrey
    try:
        r = await client.post("/v1/kb/query", json={"query": req.query, "top_k": req.top_k})
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
        "`tags` MUST contain `user:<id>` (e.g. `user:bart@proton.me`) — "
        "memories without a user tag are rejected. Add further comma-"
        "separated topic tags to improve recall (e.g. "
        "`user:bart@proton.me,topic:hardware`). Overwrites any existing "
        "value for the same (user, key) pair."
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
    entry = await memory.recall(req.key)
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
