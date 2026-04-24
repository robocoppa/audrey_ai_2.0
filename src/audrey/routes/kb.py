"""KB HTTP endpoints.

    POST /v1/kb/query         — text query, returns top-k hits
    POST /v1/kb/query/image   — image query (url or b64), returns top-k hits
    POST /v1/kb/ingest        — trigger an ingest run over one or more paths
    GET  /v1/kb/stats         — per-collection point counts

`custom-tools`' `kb_search` and `kb_image_search` tools proxy to these
two query endpoints, so when the ReAct loop dispatches a KB lookup it
comes right back into this router.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from audrey.kb.embed import ImageEmbedder, TextEmbedder
from audrey.kb.ingest import ingest_many
from audrey.kb.qdrant import KBHit, QdrantKB
from audrey.kb.user_store import user_image_collection, user_text_collection

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/kb", tags=["kb"])


class TextQuery(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    user: str | None = Field(
        default=None,
        description=(
            "Optional user id. If set and the user has a kb_user_text_<sanitized> "
            "collection, it is searched alongside the global kb_text and results "
            "are merged by score."
        ),
        max_length=200,
    )


class ImageQuery(BaseModel):
    query: str | None = Field(
        default=None,
        description="Text query — encoded via CLIP's text tower and searched against kb_images.",
        max_length=2000,
    )
    image_url: str | None = Field(default=None, description="HTTP(S) URL of the image.")
    image_b64: str | None = Field(default=None, description="Base64-encoded image bytes.")
    top_k: int = Field(default=5, ge=1, le=20)
    user: str | None = Field(
        default=None,
        description="Optional user id. Merges user's private image collection with the global one.",
        max_length=200,
    )


class IngestRequest(BaseModel):
    paths: list[str] | None = Field(
        default=None,
        description="Absolute paths to ingest. If omitted, falls back to config's kb.dataset_paths.",
    )


class Hit(BaseModel):
    score: float
    source: str
    kind: str
    chunk_idx: int
    text: str


class QueryResponse(BaseModel):
    query: str | None = None
    results: list[Hit]


@router.post("/query", response_model=QueryResponse)
async def kb_query(req: TextQuery, request: Request) -> QueryResponse:
    qdrant: QdrantKB | None = getattr(request.app.state, "qdrant", None)
    embedder: TextEmbedder | None = getattr(request.app.state, "text_embedder", None)
    if qdrant is None or embedder is None:
        raise HTTPException(status_code=503, detail="KB is not initialized")
    vec = await embedder.embed_one(req.query)
    hits = await _search_text_merged(qdrant, vec, top_k=req.top_k, user=req.user)
    return QueryResponse(
        query=req.query,
        results=[
            Hit(score=h.score, source=h.source, kind=h.kind, chunk_idx=h.chunk_idx, text=h.text)
            for h in hits
        ],
    )


async def _search_text_merged(
    qdrant: QdrantKB, vec: list[float], *, top_k: int, user: str | None,
) -> list[KBHit]:
    """Search global kb_text and, if the user has one, their kb_user_text_* too. Merge by score."""
    tasks = [qdrant.search_text(vec, top_k=top_k)]
    if user:
        user_col = user_text_collection(user)
        if await qdrant.collection_exists(user_col):
            tasks.append(qdrant.search_text(vec, top_k=top_k, collection=user_col))
    results = await asyncio.gather(*tasks)
    merged: list[KBHit] = [h for batch in results for h in batch]
    merged.sort(key=lambda h: h.score, reverse=True)
    return merged[:top_k]


async def _search_images_merged(
    qdrant: QdrantKB, vec: list[float], *, top_k: int, user: str | None,
) -> list[KBHit]:
    tasks = [qdrant.search_images(vec, top_k=top_k)]
    if user:
        user_col = user_image_collection(user)
        if await qdrant.collection_exists(user_col):
            tasks.append(qdrant.search_images(vec, top_k=top_k, collection=user_col))
    results = await asyncio.gather(*tasks)
    merged: list[KBHit] = [h for batch in results for h in batch]
    merged.sort(key=lambda h: h.score, reverse=True)
    return merged[:top_k]


@router.post("/query/image", response_model=QueryResponse)
async def kb_query_image(req: ImageQuery, request: Request) -> QueryResponse:
    qdrant: QdrantKB | None = getattr(request.app.state, "qdrant", None)
    embedder: ImageEmbedder | None = getattr(request.app.state, "image_embedder", None)
    if qdrant is None or embedder is None:
        raise HTTPException(status_code=503, detail="KB image search is not initialized")
    if not req.image_url and not req.image_b64 and not req.query:
        raise HTTPException(
            status_code=422,
            detail="One of query, image_url, or image_b64 is required.",
        )
    try:
        if req.image_url:
            vec = await embedder.embed_url(req.image_url)
        elif req.image_b64:
            vec = await embedder.embed_b64(req.image_b64)
        else:
            vec = await embedder.embed_text(req.query or "")
    except Exception as e:  # noqa: BLE001 — surface embed errors as 4xx
        raise HTTPException(status_code=422, detail=f"image embed failed: {e}") from e
    hits = await _search_images_merged(qdrant, vec, top_k=req.top_k, user=req.user)
    return QueryResponse(
        query=req.query,
        results=[
            Hit(score=h.score, source=h.source, kind=h.kind, chunk_idx=h.chunk_idx, text=h.text)
            for h in hits
        ],
    )


@router.post("/ingest")
async def kb_ingest(req: IngestRequest, request: Request) -> dict[str, Any]:
    app = request.app
    qdrant: QdrantKB | None = getattr(app.state, "qdrant", None)
    text_embedder: TextEmbedder | None = getattr(app.state, "text_embedder", None)
    image_embedder: ImageEmbedder | None = getattr(app.state, "image_embedder", None)
    if qdrant is None or text_embedder is None:
        raise HTTPException(status_code=503, detail="KB is not initialized")

    cfg = app.state.cfg
    kb_cfg = cfg.raw.get("kb", {}) or {}
    roots = [Path(p) for p in (req.paths or kb_cfg.get("dataset_paths") or [])]
    if not roots:
        raise HTTPException(status_code=400, detail="No paths provided and kb.dataset_paths is empty.")
    chunk_tokens = int(kb_cfg.get("chunk_tokens", 1000))
    overlap = int(kb_cfg.get("chunk_overlap", 100))
    stats = await ingest_many(
        roots, qdrant=qdrant, text_embedder=text_embedder,
        image_embedder=image_embedder,
        chunk_tokens=chunk_tokens, overlap_tokens=overlap,
    )
    log.info("kb.ingest (http): %s", stats.as_dict())
    return {"roots": [str(r) for r in roots], **stats.as_dict()}


@router.get("/stats")
async def kb_stats(request: Request) -> dict[str, Any]:
    qdrant: QdrantKB | None = getattr(request.app.state, "qdrant", None)
    if qdrant is None:
        raise HTTPException(status_code=503, detail="KB is not initialized")
    counts = await qdrant.counts()
    return {
        "collections": counts,
        "text_collection": qdrant.text_collection,
        "image_collection": qdrant.image_collection,
    }


__all__ = ["router"]
