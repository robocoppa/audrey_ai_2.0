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
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from audrey.auth import AuthedUser, KBCaller, require_admin, resolve_kb_caller
from audrey.kb.embed import ImageEmbedder, TextEmbedder
from audrey.kb.fusion import RRF_K, passes_evidence, reciprocal_rank_fusion
from audrey.kb.ingest import ingest_many
from audrey.kb.qdrant import KBHit, QdrantKB, SearchScope
from audrey.kb.user_store import user_image_collection, user_text_collection
from audrey.metrics import kb_search_hits, kb_search_seconds

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
    filename: str | None = Field(
        default=None,
        description=(
            "Optional filename of one of the user's own uploads. When set, the "
            "search covers only that file — the global KB is not searched at "
            "all, because a user's upload never lives there. Resolved against "
            "the uploads index, so a name that matches nothing returns no hits "
            "rather than silently widening to everything."
        ),
        max_length=500,
    )
    artifact: str | None = Field(
        default=None,
        description=(
            "Optional. For a processed video, which derived text to search: "
            "'transcript' (what was said), 'visual' (what was on screen) or "
            "'summary'. Omit to search all of them."
        ),
        pattern="^(transcript|visual|summary)$",
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
    # Phase 40. Empty for global-KB hits, which have no uploader. Without
    # these a caller cannot say *which* file an answer came from: `source` for
    # an upload is the sidecar path, which is file_id-derived and unreadable.
    filename: str = ""
    artifact: str = ""


class QueryResponse(BaseModel):
    query: str | None = None
    results: list[Hit]
    # Set only when something about the request needs saying — today, a
    # `filename` that matched no file. Empty results alone are ambiguous
    # between "that file has nothing about this" and "there is no such file",
    # and a model told the first will confidently report the wrong one.
    notice: str = ""


class StatsResponse(BaseModel):
    collections: dict[str, int] = Field(
        description=(
            "Per-collection point counts. A value of -1 means the count call "
            "to Qdrant failed for that collection (transport error, missing "
            "collection, etc.) — treat -1 as 'unknown', not 'zero'."
        )
    )
    text_collection: str
    image_collection: str


def _hybrid_cfg(request: Request) -> dict[str, Any]:
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        return {}
    return ((cfg.raw.get("kb", {}) or {}).get("hybrid", {}) or {})


def _kb_min_score(request: Request) -> float:
    """Cosine-similarity floor for KB text hits, from `kb.min_score` (default 0.0).

    Below the floor, a hit is discarded rather than returned. The KB always
    returns its top_k *nearest* vectors regardless of how far they are, so on a
    query the corpus can't answer it hands back the least-irrelevant junk — which
    then pollutes a researcher's context and reads as a real source (2026-07-15
    trace run: a vaccine query returned PowerApps / ServiceNow / Forest-Service
    docs). A floor turns "nothing relevant" into an *empty* result, which the
    researcher already handles gracefully (falls back to web/prior knowledge),
    instead of injecting off-topic text. Default 0.0 = OFF (no behavior change
    until tuned on the box), since the right threshold is corpus-dependent —
    unit-normalized nomic-embed cosine, relevant chunks typically score well
    above unrelated ones, but the exact cut must be set against real hit scores."""
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        return 0.0
    return float((cfg.raw.get("kb", {}) or {}).get("min_score", 0.0))


_NOTICE_MAX_NAMES = 20


def _unknown_file_notice(wanted: str, available: list[str]) -> str:
    """Tell the caller the file is missing, and what does exist.

    Naming the alternatives costs nothing — `_resolve_filename` has already
    read the user's rows to fail — and it closes the loop in one turn instead
    of two. Without it the model's only move is to call `list_my_files` and
    ask again, and its more likely move is to guess a second filename.

    Capped, because a prolific user's whole file list does not belong in a
    tool result. The count is still exact so the reply is never misleading
    about how much was left out.
    """
    if not available:
        return (
            f"No file named {wanted!r} was found, because this user has not "
            f"uploaded any files yet. Nothing was searched."
        )
    shown = sorted(available)[:_NOTICE_MAX_NAMES]
    more = len(available) - len(shown)
    listing = ", ".join(repr(n) for n in shown) + (f", and {more} more" if more else "")
    return (
        f"No file named {wanted!r} was found for this user, so nothing was "
        f"searched. Available files are: {listing}. Re-run with one of those "
        f"exact names, or omit the filename to search everything."
    )


async def _resolve_filename(
    request: Request, *, user: str | None, filename: str,
) -> tuple[list[str], list[str]]:
    """Filenames a user typed to the file_ids Qdrant stores. May be empty.

    Resolution happens here, in one place, rather than by filtering Qdrant on
    the `filename` payload directly. Two reasons, and the second is the real
    one:

    - **A miss is distinguishable.** Filtering on the payload returns nothing
      both for "no such file" and "that file says nothing about this", and the
      caller cannot tell which. Here an empty list means the first, and the
      route says so.
    - **The uploads index is the authority on what a user has.** It is what
      `list_my_files` lists, so a filename the model was just shown resolves
      by construction. Matching payload strings would drift the moment the two
      disagree.

    Matching is case-insensitive but otherwise exact — no stemming, no
    prefixes, no "closest match". Scoping to the wrong file answers
    confidently from the wrong source, which is worse than not scoping at all,
    so a near-miss must miss.

    Duplicates are kept: two uploads can share a filename, and "in
    standup.mp4" then honestly means both.

    Returns `(matched_file_ids, every_filename_this_user_has)`. The second is
    a by-product of the lookup rather than a second query, and it is what lets
    a miss name the alternatives instead of just denying the request.
    """
    db = getattr(request.app.state, "uploads_db", None)
    if db is None or not user:
        return [], []
    wanted = filename.strip().casefold()
    rows = await db.list_user(user)
    matched = [
        str(r["file_id"]) for r in rows
        if str(r["filename"]).strip().casefold() == wanted
    ]
    return matched, [str(r["filename"]) for r in rows]


@router.post("/query", response_model=QueryResponse)
async def kb_query(
    req: TextQuery,
    request: Request,
    caller: KBCaller = Depends(resolve_kb_caller),
) -> QueryResponse:
    qdrant: QdrantKB | None = getattr(request.app.state, "qdrant", None)
    embedder: TextEmbedder | None = getattr(request.app.state, "text_embedder", None)
    if qdrant is None or embedder is None:
        raise HTTPException(status_code=503, detail="KB is not initialized")
    effective_user = req.user if caller.is_service else caller.email

    scope: SearchScope | None = None
    notice = ""
    if req.filename or req.artifact:
        file_ids: list[str] | None = None
        if req.filename:
            file_ids, available = await _resolve_filename(
                request, user=effective_user, filename=req.filename,
            )
            if not file_ids:
                # Answer without touching Qdrant. A scoped search that cannot
                # match must not become an unscoped one, and the caller needs
                # to know the file is the problem, not the question.
                log.info(
                    "kb: filename %r matched no file for user=%s", req.filename, effective_user,
                )
                return QueryResponse(
                    query=req.query, results=[],
                    notice=_unknown_file_notice(req.filename, available),
                )
            if len(file_ids) > 1:
                # Say so rather than merging silently. The caller asked about
                # "standup.mp4" believing it named one thing; an answer
                # stitched from two different recordings under that name is
                # wrong in a way it cannot see.
                notice = (
                    f"{len(file_ids)} uploaded files are named {req.filename!r}, "
                    f"and all of them were searched — results may mix them."
                )
        scope = SearchScope(file_ids=file_ids, artifact=req.artifact)

    t0 = time.perf_counter()
    vec = await embedder.embed_one(req.query)
    hybrid = _hybrid_cfg(request)
    if hybrid.get("enabled"):
        hits, had_user = await _search_text_hybrid(
            qdrant, vec, query=req.query, top_k=req.top_k, user=effective_user,
            min_score=_kb_min_score(request), cfg=hybrid, scope=scope,
        )
    else:
        hits, had_user = await _search_text_merged(
            qdrant, vec, top_k=req.top_k, user=effective_user,
            min_score=_kb_min_score(request), scope=scope,
        )
    elapsed = time.perf_counter() - t0
    kb_search_seconds.labels(kind="text", had_user_collection=str(had_user).lower()).observe(elapsed)
    kb_search_hits.labels(kind="text").observe(len(hits))
    return QueryResponse(
        query=req.query,
        notice=notice,
        results=[
            Hit(
                score=h.score, source=h.source, kind=h.kind, chunk_idx=h.chunk_idx,
                text=h.text,
                filename=str(h.payload.get("filename") or ""),
                artifact=str(h.payload.get("artifact") or ""),
            )
            for h in hits
        ],
    )


def _scoped_to_one_users_files(scope: SearchScope | None) -> bool:
    """True when the request names particular files, so the global KB is moot.

    An upload's chunks only ever land in `kb_user_text_<user>`; the global
    collection has no `file_id` payload at all. Searching it under a file
    filter would return nothing, correctly but pointlessly — and would spend
    an embedding round trip and a Qdrant query to do it.
    """
    return scope is not None and scope.file_ids is not None


async def _search_text_merged(
    qdrant: QdrantKB, vec: list[float], *, top_k: int, user: str | None,
    min_score: float = 0.0, scope: SearchScope | None = None,
) -> tuple[list[KBHit], bool]:
    """Search global kb_text and, if the user has one, their kb_user_text_* too. Merge by score.

    Second return value is True iff the user's private collection was actually
    merged in (user supplied + collection exists). The metrics label uses it.

    Score-merge precondition: both collections must use the same embedder
    (currently 768-d nomic-embed-text, cosine) so the raw scores are
    comparable. If a per-user collection ever ships with a different model,
    switch to reciprocal-rank-fusion rather than sorting by raw score.

    `min_score` drops hits below the cosine floor (see `_kb_min_score`). To keep
    `top_k` *usable* hits when a floor is active, we over-fetch from Qdrant (the
    floor may reject the nearest few), then filter, then cap — otherwise fetching
    only `top_k` and filtering could return fewer than `top_k` real hits when the
    nearest ones are below the floor. `0.0` (the default) keeps every hit and
    fetches exactly `top_k`, so this is a no-op until tuned.
    """
    # Over-fetch when a floor is active so below-floor near-neighbours can't
    # starve real hits ranked just past top_k; cap the fetch so a huge top_k
    # can't balloon the Qdrant scan. No floor → fetch exactly top_k (unchanged).
    fetch_k = min(top_k * 4, 40) if min_score > 0.0 else top_k
    coros = (
        [] if _scoped_to_one_users_files(scope)
        else [qdrant.search_text(vec, top_k=fetch_k)]
    )
    had_user = False
    if user:
        user_col = user_text_collection(user)
        if await qdrant.collection_exists(user_col):
            coros.append(
                qdrant.search_text(vec, top_k=fetch_k, collection=user_col, scope=scope))
            had_user = True
    results = await asyncio.gather(*coros)
    merged: list[KBHit] = [h for batch in results for h in batch if h.score >= min_score]
    merged.sort(key=lambda h: h.score, reverse=True)
    return merged[:top_k], had_user


async def _search_text_hybrid(
    qdrant: QdrantKB, vec: list[float], *, query: str, top_k: int, user: str | None,
    min_score: float, cfg: dict[str, Any], scope: SearchScope | None = None,
) -> tuple[list[KBHit], bool]:
    """Dense and lexical, merged by rank, filtered by evidence (Phase 39).

    Same two return values as `_search_text_merged` so the caller and its
    metrics do not care which path ran.

    Both retrievers are asked for more than `top_k`. Fusion reorders, and the
    evidence rule removes — so fetching exactly `top_k` from each would return
    fewer than `top_k` real hits whenever anything was dropped. The over-fetch
    is capped so a large `top_k` cannot turn into an unbounded scan.

    Each retriever searches the global collection and, when the user has one,
    their private collection too. All four lists are concatenated *within*
    their own retriever before fusion, because RRF reads rank position and a
    hit's position only means something relative to the same retriever.

    **`scope` reaches both retrievers by construction** (phase 40) — see
    `QdrantKB.search_hybrid`, which takes it once per collection and fans out.
    This route cannot express a dense-only filter, which is the point: that
    bug produces a plausible, sourced, partly-wrong answer and nothing about
    it looks wrong.
    """
    fetch_k = min(max(top_k * 4, 20), 40)
    # One call per collection, each handing `search_hybrid` the scope exactly
    # once — there is no argument list here in which the dense and lexical
    # sides could disagree.
    coros = (
        [] if _scoped_to_one_users_files(scope)
        else [qdrant.search_hybrid(vec, query, top_k=fetch_k)]
    )
    had_user = False
    if user:
        user_col = user_text_collection(user)
        if await qdrant.collection_exists(user_col):
            coros.append(qdrant.search_hybrid(
                vec, query, top_k=fetch_k, collection=user_col, scope=scope))
            had_user = True

    pairs = await asyncio.gather(*coros)
    dense = [h for d, _ in pairs for h in d]
    lexical = [h for _, lex in pairs for h in lex]
    dense.sort(key=lambda h: h.score, reverse=True)
    lexical.sort(key=lambda h: h.score, reverse=True)

    fused = reciprocal_rank_fusion(
        dense, lexical, rrf_k=int(cfg.get("rrf_k", RRF_K)), query=query,
    )
    min_overlap = float(cfg.get("min_term_overlap", 0.7))
    kept = [f for f in fused if passes_evidence(
        f, min_score=min_score, min_overlap=min_overlap)]

    # Report the fused score, not the originating retriever's.
    #
    # A `KBHit` carries whichever score the retriever that found it produced,
    # and those are different scales — a cosine of 0.47 next to a BM25 score
    # of 13.8, ordered by neither. Shipped that way on 2026-08-03 it made the
    # results unreadable and any downstream comparison meaningless. The fused
    # value is the only number that describes the list it is in.
    return [replace(f.hit, score=f.score) for f in kept[:top_k]], had_user


async def _search_images_merged(
    qdrant: QdrantKB, vec: list[float], *, top_k: int, user: str | None,
) -> tuple[list[KBHit], bool]:
    # Same score-merge precondition as `_search_text_merged`: both image
    # collections use 512-d CLIP ViT-B-32, cosine.
    coros = [qdrant.search_images(vec, top_k=top_k)]
    had_user = False
    if user:
        user_col = user_image_collection(user)
        if await qdrant.collection_exists(user_col):
            coros.append(qdrant.search_images(vec, top_k=top_k, collection=user_col))
            had_user = True
    results = await asyncio.gather(*coros)
    merged: list[KBHit] = [h for batch in results for h in batch]
    merged.sort(key=lambda h: h.score, reverse=True)
    return merged[:top_k], had_user


@router.post("/query/image", response_model=QueryResponse)
async def kb_query_image(
    req: ImageQuery,
    request: Request,
    caller: KBCaller = Depends(resolve_kb_caller),
) -> QueryResponse:
    qdrant: QdrantKB | None = getattr(request.app.state, "qdrant", None)
    embedder: ImageEmbedder | None = getattr(request.app.state, "image_embedder", None)
    if qdrant is None or embedder is None:
        raise HTTPException(status_code=503, detail="KB image search is not initialized")
    if not req.image_url and not req.image_b64 and not req.query:
        raise HTTPException(
            status_code=422,
            detail="One of query, image_url, or image_b64 is required.",
        )
    t0 = time.perf_counter()
    try:
        if req.image_url:
            vec = await embedder.embed_url(req.image_url)
        elif req.image_b64:
            vec = await embedder.embed_b64(req.image_b64)
        else:
            vec = await embedder.embed_text(req.query or "")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"image embed failed: {e}") from e
    effective_user = req.user if caller.is_service else caller.email
    hits, had_user = await _search_images_merged(qdrant, vec, top_k=req.top_k, user=effective_user)
    elapsed = time.perf_counter() - t0
    kb_search_seconds.labels(kind="image", had_user_collection=str(had_user).lower()).observe(elapsed)
    kb_search_hits.labels(kind="image").observe(len(hits))
    return QueryResponse(
        query=req.query,
        results=[
            Hit(score=h.score, source=h.source, kind=h.kind, chunk_idx=h.chunk_idx, text=h.text)
            for h in hits
        ],
    )


@router.post("/ingest")
async def kb_ingest(
    req: IngestRequest,
    request: Request,
    _admin: AuthedUser = Depends(require_admin),
) -> dict[str, Any]:
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


@router.get("/stats", response_model=StatsResponse)
async def kb_stats(request: Request) -> StatsResponse:
    qdrant: QdrantKB | None = getattr(request.app.state, "qdrant", None)
    if qdrant is None:
        raise HTTPException(status_code=503, detail="KB is not initialized")
    counts = await qdrant.counts()
    return StatsResponse(
        collections=counts,
        text_collection=qdrant.text_collection,
        image_collection=qdrant.image_collection,
    )


__all__ = ["router"]
