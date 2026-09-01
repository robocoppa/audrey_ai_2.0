"""Qdrant wrapper for Audrey's knowledge base.

Two collections:
  - `kb_text`   : 768-d (nomic-embed-text via ollama /api/embed)
  - `kb_images` : 512-d (CLIP ViT-B-32 via sentence-transformers)

We create collections eagerly at startup with explicit vector dim + cosine
distance so a dim mismatch surfaces now, not on the first query. The
qdrant-client sync API is wrapped in `asyncio.to_thread` — it holds no
event loop of its own, so this is fine and keeps the orchestrator async.

Payload convention (both collections):
  {
    "source": "/datasets/geology/rocks.md",   # absolute path
    "kind":   "text" | "image",
    "text":   "...",                          # present for text chunks
    "caption": "...",                         # present for image chunks
    "chunk_idx": 0,                           # 0-based within the source
    "mtime": 1776000000,                      # source mtime when ingested
  }

Point IDs are UUIDv5(namespace=DNS, name=f"{source}:{kind}:{idx}") so
re-ingesting the same source replaces its points instead of duplicating.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from audrey.kb import bm25

log = logging.getLogger(__name__)

TEXT_DIM = 768
IMAGE_DIM = 512
_NAMESPACE = uuid.NAMESPACE_DNS

# The lexical half of hybrid retrieval (Phase 39). Named, unlike the dense
# vector, which stays unnamed — verified against Qdrant 1.18.3 on 2026-08-03
# that the two coexist and that a dense search still works with no `using=`,
# so nothing about the existing dense path changes.
#
# `Modifier.IDF` makes the server compute inverse document frequency from the
# collection's own statistics. `kb/bm25.py` therefore stores only the term
# frequency half, and a vector written today stays correct as the corpus grows.
SPARSE_NAME = "bm25"

# Text collections only. Image points carry a `caption`, and lexical search
# over captions is a real idea but not this phase's — an unused sparse config
# on every image collection is clutter that would read as an oversight.
_SPARSE_CONFIG = {SPARSE_NAME: qmodels.SparseVectorParams(
    modifier=qmodels.Modifier.IDF)}


@dataclass(slots=True)
class KBHit:
    score: float
    source: str
    kind: str
    chunk_idx: int
    text: str
    payload: dict[str, Any]


@dataclass(slots=True, frozen=True)
class SearchScope:
    """Narrows a search to particular files and/or artifact kinds (phase 40).

    One object rather than loose keyword arguments, so that "did the lexical
    side get the same filter as the dense side?" is answerable by looking at
    one value being threaded through, not by comparing two argument lists.

    `file_ids` is a list because filenames are not unique — two uploads can
    share one, and "search in standup.mp4" then legitimately means both.

    **An empty `file_ids` is refused at construction, not handled.** It would
    mean "a file was named and nothing matched", and every way of expressing
    that downstream is a trap: an empty `MatchAny` is not dependably "matches
    nothing" across qdrant-client versions and its local mode, and the way it
    fails is by matching *everything* — a scoped question silently answered
    from the whole corpus. Rather than carry a sentinel every consumer has to
    remember to check, the state is unrepresentable: a caller whose lookup
    found nothing must not search, and says so itself. `routes/kb.py` returns
    early with a notice.

    `artifact` selects among a video's derived texts — "transcript" (what was
    said), "visual" (what was on screen) or "summary".
    """

    file_ids: list[str] | None = None
    artifact: str | None = None
    # Load-bearing ownership boundary for private reads. Collection names are
    # sanitized and can collide, so routes must filter on the exact raw user
    # stored in each point as well as selecting the user collection.
    user: str | None = None
    # Durable deletion tombstones. Qdrant cleanup may be retrying, so private
    # reads must hide these ids before the old points are physically gone.
    excluded_file_ids: list[str] | None = None

    def __post_init__(self) -> None:
        if self.file_ids is not None and not self.file_ids:
            raise ValueError(
                "SearchScope(file_ids=[]) is not a scope that matches nothing — "
                "it is a bug. A lookup that resolved no files must not search."
            )

    def is_empty(self) -> bool:
        return (
            self.file_ids is None
            and self.artifact is None
            and self.user is None
            and self.excluded_file_ids is None
        )


def _as_filter(scope: SearchScope | None) -> qmodels.Filter | None:
    """`SearchScope` to a Qdrant filter, or None when nothing is scoped."""
    if scope is None or scope.is_empty():
        return None
    must: list[qmodels.FieldCondition] = []
    must_not: list[qmodels.FieldCondition] = []
    if scope.file_ids:
        must.append(qmodels.FieldCondition(
            key="file_id", match=qmodels.MatchAny(any=list(scope.file_ids)),
        ))
    if scope.artifact:
        must.append(qmodels.FieldCondition(
            key="artifact", match=qmodels.MatchValue(value=scope.artifact),
        ))
    if scope.user:
        must.append(qmodels.FieldCondition(
            key="user", match=qmodels.MatchValue(value=scope.user),
        ))
    if scope.excluded_file_ids:
        must_not.append(qmodels.FieldCondition(
            key="file_id",
            match=qmodels.MatchAny(any=list(scope.excluded_file_ids)),
        ))
    return qmodels.Filter(must=must, must_not=must_not)


def point_id(*, source: str, kind: str, idx: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{source}:{kind}:{idx}"))


class QdrantKB:
    """Thin async wrapper around qdrant-client's sync surface."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        text_collection: str = "kb_text",
        image_collection: str = "kb_images",
    ) -> None:
        self._client = QdrantClient(host=host, port=port)
        self.text_collection = text_collection
        self.image_collection = image_collection
        self._sparse_cache: dict[str, bool] = {}

    async def ensure_collections(self) -> None:
        await asyncio.to_thread(self._ensure_sync)

    async def probe(self) -> None:
        """Perform the same lightweight service read used by startup checks."""
        await asyncio.to_thread(self._client.get_collections)


    def _ensure_sync(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.text_collection not in existing:
            self._create_collection_sync(self.text_collection, dim=TEXT_DIM)
        if self.image_collection not in existing:
            self._create_collection_sync(self.image_collection, dim=IMAGE_DIM)

    async def ensure_collection(self, name: str, *, dim: int) -> None:
        """Create a named collection if missing. Used for per-user kb_user_* collections."""
        await asyncio.to_thread(self._ensure_named_sync, name, dim)

    def _ensure_named_sync(self, name: str, dim: int) -> None:
        if not self._client.collection_exists(name):
            self._create_collection_sync(name, dim=dim)

    def _create_collection_sync(self, name: str, *, dim: int) -> None:
        # New text collections are born with sparse config, so only the ones
        # that predate Phase 39 ever need migrating. A user who uploads for the
        # first time after this ships never enters the migration path at all.
        sparse = _SPARSE_CONFIG if dim == TEXT_DIM else None
        self._client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
            sparse_vectors_config=sparse,
        )
        log.info(
            "qdrant: created collection %s (dim=%d%s)",
            name, dim, ", sparse=bm25" if sparse else "",
        )

    async def has_sparse(self, collection: str) -> bool:
        """Whether `collection` can accept BM25 vectors, cached after first look.

        Load-bearing during the migration window. Upserting a sparse vector
        into a collection that has no sparse config is a hard `400 Not
        existing vector name`, so ingest has to know — and the honest answer
        differs per collection while the migration runs.

        Caching is safe in one direction only, which is the direction that
        happens: a collection gains sparse config once, at creation or
        migration, and never loses it. A `False` is therefore re-checked on
        every call and a `True` is remembered.
        """
        if self._sparse_cache.get(collection):
            return True
        present = await asyncio.to_thread(self._has_sparse_sync, collection)
        if present:
            self._sparse_cache[collection] = True
        return present

    def _has_sparse_sync(self, collection: str) -> bool:
        try:
            info = self._client.get_collection(collection)
        except Exception as e:  # noqa: BLE001 — a missing collection is a legitimate False
            log.debug("qdrant: has_sparse(%s) could not read config: %s", collection, e)
            return False
        return SPARSE_NAME in (info.config.params.sparse_vectors or {})

    async def collection_exists(self, name: str) -> bool:
        return await asyncio.to_thread(self._collection_exists_sync, name)

    def _collection_exists_sync(self, name: str) -> bool:
        # qdrant-client >=1.7 exposes `collection_exists(name)` which hits
        # `/collections/{name}/exists` — O(1) on the server, vs the previous
        # `get_collections()` round-trip that returned the full list. Called
        # on every user-scoped KB search via `_search_text_merged`, so the
        # difference matters on hot paths.
        return bool(self._client.collection_exists(name))

    async def list_collections(self) -> list[str]:
        """Return the names of every collection on this Qdrant instance.

        Public surface so callers don't reach into `_client` to enumerate
        collections — used by the uploads startup reconcile to find every
        `kb_user_*` collection without round-tripping through every known
        prefix.
        """
        return await asyncio.to_thread(self._list_collections_sync)

    def _list_collections_sync(self) -> list[str]:
        return [c.name for c in self._client.get_collections().collections]

    async def scroll_collection(
        self,
        collection: str,
        *,
        page_size: int = 256,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Walk every point in `collection` and return `[(point_id, payload), ...]`.

        Loads the whole collection into memory; intended for offline /
        admin paths (reconcile sweeps, uploads-side startup reconcile,
        per-user file lists). Don't use this on the chat hot path — use
        `search_text` / `search_images` instead.

        Returns an empty list if the collection doesn't exist; matches
        the behavior callers want for "scroll whatever's there."
        """
        if not await self.collection_exists(collection):
            return []
        return await asyncio.to_thread(self._scroll_collection_sync, collection, page_size)

    def _scroll_collection_sync(
        self, collection: str, page_size: int,
    ) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []
        next_page: Any = None
        while True:
            points, next_page = self._client.scroll(
                collection_name=collection,
                limit=page_size,
                offset=next_page,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                out.append((str(p.id), p.payload or {}))
            if next_page is None:
                break
        return out

    async def upsert_text(
        self, points: list[qmodels.PointStruct], *, collection: str | None = None,
    ) -> None:
        if not points:
            return
        await asyncio.to_thread(
            self._client.upsert,
            collection_name=collection or self.text_collection,
            points=points,
            wait=True,
        )

    async def upsert_images(
        self, points: list[qmodels.PointStruct], *, collection: str | None = None,
    ) -> None:
        if not points:
            return
        await asyncio.to_thread(
            self._client.upsert,
            collection_name=collection or self.image_collection,
            points=points,
            wait=True,
        )

    async def delete_by_source(self, source: str, *, collection: str) -> None:
        """Remove every point whose payload.source equals `source`.

        qdrant-client's `delete()` returns an `UpdateResult` with no count
        field, so no count is reported back. A source with zero matching
        points is a no-op, not an error.
        """
        flt = qmodels.Filter(
            must=[qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source))]
        )
        await asyncio.to_thread(
            self._client.delete,
            collection_name=collection,
            points_selector=qmodels.FilterSelector(filter=flt),
            wait=True,
        )

    async def search_text(
        self, vector: list[float], *, top_k: int = 5, collection: str | None = None,
        scope: SearchScope | None = None,
    ) -> list[KBHit]:
        return await self._search(
            vector, collection or self.text_collection, top_k=top_k, scope=scope,
        )

    async def search_images(
        self, vector: list[float], *, top_k: int = 5, collection: str | None = None,
        scope: SearchScope | None = None,
    ) -> list[KBHit]:
        return await self._search(
            vector, collection or self.image_collection, top_k=top_k, scope=scope,
        )

    async def search_hybrid(
        self, vector: list[float], query: str, *, top_k: int = 5,
        collection: str | None = None, scope: SearchScope | None = None,
    ) -> tuple[list[KBHit], list[KBHit]]:
        """Both retrievers over one collection under one scope. Returns (dense, lexical).

        **This exists so the scope cannot reach one retriever and not the
        other.** Phase 39 made the query path hybrid, and phase 40's plan named
        the resulting hazard precisely: filter the dense side only and BM25
        goes on returning chunks from every other file, reciprocal-rank fusion
        interleaves them, and the answer is confident, sourced, and partly
        drawn from the wrong document. Nothing errors and nothing logs.

        The first version of this took `scope=` at both call sites in
        `routes/kb.py` and pinned the pairing with a test. A test proves today's
        code passes both; it does not stop tomorrow's edit from adding a third
        call site with one argument missing. Taking the scope once, here, is
        what actually removes the failure — the caller has no way to express
        the broken state.

        Returning a tuple rather than a fused list is deliberate: fusion and
        the evidence rule are retrieval *policy* and belong with the route that
        owns the config for them, not in the storage wrapper.
        """
        return await asyncio.gather(
            self.search_text(vector, top_k=top_k, collection=collection, scope=scope),
            self.search_lexical(query, top_k=top_k, collection=collection, scope=scope),
        )

    async def search_lexical(
        self, query: str, *, top_k: int = 5, collection: str | None = None,
        scope: SearchScope | None = None,
    ) -> list[KBHit]:
        """BM25 search over `collection`, or an empty list if it has none.

        The empty list is the whole degradation story. A collection that
        predates the migration, or a query whose terms are all punctuation,
        returns nothing here and the fused result is simply the dense list —
        which is exactly the behaviour before this phase. There is no error
        path for "lexical is unavailable" because there does not need to be.

        On the hybrid path, prefer `search_hybrid` — it takes the scope once
        and fans out, so the dense and lexical sides cannot receive different
        filters. This method stays public for the callers that genuinely want
        one retriever.
        """
        target = collection or self.text_collection
        indices, values = bm25.query_vector(query)
        if not indices or not await self.has_sparse(target):
            return []
        return await asyncio.to_thread(
            self._search_lexical_sync, indices, values, target, top_k, _as_filter(scope),
        )

    def _search_lexical_sync(
        self, indices: list[int], values: list[float], collection: str, top_k: int,
        query_filter: qmodels.Filter | None = None,
    ) -> list[KBHit]:
        result = self._client.query_points(
            collection_name=collection,
            query=qmodels.SparseVector(indices=indices, values=values),
            using=SPARSE_NAME,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return [_to_hit(h) for h in getattr(result, "points", result)]

    async def _search(
        self, vector: list[float], collection: str, *, top_k: int,
        scope: SearchScope | None = None,
    ) -> list[KBHit]:
        # qdrant-client 1.12 deprecated `.search()` in favor of `.query_points()`,
        # which returns a `QueryResponse` wrapping the same `ScoredPoint` list.
        result = await asyncio.to_thread(
            self._client.query_points,
            collection_name=collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            query_filter=_as_filter(scope),
        )
        return [_to_hit(h) for h in getattr(result, "points", result)]

    async def delete_by_file_id(
        self, file_id: str, *, user: str, collection: str,
    ) -> None:
        """Delete every point matching both file_id AND user. Used by /v1/files DELETE.

        The `user` clause is load-bearing — never allow deletion scoped by
        file_id alone. (The file_id UUIDs are unguessable, but belt-and-
        suspenders: two users can't collide on a UUID, but if the API ever
        leaks an id to the wrong user, this filter prevents cross-scope delete.)

        Missing collection = no-op. The /v1/files delete route hits both the
        text and image collections without knowing which holds the file, and
        a user who has only uploaded one kind won't have the other collection.
        """
        if not await self.collection_exists(collection):
            return
        flt = qmodels.Filter(must=[
            qmodels.FieldCondition(key="file_id", match=qmodels.MatchValue(value=file_id)),
            qmodels.FieldCondition(key="user", match=qmodels.MatchValue(value=user)),
        ])
        await asyncio.to_thread(
            self._client.delete,
            collection_name=collection,
            points_selector=qmodels.FilterSelector(filter=flt),
            wait=True,
        )

    async def ensure_user_payload_indexes(self, collection: str) -> None:
        """Create `user`, `file_id` and `artifact` keyword indexes.

        `artifact` joined the list in phase 40, when the search filter started
        using it. Every caller runs this on upload and on ingest-result, and
        `create_payload_index` is idempotent, so existing collections pick the
        new index up on their next write rather than needing a migration.
        """
        await asyncio.to_thread(self._ensure_user_indexes_sync, collection)

    def _ensure_user_indexes_sync(self, collection: str) -> None:
        for field in ("user", "file_id", "artifact"):
            try:
                self._client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except UnexpectedResponse as e:
                # Qdrant returns a 4xx with "already exists" in the body when
                # the index is already present. That's the expected idempotent
                # path; anything else (5xx, schema mismatch, transport) is a
                # real failure and must surface.
                body = (e.content or b"").decode("utf-8", errors="replace").lower()
                status = e.status_code or 0
                if status < 500 and "exist" in body:
                    log.debug("qdrant: payload index %s.%s already present", collection, field)
                    continue
                raise

    async def list_user_files(
        self, *, user: str, collection: str,
    ) -> list[dict[str, Any]]:
        """Return one row per file_id in `collection` for this user.

        Scrolls every point, groups by file_id, and reports first-seen
        metadata + chunk count. For the upload UI's file list — not a
        hot path. 10k chunks is fine; a scanner PDF could blow up here,
        but the upload cap already bounds it.
        """
        return await asyncio.to_thread(self._list_user_files_sync, user, collection)

    def _list_user_files_sync(self, user: str, collection: str) -> list[dict[str, Any]]:
        if not self._client.collection_exists(collection):
            return []
        flt = qmodels.Filter(
            must=[qmodels.FieldCondition(key="user", match=qmodels.MatchValue(value=user))]
        )
        by_file: dict[str, dict[str, Any]] = {}
        next_page: Any = None
        while True:
            points, next_page = self._client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=256,
                offset=next_page,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                fid = str(payload.get("file_id") or "")
                if not fid:
                    continue
                row = by_file.setdefault(fid, {
                    "file_id": fid,
                    "filename": str(payload.get("filename") or ""),
                    "mime": str(payload.get("mime") or ""),
                    "bytes": int(payload.get("bytes") or 0),
                    "uploaded_at": str(payload.get("uploaded_at") or ""),
                    "chunks": 0,
                })
                row["chunks"] += 1
            if next_page is None:
                break
        return sorted(by_file.values(), key=lambda r: r["uploaded_at"], reverse=True)

    async def counts(self) -> dict[str, int]:
        return await asyncio.to_thread(self._counts_sync)

    def _counts_sync(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name in (self.text_collection, self.image_collection):
            try:
                info = self._client.count(collection_name=name, exact=True)
                out[name] = int(info.count)
            except Exception as e:  # noqa: BLE001 — count is best-effort
                log.warning("qdrant: count(%s) failed: %s", name, e)
                out[name] = -1
        return out

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001, S110 — shutdown path; logging would be noise
            pass


def _to_hit(scored: Any) -> KBHit:
    """One `ScoredPoint` to one `KBHit`, shared by the dense and lexical paths.

    Both retrievers must produce identically-shaped hits or the fusion in
    `kb/fusion.py` cannot key them together — it matches on
    `(source, chunk_idx)`, which comes from here.
    """
    p = scored.payload or {}
    return KBHit(
        score=float(scored.score),
        source=str(p.get("source", "")),
        kind=str(p.get("kind", "")),
        chunk_idx=int(p.get("chunk_idx", 0)),
        text=str(p.get("text") or p.get("caption") or ""),
        payload=p,
    )


_RESERVED_PAYLOAD_KEYS = frozenset({"source", "kind", "text", "caption", "chunk_idx", "mtime"})


def _check_extras(extra: dict[str, Any] | None) -> None:
    if not extra:
        return
    clobber = _RESERVED_PAYLOAD_KEYS & extra.keys()
    if clobber:
        raise ValueError(
            f"build_*_point extras cannot override reserved payload keys: {sorted(clobber)}"
        )


def build_text_point(
    *,
    source: str,
    chunk_idx: int,
    text: str,
    vector: list[float],
    mtime: float,
    extra: dict[str, Any] | None = None,
    sparse: bool = False,
) -> qmodels.PointStruct:
    """One text chunk as a Qdrant point.

    `sparse` adds the BM25 vector alongside the dense one. It defaults to
    False because the caller is the only thing that knows whether the target
    collection has sparse config — writing a named vector into a collection
    that lacks it is a hard `400`, and during the migration window some
    collections have it and some do not. `QdrantKB.has_sparse` is the answer.

    The dense vector stays under `""`, its real name for an unnamed vector, so
    every existing dense read keeps working with no `using=`.
    """
    _check_extras(extra)
    payload: dict[str, Any] = {
        "source": source,
        "kind": "text",
        "text": text,
        "chunk_idx": chunk_idx,
        "mtime": float(mtime),
    }
    if extra:
        payload.update(extra)
    vectors: Any = vector
    if sparse:
        indices, values = bm25.document_vector(text)
        vectors = {
            "": vector,
            SPARSE_NAME: qmodels.SparseVector(indices=indices, values=values),
        }
    return qmodels.PointStruct(
        id=point_id(source=source, kind="text", idx=chunk_idx),
        vector=vectors,
        payload=payload,
    )


def build_image_point(
    *,
    source: str,
    chunk_idx: int,
    caption: str,
    vector: list[float],
    mtime: float,
    extra: dict[str, Any] | None = None,
) -> qmodels.PointStruct:
    _check_extras(extra)
    payload: dict[str, Any] = {
        "source": source,
        "kind": "image",
        "caption": caption,
        "chunk_idx": chunk_idx,
        "mtime": float(mtime),
    }
    if extra:
        payload.update(extra)
    return qmodels.PointStruct(
        id=point_id(source=source, kind="image", idx=chunk_idx),
        vector=vector,
        payload=payload,
    )


def normalize_source(path: str | Path) -> str:
    return str(Path(path).resolve())


__all__ = [
    "QdrantKB", "KBHit", "SearchScope", "TEXT_DIM", "IMAGE_DIM", "SPARSE_NAME",
    "build_text_point", "build_image_point", "normalize_source", "point_id",
]
