"""Qdrant wrapper for Audrey's knowledge base.

Two collections:
  - `kb_text`   : 768-d (nomic-embed-text via ollama /api/embeddings)
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

log = logging.getLogger(__name__)

TEXT_DIM = 768
IMAGE_DIM = 512
_NAMESPACE = uuid.NAMESPACE_DNS


@dataclass(slots=True)
class KBHit:
    score: float
    source: str
    kind: str
    chunk_idx: int
    text: str
    payload: dict[str, Any]


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

    async def ensure_collections(self) -> None:
        await asyncio.to_thread(self._ensure_sync)

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
        existing = {c.name for c in self._client.get_collections().collections}
        if name not in existing:
            self._create_collection_sync(name, dim=dim)

    def _create_collection_sync(self, name: str, *, dim: int) -> None:
        self._client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )
        log.info("qdrant: created collection %s (dim=%d)", name, dim)

    async def collection_exists(self, name: str) -> bool:
        return await asyncio.to_thread(self._collection_exists_sync, name)

    def _collection_exists_sync(self, name: str) -> bool:
        return name in {c.name for c in self._client.get_collections().collections}

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

    async def delete_by_source(self, source: str, *, collection: str) -> int:
        """Remove every point whose payload.source equals `source`. Returns count deleted (best-effort)."""
        flt = qmodels.Filter(
            must=[qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source))]
        )
        result = await asyncio.to_thread(
            self._client.delete,
            collection_name=collection,
            points_selector=qmodels.FilterSelector(filter=flt),
            wait=True,
        )
        # qdrant-client returns an UpdateResult with a status; count isn't exposed here.
        return 0 if result is None else -1

    async def search_text(
        self, vector: list[float], *, top_k: int = 5, collection: str | None = None,
    ) -> list[KBHit]:
        return await self._search(vector, collection or self.text_collection, top_k=top_k)

    async def search_images(
        self, vector: list[float], *, top_k: int = 5, collection: str | None = None,
    ) -> list[KBHit]:
        return await self._search(vector, collection or self.image_collection, top_k=top_k)

    async def _search(self, vector: list[float], collection: str, *, top_k: int) -> list[KBHit]:
        # qdrant-client 1.12 deprecated `.search()` in favor of `.query_points()`,
        # which returns a `QueryResponse` wrapping the same `ScoredPoint` list.
        result = await asyncio.to_thread(
            self._client.query_points,
            collection_name=collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        hits = getattr(result, "points", result)
        out: list[KBHit] = []
        for h in hits:
            p = h.payload or {}
            out.append(KBHit(
                score=float(h.score),
                source=str(p.get("source", "")),
                kind=str(p.get("kind", "")),
                chunk_idx=int(p.get("chunk_idx", 0)),
                text=str(p.get("text") or p.get("caption") or ""),
                payload=p,
            ))
        return out

    async def delete_by_file_id(
        self, file_id: str, *, user: str, collection: str,
    ) -> None:
        """Delete every point matching both file_id AND user. Used by /v1/files DELETE.

        The `user` clause is load-bearing — never allow deletion scoped by
        file_id alone. (The file_id UUIDs are unguessable, but belt-and-
        suspenders: two users can't collide on a UUID, but if the API ever
        leaks an id to the wrong user, this filter prevents cross-scope delete.)
        """
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
        """Create `user` and `file_id` keyword indexes for a user-scoped collection."""
        await asyncio.to_thread(self._ensure_user_indexes_sync, collection)

    def _ensure_user_indexes_sync(self, collection: str) -> None:
        for field in ("user", "file_id"):
            try:
                self._client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception as e:  # noqa: BLE001 — index may already exist
                log.debug("qdrant: payload index %s.%s already present or failed: %s", collection, field, e)

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
        if collection not in {c.name for c in self._client.get_collections().collections}:
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
        except Exception:  # noqa: BLE001
            pass


def build_text_point(
    *,
    source: str,
    chunk_idx: int,
    text: str,
    vector: list[float],
    mtime: float,
    extra: dict[str, Any] | None = None,
) -> qmodels.PointStruct:
    payload: dict[str, Any] = {
        "source": source,
        "kind": "text",
        "text": text,
        "chunk_idx": chunk_idx,
        "mtime": float(mtime),
    }
    if extra:
        payload.update(extra)
    return qmodels.PointStruct(
        id=point_id(source=source, kind="text", idx=chunk_idx),
        vector=vector,
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
    "QdrantKB", "KBHit", "TEXT_DIM", "IMAGE_DIM",
    "build_text_point", "build_image_point", "normalize_source", "point_id",
]
