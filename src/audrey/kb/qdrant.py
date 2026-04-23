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
            self._client.create_collection(
                collection_name=self.text_collection,
                vectors_config=qmodels.VectorParams(size=TEXT_DIM, distance=qmodels.Distance.COSINE),
            )
            log.info("qdrant: created collection %s (dim=%d)", self.text_collection, TEXT_DIM)
        if self.image_collection not in existing:
            self._client.create_collection(
                collection_name=self.image_collection,
                vectors_config=qmodels.VectorParams(size=IMAGE_DIM, distance=qmodels.Distance.COSINE),
            )
            log.info("qdrant: created collection %s (dim=%d)", self.image_collection, IMAGE_DIM)

    async def upsert_text(self, points: list[qmodels.PointStruct]) -> None:
        if not points:
            return
        await asyncio.to_thread(
            self._client.upsert,
            collection_name=self.text_collection,
            points=points,
            wait=True,
        )

    async def upsert_images(self, points: list[qmodels.PointStruct]) -> None:
        if not points:
            return
        await asyncio.to_thread(
            self._client.upsert,
            collection_name=self.image_collection,
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

    async def search_text(self, vector: list[float], *, top_k: int = 5) -> list[KBHit]:
        return await self._search(vector, self.text_collection, top_k=top_k)

    async def search_images(self, vector: list[float], *, top_k: int = 5) -> list[KBHit]:
        return await self._search(vector, self.image_collection, top_k=top_k)

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
