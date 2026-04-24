"""Dataset crawl + ingest.

`ingest_path(path)` walks a directory, loads each supported file,
chunks it, embeds the chunks, and upserts them into Qdrant. Point IDs
are deterministic (`point_id(source, kind, idx)` in `qdrant.py`), so
re-running ingest is idempotent: unchanged chunks overwrite themselves
with identical vectors, changed chunks replace their old points.

If a file had N chunks previously and now has M < N, the tail chunks
(M..N-1) are orphaned. `ingest_file` issues a `delete_by_source` for
anything over the current chunk count before the upsert so the index
stays clean.

Images are embedded as a single point per file (CLIP produces one
vector for the whole image); no chunking.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client.http import models as qmodels

from audrey.kb.chunk import IMAGE_SUFFIXES, TEXT_SUFFIXES, Chunk, chunk_text, load_text
from audrey.kb.embed import ImageEmbedder, TextEmbedder
from audrey.kb.extract import extract_text
from audrey.kb.qdrant import QdrantKB, build_image_point, build_text_point, normalize_source

log = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestStats:
    files_seen: int = 0
    files_text: int = 0
    files_image: int = 0
    files_skipped: int = 0
    chunks_text: int = 0
    chunks_image: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, int | list[str]]:
        return {
            "files_seen": self.files_seen,
            "files_text": self.files_text,
            "files_image": self.files_image,
            "files_skipped": self.files_skipped,
            "chunks_text": self.chunks_text,
            "chunks_image": self.chunks_image,
            "errors": list(self.errors),
        }


async def ingest_path(
    root: Path,
    *,
    qdrant: QdrantKB,
    text_embedder: TextEmbedder,
    image_embedder: ImageEmbedder | None,
    chunk_tokens: int = 1000,
    overlap_tokens: int = 100,
) -> IngestStats:
    """Recursively ingest every supported file under `root`."""
    stats = IngestStats()
    if not root.exists():
        stats.errors.append(f"root does not exist: {root}")
        return stats

    for path in sorted(_iter_files(root)):
        stats.files_seen += 1
        suffix = path.suffix.lower()
        try:
            if suffix in TEXT_SUFFIXES or suffix in {".pdf", ".docx", ".html", ".htm"}:
                n = await ingest_text_file(
                    path, qdrant=qdrant, embedder=text_embedder,
                    chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
                )
                stats.files_text += 1
                stats.chunks_text += n
            elif suffix in IMAGE_SUFFIXES and image_embedder is not None:
                ok = await ingest_image_file(path, qdrant=qdrant, embedder=image_embedder)
                if ok:
                    stats.files_image += 1
                    stats.chunks_image += 1
                else:
                    stats.files_skipped += 1
            else:
                stats.files_skipped += 1
        except Exception as e:  # noqa: BLE001 — keep the crawl going
            log.warning("kb.ingest: %s failed: %s", path, e)
            stats.errors.append(f"{path}: {e}")
    log.info(
        "kb.ingest: root=%s seen=%d text=%d(%d chunks) images=%d skipped=%d errors=%d",
        root, stats.files_seen, stats.files_text, stats.chunks_text,
        stats.files_image, stats.files_skipped, len(stats.errors),
    )
    return stats


async def ingest_text_file(
    path: Path,
    *,
    qdrant: QdrantKB,
    embedder: TextEmbedder,
    chunk_tokens: int,
    overlap_tokens: int,
) -> int:
    raw = load_text(path)
    if not raw:
        return 0
    chunks: list[Chunk] = chunk_text(raw, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        return 0
    source = normalize_source(path)
    mtime = path.stat().st_mtime
    vectors = await embedder.embed_many([c.text for c in chunks])
    # Clear any stale points from a previous larger ingest; the upsert
    # below rewrites the current range with deterministic IDs.
    await qdrant.delete_by_source(source, collection=qdrant.text_collection)
    points: list[qmodels.PointStruct] = [
        build_text_point(
            source=source, chunk_idx=c.idx, text=c.text,
            vector=v, mtime=mtime,
        )
        for c, v in zip(chunks, vectors, strict=True)
    ]
    await qdrant.upsert_text(points)
    return len(points)


async def ingest_image_file(
    path: Path,
    *,
    qdrant: QdrantKB,
    embedder: ImageEmbedder,
) -> bool:
    source = normalize_source(path)
    mtime = path.stat().st_mtime
    try:
        vec = await embedder.embed_path(path)
    except Exception as e:  # noqa: BLE001 — bad image shouldn't poison the crawl
        log.warning("kb.ingest: image %s failed: %s", path, e)
        return False
    point = build_image_point(
        source=source, chunk_idx=0, caption=path.name,
        vector=vec, mtime=mtime,
    )
    await qdrant.upsert_images([point])
    return True


def _iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            yield p


async def ingest_many(
    roots: list[Path],
    *,
    qdrant: QdrantKB,
    text_embedder: TextEmbedder,
    image_embedder: ImageEmbedder | None,
    chunk_tokens: int = 1000,
    overlap_tokens: int = 100,
) -> IngestStats:
    """Ingest every root path sequentially, merging stats."""
    merged = IngestStats()
    for root in roots:
        s = await ingest_path(
            root, qdrant=qdrant, text_embedder=text_embedder,
            image_embedder=image_embedder,
            chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
        )
        merged.files_seen += s.files_seen
        merged.files_text += s.files_text
        merged.files_image += s.files_image
        merged.files_skipped += s.files_skipped
        merged.chunks_text += s.chunks_text
        merged.chunks_image += s.chunks_image
        merged.errors.extend(s.errors)
    return merged


async def ingest_user_text_file(
    path: Path,
    *,
    qdrant: QdrantKB,
    embedder: TextEmbedder,
    collection: str,
    user: str,
    file_id: str,
    filename: str,
    mime: str,
    uploaded_at: str | None = None,
    chunk_tokens: int = 1000,
    overlap_tokens: int = 100,
) -> int:
    """Ingest a single uploaded file into a user-scoped text collection.

    Mirrors `ingest_text_file` but writes to `collection` (e.g.
    `kb_user_text_bart_proton_me`) with user/file metadata in the payload.
    Delete-before-upsert clears any prior points for the same file_id.
    """
    raw = extract_text(path)  # raises EmptyExtractionError on scanned PDFs etc.
    chunks: list[Chunk] = chunk_text(raw, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    if not chunks:
        return 0

    source = normalize_source(path)
    mtime = path.stat().st_mtime
    size_bytes = path.stat().st_size
    stamp = uploaded_at or _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    vectors = await embedder.embed_many([c.text for c in chunks])

    await qdrant.delete_by_file_id(file_id, user=user, collection=collection)

    extras = {
        "user": user,
        "file_id": file_id,
        "filename": filename,
        "mime": mime,
        "bytes": int(size_bytes),
        "uploaded_at": stamp,
    }
    points: list[qmodels.PointStruct] = [
        build_text_point(
            source=source, chunk_idx=c.idx, text=c.text,
            vector=v, mtime=mtime, extra=extras,
        )
        for c, v in zip(chunks, vectors, strict=True)
    ]
    await qdrant.upsert_text(points, collection=collection)
    return len(points)


async def ingest_user_image_file(
    path: Path,
    *,
    qdrant: QdrantKB,
    embedder: ImageEmbedder,
    collection: str,
    user: str,
    file_id: str,
    filename: str,
    mime: str,
    uploaded_at: str | None = None,
) -> bool:
    source = normalize_source(path)
    mtime = path.stat().st_mtime
    size_bytes = path.stat().st_size
    stamp = uploaded_at or _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    try:
        vec = await embedder.embed_path(path)
    except Exception as e:  # noqa: BLE001
        log.warning("kb.ingest: user image %s failed: %s", path, e)
        return False

    await qdrant.delete_by_file_id(file_id, user=user, collection=collection)

    extras = {
        "user": user,
        "file_id": file_id,
        "filename": filename,
        "mime": mime,
        "bytes": int(size_bytes),
        "uploaded_at": stamp,
    }
    point = build_image_point(
        source=source, chunk_idx=0, caption=filename,
        vector=vec, mtime=mtime, extra=extras,
    )
    await qdrant.upsert_images([point], collection=collection)
    return True


__all__ = [
    "IngestStats", "ingest_path", "ingest_many",
    "ingest_text_file", "ingest_image_file",
    "ingest_user_text_file", "ingest_user_image_file",
]
