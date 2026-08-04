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

from audrey.kb.chunk import (
    IMAGE_SUFFIXES,
    TEXT_SUFFIXES,
    Chunk,
    chunk_segments,
    chunk_text,
    load_text,
)
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
    sparse = await qdrant.has_sparse(qdrant.text_collection)
    points: list[qmodels.PointStruct] = [
        build_text_point(
            source=source, chunk_idx=c.idx, text=c.text,
            vector=v, mtime=mtime, sparse=sparse,
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
        if not p.is_file():
            continue
        # Skip the file if any path component (relative to root) starts with
        # a dot — so a stray `.git/` or `.cache/` under a topic dir doesn't
        # have its non-dot children ingested.
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
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
    sparse = await qdrant.has_sparse(collection)
    points: list[qmodels.PointStruct] = [
        build_text_point(
            source=source, chunk_idx=c.idx, text=c.text,
            vector=v, mtime=mtime, extra=extras, sparse=sparse,
        )
        for c, v in zip(chunks, vectors, strict=True)
    ]
    await qdrant.upsert_text(points, collection=collection)
    return len(points)


async def ingest_transcript_segments(
    segments: list[dict],
    *,
    sidecar: Path,
    qdrant: QdrantKB,
    embedder: TextEmbedder,
    collection: str,
    user: str,
    file_id: str,
    filename: str,
    mime: str,
    source_bytes: int,
    uploaded_at: str | None = None,
    chunk_tokens: int = 250,
    overlap_tokens: int = 40,
    delete_existing: bool = True,
) -> int:
    """Ingest whisper segments as timestamped chunks.

    Separate from `ingest_user_text_file` because a transcript is not a
    document. It arrives already split on natural pauses, it carries a
    timestamp per line that must not be embedded, and it needs much smaller
    chunks — see `chunk_segments` for the measurements that forced all three.

    `sidecar` is the human-readable `[HH:MM:SS] line` file. It stays the
    identity anchor — `source` in the payload, and what `delete_by_file_id`
    and the reconcile sweep key on — but its *contents* are never what gets
    embedded. Chunk text is built from `segments` without the timestamps.

    Delete-before-upsert, matching `ingest_user_text_file`, so a re-run
    replaces its predecessor rather than doubling it.

    `delete_existing=False` when the caller has already cleared this
    `file_id`. Phase 36 needs that: `delete_by_file_id` removes *every* point
    for the file, so a transcript ingest running after a frame ingest would
    delete the frames it was supposed to sit alongside. The route clears once
    and both artifacts write into the space it made.
    """
    chunks = chunk_segments(
        segments, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
    )
    if not chunks:
        return 0

    source = normalize_source(sidecar)
    # One stat, not one per chunk — it was inside the comprehension below.
    stat = sidecar.stat()
    stamp = uploaded_at or _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    vectors = await embedder.embed_many([c.text for c in chunks])

    if delete_existing:
        await qdrant.delete_by_file_id(file_id, user=user, collection=collection)
    sparse = await qdrant.has_sparse(collection)

    points: list[qmodels.PointStruct] = [
        build_text_point(
            source=source, chunk_idx=c.idx, text=c.text, vector=v,
            mtime=stat.st_mtime, sparse=sparse,
            extra={
                "user": user,
                "file_id": file_id,
                "filename": filename,
                "mime": mime,
                # The SOURCE video's size, not the sidecar's. Every payload
                # field here describes the uploaded file this `file_id` names,
                # and `reconcile_with_qdrant` rebuilds the uploads row from
                # exactly these on every boot. Using the transcript's own size
                # billed a 288 MB video as 9 KB against a 1 GiB quota — and
                # did it silently, one boot after the ingest.
                "bytes": int(source_bytes),
                "uploaded_at": stamp,
                # The timestamps ride in the payload so a hit can say *where*
                # in the video it was said, without those characters diluting
                # the vector that found it.
                "t_start": c.t_start,
                "t_end": c.t_end,
                "artifact": "transcript",
            },
        )
        for c, v in zip(chunks, vectors, strict=True)
    ]
    await qdrant.upsert_text(points, collection=collection)
    log.info(
        "ingest: transcript %s -> %d chunks (%d segments, %d tokens/chunk)",
        file_id, len(points), len(segments), chunk_tokens,
    )
    return len(points)


async def ingest_summary(
    summary: str,
    *,
    sidecar: Path,
    qdrant: QdrantKB,
    embedder: TextEmbedder,
    collection: str,
    user: str,
    file_id: str,
    filename: str,
    mime: str,
    source_bytes: int,
    uploaded_at: str | None = None,
) -> int:
    """Ingest a video's summary as one searchable chunk (Phase 37).

    Stored on the row, it answers "what is this video" in the file list.
    Ingested here, it answers the same question in chat without pulling two
    hundred transcript chunks into context. One extra chunk per video is a
    rounding error against the transcript it summarises.

    Deliberately **not** chunked. A summary that needed splitting would no
    longer be a summary, and its value in retrieval is that the whole thing
    fits in one hit — a half-summary answers nothing.

    Its own sidecar name for the same reason as the frames: `point_id` is
    `(source, kind, chunk_idx)`, so sharing a source with either of the other
    artifacts would collide on chunk 0.
    """
    text = summary.strip()
    if not text:
        return 0

    sidecar.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(sidecar.write_text, text, "utf-8")
    source = normalize_source(sidecar)
    stat = sidecar.stat()
    stamp = uploaded_at or _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    vectors = await embedder.embed_many([text])

    point = build_text_point(
        source=source, chunk_idx=0, text=text, vector=vectors[0],
        mtime=stat.st_mtime, sparse=await qdrant.has_sparse(collection),
        extra={
            "user": user,
            "file_id": file_id,
            "filename": filename,
            "mime": mime,
            "bytes": int(source_bytes),
            "uploaded_at": stamp,
            "t_start": 0.0,
            "t_end": 0.0,
            "artifact": "summary",
        },
    )
    await qdrant.upsert_text([point], collection=collection)
    log.info("ingest: summary %s -> 1 chunk (%d chars)", file_id, len(text))
    return 1


async def ingest_frame_descriptions(
    frames: list[dict],
    *,
    sidecar: Path,
    qdrant: QdrantKB,
    embedder: TextEmbedder,
    collection: str,
    user: str,
    file_id: str,
    filename: str,
    mime: str,
    source_bytes: int,
    uploaded_at: str | None = None,
    chunk_tokens: int = 250,
    overlap_tokens: int = 40,
    delete_existing: bool = False,
) -> int:
    """Ingest keyframe descriptions as timestamped chunks (Phase 36).

    Each `frame` is `{t_start, t_end, text}` — the prose a `vl` model produced
    for one keyframe, and the span of video that keyframe stands in for.

    **Chunked per frame, not across frames.** Two descriptions are about two
    different moments, so letting a chunk straddle them would produce text
    that was never true of either and attach it to whichever timestamp came
    first. Within one description the ordinary chunker applies: a dense slide
    transcribed verbatim can be long, and a 250-token limit keeps these the
    same size as transcript chunks so neither artifact outranks the other for
    reasons of length alone.

    Its own `sidecar` name, distinct from the transcript's. Both artifacts
    live under one `file_id` in one collection, and `point_id` is derived from
    `(source, kind, chunk_idx)` — so sharing a source would make frame chunk 0
    and transcript chunk 0 the same point, each silently overwriting the
    other.

    `delete_existing` defaults to **False**, the opposite of the transcript
    path. `delete_by_file_id` removes every point for the file including the
    transcript, and in the one caller that matters the route has already
    cleared the file_id once for both artifacts.
    """
    prepared: list[tuple[Chunk, dict]] = []
    for frame in frames:
        text = str(frame.get("text") or "").strip()
        if not text:
            continue
        for piece in chunk_text(
            text, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens,
        ):
            prepared.append((piece, frame))
    if not prepared:
        return 0

    source = normalize_source(sidecar)
    stat = sidecar.stat()
    stamp = uploaded_at or _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    vectors = await embedder.embed_many([c.text for c, _ in prepared])

    if delete_existing:
        await qdrant.delete_by_file_id(file_id, user=user, collection=collection)
    sparse = await qdrant.has_sparse(collection)

    points: list[qmodels.PointStruct] = [
        build_text_point(
            # Numbered across the whole set, not per frame — `chunk_idx` is
            # half of the point id, so restarting at 0 for each frame would
            # collapse every frame's first chunk onto one point.
            source=source, chunk_idx=idx, text=c.text, vector=v,
            mtime=stat.st_mtime, sparse=sparse,
            extra={
                "user": user,
                "file_id": file_id,
                "filename": filename,
                "mime": mime,
                "bytes": int(source_bytes),
                "uploaded_at": stamp,
                "t_start": float(frame.get("t_start") or 0.0),
                "t_end": float(frame.get("t_end") or 0.0),
                # The discriminator that lets a caller tell "this was said"
                # from "this was shown" — the two answer different questions
                # about the same second of video.
                "artifact": "visual",
            },
        )
        for idx, ((c, frame), v) in enumerate(zip(prepared, vectors, strict=True))
    ]
    await qdrant.upsert_text(points, collection=collection)
    log.info(
        "ingest: frames %s -> %d chunks from %d descriptions",
        file_id, len(points), len(frames),
    )
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
