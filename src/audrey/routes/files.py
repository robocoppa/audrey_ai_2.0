"""Per-user file uploads.

    POST   /v1/files           — multipart upload; stream to disk, validate,
                                 ingest into the user's kb_user_text_* /
                                 kb_user_images_* collections.
    GET    /v1/files           — list the caller's files (one row per file_id).
    DELETE /v1/files/{file_id} — purge all points for a file + delete bytes.

Identity: every endpoint depends on `require_user`, which proxies the
browser's `Authorization: Bearer <jwt>` to OWUI and returns an `AuthedUser`.
The caller's email *is* the user id — no `?user=` query param, no `X-User`
header, no form field. Spoofing requires forging a token OWUI would accept.

Safety layers (all mandatory):

  - Token validation: `require_user` → 401 on missing/invalid token.
  - Size cap: `kb.max_upload_mb` enforced while streaming (stop + 413 at limit).
  - Mime sniff: libmagic reads the saved bytes — extension is a hint, sniff
    is the gate. Whitelist in `kb.extract.ALLOWED_MIMES`.
  - Per-user byte quota: sum of already-stored `bytes` payload field must be
    under `kb.max_user_bytes` *before* ingest.
  - User isolation: every Qdrant read/write is scoped by both `file_id`
    AND `user` in payload filters. See `QdrantKB.delete_by_file_id`.
  - Filename sanitization: we keep the original filename for display, but
    the bytes land at `<upload_root>/<sanitized_user>/<file_id><ext>` —
    the client-supplied name is never used as a path segment.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from audrey.auth import AuthedUser, require_user
from audrey.kb.extract import (
    ALLOWED_EXTENSIONS,
    ALLOWED_IMAGE_MIMES,
    ALLOWED_TEXT_MIMES,
    EmptyExtractionError,
    UnsupportedMimeError,
    is_image_mime,
    is_text_mime,
    sniff_mime,
)
from audrey.kb.ingest import ingest_user_image_file, ingest_user_text_file
from audrey.kb.qdrant import QdrantKB
from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import (
    ensure_user_collections,
    sanitize_user,
    user_image_collection,
    user_text_collection,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/files", tags=["files"])


class FileRow(BaseModel):
    file_id: str
    filename: str
    mime: str
    bytes: int
    uploaded_at: str
    chunks: int


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    mime: str
    bytes: int
    kind: str                # "text" | "image"
    collection: str
    chunks: int              # text only; 1 for images


class Limits(BaseModel):
    """The gates an upload has to clear, published so the client can pre-check.

    Advisory only. Every value here is re-enforced server-side on the way in
    (`_stream_to_disk` cap, the quota checks, the libmagic sniff); shipping
    them to the browser just means a doomed upload can be refused before the
    bytes go on the wire instead of after.
    """

    max_upload_bytes: int
    max_user_bytes: int
    allowed_extensions: list[str]


class ListResponse(BaseModel):
    user: str
    files: list[FileRow]
    total_bytes: int
    limits: Limits


class DeleteResponse(BaseModel):
    file_id: str
    deleted: bool


def _upload_root(request: Request) -> Path:
    cfg = request.app.state.cfg
    kb_cfg = cfg.raw.get("kb", {}) or {}
    root = Path(kb_cfg.get("upload_root", "/data/uploads"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _max_upload_bytes(request: Request) -> int:
    cfg = request.app.state.cfg
    kb_cfg = cfg.raw.get("kb", {}) or {}
    return int(kb_cfg.get("max_upload_mb", 50)) * 1024 * 1024


def _max_user_bytes(request: Request) -> int:
    cfg = request.app.state.cfg
    kb_cfg = cfg.raw.get("kb", {}) or {}
    return int(kb_cfg.get("max_user_bytes", 1024 * 1024 * 1024))


async def _stream_to_disk(upload: UploadFile, dest: Path, *, limit_bytes: int) -> int:
    """Stream upload bytes to disk, stopping at limit_bytes. Returns written size.

    Returns -1 if the cap would be exceeded (caller should 413 and unlink dest).
    Checks the cap *before* extending — a single oversized chunk can't push
    `written` past the limit. Same defense-in-depth pattern we use in
    `kb/embed._fetch_image`.
    """
    written = 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            if written + len(chunk) > limit_bytes:
                return -1
            written += len(chunk)
            f.write(chunk)
    return written


def _get_uploads_db(request: Request) -> UploadsDB:
    db: UploadsDB | None = getattr(request.app.state, "uploads_db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Uploads index is not initialized.")
    return db


@router.post("", response_model=UploadResponse)
async def upload_file(
    request: Request,
    me: AuthedUser = Depends(require_user),
    file: UploadFile = File(...),
) -> UploadResponse:
    """Accept one file, validate, extract, ingest into the caller's user collections."""
    user = me.email

    qdrant: QdrantKB | None = getattr(request.app.state, "qdrant", None)
    text_embedder = getattr(request.app.state, "text_embedder", None)
    image_embedder = getattr(request.app.state, "image_embedder", None)
    if qdrant is None or text_embedder is None:
        raise HTTPException(status_code=503, detail="KB is not initialized.")
    db = _get_uploads_db(request)

    # Ensure user collections + indexes exist before we write.
    text_col, image_col = await ensure_user_collections(qdrant, user)

    max_upload = _max_upload_bytes(request)
    max_total = _max_user_bytes(request)
    root = _upload_root(request)
    slug = sanitize_user(user)
    file_id = str(uuid.uuid4())
    ext = Path(file.filename or "").suffix.lower()
    dest = root / slug / f"{file_id}{ext}"

    # Pre-flight quota check before we touch disk: if the user is already at
    # or over their byte budget, every byte we stream is wasted I/O on what
    # we'll reject anyway. Post-stream check below still runs (we only know
    # the actual upload size after streaming) — this is an additional guard.
    already = await db.user_total_bytes(user)
    if already >= max_total:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Per-user storage quota already at or over the limit: "
                f"{already // (1024 * 1024)}MB >= {max_total // (1024 * 1024)}MB."
            ),
        )

    written = await _stream_to_disk(file, dest, limit_bytes=max_upload)
    if written < 0:
        _safe_unlink(dest)
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {max_upload // (1024 * 1024)} MB limit.",
        )
    if written == 0:
        _safe_unlink(dest)
        raise HTTPException(status_code=422, detail="Empty upload.")

    # Mime gate. Trust the sniffed bytes, not the client-declared type.
    mime = sniff_mime(dest)
    if mime not in (ALLOWED_TEXT_MIMES | ALLOWED_IMAGE_MIMES):
        _safe_unlink(dest)
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported mime: {mime!r}. Allowed: {sorted(ALLOWED_TEXT_MIMES | ALLOWED_IMAGE_MIMES)}",
        )

    # Post-stream quota check — actual upload size only becomes known after
    # streaming. The pre-flight above catches the already-over case at the
    # wire; this catches the case where this upload itself crosses the line.
    if already + written > max_total:
        _safe_unlink(dest)
        raise HTTPException(
            status_code=413,
            detail=(
                f"Per-user storage quota exceeded: "
                f"{(already + written) // (1024 * 1024)}MB > {max_total // (1024 * 1024)}MB."
            ),
        )

    # Strip directory components, then cap at NAME_MAX (255) so a runaway
    # 10 MB filename string can't bloat sqlite or the Qdrant payload.
    filename = Path(file.filename or file_id).name[:255]
    kind = "image" if is_image_mime(mime) else "text"
    # Stamp once here so the qdrant payload + sqlite row agree to the second.
    stamp = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    try:
        if kind == "text":
            if not is_text_mime(mime):
                raise UnsupportedMimeError(f"not a text mime: {mime}")
            n_chunks = await ingest_user_text_file(
                dest, qdrant=qdrant, embedder=text_embedder,
                collection=text_col, user=user, file_id=file_id,
                filename=filename, mime=mime, uploaded_at=stamp,
            )
            collection = text_col
        else:
            if image_embedder is None:
                raise HTTPException(status_code=503, detail="Image embedder not initialized.")
            ok = await ingest_user_image_file(
                dest, qdrant=qdrant, embedder=image_embedder,
                collection=image_col, user=user, file_id=file_id,
                filename=filename, mime=mime, uploaded_at=stamp,
            )
            if not ok:
                raise HTTPException(status_code=422, detail="Image embedding failed.")
            n_chunks = 1
            collection = image_col
    except EmptyExtractionError as e:
        _safe_unlink(dest)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except UnsupportedMimeError as e:
        _safe_unlink(dest)
        raise HTTPException(status_code=415, detail=str(e)) from e
    except HTTPException:
        _safe_unlink(dest)
        raise
    except Exception as e:
        _safe_unlink(dest)
        log.exception("files: ingest failed for %s (%s): %s", filename, user, e)
        raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e

    # Index the upload AFTER qdrant succeeded. If sqlite fails, roll back
    # qdrant so list/quota stay coherent — better to drop the upload than
    # ship a phantom file the user can't see or delete. The rollback itself
    # is wrapped: if Qdrant is the one that's flapping, the second exception
    # would mask the first and could also leave behind points the sqlite
    # row never recorded. Best-effort rollback + log + the next boot's
    # `reconcile_with_qdrant` sweep is the recovery story.
    try:
        await db.record_upload(
            file_id=file_id, user=user, filename=filename, mime=mime,
            bytes_=written, kind=kind, collection=collection,
            chunks=n_chunks, uploaded_at=stamp,
        )
    except Exception as e:
        log.exception("files: uploads_db.record failed for %s (%s): %s", filename, user, e)
        try:
            await qdrant.delete_by_file_id(file_id, user=user, collection=collection)
        except Exception as rollback_err:  # noqa: BLE001 — must not mask the original error
            log.error(
                "files: qdrant rollback ALSO failed for file_id=%s user=%s collection=%s: %s "
                "(orphan points will be cleaned up by next boot's reconcile_with_qdrant)",
                file_id, user, collection, rollback_err,
            )
        _safe_unlink(dest)
        raise HTTPException(status_code=500, detail=f"Index write failed: {e}") from e

    log.info(
        "files: user=%s file_id=%s filename=%r mime=%s bytes=%d kind=%s chunks=%d",
        user, file_id, filename, mime, written, kind, n_chunks,
    )
    return UploadResponse(
        file_id=file_id, filename=filename, mime=mime, bytes=written,
        kind=kind, collection=collection, chunks=n_chunks,
    )


@router.get("", response_model=ListResponse)
async def list_files(
    request: Request, me: AuthedUser = Depends(require_user),
) -> ListResponse:
    user = me.email
    db = _get_uploads_db(request)

    rows = await db.list_user(user)
    files = [FileRow(**{k: row[k] for k in (
        "file_id", "filename", "mime", "bytes", "uploaded_at", "chunks",
    )}) for row in rows]
    total = sum(r.bytes for r in files)
    return ListResponse(
        user=user, files=files, total_bytes=total,
        limits=Limits(
            max_upload_bytes=_max_upload_bytes(request),
            max_user_bytes=_max_user_bytes(request),
            allowed_extensions=sorted(ALLOWED_EXTENSIONS),
        ),
    )


@router.delete("/{file_id}", response_model=DeleteResponse)
async def delete_file(
    file_id: str, request: Request, me: AuthedUser = Depends(require_user),
) -> DeleteResponse:
    user = me.email
    qdrant: QdrantKB | None = getattr(request.app.state, "qdrant", None)
    if qdrant is None:
        raise HTTPException(status_code=503, detail="KB is not initialized.")
    db = _get_uploads_db(request)

    # sqlite first — once the index row is gone, list/quota immediately
    # reflect the delete even if the qdrant calls below take a beat.
    deleted_row = await db.delete_upload(file_id, user=user)

    # Delete from both collections; a given file_id only lives in one, but
    # scoped double-filter on (file_id, user) makes unscoped calls safe.
    text_col = user_text_collection(user)
    image_col = user_image_collection(user)
    await asyncio.gather(
        qdrant.delete_by_file_id(file_id, user=user, collection=text_col),
        qdrant.delete_by_file_id(file_id, user=user, collection=image_col),
    )

    # Best-effort bytes cleanup. We don't know the extension, so glob.
    root = _upload_root(request) / sanitize_user(user)
    for p in root.glob(f"{file_id}.*"):
        _safe_unlink(p)
    bare = root / file_id
    _safe_unlink(bare)

    log.info("files: delete user=%s file_id=%s indexed=%s", user, file_id, deleted_row)
    # `deleted` reflects whether sqlite had a row to remove. The Qdrant
    # delete-by-filter and the disk unlink are best-effort cleanup that
    # both no-op gracefully on missing data, so the sqlite outcome is
    # the honest signal to the caller.
    return DeleteResponse(file_id=file_id, deleted=deleted_row)


def _safe_unlink(p: Path) -> None:
    try:
        p.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        log.warning("files: unlink failed for %s: %s", p, e)


__all__ = ["router"]
