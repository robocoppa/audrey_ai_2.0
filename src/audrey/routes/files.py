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
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from audrey.auth import AuthedUser, require_user
from audrey.kb.extract import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIMES,
    EmptyExtractionError,
    UnsupportedMimeError,
    is_image_mime,
    is_text_mime,
    is_video_mime,
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
    # 'ready' | 'pending'. Defaulted so a row written before the column
    # existed still validates after the additive migration.
    status: str = "ready"


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    mime: str
    bytes: int
    kind: str                # "text" | "image" | "video"
    collection: str          # empty for video — nothing is indexed yet
    chunks: int              # text only; 1 for images; 0 for video
    status: str = "ready"    # "ready" | "pending" (video awaits the worker)


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
    # Chunked transport (32b). The client picks single-shot below
    # `max_upload_bytes` and sessions above it, so both numbers have to
    # travel together or it cannot make that call.
    chunked_max_bytes: int
    part_size: int


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

    return await _validate_and_ingest(
        request, dest,
        user=user, file_id=file_id, filename=file.filename or file_id,
        written=written, already=already, max_total=max_total,
        qdrant=qdrant, text_embedder=text_embedder, image_embedder=image_embedder,
        db=db, text_col=text_col, image_col=image_col,
    )


async def _validate_and_ingest(
    request: Request,
    dest: Path,
    *,
    user: str,
    file_id: str,
    filename: str,
    written: int,
    already: int,
    max_total: int,
    qdrant: QdrantKB,
    text_embedder,
    image_embedder,
    db: UploadsDB,
    text_col: str,
    image_col: str,
) -> UploadResponse:
    """Everything after the bytes are on disk: sniff, quota, ingest, index.

    Shared by the single-shot route and the chunked-session `/complete`, so
    the two transports cannot drift on validation. `dest` is unlinked on
    every failure path — the caller owns nothing once this is entered.
    """
    # Mime gate. Trust the sniffed bytes, not the client-declared type.
    mime = sniff_mime(dest)
    if mime not in ALLOWED_MIMES:
        _safe_unlink(dest)
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported mime: {mime!r}. Allowed: {sorted(ALLOWED_MIMES)}",
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
    filename = Path(filename or file_id).name[:255]
    if is_video_mime(mime):
        kind = "video"
    elif is_image_mime(mime):
        kind = "image"
    else:
        kind = "text"
    # Stamp once here so the qdrant payload + sqlite row agree to the second.
    stamp = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")

    # Video is stored, not extracted. There is no loader that reads a video
    # and no embedder that takes one, so it gets a row and its bytes and
    # nothing else — `status='pending'` is the honest description, and the
    # Phase 32c media worker is what will move it to 'ready'. Returning early
    # keeps it out of the try/except below, whose whole job is ingest.
    if kind == "video":
        try:
            await db.record_upload(
                file_id=file_id, user=user, filename=filename, mime=mime,
                bytes_=written, kind=kind, collection="", chunks=0,
                uploaded_at=stamp, status="pending",
            )
        except Exception as e:
            _safe_unlink(dest)
            log.exception("files: uploads_db.record failed for %s (%s): %s", filename, user, e)
            raise HTTPException(status_code=500, detail=f"Index write failed: {e}") from e
        log.info(
            "files: stored pending video user=%s file_id=%s filename=%r bytes=%d",
            user, file_id, filename, written,
        )
        return UploadResponse(
            file_id=file_id, filename=filename, mime=mime, bytes=written,
            kind=kind, collection="", chunks=0, status="pending",
        )

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


# ─── Chunked upload sessions (Phase 32b) ──────────────────────────────
#
# A single POST cannot carry a large file: cloudflared fronts this app and
# the CDN edge refuses request bodies past its plan limit (100 MB on
# Free/Pro/Business) before they ever reach us — the user sees a 413 whose
# body is an HTML error page, not our JSON. Slicing the file into parts that
# each clear that ceiling is the only transport that works through the
# tunnel, so the per-part size is what matters and the whole-file size stops
# being a transport concern at all.


class SessionOpenRequest(BaseModel):
    filename: str
    total_bytes: int


class SessionOpenResponse(BaseModel):
    upload_id: str
    part_size: int
    parts_total: int
    received_parts: list[int]   # non-empty when resuming


class PartResponse(BaseModel):
    upload_id: str
    part_no: int
    bytes: int
    parts_received: int
    parts_total: int


def _chunked_cfg(request: Request) -> dict:
    cfg = request.app.state.cfg
    kb_cfg = cfg.raw.get("kb", {}) or {}
    return kb_cfg.get("chunked", {}) or {}


def _part_size(request: Request) -> int:
    """Bytes per part. Must stay well under the edge's body cap."""
    return int(_chunked_cfg(request).get("part_size_mb", 8)) * 1024 * 1024


def _chunked_max_bytes(request: Request) -> int:
    """Whole-file ceiling for a chunked upload.

    Deliberately separate from `kb.max_upload_mb`, which caps the single-shot
    route. That cap exists because a single-shot body has to survive the
    tunnel; a chunked upload does not, so holding it to the same number would
    defeat the entire point of chunking.
    """
    return int(_chunked_cfg(request).get("max_upload_mb", 2048)) * 1024 * 1024


def _session_dir(request: Request, user: str, upload_id: str) -> Path:
    """Where a session's parts live: <root>/<user>/.sessions/<upload_id>/.

    Under the per-user directory so a stray sweep bug can't cross users, and
    dot-prefixed so it never collides with a stored `<file_id><ext>`.
    """
    return _upload_root(request) / sanitize_user(user) / ".sessions" / upload_id


@router.post("/upload-sessions", response_model=SessionOpenResponse)
async def open_upload_session(
    body: SessionOpenRequest,
    request: Request,
    me: AuthedUser = Depends(require_user),
) -> SessionOpenResponse:
    """Reserve an upload id and tell the client how to slice the file."""
    user = me.email
    db = _get_uploads_db(request)

    if body.total_bytes <= 0:
        raise HTTPException(status_code=422, detail="total_bytes must be positive.")

    max_bytes = _chunked_max_bytes(request)
    if body.total_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {max_bytes // (1024 * 1024)} MB chunked limit.",
        )

    # Same pre-flight the single-shot route does: refuse before the client
    # spends forty requests on something the quota will reject at the end.
    max_total = _max_user_bytes(request)
    already = await db.user_total_bytes(user)
    if already + body.total_bytes > max_total:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Per-user storage quota exceeded: "
                f"{(already + body.total_bytes) // (1024 * 1024)}MB > "
                f"{max_total // (1024 * 1024)}MB."
            ),
        )

    part_size = _part_size(request)
    parts_total = (body.total_bytes + part_size - 1) // part_size
    upload_id = str(uuid.uuid4())
    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")

    _session_dir(request, user, upload_id).mkdir(parents=True, exist_ok=True)
    await db.open_session(
        upload_id=upload_id, user=user,
        filename=Path(body.filename or "upload").name[:255],
        total_bytes=body.total_bytes, part_size=part_size,
        parts_total=parts_total, now=now,
    )
    log.info(
        "files: session open user=%s upload_id=%s parts=%d bytes=%d",
        user, upload_id, parts_total, body.total_bytes,
    )
    return SessionOpenResponse(
        upload_id=upload_id, part_size=part_size,
        parts_total=parts_total, received_parts=[],
    )


@router.put("/upload-sessions/{upload_id}/parts/{part_no}", response_model=PartResponse)
async def upload_part(
    upload_id: str,
    part_no: int,
    request: Request,
    me: AuthedUser = Depends(require_user),
) -> PartResponse:
    """Stream one part to disk. Idempotent — a retried part overwrites."""
    user = me.email
    db = _get_uploads_db(request)

    session = await db.get_session(upload_id, user=user)
    if session is None:
        raise HTTPException(status_code=404, detail="No such upload session.")
    if not (0 <= part_no < int(session["parts_total"])):
        raise HTTPException(
            status_code=422,
            detail=f"part_no out of range (0..{int(session['parts_total']) - 1}).",
        )

    part_size = int(session["part_size"])
    dest = _session_dir(request, user, upload_id) / f"{part_no:06d}.part"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Cap each part at the negotiated size. Without this a client could send
    # one enormous "part" and walk straight past the whole-file ceiling that
    # `open_upload_session` enforced against the declared total.
    written = 0
    try:
        with dest.open("wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                if written + len(chunk) > part_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Part exceeds the negotiated {part_size} byte size.",
                    )
                written += len(chunk)
                f.write(chunk)
    except HTTPException:
        _safe_unlink(dest)
        raise
    except Exception as e:
        _safe_unlink(dest)
        log.exception("files: part write failed %s/%s: %s", upload_id, part_no, e)
        raise HTTPException(status_code=500, detail=f"Part write failed: {e}") from e

    if written == 0:
        _safe_unlink(dest)
        raise HTTPException(status_code=422, detail="Empty part.")

    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    await db.record_part(upload_id=upload_id, part_no=part_no, bytes_=written, now=now)
    parts_received, _ = await db.session_progress(upload_id)
    return PartResponse(
        upload_id=upload_id, part_no=part_no, bytes=written,
        parts_received=parts_received, parts_total=int(session["parts_total"]),
    )


@router.post("/upload-sessions/{upload_id}/complete", response_model=UploadResponse)
async def complete_upload_session(
    upload_id: str,
    request: Request,
    me: AuthedUser = Depends(require_user),
) -> UploadResponse:
    """Assemble the parts and hand the result to the shared ingest path."""
    user = me.email
    db = _get_uploads_db(request)

    qdrant: QdrantKB | None = getattr(request.app.state, "qdrant", None)
    text_embedder = getattr(request.app.state, "text_embedder", None)
    image_embedder = getattr(request.app.state, "image_embedder", None)
    if qdrant is None or text_embedder is None:
        raise HTTPException(status_code=503, detail="KB is not initialized.")

    session = await db.get_session(upload_id, user=user)
    if session is None:
        raise HTTPException(status_code=404, detail="No such upload session.")

    parts_total = int(session["parts_total"])
    received = await db.received_part_numbers(upload_id)
    missing = sorted(set(range(parts_total)) - received)
    if missing:
        # Don't assemble a file with holes in it — the sniff would read
        # whatever landed and the user would get a baffling 415.
        raise HTTPException(
            status_code=409,
            detail=f"Incomplete upload: {len(missing)} part(s) missing, first is {missing[0]}.",
        )

    text_col, image_col = await ensure_user_collections(qdrant, user)
    max_total = _max_user_bytes(request)
    already = await db.user_total_bytes(user)

    filename = str(session["filename"])
    file_id = str(uuid.uuid4())
    ext = Path(filename).suffix.lower()
    dest = _upload_root(request) / sanitize_user(user) / f"{file_id}{ext}"
    session_dir = _session_dir(request, user, upload_id)

    written = await asyncio.to_thread(_assemble_parts, session_dir, dest, parts_total)
    if written == 0:
        _safe_unlink(dest)
        await _drop_session(db, session_dir, upload_id)
        raise HTTPException(status_code=422, detail="Empty upload.")

    try:
        result = await _validate_and_ingest(
            request, dest,
            user=user, file_id=file_id, filename=filename,
            written=written, already=already, max_total=max_total,
            qdrant=qdrant, text_embedder=text_embedder, image_embedder=image_embedder,
            db=db, text_col=text_col, image_col=image_col,
        )
    finally:
        # Parts are scratch either way: on success they're redundant, on
        # failure the client must reopen a session rather than retry into a
        # half-validated one.
        await _drop_session(db, session_dir, upload_id)

    log.info(
        "files: session complete user=%s upload_id=%s file_id=%s bytes=%d",
        user, upload_id, file_id, written,
    )
    return result


def _assemble_parts(session_dir: Path, dest: Path, parts_total: int) -> int:
    """Concatenate parts 0..n-1 into `dest`. Returns bytes written.

    Sync on purpose — this is blocking file I/O and the caller runs it in a
    thread rather than stalling the event loop on a multi-hundred-MB copy.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dest.open("wb") as out:
        for n in range(parts_total):
            part = session_dir / f"{n:06d}.part"
            with part.open("rb") as src:
                while True:
                    buf = src.read(1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
                    written += len(buf)
    return written


async def _drop_session(db: UploadsDB, session_dir: Path, upload_id: str) -> None:
    """Forget a session: its rows and its parts on disk. Best effort."""
    await db.close_session(upload_id)
    await asyncio.to_thread(shutil.rmtree, session_dir, True)


@router.get("", response_model=ListResponse)
async def list_files(
    request: Request, me: AuthedUser = Depends(require_user),
) -> ListResponse:
    user = me.email
    db = _get_uploads_db(request)

    rows = await db.list_user(user)
    files = [FileRow(**{k: row[k] for k in (
        "file_id", "filename", "mime", "bytes", "uploaded_at", "chunks", "status",
    )}) for row in rows]
    total = sum(r.bytes for r in files)
    return ListResponse(
        user=user, files=files, total_bytes=total,
        limits=Limits(
            max_upload_bytes=_max_upload_bytes(request),
            max_user_bytes=_max_user_bytes(request),
            allowed_extensions=sorted(ALLOWED_EXTENSIONS),
            chunked_max_bytes=_chunked_max_bytes(request),
            part_size=_part_size(request),
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
