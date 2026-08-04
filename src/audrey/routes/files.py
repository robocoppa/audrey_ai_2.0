"""Per-user file uploads.

    POST   /v1/files           — multipart upload; stream to disk, validate,
                                 ingest into the user's kb_user_text_* /
                                 kb_user_images_* collections.
    GET    /v1/files           — list the caller's files (one row per file_id).
    DELETE /v1/files/{file_id} — purge all points for a file + delete bytes.

    POST   /v1/files/upload-sessions            — open a chunked upload.
    PUT    .../upload-sessions/{id}/parts/{n}   — one part.
    POST   .../upload-sessions/{id}/complete    — assemble and ingest.

    POST   /v1/files/jobs/claim               — SERVICE: lease a pending job.
    POST   /v1/files/{file_id}/ingest-result  — SERVICE: worker output.
    POST   /v1/files/{file_id}/ingest-failed  — SERVICE: worker gave up.

Identity: every *user* endpoint depends on `require_user`, which proxies the
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

The three `jobs`/`ingest-*` routes are the exception to all of the above: they
authenticate with `require_service` and carry no user identity of their own,
because the media worker acts on behalf of whoever uploaded the file. A user
JWT must never reach them — see `auth.require_service`.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel

from audrey.auth import AuthedUser, require_service, require_user
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
from audrey.kb.ingest import (
    ingest_transcript_segments,
    ingest_user_image_file,
    ingest_user_text_file,
)
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
    # 'ready' | 'pending' | 'processing' | 'failed'. Defaulted so a row written
    # before the column existed still validates after the additive migration.
    status: str = "ready"
    # Only set on 'failed'. Shown to the user — a row that stops moving without
    # saying why is the failure this field exists to prevent.
    failure_reason: str = ""
    # Seconds of audio, for video only. 0 everywhere else and for a video with
    # no audio stream, so it is only meaningful read alongside `kind`.
    duration_s: float = 0.0


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
    # Phase 35 media worker is what will move it to 'ready'. Returning early
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


# ─── Chunked upload sessions (Phase 32) ───────────────────────────────
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


# ─── Video job lifecycle (Phase 33) ───────────────────────────────────
#
# A video uploads to `status='pending'` and nothing in this process will ever
# read it — extraction is minutes of CPU and GPU and cannot live in a request.
# These three routes are how a separate media worker takes ownership of that
# row, and how it gives it back.
#
# The worker PULLS. Audrey holds no queue, no retry policy and no address for
# a container it does not own; a worker that is down just means rows keep
# accumulating as 'pending', which is where they already sit today.
#
# Service-token only. They hand out filesystem paths and write into an
# arbitrary user's collection, so a user JWT must not reach them.


class JobClaim(BaseModel):
    file_id: str
    user: str
    filename: str
    mime: str
    bytes: int
    path: str
    lease_id: str
    attempts: int


class TranscriptSegment(BaseModel):
    t_start: float
    t_end: float
    text: str


class IngestResultRequest(BaseModel):
    lease_id: str
    duration_s: float | None = None
    # Phase 33 ships the envelope and the state machine around it. Phase 35
    # fills this with real whisper output; until then a stub worker posts one
    # segment, which is enough to prove the path end to end.
    segments: list[TranscriptSegment] = []


class IngestFailedRequest(BaseModel):
    lease_id: str
    reason: str


class JobResultResponse(BaseModel):
    file_id: str
    status: str
    chunks: int


def _video_cfg(request: Request) -> dict:
    cfg = request.app.state.cfg
    kb_cfg = cfg.raw.get("kb", {}) or {}
    return kb_cfg.get("video", {}) or {}


def _lease_minutes(request: Request) -> int:
    return int(_video_cfg(request).get("lease_minutes", 30))


def _max_attempts(request: Request) -> int:
    return int(_video_cfg(request).get("max_attempts", 3))


def _transcript_chunk_tokens(request: Request) -> int:
    """Tokens per transcript chunk, from `kb.video.transcript_chunk_tokens`.

    Much smaller than the 1000 used for documents. A 1000-token chunk of
    speech is three-plus minutes of talking, and on the first real video a
    25-word verbatim quote scored 0.586 against its own chunk — barely over
    the 0.53 floor, when an exact match should be near 0.9.
    """
    return int(_video_cfg(request).get("transcript_chunk_tokens", 250))


def _hhmmss(seconds: float) -> str:
    """Timestamp prefix for a transcript line, so a chunk carries its own position."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _source_path(request: Request, row) -> Path:
    """Where an uploaded file's bytes live on disk.

    The claim hands this to the worker and the requeue stats it, so the two
    must agree on the layout — one function rather than two copies of the
    same three-part join.
    """
    ext = Path(str(row["filename"])).suffix.lower()
    return (
        _upload_root(request) / sanitize_user(str(row["user"])) / f"{row['file_id']}{ext}"
    )


# `response_model=None`: the return is either a JobClaim or a bare 204
# Response, and FastAPI cannot build one response model from that union.
@router.post("/jobs/claim", response_model=None)
async def claim_job(
    request: Request, _: None = Depends(require_service),
) -> JobClaim | Response:
    """Lease the oldest pending upload, or 204 when the queue is empty.

    An empty queue is the steady state, not an error — a worker polling an idle
    Audrey must not fill the log with failures.

    The sweep runs here rather than on a timer. The only thing that needs
    expired leases returned is a worker asking for work, so doing it on the
    claim keeps recovery in one place with no background task to supervise.
    """
    db = _get_uploads_db(request)

    expiry = _dt.datetime.now(_dt.UTC) - _dt.timedelta(minutes=_lease_minutes(request))
    await db.sweep_expired_leases(
        expired_before=expiry.isoformat(timespec="seconds"),
        max_attempts=_max_attempts(request),
    )

    lease_id = str(uuid.uuid4())
    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    row = await db.claim_job(lease_id=lease_id, now=now)
    if row is None:
        return Response(status_code=204)

    path = _source_path(request, row)
    log.info(
        "files: leased job file_id=%s user=%s attempt=%d",
        row["file_id"], row["user"], row["attempts"],
    )
    return JobClaim(
        file_id=str(row["file_id"]), user=str(row["user"]),
        filename=str(row["filename"]), mime=str(row["mime"]),
        bytes=int(row["bytes"]), path=str(path),
        lease_id=lease_id, attempts=int(row["attempts"]),
    )


@router.post("/{file_id}/ingest-result", response_model=JobResultResponse)
async def ingest_result(
    file_id: str,
    body: IngestResultRequest,
    request: Request,
    _: None = Depends(require_service),
) -> JobResultResponse:
    """Take a worker's output, ingest it, and flip the row to 'ready'.

    Audrey does the ingest, not the worker. `UploadsDB` is a single connection
    with no WAL guarded by an in-process lock — a second container writing that
    file breaks the single-writer contract `reconcile_with_qdrant` depends on,
    and breaks it quietly rather than loudly.
    """
    db = _get_uploads_db(request)
    qdrant: QdrantKB | None = getattr(request.app.state, "qdrant", None)
    text_embedder = getattr(request.app.state, "text_embedder", None)
    if qdrant is None or text_embedder is None:
        raise HTTPException(status_code=503, detail="KB is not initialized.")

    row = await db.get_upload(file_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such file.")
    if row["lease_id"] != body.lease_id or row["status"] != "processing":
        # Refuse before ingesting, not after. A stale worker's transcript must
        # not reach Qdrant at all — writing it and then declining to update the
        # row would leave one run's chunks under another run's row.
        raise HTTPException(
            status_code=409,
            detail="Lease is no longer valid — this job was reclaimed.",
        )

    user = str(row["user"])
    stamp = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")

    transcript = "\n".join(
        f"[{_hhmmss(s.t_start)}] {s.text.strip()}"
        for s in body.segments if s.text.strip()
    )
    chunks = 0
    collection = ""
    # A silent or music-only video produces no text. That is a fact about the
    # file, not a defect in it — the row still completes, with nothing indexed
    # and no collection claimed. Deliberately NOT the `EmptyExtractionError`
    # treatment a scanned PDF gets, where empty means unusable.
    if transcript:
        # Ingest through the ordinary text path rather than a video-shaped one.
        # The sidecar is written beside the source under the same file_id, so
        # the row and its chunks agree on identity and delete-by-file_id still
        # collects both. It stays the human-readable artifact — timestamps and
        # all — but its contents are NOT what gets embedded: see
        # `ingest_transcript_segments`.
        text_col, _image_col = await ensure_user_collections(qdrant, user)
        collection = text_col
        sidecar = (
            _upload_root(request) / sanitize_user(user) / f"{file_id}.transcript.txt"
        )
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(sidecar.write_text, transcript, "utf-8")
        try:
            chunks = await ingest_transcript_segments(
                [s.model_dump() for s in body.segments if s.text.strip()],
                sidecar=sidecar, qdrant=qdrant, embedder=text_embedder,
                collection=text_col, user=user, file_id=file_id,
                filename=str(row["filename"]), mime=str(row["mime"]),
                source_bytes=int(row["bytes"]),
                uploaded_at=stamp,
                chunk_tokens=_transcript_chunk_tokens(request),
            )
        except Exception as e:
            log.exception("files: transcript ingest failed for %s: %s", file_id, e)
            await db.fail_job(
                file_id=file_id, lease_id=body.lease_id,
                reason=f"transcript ingest failed: {e}",
            )
            raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e

    if not await db.complete_job(
        file_id=file_id, lease_id=body.lease_id, collection=collection, chunks=chunks,
        duration_s=float(body.duration_s or 0.0),
    ):
        # Valid when checked above and not now, so the sweep ran in between.
        # The chunks are already in Qdrant under this file_id; the newer
        # lease's delete-before-upsert will clear them.
        raise HTTPException(
            status_code=409, detail="Lease expired during ingest — job was reclaimed.",
        )

    log.info(
        "files: job complete file_id=%s user=%s chunks=%d segments=%d",
        file_id, user, chunks, len(body.segments),
    )
    return JobResultResponse(file_id=file_id, status="ready", chunks=chunks)


@router.post("/{file_id}/ingest-failed", response_model=JobResultResponse)
async def ingest_failed(
    file_id: str,
    body: IngestFailedRequest,
    request: Request,
    _: None = Depends(require_service),
) -> JobResultResponse:
    """Record why a job could not be done, so the user is told rather than left waiting."""
    db = _get_uploads_db(request)
    row = await db.get_upload(file_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such file.")
    if not await db.fail_job(
        file_id=file_id, lease_id=body.lease_id, reason=body.reason,
    ):
        raise HTTPException(
            status_code=409,
            detail="Lease is no longer valid — this job was reclaimed.",
        )
    log.info("files: job failed file_id=%s reason=%r", file_id, body.reason[:200])
    return JobResultResponse(file_id=file_id, status="failed", chunks=0)


@router.post("/{file_id}/requeue", response_model=JobResultResponse)
async def requeue_job(
    file_id: str,
    request: Request,
    force: bool = False,
    _: None = Depends(require_service),
) -> JobResultResponse:
    """Send a processed video back to the queue to be done again.

    Two callers. An operator whose video failed for a reason since fixed — a
    bad codec, a worker bug — which otherwise has no route back into the queue
    at all, only delete-and-re-upload. And development against a real file:
    Phases 34-38 each run a new worker over the same video repeatedly, and
    re-uploading hundreds of megabytes per iteration is the kind of friction
    that stops things from being tested.

    Service-token only, like the other job routes. A user-facing "reprocess"
    button is a separate decision — this one is for the operator.

    **A `processing` row is refused unless `force=true`.** Requeueing a live
    job clears its lease, so the worker finishes, gets a `409`, and its work is
    discarded — 74 seconds of whisper on the run that prompted this guard.
    Nothing breaks and nothing is corrupted; the cost is silent and paid in
    CPU. Taking a job back from a worker is still allowed, because a genuinely
    stuck one needs it, but it should be something you meant to do.
    """
    db = _get_uploads_db(request)
    qdrant: QdrantKB | None = getattr(request.app.state, "qdrant", None)
    if qdrant is None:
        raise HTTPException(status_code=503, detail="KB is not initialized.")

    row = await db.get_upload(file_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such file.")

    if row["status"] == "processing" and not force:
        # Refuse before touching Qdrant, so a refused call changes nothing at
        # all — including for the worker that is still running.
        raise HTTPException(
            status_code=409,
            detail=(
                f"{file_id} is being processed right now (lease "
                f"{row['lease_id']}, since {row['leased_at']}). Requeueing "
                "would discard that work. Retry with ?force=true if that is "
                "what you want."
            ),
        )

    user = str(row["user"])
    # Qdrant first, and fatally. `ingest_user_text_file` deletes by file_id
    # before upserting, so a re-run that produces a transcript would clear
    # these anyway — but a re-run that produces *no* transcript never calls it,
    # and the old chunks would stay searchable under a row claiming none. The
    # reconcile sweep can't catch that either: it exempts `chunks = 0` rows by
    # design. So the points go now, and if this raises, the row is left alone
    # and the caller retries against unchanged state.
    #
    # Text only. A transcript is the only thing this path ever puts in Qdrant;
    # nothing here writes to the image collection.
    await qdrant.delete_by_file_id(
        file_id, user=user, collection=user_text_collection(user),
    )

    # Re-read the size from the file itself rather than trusting the row.
    #
    # A re-run takes its `source_bytes` from this row, so a wrong number here
    # is copied onto the new points and survives the requeue that was meant to
    # fix it. Rows can be wrong: until 2026-08-03 transcript points carried the
    # *sidecar's* size, and `reconcile_with_qdrant` writes payload bytes back
    # onto the row at every boot — which is how a 288 MB video came to be
    # billed as 9 KB against a 1 GiB quota. Once that has happened the file on
    # disk is the only surviving truth.
    #
    # A missing source is not fatal here. The claim would hand the worker a
    # path that does not exist and the job fails with that as its reason,
    # which says more than a 404 from this route would.
    source = _source_path(request, row)
    try:
        source_bytes = (await asyncio.to_thread(source.stat)).st_size
    except OSError:
        log.warning(
            "files: requeue could not stat %s — leaving bytes=%s as recorded",
            source, row["bytes"],
        )
        source_bytes = None

    if not await db.requeue_job(file_id, bytes_=source_bytes):
        raise HTTPException(status_code=404, detail="No such file.")

    log.info(
        "files: requeued file_id=%s user=%s (was %s%s)",
        file_id, user, row["status"], ", FORCED over a live lease" if force else "",
    )
    return JobResultResponse(file_id=file_id, status="pending", chunks=0)


@router.get("", response_model=ListResponse)
async def list_files(
    request: Request, me: AuthedUser = Depends(require_user),
) -> ListResponse:
    user = me.email
    db = _get_uploads_db(request)

    rows = await db.list_user(user)
    files = [FileRow(**{k: row[k] for k in (
        "file_id", "filename", "mime", "bytes", "uploaded_at", "chunks",
        "status", "failure_reason", "duration_s",
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
