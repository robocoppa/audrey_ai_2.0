"""Per-user file uploads.

    POST   /v1/files           — multipart upload; stream to disk, validate,
                                 ingest into the user's kb_user_text_* /
                                 kb_user_images_* collections.
    GET    /v1/files           — list the caller's files (one row per file_id).
    DELETE /v1/files/{file_id} — purge all points for a file + delete bytes.

    POST   /v1/files/upload-sessions            — open a chunked upload.
    PUT    .../upload-sessions/{id}/parts/{n}   — one part.
    POST   .../upload-sessions/{id}/complete    — assemble and ingest.

    POST   /v1/files/from-url                 — accept a video URL; the bytes
                                                are downloaded elsewhere, later.

    POST   /v1/files/jobs/claim               — SERVICE: lease a pending job.
    POST   /v1/files/{file_id}/ingest-result  — SERVICE: worker output.
    POST   /v1/files/{file_id}/ingest-failed  — SERVICE: worker gave up.
    POST   /v1/files/fetch/claim              — SERVICE: lease a download.
    POST   /v1/files/fetch/{file_id}/result   — SERVICE: download landed.
    POST   /v1/files/fetch/{file_id}/failed   — SERVICE: download gave up.
    POST   /v1/files/list                     — SERVICE: a named user's files,
                                                shaped for a model (phase 40).

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

The `jobs`/`ingest-*` and `fetch/*` routes are the exception to all of the
above: they authenticate with `require_service` and carry no user identity of
their own, because the media worker and the media fetcher act on behalf of
whoever owns the row. A user JWT must never reach them — see
`auth.require_service`.

`POST /v1/files/from-url` is a *user* route and keeps every gate above, with
one addition: a host allowlist, because it is the only route here that makes
this server originate an outbound request. What it does not do is download
anything — see the Phase 41 section for why the bytes are somebody else's job.

`POST /v1/files/list` is the same exception for the same reason, and one step
more dangerous: it *names* its user in the body, so it is only safe while the
sole caller holding the service token — the tools-server — has that argument
overwritten by `tools/dispatch.py`'s `_USER_SCOPED_TOOLS`. The route and that
set are one mechanism in two files.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import logging
import shutil
import urllib.parse
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
    ingest_frame_descriptions,
    ingest_summary,
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
from audrey.pipeline.summarise import summarise_video

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
    # One-paragraph summary of a processed video (Phase 37). Empty for
    # everything else, and for a video whose summary call failed — which is a
    # missing field, never a failed row.
    summary: str = ""
    # When the uploaded bytes were reclaimed (Phase 38). Empty while the source
    # is still on disk. Surfaced rather than kept internal because `bytes` goes
    # on saying 288 MB after the file is gone — without this the file list
    # cannot explain why that no longer counts against the quota, and a user
    # would read the difference as an accounting bug.
    source_freed_at: str = ""
    # When a worker claimed this row (Phase 40). Empty unless `status` is
    # 'processing' or 'fetching'. Surfaced so the page can say how long a job
    # has been running: a 458-second ingest that reports nothing while it runs
    # is indistinguishable from one that has hung.
    leased_at: str = ""
    # Where a fetched video came from (Phase 41). Empty for an ordinary upload.
    # Shown rather than kept internal because it is the only thing on the row a
    # user can act on while it is still downloading — "is it fetching the right
    # video?" is answerable from the link and from nothing else, since the
    # filename is a placeholder until yt-dlp reports the real title.
    source_url: str = ""


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
    # The server's clock at the moment this list was built (Phase 40).
    #
    # Elapsed time is the point of `leased_at`, and a browser computing
    # `Date.now() - leased_at` computes it against the *browser's* clock. A
    # laptop that slept through a timezone, or is simply minutes out, then
    # renders "processing for 47 hours" or a negative — which reads as a
    # broken job queue and sends someone to debug the worker. Anchoring on a
    # timestamp from the same clock that wrote `leased_at` removes the whole
    # class of error; the page ticks forward from here locally.
    server_time: str


class DeleteResponse(BaseModel):
    file_id: str
    deleted: bool


class JobResultResponse(BaseModel):
    """What a stage reports back after taking a row off its queue.

    Up here rather than beside the video-job routes that first used it because
    the Phase 41 fetch routes are declared earlier in the file and name it as a
    `response_model` — a decorator argument, so it is evaluated at import time
    and has to already exist. Shared by both stages, which is honest: they are
    the same handover with different nouns.
    """

    file_id: str
    status: str
    chunks: int


class ServiceListRequest(BaseModel):
    user: str


class ModelFileRow(BaseModel):
    """One file as a *model* should see it — deliberately not `FileRow`.

    `FileRow` answers a browser rendering a management table, so it carries
    `bytes`, `collection`, `file_id` and `source_freed_at`. None of that helps
    a language model answer "what videos do I have?", and every extra field is
    context spent plus one more thing to be wrong about. What is left is what a
    model can say something true with.

    `file_id` is omitted on purpose. The scoping filter (step 3) resolves a
    *filename*, because a filename is the only handle a model can carry from a
    listing into a follow-up question without mangling it — a UUID is exactly
    the kind of token that comes back one character different.
    """

    filename: str
    kind: str                # "text" | "image" | "video"
    status: str              # "ready" | "pending" | "processing" | "failed"
    uploaded_at: str
    # Video only, and 0 for a video with no audio stream — so it is only
    # meaningful read alongside `kind`.
    duration_s: float = 0.0
    # Phase 37's one-paragraph summary. Empty for anything that is not a
    # processed video, and for a video whose summary call failed.
    summary: str = ""
    # Only set on 'failed'. Included for the same reason the upload page shows
    # it: a row that stopped moving without saying why is the failure this
    # field exists to prevent, and that is as true in a chat answer as on a page.
    failure_reason: str = ""
    # Seconds this file has been queued or being worked on, 0 once it is done
    # (Phase 40). Computed server-side rather than shipping `leased_at` and a
    # clock: "how long has my video been processing?" is a question a user
    # asks in chat, and date arithmetic against an unstated now is exactly
    # what a language model should not be doing.
    waiting_for_s: float = 0.0


class ServiceListResponse(BaseModel):
    user: str
    files: list[ModelFileRow]


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


def _waiting_for_s(row: dict, *, now: _dt.datetime) -> float:
    """Seconds a row has been queued or in progress. 0 when it is neither.

    Measured from `leased_at` once a worker holds it and from `uploaded_at`
    before that, because those answer different questions — "it is being
    worked on and has taken this long" versus "it is still behind other work".
    A user deciding whether to wait or come back needs to tell those apart.

    Clamped at 0. The timestamps are written by this process's own clock, so
    a negative should be impossible; if one ever appears — a clock step, an
    NTP correction, a restored backup — reporting it as 0 is the honest
    degradation. A file that has been processing for "-3 minutes" reads as a
    broken queue and sends someone to debug the worker.

    The two Phase 41 fetch states are in the set for the same reason the ingest
    ones are: a download is the stage a user is *most* likely to be watching,
    because it is the one they just started by hand.
    """
    if row["status"] not in ("fetch_pending", "fetching", "pending", "processing"):
        return 0.0
    stamp = str(row["leased_at"] or "") or str(row["uploaded_at"] or "")
    if not stamp:
        return 0.0
    try:
        started = _dt.datetime.fromisoformat(stamp)
    except ValueError:
        return 0.0
    if started.tzinfo is None:
        started = started.replace(tzinfo=_dt.UTC)
    return max(0.0, (now - started).total_seconds())


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


# ─── URL fetch lifecycle (Phase 41) ───────────────────────────────────
#
# One stage in FRONT of the video job lifecycle below. A user pastes a link,
# this accepts it and returns; a `media-fetcher` container downloads the bytes
# and hands the row to the 'pending' queue, from where everything downstream is
# phases 33-38 unchanged and unaware a URL was ever involved.
#
# Declared before the `/{file_id}/...` routes deliberately. FastAPI matches in
# declaration order, and `/fetch/claim` is a two-segment path like
# `/{file_id}/ingest-result` — they cannot collide today because the second
# segments differ, but the ordering makes that a property of the file rather
# than a coincidence of naming.
#
# The download happens in NEITHER `audrey-ai` nor `media-worker`, and both
# exclusions are load-bearing rather than stylistic — see the phase 41 plan.
# The short version: `media-worker` is on an `internal: true` network on
# purpose, and giving it egress would restore its route to the host-published
# ollama port, undoing phase 34's fairness invariant.


class UrlIngestRequest(BaseModel):
    url: str


_FETCH_SCHEMES = frozenset({"http", "https"})


def _fetch_cfg(request: Request) -> dict:
    cfg = request.app.state.cfg
    kb_cfg = cfg.raw.get("kb", {}) or {}
    return kb_cfg.get("fetch", {}) or {}


def _allowed_fetch_hosts(request: Request) -> frozenset[str]:
    """Hosts this deployment will download from, from `kb.fetch.allowed_hosts`.

    **An absent or empty list refuses everything**, and that is the safe
    direction rather than an oversight: this feeds a downloader with write
    access to every user's upload directory, so a deployment whose config
    predates the setting must not silently acquire an open one. The failure
    mode of the safe default is a clear 403 on the first paste; the failure
    mode of the other default is not visible at all.

    Exact matches only — no suffix logic. `youtube.com` in the list must not
    admit `youtube.com.evil.test`, and the moment matching becomes "endswith"
    that is exactly what it admits.
    """
    raw = _fetch_cfg(request).get("allowed_hosts") or []
    return frozenset(str(h).strip().lower() for h in raw if str(h).strip())


def _max_fetch_bytes(request: Request) -> int:
    """Per-URL ceiling. The size is unknown at accept time, so this stands in."""
    return int(_fetch_cfg(request).get("max_bytes_mb", 2048)) * 1024 * 1024


def _fetch_lease_minutes(request: Request) -> int:
    return int(_fetch_cfg(request).get("lease_minutes", 20))


def _fetch_max_attempts(request: Request) -> int:
    return int(_fetch_cfg(request).get("max_attempts", 3))


def _validate_fetch_url(url: str, allowed: frozenset[str]) -> str:
    """Check scheme and host, or raise the HTTPException the caller should send.

    Returns the URL stripped, not rebuilt. `find_by_source_url` matches on the
    exact string and the page displays it, so normalising here would mean two
    spellings of the same paste no longer recognise each other.
    """
    cleaned = url.strip()
    if not cleaned:
        raise HTTPException(status_code=422, detail="url is required.")
    try:
        parsed = urllib.parse.urlparse(cleaned)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Unparseable URL: {e}") from e

    if parsed.scheme.lower() not in _FETCH_SCHEMES:
        raise HTTPException(
            status_code=422,
            detail=f"Only http and https URLs can be fetched, not {parsed.scheme!r}.",
        )
    # `hostname`, never `netloc`. It lowercases, drops the port, and — the part
    # that matters — drops any `user:pass@` prefix. Matching an exact-host set
    # against `netloc` would let `https://www.youtube.com@evil.test/x` pass a
    # membership test while resolving to `evil.test`.
    host = (parsed.hostname or "").lower()
    if not host:
        raise HTTPException(status_code=422, detail="URL has no host.")
    if host not in allowed:
        raise HTTPException(
            status_code=403,
            detail=(
                f"{host} is not a site this server downloads from. Allowed: "
                f"{', '.join(sorted(allowed)) or '(none configured)'}."
            ),
        )
    return cleaned


def _url_placeholder_name(url: str) -> str:
    """A filename to show while the real title is unknown.

    The row is visible the instant it is accepted, and a blank name in the list
    reads as a broken row. The video id is the best available stand-in: short,
    stable, and recognisable next to the link it came from. `complete_fetch`
    replaces it with the real title.
    """
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query or "")
    candidate = (query.get("v") or [""])[0].strip()
    if not candidate:
        candidate = Path(parsed.path or "").name.strip()
    return (candidate or "video")[:255]


@router.post("/from-url", response_model=UploadResponse)
async def ingest_from_url(
    body: UrlIngestRequest,
    request: Request,
    me: AuthedUser = Depends(require_user),
) -> UploadResponse:
    """Accept a video URL and return immediately. The bytes arrive later.

    **Nothing is downloaded in this request.** A 20-minute video is not an HTTP
    request, and a route that waited for one would hold a worker for the whole
    download and time out at the tunnel long before finishing.

    The gates, in the order they run and for the reason they are in that order:

      1. **Host allowlist, before anything else.** Not "whatever yt-dlp
         accepts", which is ~1800 sites including plenty that serve arbitrary
         files — that would make this a general-purpose file downloader with a
         video-shaped name.
      2. **Duplicate URL.** Cheap, and it is the single most likely mistake: a
         double-clicked button or a link already in the list, each costing a
         second download of the same 300 MB.
      3. **Quota.** Against a configured per-URL ceiling, because the real size
         is not knowable yet. `fetch-result` re-checks it against the bytes that
         actually landed — this one only stops an account that is already full
         from starting a download it could never store.
    """
    user = me.email
    db = _get_uploads_db(request)

    source_url = _validate_fetch_url(body.url, _allowed_fetch_hosts(request))

    existing = await db.find_by_source_url(user=user, source_url=source_url)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"That link is already in your files as {existing['filename']!r} "
                f"({existing['status']}). Delete it first to fetch it again."
            ),
        )

    max_total = _max_user_bytes(request)
    already = await db.user_total_bytes(user)
    ceiling = _max_fetch_bytes(request)
    if already + ceiling > max_total:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Not enough room: a fetch can use up to "
                f"{ceiling // (1024 * 1024)}MB and you are at "
                f"{already // (1024 * 1024)}MB of "
                f"{max_total // (1024 * 1024)}MB."
            ),
        )

    file_id = str(uuid.uuid4())
    stamp = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    filename = _url_placeholder_name(source_url)
    await db.record_url_fetch(
        file_id=file_id, user=user, source_url=source_url,
        filename=filename, uploaded_at=stamp,
    )
    log.info(
        "files: queued url fetch user=%s file_id=%s url=%r", user, file_id, source_url,
    )
    return UploadResponse(
        file_id=file_id, filename=filename, mime="", bytes=0,
        kind="video", collection="", chunks=0, status="fetch_pending",
    )


class FetchClaim(BaseModel):
    """A download job, handed to `media-fetcher` on its poll.

    The caps travel with the job for the same reason `FrameSettings` does: the
    fetcher does not mount `config.yaml`, and a parallel set of environment
    variables is a second source of truth that drifts silently. A knob changed
    here takes effect on the next claim.
    """

    file_id: str
    user: str
    source_url: str
    lease_id: str
    attempts: int
    # Where the finished file must end up. A directory rather than a full path
    # because the extension is not known until yt-dlp has picked a container —
    # the fetcher names the file `<file_id><ext>` and reports an extension that
    # matches, which is what `fetch-result` verifies rather than assumes.
    dest_dir: str
    lease_seconds: int = 1200
    max_bytes: int = 2 * 1024 * 1024 * 1024
    # Refuse a six-hour stream from the metadata pass, before a byte is
    # downloaded. 0 disables the check.
    max_duration_s: float = 0.0


class FetchResultRequest(BaseModel):
    lease_id: str
    # The real title, with the extension of the file actually written. The
    # extension is not decoration: `_source_path` rebuilds the on-disk path
    # from it, so a mismatch means the worker is handed a path to nothing.
    filename: str
    duration_s: float = 0.0


class FetchFailedRequest(BaseModel):
    lease_id: str
    reason: str


@router.post("/fetch/claim", response_model=None)
async def claim_fetch(
    request: Request, _: None = Depends(require_service),
) -> FetchClaim | Response:
    """Lease the oldest accepted URL, or 204 when nothing is waiting.

    Sweeps its own stage's expired leases first, exactly as `/jobs/claim` does:
    the only thing that needs a dead fetcher's job returned is another fetcher
    asking for work, so recovery rides the poll and there is still no background
    task to supervise.
    """
    db = _get_uploads_db(request)

    expiry = _dt.datetime.now(_dt.UTC) - _dt.timedelta(
        minutes=_fetch_lease_minutes(request),
    )
    await db.sweep_expired_leases(
        expired_before=expiry.isoformat(timespec="seconds"),
        max_attempts=_fetch_max_attempts(request),
        stage="fetching",
    )

    lease_id = str(uuid.uuid4())
    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    row = await db.claim_fetch(lease_id=lease_id, now=now)
    if row is None:
        return Response(status_code=204)

    user = str(row["user"])
    dest_dir = _upload_root(request) / sanitize_user(user)
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "files: leased fetch file_id=%s user=%s attempt=%d url=%r",
        row["file_id"], user, row["attempts"], row["source_url"],
    )
    cfg = _fetch_cfg(request)
    return FetchClaim(
        file_id=str(row["file_id"]), user=user,
        source_url=str(row["source_url"]), lease_id=lease_id,
        attempts=int(row["attempts"]), dest_dir=str(dest_dir),
        lease_seconds=_fetch_lease_minutes(request) * 60,
        max_bytes=_max_fetch_bytes(request),
        max_duration_s=float(cfg.get("max_duration_s", 0) or 0),
    )


@router.post("/fetch/{file_id}/result", response_model=JobResultResponse)
async def fetch_result(
    file_id: str,
    body: FetchResultRequest,
    request: Request,
    _: None = Depends(require_service),
) -> JobResultResponse:
    """Take a completed download and hand the row to the video queue.

    The fetcher reports what it downloaded; this route **verifies it** rather
    than recording it. Three checks, each of which has to happen here because
    the fetcher is a separate container that could be wrong, stale, or broken:

      1. **The file is where it says it is.** `_source_path` rebuilds the path
         from `file_id` plus the reported filename's extension, so a filename
         whose extension disagrees with the file on disk hands the media worker
         a path to nothing — a failure that would otherwise surface three
         attempts later as an unexplained ingest error.
      2. **The bytes are a video.** The same libmagic gate an upload passes.
         Without it, a downloader that saved an HTML "video unavailable" page
         pushes that into the transcription queue.
      3. **The quota, for real this time.** `from-url` could only check a
         configured ceiling; this is the first moment the true size exists.

    A failed check fails the *row*, unlinks the bytes, and says why. Leaving a
    row in 'fetching' with a rejected file on disk would consume quota for
    something no one can see.
    """
    db = _get_uploads_db(request)
    row = await db.get_upload(file_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such file.")
    if row["lease_id"] != body.lease_id or row["status"] != "fetching":
        raise HTTPException(
            status_code=409,
            detail="Lease is no longer valid — this fetch was reclaimed.",
        )

    user = str(row["user"])
    filename = Path(body.filename or "").name[:255]
    if not filename:
        raise HTTPException(status_code=422, detail="filename is required.")

    # Rebuild the path the same way every other reader of this row will.
    dest = _source_path(request, {**row, "filename": filename})

    async def rejected(status_code: int, detail: str) -> HTTPException:
        """Fail the row, drop the bytes, and hand back the error to raise.

        Returns rather than raises so every call site reads `raise await
        rejected(...)` — the rejection is visible as a `raise` at the point it
        happens, instead of being hidden inside a helper that callers have to
        know never returns.
        """
        await asyncio.to_thread(_safe_unlink, dest)
        await db.fail_job(
            file_id=file_id, lease_id=body.lease_id, reason=detail, stage="fetching",
        )
        return HTTPException(status_code=status_code, detail=detail)

    try:
        written = (await asyncio.to_thread(dest.stat)).st_size
    except OSError:
        raise await rejected(
            422,
            f"the download reported {filename!r} but nothing is at {dest.name} — "
            "the extension in the reported filename must match the file written",
        ) from None

    if written == 0:
        raise await rejected(422, "the download produced an empty file")

    mime = await asyncio.to_thread(sniff_mime, dest)
    if mime not in ALLOWED_MIMES or not is_video_mime(mime):
        raise await rejected(
            415,
            f"the download is not a video: {mime!r}. This is usually a site "
            "returning an error page rather than the media",
        )

    max_total = _max_user_bytes(request)
    # This row still carries bytes=0, so the sum excludes it — `already` is
    # genuinely everything else the user is storing.
    already = await db.user_total_bytes(user)
    if already + written > max_total:
        raise await rejected(
            413,
            f"the download is {written // (1024 * 1024)}MB and would put you over "
            f"your {max_total // (1024 * 1024)}MB quota",
        )

    if not await db.complete_fetch(
        file_id=file_id, lease_id=body.lease_id, filename=filename,
        mime=mime, bytes_=written,
    ):
        # Swept between the guard above and here. The bytes stay on disk at a
        # path derived from this file_id, so the re-run overwrites them rather
        # than orphaning them — deleting here would race the fetcher that has
        # already been handed the job.
        raise HTTPException(
            status_code=409, detail="Lease expired during the handover — fetch reclaimed.",
        )

    log.info(
        "files: fetch complete file_id=%s user=%s filename=%r bytes=%d mime=%s "
        "duration=%.1fs — now pending for the media worker",
        file_id, user, filename, written, mime, body.duration_s,
    )
    return JobResultResponse(file_id=file_id, status="pending", chunks=0)


@router.post("/fetch/{file_id}/failed", response_model=JobResultResponse)
async def fetch_failed(
    file_id: str,
    body: FetchFailedRequest,
    request: Request,
    _: None = Depends(require_service),
) -> JobResultResponse:
    """Record why a download could not be done.

    **The reason is the deliverable here, not the happy path.** Private video,
    region blocked, members-only, age-gated, live stream, deleted — these are
    the common cases, not edge cases, and a generic "download failed" is the
    message that generates the support question this whole field exists to
    prevent. The fetcher maps what it can and passes the rest through; this end
    stores it verbatim and does not editorialise, exactly as `ingest-failed`
    does for the worker.
    """
    db = _get_uploads_db(request)
    row = await db.get_upload(file_id)
    if row is None:
        raise HTTPException(status_code=404, detail="No such file.")
    if not await db.fail_job(
        file_id=file_id, lease_id=body.lease_id, reason=body.reason, stage="fetching",
    ):
        raise HTTPException(
            status_code=409,
            detail="Lease is no longer valid — this fetch was reclaimed.",
        )
    log.info("files: fetch failed file_id=%s reason=%r", file_id, body.reason[:200])
    return JobResultResponse(file_id=file_id, status="failed", chunks=0)


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


class FrameSettings(BaseModel):
    """Visual-pass knobs, handed to the worker with its job (Phase 36).

    Phase 34 established that the worker reads env rather than `config.yaml`,
    because it does not mount that file. These settings arrive on the claim
    instead, which keeps `kb.video.*` the single source of truth without
    either mounting config into a second container or maintaining a parallel
    set of environment variables that can silently disagree with it.

    A knob that changes here takes effect on the next *claim*, not the next
    worker restart — which is the behaviour you want while calibrating a
    threshold against real footage.
    """

    interval_s: float = 30.0
    keyframes_max: int = 24
    max_width: int = 1280
    dedup_distance: int = 8


class JobClaim(BaseModel):
    file_id: str
    user: str
    filename: str
    mime: str
    bytes: int
    path: str
    lease_id: str
    attempts: int
    frames: FrameSettings = FrameSettings()
    # Seconds this lease is good for, so the worker can budget against the
    # clock that will actually reclaim its job.
    #
    # Without it the stage budgets are set independently and can exceed the
    # lease between them: 1440s of transcription plus 900s of frames is 39
    # minutes against a 30-minute lease, so a long video would be swept
    # mid-describe, re-claimed, and burn its attempts doing the same thing
    # again. The worker cannot read `config.yaml`, so the number comes with
    # the job.
    lease_seconds: int = 1800


class TranscriptSegment(BaseModel):
    t_start: float
    t_end: float
    text: str


class FrameDescription(BaseModel):
    """One keyframe's prose, and the span of video it speaks for."""

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
    # Phase 36. Independent of `segments`: a silent video has frames and no
    # speech, an audio-only upload has speech and no frames, and both are
    # ordinary successful jobs.
    frames: list[FrameDescription] = []
    # How many keyframes the worker meant to describe. When this exceeds
    # `len(frames)` some were dropped — the budget ran out, a frame could not
    # be read, or the describe route refused it. The worker's log says which;
    # this end only carries the count, and must not guess at the reason.
    frames_planned: int | None = None


class IngestFailedRequest(BaseModel):
    lease_id: str
    reason: str


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


async def _reclaim_sources(request: Request, db) -> int:
    """Unlink processed videos whose bytes are past the retention window.

    Returns how many were reclaimed. **Never raises** — this runs on the claim
    path, and a disk error while tidying up must not stop a worker from getting
    work. A source that could not be unlinked stays eligible and is simply
    retried on the next poll.

    ## Why here, and not on a timer

    Nothing else in this file has a background task, deliberately: phase 33 put
    the lease sweep on the claim for the same reason, because a poll every ten
    seconds is already a heartbeat and a supervised task is one more thing that
    can silently stop. Reclamation has no deadline — a video freed a minute
    late costs nothing — so it inherits that heartbeat rather than growing its
    own.

    ## The order that matters

    sqlite is marked FIRST, and only then is the file unlinked. Both orders
    lose something if the process dies between the two steps, and they are not
    the same loss: unlink-then-mark leaves a row that still bills the user for
    bytes that no longer exist, with nothing left on disk to prove otherwise
    and no path that would ever correct it. Mark-then-unlink leaves an orphaned
    file that costs disk and is invisible to the quota — recoverable by hand,
    and never wrong about what the user owes.
    """
    video_cfg = _video_cfg(request)
    # Defaults to keeping. An absent config key must not be able to delete a
    # user's file — this is the only irreversible operation in the pipeline,
    # and a deployment whose `config.yaml` predates the setting is exactly the
    # deployment least expecting it.
    if bool(video_cfg.get("keep_source", True)):
        return 0

    hours = float(video_cfg.get("source_retention_hours", 24))
    cutoff = _dt.datetime.now(_dt.UTC) - _dt.timedelta(hours=hours)
    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")

    try:
        rows = await db.reclaimable_sources(
            completed_before=cutoff.isoformat(timespec="seconds"),
        )
    except Exception as e:  # noqa: BLE001 — tidying must never break a claim
        log.warning("files: could not list reclaimable sources: %s", e)
        return 0

    freed = 0
    for row in rows:
        if not await db.mark_source_freed(str(row["file_id"]), freed_at=now):
            # Another sweep took it first. The transition is the guard.
            continue
        path = _source_path(request, row)
        await asyncio.to_thread(_safe_unlink, path)
        freed += 1
        log.info(
            "files: reclaimed source for %s (%s, %d bytes) — %s retention "
            "window elapsed; this video can no longer be requeued",
            row["file_id"], row["filename"], int(row["bytes"] or 0),
            f"{hours:g}h",
        )
    return freed


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
    await _reclaim_sources(request, db)

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
    video_cfg = _video_cfg(request)
    return JobClaim(
        file_id=str(row["file_id"]), user=str(row["user"]),
        filename=str(row["filename"]), mime=str(row["mime"]),
        bytes=int(row["bytes"]), path=str(path),
        lease_id=lease_id, attempts=int(row["attempts"]),
        lease_seconds=_lease_minutes(request) * 60,
        frames=FrameSettings(
            interval_s=float(video_cfg.get("frame_interval_s", 30)),
            keyframes_max=int(video_cfg.get("keyframes_max", 24)),
            max_width=int(video_cfg.get("frame_max_width", 1280)),
            dedup_distance=int(video_cfg.get("frame_dedup_distance", 8)),
        ),
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
    frames = [f.model_dump() for f in body.frames if f.text.strip()]
    chunks = 0
    collection = ""

    # Clear the file_id once, here, rather than inside each ingest call.
    #
    # `delete_by_file_id` removes *every* point for the file, so the transcript
    # ingest deleting on its own way in would take out frame descriptions
    # written moments earlier — and it would do it in whichever order the two
    # calls happened to run. One delete up front means both artifacts write
    # into the space it made, and a re-run still replaces its predecessor
    # rather than doubling it.
    if transcript or frames:
        await qdrant.delete_by_file_id(
            file_id, user=user, collection=user_text_collection(user),
        )

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
                delete_existing=False,
            )
        except Exception as e:
            log.exception("files: transcript ingest failed for %s: %s", file_id, e)
            await db.fail_job(
                file_id=file_id, lease_id=body.lease_id,
                reason=f"transcript ingest failed: {e}",
            )
            raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e

    # Phase 36. Independent of the transcript in both directions: a silent
    # video has frames and no speech, an audio-only upload has speech and no
    # frames, and both are ordinary successful jobs.
    if frames:
        text_col, _image_col = await ensure_user_collections(qdrant, user)
        collection = text_col
        frame_sidecar = (
            _upload_root(request) / sanitize_user(user) / f"{file_id}.frames.txt"
        )
        frame_sidecar.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            frame_sidecar.write_text,
            "\n\n".join(f"[{_hhmmss(f['t_start'])}] {f['text'].strip()}" for f in frames),
            "utf-8",
        )
        try:
            chunks += await ingest_frame_descriptions(
                frames,
                sidecar=frame_sidecar, qdrant=qdrant, embedder=text_embedder,
                collection=text_col, user=user, file_id=file_id,
                filename=str(row["filename"]), mime=str(row["mime"]),
                source_bytes=int(row["bytes"]),
                uploaded_at=stamp,
                chunk_tokens=_transcript_chunk_tokens(request),
            )
        except Exception as e:
            log.exception("files: frame ingest failed for %s: %s", file_id, e)
            await db.fail_job(
                file_id=file_id, lease_id=body.lease_id,
                reason=f"frame ingest failed: {e}",
            )
            raise HTTPException(status_code=500, detail=f"Ingest failed: {e}") from e

    # Phase 37. Last, because it reads what the other two produced — and
    # fail-soft, because by now they are ingested and already useful. A
    # summary that could not be written is a missing field, never a failed
    # row: the video is still searchable by everything it said and showed.
    summary = ""
    if transcript or frames:
        try:
            summary = await summarise_video(
                [s.model_dump() for s in body.segments if s.text.strip()],
                frames,
                filename=str(row["filename"]),
                duration_s=float(body.duration_s or 0.0),
                ollama=request.app.state.ollama,
                registry=request.app.state.registry,
                gate=request.app.state.gate,
                cfg=request.app.state.cfg,
                user_id=user,
            )
        except Exception as e:  # noqa: BLE001 — every failure here is survivable
            log.warning("files: summary failed for %s (row stays ready): %s", file_id, e)

    if summary:
        try:
            chunks += await ingest_summary(
                summary,
                sidecar=(
                    _upload_root(request) / sanitize_user(user) / f"{file_id}.summary.txt"
                ),
                qdrant=qdrant, embedder=text_embedder,
                collection=collection or user_text_collection(user),
                user=user, file_id=file_id,
                filename=str(row["filename"]), mime=str(row["mime"]),
                source_bytes=int(row["bytes"]), uploaded_at=stamp,
            )
        except Exception as e:  # noqa: BLE001 — same reasoning; keep the text on the row
            log.warning("files: summary ingest failed for %s: %s", file_id, e)

    if body.frames_planned is not None and len(frames) < body.frames_planned:
        # Not a failure. Unlike a transcript, frame descriptions are
        # independently timestamped, so an incomplete set is correct about
        # what it covers rather than silently wrong about the whole. Logged
        # because the alternative is a video that is quietly less searchable
        # than the one next to it, for no visible reason.
        #
        # **This must not name a cause.** It used to end "the visual pass ran
        # out of budget", which is one of three reasons `describe_frames`
        # returns short — the others being a frame it could not read and a
        # frame the describe route refused. On 2026-08-04 a `num_predict` cap
        # made three frames come back empty, the route 502'd them, and this
        # line reported a budget that had not been touched. It cost a round of
        # diagnosis pointed at the wrong subsystem.
        #
        # The worker knows which frames were dropped and why, and logs it per
        # frame. This end only knows the count, so the count is all it says.
        log.warning(
            "files: %s described %d of %d planned keyframes — see the "
            "media-worker log for which frames were dropped and why",
            file_id, len(frames), body.frames_planned,
        )

    if not await db.complete_job(
        file_id=file_id, lease_id=body.lease_id, collection=collection, chunks=chunks,
        duration_s=float(body.duration_s or 0.0), summary=summary,
        completed_at=stamp,
    ):
        # Valid when checked above and not now, so the sweep ran in between.
        # The chunks are already in Qdrant under this file_id; the newer
        # lease's delete-before-upsert will clear them.
        raise HTTPException(
            status_code=409, detail="Lease expired during ingest — job was reclaimed.",
        )

    log.info(
        "files: job complete file_id=%s user=%s chunks=%d segments=%d frames=%d",
        file_id, user, chunks, len(body.segments), len(frames),
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

    if row["source_freed_at"]:
        # Phase 38 reclaimed the bytes. `force` deliberately does not override
        # this — the other guard protects work that is merely in progress, and
        # can be overruled by someone who means it. There is nothing to
        # overrule here.
        #
        # Refusing costs the caller one clear error. Proceeding would delete
        # the video's existing chunks (below), queue a job against a path that
        # no longer exists, and burn all three attempts failing on it — ending
        # with a video that WAS fully searchable and now is not, and no way
        # back short of re-uploading the file.
        raise HTTPException(
            status_code=409,
            detail=(
                f"{file_id}'s source was reclaimed on {row['source_freed_at']} "
                "and cannot be re-processed. Its transcript, descriptions and "
                "summary are still searchable; re-upload the file to run it "
                "again. Set kb.video.keep_source to stop this happening to "
                "videos uploaded from now on."
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
    # Projected straight from what `FileRow` declares, not a third hand-written
    # column list. There was one here, and it had already drifted: `bytes`,
    # `status`, `summary` and the rest were listed, `source_freed_at` was not —
    # so the reclamation marker phase 38 added to `FileRow` and to the SELECT
    # was silently `""` on every response, and the page's strikethrough could
    # never render. Nothing errored, because the field has a default.
    #
    # "Adding a column means three places" was the lesson from the `summary`
    # 500; this was the fourth, and it fails *quietly* rather than loudly.
    # Driving the projection from the model removes it as a place at all —
    # a field FileRow declares but the SELECT omits now raises `KeyError`
    # here, which `TestListReturnsEveryFileRowField` already pins.
    files = [FileRow(**{k: row[k] for k in FileRow.model_fields}) for row in rows]
    total = sum(r.bytes for r in files)
    return ListResponse(
        user=user, files=files, total_bytes=total,
        server_time=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        limits=Limits(
            max_upload_bytes=_max_upload_bytes(request),
            max_user_bytes=_max_user_bytes(request),
            allowed_extensions=sorted(ALLOWED_EXTENSIONS),
            chunked_max_bytes=_chunked_max_bytes(request),
            part_size=_part_size(request),
        ),
    )


@router.post("/list", response_model=ServiceListResponse)
async def list_files_for_user(
    body: ServiceListRequest,
    request: Request,
    _: None = Depends(require_service),
) -> ServiceListResponse:
    """SERVICE: one named user's files, shaped for a model rather than a page.

    Exists because the tools-server holds `KB_SERVICE_TOKEN` and **cannot
    obtain a user JWT**, so it can never call `GET /v1/files` — that route
    depends on `require_user`. Four phases in this campaign have now planned
    around a service-token act-as that does not exist; check what a route
    actually depends on before designing against it.

    `require_service`, not `resolve_kb_caller`. The latter also accepts a user
    JWT, and this route names its target user in the body — so a JWT-bearing
    caller could read anyone's file list by typing a different email. The
    security property here is the same one the job routes rely on: holding a
    valid *user* token must not be enough to reach it.

    That property alone is not sufficient, because the tools-server passes the
    tool's `user` argument through to this body. The other half lives in
    `tools/dispatch.py`, whose `_USER_SCOPED_TOOLS` set overwrites the
    model-supplied `user` with the authenticated pipeline user. **Both halves
    are required.** Without the dispatcher's overwrite, a prompt naming another
    user's email reaches their file list through this route.

    POST rather than GET because the target is an email address: a query string
    puts it in every access and proxy log along the way, and the sibling service
    routes are POST regardless.

    An unknown user gets an empty list, not a 404 — "you have not uploaded
    anything" is a true and useful answer, and distinguishing an unknown user
    from an empty one would leak which addresses exist.
    """
    user = body.user.strip()
    if not user:
        raise HTTPException(status_code=422, detail="user is required.")

    db = _get_uploads_db(request)
    rows = await db.list_user(user)
    now = _dt.datetime.now(_dt.UTC)
    files = [
        ModelFileRow(
            filename=str(row["filename"]),
            kind=str(row["kind"]),
            status=str(row["status"]),
            uploaded_at=str(row["uploaded_at"]),
            duration_s=float(row["duration_s"]),
            summary=str(row["summary"]),
            failure_reason=str(row["failure_reason"]),
            waiting_for_s=_waiting_for_s(row, now=now),
        )
        for row in rows
    ]
    log.info("files: service list user=%s files=%d", user, len(files))
    return ServiceListResponse(user=user, files=files)


# ─── Reading an artifact back (2026-08-05) ────────────────────────────
#
# Ingest writes three sidecars beside the source and, until this route, nothing
# ever read any of them back. "Give me the transcript" is the most obvious
# question to ask of an ingested video and the only answer available was
# `kb_search`, which returns ranked fragments — at the fast path's cap, one
# 992-char chunk. A model asked for a transcript reported that it could give
# only a partial excerpt, which was true, and then invented a way to ask for
# the rest, which was not.
#
# Paged rather than whole, and that is not a limitation to apologise for. A
# two-hour transcript cannot enter a chat turn at any budget, so the choice is
# between an honest page with a stated position and a silent truncation. The
# tool asks for a range and is told where it is in the document.

#: `artifact` names as the Qdrant payload spells them → the sidecar on disk.
#: **`visual` is written to `.frames.txt`**, and the mismatch is deliberate
#: history rather than a bug: the payload names what the text *is* (a visual
#: description) and the file names where it came *from* (keyframes). Anything
#: reading one by the other's name finds nothing, so the map is explicit.
_ARTIFACT_SIDECARS: dict[str, str] = {
    "transcript": "transcript.txt",
    "visual": "frames.txt",
    "summary": "summary.txt",
}


class ArtifactRequest(BaseModel):
    user: str
    filename: str
    artifact: str = "transcript"
    offset: int = 0
    limit: int = 4000


class ArtifactResponse(BaseModel):
    filename: str
    artifact: str
    text: str
    offset: int
    # Where a follow-up call should start, or None at the end of the document.
    # An explicit end marker rather than leaving the caller to compare
    # `offset + len(text)` against `total_chars`: the whole failure this route
    # exists to fix was a model reasoning about how much it was missing.
    next_offset: int | None
    total_chars: int
    note: str = ""


@router.post("/artifact", response_model=ArtifactResponse)
async def read_artifact(
    body: ArtifactRequest,
    request: Request,
    _: None = Depends(require_service),
) -> ArtifactResponse:
    """SERVICE: one page of a file's transcript, visual descriptions or summary.

    `require_service` and a user named in the body, exactly like
    `POST /v1/files/list` — and with exactly the same second half: the
    dispatcher's `_USER_SCOPED_TOOLS` overwrites the model-supplied `user`.
    **Both halves are required.** This route hands back a user's document
    contents verbatim, so without the dispatcher's overwrite a prompt naming
    another address reads their transcripts.

    Pages are snapped to a line boundary. Transcript lines are `[00:04:12]
    text`, so a page cut mid-line hands the model a fragment with no timestamp
    and half a sentence — and the model cannot tell that from the real end of
    the document.
    """
    user = body.user.strip()
    if not user:
        raise HTTPException(status_code=422, detail="user is required.")

    artifact = body.artifact.strip().lower() or "transcript"
    if artifact not in _ARTIFACT_SIDECARS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown artifact {artifact!r}. Available: "
                f"{', '.join(sorted(_ARTIFACT_SIDECARS))}."
            ),
        )

    db = _get_uploads_db(request)
    wanted = body.filename.strip().casefold()
    rows = [r for r in await db.list_user(user)
            if str(r["filename"]).strip().casefold() == wanted]
    if not rows:
        # Name the alternatives rather than only denying, the same way
        # `_unknown_file_notice` does for a scoped search. A model that
        # mistyped a filename can correct itself in one turn; one told only
        # "not found" reports the file as missing.
        names = sorted({str(r["filename"]) for r in await db.list_user(user)})
        raise HTTPException(
            status_code=404,
            detail=(
                f"No file named {body.filename!r}. You have: "
                f"{', '.join(names) if names else '(nothing uploaded)'}."
            ),
        )

    # Newest first — `list_user` orders by `uploaded_at DESC`, so a filename
    # reused across two uploads resolves to the more recent one. Said out loud
    # in `note`, because silently picking one of two identically named files is
    # how a confident answer comes from the wrong source.
    row = rows[0]
    note = ""
    if len(rows) > 1:
        note = (
            f"{len(rows)} files are named {body.filename!r}; this is the most "
            f"recent, uploaded {row['uploaded_at']}. "
        )

    path = (
        _upload_root(request) / sanitize_user(user)
        / f"{row['file_id']}.{_ARTIFACT_SIDECARS[artifact]}"
    )
    try:
        text = await asyncio.to_thread(path.read_text, "utf-8")
    except OSError:
        # Distinguish the three reasons this file is absent, because they need
        # three different answers and a bare 404 gets reported as "the video
        # has no transcript" for all of them.
        status = str(row["status"])
        if status != "ready":
            detail = (
                f"{row['filename']!r} is {status}, so it has no {artifact} yet."
            )
        elif str(row["kind"]) != "video":
            detail = (
                f"{row['filename']!r} is a {row['kind']} file — {artifact} "
                "exists only for video."
            )
        else:
            # State the fact and STOP. This used to append "a video with no
            # speech has no transcript, and one whose frames were all
            # near-identical has no visual pass" — two general explanations,
            # offered as help.
            #
            # A model repeated them back as findings about the file. Asked to
            # compare two videos on 2026-08-06, it reported that silent.mp4's
            # "frames are all near-identical", which nothing had measured — the
            # sentence was a worked example of why an artifact can be absent,
            # and it read as a description of this one.
            #
            # An error message is evidence to whoever receives it. Anything in
            # here that sounds like an observation will be reported as one, so
            # it may only contain things that were actually checked.
            detail = (
                f"{row['filename']!r} finished processing and produced no "
                f"{artifact}. There is nothing to read and nothing is known "
                f"about what this file contains. Report that it has no "
                f"{artifact} and stop — do not describe its contents, do not "
                "guess at a cause, and do not attribute anything to it from "
                "another file or from its summary."
            )
        raise HTTPException(status_code=404, detail=detail) from None

    total = len(text)
    offset = max(0, min(body.offset, total))
    limit = max(1, body.limit)
    end = min(offset + limit, total)
    # Snap forward to the end of the line so a page break lands between
    # transcript lines rather than inside one. Only when there is more to come;
    # at the end of the document there is nothing to snap to.
    if end < total:
        newline = text.find("\n", end)
        end = total if newline == -1 else newline + 1

    page = text[offset:end]
    next_offset = end if end < total else None
    if next_offset is not None:
        note += (
            f"This is characters {offset:,}-{end:,} of {total:,}. Call again "
            f"with offset={end} for the next page."
        )

    log.info(
        "files: artifact read user=%s file=%r artifact=%s %d-%d of %d",
        user, row["filename"], artifact, offset, end, total,
    )
    return ArtifactResponse(
        filename=str(row["filename"]), artifact=artifact, text=page,
        offset=offset, next_offset=next_offset, total_chars=total,
        note=note.strip(),
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
