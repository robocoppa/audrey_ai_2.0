"""Owner-bound file operations shared by native routes, Projects and bot inputs.

Principal storage namespaces remain opaque, including legacy email namespaces.
Worker protocols and low-level lifecycle helpers stay in ``routes.files``.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import uuid
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile

from audrey.identity import Principal
from audrey.kb.extract import ALLOWED_EXTENSIONS
from audrey.kb.file_deletion import FileDeletionWorker
from audrey.routes import files as uploads


async def upload_file(
    request: Request, principal: Principal, file: UploadFile,
) -> uploads.UploadResponse:
    """Accept one file, validate, extract, ingest into the caller's user collections."""
    uploads._refuse_unavailable_private_processing()
    user = principal.storage_namespace

    qdrant: uploads.QdrantKB | None = getattr(request.app.state, "qdrant", None)
    text_embedder = getattr(request.app.state, "text_embedder", None)
    image_embedder = getattr(request.app.state, "image_embedder", None)
    if qdrant is None or text_embedder is None:
        raise HTTPException(status_code=503, detail="KB is not initialized.")
    storage = uploads._get_storage_lifecycle(request)
    max_upload = uploads._max_upload_bytes(request)
    max_total = uploads._max_user_bytes(request)
    root = uploads._upload_root(request)
    slug = uploads.sanitize_user(user)
    file_id = str(uuid.uuid4())
    ext = Path(file.filename or "").suffix.lower()
    dest = root / slug / f"{file_id}{ext}"

    declared = max_upload if file.size is None else int(file.size)
    if declared <= 0:
        raise HTTPException(status_code=422, detail="Empty upload.")
    if declared > max_upload:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {max_upload // (1024 * 1024)} MB limit.",
        )

    await uploads._sweep_stale_upload_sessions(request, storage)
    now, expired_before = uploads._reservation_window(request)
    try:
        reservation = await storage.reserve_single_upload(
            reservation_id=file_id,
            user=user,
            bytes_=declared,
            max_user_bytes=max_total,
            now=now,
            expired_before=expired_before,
        )
    except uploads.QuotaExceededError as e:
        raise uploads._quota_http_error(e) from e

    try:
        # Collection creation is work too, so it happens only after capacity is
        # held. Every exit below releases the reservation unless commit already
        # consumed it.
        text_col, image_col = await uploads.ensure_user_collections(qdrant, user)
        written = await uploads._stream_to_disk(file, dest, limit_bytes=max_upload)
        if written < 0:
            uploads._safe_unlink(dest)
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds {max_upload // (1024 * 1024)} MB limit.",
            )
        if written == 0:
            uploads._safe_unlink(dest)
            raise HTTPException(status_code=422, detail="Empty upload.")

        try:
            reservation = await storage.reserve_single_upload(
                reservation_id=file_id,
                user=user,
                bytes_=written,
                max_user_bytes=max_total,
                now=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                expired_before=expired_before,
            )
        except uploads.QuotaExceededError as e:
            uploads._safe_unlink(dest)
            raise uploads._quota_http_error(e) from e

        return await uploads._validate_and_ingest(
            request,
            dest,
            user=user,
            file_id=file_id,
            filename=file.filename or file_id,
            written=written,
            max_total=max_total,
            qdrant=qdrant,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            storage=storage,
            reservation=reservation,
            text_col=text_col,
            image_col=image_col,
        )
    finally:
        await storage.release(reservation)


async def open_upload_session(
    body: uploads.SessionOpenRequest, request: Request, principal: Principal,
) -> uploads.SessionOpenResponse:
    """Reserve an upload id and tell the client how to slice the file."""
    uploads._refuse_unavailable_private_processing()
    user = principal.storage_namespace
    storage = uploads._get_storage_lifecycle(request)

    if body.total_bytes <= 0:
        raise HTTPException(status_code=422, detail="total_bytes must be positive.")

    max_bytes = uploads._chunked_max_bytes(request)
    if body.total_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds {max_bytes // (1024 * 1024)} MB chunked limit.",
        )

    max_total = uploads._max_user_bytes(request)
    part_size = uploads._part_size(request)
    parts_total = (body.total_bytes + part_size - 1) // part_size
    upload_id = str(uuid.uuid4())
    await uploads._sweep_stale_upload_sessions(request, storage)
    now, expired_before = uploads._reservation_window(request)
    try:
        reservation = await storage.open_chunk_session(
            upload_id=upload_id,
            user=user,
            filename=Path(body.filename or "upload").name[:255],
            total_bytes=body.total_bytes,
            part_size=part_size,
            parts_total=parts_total,
            max_user_bytes=max_total,
            now=now,
            expired_before=expired_before,
        )
    except uploads.QuotaExceededError as e:
        raise uploads._quota_http_error(e) from e

    try:
        uploads._session_dir(request, user, upload_id).mkdir(parents=True, exist_ok=True)
    except Exception:
        await storage.release(reservation)
        raise
    uploads.log.info(
        "files: session open user=%s upload_id=%s parts=%d bytes=%d",
        user, upload_id, parts_total, body.total_bytes,
    )
    return uploads.SessionOpenResponse(
        upload_id=upload_id, part_size=part_size,
        parts_total=parts_total, received_parts=[],
    )


async def upload_part(
    upload_id: str, part_no: int, request: Request, principal: Principal,
) -> uploads.PartResponse:
    """Stream one part to disk. Idempotent — a retried part overwrites."""
    uploads._refuse_unavailable_private_processing()
    user = principal.storage_namespace
    db = uploads._get_uploads_db(request)
    storage = uploads._get_storage_lifecycle(request)

    session = await db.get_session(upload_id, user=user)
    if session is None:
        raise HTTPException(status_code=404, detail="No such upload session.")
    uploads._refuse_unavailable_private_processing(session)
    if not (0 <= part_no < int(session["parts_total"])):
        raise HTTPException(
            status_code=422,
            detail=f"part_no out of range (0..{int(session['parts_total']) - 1}).",
        )

    part_size = int(session["part_size"])
    dest = uploads._session_dir(request, user, upload_id) / f"{part_no:06d}.part"
    # Keep a valid prior part intact until both the bytes and the reservation
    # update succeed. The final replace is atomic because both paths are siblings.
    incoming = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.incoming")

    # The shared writer awaits every bounded-pool disk write before it accepts
    # another request chunk, and returns before writing a chunk over the cap.
    try:
        written = await uploads._write_chunks_to_disk(
            request.stream(), incoming, limit_bytes=part_size,
        )
        if written < 0:
            raise HTTPException(
                status_code=413,
                detail=f"Part exceeds the negotiated {part_size} byte size.",
            )
    except HTTPException:
        uploads._safe_unlink(incoming)
        raise
    except asyncio.CancelledError:
        uploads._safe_unlink(incoming)
        raise
    except Exception as e:
        uploads._safe_unlink(incoming)
        uploads.log.exception("files: part write failed %s/%s: %s", upload_id, part_no, e)
        raise HTTPException(status_code=500, detail=f"Part write failed: {e}") from e

    if written == 0:
        uploads._safe_unlink(incoming)
        raise HTTPException(status_code=422, detail="Empty part.")

    now = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
    reservation = uploads.StorageReservation(
        upload_id, user, "chunked", int(session["total_bytes"]),
    )
    try:
        await storage.record_chunk_part(
            reservation, part_no=part_no, bytes_=written, now=now,
        )
    except ValueError as e:
        uploads._safe_unlink(incoming)
        raise HTTPException(
            status_code=409,
            detail=str(e),
        ) from e
    except asyncio.CancelledError:
        uploads._safe_unlink(incoming)
        await uploads._drop_session(storage, dest.parent, reservation)
        raise
    except Exception as e:
        uploads._safe_unlink(incoming)
        await uploads._drop_session(storage, dest.parent, reservation)
        uploads.log.exception(
            "files: part accounting failed %s/%s: %s", upload_id, part_no, e,
        )
        raise HTTPException(
            status_code=500, detail=f"Part accounting failed: {e}",
        ) from e

    try:
        await asyncio.to_thread(incoming.replace, dest)
    except asyncio.CancelledError:
        uploads._safe_unlink(incoming)
        await uploads._drop_session(storage, dest.parent, reservation)
        raise
    except OSError as e:
        uploads._safe_unlink(incoming)
        await uploads._drop_session(storage, dest.parent, reservation)
        uploads.log.exception("files: part commit failed %s/%s: %s", upload_id, part_no, e)
        raise HTTPException(status_code=500, detail=f"Part commit failed: {e}") from e

    parts_received, _ = await db.session_progress(upload_id)
    return uploads.PartResponse(
        upload_id=upload_id, part_no=part_no, bytes=written,
        parts_received=parts_received, parts_total=int(session["parts_total"]),
    )


async def complete_upload_session(
    upload_id: str, request: Request, principal: Principal,
) -> uploads.UploadResponse:
    """Assemble the parts and hand the result to the shared ingest path."""
    uploads._refuse_unavailable_private_processing()
    user = principal.storage_namespace
    db = uploads._get_uploads_db(request)
    storage = uploads._get_storage_lifecycle(request)

    qdrant: uploads.QdrantKB | None = getattr(request.app.state, "qdrant", None)
    text_embedder = getattr(request.app.state, "text_embedder", None)
    image_embedder = getattr(request.app.state, "image_embedder", None)
    if qdrant is None or text_embedder is None:
        raise HTTPException(status_code=503, detail="KB is not initialized.")

    session = await db.get_session(upload_id, user=user)
    if session is None:
        raise HTTPException(status_code=404, detail="No such upload session.")
    uploads._refuse_unavailable_private_processing(session)

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

    text_col, image_col = await uploads.ensure_user_collections(qdrant, user)
    max_total = uploads._max_user_bytes(request)
    reservation = uploads.StorageReservation(upload_id, user, "chunked", int(session["total_bytes"]))

    filename = str(session["filename"])
    file_id = str(uuid.uuid4())
    ext = Path(filename).suffix.lower()
    dest = uploads._upload_root(request) / uploads.sanitize_user(user) / f"{file_id}{ext}"
    session_dir = uploads._session_dir(request, user, upload_id)

    written = await asyncio.to_thread(uploads._assemble_parts, session_dir, dest, parts_total)
    if written != int(session["total_bytes"]):
        uploads._safe_unlink(dest)
        await uploads._drop_session(storage, session_dir, reservation)
        raise HTTPException(
            status_code=422,
            detail=(
                f"Assembled size {written} does not match the declared "
                f"{int(session['total_bytes'])} bytes."
            ),
        )

    try:
        result = await uploads._validate_and_ingest(
            request,
            dest,
            user=user,
            file_id=file_id,
            filename=filename,
            written=written,
            max_total=max_total,
            qdrant=qdrant,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            storage=storage,
            reservation=reservation,
            text_col=text_col,
            image_col=image_col,
        )
    finally:
        # Parts are scratch either way: on success they're redundant, on
        # failure the client must reopen a session rather than retry into a
        # half-validated one.
        await uploads._drop_session(storage, session_dir, reservation)

    uploads.log.info(
        "files: session complete user=%s upload_id=%s file_id=%s bytes=%d",
        user, upload_id, file_id, written,
    )
    return result


async def ingest_from_url(
    body: uploads.UrlIngestRequest, request: Request, principal: Principal,
) -> uploads.UploadResponse:
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
      3. **Quota reservation.** Hold the configured per-URL ceiling because the
         real size is not knowable yet. `fetch-result` verifies the actual bytes
         fit that hold and converts it to stored usage.
    """
    uploads._refuse_unavailable_private_processing()
    user = principal.storage_namespace
    db = uploads._get_uploads_db(request)
    storage = uploads._get_storage_lifecycle(request)

    source_url = uploads._validate_fetch_url(body.url, uploads._allowed_fetch_hosts(request))

    existing = await db.find_by_source_url(user=user, source_url=source_url)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"That link is already in your files as {existing['filename']!r} "
                f"({existing['status']}). Delete it first to fetch it again."
            ),
        )

    max_total = uploads._max_user_bytes(request)
    ceiling = uploads._max_fetch_bytes(request)
    file_id = str(uuid.uuid4())
    filename = uploads._url_placeholder_name(source_url)
    await uploads._sweep_stale_upload_sessions(request, storage)
    now, expired_before = uploads._reservation_window(request)
    try:
        await storage.reserve_url_fetch(
            file_id=file_id,
            user=user,
            source_url=source_url,
            filename=filename,
            ceiling_bytes=ceiling,
            max_user_bytes=max_total,
            now=now,
            expired_before=expired_before,
        )
    except uploads.QuotaExceededError as e:
        raise uploads._quota_http_error(e) from e
    uploads.log.info(
        "files: queued url fetch user=%s file_id=%s url=%r", user, file_id, source_url,
    )
    return uploads.UploadResponse(
        file_id=file_id, filename=filename, mime="", bytes=0,
        kind="video", collection="", chunks=0, status="fetch_pending",
    )


async def list_files(
    request: Request, principal: Principal,
) -> uploads.ListResponse:
    user = principal.storage_namespace
    db = uploads._get_uploads_db(request)

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
    files = [uploads.FileRow(**{k: row[k] for k in uploads.FileRow.model_fields}) for row in rows]
    total = sum(r.bytes for r in files)
    return uploads.ListResponse(
        user=user, files=files, total_bytes=total,
        server_time=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        limits=uploads.Limits(
            max_upload_bytes=uploads._max_upload_bytes(request),
            max_user_bytes=uploads._max_user_bytes(request),
            allowed_extensions=sorted(ALLOWED_EXTENSIONS),
            chunked_max_bytes=uploads._chunked_max_bytes(request),
            part_size=uploads._part_size(request),
            fetch_hosts=sorted(uploads._allowed_fetch_hosts(request)),
        ),
    )


async def delete_file(
    file_id: str, request: Request, principal: Principal,
) -> uploads.DeleteResponse:
    user = principal.storage_namespace
    worker: FileDeletionWorker | None = getattr(
        request.app.state,
        "file_deletions",
        None,
    )
    if worker is None:
        raise HTTPException(status_code=503, detail="File deletion is not initialized.")

    result = await worker.request(file_id, user=user)
    uploads.log.info(
        "files: delete user=%s file_id=%s known=%s completed=%s",
        user,
        file_id,
        result.known,
        result.completed,
    )
    return uploads.DeleteResponse(
        file_id=file_id,
        deleted=result.known,
        pending_cleanup=result.known and not result.completed,
    )
