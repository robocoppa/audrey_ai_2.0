"""Service-only evidence reads for owner-bound model tools."""
from __future__ import annotations

import asyncio
import base64
import contextlib
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from audrey.auth import require_service
from audrey.kb.table_analysis import TableAnalysisError, TableAnalysisRequest, analyze_table
from audrey.kb.user_store import sanitize_user
from audrey.media.visual_render import VisualRenderError, render_pdf_page, render_video_frame
from audrey.models.ollama import OllamaError
from audrey.pipeline.vision import (
    DESCRIBE_SYSTEM,
    QUESTION_HINT,
    VisionUnavailableError,
    describe_one_image,
)
from audrey.routes.files import (
    _file_operation_guard,
    _get_uploads_db,
    _source_path,
    _upload_root,
)

router = APIRouter(prefix="/v1/files", tags=["files"])
_visual_slots = asyncio.Semaphore(2)


class OwnedFileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=500)


class TableToolRequest(TableAnalysisRequest):
    user: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=500)


class PDFPageRequest(OwnedFileRequest):
    page: int = Field(ge=1, le=10000)
    question: str = Field(default="", max_length=1000)


class VideoFrameRequest(OwnedFileRequest):
    timestamp_s: Annotated[float, Field(ge=0, le=86400, allow_inf_nan=False)]
    question: str = Field(default="", max_length=1000)


async def _owned_source(body: Any, request: Request) -> tuple[dict, Path, str]:
    rows = await _get_uploads_db(request).list_user(body.user.strip())
    matches = [
        row for row in rows
        if str(row["filename"]).strip().casefold() == body.filename.strip().casefold()
    ]
    if not matches:
        raise HTTPException(status_code=404, detail="File not found.")
    if matches[0].get("private_required", 0):
        # HTTP callers cannot establish an enforced private processing scope
        # yet. Refuse before reading bytes, rendering or selecting vision.
        raise HTTPException(status_code=409, detail="private_processing_unavailable")
    matches = [row for row in matches if not row.get("private_required", 0)]
    row = {**dict(matches[0]), "user": body.user.strip()}
    if row["status"] != "ready":
        raise HTTPException(status_code=409, detail="File is not Ready for inspection.")
    if row.get("source_freed_at"):
        raise HTTPException(
            status_code=409,
            detail="Original file is unavailable. Existing transcript and visual notes can still be read.",
        )
    directory = (_upload_root(request) / sanitize_user(body.user.strip())).resolve()
    try:
        path = _source_path(request, row).resolve(strict=True)
        if not path.is_relative_to(directory) or not path.is_file():
            raise OSError
    except OSError as exc:
        raise HTTPException(status_code=409, detail="Original file is unavailable.") from exc
    note = (
        f"{len(matches)} uploads share this name; using the most recent ({row['uploaded_at']})."
        if len(matches) > 1 else ""
    )
    return row, path, note


@router.post("/table")
async def read_table(
    body: TableToolRequest, request: Request, _: None = Depends(require_service),
) -> dict:
    row, path, note = await _owned_source(body, request)
    spec = TableAnalysisRequest.model_validate(body.model_dump(exclude={"user", "filename"}))
    try:
        async with contextlib.asynccontextmanager(_file_operation_guard)(row["file_id"], request):
            mime = "text/csv" if path.suffix.lower() == ".csv" else str(row["mime"])
            result = await asyncio.to_thread(analyze_table, path, mime, spec)
    except TableAnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=409, detail="Original file is unavailable.") from exc
    return {
        **result, "filename": row["filename"], "artifact": "table",
        "file_id": row["file_id"], "note": note,
    }


async def _inspect(body: PDFPageRequest | VideoFrameRequest, request: Request) -> dict:
    row, path, note = await _owned_source(body, request)
    pdf = isinstance(body, PDFPageRequest)
    if pdf and row["mime"] != "application/pdf":
        raise HTTPException(status_code=422, detail="This tool inspects PDF files only.")
    if not pdf and row["kind"] != "video":
        raise HTTPException(status_code=422, detail="This tool inspects video files only.")
    if not pdf and row.get("duration_s", 0) > 0 and body.timestamp_s >= row["duration_s"]:
        raise HTTPException(status_code=422, detail="Timestamp is beyond the video's duration.")
    state = request.app.state
    if any(getattr(state, name, None) is None for name in ("ollama", "registry", "health", "gate")):
        raise HTTPException(status_code=503, detail="Vision service is not initialized.")
    try:
        async with asyncio.timeout(125), _visual_slots:
            async with contextlib.asynccontextmanager(_file_operation_guard)(row["file_id"], request):
                if pdf:
                    image, total_pages = await render_pdf_page(path, body.page)
                else:
                    image = await render_video_frame(path, body.timestamp_s)
                    total_pages = None
            mime = "image/png" if pdf else "image/jpeg"
            url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
            description, model, _ = await describe_one_image(
                url, ollama=state.ollama, registry=state.registry, health=state.health,
                gate=state.gate, cfg=state.cfg, user_id=body.user.strip(),
                user_question=body.question, system=DESCRIBE_SYSTEM,
                hint_template=QUESTION_HINT,
            )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Visual inspection timed out.") from exc
    except VisualRenderError as exc:
        status = 503 if "unavailable" in str(exc) else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=409, detail="Original file is unavailable.") from exc
    except VisionUnavailableError as exc:
        raise HTTPException(status_code=503, detail="No vision model is available.") from exc
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail="The vision model could not inspect this image.") from exc
    if not description.strip():
        raise HTTPException(status_code=502, detail="The vision model returned no observation.")
    result = {
        "filename": row["filename"], "file_id": row["file_id"],
        "artifact": "pdf_page" if pdf else "video_frame", "model": model,
        "description": description[:2000], "description_truncated": len(description) > 2000,
        "note": note,
    }
    if pdf:
        result.update(page=body.page, total_pages=total_pages)
    else:
        result.update(
            timestamp_s=body.timestamp_s,
            coverage="One still frame near this timestamp; it does not establish movement or speech.",
        )
    return result


@router.post("/inspect-pdf-page")
async def inspect_pdf_page(
    body: PDFPageRequest, request: Request, _: None = Depends(require_service),
) -> dict:
    return await _inspect(body, request)


@router.post("/inspect-video-frame")
async def inspect_video_frame(
    body: VideoFrameRequest, request: Request, _: None = Depends(require_service),
) -> dict:
    return await _inspect(body, request)
