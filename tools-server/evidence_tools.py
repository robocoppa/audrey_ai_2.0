"""Owner-bound proxies for calculations and deeper file/history evidence."""
from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

try:
    # Docker copies the shared backend module beside this sidecar.
    from table_analysis import TableAnalysisRequest
except ModuleNotFoundError:
    # Laptop tests use the installed source package.
    from audrey.kb.table_analysis import TableAnalysisRequest

router = APIRouter(tags=["tools"])


class OwnedFileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user: str = Field(
        min_length=1, max_length=200,
        description="Authenticated owner; Audrey fills this in automatically.",
    )
    filename: str = Field(
        min_length=1, max_length=500,
        description="Exact owned filename from attachments or list_my_files. The most recent upload with this name is used.",
    )


class AnalyzeTableRequest(TableAnalysisRequest):
    user: str = Field(min_length=1, max_length=200, description="Owner filled in automatically by Audrey.")
    filename: str = Field(min_length=1, max_length=500, description="Exact owned CSV/XLSX filename.")


class PDFPageRequest(OwnedFileRequest):
    page: int = Field(ge=1, le=10000, description="PDF page number, starting at 1.")
    question: str = Field(default="", max_length=1000, description="Visual detail to inspect.")


class VideoFrameRequest(OwnedFileRequest):
    timestamp_s: Annotated[float, Field(
        ge=0, le=86400, allow_inf_nan=False,
        description="Position in seconds, e.g. 135 for 2:15. Reads one still frame.",
    )]
    question: str = Field(default="", max_length=1000, description="Visible detail to inspect at this moment.")


class OpenChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user: str = Field(min_length=1, max_length=200, description="Owner filled in automatically by Audrey.")
    conversation_id: str = Field(
        min_length=1, max_length=200,
        description="Saved conversation ID returned by chat_history_search.",
    )
    offset: int = Field(default=0, ge=0, description="Use next_offset from the prior page.")


def _require(request: Request, *components: str) -> None:
    unavailable = request.app.state.capabilities.unavailable(list(components))
    if unavailable:
        raise HTTPException(status_code=503, detail={
            "error": "capability_unavailable", "components": unavailable,
        })


async def _proxy(request: Request, path: str, body: dict, *, timeout_s: float = 25) -> dict:
    try:
        response = await request.app.state.audrey.post(path, json=body, timeout=timeout_s)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Evidence read timed out.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Audrey evidence service is unavailable.") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "Evidence read failed.")
        except (ValueError, AttributeError):
            detail = "Evidence read failed."
        raise HTTPException(status_code=response.status_code, detail=detail)
    try:
        result = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Evidence service returned invalid data.") from exc
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="Evidence service returned invalid data.")
    return result


@router.post("/analyze_table", operation_id="analyze_table", description=(
    "Read and analyze an owned CSV or Excel XLSX file. Start with action='inspect' "
    "to see sheet names, column names and sample rows. Then use action='rows' "
    "with filters, 'aggregate' with sum/mean/min/max/count and optional group_by, "
    "or 'duplicates' with columns. First row is the header. Results state row "
    "coverage, next_column_offset for remaining exact headers, and next_offset for rows; totals cover the full selected sheet after filters, "
    "never an incomplete scan. Decimal totals are strings. Formula cells use "
    "cached values; uncached formulas remain unknown and counts are disclosed. "
    "Use calculate for separate arithmetic. Files are read only; no sheets are edited."
))
async def analyze_table(body: AnalyzeTableRequest, request: Request) -> dict:
    _require(request, "audrey_files")
    return await _proxy(request, "/v1/files/table", body.model_dump())


@router.post("/inspect_pdf_page", operation_id="inspect_pdf_page", description=(
    "Look at one actual page of an owned PDF with a vision model. Use for charts, "
    "diagrams, drawing labels or page layout that text extraction misses. "
    "Return visible observations with filename/page attribution; uncertain or "
    "unreadable details stay uncertain. For ordinary document text use get_file_text "
    "or kb_search. This costs a vision call and reads only the requested page."
))
async def inspect_pdf_page(body: PDFPageRequest, request: Request) -> dict:
    _require(request, "audrey_files")
    return await _proxy(request, "/v1/files/inspect-pdf-page", body.model_dump(), timeout_s=140)


@router.post("/inspect_video_frame", operation_id="inspect_video_frame", description=(
    "Inspect an actual still frame near a timestamp in an owned Ready video. "
    "Use when a specific visible moment matters; timestamp_s is seconds. "
    "It does not establish motion, events between frames, or spoken words. "
    "Read the transcript for speech and visual notes for broad coverage. "
    "The original video must still exist; a reclaimed original cannot be inspected "
    "and is not downloaded automatically. Return observed details and timestamp."
))
async def inspect_video_frame(body: VideoFrameRequest, request: Request) -> dict:
    _require(request, "audrey_files")
    return await _proxy(request, "/v1/files/inspect-video-frame", body.model_dump(), timeout_s=140)


@router.post("/open_chat_history", operation_id="open_chat_history", description=(
    "Read one owned saved native or imported conversation after locating its ID "
    "with chat_history_search. Use for the complete prior decision or exchange "
    "when the search snippet is insufficient. Returns chronological saved user/"
    "assistant text in pages, with total_chars and next_offset. Follow next_offset "
    "when more context is needed; do not imply unread parts were reviewed. "
    "Incomplete answers are labeled. Deleted, foreign and unimported legacy "
    "conversations are unavailable. Do not call to personalize ordinary answers."
))
async def open_chat_history(body: OpenChatRequest, request: Request) -> dict:
    _require(request, "audrey_history", "chat_archive_source")
    archive = request.app.state.chat_archive
    cutoff = await archive.conversation_read_cutoff(
        user=body.user, conversation_id=body.conversation_id,
    )
    payload = {**body.model_dump(), "limit": 2000, "visible_after": cutoff or None}
    result = await _proxy(request, "/v1/chat/history/conversation", payload)
    if not isinstance(result.get("conversation_id"), str) or not result["conversation_id"]:
        raise HTTPException(status_code=502, detail="History service returned invalid data.")
    # An imported ID may resolve to a canonical ID with its own tombstone.
    if result.get("conversation_id") != body.conversation_id:
        canonical_cutoff = await archive.conversation_read_cutoff(
            user=body.user, conversation_id=result["conversation_id"],
        )
        if canonical_cutoff and canonical_cutoff > cutoff:
            payload["visible_after"] = canonical_cutoff
            result = await _proxy(request, "/v1/chat/history/conversation", payload)
    return result
