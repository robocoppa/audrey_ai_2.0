"""Read-only proxies for configured public reference libraries."""

from __future__ import annotations

from typing import Literal

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(tags=["tools"])


class ReferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    library_id: str = Field(
        default="irs-federal",
        pattern=r"^[a-z][a-z0-9-]{0,63}$",
        description="Registered public library ID. Use irs-federal for the owned English IRS archive.",
    )


class ReferenceLookupRequest(ReferenceRequest):
    product_number: str | None = Field(
        default=None, min_length=1, max_length=100,
        description="Exact official product number, e.g. Form 1040 or Publication 17.",
    )
    query: str | None = Field(
        default=None, min_length=1, max_length=200,
        description="Words to match in catalog titles or product numbers; this is catalog lookup, not document-content search.",
    )
    catalog: Literal["current", "prior", "bulletins"] | None = Field(
        default="current", description="Current catalog by default. Explicit prior/bulletins lookups are limited to the server-controlled recent-history window; old editions are unavailable.",
    )
    offset: int = Field(default=0, ge=0, le=100000, description="Use next_offset from the previous catalog page.")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum catalog entries in this page.")
    tax_year: int | None = Field(
        default=None, ge=1800, le=2200,
        description="Requested applicability filter. Current IRS library applicability is unknown, so this filter returns an explicit error rather than guessing from revision dates.",
    )


class ReferenceReadRequest(ReferenceRequest):
    catalog: Literal["current", "prior", "bulletins"] | None = Field(
        default="current", description="Current catalog by default. For an explicitly selected recent historical result, pass its matching prior or bulletins catalog; the server controls permitted editions.",
    )
    document_id: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Full 64-character document ID returned by reference_lookup; never a filename, path, URL or abbreviated ID.",
    )
    page: int = Field(default=1, ge=1, le=10000, description="PDF page number, starting at 1.")
    offset: int = Field(
        default=0, ge=0, le=200000,
        description="Character offset within this page. Use next_offset to read the rest of a long page.",
    )
    limit: int = Field(default=1200, ge=1, le=4000, description="Maximum extracted page-text characters.")
    revision_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$",
        description="Optional full revision checksum from the catalog to pin the cited edition.",
    )


def _require_reference_service(request: Request) -> None:
    unavailable = request.app.state.capabilities.unavailable(["audrey_references"])
    if unavailable:
        raise HTTPException(status_code=503, detail={
            "error": "capability_unavailable", "components": unavailable,
        })


async def _proxy(request: Request, path: str, body: dict) -> dict:
    try:
        response = await request.app.state.audrey.post(path, json=body, timeout=25)
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Reference read timed out.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Audrey reference service is unavailable.") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "Reference read failed.")
        except (ValueError, AttributeError):
            detail = "Reference read failed."
        raise HTTPException(status_code=response.status_code, detail=detail)
    try:
        result = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Reference service returned invalid data.") from exc
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="Reference service returned invalid data.")
    return result


@router.post("/reference_lookup", operation_id="reference_lookup", description=(
    "Look up or browse the owned public English IRS reference archive by exact "
    "product number or catalog title words. Omit product_number/query to browse. "
    "Use library_id='irs-federal'; the current catalog is the default. Explicit "
    "prior/bulletins lookups use only recent historical editions permitted by the "
    "server; the model cannot change that window or open older archive documents. "
    "Follow returned document IDs and next_offset. This searches catalog metadata, "
    "not PDF contents, and makes no web, embedding "
    "or model calls. Revisions are retained separately; revision dates are not tax "
    "years. Tax-year applicability is unknown unless explicitly established from "
    "the document; a tax_year filter is currently rejected. Read relevant pages "
    "with read_reference and cite the returned edition/page/source. These tools "
    "read public references only, never a person's uploads, projects or tax records."
))
async def reference_lookup(body: ReferenceLookupRequest, request: Request) -> dict:
    _require_reference_service(request)
    return await _proxy(request, "/v1/references/catalog", body.model_dump())


@router.post("/read_reference", operation_id="read_reference", description=(
    "Read extracted text from one actual page of a retained public English IRS PDF. "
    "First obtain its full document_id from reference_lookup; only current catalog "
    "items and server-permitted recent historical editions are available. Older "
    "archive IDs are refused. For a recent historical lookup result, explicitly "
    "pass its matching prior or bulletins catalog; otherwise current scope applies. "
    "Optionally pin its "
    "revision_sha256. Follow next_offset for more text on the same page, and page "
    "numbers for other pages. Preserve returned edition, page and official-source "
    "citations. Revision dates are not tax years: applicability remains unknown "
    "until verified from the source. This is text extraction only, with no OCR or "
    "vision; blank/scanned pages and unreviewed pages cannot establish wording. "
    "No web fetch, embedding or model calls occur. It cannot open private uploads "
    "or any model-chosen filesystem path."
))
async def read_reference(body: ReferenceReadRequest, request: Request) -> dict:
    _require_reference_service(request)
    return await _proxy(request, "/v1/references/read", body.model_dump())
