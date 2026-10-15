"""Bounded public-reference operations; private evidence has separate routes."""
from __future__ import annotations

import asyncio
import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from audrey.auth import require_principal, require_service
from audrey.references.library import (
    ReferenceLibraryRegistry,
    ReferenceNotFound,
    ReferenceQueryError,
    ReferenceReadError,
    ReferenceUnavailable,
)

router = APIRouter(prefix="/v1/references", tags=["references"])
_read_slots = asyncio.Semaphore(2)
# Fits the deployed fast-worker tool context while preserving valid JSON and
# its next_offset. A clipped JSON string would lose both citation and paging.
_RESULT_CHARS = 3500


class LibraryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    library_id: str = Field(default="irs-federal", pattern=r"^[a-z][a-z0-9-]{0,63}$")


class CatalogRequest(LibraryRequest):
    product_number: str | None = Field(default=None, min_length=1, max_length=100)
    query: str | None = Field(default=None, min_length=1, max_length=200)
    catalog: Literal["current", "prior", "bulletins"] | None = "current"
    offset: int = Field(default=0, ge=0, le=100000)
    limit: int = Field(default=5, ge=1, le=20)
    tax_year: int | None = Field(default=None, ge=1800, le=2200)


class ReadRequest(LibraryRequest):
    document_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog: Literal["current", "prior", "bulletins"] | None = "current"
    revision_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    page: int = Field(default=1, ge=1, le=10000)
    offset: int = Field(default=0, ge=0, le=200000)
    limit: int = Field(default=1200, ge=1, le=4000)


def _registry(request: Request) -> ReferenceLibraryRegistry:
    registry = getattr(request.app.state, "reference_libraries", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Reference libraries are not initialized.")
    return registry


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ReferenceNotFound):
        return HTTPException(status_code=404, detail="Reference document or library not found.")
    if isinstance(exc, ReferenceUnavailable):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=422, detail=str(exc))


def _size(result: dict) -> int:
    return len(json.dumps(result, ensure_ascii=False))


def _bounded_catalog(result: dict) -> dict:
    result = {**result, "items": list(result["items"])}
    while _size(result) > _RESULT_CHARS and len(result["items"]) > 1:
        result["items"].pop()
        result.update(
            returned=len(result["items"]), has_more=True,
            next_offset=result["offset"] + len(result["items"]),
        )
    if _size(result) > _RESULT_CHARS:
        raise ReferenceReadError("Reference metadata exceeds the bounded result size.")
    return result


def _bounded_read(result: dict) -> dict:
    result = dict(result)
    if _size(result) <= _RESULT_CHARS:
        return result
    original = result["text"]
    low, high = 0, len(original)
    while low < high:
        candidate = (low + high + 1) // 2
        result.update(
            text=original[:candidate], next_offset=result["offset"] + candidate, has_more=True,
        )
        if _size(result) <= _RESULT_CHARS:
            low = candidate
        else:
            high = candidate - 1
    result.update(text=original[:low], next_offset=result["offset"] + low, has_more=True)
    if low == 0 or _size(result) > _RESULT_CHARS:
        raise ReferenceReadError("Reference metadata exceeds the bounded result size.")
    return result


@router.get("/libraries")
async def libraries(request: Request, _: object = Depends(require_principal)) -> dict:
    return await asyncio.to_thread(_registry(request).list_libraries)


@router.post("/catalog")
async def catalog(
    body: CatalogRequest, request: Request, _: None = Depends(require_service),
) -> dict:
    try:
        result = await asyncio.to_thread(
            _registry(request).catalog, body.library_id,
            product_number=body.product_number, query=body.query,
            catalog=body.catalog, offset=body.offset, limit=body.limit, year=body.tax_year,
        )
        return _bounded_catalog(result)
    except (ReferenceNotFound, ReferenceUnavailable, ReferenceQueryError, ReferenceReadError) as exc:
        raise _http_error(exc) from exc


@router.post("/read")
async def read(
    body: ReadRequest, request: Request, _: None = Depends(require_service),
) -> dict:
    try:
        async with _read_slots:
            result = await asyncio.to_thread(
                _registry(request).read, body.library_id, body.document_id,
                catalog=body.catalog, page=body.page, offset=body.offset, limit=body.limit,
                revision_sha256=body.revision_sha256,
            )
        return _bounded_read(result)
    except (ReferenceNotFound, ReferenceUnavailable, ReferenceQueryError, ReferenceReadError) as exc:
        raise _http_error(exc) from exc
