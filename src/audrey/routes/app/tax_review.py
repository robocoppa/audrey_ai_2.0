"""Owner-only saved Tax review case and explicit user confirmation actions."""
import asyncio
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from audrey.auth import require_scope
from audrey.identity import Principal
from audrey.tax_review import (
    CreateTaxCaseRequest,
    TaxCaseActionRequest,
    apply_tax_case_action,
    create_tax_case,
    ensure_tax_project,
    get_project_tax_cases,
    get_tax_case,
    prepare_tax_workbook,
)

router = APIRouter(tags=["application-tax-review"])
_access = require_scope("compat:full")


@router.get("/conversations/{conversation_id}/tax-review")
async def get_case(
    conversation_id: str, request: Request, principal: Principal = Depends(_access),
) -> dict:
    return {"case": await get_tax_case(request.app, principal, conversation_id)}


@router.get("/projects/{project_id}/tax-reviews")
async def project_cases(
    project_id: str, request: Request, principal: Principal = Depends(_access),
) -> dict:
    return {"items": await get_project_tax_cases(request.app, principal, project_id)}


@router.post("/conversations/{conversation_id}/tax-review", status_code=201)
async def create_case(
    conversation_id: str, body: CreateTaxCaseRequest, request: Request,
    principal: Principal = Depends(_access),
) -> dict:
    return await create_tax_case(request.app, principal, conversation_id, body)


@router.post("/conversations/{conversation_id}/tax-review/project")
async def ensure_case_project(
    conversation_id: str, request: Request, principal: Principal = Depends(_access),
) -> dict:
    return await ensure_tax_project(request.app, principal, conversation_id)


@router.post("/conversations/{conversation_id}/tax-review/actions")
async def case_action(
    conversation_id: str, body: TaxCaseActionRequest, request: Request,
    principal: Principal = Depends(_access),
) -> dict:
    return await apply_tax_case_action(request.app, principal, conversation_id, body)


class TaxWorkbookRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=1)


@router.post("/conversations/{conversation_id}/tax-review/exports")
async def prepare_workbook(
    conversation_id: str, body: TaxWorkbookRequest, request: Request,
    principal: Principal = Depends(_access),
) -> dict:
    return {"export": await prepare_tax_workbook(
        request.app, principal, conversation_id, expected_revision=body.expected_revision,
    )}


@router.get("/conversations/{conversation_id}/tax-review/exports/{export_id}/download")
async def download_workbook(
    conversation_id: str, export_id: str, request: Request,
    principal: Principal = Depends(_access),
) -> StreamingResponse:
    from audrey.tax_exports import TaxExportError

    case = await get_tax_case(request.app, principal, conversation_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Tax review case not found.")
    exports = getattr(request.app.state, "tax_review_exports", None)
    if exports is None:
        raise HTTPException(status_code=503, detail="Workbook downloads are temporarily unavailable.")
    try:
        artifact = await exports.download(user_id=principal.user_id, case=case, export_id=export_id)
    except TaxExportError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None
    # Rendering ran outside the database transaction. Reauthorize before serving
    # bytes in case the account, conversation or project changed in that await.
    try:
        current = await get_tax_case(request.app, principal, conversation_id)
        if current is None or current["id"] != case["id"]:
            await exports.delete_case(user_id=principal.user_id, case_id=case["id"])
            raise HTTPException(status_code=404, detail="Tax review case not found.")
    except BaseException:
        artifact.close()
        raise

    async def content():
        try:
            while chunk := await asyncio.to_thread(artifact.file.read, 64 * 1024):
                yield chunk
        finally:
            artifact.close()

    # The file was opened while the cache lock was held. Its descriptor remains
    # valid if eviction/deletion unlinks the path during an authorized download.
    return StreamingResponse(
        content(), background=BackgroundTask(artifact.close),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
                 "Content-Disposition": f'attachment; filename="{artifact.filename}"',
                 "Content-Length": str(artifact.size),
                 "X-Tax-Case-Revision": str(artifact.case_revision)},
    )


@router.get("/conversations/{conversation_id}/tax-review/exports/{export_id}/preview")
async def preview_workbook(
    conversation_id: str, export_id: str, request: Request,
    sheet: int | None = Query(default=None, ge=0, le=7),
    row_offset: int = Query(default=0, ge=0, le=10_000),
    principal: Principal = Depends(_access),
):
    import io

    from fastapi.responses import JSONResponse

    from audrey.file_previews import MAX_SOURCE_BYTES, workbook_preview
    from audrey.routes.files import _settle_private_persistence
    from audrey.tax_exports import TaxExportError

    case = await get_tax_case(request.app, principal, conversation_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Tax review case not found.")
    exports = getattr(request.app.state, "tax_review_exports", None)
    if exports is None:
        raise HTTPException(status_code=503, detail="Workbook previews are temporarily unavailable.")
    try:
        artifact = await exports.download(user_id=principal.user_id, case=case, export_id=export_id)
    except TaxExportError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None
    try:
        if artifact.size > MAX_SOURCE_BYTES:
            raise ValueError("Preview limit")
        # Finish the short descriptor read before closing it on cancellation.
        data = await _settle_private_persistence(asyncio.to_thread(artifact.file.read, MAX_SOURCE_BYTES + 1))
        if len(data) > MAX_SOURCE_BYTES:
            raise ValueError("Preview limit")
        result = await asyncio.to_thread(workbook_preview, io.BytesIO(data), sheet=sheet, row_offset=row_offset)
    except Exception:  # noqa: BLE001 - never expose parser contents or host paths
        raise HTTPException(status_code=422, detail="Preview is unavailable. You can still download the workbook.") from None
    finally:
        artifact.close()
    current = await get_tax_case(request.app, principal, conversation_id)
    if current is None or current["id"] != case["id"]:
        raise HTTPException(status_code=404, detail="Tax review case not found.")
    return JSONResponse(
        {"id": export_id, "filename": artifact.filename, **result},
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff",
                 "X-Tax-Case-Revision": str(artifact.case_revision)},
    )


class PrivateTaskAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=1)
    action: Literal["pause", "cancel"]


@router.get("/conversations/{conversation_id}/private-task")
async def get_private_task(conversation_id: str, request: Request,
                           principal: Principal = Depends(_access)) -> dict:
    from audrey.app_state.private_tasks import PrivateTaskAccessError
    from audrey.private_reviews import public_private_task
    from audrey.routes.app.runs import _manager

    try:
        task = await _manager(request).private_reviews.latest(principal.user_id, conversation_id)
    except PrivateTaskAccessError:
        raise HTTPException(status_code=404, detail="Private review not found.") from None
    return {"task": public_private_task(task)}


@router.post("/private-tasks/{task_id}/actions")
async def private_task_action(task_id: str, body: PrivateTaskAction, request: Request,
                              principal: Principal = Depends(_access)) -> dict:
    from audrey.app_state.private_tasks import PrivateTaskAccessError, PrivateTaskError
    from audrey.private_reviews import public_private_task
    from audrey.routes.app.runs import _manager

    try:
        task = await _manager(request).private_reviews.control(
            user_id=principal.user_id, task_id=task_id,
            expected_revision=body.expected_revision, action=body.action,
        )
    except PrivateTaskAccessError:
        raise HTTPException(status_code=404, detail="Private review not found.") from None
    except PrivateTaskError:
        raise HTTPException(status_code=409, detail="Private review changed; reload it.") from None
    if task is None:
        raise HTTPException(status_code=404, detail="Private review not found.")
    return {"task": public_private_task(task)}
