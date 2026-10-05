"""Owner-visible native approval controls for document jobs."""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from audrey.app_state import (
    ApplicationStore,
    DocumentApprovalError,
    DocumentApprovalExpiredError,
    DocumentApprovalRecord,
    DocumentConflictError,
    DocumentJobRecord,
    InvalidDocumentOperationError,
)
from audrey.auth import require_provider_principal, require_scope
from audrey.documents import (
    DOCX_MIME,
    PROJECT_BRIEF_TEMPLATE_ID,
    TEMPLATE_OPERATION,
    DocumentTemplateCatalog,
    DocumentTemplateError,
    ProjectBriefFields,
    TemplateDocumentArguments,
)
from audrey.identity import Principal

router = APIRouter(prefix="/document-jobs", tags=["document-jobs"])
_document_access = require_scope("compat:full")


class DocumentApprovalResponse(BaseModel):
    id: str
    operation_digest: str
    expires_at: str
    decision: str
    decided_at: str | None
    used_at: str | None


class DocumentJobResponse(BaseModel):
    id: str
    input_version_id: str
    output_version_id: str | None
    operation: str
    operation_digest: str
    output_mime: str
    summary: str
    preview: str
    status: str
    attempts: int
    error_code: str
    created_at: str
    updated_at: str
    completed_at: str | None
    approval: DocumentApprovalResponse


class DocumentJobListResponse(BaseModel):
    items: list[DocumentJobResponse]


class DocumentDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved", "rejected"]
    operation_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class DocumentTemplateResponse(BaseModel):
    id: str
    name: str
    description: str
    version: int
    fields: list[str]


class DocumentTemplateListResponse(BaseModel):
    items: list[DocumentTemplateResponse]


class CreateProjectBriefRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    template_id: Literal["project-brief-v1"] = PROJECT_BRIEF_TEMPLATE_ID
    filename: str = Field(min_length=1, max_length=255)
    fields: ProjectBriefFields
    idempotency_key: str = Field(min_length=1, max_length=200)


def _store(request: Request) -> ApplicationStore:
    store = getattr(request.app.state, "application_store", None)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Audrey application state is not initialized.",
        )
    return store


def _catalog(request: Request) -> DocumentTemplateCatalog:
    catalog = getattr(request.app.state, "document_templates", None)
    if catalog is None:
        raise HTTPException(
            status_code=503,
            detail="Document templates are not initialized.",
        )
    return catalog


def _wake_worker(request: Request) -> None:
    worker = getattr(request.app.state, "document_template_worker", None)
    if worker is not None:
        worker.wake()


def _response(
    job: DocumentJobRecord,
    approval: DocumentApprovalRecord,
) -> DocumentJobResponse:
    return DocumentJobResponse(
        id=job.job_id,
        input_version_id=job.input_version_id,
        output_version_id=job.output_version_id,
        operation=job.operation,
        operation_digest=job.operation_digest,
        output_mime=job.output_mime,
        summary=job.summary,
        preview=job.preview,
        status=job.status,
        attempts=job.attempts,
        error_code=job.error_code,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        approval=DocumentApprovalResponse(
            id=approval.approval_id,
            operation_digest=approval.operation_digest,
            expires_at=approval.expires_at,
            decision=approval.decision,
            decided_at=approval.decided_at,
            used_at=approval.used_at,
        ),
    )


@router.get("/templates", response_model=DocumentTemplateListResponse)
async def list_document_templates(
    request: Request,
    _principal: Principal = Depends(_document_access),
) -> DocumentTemplateListResponse:
    return DocumentTemplateListResponse(items=[
        DocumentTemplateResponse(
            id=descriptor.template_id,
            name=descriptor.name,
            description=descriptor.description,
            version=descriptor.version,
            fields=list(descriptor.fields),
        )
        for descriptor in _catalog(request).list()
    ])


@router.post("/template-to-docx", response_model=DocumentJobResponse, status_code=201)
async def create_project_brief(
    payload: CreateProjectBriefRequest,
    request: Request,
    principal: Principal = Depends(_document_access),
) -> DocumentJobResponse:
    catalog = _catalog(request)
    try:
        descriptor = catalog.descriptor(payload.template_id)
        arguments = TemplateDocumentArguments(
            template_id=descriptor.template_id,
            template_sha256=descriptor.content_sha256,
            filename=payload.filename,
            fields=payload.fields,
        )
        source = await _store(request).document_tools.register_source_version(
            user_id=principal.user_id,
            file_id=f"template_{descriptor.template_id}",
            filename=f"{descriptor.template_id}.docx",
            mime=DOCX_MIME,
            bytes_count=descriptor.bytes_count,
            content_sha256=descriptor.content_sha256,
        )
        job, approval = await _store(request).document_tools.create_request(
            user_id=principal.user_id,
            input_version_id=source.version_id,
            operation=TEMPLATE_OPERATION,
            arguments=arguments.model_dump(mode="json"),
            output_mime=DOCX_MIME,
            summary=f"Create {payload.filename} from the reviewed Project brief template.",
            preview=(
                f"Project brief: {payload.fields.title} for {payload.fields.prepared_for}; "
                f"{len(payload.fields.objectives)} objectives and "
                f"{len(payload.fields.next_steps)} next steps."
            ),
            idempotency_key=payload.idempotency_key,
            requested_by_kind="user",
            requested_by_id=principal.user_id,
        )
    except (
        DocumentTemplateError,
        InvalidDocumentOperationError,
        ValidationError,
    ) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DocumentConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _response(job, approval)


@router.get("", response_model=DocumentJobListResponse)
async def list_document_jobs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    principal: Principal = Depends(_document_access),
) -> DocumentJobListResponse:
    records = await _store(request).document_tools.list_jobs(
        user_id=principal.user_id,
        limit=limit,
    )
    return DocumentJobListResponse(
        items=[_response(job, approval) for job, approval in records]
    )


@router.get("/{job_id}", response_model=DocumentJobResponse)
async def get_document_job(
    job_id: str,
    request: Request,
    principal: Principal = Depends(_document_access),
) -> DocumentJobResponse:
    record = await _store(request).document_tools.get_job(
        user_id=principal.user_id,
        job_id=job_id,
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Document job not found.")
    return _response(*record)


@router.post("/{job_id}/decision", response_model=DocumentJobResponse)
async def decide_document_job(
    job_id: str,
    payload: DocumentDecisionRequest,
    request: Request,
    principal: Principal = Depends(require_provider_principal),
) -> DocumentJobResponse:
    if "bots" in principal.groups:
        raise HTTPException(
            status_code=403,
            detail="Bot accounts cannot approve document operations.",
        )
    try:
        record = await _store(request).document_tools.decide(
            user_id=principal.user_id,
            job_id=job_id,
            actor_user_id=principal.user_id,
            operation_digest=payload.operation_digest,
            decision=payload.decision,
        )
    except DocumentApprovalExpiredError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except DocumentApprovalError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidDocumentOperationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Document job not found.")
    if record[0].status == "queued":
        _wake_worker(request)
    return _response(*record)


@router.post("/{job_id}/cancel", response_model=DocumentJobResponse)
async def cancel_document_job(
    job_id: str,
    request: Request,
    principal: Principal = Depends(_document_access),
) -> DocumentJobResponse:
    try:
        record = await _store(request).document_tools.cancel(
            user_id=principal.user_id,
            job_id=job_id,
        )
    except DocumentConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Document job not found.")
    return _response(*record)


__all__ = [
    "CreateProjectBriefRequest",
    "DocumentApprovalResponse",
    "DocumentDecisionRequest",
    "DocumentJobListResponse",
    "DocumentJobResponse",
    "DocumentTemplateListResponse",
    "DocumentTemplateResponse",
    "router",
]
