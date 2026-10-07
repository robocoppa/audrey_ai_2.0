"""Ready, owned file metadata for native chat automatic skill selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from audrey.app_state import AttachmentSnapshot, MessageRecord
from audrey.identity import Principal
from audrey.skills.models import ResolvedSkill
from audrey.skills.registry import SkillRegistry, skill_mode_for_virtual_model
from audrey.skills.selection import SkillEvidence, select_automatic_skill

if TYPE_CHECKING:
    from audrey.project_context import ProjectContextSnapshot


def _file_evidence(
    attachments: tuple[AttachmentSnapshot, ...],
    project_context: ProjectContextSnapshot | None,
) -> tuple[SkillEvidence, ...]:
    snapshots = (
        *attachments,
        *(project_context.files if project_context is not None else ()),
    )
    evidence: list[SkillEvidence] = []
    seen: set[str] = set()
    for file in snapshots:
        if file.file_id in seen:
            continue
        seen.add(file.file_id)
        evidence.append(
            SkillEvidence(
                filename=file.filename,
                kind="document" if file.kind == "text" else file.kind,
            )
        )
    return tuple(evidence)


async def resolve_native_auto_skill(
    request: Request,
    principal: Principal,
    *,
    registry: SkillRegistry | None,
    virtual_model: str,
    attachments: tuple[AttachmentSnapshot, ...],
    project_context: ProjectContextSnapshot | None,
    previous_records: Sequence[MessageRecord],
    prompt: str,
) -> ResolvedSkill | None:
    """Select from file metadata after native ownership and readiness checks.

    Historical snapshots require fresh validation because their files may have
    been removed or returned to processing since the attachment was saved.
    """

    if (
        registry is None
        or not getattr(registry, "auto_select", False)
        or not registry.enabled
        or principal.role == "bot"
        or "bots" in principal.groups
        or virtual_model.startswith("audrey_passthrough/")
    ):
        return None

    evidence_attachments = attachments
    if not evidence_attachments:
        last_attached_message = next(
            (
                record
                for record in reversed(previous_records)
                if record.role == "user" and record.attachments
            ),
            None,
        )
        if last_attached_message is not None:
            # Saved metadata can rule out ordinary chat, but can never authorize
            # selection until those attachment IDs are freshly owner-validated.
            preliminary = _file_evidence(last_attached_message.attachments, project_context)
            if select_automatic_skill(prompt, preliminary) is None:
                return None
            # Keep this import local: native routes import this module too.
            from audrey.routes.app.files import resolve_owned_attachments

            file_ids = tuple(
                dict.fromkeys(
                    attachment.file_id for attachment in last_attached_message.attachments
                )
            )
            try:
                evidence_attachments = await resolve_owned_attachments(
                    request,
                    principal,
                    file_ids,
                )
            except HTTPException:
                # Never silently shrink missing historical evidence to an older
                # attachment or a different file available in the project.
                return None

    evidence = _file_evidence(evidence_attachments, project_context)
    if not evidence:
        return None
    return registry.resolve_auto(
        prompt=prompt,
        virtual_model=virtual_model,
        mode=skill_mode_for_virtual_model(virtual_model),
        files=evidence,
    )
