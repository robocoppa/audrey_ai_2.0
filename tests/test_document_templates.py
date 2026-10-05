"""Bounded server-owned DOCX template contracts."""

from __future__ import annotations

import io
import zipfile

import pytest
from docx import Document
from pydantic import ValidationError

from audrey.documents import (
    DOCX_MIME,
    PROJECT_BRIEF_TEMPLATE_ID,
    DocumentTemplateCatalog,
    DocumentTemplateError,
    ProjectBriefFields,
    TemplateDocumentArguments,
)


def _arguments(catalog: DocumentTemplateCatalog) -> TemplateDocumentArguments:
    descriptor = catalog.descriptor(PROJECT_BRIEF_TEMPLATE_ID)
    return TemplateDocumentArguments(
        template_id=descriptor.template_id,
        template_sha256=descriptor.content_sha256,
        filename="North Star Project Brief.docx",
        fields=ProjectBriefFields(
            title="North Star rollout",
            prepared_for="Example Operations",
            prepared_on="October 4, 2026",
            summary="A concise plan for the first controlled rollout.",
            objectives=["Confirm ownership", "Measure adoption"],
            next_steps=["Approve the pilot", "Review results after two weeks"],
        ),
    )


def test_project_brief_template_renders_and_reopens_with_expected_plain_text():
    catalog = DocumentTemplateCatalog()
    descriptor = catalog.descriptor(PROJECT_BRIEF_TEMPLATE_ID)

    assert descriptor.version == 1
    assert descriptor.bytes_count > 0
    assert len(descriptor.content_sha256) == 64
    rendered = catalog.render(_arguments(catalog))

    assert rendered.content.startswith(b"PK")
    assert rendered.content_sha256
    assert all(value in rendered.text for value in rendered.expected_text)
    reopened = Document(io.BytesIO(rendered.content))
    assert reopened.core_properties.title == "Audrey Project Brief Template"
    with zipfile.ZipFile(io.BytesIO(rendered.content)) as archive:
        names = set(archive.namelist())
        assert "word/document.xml" in names
        assert not any(name.startswith("word/media/") for name in names)
        assert not any(
            name.casefold().endswith((
                ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".svg", ".tif",
                ".tiff", ".webp",
            ))
            for name in names
        )
        assert not any("vbaproject" in name.casefold() for name in names)
        relationships = b"".join(
            archive.read(name) for name in names if name.endswith(".rels")
        )
        assert b'TargetMode="External"' not in relationships


def test_template_arguments_reject_paths_unknown_templates_and_control_text():
    catalog = DocumentTemplateCatalog()
    arguments = _arguments(catalog).model_dump()
    arguments["filename"] = "../outside.docx"
    with pytest.raises(ValidationError, match=r"plain \.docx name"):
        TemplateDocumentArguments.model_validate(arguments)

    arguments = _arguments(catalog).model_dump()
    arguments["template_id"] = "unknown"
    with pytest.raises(DocumentTemplateError, match="unknown"):
        catalog.validate_arguments(arguments)

    fields = _arguments(catalog).fields.model_dump()
    fields["summary"] = "contains\x00control"
    with pytest.raises(ValidationError, match="control"):
        ProjectBriefFields.model_validate(fields)


def test_template_digest_change_requires_a_new_approval():
    catalog = DocumentTemplateCatalog()
    arguments = _arguments(catalog).model_dump()
    arguments["template_sha256"] = "0" * 64

    with pytest.raises(DocumentTemplateError, match="digest"):
        catalog.validate_arguments(arguments)


def _rewrite_package(
    content: bytes,
    *,
    replacements: dict[str, bytes] | None = None,
    extra: tuple[str, bytes] | None = None,
) -> bytes:
    source = io.BytesIO(content)
    output = io.BytesIO()
    with zipfile.ZipFile(source) as incoming, zipfile.ZipFile(output, "w") as outgoing:
        for info in incoming.infolist():
            value = (replacements or {}).get(info.filename, incoming.read(info.filename))
            outgoing.writestr(info, value)
        if extra is not None:
            outgoing.writestr(*extra)
    return output.getvalue()


def test_template_verification_rejects_images_and_external_relationships():
    catalog = DocumentTemplateCatalog()
    arguments = _arguments(catalog)
    rendered = catalog.render(arguments)
    with_image = _rewrite_package(
        rendered.content,
        extra=("word/media/image1.png", b"not-an-image"),
    )
    with pytest.raises(DocumentTemplateError, match="active content"):
        catalog.verify(with_image, arguments)

    with zipfile.ZipFile(io.BytesIO(rendered.content)) as archive:
        relationships_name = next(
            name for name in archive.namelist() if name.endswith(".rels")
        )
        relationships = archive.read(relationships_name).replace(
            b"</Relationships>",
            b'<Relationship Id="external" Type="example" '
            b'Target="https://example.com" TargetMode="External"/></Relationships>',
        )
    with_external = _rewrite_package(
        rendered.content,
        replacements={relationships_name: relationships},
    )
    with pytest.raises(DocumentTemplateError, match="external relationship"):
        catalog.verify(with_external, arguments)


def test_template_output_mime_is_docx():
    assert DOCX_MIME == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
