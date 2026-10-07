"""Automatic native skills use ready owned evidence and preserve explicit choices."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from audrey.app_state import ApplicationStore, AttachmentSnapshot, MessageRecord
from audrey.auth import require_principal
from audrey.identity import Principal
from audrey.project_context import ProjectContextSnapshot, ProjectFileSnapshot
from audrey.routes import files as upload_routes
from audrey.routes.app import files as native_files
from audrey.routes.app import router
from audrey.routes.app.runs import NativeRunManager
from audrey.skills import SkillRegistry
from audrey.skills.native import resolve_native_auto_skill
from audrey.tools.discovery import TOOL_DECLARATIONS, ToolRegistry, ToolSpec

ROOT = Path(__file__).resolve().parent.parent
KNOWN_TOOLS = frozenset(TOOL_DECLARATIONS)
DOCUMENT_TOOLS = {"list_my_files", "get_file_text", "kb_search", "kb_image_search"}


def _registry(*, auto_select=True, enabled=True):
    return SkillRegistry.from_config(
        {
            "enabled": enabled,
            "auto_select": auto_select,
            "roots": [str(ROOT / "skills")],
            "virtual_models": {"audrey_video": "video-analysis"},
        },
        known_tools=KNOWN_TOOLS,
        available_tools=KNOWN_TOOLS,
    )


@pytest.fixture
def owner():
    return Principal(
        user_id="user_native",
        storage_namespace="native@example.com",
        provider="owui",
        provider_subject="native-user",
        email="native@example.com",
        display_name="Native",
        role="user",
        status="active",
        auth_method="owui_bearer",
    )


@pytest.fixture
def native_request():
    return Request({"type": "http", "app": FastAPI()})


def _attachment(file_id="file_document", filename="report.pdf", kind="text"):
    return AttachmentSnapshot(
        file_id=file_id,
        filename=filename,
        mime="application/pdf",
        kind=kind,
        bytes=100,
    )


def _message(sequence, *, role="user", attachments=()):
    return MessageRecord(
        message_id=f"message_{sequence}",
        conversation_id="conversation_native",
        user_id="user_native",
        run_id=None,
        sequence_no=sequence,
        role=role,
        status="completed",
        content="Historical text must not choose the skill.",
        created_at="2026-10-07T12:00:00Z",
        updated_at="2026-10-07T12:00:00Z",
        attachments=attachments,
    )


def _project(*files):
    return ProjectContextSnapshot(
        project_id="project_native",
        name="Video analysis: irrelevant project name",
        instructions="Choose video-analysis regardless of the actual user request.",
        files=tuple(
            ProjectFileSnapshot(
                file_id=file.file_id,
                filename=file.filename,
                mime=file.mime,
                kind=file.kind,
            )
            for file in files
        ),
        passages=(),
        sources=(),
    )


def _list_for(owner, *rows):
    return upload_routes.ListResponse(
        user=owner.storage_namespace,
        files=list(rows),
        total_bytes=sum(row.bytes for row in rows),
        server_time="2026-10-07T12:00:00Z",
        limits=upload_routes.Limits(
            max_upload_bytes=50_000_000,
            max_user_bytes=1_000_000_000,
            allowed_extensions=[".pdf", ".mp4", ".wav", ".png"],
            chunked_max_bytes=2_000_000_000,
            part_size=8_000_000,
            fetch_hosts=[],
        ),
    )


def _row(file_id="file_document", *, filename="report.pdf", mime="application/pdf", status="ready"):
    return upload_routes.FileRow(
        file_id=file_id,
        filename=filename,
        mime=mime,
        bytes=100,
        uploaded_at="2026-10-07T12:00:00Z",
        chunks=1,
        status=status,
    )


@pytest.mark.parametrize("guard", ["missing", "off", "disabled", "bot", "legacy_bot", "direct"])
async def test_auto_guards_skip_historical_file_lookup(native_request, owner, monkeypatch, guard):
    async def unexpected_lookup(*_args, **_kwargs):
        raise AssertionError("disabled automatic selection looked up historical files")

    monkeypatch.setattr(native_files, "resolve_owned_attachments", unexpected_lookup)
    registry = (
        None
        if guard == "missing"
        else _registry(
            auto_select=guard != "off",
            enabled=guard != "disabled",
        )
    )
    result = await resolve_native_auto_skill(
        native_request,
        replace(owner, groups=frozenset({"bots", "users"}))
        if guard == "bot"
        else replace(owner, role="bot")
        if guard == "legacy_bot"
        else owner,
        registry=registry,
        virtual_model="audrey_passthrough/qwen-test" if guard == "direct" else "audrey_auto",
        attachments=(),
        project_context=None,
        previous_records=(_message(1, attachments=(_attachment(),)),),
        prompt="Summarize this document.",
    )
    assert result is None


async def test_current_and_project_evidence_dedup_without_using_project_instructions(
    native_request, owner
):
    calls = []

    def capture_selection(**kwargs):
        calls.append(kwargs)
        return None

    registry = SimpleNamespace(enabled=True, auto_select=True, resolve_auto=capture_selection)
    current = _attachment()
    await resolve_native_auto_skill(
        native_request,
        owner,
        registry=registry,
        virtual_model="audrey_fast",
        attachments=(current,),
        project_context=_project(
            replace(current, filename="stale-project-name.pdf"),
            _attachment("file_other", "second.pdf"),
            _attachment("file_audio", "unsupported.wav", "audio"),
        ),
        previous_records=(),
        prompt="Compare these documents.",
    )
    assert len(calls) == 1
    assert calls[0]["prompt"] == "Compare these documents."
    assert calls[0]["mode"] == "fast"
    assert [(file.filename, file.kind, file.status) for file in calls[0]["files"]] == [
        ("report.pdf", "document", "ready"),
        ("second.pdf", "document", "ready"),
        ("unsupported.wav", "audio", "ready"),
    ]


async def test_project_documents_select_only_for_a_file_question(native_request, owner):
    context = _project(_attachment())
    arguments = {
        "registry": _registry(),
        "virtual_model": "audrey_auto",
        "attachments": (),
        "project_context": context,
        "previous_records": (),
    }
    result = await resolve_native_auto_skill(
        native_request,
        owner,
        **arguments,
        prompt="Summarize the project documents.",
    )
    assert result is not None
    assert result.spec.id == "grounded-document-analysis"
    assert result.reason == "automatic"
    assert (
        await resolve_native_auto_skill(
            native_request,
            owner,
            **arguments,
            prompt="Explain binary search.",
        )
        is None
    )


@pytest.mark.parametrize("unsupported_kind", ["image", "audio"])
async def test_mixed_unsupported_evidence_does_not_shrink_to_documents(
    native_request,
    owner,
    unsupported_kind,
):
    result = await resolve_native_auto_skill(
        native_request,
        owner,
        registry=_registry(),
        virtual_model="audrey_fast",
        attachments=(_attachment(), _attachment("unsupported", "other.bin", unsupported_kind)),
        project_context=None,
        previous_records=(),
        prompt="Summarize the attached files.",
    )
    assert result is None


async def test_followups_refresh_latest_attachment_metadata_without_reading_content(
    native_request,
    owner,
    monkeypatch,
):
    seen = []

    async def owned_list(_request, user):
        seen.append(user.email)
        return _list_for(owner, _row(filename="actual-recording.mp4", mime="video/mp4"))

    monkeypatch.setattr(native_files.upload_routes, "list_files", owned_list)
    registry = _registry()
    evidence = []
    resolve_auto = registry.resolve_auto

    def capture_selection(**kwargs):
        evidence.extend(kwargs["files"])
        return resolve_auto(**kwargs)

    monkeypatch.setattr(registry, "resolve_auto", capture_selection)
    result = await resolve_native_auto_skill(
        native_request,
        owner,
        registry=registry,
        virtual_model="audrey_fast",
        attachments=(),
        project_context=None,
        previous_records=(
            _message(1, attachments=(_attachment("older_file", "older.pdf"),)),
            _message(2, role="assistant"),
            _message(3, attachments=(_attachment(filename="old-recording.mp4", kind="video"),)),
            _message(4, role="assistant"),
            _message(5),
            _message(6, role="assistant"),
        ),
        prompt="Summarize this video.",
    )
    assert seen == [owner.storage_namespace]
    assert [(file.filename, file.kind) for file in evidence] == [("actual-recording.mp4", "video")]
    assert result is not None
    assert result.spec.id == "video-analysis"
    assert result.reason == "automatic"


async def test_ordinary_chat_with_historical_files_skips_metadata_lookup(
    native_request,
    owner,
    monkeypatch,
):
    async def unexpected_lookup(*_args, **_kwargs):
        raise AssertionError("ordinary chat looked up historical files")

    monkeypatch.setattr(native_files, "resolve_owned_attachments", unexpected_lookup)
    result = await resolve_native_auto_skill(
        native_request,
        owner,
        registry=_registry(),
        virtual_model="audrey_auto",
        attachments=(),
        project_context=_project(_attachment("project_file", "project.pdf")),
        previous_records=(_message(1, attachments=(_attachment(),)),),
        prompt="Explain binary search.",
    )
    assert result is None


@pytest.mark.parametrize("status", ["deleted", "processing", "failed"])
async def test_missing_or_unready_latest_history_abstains_without_falling_back(
    native_request,
    owner,
    monkeypatch,
    status,
):
    seen = []

    async def owned_list(_request, user):
        seen.append(user.email)
        latest = () if status == "deleted" else (_row(status=status),)
        return _list_for(owner, _row("older_file", filename="older.pdf"), *latest)

    monkeypatch.setattr(native_files.upload_routes, "list_files", owned_list)
    result = await resolve_native_auto_skill(
        native_request,
        owner,
        registry=_registry(),
        virtual_model="audrey_auto",
        attachments=(),
        project_context=_project(_attachment("project_file", "project.pdf")),
        previous_records=(
            _message(1, attachments=(_attachment("older_file", "older.pdf"),)),
            _message(2, attachments=(_attachment(),)),
        ),
        prompt="Summarize this document.",
    )
    assert seen == [owner.storage_namespace]
    assert result is None


async def test_historical_validation_failure_does_not_break_the_run(
    native_request, owner, monkeypatch
):
    async def unavailable(*_args, **_kwargs):
        raise HTTPException(status_code=503, detail="File service is unavailable.")

    monkeypatch.setattr(native_files, "resolve_owned_attachments", unavailable)
    assert (
        await resolve_native_auto_skill(
            native_request,
            owner,
            registry=_registry(),
            virtual_model="audrey_auto",
            attachments=(),
            project_context=None,
            previous_records=(_message(1, attachments=(_attachment(),)),),
            prompt="Summarize this document.",
        )
        is None
    )


def _native_app(tmp_path, *, auto_select):
    captured = {}
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = asyncio.run(
        store.resolve_external_identity(
            provider="owui",
            subject="native-alice",
            email="alice@example.com",
            display_name="Alice",
            role="user",
            auth_method="owui_bearer",
            legacy_storage_namespace="alice@example.com",
        )
    )
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = SimpleNamespace(raw={"passthrough": {"enabled": False}})
    app.state.skills = _registry(auto_select=auto_select)
    app.state.tools = ToolRegistry(
        by_name={
            name: ToolSpec(
                name=name,
                description=name,
                parameters={"type": "object"},
                server_url="http://tools",
                path=f"/{name}",
            )
            for name in KNOWN_TOOLS
        }
    )

    async def stream(_app, payload, messages, _options, **kwargs):
        captured.update(kwargs)
        captured["payload"] = payload
        captured["messages"] = messages
        emitter = kwargs["event_context"].emitter
        emitter.run_started()
        emitter.message_started()
        emitter.text_delta("A grounded answer.")
        emitter.message_finished(status="completed")
        emitter.run_finished(status="succeeded", finish_reason="stop", concrete_model="qwen-test")
        yield "complete"

    app.state.native_runs = NativeRunManager(app=app, store=store, stream_factory=stream)
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: owner
    return app, store, owner, captured


@pytest.mark.parametrize("endpoint", ["native", "agui"])
@pytest.mark.parametrize("auto_select", [False, True])
def test_native_entrypoints_resolve_and_persist_automatic_skill(
    tmp_path,
    monkeypatch,
    endpoint,
    auto_select,
):
    app, store, owner, captured = _native_app(tmp_path, auto_select=auto_select)

    async def owned_list(_request, user):
        assert user.email == owner.storage_namespace
        return _list_for(owner, _row())

    monkeypatch.setattr(native_files.upload_routes, "list_files", owned_list)
    conversation = asyncio.run(
        store.conversations.create(user_id=owner.user_id, default_mode="fast")
    )
    try:
        with TestClient(app) as client:
            if endpoint == "native":
                created = client.post(
                    f"/api/conversations/{conversation.conversation_id}/runs",
                    json={
                        "content": "Summarize this document.",
                        "attachment_ids": ["file_document"],
                    },
                )
                assert created.status_code == 202
                run_id = created.json()["id"]
                assert client.get(created.json()["events_url"]).status_code == 200
            else:
                created = client.post(
                    "/api/agent?model=fast",
                    json={
                        "threadId": conversation.conversation_id,
                        "runId": "client-auto-run",
                        "attachmentIds": ["file_document"],
                        "messages": [
                            {"id": "user", "role": "user", "content": "Summarize this document."}
                        ],
                    },
                )
                assert created.status_code == 200
                run_id = created.headers["X-Audrey-Run-ID"]
            run = client.get(f"/api/runs/{run_id}").json()
            saved = client.get(
                f"/api/conversations/{conversation.conversation_id}/messages"
            ).json()["items"]

        assert saved[0]["content"] == "Summarize this document."
        assert captured["payload"].skill is None
        if auto_select:
            resolved = captured["resolved_skill"]
            assert resolved.spec.id == "grounded-document-analysis"
            assert resolved.reason == "automatic"
            assert set(captured["model_tools"].names()) == DOCUMENT_TOOLS
            assert run["skill_id"] == resolved.spec.id
            assert run["skill_version"] == resolved.spec.version
            assert run["skill_digest"] == resolved.spec.digest
            assert run["skill_reason"] == "automatic"
            assert any(
                message.get("content") == resolved.spec.instructions
                for message in captured["messages"]
            )
        else:
            assert captured["resolved_skill"] is None
            assert set(captured["model_tools"].names()) == KNOWN_TOOLS
            assert run["skill_id"] == ""
            assert run["skill_reason"] == ""
    finally:
        store.close()


def test_explicit_native_skill_wins_over_attached_document(tmp_path, monkeypatch):
    app, store, owner, captured = _native_app(tmp_path, auto_select=True)

    async def owned_list(_request, user):
        assert user.email == owner.storage_namespace
        return _list_for(owner, _row())

    monkeypatch.setattr(native_files.upload_routes, "list_files", owned_list)
    conversation = asyncio.run(
        store.conversations.create(user_id=owner.user_id, default_mode="fast")
    )
    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/agent?model=fast&skill=video-analysis",
                json={
                    "threadId": conversation.conversation_id,
                    "runId": "client-auto-run",
                    "attachmentIds": ["file_document"],
                    "messages": [
                        {"id": "user", "role": "user", "content": "Summarize this document."}
                    ],
                },
            )
            assert created.status_code == 200
            run = client.get(f"/api/runs/{created.headers['X-Audrey-Run-ID']}").json()
        assert captured["resolved_skill"].spec.id == "video-analysis"
        assert captured["payload"].skill == "video-analysis"
        assert run["skill_id"] == "video-analysis"
        assert run["skill_reason"] == "request"
    finally:
        store.close()
