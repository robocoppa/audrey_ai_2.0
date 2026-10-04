"""Schema 20 exact approvals and immutable document derivation contracts."""

from __future__ import annotations

import datetime as dt
import hashlib
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.app_state import (
    ApplicationStore,
    DocumentApprovalError,
    DocumentApprovalExpiredError,
    DocumentConflictError,
    DocumentLeaseError,
    InvalidDocumentOperationError,
)
from audrey.auth import require_principal, require_provider_principal
from audrey.identity import Principal
from audrey.routes.app import router


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


async def _resolve(
    store: ApplicationStore,
    *,
    subject: str,
    email: str,
) -> Principal:
    return await store.resolve_external_identity(
        provider="owui",
        subject=subject,
        email=email,
        display_name=email.split("@", maxsplit=1)[0].title(),
        role="user",
        auth_method="owui_bearer",
        legacy_storage_namespace=email,
    )


async def _source(store: ApplicationStore, owner: Principal, *, suffix: str = "one"):
    return await store.document_tools.register_source_version(
        user_id=owner.user_id,
        file_id="file_contract",
        filename="contract.txt",
        mime="text/plain",
        bytes_count=len(suffix),
        content_sha256=_hash(suffix),
    )


async def _request(
    store: ApplicationStore,
    owner: Principal,
    version_id: str,
    *,
    key: str = "request-one",
    arguments: dict[str, object] | None = None,
    expires_at: str | None = None,
    requested_by_kind: str = "model",
):
    return await store.document_tools.create_request(
        user_id=owner.user_id,
        input_version_id=version_id,
        operation="template_to_docx",
        arguments=arguments or {"fields": {"client": "Acme", "year": 2026}},
        output_mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        summary="Create a DOCX from the reviewed private template.",
        preview="Client: Acme\nYear: 2026",
        idempotency_key=key,
        requested_by_kind=requested_by_kind,
        requested_by_id=owner.user_id,
        expires_at=expires_at,
    )


@pytest.mark.asyncio
async def test_schema_20_versions_are_owner_scoped_and_immutable(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    alice = await _resolve(store, subject="docs-alice", email="alice@example.com")
    bob = await _resolve(store, subject="docs-bob", email="bob@example.com")
    try:
        version = await _source(store, alice)
        repeated = await _source(store, alice)
        assert store.schema_version == 20
        assert repeated == version
        assert version.version_number == 1
        assert version.origin == "upload"
        assert await store.document_tools.get_version(
            user_id=bob.user_id,
            version_id=version.version_id,
        ) is None
    finally:
        store.close()

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {
            "app_file_versions",
            "app_document_jobs",
            "app_document_approvals",
            "app_file_derivations",
        } <= tables
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE app_file_versions SET filename = 'changed.txt' "
                "WHERE version_id = ?",
                (version.version_id,),
            )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.asyncio
async def test_request_digest_idempotency_bounds_and_bot_default_deny(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _resolve(store, subject="docs-owner", email="owner@example.com")
    try:
        version = await _source(store, owner)
        job, approval = await _request(store, owner, version.version_id)
        same_job, same_approval = await _request(store, owner, version.version_id)
        assert same_job.job_id == job.job_id
        assert same_approval.approval_id == approval.approval_id
        assert job.operation_digest == approval.operation_digest
        assert len(job.operation_digest) == 64

        with pytest.raises(DocumentConflictError, match="idempotency key"):
            await _request(
                store,
                owner,
                version.version_id,
                arguments={"fields": {"client": "Changed"}},
            )
        with pytest.raises(InvalidDocumentOperationError, match="allowlisted"):
            await _request(
                store,
                owner,
                version.version_id,
                key="bot-request",
                requested_by_kind="bot",
            )
        with pytest.raises(InvalidDocumentOperationError, match="too many"):
            await _request(
                store,
                owner,
                version.version_id,
                key="oversized-request",
                arguments={"values": list(range(300))},
            )
    finally:
        store.close()


@pytest.mark.asyncio
async def test_changed_expired_rejected_and_stale_approvals_execute_nothing(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = await _resolve(store, subject="docs-approval", email="owner@example.com")
    try:
        version = await _source(store, owner)
        job, approval = await _request(store, owner, version.version_id)
        with pytest.raises(DocumentApprovalError, match="digest"):
            await store.document_tools.decide(
                user_id=owner.user_id,
                job_id=job.job_id,
                actor_user_id=owner.user_id,
                operation_digest="0" * 64,
                decision="approved",
            )
        unchanged = await store.document_tools.get_job(
            user_id=owner.user_id,
            job_id=job.job_id,
        )
        assert unchanged is not None and unchanged[0].status == "awaiting_approval"
        with pytest.raises(DocumentApprovalError, match="owner"):
            await store.document_tools.decide(
                user_id=owner.user_id,
                job_id=job.job_id,
                actor_user_id="usr_someone_else",
                operation_digest=approval.operation_digest,
                decision="approved",
            )

        rejected_job, rejected_approval = await store.document_tools.decide(
            user_id=owner.user_id,
            job_id=job.job_id,
            actor_user_id=owner.user_id,
            operation_digest=approval.operation_digest,
            decision="rejected",
        ) or (None, None)
        assert rejected_job is not None and rejected_job.status == "rejected"
        assert rejected_approval is not None and rejected_approval.decision == "rejected"
        assert await store.document_tools.claim_next(
            lease_id="rejected-lease",
            lease_seconds=30,
        ) is None

        expiry = dt.datetime.now(dt.UTC) + dt.timedelta(minutes=1)
        expiring, expiring_approval = await _request(
            store,
            owner,
            version.version_id,
            key="expiring",
            expires_at=expiry.isoformat(),
        )
        with pytest.raises(DocumentApprovalExpiredError):
            await store.document_tools.decide(
                user_id=owner.user_id,
                job_id=expiring.job_id,
                actor_user_id=owner.user_id,
                operation_digest=expiring_approval.operation_digest,
                decision="approved",
                now=(expiry + dt.timedelta(seconds=1)).isoformat(),
            )
        expired = await store.document_tools.get_job(
            user_id=owner.user_id,
            job_id=expiring.job_id,
        )
        assert expired is not None and expired[0].status == "expired"

        stale_job, stale_approval = await _request(
            store,
            owner,
            version.version_id,
            key="stale",
        )
        newer = await _source(store, owner, suffix="two")
        assert newer.version_number == 2
        with pytest.raises(DocumentApprovalError, match="stale"):
            await store.document_tools.decide(
                user_id=owner.user_id,
                job_id=stale_job.job_id,
                actor_user_id=owner.user_id,
                operation_digest=stale_approval.operation_digest,
                decision="approved",
            )
        stale = await store.document_tools.get_job(
            user_id=owner.user_id,
            job_id=stale_job.job_id,
        )
        assert stale is not None
        assert (stale[0].status, stale[0].error_code) == ("failed", "stale_input")
    finally:
        store.close()


@pytest.mark.asyncio
async def test_approved_job_claims_once_and_publishes_one_derivation(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    owner = await _resolve(store, subject="docs-publish", email="owner@example.com")
    try:
        source = await _source(store, owner)
        job, approval = await _request(store, owner, source.version_id)
        decided = await store.document_tools.decide(
            user_id=owner.user_id,
            job_id=job.job_id,
            actor_user_id=owner.user_id,
            operation_digest=approval.operation_digest,
            decision="approved",
        )
        assert decided is not None and decided[0].status == "queued"
        claimed = await store.document_tools.claim_next(
            lease_id="worker-one",
            lease_seconds=300,
        )
        assert claimed is not None
        assert (claimed.job_id, claimed.attempts) == (job.job_id, 1)
        assert await store.document_tools.claim_next(
            lease_id="worker-two",
            lease_seconds=300,
        ) is None

        with pytest.raises(DocumentConflictError, match="mime"):
            await store.document_tools.publish(
                user_id=owner.user_id,
                job_id=job.job_id,
                lease_id="worker-one",
                output_file_id="file_generated_docx",
                filename="generated.pdf",
                mime="application/pdf",
                bytes_count=512,
                content_sha256=_hash("generated-docx"),
                worker_version="document-worker/1",
            )
        published, output = await store.document_tools.publish(
            user_id=owner.user_id,
            job_id=job.job_id,
            lease_id="worker-one",
            output_file_id="file_generated_docx",
            filename="generated.docx",
            mime=job.output_mime,
            bytes_count=512,
            content_sha256=_hash("generated-docx"),
            worker_version="document-worker/1",
        )
        assert published.status == "succeeded"
        assert output.parent_version_id == source.version_id
        assert output.origin == "derivation"

        repeated_job, repeated_output = await store.document_tools.publish(
            user_id=owner.user_id,
            job_id=job.job_id,
            lease_id="worker-one",
            output_file_id="file_generated_docx",
            filename="generated.docx",
            mime=job.output_mime,
            bytes_count=512,
            content_sha256=_hash("generated-docx"),
            worker_version="document-worker/1",
        )
        assert repeated_job.output_version_id == output.version_id
        assert repeated_output.version_id == output.version_id
        with pytest.raises(DocumentConflictError, match="another output"):
            await store.document_tools.publish(
                user_id=owner.user_id,
                job_id=job.job_id,
                lease_id="worker-one",
                output_file_id="file_generated_docx",
                filename="generated.docx",
                mime=job.output_mime,
                bytes_count=513,
                content_sha256=_hash("changed-output"),
                worker_version="document-worker/1",
            )
    finally:
        store.close()

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM app_file_versions WHERE origin = 'derivation'"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT COUNT(*) FROM app_file_derivations"
        ).fetchone()[0] == 1
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(app_document_approvals)")
        }
        assert not ({"arguments_json", "summary", "preview"} & columns)


@pytest.mark.asyncio
async def test_expired_worker_lease_recovers_after_restart_without_stale_publication(tmp_path):
    path = tmp_path / "app.sqlite"
    store = ApplicationStore(path)
    owner = await _resolve(store, subject="docs-restart", email="owner@example.com")
    source = await _source(store, owner)
    job, approval = await _request(store, owner, source.version_id)
    await store.document_tools.decide(
        user_id=owner.user_id,
        job_id=job.job_id,
        actor_user_id=owner.user_id,
        operation_digest=approval.operation_digest,
        decision="approved",
    )
    lease_started = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=20)
    first = await store.document_tools.claim_next(
        lease_id="before-restart",
        lease_seconds=10,
        now=lease_started.isoformat(),
    )
    assert first is not None and first.attempts == 1
    store.close()

    reopened = ApplicationStore(path)
    try:
        with pytest.raises(DocumentLeaseError, match="stale"):
            await reopened.document_tools.publish(
                user_id=owner.user_id,
                job_id=job.job_id,
                lease_id="before-restart",
                output_file_id="file_expired",
                filename="expired.docx",
                mime=job.output_mime,
                bytes_count=1,
                content_sha256=_hash("expired"),
                worker_version="document-worker/1",
            )
        with pytest.raises(DocumentLeaseError, match="stale"):
            await reopened.document_tools.fail(
                user_id=owner.user_id,
                job_id=job.job_id,
                lease_id="before-restart",
                error_code="expired_worker",
            )
        recovered = await reopened.document_tools.claim_next(
            lease_id="after-restart",
            lease_seconds=300,
            now=(lease_started + dt.timedelta(seconds=11)).isoformat(),
        )
        assert recovered is not None
        assert (recovered.job_id, recovered.attempts) == (job.job_id, 2)
        with pytest.raises(DocumentLeaseError, match="stale"):
            await reopened.document_tools.publish(
                user_id=owner.user_id,
                job_id=job.job_id,
                lease_id="before-restart",
                output_file_id="file_stale",
                filename="stale.docx",
                mime=job.output_mime,
                bytes_count=1,
                content_sha256=_hash("stale"),
                worker_version="document-worker/1",
            )
        cancelled = await reopened.document_tools.cancel(
            user_id=owner.user_id,
            job_id=job.job_id,
        )
        assert cancelled is not None and cancelled[0].status == "cancelled"
        with pytest.raises(DocumentLeaseError, match="stale"):
            await reopened.document_tools.publish(
                user_id=owner.user_id,
                job_id=job.job_id,
                lease_id="after-restart",
                output_file_id="file_cancelled",
                filename="cancelled.docx",
                mime=job.output_mime,
                bytes_count=1,
                content_sha256=_hash("cancelled"),
                worker_version="document-worker/1",
            )
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_document_job_routes_hide_other_owners_and_block_bot_approval(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    alice = await _resolve(store, subject="route-alice", email="alice@example.com")
    bob = await _resolve(store, subject="route-bob", email="bob@example.com")
    source = await _source(store, alice)
    job, approval = await _request(store, alice, source.version_id)
    current = {"principal": alice}
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: current["principal"]
    app.dependency_overrides[require_provider_principal] = lambda: current["principal"]

    try:
        with TestClient(app) as client:
            listed = client.get("/api/document-jobs")
            assert listed.status_code == 200
            item = listed.json()["items"][0]
            assert item["id"] == job.job_id
            assert "arguments" not in item
            assert item["approval"]["operation_digest"] == approval.operation_digest

            current["principal"] = bob
            assert client.get(f"/api/document-jobs/{job.job_id}").status_code == 404

            current["principal"] = Principal(
                user_id=alice.user_id,
                storage_namespace=alice.storage_namespace,
                provider=alice.provider,
                provider_subject=alice.provider_subject,
                email=alice.email,
                display_name=alice.display_name,
                role=alice.role,
                status=alice.status,
                auth_method=alice.auth_method,
                groups=frozenset({"bots", "users"}),
            )
            blocked = client.post(
                f"/api/document-jobs/{job.job_id}/decision",
                json={
                    "decision": "approved",
                    "operation_digest": approval.operation_digest,
                },
            )
            assert blocked.status_code == 403

            current["principal"] = alice
            approved = client.post(
                f"/api/document-jobs/{job.job_id}/decision",
                json={
                    "decision": "approved",
                    "operation_digest": approval.operation_digest,
                },
            )
            assert approved.status_code == 200
            assert approved.json()["status"] == "queued"
    finally:
        store.close()
