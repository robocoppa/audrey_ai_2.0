"""Native application resource contracts."""

import asyncio
import datetime as dt
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.app_state import ApplicationStore
from audrey.auth import require_principal, require_provider_principal
from audrey.identity import Principal
from audrey.routes.app import router


def _principal() -> Principal:
    return Principal(
        user_id="usr_123",
        storage_namespace="alice@example.com",
        provider="owui",
        provider_subject="provider-secret-subject",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        status="active",
        auth_method="owui_bearer",
    )


def _persist_principal(
    store: ApplicationStore,
    template: Principal,
) -> Principal:
    return asyncio.run(
        store.resolve_external_identity(
            provider="owui",
            subject=template.provider_subject,
            email=template.email,
            display_name=template.display_name,
            role=template.role,
            auth_method=template.auth_method,
            legacy_storage_namespace=template.storage_namespace,
        )
    )


def test_get_me_returns_audrey_id_without_internal_identity_fields():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_principal] = _principal

    response = TestClient(app).get("/api/me")

    assert response.status_code == 200
    assert response.json() == {
        "id": "usr_123",
        "email": "alice@example.com",
        "display_name": "Alice",
        "role": "user",
        "status": "active",
        "auth_provider": "owui",
    }
    assert "storage_namespace" not in response.text
    assert "provider-secret-subject" not in response.text


def test_get_me_requires_authentication():
    app = FastAPI()
    app.include_router(router)

    response = TestClient(app).get("/api/me")

    assert response.status_code == 401


def test_patch_me_updates_only_the_provider_authenticated_profile(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)
    owner = _persist_principal(store, _principal())
    app.dependency_overrides[require_provider_principal] = lambda: owner

    try:
        response = TestClient(app).patch(
            "/api/me",
            json={"display_name": "  Alice Builder  "},
        )
        refreshed = asyncio.run(
            store.resolve_external_identity(
                provider=owner.provider,
                subject=owner.provider_subject,
                email=owner.email,
                display_name="Provider Name",
                role=owner.role,
                auth_method=owner.auth_method,
                legacy_storage_namespace=owner.storage_namespace,
                sync_display_name=False,
            )
        )
    finally:
        store.close()

    assert response.status_code == 200
    assert response.json() == {
        "id": owner.user_id,
        "email": "alice@example.com",
        "display_name": "Alice Builder",
        "role": "user",
        "status": "active",
        "auth_provider": "owui",
    }
    assert refreshed.display_name == "Alice Builder"


def test_patch_me_rejects_personal_tokens_and_blank_names(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = _persist_principal(store, _principal())
    issued = asyncio.run(
        store.create_personal_token(
            user_id=owner.user_id,
            name="Native account",
            scopes=["account:read"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
        )
    )
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)
    app.dependency_overrides[require_provider_principal] = lambda: owner

    try:
        blank = TestClient(app).patch("/api/me", json={"display_name": "   "})
        app.dependency_overrides.clear()
        personal = TestClient(app).patch(
            "/api/me",
            headers={"Authorization": f"Bearer {issued.token}"},
            json={"display_name": "Token Rename"},
        )
    finally:
        store.close()

    assert blank.status_code == 422
    assert blank.json()["detail"] == "display name is required"
    assert personal.status_code == 403
    assert personal.json()["detail"] == (
        "External provider authentication is required for this operation."
    )


def test_token_lifecycle_returns_secret_only_on_create(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)
    owner = _persist_principal(store, _principal())
    app.dependency_overrides[require_provider_principal] = lambda: owner

    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/tokens",
                json={
                    "name": "Laptop eval",
                    "scopes": ["compat:full", "account:read"],
                    "expires_in_days": 30,
                },
            )
            assert created.status_code == 201
            body = created.json()
            token_id = body["id"]
            assert body["token"].startswith(f"aud_{token_id}.")
            assert body["scopes"] == ["account:read", "compat:full"]
            assert body["revoked_at"] is None

            listed = client.get("/api/tokens")
            assert listed.status_code == 200
            assert listed.json()["items"] == [
                {key: value for key, value in body.items() if key != "token"}
            ]
            assert "token" not in listed.text

            revoked = client.delete(f"/api/tokens/{token_id}")
            assert revoked.status_code == 200
            assert revoked.json() == {"id": body["id"], "revoked": True}

            after = client.get("/api/tokens").json()["items"][0]
            assert after["revoked_at"]
            assert "token" not in after
    finally:
        store.close()


def test_token_route_rejects_non_expiring_request(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)
    owner = _persist_principal(store, _principal())
    app.dependency_overrides[require_provider_principal] = lambda: owner

    try:
        response = TestClient(app).post(
            "/api/tokens",
            json={
                "name": "Permanent token",
                "scopes": ["compat:full"],
                "expires_in_days": None,
            },
        )
    finally:
        store.close()

    assert response.status_code == 422


def test_token_routes_do_not_cross_owner_boundary(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)

    first = _principal()
    second = Principal(
        user_id="usr_456",
        storage_namespace="bob@example.com",
        provider="owui",
        provider_subject="owui-bob",
        email="bob@example.com",
        display_name="Bob",
        role="user",
        status="active",
        auth_method="owui_bearer",
    )

    actual_first = _persist_principal(store, first)
    actual_second = _persist_principal(store, second)
    app.dependency_overrides[require_provider_principal] = lambda: actual_first
    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/tokens",
                json={"name": "First", "scopes": ["compat:full"]},
            )
            token_id = created.json()["id"]

            app.dependency_overrides[require_provider_principal] = lambda: actual_second
            assert client.get("/api/tokens").json() == {"items": []}
            hidden = client.delete(f"/api/tokens/{token_id}")
            assert hidden.status_code == 404
    finally:
        store.close()


def test_personal_token_needs_account_read_scope_for_me():
    principal = Principal(
        user_id="usr_123",
        storage_namespace="alice@example.com",
        provider="audrey",
        provider_subject="pat_123",
        email="alice@example.com",
        display_name="Alice",
        role="user",
        status="active",
        auth_method="personal_token",
        scopes=frozenset({"compat:full"}),
    )
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_principal] = lambda: principal

    response = TestClient(app).get("/api/me")

    assert response.status_code == 403
    assert response.json()["detail"] == ("Personal access token lacks account:read scope.")


def test_personal_token_authenticates_through_native_me_route(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    owner = _persist_principal(store, _principal())
    issued = asyncio.run(
        store.create_personal_token(
            user_id=owner.user_id,
            name="Native account",
            scopes=["account:read"],
            expires_at=(dt.datetime.now(dt.UTC) + dt.timedelta(days=30)).isoformat(),
        )
    )
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)

    try:
        response = TestClient(app).get(
            "/api/me",
            headers={"Authorization": f"Bearer {issued.token}"},
        )
    finally:
        store.close()

    assert response.status_code == 200
    assert response.json()["id"] == owner.user_id
    assert response.json()["auth_provider"] == "audrey"


def _conversation_app(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    app = FastAPI()
    app.state.application_store = store
    app.include_router(router)
    owner = _persist_principal(store, _principal())
    app.dependency_overrides[require_principal] = lambda: owner
    return app, store, owner


def test_conversation_routes_create_update_archive_list_and_delete(tmp_path):
    app, store, _ = _conversation_app(tmp_path)
    archive_wakes: list[bool] = []
    app.state.archive_projector = SimpleNamespace(
        wake=lambda: archive_wakes.append(True)
    )
    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/conversations",
                json={"title": "Native chat", "default_mode": "deep"},
            )
            assert created.status_code == 201
            conversation = created.json()
            conversation_id = conversation["id"]
            assert conversation["title"] == "Native chat"
            assert conversation["default_mode"] == "deep"
            assert "user_id" not in created.text

            fetched = client.get(f"/api/conversations/{conversation_id}")
            assert fetched.status_code == 200
            assert fetched.json() == conversation

            changed = client.patch(
                f"/api/conversations/{conversation_id}",
                json={"title": "Renamed", "default_mode": "video"},
            )
            assert changed.status_code == 200
            assert changed.json()["title"] == "Renamed"
            assert changed.json()["default_mode"] == "video"

            archived = client.patch(
                f"/api/conversations/{conversation_id}",
                json={"archived": True},
            )
            assert archived.status_code == 200
            assert archived.json()["archived_at"] is not None
            assert client.get("/api/conversations").json()["items"] == []
            archived_list = client.get("/api/conversations?archived=true").json()
            assert [item["id"] for item in archived_list["items"]] == [conversation_id]

            deleted = client.delete(f"/api/conversations/{conversation_id}")
            assert deleted.status_code == 204
            assert client.get(f"/api/conversations/{conversation_id}").status_code == 404
            assert archive_wakes == [True]
    finally:
        store.close()


def test_conversation_and_message_routes_paginate_without_overlap(tmp_path):
    app, store, owner = _conversation_app(tmp_path)
    try:
        conversation_ids = [
            asyncio.run(
                store.conversations.create(user_id=owner.user_id, title=f"Chat {index}")
            ).conversation_id
            for index in range(3)
        ]
        target_id = conversation_ids[0]
        for prompt, answer in (("one", "first"), ("two", "second")):
            started = asyncio.run(
                store.conversations.begin_run(
                    user_id=owner.user_id,
                    conversation_id=target_id,
                    user_content=prompt,
                )
            )
            assert started is not None
            asyncio.run(
                store.conversations.finish_run(
                    user_id=owner.user_id,
                    run_id=started.run.run_id,
                    outcome="succeeded",
                    assistant_content=answer,
                )
            )

        with TestClient(app) as client:
            first = client.get("/api/conversations?limit=2")
            assert first.status_code == 200
            first_body = first.json()
            assert len(first_body["items"]) == 2
            assert first_body["next_cursor"]
            second = client.get(
                "/api/conversations",
                params={"limit": 2, "cursor": first_body["next_cursor"]},
            )
            assert second.status_code == 200
            listed_ids = [item["id"] for item in first_body["items"] + second.json()["items"]]
            assert len(listed_ids) == len(set(listed_ids)) == 3
            assert set(listed_ids) == set(conversation_ids)

            searched = client.get(
                "/api/conversations",
                params={"limit": 1, "q": "chat"},
            )
            assert searched.status_code == 200
            searched_body = searched.json()
            assert len(searched_body["items"]) == 1
            assert searched_body["next_cursor"]
            assert client.get(
                "/api/conversations",
                params={
                    "limit": 1,
                    "q": "different search",
                    "cursor": searched_body["next_cursor"],
                },
            ).status_code == 422

            exact_search = client.get(
                "/api/conversations",
                params={"q": "Chat 0"},
            )
            assert [item["id"] for item in exact_search.json()["items"]] == [
                target_id
            ]

            messages = client.get(f"/api/conversations/{target_id}/messages?limit=3")
            assert messages.status_code == 200
            first_messages = messages.json()
            assert [item["sequence"] for item in first_messages["items"]] == [1, 2, 3]
            assert first_messages["next_cursor"]
            remaining = client.get(
                f"/api/conversations/{target_id}/messages",
                params={"limit": 3, "cursor": first_messages["next_cursor"]},
            )
            assert remaining.status_code == 200
            assert [item["sequence"] for item in remaining.json()["items"]] == [4]
            assert "user_id" not in messages.text
    finally:
        store.close()


def test_conversation_routes_hide_cross_owner_resources(tmp_path):
    app, store, alice = _conversation_app(tmp_path)
    bob_template = Principal(
        user_id="ignored",
        storage_namespace="bob@example.com",
        provider="owui",
        provider_subject="owui-bob",
        email="bob@example.com",
        display_name="Bob",
        role="user",
        status="active",
        auth_method="owui_bearer",
    )
    bob = _persist_principal(store, bob_template)
    conversation = asyncio.run(store.conversations.create(user_id=alice.user_id))
    app.dependency_overrides[require_principal] = lambda: bob
    try:
        with TestClient(app) as client:
            conversation_path = f"/api/conversations/{conversation.conversation_id}"
            assert client.get(conversation_path).status_code == 404
            assert client.patch(conversation_path, json={"title": "Intrusion"}).status_code == 404
            assert client.delete(conversation_path).status_code == 404
            assert client.get(f"{conversation_path}/messages").status_code == 404
            assert client.get("/api/conversations").json()["items"] == []
    finally:
        store.close()


def test_conversation_routes_guard_active_runs_and_reject_bad_cursors(tmp_path):
    app, store, owner = _conversation_app(tmp_path)
    conversation = asyncio.run(store.conversations.create(user_id=owner.user_id))
    started = asyncio.run(
        store.conversations.begin_run(
            user_id=owner.user_id,
            conversation_id=conversation.conversation_id,
            user_content="Still running",
        )
    )
    assert started is not None
    try:
        with TestClient(app) as client:
            path = f"/api/conversations/{conversation.conversation_id}"
            assert client.patch(path, json={"archived": True}).status_code == 409
            assert client.delete(path).status_code == 409
            assert client.get("/api/conversations?cursor=not-base64").status_code == 422
            assert client.get(f"{path}/messages?cursor=not-base64").status_code == 422
            assert client.patch(path, json={}).status_code == 422
    finally:
        store.close()


def test_conversation_routes_require_auth_and_compat_scope(tmp_path):
    unauthenticated = FastAPI()
    unauthenticated.include_router(router)
    assert TestClient(unauthenticated).get("/api/conversations").status_code == 401

    app, store, owner = _conversation_app(tmp_path)
    personal = Principal(
        user_id=owner.user_id,
        storage_namespace=owner.storage_namespace,
        provider="audrey",
        provider_subject="pat_missing_scope",
        email=owner.email,
        display_name=owner.display_name,
        role="user",
        status="active",
        auth_method="personal_token",
        scopes=frozenset({"account:read"}),
    )
    app.dependency_overrides[require_principal] = lambda: personal
    try:
        response = TestClient(app).get("/api/conversations")
        assert response.status_code == 403
        assert response.json()["detail"] == (
            "Personal access token lacks compat:full scope."
        )
    finally:
        store.close()
