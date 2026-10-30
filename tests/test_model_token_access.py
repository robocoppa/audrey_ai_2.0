"""Native model grants follow a personal token owner's current account authority."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from audrey.app_state import ApplicationStore
from audrey.model_catalog import catalog_for_principal, resolve_model


async def _admin(store, name):
    return await store.resolve_external_identity(
        provider="cloudflare_access", subject=name, email=f"{name}@example.com",
        display_name=name, role="admin", auth_method="cloudflare_access",
        initial_status="active",
    )


def _runtime(*names):
    cfg = SimpleNamespace(raw={"passthrough": {"enabled": True}}, model_registry={})
    ollama = SimpleNamespace(
        tags=AsyncMock(return_value=[{"name": name} for name in names]),
    )
    return cfg, ollama


async def _token_catalog(store, cfg, ollama, token):
    # Match native request authentication: never reuse an earlier principal.
    principal = await store.authenticate_personal_token(token)
    models = await catalog_for_principal(cfg, store, principal, ollama=ollama)
    return principal, {model.id: model for model in models}


async def test_admin_personal_token_has_restricted_but_not_disabled_native_models(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        owner = await _admin(store, "token-owner")
        cfg, ollama = _runtime("restricted:latest", "disabled:latest")
        for name, enabled in (("restricted:latest", True), ("disabled:latest", False)):
            await store.set_model_access_policy(
                actor_user_id=owner.user_id, model_id=f"direct/{name}",
                enabled=enabled, audience="admins",
            )
        issued = await store.create_personal_token(
            user_id=owner.user_id, name="Native administration account client",
            scopes=["compat:full"], expires_at=None,
        )

        principal, models = await _token_catalog(store, cfg, ollama, issued.token)

        assert principal.auth_method == "personal_token"
        assert models["direct/restricted:latest"].visibility == "restricted"
        assert "direct/disabled:latest" not in models
        assert await resolve_model(
            cfg, store, principal, "direct/restricted:latest", ollama=ollama,
        ) is not None
        assert await resolve_model(
            cfg, store, principal, "direct/disabled:latest", ollama=ollama,
        ) is None
        assert ollama.tags.await_count == 1
    finally:
        store.close()


@pytest.mark.parametrize("revocation", ["owner_membership", "model_publication"])
async def test_token_model_access_changes_on_owner_demotion_and_grant_revocation(
    tmp_path, revocation,
):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        operator = await _admin(store, "operator")
        owner = await _admin(store, "token-owner")
        cfg, ollama = _runtime("restricted:latest", "tester:latest", "public:latest")
        for name, audience in (
            ("restricted:latest", "admins"),
            ("tester:latest", "testers"),
            ("public:latest", "users"),
        ):
            await store.set_model_access_policy(
                actor_user_id=operator.user_id, model_id=f"direct/{name}",
                enabled=True, audience=audience,
            )
        issued = await store.create_personal_token(
            user_id=owner.user_id, name="Native client", scopes=["compat:full"],
            expires_at=None,
        )
        principal, models = await _token_catalog(store, cfg, ollama, issued.token)
        assert principal.is_admin
        assert {model_id for model_id in models if model_id.startswith("direct/")} == {
            "direct/restricted:latest", "direct/tester:latest", "direct/public:latest",
        }

        await store.admin_update_user(
            actor_user_id=operator.user_id, target_user_id=owner.user_id,
            groups=["users", "testers"],
        )
        principal, models = await _token_catalog(store, cfg, ollama, issued.token)
        assert not principal.is_admin
        assert "direct/restricted:latest" not in models
        assert {"direct/tester:latest", "direct/public:latest"} <= models.keys()
        # Demotion must take effect while the same model inventory remains cached.
        assert ollama.tags.await_count == 1

        if revocation == "owner_membership":
            await store.admin_update_user(
                actor_user_id=operator.user_id, target_user_id=owner.user_id,
                groups=["users"],
            )
        else:
            await store.set_model_publication_profile(
                actor_user_id=operator.user_id, model_id="direct/tester:latest",
                visibility="restricted", roles=[], display_name="Withdrawn tester model",
            )

        principal, models = await _token_catalog(store, cfg, ollama, issued.token)
        assert "direct/restricted:latest" not in models
        assert "direct/tester:latest" not in models
        assert "direct/public:latest" in models
        assert await resolve_model(
            cfg, store, principal, "direct/tester:latest", ollama=ollama,
        ) is None
    finally:
        store.close()
