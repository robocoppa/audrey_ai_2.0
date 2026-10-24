"""Catalog caching must preserve current per-owner authorization and admin saves."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

from audrey.app_state import ApplicationStore
from audrey.model_catalog import catalog_for_principal, configured_models


def _cfg():
    return SimpleNamespace(
        raw={"passthrough": {"enabled": True, "allowed_models": ["test:latest"]}},
        model_registry={},
    )


async def _principal(store, name, role="user"):
    return await store.resolve_external_identity(
        provider="cloudflare_access", subject=name, email=name + "@example.com",
        display_name=name, role=role, auth_method="cloudflare_access",
        initial_status="active",
    )


async def test_catalog_cache_shares_inventory_without_sharing_principal_access(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        cfg = _cfg()
        admin = await _principal(store, "admin", "admin")
        user = await _principal(store, "user")
        ollama = SimpleNamespace(tags=AsyncMock(return_value=[{"name": "test:latest"}]))
        readers = ["list_model_access_policies", "list_model_publication_profiles", "list_model_display_order"]
        for name in readers:
            setattr(store, name, AsyncMock(wraps=getattr(store, name)))
        catalogs = await asyncio.gather(
            catalog_for_principal(cfg, store, admin, ollama=ollama, include_hidden=True),
            catalog_for_principal(cfg, store, user, ollama=ollama),
        )
        assert "direct/test:latest" in {m.id for m in catalogs[0]}
        assert "direct/test:latest" not in {m.id for m in catalogs[1]}
        assert await catalog_for_principal(
            cfg, store, replace(user, status="disabled"), ollama=ollama,
        ) == ()
        assert ollama.tags.await_count == 1
        assert all(getattr(store, name).await_count == 1 for name in readers)
    finally:
        store.close()


async def test_admin_publication_and_revocation_invalidate_catalog_immediately(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        cfg = _cfg()
        admin = await _principal(store, "admin", "admin")
        user = await _principal(store, "user")
        ollama = SimpleNamespace(tags=AsyncMock(return_value=[{"name": "test:latest"}]))
        async def available():
            return {m.id for m in await catalog_for_principal(cfg, store, user, ollama=ollama)}
        assert "direct/test:latest" not in await available()
        await store.set_model_access_policy(
            actor_user_id=admin.user_id, model_id="direct/test:latest", enabled=True, audience="users",
        )
        assert "direct/test:latest" in await available()
        profile = await store.set_model_publication_profile(
            actor_user_id=admin.user_id, model_id="direct/test:latest",
            visibility="restricted", roles=[], display_name="Restricted model",
        )
        assert profile.visibility == "restricted"
        assert "direct/test:latest" not in await available()
        # Existing installations keep their SQL CHECK encoding, with no schema
        # migration or rewrite of administrator grants.
        assert store._conn.execute(
            "SELECT visibility FROM model_publication_profiles WHERE model_id=?",
            ("direct/test:latest",),
        ).fetchone()[0] == "private"
        admin_models = await catalog_for_principal(cfg, store, admin, ollama=ollama)
        assert next(m for m in admin_models if m.id == "direct/test:latest").visibility == "restricted"
        await store.delete_model_publication_profile(
            actor_user_id=admin.user_id, model_id="direct/test:latest",
        )
        assert "direct/test:latest" in await available()
        await store.set_model_access_policy(
            actor_user_id=admin.user_id, model_id="direct/test:latest", enabled=False, audience="users",
        )
        assert "direct/test:latest" not in await available()
    finally:
        store.close()


async def test_catalog_inventory_expires_and_app_instances_are_isolated(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("audrey.model_catalog.time.monotonic", lambda: clock[0])
    first = ApplicationStore(tmp_path / "first.sqlite")
    second = ApplicationStore(tmp_path / "second.sqlite")
    try:
        cfg = _cfg()
        one = await _principal(first, "one", "admin")
        two = await _principal(second, "two")
        ollama = SimpleNamespace(tags=AsyncMock(return_value=[{"name": "test:latest"}]))
        await catalog_for_principal(cfg, first, one, ollama=ollama)
        assert "direct/test:latest" not in {
            m.id for m in await catalog_for_principal(cfg, second, two, ollama=ollama)
        }
        assert ollama.tags.await_count == 2
        ollama.tags.return_value.append({"name": "added:latest"})
        assert "direct/added:latest" not in {
            m.id for m in await catalog_for_principal(cfg, first, one, ollama=ollama)
        }
        clock[0] += 31
        assert "direct/added:latest" in {
            m.id for m in await catalog_for_principal(cfg, first, one, ollama=ollama)
        }
        assert ollama.tags.await_count == 3
    finally:
        first.close()
        second.close()


async def test_admin_save_during_inventory_refresh_cannot_publish_stale_grants(tmp_path):
    store = ApplicationStore(tmp_path / "app.sqlite")
    try:
        cfg = _cfg()
        admin = await _principal(store, "admin", "admin")
        user = await _principal(store, "user")
        started, release = asyncio.Event(), asyncio.Event()
        async def tags():
            started.set()
            await release.wait()
            return [{"name": "test:latest"}]
        ollama = SimpleNamespace(tags=AsyncMock(side_effect=tags))
        pending = asyncio.create_task(catalog_for_principal(cfg, store, user, ollama=ollama))
        await started.wait()
        await store.set_model_access_policy(
            actor_user_id=admin.user_id, model_id="direct/test:latest", enabled=True, audience="users",
        )
        release.set()
        assert "direct/test:latest" in {m.id for m in await pending}
        assert ollama.tags.await_count == 2
    finally:
        store.close()


def test_private_catalog_order_tracks_local_workflow(monkeypatch):
    import audrey.model_catalog as catalog
    workflows = list(catalog._WORKFLOWS)
    local = next(model for model in workflows if model.id == "local")
    workflows.remove(local)
    workflows.insert(1, local)
    monkeypatch.setattr(catalog, "_WORKFLOWS", tuple(workflows))
    cfg = _cfg()
    cfg.raw["private_processing"] = {"enabled": True}
    ids = [model.id for model in configured_models(cfg)]
    assert ids.index("private") == ids.index("local") + 1
