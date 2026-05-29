"""Tests for the audrey_passthrough/<concrete> route surface.

Covers model-string parsing, gating (disabled, role, allowed_models),
and the /v1/models prefix expansion. We test the helper functions
directly rather than spinning up the full FastAPI app — same pattern
as test_admin_routes.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from audrey.routes.openai import (
    PASSTHROUGH_BARE,
    PASSTHROUGH_PREFIX,
    _is_passthrough,
    _passthrough_concrete,
    _resolve_passthrough_model,
    list_models,
)

# ─── _is_passthrough / _passthrough_concrete ───────────────────────────

def test_is_passthrough_recognizes_bare_and_prefixed():
    assert _is_passthrough(PASSTHROUGH_BARE)
    assert _is_passthrough(f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k")
    assert _is_passthrough("audrey_passthrough/anything")


def test_is_passthrough_rejects_other_virtual_models():
    assert not _is_passthrough("audrey_fast")
    assert not _is_passthrough("audrey_deep")
    assert not _is_passthrough("qwen3.6:35b")
    assert not _is_passthrough("")


def test_passthrough_concrete_strips_prefix():
    assert _passthrough_concrete(f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k") == "qwen3.6:35b-64k"
    assert _passthrough_concrete(f"{PASSTHROUGH_PREFIX}llama3.3:70b") == "llama3.3:70b"


def test_passthrough_concrete_rejects_bare_form():
    # The bare `audrey_passthrough` must include /<concrete>; bare is
    # a 400, not silent fallback.
    with pytest.raises(HTTPException) as exc_info:
        _passthrough_concrete(PASSTHROUGH_BARE)
    assert exc_info.value.status_code == 400
    assert "audrey_passthrough/<model_name>" in exc_info.value.detail


def test_passthrough_concrete_rejects_empty_suffix():
    with pytest.raises(HTTPException) as exc_info:
        _passthrough_concrete(f"{PASSTHROUGH_PREFIX}   ")
    assert exc_info.value.status_code == 400
    assert "Empty concrete" in exc_info.value.detail


# ─── _resolve_passthrough_model ────────────────────────────────────────

def _stub_cfg(passthrough_block: dict | None) -> SimpleNamespace:
    """Build the minimal cfg shape the resolver reads."""
    return SimpleNamespace(raw={"passthrough": passthrough_block} if passthrough_block is not None else {})


def _stub_registry(location: str = "local") -> SimpleNamespace:
    """Registry stub: location_of returns whatever we configure."""
    return SimpleNamespace(location_of=lambda _name: location)


def _stub_user(role: str = "user", email: str = "alice@example.com") -> SimpleNamespace:
    return SimpleNamespace(email=email, role=role, owui_id="abc")


def test_resolve_rejects_when_passthrough_disabled():
    cfg = _stub_cfg({"enabled": False, "allowed_models": ["qwen3.6:35b-64k"]})
    with pytest.raises(HTTPException) as exc_info:
        _resolve_passthrough_model(
            f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
            cfg, _stub_registry(), _stub_user(),
        )
    assert exc_info.value.status_code == 403
    assert "disabled" in exc_info.value.detail.lower()


def test_resolve_rejects_when_no_passthrough_block_in_config():
    # Missing block is the same as disabled.
    cfg = _stub_cfg(None)
    with pytest.raises(HTTPException) as exc_info:
        _resolve_passthrough_model(
            f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
            cfg, _stub_registry(), _stub_user(),
        )
    assert exc_info.value.status_code == 403


def test_resolve_rejects_model_not_in_allowed_list():
    cfg = _stub_cfg({"enabled": True, "allowed_models": ["qwen3.6:35b-64k"]})
    with pytest.raises(HTTPException) as exc_info:
        _resolve_passthrough_model(
            f"{PASSTHROUGH_PREFIX}llama3.3:70b",
            cfg, _stub_registry(), _stub_user(),
        )
    assert exc_info.value.status_code == 403
    assert "llama3.3:70b" in exc_info.value.detail


def test_resolve_allows_any_model_when_allowed_list_is_empty():
    # Empty/missing allowed_models means no allowlist enforcement —
    # any registered concrete model goes through. Useful for dev,
    # cfg.yaml ships with a non-empty list.
    cfg = _stub_cfg({"enabled": True, "allowed_models": []})
    concrete, location = _resolve_passthrough_model(
        f"{PASSTHROUGH_PREFIX}some-model:tag",
        cfg, _stub_registry(location="local"), _stub_user(),
    )
    assert concrete == "some-model:tag"
    assert location == "local"


def test_resolve_returns_concrete_and_location_on_allow():
    cfg = _stub_cfg({"enabled": True, "allowed_models": ["qwen3.6:35b-64k"]})
    concrete, location = _resolve_passthrough_model(
        f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
        cfg, _stub_registry(location="local"), _stub_user(),
    )
    assert concrete == "qwen3.6:35b-64k"
    assert location == "local"


def test_resolve_propagates_cloud_location_from_registry():
    cfg = _stub_cfg({"enabled": True, "allowed_models": ["cloud-only-model"]})
    _concrete, location = _resolve_passthrough_model(
        f"{PASSTHROUGH_PREFIX}cloud-only-model",
        cfg, _stub_registry(location="cloud"), _stub_user(),
    )
    assert location == "cloud"


def test_resolve_enforces_required_role():
    cfg = _stub_cfg({
        "enabled": True,
        "allowed_models": ["qwen3.6:35b-64k"],
        "require_role": "admin",
    })
    with pytest.raises(HTTPException) as exc_info:
        _resolve_passthrough_model(
            f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
            cfg, _stub_registry(), _stub_user(role="user"),
        )
    assert exc_info.value.status_code == 403
    assert "admin" in exc_info.value.detail


def test_resolve_admin_role_allows_admin_user():
    cfg = _stub_cfg({
        "enabled": True,
        "allowed_models": ["qwen3.6:35b-64k"],
        "require_role": "admin",
    })
    concrete, _ = _resolve_passthrough_model(
        f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
        cfg, _stub_registry(), _stub_user(role="admin"),
    )
    assert concrete == "qwen3.6:35b-64k"


def test_resolve_null_role_means_any_authenticated_user():
    cfg = _stub_cfg({
        "enabled": True,
        "allowed_models": ["qwen3.6:35b-64k"],
        "require_role": None,
    })
    concrete, _ = _resolve_passthrough_model(
        f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k",
        cfg, _stub_registry(), _stub_user(role="user"),
    )
    assert concrete == "qwen3.6:35b-64k"


def test_resolve_bare_form_is_400_not_403():
    # The bare audrey_passthrough form is a malformed request (400),
    # not an auth/policy issue (403). Distinct so clients can tell
    # them apart.
    cfg = _stub_cfg({"enabled": True, "allowed_models": ["qwen3.6:35b-64k"]})
    with pytest.raises(HTTPException) as exc_info:
        _resolve_passthrough_model(
            PASSTHROUGH_BARE, cfg, _stub_registry(), _stub_user(),
        )
    assert exc_info.value.status_code == 400


# ─── /v1/models listing ────────────────────────────────────────────────

def _stub_request_for_listing(passthrough_block: dict | None) -> SimpleNamespace:
    cfg = _stub_cfg(passthrough_block)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(cfg=cfg)))


@pytest.mark.asyncio
async def test_list_models_includes_passthrough_when_enabled():
    request = _stub_request_for_listing({
        "enabled": True,
        "allowed_models": ["qwen3.6:35b-64k", "qwen3-coder-next-64k:latest"],
    })
    resp = await list_models(request)
    ids = {entry["id"] for entry in resp["data"]}
    assert f"{PASSTHROUGH_PREFIX}qwen3.6:35b-64k" in ids
    assert f"{PASSTHROUGH_PREFIX}qwen3-coder-next-64k:latest" in ids
    # Pipeline virtual models still listed.
    assert "audrey_fast" in ids
    assert "audrey_deep" in ids


@pytest.mark.asyncio
async def test_list_models_omits_passthrough_when_disabled():
    request = _stub_request_for_listing({
        "enabled": False,
        "allowed_models": ["qwen3.6:35b-64k"],
    })
    resp = await list_models(request)
    ids = {entry["id"] for entry in resp["data"]}
    # No passthrough entries.
    assert not any(i.startswith(PASSTHROUGH_PREFIX) for i in ids)
    # Pipeline virtual models still listed — disabling passthrough
    # doesn't kill the rest of the listing.
    assert "audrey_fast" in ids


@pytest.mark.asyncio
async def test_list_models_omits_passthrough_when_no_block():
    request = _stub_request_for_listing(None)
    resp = await list_models(request)
    ids = {entry["id"] for entry in resp["data"]}
    assert not any(i.startswith(PASSTHROUGH_PREFIX) for i in ids)


@pytest.mark.asyncio
async def test_list_models_handles_empty_allowed_list():
    # Enabled but with no allowed_models means nothing to list under
    # the prefix — passthrough still works at the route level (no
    # allowlist enforcement), but the listing has nothing to show.
    request = _stub_request_for_listing({
        "enabled": True,
        "allowed_models": [],
    })
    resp = await list_models(request)
    ids = {entry["id"] for entry in resp["data"]}
    assert not any(i.startswith(PASSTHROUGH_PREFIX) for i in ids)
