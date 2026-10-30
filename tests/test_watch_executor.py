"""Public watch comparisons are complete, evidenced and local-only."""

from __future__ import annotations

import hashlib
import json
import socket
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from audrey.agent_dispatch import AgentDispatchStoppedError
from audrey.app_state import ApplicationStore
from audrey.models.ollama import GenerationUsageError, OllamaClient
from audrey.net import public_fetch
from audrey.net.public_fetch import PublicAsset
from audrey.pipeline.fair_gate import FairLocalGate
from audrey.processing_policy import ProcessingPolicy, processing_scope
from audrey.routes.inflight import UserInflightRegistry
from audrey.watch_executor import (
    EXTRACTION_VERSION,
    WatchCheckError,
    WatchExecutor,
    evidence_diff,
    extract_public_text,
    validate_watch_url,
)
from audrey.watch_runtime import WatchRuntime

_MODEL = "qwen3.8:latest"
_URL = "https://example.org/releases"


class Repository:
    def __init__(self):
        self.operations = []
        self.revoked = False

    async def validate_lease(self, **_lease):
        if self.revoked:
            raise AgentDispatchStoppedError("The watch definition changed.")

    async def reserve_operation(self, *, kind, **lease):
        await self.validate_lease(**lease)
        self.operations.append(kind)
        return str(len(self.operations))


@pytest.fixture
async def state(tmp_path):
    store = ApplicationStore(tmp_path / "application.sqlite")
    owner = await store.resolve_external_identity(
        provider="test", subject="alice", email="alice@example.com", display_name="Alice",
        role="admin", auth_method="test",
    )
    repository = Repository()
    store.watches = repository
    app = FastAPI()
    app.state.application_store = store
    app.state.cfg = SimpleNamespace(raw={}, env=SimpleNamespace(ollama_host="http://ollama.test"))
    app.state.gate = FairLocalGate()
    app.state.inflight = UserInflightRegistry(max_inflight_per_user=2)
    value = SimpleNamespace(
        store=store, owner=owner, app=app, repository=repository, calls=[], clients=[],
        fetches=[], resolved=[], client_options=[], on_chat=None, on_tags=None,
        asset=PublicAsset(b"Release 2 is available.", "text/plain", _URL),
        installed=[{"name": _MODEL, "model": _MODEL, "digest": "a" * 64}],
        metadata={"details": {"format": "gguf"}, "model_info": {
            "general.architecture": "qwen3", "general.parameter_count": 27000000000,
        }},
        response={"done": True, "done_reason": "stop", "eval_count": 12,
                  "message": {"content": json.dumps({"significant": True, "summary": "Release 2 was announced."})}},
    )

    async def provider(request):
        if request.url.path == "/api/tags":
            if value.on_tags:
                await value.on_tags()
            return httpx.Response(200, json={"models": value.installed})
        if request.url.path == "/api/show":
            return httpx.Response(200, json=value.metadata)
        assert request.url.path == "/api/chat"
        value.calls.append(json.loads(request.content))
        if value.on_chat:
            await value.on_chat()
        return httpx.Response(200, json=value.response)

    def client_factory(endpoint, **kwargs):
        value.client_options.append((endpoint, kwargs))
        client = OllamaClient(endpoint, transport=httpx.MockTransport(provider), **kwargs)
        value.clients.append(client)
        return client

    async def resolve(url, **kwargs):
        assert kwargs["no_query"] is True
        await kwargs["authorize"]()
        value.resolved.append(url)
        return url

    async def fetch(url, **kwargs):
        assert kwargs["max_bytes"] == 256 * 1024
        assert kwargs["no_query"] is True and kwargs["strict_length"] is True
        await kwargs["authorize"]()
        value.fetches.append(url)
        return value.asset

    value.executor = WatchExecutor(app, client_factory=client_factory, resolver=resolve, fetcher=fetch)
    yield value
    for client in value.clients:
        await client.aclose()
    store.close()


def _run(state, *, old="Release 1 is available.", **changes):
    baseline = None if old is None else {
        "text": old, "sha256": hashlib.sha256(old.encode()).hexdigest(),
        "final_url": _URL, "extraction_version": EXTRACTION_VERSION, "run_id": "baseline",
    }
    return {
        "run_id": "run", "watch_id": "watch", "user_id": state.owner.user_id,
        "lease_token": "lease", "fence": 1, "url": _URL,
        "criterion": "New release announcements", "baseline": baseline, **changes,
    }


async def _add_owner_to_bots(state):
    await state.store.admin_update_user(
        actor_user_id=state.owner.user_id,
        target_user_id=state.owner.user_id,
        groups=[*state.owner.groups, "bots"],
    )
    principal = await state.store.get_principal(user_id=state.owner.user_id)
    assert principal.role == "admin" and "bots" in principal.groups


@pytest.mark.parametrize("old,outcome", [(None, "baseline"), ("Release 2 is available.", "unchanged")])
async def test_baseline_and_unchanged_skip_all_model_work(state, old, outcome):
    result = await state.executor(_run(state, old=old))
    assert result["outcome"] == outcome and result["significant"] is None
    assert result["text"] == "Release 2 is available." and result["diff"] == ""
    assert state.repository.operations == ["fetch"]
    assert not state.calls and not state.clients


@pytest.mark.parametrize("identity", ["final_url", "extraction_version"])
async def test_incompatible_source_identity_establishes_baseline_without_alert(state, identity):
    run = _run(state)
    run["baseline"][identity] = "different"
    result = await state.executor(run)
    assert result["outcome"] == "baseline" and result["significant"] is None
    assert not state.clients


async def test_corrupted_saved_baseline_is_not_reported_as_unchanged(state):
    run = _run(state, old="Release 2 is available.")
    run["baseline"]["text"] = "Corrupted content"
    with pytest.raises(WatchCheckError, match="baseline could not be verified"):
        await state.executor(run)
    assert not state.clients


async def test_change_too_large_for_complete_evidence_fails_before_model_dispatch(state):
    state.asset = PublicAsset(b"b" * 10000, "text/plain", _URL)
    with pytest.raises(WatchCheckError, match="complete-comparison limit"):
        await state.executor(_run(state, old="a" * 10000))
    assert not state.clients and state.repository.operations == ["fetch"]


@pytest.mark.parametrize("significant", [True, False])
async def test_observed_change_and_local_judgment_are_distinct_with_exact_evidence(state, significant):
    state.response["message"]["content"] = json.dumps({"significant": significant, "summary": "Release number changed."})
    run = _run(state)
    result = await state.executor(run)
    assert result["outcome"] == "changed" and result["significant"] is significant
    evidence = json.loads(result["diff"])
    for name, text in (("old", run["baseline"]["text"]), ("new", result["text"])):
        excerpt = evidence[name]
        assert text[excerpt["start"]:excerpt["end"]] == excerpt["text"]
    request = state.calls[0]
    assert request["model"] == _MODEL and request["think"] is False
    assert request["options"] == {"temperature": 0, "num_predict": 1024}
    assert request["format"]["additionalProperties"] is False and not request.get("tools")
    assert json.loads(request["messages"][1]["content"])["evidence"] == evidence
    assert state.client_options == [("http://ollama.test", {"trust_env": False})]
    assert state.repository.operations == ["fetch", "model"]
    assert all(client._client.is_closed for client in state.clients)


@pytest.mark.parametrize("failure", ["length", "incomplete", "tools", "malformed", "string_bool", "empty", "extra", "usage"])
async def test_incomplete_or_invalid_model_judgment_never_returns_a_successful_snapshot(state, failure):
    if failure == "length":
        state.response["done_reason"] = "length"
    elif failure == "incomplete":
        state.response["done"] = False
    elif failure == "tools":
        state.response["message"]["tool_calls"] = [{"function": {"name": "web_fetch"}}]
    elif failure == "usage":
        state.response.pop("eval_count")
    else:
        state.response["message"]["content"] = {
            "malformed": "not JSON", "string_bool": '{"significant":"yes","summary":"Change"}',
            "empty": '{"significant":true,"summary":" "}',
            "extra": '{"significant":true,"summary":"Change","action":"send"}',
        }[failure]
    with pytest.raises((WatchCheckError, GenerationUsageError)):
        await state.executor(_run(state))
    assert len(state.calls) == 1 and state.repository.operations == ["fetch", "model"]


@pytest.mark.parametrize("failure", ["missing", "remote", "digest"])
async def test_unverified_local_model_stops_without_any_content_or_cloud_call(state, failure):
    if failure == "missing":
        state.installed.clear()
    elif failure == "remote":
        state.metadata["remote_host"] = "https://cloud.test"
    else:
        async def replace_digest():
            state.installed[0]["digest"] = "b" * 64 if state.installed[0]["digest"] == "a" * 64 else "a" * 64
        state.on_tags = replace_digest
    with pytest.raises(WatchCheckError):
        await state.executor(_run(state))
    assert not state.calls


@pytest.mark.parametrize("phase", ["before_fetch", "model_metadata", "model_response"])
async def test_lease_revocation_stops_network_or_result_delivery(state, phase):
    if phase == "before_fetch":
        state.repository.revoked = True
    elif phase == "model_metadata":
        async def revoke():
            state.repository.revoked = True
        state.on_tags = revoke
    else:
        async def revoke():
            state.repository.revoked = True
        state.on_chat = revoke
    with pytest.raises(AgentDispatchStoppedError):
        await state.executor(_run(state))
    assert len(state.calls) == (1 if phase == "model_response" else 0)
    if phase == "before_fetch":
        assert not state.resolved and not state.fetches


@pytest.mark.parametrize("phase", ["before_fetch", "model_metadata", "model_response"])
async def test_canonical_bot_membership_revocation_stops_fetch_generation_and_result(state, phase):
    if phase == "before_fetch":
        original_resolver = state.executor.resolver

        async def grant_bots_during_resolution(url, **kwargs):
            result = await original_resolver(url, **kwargs)
            await _add_owner_to_bots(state)
            return result

        state.executor.resolver = grant_bots_during_resolution
    elif phase == "model_metadata":
        async def grant_bots():
            await _add_owner_to_bots(state)

        state.on_tags = grant_bots
    else:
        async def grant_bots():
            await _add_owner_to_bots(state)

        state.on_chat = grant_bots
    with pytest.raises(AgentDispatchStoppedError, match="Watch account access is unavailable"):
        await state.executor(_run(state))
    assert len(state.calls) == (1 if phase == "model_response" else 0)
    assert all(client._client.is_closed for client in state.clients)
    if phase == "before_fetch":
        assert state.resolved and not state.fetches and not state.clients


async def test_pause_during_model_access_lookup_fences_the_following_network_dispatch(state, monkeypatch):
    async def revoke_during_lookup(*_args, **_kwargs):
        state.repository.revoked = True
        return object()

    monkeypatch.setattr("audrey.watch_executor.resolve_model", revoke_during_lookup)
    with pytest.raises(AgentDispatchStoppedError):
        await state.executor(_run(state, old=None))
    assert not state.resolved and not state.fetches and not state.clients


@pytest.mark.parametrize("boundary", ["disabled", "agents_disabled", "bot", "bots", "model", "private"])
async def test_live_account_and_policy_requirements_apply_before_even_a_baseline_fetch(state, boundary, monkeypatch):
    if boundary == "disabled":
        state.app.state.cfg.raw["watches"] = {"enabled": False}
    elif boundary == "agents_disabled":
        state.app.state.cfg.raw["agent_tasks"] = {"enabled": False}
    elif boundary == "bot":
        async def bot_principal(**_kwargs):
            return replace(state.owner, role="bot")
        monkeypatch.setattr(state.store, "get_principal", bot_principal)
    elif boundary == "bots":
        await _add_owner_to_bots(state)
    elif boundary == "model":
        await state.store.set_model_access_policy(actor_user_id=state.owner.user_id, model_id="local", enabled=False, audience="users")
    if boundary == "private":
        with processing_scope(ProcessingPolicy(local_models={_MODEL: "a" * 64}, ollama_base_url="http://ollama.test")):
            with pytest.raises(AgentDispatchStoppedError):
                await state.executor(_run(state, old=None))
    else:
        with pytest.raises(AgentDispatchStoppedError):
            await state.executor(_run(state, old=None))
    assert not state.resolved and not state.fetches and not state.clients


def test_template_normalization_removes_noise_and_retains_substantive_article_header():
    asset = PublicAsset(b"""<html><body><header>Site clock 12:00</header><nav>Menu</nav>
    <main><article><header><h1>Release 2</h1></header><p>Price is $27.50.</p>
    <script>steal()</script><style>.x{}</style><p hidden>Hidden</p></article></main>
    <footer>Copyright 2026</footer></body></html>""", "text/html", _URL)
    assert extract_public_text(asset) == "Release 2\nPrice is $27.50."


@pytest.mark.parametrize("data,mime", [
    (b"", "text/plain"), (b"<script>Only script</script>", "text/html"),
    (b"PDF bytes", "application/pdf"), (b"\xff", "text/plain"),
    (b"\x00binary", "text/plain"), (b"x" * 64001, "text/plain"),
    (b"x" * (256 * 1024 + 1), "text/html"),
])
def test_empty_unsupported_or_excessive_text_is_rejected_without_truncation(data, mime):
    with pytest.raises(WatchCheckError):
        extract_public_text(PublicAsset(data, mime, _URL))


def test_diff_limit_never_silently_omits_changed_text():
    with pytest.raises(WatchCheckError, match="complete-comparison limit"):
        evidence_diff("a" * 10000, "b" * 10000)
    old, new = "é" * 6000 + "old", "é" * 6000 + "new"
    evidence = json.loads(evidence_diff(old, new))
    assert evidence["old"]["text"] == old[evidence["old"]["start"]:evidence["old"]["end"]]


@pytest.mark.parametrize("url", ["https://user:secret@example.org/", "file:///tmp/a", "http://example.org:8000/", "https://example.org/?q=change"])
def test_watch_url_rejects_authentication_queries_and_nonpublic_protocols(url):
    with pytest.raises(ValueError):
        validate_watch_url(url)


class Body(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"hello"


@pytest.fixture
def dns(monkeypatch):
    calls = []

    def resolve(host, port, **_kwargs):
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port))]

    monkeypatch.setattr(public_fetch.socket, "getaddrinfo", resolve)
    return calls


async def test_watch_fetch_rejects_redirect_queries_before_their_dns_or_socket(dns):
    requests = []

    def serve(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://another.org/?token=secret"})

    resolved = await public_fetch.resolve_public_url(_URL, no_query=True)
    with pytest.raises(HTTPException):
        await public_fetch.fetch_public_asset(resolved, max_bytes=20, no_query=True, transport=httpx.MockTransport(serve))
    assert len(requests) == 1 and dns == ["example.org."]


async def test_watch_fetch_rejects_partial_declared_body(dns):
    resolved = await public_fetch.resolve_public_url(_URL)
    with pytest.raises(HTTPException, match="incomplete"):
        await public_fetch.fetch_public_asset(
            resolved, max_bytes=20, strict_length=True,
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, headers={"content-length": "12"}, stream=Body())),
        )


async def test_fetch_rechecks_authority_before_redirect_dns_and_socket(dns):
    allowed = True
    requests = []

    async def authorize():
        if not allowed:
            raise AgentDispatchStoppedError("Watch paused.")

    def serve(request):
        nonlocal allowed
        requests.append(request)
        allowed = False
        return httpx.Response(302, headers={"location": "https://another.org/final"})

    resolved = await public_fetch.resolve_public_url(_URL, authorize=authorize)
    with pytest.raises(AgentDispatchStoppedError):
        await public_fetch.fetch_public_asset(resolved, max_bytes=20, authorize=authorize, transport=httpx.MockTransport(serve))
    assert len(requests) == 1 and dns == ["example.org."]


async def test_real_repository_runtime_snapshots_and_notices_preserve_prior_baseline_on_failed_judgment(state):
    # Reattach the canonical repository so leases, receipts, snapshots and
    # notices are exercised together with the real mocked Ollama transport.
    from audrey.app_state.watches import WatchesRepository

    repository = WatchesRepository(state.store._conn, state.store._lock)
    state.store.watches = repository
    watch = await repository.create(
        user_id=state.owner.user_id, title="Release updates", url=_URL,
        criterion="New release announcements", request_id="integration",
        schedule={"kind": "interval", "timezone": "UTC", "interval_minutes": 60},
    )
    runtime = WatchRuntime(repository, state.executor)

    async def check():
        await repository.run_now(user_id=state.owner.user_id, watch_id=watch["id"], expected_revision=watch["revision"])
        assert await runtime.run_once()
        return (await repository.list_runs(user_id=state.owner.user_id, watch_id=watch["id"]))[0]

    assert (await check())["outcome"] == "baseline"
    assert not state.clients
    first_baseline = repository._baseline(watch["id"])
    assert (await check())["outcome"] == "unchanged"
    assert await repository.unread_count(user_id=state.owner.user_id) == 0
    state.asset = PublicAsset(b"Release 3 is available.", "text/plain", _URL)
    changed = await check()
    assert changed["outcome"] == "changed" and changed["significant"] is True
    assert await repository.unread_count(user_id=state.owner.user_id) == 1
    valid_baseline = repository._baseline(watch["id"])
    assert valid_baseline["text"] == "Release 3 is available."
    assert first_baseline["text"] == "Release 2 is available."

    state.asset = PublicAsset(b"Release 4 is available.", "text/plain", _URL)
    state.response["done_reason"] = "length"
    failed = await check()
    assert failed["status"] == "queued" and failed["outcome"] is None
    assert "did not finish" in failed["error"]
    assert repository._baseline(watch["id"]) == valid_baseline
    assert await repository.unread_count(user_id=state.owner.user_id) == 1
    # The failed occurrence is in durable backoff; repeated wakes cannot retry
    # immediately or manufacture another notice.
    assert not await runtime.run_once()
