"""Hermetic contract for the unexposed local-processing foundation."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from audrey.processing_policy import (
    LOCAL_PROCESSING_TOOLS,
    ProcessingPolicy,
    ProcessingPolicyError,
    current_processing_policy,
    processing_scope,
)

DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64
MODEL = "qwen3.8:latest"
OLLAMA = "http://ollama:11434"
TOOLS = "http://custom-tools:8001"


def policy(**overrides):
    arguments = {
        "local_models": {MODEL: DIGEST},
        "ollama_base_url": OLLAMA,
        "allowed_tools": frozenset({"calculate", "read_reference"}),
        "tool_servers": frozenset({TOOLS}),
    }
    return ProcessingPolicy(**(arguments | overrides))


def evidence():
    return {
        "installed_models": [{"name": MODEL, "model": MODEL, "digest": DIGEST}],
        "metadata": {"model_info": {"general.architecture": "qwen3", "general.parameter_count": 27_000_000_000}, "details": {"format": "gguf"}},
    }


def test_default_is_unrestricted_and_scope_restores_after_failure():
    assert current_processing_policy() is None
    protected = policy()
    with pytest.raises(ValueError), processing_scope(protected):
        assert current_processing_policy() is protected
        raise ValueError("ordinary failure")
    assert current_processing_policy() is None


def test_policy_copies_and_freezes_all_operator_supplied_state():
    models = {MODEL: DIGEST}
    tools = {"calculate"}
    servers = {TOOLS}
    protected = policy(local_models=models, allowed_tools=tools, tool_servers=servers)
    models[MODEL] = OTHER_DIGEST
    tools.add("read_reference")
    servers.add("https://example.com")
    assert dict(protected.local_models) == {MODEL: DIGEST}
    assert protected.allowed_tools == frozenset({"calculate"})
    assert protected.tool_servers == frozenset({TOOLS})
    with pytest.raises(TypeError):
        protected.local_models[MODEL] = OTHER_DIGEST
    with pytest.raises(FrozenInstanceError):
        protected.ollama_base_url = "https://example.com"


@pytest.mark.parametrize("models", [{}, None, [], {"bad model": DIGEST}, {"x:cloud": DIGEST}, {"x:397b-cloud": DIGEST}, {MODEL: "not-a-digest"}, {MODEL: DIGEST, "qwen3.8": DIGEST}])
def test_invalid_or_empty_model_allowlist_is_refused(models):
    with pytest.raises(ProcessingPolicyError) as caught:
        policy(local_models=models)
    assert caught.value.code == "invalid_processing_policy"


@pytest.mark.parametrize("name", ["qwen3.8:cloud", "qwen3.8:397b-cloud", "qwen3.8:other", "Qwen3.8:latest", "qwen3.8:latest ", "qwen3.8:latest\n", "../qwen3.8", "/qwen3.8", "qwen3.8:", "qwen3.8:latest:extra", None, 3])
def test_model_admission_is_exact_and_fails_closed(name):
    with pytest.raises(ProcessingPolicyError) as caught:
        policy().assert_model_allowed(name)
    assert caught.value.code == "local_model_not_allowed"


def test_only_default_latest_and_sha256_digest_prefix_are_normalized():
    protected = policy(local_models={"qwen3.8": "sha256:" + DIGEST})
    assert dict(protected.local_models) == {MODEL: DIGEST}
    protected.assert_model_allowed(MODEL)
    protected.assert_model_allowed("qwen3.8")
    proof = evidence()
    proof["installed_models"][0]["digest"] = "sha256:" + DIGEST
    protected.assert_local_model_metadata("qwen3.8", **proof)


@pytest.mark.parametrize("url", ["https://user:secret@example.com", "http://ollama:11434?", "http://ollama:11434#", "ftp://ollama", "http://", "http://ollama:abc", "http://ollama:11434/../api", "http://ollama:11434/%2e%2e/api", " http://ollama:11434", "http://ollama:11434\n", "http://ollama:11434\\@example.com"])
def test_invalid_service_endpoints_are_refused(url):
    with pytest.raises(ProcessingPolicyError):
        policy(ollama_base_url=url)


def test_endpoint_identity_canonicalizes_case_default_ports_and_trailing_slash():
    protected = policy(ollama_base_url="http://OLLAMA:80/api/", tool_servers={"https://CUSTOM-TOOLS:443/"})
    protected.assert_ollama_endpoint("http://ollama/api")
    protected.assert_tool_allowed("calculate", server_url="https://custom-tools")
    ipv6 = policy(ollama_base_url="http://[::1]:11434/")
    ipv6.assert_ollama_endpoint("http://[0:0:0:0:0:0:0:1]:11434")


@pytest.mark.parametrize("url", ["https://ollama:11434", "http://ollama:11435", "http://other:11434", "http://ollama:11434/other", "http://user:secret@ollama:11434", "bad"])
def test_wrong_ollama_endpoint_is_refused(url):
    with pytest.raises(ProcessingPolicyError) as caught:
        policy().assert_ollama_endpoint(url)
    assert caught.value.code == "local_model_endpoint_blocked"


def test_local_metadata_passes_with_unrelated_cloud_catalog_entries():
    proof = evidence()
    proof["installed_models"].append({"name": "kimi-k3:cloud", "digest": OTHER_DIGEST})
    policy().assert_local_model_metadata(MODEL, **proof)


@pytest.mark.parametrize("location", ["installed_models", "metadata"])
@pytest.mark.parametrize("field", ["remote_host", "remote_model"])
@pytest.mark.parametrize("value", ["https://ollama.com", "", 0, [], {}])
def test_any_ambiguous_or_remote_marker_is_refused(location, field, value):
    proof = evidence()
    target = proof[location][0] if location == "installed_models" else proof[location]
    target[field] = value
    with pytest.raises(ProcessingPolicyError) as caught:
        policy().assert_local_model_metadata(MODEL, **proof)
    assert caught.value.code == "local_model_remote"


@pytest.mark.parametrize("value", [None, False])
def test_explicit_absent_remote_markers_are_accepted(value):
    proof = evidence()
    for target in (proof["installed_models"][0], proof["metadata"]):
        target.update(remote_host=value, remote_model=value)
    policy().assert_local_model_metadata(MODEL, **proof)


@pytest.mark.parametrize("proof,code", [
    ({"installed_models": [], "metadata": {}}, "local_model_not_installed"),
    ({"installed_models": [{"name": MODEL, "digest": OTHER_DIGEST}], "metadata": {}}, "local_model_digest_mismatch"),
    ({"installed_models": [{"name": MODEL}], "metadata": {}}, "local_model_metadata_invalid"),
    ({"installed_models": [{"name": MODEL, "model": "different:latest", "digest": DIGEST}], "metadata": {}}, "local_model_metadata_invalid"),
    ({"installed_models": "invalid", "metadata": {}}, "local_model_metadata_invalid"),
    ({"installed_models": [None], "metadata": {}}, "local_model_metadata_invalid"),
])
def test_installed_identity_must_be_unambiguous_and_match_digest(proof, code):
    with pytest.raises(ProcessingPolicyError) as caught:
        policy().assert_local_model_metadata(MODEL, **proof)
    assert caught.value.code == code


def test_duplicate_installed_identity_is_refused():
    proof = evidence()
    proof["installed_models"] *= 2
    with pytest.raises(ProcessingPolicyError) as caught:
        policy().assert_local_model_metadata(MODEL, **proof)
    assert caught.value.code == "local_model_metadata_invalid"


@pytest.mark.parametrize("metadata", [None, {}, {"model_info": {}, "details": {"format": "gguf"}}, {"model_info": {"x": 1}}, {"model_info": {"x": 1}, "details": {"format": "other"}}, {"model_info": [], "details": {"format": "gguf"}}])
def test_local_weights_proof_is_required(metadata):
    proof = evidence() | {"metadata": metadata}
    with pytest.raises(ProcessingPolicyError) as caught:
        policy().assert_local_model_metadata(MODEL, **proof)
    assert caught.value.code == "local_model_metadata_invalid"


@pytest.mark.parametrize("changes", [
    {"general.architecture": ""}, {"general.architecture": " "},
    {"general.architecture": 1}, {"general.architecture": None},
    {"general.parameter_count": True}, {"general.parameter_count": 0},
    {"general.parameter_count": -1}, {"general.parameter_count": "27000000000"},
    {"general.parameter_count": None}, {"general.parameter_count": 1.0},
])
def test_local_weight_architecture_and_parameter_count_are_required(changes):
    proof = evidence()
    proof["metadata"]["model_info"].update(changes)
    with pytest.raises(ProcessingPolicyError) as caught:
        policy().assert_local_model_metadata(MODEL, **proof)
    assert caught.value.code == "local_model_metadata_invalid"


def test_unknown_model_is_refused_before_inspecting_metadata():
    with pytest.raises(ProcessingPolicyError) as caught:
        policy().assert_local_model_metadata("unknown", installed_models=None, metadata=None)
    assert caught.value.code == "local_model_not_allowed"


@pytest.mark.parametrize("name", ["web_search", "web_fetch", "memory_store", "memory_recall", "memory_search", "chat_history_search", "open_chat_history", "inspect_pdf_page", "inspect_video_frame", "unknown", "Calculate", "calculate "])
def test_external_model_using_write_and_unknown_tools_cannot_be_authorized(name):
    with pytest.raises(ProcessingPolicyError):
        policy(allowed_tools={name})


def test_default_tool_policy_denies_all_and_model_free_tools_need_exact_endpoint():
    protected = ProcessingPolicy(local_models={MODEL: DIGEST}, ollama_base_url=OLLAMA)
    for name in LOCAL_PROCESSING_TOOLS:
        with pytest.raises(ProcessingPolicyError):
            protected.assert_tool_allowed(name, server_url=TOOLS)
    protected = policy(allowed_tools=set(LOCAL_PROCESSING_TOOLS))
    for name in LOCAL_PROCESSING_TOOLS:
        protected.assert_tool_allowed(name, server_url=TOOLS)
    with pytest.raises(ProcessingPolicyError) as caught:
        protected.assert_tool_allowed("calculate", server_url="https://external.example")
    assert caught.value.code == "local_tool_endpoint_blocked"


def test_tool_name_is_refused_before_even_checking_endpoint():
    with pytest.raises(ProcessingPolicyError) as caught:
        policy().assert_tool_allowed("web_search", server_url="bad")
    assert caught.value.code == "local_tool_not_allowed"


@pytest.mark.parametrize("changes", [None, {"local_models": {MODEL: OTHER_DIGEST}}, {"local_models": {MODEL: DIGEST, "other": DIGEST}}, {"allowed_tools": {"calculate", "read_reference", "get_file_text"}}, {"tool_servers": {TOOLS, "http://other:8001"}}, {"ollama_base_url": "http://other:11434"}])
def test_nested_scope_cannot_weaken_any_protected_boundary(changes):
    protected = policy()
    child = None if changes is None else policy(**changes)
    with processing_scope(protected):
        with pytest.raises(ProcessingPolicyError) as caught, processing_scope(child):
            pytest.fail("weakening must be refused")
        assert caught.value.code == "processing_policy_downgrade"
        assert current_processing_policy() is protected
    assert current_processing_policy() is None


def test_nested_stricter_scope_restores_parent():
    parent = policy(local_models={MODEL: DIGEST, "other": OTHER_DIGEST})
    child = policy(allowed_tools={"calculate"})
    with processing_scope(parent):
        with processing_scope(child):
            assert current_processing_policy() is child
        assert current_processing_policy() is parent


async def test_async_children_inherit_protection_and_parallel_requests_are_isolated():
    protected = policy()
    started = asyncio.Event()
    release = asyncio.Event()

    async def child():
        assert current_processing_policy() is protected
        started.set()
        await release.wait()
        return current_processing_policy()

    async def ordinary():
        await started.wait()
        assert current_processing_policy() is None
        release.set()

    ordinary_task = asyncio.create_task(ordinary())
    with processing_scope(protected):
        child_task = asyncio.create_task(child())
        await started.wait()
    assert current_processing_policy() is None
    assert await child_task is protected
    await ordinary_task


def test_safe_errors_never_echo_untrusted_model_endpoint_or_tool_values():
    marker = "private-tax-value"
    checks = [
        lambda: policy().assert_model_allowed(marker),
        lambda: policy().assert_ollama_endpoint(f"https://{marker}.example"),
        lambda: policy().assert_tool_allowed(marker),
        lambda: policy().assert_tool_allowed("calculate", server_url=f"https://{marker}.example"),
    ]
    for check in checks:
        with pytest.raises(ProcessingPolicyError) as caught:
            check()
        assert marker not in str(caught.value)
        assert marker not in repr(caught.value)
