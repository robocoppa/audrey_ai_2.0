"""Tests for `/v1/media/describe` (Phase 36).

This route exists because the phase plan's transport did not: it had the
worker call `/v1/chat/completions` "acting as the uploading user via the Phase
31 service-token act-as", and that route depends on `require_user`, which
wants a real OWUI bearer. The act-as lives only on the KB query routes.

Since the alternative considered was widening `require_user` on the endpoint
every OWUI user hits, the auth tests here are the ones that matter most — a
route that grants an arbitrary identity must be reachable by the service token
and by nothing else.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from audrey.models.ollama import OllamaError
from audrey.pipeline.vision import VisionUnavailableError
from audrey.routes.media import router

SECRET = "s3cr3t-service-token"  # noqa: S105  (test fixture, not a real secret)
JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0 not really a jpeg").decode()


class _Inflight:
    """Records which bucket the call was accounted to."""

    def __init__(self) -> None:
        self.buckets: list[str] = []

    def slot(self, user_id):
        self.buckets.append(user_id)

        class _Ctx:
            async def __aenter__(self): return None
            async def __aexit__(self, *a): return False
        return _Ctx()


def _build_app(*, describe=None, service_token: str = SECRET) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.cfg = SimpleNamespace(
        env=SimpleNamespace(kb_service_token=service_token, owui_url="http://owui"),
        raw={"vision": {"timeout_s": 120}},
    )
    for attr in ("ollama", "registry", "health", "gate"):
        setattr(app.state, attr, object())
    app.state.inflight = _Inflight()
    app.state.captured = {}

    async def _default(data_url, **kwargs):
        app.state.captured = {"data_url": data_url, **kwargs}
        return "A whiteboard reading DEPLOY FRIDAY.", "qwen3-vl:32b"

    app.state.describe_impl = describe or _default
    return app


@pytest.fixture
def patched(monkeypatch):
    """Swap `describe_one_image` for whatever the app carries."""
    def _make(**kwargs) -> TestClient:
        app = _build_app(**kwargs)

        async def _call(data_url, **kw):
            return await app.state.describe_impl(data_url, **kw)

        monkeypatch.setattr("audrey.routes.media.describe_one_image", _call)
        return TestClient(app)
    return _make


def _body(**over):
    return {"user": "bart@proton.me", "image_b64": JPEG_B64, **over}


class TestAuth:
    """The reason this route is narrow. It accepts a `user` field and acts as
    that identity, so reaching it must require the service token."""

    def test_no_token_is_401(self, patched):
        assert patched().post("/v1/media/describe", json=_body()).status_code == 401

    def test_a_wrong_service_token_is_401(self, patched):
        r = patched().post("/v1/media/describe", json=_body(),
                           headers={"X-Audrey-Service-Token": "nope"})
        assert r.status_code == 401

    def test_a_user_bearer_is_not_enough(self, patched):
        """A logged-in user must not be able to name someone else and spend
        their GPU slice — or to reach the vl pool outside the chat surface."""
        r = patched().post("/v1/media/describe", json=_body(),
                           headers={"Authorization": "Bearer a-real-owui-jwt"})
        assert r.status_code == 401

    def test_a_blank_configured_secret_never_authenticates(self, patched):
        client = patched(service_token="")
        assert client.post("/v1/media/describe", json=_body(),
                           headers={"X-Audrey-Service-Token": ""}).status_code == 401


class TestFairness:
    """The property the plan wanted from passthrough, and the only reason it
    mattered which identity the call ran as."""

    def test_the_gate_is_told_the_uploading_user(self, patched):
        client = patched()
        client.post("/v1/media/describe", json=_body(user="bart@proton.me"),
                    headers={"X-Audrey-Service-Token": SECRET})

        assert client.app.state.captured["user_id"] == "bart@proton.me"

    def test_the_inflight_slot_is_the_uploading_user(self, patched):
        """A giant video should slow its own owner's chat and leave everyone
        else's alone. A shared service identity would pool every ingest into
        one slice and blur exactly that."""
        client = patched()
        client.post("/v1/media/describe", json=_body(user="someone@else.com"),
                    headers={"X-Audrey-Service-Token": SECRET})

        assert client.app.state.inflight.buckets == ["someone@else.com"]


class TestImageHandling:
    def test_the_data_uri_is_built_here_not_by_the_caller(self, patched):
        """`ollama.py` silently drops `http(s)://` image URLs and answers
        blind. Taking raw base64 means there is no field to put a URL in."""
        client = patched()
        client.post("/v1/media/describe", json=_body(),
                    headers={"X-Audrey-Service-Token": SECRET})

        assert client.app.state.captured["data_url"].startswith("data:image/jpeg;base64,")

    def test_invalid_base64_is_refused_before_a_gpu_slot_is_taken(self, patched):
        client = patched()
        r = client.post("/v1/media/describe", json=_body(image_b64="not base64!!"),
                        headers={"X-Audrey-Service-Token": SECRET})

        assert r.status_code == 422
        assert client.app.state.inflight.buckets == []

    def test_an_unexpected_mime_is_refused(self, patched):
        r = patched().post("/v1/media/describe", json=_body(mime="application/pdf"),
                           headers={"X-Audrey-Service-Token": SECRET})
        assert r.status_code == 422

    def test_the_hint_reaches_the_model_as_a_question_not_an_instruction(self, patched):
        client = patched()
        client.post("/v1/media/describe",
                    json=_body(hint="what does the whiteboard say"),
                    headers={"X-Audrey-Service-Token": SECRET})

        assert client.app.state.captured["user_question"] == "what does the whiteboard say"


class TestFailureModes:
    """An ingest has no user waiting, so it fails loudly rather than degrading
    the way `describe_images` does for chat."""

    def test_no_healthy_vl_model_is_503(self, patched):
        async def _boom(data_url, **kw):
            raise VisionUnavailableError("no healthy vl model is available")

        r = patched(describe=_boom).post(
            "/v1/media/describe", json=_body(),
            headers={"X-Audrey-Service-Token": SECRET})

        # 503 not 502: retrying later may well succeed.
        assert r.status_code == 503

    def test_a_model_failure_is_502(self, patched):
        async def _boom(data_url, **kw):
            raise OllamaError("timed out")

        r = patched(describe=_boom).post(
            "/v1/media/describe", json=_body(),
            headers={"X-Audrey-Service-Token": SECRET})
        assert r.status_code == 502

    def test_an_empty_description_is_a_failure_not_a_success(self, patched):
        """It would otherwise be ingested as an empty chunk, and nobody would
        ever notice that a frame had not been described."""
        async def _empty(data_url, **kw):
            return "", "qwen3-vl:32b"

        r = patched(describe=_empty).post(
            "/v1/media/describe", json=_body(),
            headers={"X-Audrey-Service-Token": SECRET})
        assert r.status_code == 502

    def test_an_uninitialised_app_is_503_not_a_500(self, patched):
        client = patched()
        client.app.state.gate = None
        r = client.post("/v1/media/describe", json=_body(),
                        headers={"X-Audrey-Service-Token": SECRET})
        assert r.status_code == 503


class TestSuccess:
    def test_the_description_and_model_come_back(self, patched):
        r = patched().post("/v1/media/describe", json=_body(),
                           headers={"X-Audrey-Service-Token": SECRET})

        assert r.status_code == 200
        body = r.json()
        assert body["description"] == "A whiteboard reading DEPLOY FRIDAY."
        assert body["model"] == "qwen3-vl:32b"
        assert body["elapsed_s"] >= 0.0

    def test_the_latency_is_recorded_for_phase_38(self, patched):
        """Without a per-frame number, "video ingest is slow" is not an
        actionable sentence and phase 38 is guesswork."""
        from audrey.metrics import video_describe_seconds

        before = video_describe_seconds._sum.get()
        patched().post("/v1/media/describe", json=_body(),
                       headers={"X-Audrey-Service-Token": SECRET})

        assert video_describe_seconds._sum.get() >= before
