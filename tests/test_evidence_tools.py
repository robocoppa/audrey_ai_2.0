"""Owner-scoped evidence reads and bounded renderer process lifecycles."""
from __future__ import annotations

import asyncio
import base64
import io
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from PIL import Image
from pypdf import PdfWriter

from audrey.kb.uploads_db import UploadsDB
from audrey.kb.user_store import sanitize_user
from audrey.media import visual_render
from audrey.models.ollama import OllamaError
from audrey.pipeline.vision import VisionUnavailableError
from audrey.routes import evidence_tools

_SECRET = "test-evidence-service-secret"  # noqa: S105
_OWNER = "alice@example.com"
_HEADERS = {"X-Audrey-Service-Token": _SECRET}
_FID = "11111111-1111-1111-1111-111111111111"
_OTHER_FID = "22222222-2222-2222-2222-222222222222"
_IMAGE = b"\x89PNG\r\n\x1a\nrendered-page"


@pytest.fixture
def db(tmp_path):
    store = UploadsDB(tmp_path / "uploads.sqlite")
    yield store
    store.close()


@pytest.fixture
def app(db, tmp_path):
    instance = FastAPI()
    instance.include_router(evidence_tools.router)
    instance.state.uploads_db = db
    instance.state.cfg = SimpleNamespace(
        env=SimpleNamespace(kb_service_token=_SECRET),
        raw={"kb": {"upload_root": str(tmp_path / "uploads")}},
    )
    for name in ("ollama", "registry", "health", "gate"):
        setattr(instance.state, name, object())
    return instance


@pytest.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test",
    ) as instance:
        yield instance


async def _record(db, tmp_path, *, filename="drawing.pdf", file_id=_FID,
                  user=_OWNER, status="ready", mime="application/pdf", kind="text",
                  uploaded_at="2026-10-13T09:00:00+00:00", content=b"owned bytes",
                  duration=0):
    await db.record_upload(
        file_id=file_id, user=user, filename=filename, mime=mime, bytes_=len(content),
        kind=kind, collection="owned", chunks=1, uploaded_at=uploaded_at,
        status="pending" if duration else status,
    )
    if duration:
        lease = f"lease-{file_id}"
        assert await db.claim_job(lease_id=lease, now="2026-10-13T10:00:00+00:00")
        assert await db.complete_job(
            file_id=file_id, lease_id=lease, collection="owned", chunks=1,
            duration_s=duration,
        )
    directory = tmp_path / "uploads" / sanitize_user(user)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{file_id}{Path(filename).suffix.lower()}"
    path.write_bytes(content)
    return path


async def _post(client, suffix="inspect-pdf-page", **fields):
    return await client.post(
        f"/v1/files/{suffix}", headers=_HEADERS,
        json={"user": _OWNER, "filename": "drawing.pdf", "page": 1, **fields},
    )


@pytest.mark.parametrize(("suffix", "fields"), [
    ("inspect-pdf-page", {"page": 1}),
    ("inspect-video-frame", {"timestamp_s": 1.5}),
    ("table", {}),
])
async def test_evidence_routes_require_service_secret(client, suffix, fields):
    body = {"user": _OWNER, "filename": "owned", **fields}
    for headers in ({}, {"X-Audrey-Service-Token": "wrong"},
                    {"Authorization": "Bearer signed-in-user"}):
        response = await client.post(f"/v1/files/{suffix}", headers=headers, json=body)
        assert response.status_code == 401
        assert response.json() == {"detail": "Service token required."}


async def test_pdf_exact_page_image_and_owner_reach_vision(client, db, tmp_path, monkeypatch):
    path = await _record(db, tmp_path)
    render = AsyncMock(return_value=(_IMAGE, 7))
    vision = AsyncMock(return_value=("A load-bearing beam crosses the room.", "vision-model", {}))
    monkeypatch.setattr(evidence_tools, "render_pdf_page", render)
    monkeypatch.setattr(evidence_tools, "describe_one_image", vision)
    response = await _post(client, page=3, question="What does the drawing show?")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["page"] == 3 and result["total_pages"] == 7
    assert result["artifact"] == "pdf_page" and result["model"] == "vision-model"
    assert result["file_id"] == _FID
    render.assert_awaited_once_with(path.resolve(), 3)
    args, kwargs = vision.await_args
    assert args == (f"data:image/png;base64,{base64.b64encode(_IMAGE).decode()}",)
    assert kwargs["user_id"] == _OWNER
    assert kwargs["user_question"] == "What does the drawing show?"
    assert str(tmp_path) not in response.text
    assert base64.b64encode(_IMAGE).decode() not in response.text
    assert "owned bytes" not in response.text


async def test_video_exact_timestamp_and_still_frame_coverage(client, db, tmp_path, monkeypatch):
    path = await _record(
        db, tmp_path, filename="clip.mp4", mime="video/mp4", kind="video", duration=5,
    )
    render = AsyncMock(return_value=b"jpeg-frame")
    vision = AsyncMock(return_value=("A person holds a wrench.", "vision-model", {}))
    monkeypatch.setattr(evidence_tools, "render_video_frame", render)
    monkeypatch.setattr(evidence_tools, "describe_one_image", vision)
    response = await client.post(
        "/v1/files/inspect-video-frame", headers=_HEADERS,
        json={"user": _OWNER, "filename": "clip.mp4", "timestamp_s": 2.25, "question": "What is visible?"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["timestamp_s"] == 2.25 and result["artifact"] == "video_frame"
    assert "does not establish movement or speech" in result["coverage"]
    render.assert_awaited_once_with(path.resolve(), 2.25)
    assert vision.await_args.args[0] == "data:image/jpeg;base64,anBlZy1mcmFtZQ=="
    assert str(tmp_path) not in response.text


async def test_foreign_and_missing_file_have_identical_not_found(client, db, tmp_path):
    await _record(db, tmp_path, user="bob@example.com")
    foreign = await _post(client)
    missing = await _post(client, filename="missing.pdf")
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json() == {"detail": "File not found."}
    assert str(tmp_path) not in foreign.text


async def test_duplicate_name_uses_newest_owned_source(client, db, tmp_path, monkeypatch):
    await _record(db, tmp_path, uploaded_at="2026-10-12T09:00:00+00:00")
    newest = await _record(
        db, tmp_path, file_id=_OTHER_FID, uploaded_at="2026-10-13T09:00:00+00:00",
    )
    await _record(db, tmp_path, user="bob@example.com", file_id="foreign",
                  uploaded_at="2026-10-14T09:00:00+00:00", content=b"FOREIGN-SECRET")
    render = AsyncMock(return_value=(_IMAGE, 1))
    monkeypatch.setattr(evidence_tools, "render_pdf_page", render)
    monkeypatch.setattr(evidence_tools, "describe_one_image",
                        AsyncMock(return_value=("visible", "vision-model", {})))
    response = await _post(client, filename=" DRAWING.PDF ")
    assert response.status_code == 200, response.text
    assert response.json()["file_id"] == _OTHER_FID
    assert response.json()["note"] == (
        "2 uploads share this name; using the most recent (2026-10-13T09:00:00+00:00)."
    )
    render.assert_awaited_once_with(newest.resolve(), 1)
    assert "FOREIGN-SECRET" not in response.text


@pytest.mark.parametrize("state", ["pending", "failed", "missing", "freed", "escaped"])
async def test_unready_or_unavailable_original_does_not_render(
    client, db, tmp_path, monkeypatch, state,
):
    path = await _record(db, tmp_path, status=state if state in {"pending", "failed"} else "ready")
    if state == "missing":
        path.unlink()
    elif state == "freed":
        assert await db.mark_source_freed(_FID, freed_at="2026-10-13T11:00:00+00:00")
    elif state == "escaped":
        path.unlink()
        external = tmp_path / "foreign-secret.pdf"
        external.write_bytes(b"FOREIGN-SECRET")
        path.symlink_to(external)
    render = AsyncMock()
    monkeypatch.setattr(evidence_tools, "render_pdf_page", render)
    response = await _post(client)
    assert response.status_code == 409
    render.assert_not_called()
    assert str(tmp_path) not in response.text and "FOREIGN-SECRET" not in response.text


@pytest.mark.parametrize(("suffix", "fields", "mime", "kind"), [
    ("inspect-pdf-page", {"page": 1}, "text/plain", "text"),
    ("inspect-video-frame", {"timestamp_s": 1}, "audio/mpeg", "audio"),
])
async def test_wrong_file_kind_is_rejected(client, db, tmp_path, suffix, fields, mime, kind):
    await _record(db, tmp_path, mime=mime, kind=kind)
    response = await client.post(
        f"/v1/files/{suffix}", headers=_HEADERS,
        json={"user": _OWNER, "filename": "drawing.pdf", **fields},
    )
    assert response.status_code == 422


async def test_timestamp_at_duration_is_rejected_before_render(client, db, tmp_path, monkeypatch):
    await _record(db, tmp_path, filename="clip.mp4", mime="video/mp4", kind="video", duration=5)
    render = AsyncMock()
    monkeypatch.setattr(evidence_tools, "render_video_frame", render)
    response = await client.post(
        "/v1/files/inspect-video-frame", headers=_HEADERS,
        json={"user": _OWNER, "filename": "clip.mp4", "timestamp_s": 5},
    )
    assert response.status_code == 422
    render.assert_not_called()


@pytest.mark.parametrize(("error", "status"), [
    (VisionUnavailableError("private details"), 503),
    (OllamaError("private transport details"), 502),
    (visual_render.VisualRenderError("pdfinfo is unavailable"), 503),
    (visual_render.VisualRenderError("Page must be between 1 and 3."), 422),
    (TimeoutError(), 504),
])
async def test_visual_errors_are_explicit_without_provider_details(
    client, db, tmp_path, monkeypatch, error, status,
):
    await _record(db, tmp_path)
    render = AsyncMock(return_value=(_IMAGE, 3))
    vision = AsyncMock(return_value=("visible", "vision-model", {}))
    if isinstance(error, (VisionUnavailableError, OllamaError)):
        vision.side_effect = error
    else:
        render.side_effect = error
    monkeypatch.setattr(evidence_tools, "render_pdf_page", render)
    monkeypatch.setattr(evidence_tools, "describe_one_image", vision)
    response = await _post(client)
    assert response.status_code == status
    assert "private" not in response.text
    assert str(tmp_path) not in response.text


async def test_empty_vision_and_bounded_observations(client, db, tmp_path, monkeypatch):
    await _record(db, tmp_path)
    monkeypatch.setattr(evidence_tools, "render_pdf_page",
                        AsyncMock(return_value=(_IMAGE, 1)))
    vision = AsyncMock(return_value=(" \n ", "vision-model", {}))
    monkeypatch.setattr(evidence_tools, "describe_one_image", vision)
    empty = await _post(client)
    assert empty.status_code == 502
    assert empty.json()["detail"] == "The vision model returned no observation."
    vision.return_value = ("x" * 2500, "vision-model", {})
    bounded = await _post(client)
    assert bounded.status_code == 200
    assert len(bounded.json()["description"]) == 2000
    assert bounded.json()["description_truncated"] is True


async def test_csv_route_analyzes_owned_source_and_hides_source_path(client, db, tmp_path):
    await _record(
        db, tmp_path, filename="costs.csv", mime="text/csv",
        content=b"Category,Amount\nmaterials,10.25\nmaterials,2.5\nlabor,20\n",
    )
    response = await client.post(
        "/v1/files/table", headers=_HEADERS, json={
            "user": _OWNER, "filename": "costs.csv", "action": "aggregate",
            "group_by": ["Category"], "aggregations": [{"op": "sum", "column": "Amount"}],
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["file_id"] == _FID and result["artifact"] == "table"
    groups = {entry["key"][0]: entry["aggregations"][0]["value"] for entry in result["groups"]}
    assert groups == {"materials": "12.75", "labor": "20"}
    assert str(tmp_path) not in response.text
    assert "Category,Amount" not in response.text


class _WaitingProcess:
    def __init__(self):
        self.returncode = None
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()
        self.waits = 0
        self.kills = 0

    async def wait(self):
        self.waits += 1
        self.started.set()
        await self.stopped.wait()
        return self.returncode

    def kill(self):
        self.kills += 1
        self.returncode = -9
        self.stopped.set()


@pytest.mark.parametrize("cancel", [False, True])
async def test_renderer_timeout_or_cancellation_kills_and_reaps_child(monkeypatch, cancel, tmp_path):
    process = _WaitingProcess()
    captured = {}

    async def launch(*args, **kwargs):
        captured.update(kwargs)
        return process

    monkeypatch.setattr(visual_render.shutil, "which", lambda _: "/usr/bin/renderer")
    monkeypatch.setattr(visual_render.asyncio, "create_subprocess_exec", launch)
    monkeypatch.setattr(visual_render, "RENDER_TIMEOUT_S", 10 if cancel else 0.01)
    task = asyncio.create_task(visual_render._run("renderer", "owned-file", output=tmp_path / "render.png"))
    await process.started.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await task
    assert process.kills == 1 and process.waits == 2
    assert captured["stdout"].closed
    assert process.returncode == -9


async def test_child_exit_during_kill_still_reaps_and_preserves_timeout(monkeypatch):
    class ExitingProcess(_WaitingProcess):
        def kill(self):
            self.kills += 1
            self.returncode = 0
            self.stopped.set()
            raise ProcessLookupError("Child exited between poll and signal")

    process = ExitingProcess()
    monkeypatch.setattr(visual_render.shutil, "which", lambda _: "/usr/bin/renderer")
    monkeypatch.setattr(
        visual_render.asyncio, "create_subprocess_exec", AsyncMock(return_value=process),
    )
    monkeypatch.setattr(visual_render, "RENDER_TIMEOUT_S", 0.01)
    with pytest.raises(TimeoutError):
        await visual_render._run("renderer", "owned-file")
    assert process.kills == 1 and process.waits == 2


@pytest.mark.parametrize("video", [False, True])
async def test_renderer_failure_removes_temporary_directory(tmp_path, monkeypatch, video):
    paths = []
    original = tempfile.TemporaryDirectory

    def make_directory(**kwargs):
        directory = original(dir=tmp_path, **kwargs)
        paths.append(Path(directory.name))
        return directory

    monkeypatch.setattr(visual_render.tempfile, "TemporaryDirectory", make_directory)
    monkeypatch.setattr(visual_render, "_run",
                        AsyncMock(side_effect=visual_render.VisualRenderError("bad source")))
    with pytest.raises(visual_render.VisualRenderError, match="bad source"):
        if video:
            await visual_render.render_video_frame(tmp_path / "clip.mp4", 1)
        else:
            await visual_render.render_pdf_page(tmp_path / "drawing.pdf", 1)
    assert paths and all(not path.exists() for path in paths)


async def test_renderer_cancellation_removes_temporary_directory(tmp_path, monkeypatch):
    paths = []
    original = tempfile.TemporaryDirectory
    started = asyncio.Event()

    def make_directory(**kwargs):
        directory = original(dir=tmp_path, **kwargs)
        paths.append(Path(directory.name))
        return directory

    async def stalled_render(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(visual_render.tempfile, "TemporaryDirectory", make_directory)
    monkeypatch.setattr(visual_render, "_run", stalled_render)
    task = asyncio.create_task(visual_render.render_video_frame(tmp_path / "clip.mp4", 1))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert paths and all(not path.exists() for path in paths)


async def test_missing_renderer_and_empty_or_oversized_images_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(visual_render.shutil, "which", lambda _: None)
    with pytest.raises(visual_render.VisualRenderError, match="unavailable"):
        await visual_render._run("missing-renderer")
    image = tmp_path / "image.png"
    image.write_bytes(b"")
    with pytest.raises(visual_render.VisualRenderError, match="no usable bounded image"):
        visual_render._read_image(image)
    monkeypatch.setattr(visual_render, "MAX_IMAGE_BYTES", 4)
    image.write_bytes(b"12345")
    with pytest.raises(visual_render.VisualRenderError, match="no usable bounded image"):
        visual_render._read_image(image)


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("pdfinfo", "pdftoppm")),
    reason="PDF render binaries unavailable",
)
async def test_real_pdf_renderer_selects_page_and_bounds_dimensions(tmp_path):
    path = tmp_path / "two-pages.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=80, height=100)
    writer.add_blank_page(width=200, height=100)
    writer.write(path)
    image, pages = await visual_render.render_pdf_page(path, 2)
    assert pages == 2
    with Image.open(io.BytesIO(image)) as picture:
        assert picture.format == "PNG"
        assert picture.size == (1600, 800)
    with pytest.raises(visual_render.VisualRenderError, match="between 1 and 2"):
        await visual_render.render_pdf_page(path, 3)


def _make_tiny_video(path):
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "color=c=red:s=32x32:d=1:r=2", "-c:v", "mpeg4", "-threads", "1", "-y", str(path)],
        check=True, timeout=15, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg unavailable")
async def test_real_video_renderer_returns_requested_still_frame(tmp_path):
    path = tmp_path / "red.mp4"
    await asyncio.to_thread(_make_tiny_video, path)
    image = await visual_render.render_video_frame(path, 0.25)
    with Image.open(io.BytesIO(image)) as picture:
        assert picture.format == "JPEG"
        assert max(picture.size) <= 1280
        red, green, blue = picture.convert("RGB").getpixel((0, 0))
        assert red > 200 and green < 50 and blue < 50
